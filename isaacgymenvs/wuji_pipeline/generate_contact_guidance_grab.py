# Generate GRAB-GROUND-TRUTH contact-guidance data for wuji references.
#
# Unlike generate_contact_guidance.py (A version, which projects the *retargeted* wuji
# fingertips onto the object surface and inherits retargeting error), this reads GRAB's
# real human-grasp contact annotation: contact['object'] (n_grab_frames x 53890 obj verts),
# value = body-part id touching that object vertex (GRAB tools/utils.py contact_ids).
#
# Pipeline (all confirmed, see docs/grab_contact_guidance_plan.md):
#  1. parse traj -> GRAB npz (subject + sequence).
#  2. frame align: replicate assemble_wuji_reference.fit_crop -- the wuji 300-frame ref is a
#     crop [c0,c1] of the GRAB frames (found by matching object angular-speed), resampled to 300.
#     => grab_idx[i] = round(linspace(c0,c1-1,300))[i] is the GRAB frame for wuji frame i.
#  3. mesh: sim obj mesh assets/rsc/objs/meshes/{traj}.obj == GRAB ply * 1.25, SAME vertex order
#     (verified max|sim - grab*1.25|=0). So contact vertex index i -> sim_mesh.vertices[i] IS the
#     object-local contact point in the exact frame the reward uses (no transform needed).
#  4. per wuji frame, per finger: collect contacted object verts whose label is in that finger's
#     id set -> contact_pos_local = centroid of those sim verts; flag = 1 if any.
#
# Output matches A version exactly (loader reads contact_pos_local (T,5,3) + contact_flag (T,5)):
# columns are [thumb, index, middle, ring, pinky] (reward cols 0..4 -> th/ff/mf/rf/lf).
import os, argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rsp

# GRAB tools/utils.py contact_ids -> reward finger column [thumb, index, middle, ring, pinky]
FINGER_IDS = {
    0: [53, 54, 55],   # thumb  (R_Thumb1/2/3)  -> reward col0 right_hand_th_pos
    1: [41, 42, 43],   # index  (R_Index1/2/3)  -> col1 ff
    2: [44, 45, 46],   # middle (R_Middle1/2/3) -> col2 mf
    3: [50, 51, 52],   # ring   (R_Ring1/2/3)   -> col3 rf
    4: [47, 48, 49],   # pinky  (R_Pinky1/2/3)  -> col4 lf
}
FNAME = ["thumb", "index", "middle", "ring", "pinky"]
PALM_ID = 22           # R_Hand (palm) -- logged only, not a fingertip column
SIM_SCALE = 1.25       # sim mesh = GRAB ply * 1.25 (uniform), same vertex ordering


def ang_speed_aa(aa):                      # (T,3) axis-angle -> (T-1,)
    Rs = Rsp.from_rotvec(aa); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def ang_speed_quat(q):                      # (T,4) xyzw -> (T-1,)
    Rs = Rsp.from_quat(q); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def resample(sig, n):
    T = len(sig); xs = np.linspace(0, T - 1, n)
    lo = np.floor(xs).astype(int); hi = np.minimum(lo + 1, T - 1); w = xs - lo
    ws = (n,) + (1,) * (sig.ndim - 1)
    return sig[lo] * (1 - w).reshape(ws) + sig[hi] * w.reshape(ws)
def fit_crop(gsp, rsp):
    """Identical to assemble_wuji_reference.fit_crop: crop of grab speed best matching ref speed."""
    n = len(rsp); rn = (rsp - rsp.mean()) / (rsp.std() + 1e-9); G = len(gsp); best = (1e9, None, None)
    for c0 in range(0, G - n, 4):
        for c1 in range(c0 + n, G, 4):
            g = resample(gsp[c0:c1], n); gn = (g - g.mean()) / (g.std() + 1e-9)
            e = np.mean((gn - rn) ** 2)
            if e < best[0]: best = (e, c0, c1)
    _, b0, b1 = best
    for c0 in range(max(0, b0 - 4), b0 + 5):
        for c1 in range(b1 - 4, min(G, b1 + 5)):
            if c1 - c0 < n: continue
            g = resample(gsp[c0:c1], n); gn = (g - g.mean()) / (g.std() + 1e-9)
            e = np.mean((gn - rn) ** 2)
            if e < best[0]: best = (e, c0, c1)
    return best


def traj_to_grab_npz(traj, grab_root):
    # ori_grab_s2_cubesmall_inspect_1 -> s2 / cubesmall_inspect_1.npz
    assert traj.startswith("ori_grab_"), traj
    rest = traj[len("ori_grab_"):]            # s2_cubesmall_inspect_1
    subj, seq = rest.split("_", 1)            # s2 , cubesmall_inspect_1
    return os.path.join(grab_root, subj, f"{seq}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="ori_grab_s2_cubesmall_inspect_1")
    ap.add_argument("--grab_root", default="../GRAB/unzipped/grab")
    ap.add_argument("--allegro_ref_dir", default="./data/GRAB_Tracking_PK_reduced_300/data")
    ap.add_argument("--sim_mesh_dir", default="../assets/rsc/objs/meshes")
    ap.add_argument("--out_dir", default="./data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact_grab")
    ap.add_argument("--nframes", type=int, default=300)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    grab_npz = traj_to_grab_npz(args.traj, args.grab_root)
    allegro_ref = os.path.join(args.allegro_ref_dir, f"passive_active_info_{args.traj}_nf_300.npy")
    sim_mesh = os.path.join(args.sim_mesh_dir, f"{args.traj}.obj")
    for p in [grab_npz, allegro_ref, sim_mesh]:
        assert os.path.exists(p), f"missing {p}"

    # --- load GRAB ground-truth contact + object orient ---
    g = np.load(grab_npz, allow_pickle=True)
    grab_aa = np.asarray(g["object"].item()["params"]["global_orient"])   # (G,3)
    co = np.asarray(g["contact"].item()["object"])                        # (G,53890) int8 part-ids
    G = co.shape[0]

    # --- frame align (replicate the retargeting's fit_crop) ---
    ref = np.load(allegro_ref, allow_pickle=True).item()
    ref_q = ref["object_rot_quat"]; N = args.nframes
    gs = ang_speed_aa(grab_aa); rs = ang_speed_quat(ref_q)
    err, c0, c1 = fit_crop(gs, rs)
    corr = np.corrcoef(resample(gs[c0:c1], len(rs)), rs)[0, 1]
    grab_idx = np.round(np.linspace(c0, c1 - 1, N)).astype(int)
    print(f"[align] crop GRAB[{c0}:{c1}] ({G} frames) -> {N}   normMSE={err:.4f}  angspeed-corr={corr:.4f}")

    # --- sim mesh verts = object-local contact points (verified == GRAB ply * 1.25, same order) ---
    sv = np.asarray(trimesh.load(sim_mesh, process=False, force="mesh").vertices, np.float64)  # (53890,3)
    print(f"[mesh] {sim_mesh} verts={sv.shape}  (GRAB ply x{SIM_SCALE}, contact idx i -> sim vert i)")

    # --- per wuji frame, per finger: centroid of GRAB-truth contacted verts ---
    flag = np.zeros((N, 5), np.float32)
    cp_local = np.zeros((N, 5, 3), np.float32)
    ncontact = np.zeros((N, 5), np.int32)
    palm_frames = 0
    for i in range(N):
        labels = co[grab_idx[i]]
        if np.any(labels == PALM_ID): palm_frames += 1
        for f, ids in FINGER_IDS.items():
            mask = np.isin(labels, ids)
            nc = int(mask.sum()); ncontact[i, f] = nc
            if nc > 0:
                flag[i, f] = 1.0
                cp_local[i, f] = sv[mask].mean(0)

    # --- report ---
    print("=== per-finger GRAB-truth contact %frames + frame-range + avg #verts ===")
    for f in range(5):
        fr = np.where(flag[:, f] > 0)[0]
        rng = f"frames[{fr.min()}-{fr.max()}]" if len(fr) else "(never)"
        avgv = ncontact[fr, f].mean() if len(fr) else 0
        print(f"  {FNAME[f]:6s}: {flag[:,f].mean()*100:5.1f}%  {rng}  avg#verts={avgv:.0f}")
    anyc = (flag.sum(1) > 0)
    print(f"palm(R_Hand 22) present in {palm_frames}/{N} frames; any-finger contact {anyc.sum()}/{N} frames")
    print("=== 接触时间线(每20帧, 数字=该帧接触的手指数)===")
    print("  " + " ".join(f"{int(flag[t].sum())}" for t in range(0, N, 20)))

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.traj}_contact.npy")
        np.save(out, {"contact_flag": flag, "contact_pos_local": cp_local,
                      "contact_dist": np.zeros((N, 5), np.float32),  # truth: no fingertip-dist, placeholder
                      "quat_conv": "xyzw", "thresh": 0.0, "source": "grab_truth",
                      "grab_crop": [int(c0), int(c1)], "grab_idx": grab_idx.astype(np.int32)})
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
