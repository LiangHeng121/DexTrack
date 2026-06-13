# Generate B2 contact guidance: GRAB-truth FLAG + point = the human-contacted vertex NEAREST the
# wuji reference fingertip (NOT the centroid). Fixes the B(centroid) artifact: on broad contact
# patches (a whole cube face) the centroid drifts ~2cm from where the wuji finger actually is,
# spuriously fighting the reference grasp. The nearest-patch-vertex point sits ~where the wuji tip
# is (≈ A's projection point, <1mm on cube), so the ONLY thing that differs from A is the truth FLAG
# (which finger / which frames really touch) -- the part that matters on objects where A's geometric
# <1.2cm flag mis-fires (e.g. flute over-detects 30-40%).
#
# = "truth flag + fingertip-anchored point within the human contact region". Aligns with SPIDER
# (target a robot-reachable point inside the contact, not the exact human point).
import os, argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rsp

FINGER_IDS = {  # GRAB tools/utils.py contact_ids -> reward col [thumb,index,middle,ring,pinky]
    0: [53, 54, 55], 1: [41, 42, 43], 2: [44, 45, 46], 3: [50, 51, 52], 4: [47, 48, 49],
}
FNAME = ["thumb", "index", "middle", "ring", "pinky"]
TIP_KEYS = [f"right_finger{i}_tip_link" for i in range(1, 6)]   # finger1..5 = col0..4 (th,ff,mf,rf,lf)
PALM_ID = 22


def ang_speed_aa(aa): Rs = Rsp.from_rotvec(aa); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def ang_speed_quat(q): Rs = Rsp.from_quat(q); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def resample(sig, n):
    T = len(sig); xs = np.linspace(0, T - 1, n)
    lo = np.floor(xs).astype(int); hi = np.minimum(lo + 1, T - 1); w = xs - lo
    ws = (n,) + (1,) * (sig.ndim - 1)
    return sig[lo] * (1 - w).reshape(ws) + sig[hi] * w.reshape(ws)
def fit_crop(gsp, rsp):
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
    assert traj.startswith("ori_grab_"), traj
    subj, seq = traj[len("ori_grab_"):].split("_", 1)
    return os.path.join(grab_root, subj, f"{seq}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="ori_grab_s2_cubesmall_inspect_1")
    ap.add_argument("--grab_root", default="../GRAB/unzipped/grab")
    ap.add_argument("--allegro_ref_dir", default="./data/GRAB_Tracking_PK_reduced_300/data")
    ap.add_argument("--fpos_dir", default="./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data")
    ap.add_argument("--sim_mesh_dir", default="../assets/rsc/objs/meshes")  # 弃用(可能被抽稀)
    ap.add_argument("--grab_mesh_dir", default="../GRAB/unzipped/tools/object_meshes/contact_meshes")
    ap.add_argument("--mesh_scale", type=float, default=1.25)               # 仿真物体 = GRAB mesh x1.25
    ap.add_argument("--out_dir", default="./data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact_grab2")
    ap.add_argument("--nframes", type=int, default=300)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    grab_npz = traj_to_grab_npz(args.traj, args.grab_root)
    allegro_ref = os.path.join(args.allegro_ref_dir, f"passive_active_info_{args.traj}_nf_300.npy")
    fpos = os.path.join(args.fpos_dir, f"wuji_passive_active_info_{args.traj}_nf_300.npy")
    _subj, _seq = args.traj[len("ori_grab_"):].split("_", 1)
    obj_name = _seq.split("_")[0]                                   # ori_grab_s4_bowl_drink_2 -> bowl
    grab_mesh = os.path.join(args.grab_mesh_dir, f"{obj_name}.ply")
    for p in [grab_npz, allegro_ref, fpos, grab_mesh]:
        assert os.path.exists(p), f"missing {p}"

    g = np.load(grab_npz, allow_pickle=True)
    grab_aa = np.asarray(g["object"].item()["params"]["global_orient"])
    co = np.asarray(g["contact"].item()["object"])                        # (G,53890)

    # frame align
    ref = np.load(allegro_ref, allow_pickle=True).item(); N = args.nframes
    err, c0, c1 = fit_crop(ang_speed_aa(grab_aa), ang_speed_quat(ref["object_rot_quat"]))
    grab_idx = np.round(np.linspace(c0, c1 - 1, N)).astype(int)
    corr = np.corrcoef(resample(ang_speed_aa(grab_aa)[c0:c1], len(ang_speed_quat(ref["object_rot_quat"]))),
                       ang_speed_quat(ref["object_rot_quat"]))[0, 1]
    print(f"[align] crop GRAB[{c0}:{c1}] -> {N}  corr={corr:.4f}")

    # 接触点 = GRAB 原 mesh 顶点 x scale (接触数组就是对它索引的, 同序同数).
    # 不用仿真 mesh: 后者可能被抽稀/重排序(如 bowl 25107≠25119), 按 GRAB 索引会取错点甚至越界.
    sv = np.asarray(trimesh.load(grab_mesh, process=False, force="mesh").vertices, np.float64) * args.mesh_scale
    assert sv.shape[0] == co.shape[1], f"GRAB mesh verts {sv.shape[0]} != contact verts {co.shape[1]}"

    # wuji reference fingertips -> object-local (same method as A version generate_contact_guidance.py)
    fp = np.load(fpos, allow_pickle=True).item()
    transl = fp["object_transl"].astype(np.float64); quat = fp["object_rot_quat"].astype(np.float64)
    lk = fp["link_key_to_link_pos"]
    tips_w = np.stack([np.asarray(lk[k], np.float64) for k in TIP_KEYS], axis=1)   # (N,5,3) world
    Rm = Rsp.from_quat(quat).as_matrix()                                            # xyzw
    tips_l = np.einsum("tij,tkj->tki", np.transpose(Rm, (0, 2, 1)), tips_w - transl[:, None, :])  # (N,5,3)

    flag = np.zeros((N, 5), np.float32); cp_local = np.zeros((N, 5, 3), np.float32)
    chosen_dist = np.full((N, 5), np.nan)        # wuji tip -> chosen point (sanity: should be small ~A)
    for i in range(N):
        labels = co[grab_idx[i]]
        for f, ids in FINGER_IDS.items():
            idx = np.where(np.isin(labels, ids))[0]
            if len(idx) == 0: continue
            pts = sv[idx]                                  # contacted verts (object-local)
            j = np.argmin(np.linalg.norm(pts - tips_l[i, f], axis=1))   # nearest to wuji tip
            flag[i, f] = 1.0; cp_local[i, f] = pts[j]
            chosen_dist[i, f] = np.linalg.norm(pts[j] - tips_l[i, f])

    print("=== per-finger: truth contact %frames + wuji-tip->chosen-point dist (should be ~A, small) ===")
    for f in range(5):
        m = flag[:, f] > 0
        if m.sum() == 0: print(f"  {FNAME[f]:6s}: (never)"); continue
        print(f"  {FNAME[f]:6s}: {m.mean()*100:5.1f}%  tip->pt={np.nanmean(chosen_dist[m, f])*100:.2f}cm")

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.traj}_contact.npy")
        np.save(out, {"contact_flag": flag, "contact_pos_local": cp_local,
                      "contact_dist": np.nan_to_num(chosen_dist).astype(np.float32),
                      "quat_conv": "xyzw", "thresh": 0.0, "source": "grab_truth_nearest",
                      "grab_crop": [int(c0), int(c1)], "grab_idx": grab_idx.astype(np.int32)})
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
