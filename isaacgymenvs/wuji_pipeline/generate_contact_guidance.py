# Generate SPIDER-style contact-guidance data for wuji references (geometric, no external annot).
# Per frame, per fingertip: nearest point on the object surface (in OBJECT-LOCAL frame) + distance
# + contact flag (dist < thresh). The contact point is stored object-local so RL can transform it by
# the live object pose. Inputs: FPOS reference (5 fingertip world pos + object pose) + object mesh.
import os, sys, argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rsp
from scipy.spatial import cKDTree

TIP_KEYS = [f"right_finger{i}_tip_link" for i in range(1, 6)]

def quat_to_R(quat, conv):
    # quat: (T,4). conv 'xyzw' or 'wxyz'
    q = quat if conv == "xyzw" else quat[:, [1, 2, 3, 0]]
    return Rsp.from_quat(q).as_matrix()  # (T,3,3)

def fingertips_to_local(tips_w, transl, R):
    # tips_w (T,5,3), transl (T,3), R (T,3,3) -> object-local (T,5,3)
    rel = tips_w - transl[:, None, :]
    return np.einsum("tij,tkj->tki", np.transpose(R, (0, 2, 1)), rel)  # R^T @ rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="ori_grab_s2_cubesmall_inspect_1")
    ap.add_argument("--data_dir", default="./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data")
    ap.add_argument("--mesh_dir", default="../assets/rsc/objs/meshes")
    ap.add_argument("--thresh", type=float, default=0.012)  # 1.2cm contact threshold
    ap.add_argument("--out_dir", default="./data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ref_fn = os.path.join(args.data_dir, f"wuji_passive_active_info_{args.traj}_nf_300.npy")
    d = np.load(ref_fn, allow_pickle=True).item()
    transl = d["object_transl"].astype(np.float64)           # (T,3)
    quat = d["object_rot_quat"].astype(np.float64)           # (T,4)
    lk = d["link_key_to_link_pos"]
    tips_w = np.stack([np.asarray(lk[k], np.float64) for k in TIP_KEYS], axis=1)  # (T,5,3)
    T = transl.shape[0]

    mesh_fn = os.path.join(args.mesh_dir, f"{args.traj}.obj")
    mesh = trimesh.load(mesh_fn, force="mesh")
    verts = np.asarray(mesh.vertices, np.float64)
    tree = cKDTree(verts)   # nearest-vertex query (dense mesh -> ~nearest surface point)
    print(f"traj={args.traj} frames={T} mesh={mesh_fn} verts={len(verts)} bbox={mesh.extents.round(3)}")

    # quat convention is FIXED to xyzw (confirmed: task feeds object_rot_quat into gymapi.Quat(x,y,z,w)
    # at line ~5105 and scipy R.from_quat at ~4640, both xyzw). Print both medians only as a sanity check.
    conv = "xyzw"
    for c in ["xyzw", "wxyz"]:
        R_ = quat_to_R(quat, c); tl = fingertips_to_local(tips_w, transl, R_)
        dd_, _ = tree.query(tl.reshape(-1, 3))
        print(f"  [sanity] conv={c}: median fingertip-surface dist={np.median(dd_)*100:.2f}cm  min={dd_.min()*100:.2f}cm")
    R = quat_to_R(quat, conv)
    tips_l = fingertips_to_local(tips_w, transl, R)
    dd, ii = tree.query(tips_l.reshape(-1, 3))
    dist = dd.reshape(T, 5); cp_local = verts[ii].reshape(T, 5, 3)
    print(f"=> using FIXED quat convention: {conv}")

    flag = (dist < args.thresh).astype(np.float32)            # (T,5)
    print("=== per-finger contact %frames + contact frame-range ===")
    for i, k in enumerate(TIP_KEYS):
        fr = np.where(flag[:, i] > 0)[0]
        rng = f"frames[{fr.min()}-{fr.max()}]" if len(fr) else "(never)"
        print(f"  f{i+1}: {flag[:,i].mean()*100:5.1f}%  minDist={dist[:,i].min()*100:.2f}cm  {rng}")
    # contact pattern over time (any finger), sampled
    anyc = (flag.sum(1) > 0).astype(int)
    print("=== 接触时间线(每20帧, 数字=该帧接触的手指数)===")
    print("  " + " ".join(f"{int(flag[t].sum())}" for t in range(0, T, 20)))
    print(f"  抓握期(有接触)帧数 {anyc.sum()}/{T}; 物体最大升幅 {(transl[:,2].max()-transl[:,2].min())*100:.1f}cm")

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.traj}_contact.npy")
        np.save(out, {"contact_flag": flag, "contact_pos_local": cp_local.astype(np.float32),
                      "contact_dist": dist.astype(np.float32), "quat_conv": conv, "thresh": args.thresh})
        print(f"saved -> {out}")

if __name__ == "__main__":
    main()
