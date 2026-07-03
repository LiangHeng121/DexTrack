"""GRAB raw sequence -> HO-Cap schema (for TopoRetarget PoC).

Consumes a (T,21,3) MediaPipe-order keypoint array produced by
grab_to_mano_keypoints.py (NB: manopth's line-260 reorder already emits
thumb-first MediaPipe order -- the grab_to_mano docstring's "MANO order"
label is wrong; verified against manopth/manolayer.py:260 and
rederive_wuji_global.py TIP_KP=[4,8,12,16,20]). So NO finger reorder here.

Emits a HO-Cap-schema dir:
  <out>/motions/<clip>.npz         mediapipe_r_world, wrist_t_r, wrist_q_r(wxyz),
                                    object_t(T,1,3), object_q(T,1,4 wxyz), fps
  <out>/motions/<clip>.meta.json   handedness/objects[].asset_name/quaternion_order
  <out>/assets/<obj>/mesh_med.stl  object mesh
  <out>/assets/<obj>/asset.json

wrist_q is computed from the keypoints via the new repo's own
estimate_frame_from_hand_points + OPERATOR2MANO_RIGHT, so io/hocap.py's
wrist_q-based palm alignment is byte-identical to the MANUS SVD path the
retargeter was designed for (sidesteps any GRAB global_orient convention).

Run with the wuji_retarget_pen_min pixi env python (has hand_retarget + trimesh).
"""
import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from hand_retarget.preprocess._mano_frame import (
    OPERATOR2MANO_RIGHT,
    estimate_frame_from_hand_points,
)


def compute_wrist_q_wxyz(kp):
    """(T,21,3) MediaPipe-order kp -> (T,4) wrist quat wxyz matching io alignment."""
    T = kp.shape[0]
    q = np.zeros((T, 4), dtype=np.float64)
    for t in range(T):
        kpc = kp[t] - kp[t, 0:1]
        frame = estimate_frame_from_hand_points(kpc)      # (3,3) MANO wrist frame
        R_align = frame @ OPERATOR2MANO_RIGHT             # matches apply_mediapipe_transformations
        q_xyzw = R.from_matrix(R_align).as_quat()
        q[t] = q_xyzw[[3, 0, 1, 2]]                        # -> wxyz
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-npz", required=True)
    ap.add_argument("--kp", required=True, help="(T,21,3) MediaPipe-order kp from grab_to_mano_keypoints.py")
    ap.add_argument("--obj-mesh", required=True, help="canonical object mesh (.ply/.stl) matching GRAB object frame")
    ap.add_argument("--obj-name", required=True)
    ap.add_argument("--out-dir", required=True, help="HO-Cap dataset root to create")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--nframes", type=int, default=300)
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-end", type=int, default=-1, help="inclusive; -1 = last")
    ap.add_argument("--mesh-scale", type=float, default=1.0, help="scale object mesh about origin (1.25 = training convention)")
    ap.add_argument("--grab-idx-start", type=float, default=None, help="fractional-frame resample start (e.g. 160)")
    ap.add_argument("--grab-idx-end", type=float, default=None, help="fractional-frame resample end inclusive (e.g. 458)")
    ap.add_argument("--canon-sim", default=None, help="npy {s,R,t}: GRAB->DexTrack canonical similarity applied to keypoints+object")
    ap.add_argument("--object-ref-npy", default=None, help="training .npy whose object_transl/rot_quat (canonical) OVERRIDE the GRAB object")
    args = ap.parse_args()

    def resample_frac(sig, gidx):
        lo = np.floor(gidx).astype(int); hi = np.minimum(lo + 1, len(sig) - 1); w = gidx - lo
        ws = (len(gidx),) + (1,) * (sig.ndim - 1)
        return sig[lo] * (1 - w).reshape(ws) + sig[hi] * w.reshape(ws)

    kp = np.load(args.kp).astype(np.float64)              # (T,21,3) world, meters, MediaPipe order
    d = np.load(args.grab_npz, allow_pickle=True)
    obj = d["object"].item()["params"]
    obj_transl = np.asarray(obj["transl"], dtype=np.float64)       # (T,3) world meters
    obj_orient = np.asarray(obj["global_orient"], dtype=np.float64)  # (T,3) axis-angle
    fps = float(d["framerate"])
    T = kp.shape[0]
    assert obj_transl.shape[0] == T, (obj_transl.shape, T)

    if args.grab_idx_start is not None:
        gidx = np.linspace(args.grab_idx_start, args.grab_idx_end, args.nframes)
        kp = resample_frac(kp, gidx); obj_transl = resample_frac(obj_transl, gidx); obj_orient = resample_frac(obj_orient, gidx)
        print(f"  fractional resample grab[{args.grab_idx_start}:{args.grab_idx_end}] -> {args.nframes} frames")
    else:
        lo = max(0, args.frame_start)
        hi = (T - 1) if args.frame_end < 0 else min(T - 1, args.frame_end)
        idx = np.linspace(lo, hi, min(args.nframes, hi - lo + 1)).round().astype(int)
        kp = kp[idx]; obj_transl = obj_transl[idx]; obj_orient = obj_orient[idx]
    n = len(kp)

    # GRAB -> DexTrack canonical similarity on keypoints+object, so retargeter output
    # lands directly in the canonical (training/FPOS) frame.
    if args.canon_sim:
        sim = np.load(args.canon_sim, allow_pickle=True).item()
        s, Rw, tt = float(sim["s"]), np.asarray(sim["R"]), np.asarray(sim["t"])
        kp = (s * (kp.reshape(-1, 3) @ Rw.T) + tt).reshape(n, 21, 3)
        obj_transl = s * (obj_transl @ Rw.T) + tt
        print(f"  applied canonical similarity s={s:.4f} to keypoints + object")

    wrist_t_r = kp[:, 0, :].astype(np.float64)                     # wrist keypoint (viz only)
    wrist_q_r = compute_wrist_q_wxyz(kp)                           # wxyz

    if args.object_ref_npy:
        ref = np.load(args.object_ref_npy, allow_pickle=True).item()
        object_t = ref["object_transl"][:n].astype(np.float64).reshape(n, 1, 3)
        object_q = ref["object_rot_quat"][:n][:, [3, 0, 1, 2]].astype(np.float64).reshape(n, 1, 4)  # xyzw->wxyz
        print(f"  object pose from ref {os.path.basename(args.object_ref_npy)} (canonical, identical to FPOS)")
    else:
        Ro = (R.from_matrix(np.asarray(sim["R"])) * R.from_rotvec(obj_orient)) if args.canon_sim else R.from_rotvec(obj_orient)
        object_q = Ro.as_quat()[:, [3, 0, 1, 2]].reshape(n, 1, 4)
        object_t = obj_transl.reshape(n, 1, 3)

    # dirs
    mot = os.path.join(args.out_dir, "motions")
    asdir = os.path.join(args.out_dir, "assets", args.obj_name)
    os.makedirs(mot, exist_ok=True)
    os.makedirs(asdir, exist_ok=True)

    # mesh -> mesh_med.stl
    mesh = trimesh.load(args.obj_mesh, process=False)
    if args.mesh_scale != 1.0:
        mesh.apply_scale(args.mesh_scale)          # about origin, matches grab_object_cfg tm.apply_scale
        print(f"  scaled object mesh about origin by {args.mesh_scale}")
    stl_path = os.path.join(asdir, "mesh_med.stl")
    mesh.export(stl_path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"[mesh] {args.obj_mesh} -> {stl_path}  bbox(m)={np.round(ext,4)}  verts={len(mesh.vertices)}")

    with open(os.path.join(asdir, "asset.json"), "w") as f:
        json.dump({"asset_name": args.obj_name, "shape": "mesh",
                   "stl_variants": [{"tier": "med", "path": "mesh_med.stl"}],
                   "notes": f"GRAB object mesh {os.path.basename(args.obj_mesh)} (meters, unscaled)"}, f, indent=2)

    npz_path = os.path.join(mot, f"{args.clip}.npz")
    np.savez(
        npz_path,
        mediapipe_r_world=kp.astype(np.float32),
        wrist_t_r=wrist_t_r.astype(np.float32),
        wrist_q_r=wrist_q_r.astype(np.float32),
        object_t=object_t.astype(np.float32),
        object_q=object_q.astype(np.float32),
        fps=np.float32(fps),
        mediapipe_l_world=np.array(None, dtype=object),
        wrist_t_l=np.array(None, dtype=object),
        wrist_q_l=np.array(None, dtype=object),
    )
    with open(os.path.join(mot, f"{args.clip}.meta.json"), "w") as f:
        json.dump({"handedness": "right",
                   "objects": [{"asset_name": args.obj_name}],
                   "retargeting": {"quaternion_order": "wxyz"},
                   "source": f"GRAB {os.path.basename(args.grab_npz)} subsampled {T}->{n}"}, f, indent=2)

    print(f"[hocap] wrote {npz_path}  frames={n}  fps={fps}")
    print(f"[hocap] mediapipe_r_world {kp.shape}  wrist_q_r wxyz  object_t {object_t.shape}")
    print(f"[hocap] object z range (world m): {object_t[:,0,2].min():.3f}..{object_t[:,0,2].max():.3f}")


if __name__ == "__main__":
    main()
