"""TopoRetarget motion.npz (canonical frame) -> DexTrack training-format .npy.

Uses TopoRetarget's OWN coordinated 26-DOF (wrist6dof from its SQP wrist +
20 finger) -- NO Kabsch rederive (that was the old pipeline's patch for a
wrist-less retargeter; TopoRetarget optimizes wrist+finger jointly and its low
penetration lives in that coupled solution).

Output keys (exact training format, verified against commands.py loader):
  robot_delta_states_weights_np (T,26)  [wrist tx,ty,tz, rx,ry,rz(euler XYZ), finger1..5 j1..4]
  object_transl (T,3)                   canonical (copied from FPOS ref -> identical object)
  object_rot_quat (T,4) xyzw            canonical (copied from FPOS ref)
  link_key_to_link_pos {palm + 5 tips}  FK world positions via the fly-hand MJCF

Run with wuji_retarget_pen_min pixi env python (mujoco for FK).
"""
import argparse
import pathlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HAND = "/data/home/liangheng/DexTrack/wuji-mjlab/src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_fly_mjlab.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="TopoRetarget motion.npz (canonical frame)")
    ap.add_argument("--object-ref-npy", required=True, help="FPOS .npy to copy canonical object_transl/rot_quat from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.motion, allow_pickle=True)
    wt = d["retargeted_wrist_t_r"].astype(np.float64)                 # (T,3) canonical palm pos
    wq = d["retargeted_wrist_q_r"].astype(np.float64)                 # (T,4) wxyz canonical palm quat
    fingers = d["retargeted_qpos_r"].astype(np.float64)               # (T,20)
    T = len(wt)
    euler = R.from_quat(wq[:, [1, 2, 3, 0]]).as_euler("XYZ")          # wxyz->xyzw-> intrinsic XYZ

    # Clamp finger DOFs to the mjlab training MJCF limits. TopoRetarget's WH110 model
    # has a slightly wider thumb range than right_fly_mjlab.xml (<=~1.1deg), so its raw
    # finger qpos can exceed the mjlab limits by <=0.019 rad -- mjlab would clamp these
    # at load anyway, and FPOS sits inside. Clamp here so the stored data is valid in
    # the exact model it is trained in (sub-mm fingertip effect).
    m = mujoco.MjModel.from_xml_path(HAND)
    lo = np.full(26, -np.inf); hi = np.full(26, np.inf)
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        if m.jnt_limited[j]:
            lo[a] = m.jnt_range[j, 0]; hi[a] = m.jnt_range[j, 1]
    fingers_clamped = np.clip(fingers, lo[6:26], hi[6:26])
    _nclamp = int((np.abs(fingers_clamped - fingers) > 1e-6).any(1).sum())
    _maxclamp = float(np.abs(fingers_clamped - fingers).max())
    print(f"[assemble] finger clamp to mjlab limits: {_nclamp} frames touched, max {_maxclamp:.4f} rad ({np.degrees(_maxclamp):.2f} deg)")
    fingers = fingers_clamped
    q26 = np.concatenate([wt, euler, fingers], axis=1).astype(np.float32)  # (T,26)

    ref = np.load(args.object_ref_npy, allow_pickle=True).item()
    object_transl = ref["object_transl"][:T].astype(np.float32)
    object_rot_quat = ref["object_rot_quat"][:T].astype(np.float32)   # xyzw

    # link_key_to_link_pos from motion.npz retargeted_joint_pos_r (T,26,3), canonical
    # world: index 0 = palm; finger f (0..4) links at 1+5f..5+5f, tip at 5+5f.
    jp = d["retargeted_joint_pos_r"].astype(np.float32)               # (T,26,3)
    link_key = {"right_palm_link": jp[:, 0].copy()}
    for i in range(1, 6):
        link_key[f"right_finger{i}_tip_link"] = jp[:, 5 + 5 * (i - 1)].copy()

    tips = np.stack([link_key[f"right_finger{i}_tip_link"] for i in range(1, 6)], axis=1)  # (T,5,3)
    dtip = np.linalg.norm(tips - object_transl[:, None, :], axis=2)
    pos = {"right_palm_link": link_key["right_palm_link"]}
    print(f"[assemble] frames={T}  q26 range=[{q26.min():.3f},{q26.max():.3f}]  nan={np.isnan(q26).any()}")
    print(f"[assemble] palm z range: {pos['right_palm_link'][:,2].min():.3f}..{pos['right_palm_link'][:,2].max():.3f}")
    print(f"[assemble] mean tip->obj-center dist: {dtip.mean()*100:.1f}cm  (apple radius ~5.5cm => fingers on/near surface)")

    out = {"robot_delta_states_weights_np": q26, "object_transl": object_transl,
           "object_rot_quat": object_rot_quat, "link_key_to_link_pos": link_key}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)
    print(f"[assemble] saved -> {args.out}")


if __name__ == "__main__":
    main()
