"""B4: 21 MediaPipe-order hand keypoints -> per-frame wuji qpos(20).

Runs the official wuji Retargeter sequentially (warm-started, low-pass filtered)
over a (T,21,3) keypoint sequence produced by B3a (grab_to_mano_keypoints.py).
manopth's 21-joint output order already matches MediaPipe order, so no finger
reorder is needed.

Run in the `wuji-retarget` conda env (pinocchio + nlopt).

Usage:
    conda activate wuji-retarget
    python wuji_pipeline/retarget_keypoints_to_wuji.py \
        --kp wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy \
        --config wuji/wuji-retargeting/example/config/retarget_manus_right.yaml \
        --out wuji_pipeline/out/s2_cubesmall_inspect_1_wuji_qpos20.npy
"""
import argparse
import os

import numpy as np
from wuji_retargeting.retarget import Retargeter
from wuji_retargeting.robot import RobotWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", required=True, help="(T,21,3) keypoints, MediaPipe order, meters")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="scale hand about wrist before retargeting (use 1.25 to match DexTrack 5cm object)")
    args = ap.parse_args()

    kp = np.load(args.kp).astype(np.float64)  # (T,21,3)
    T = kp.shape[0]
    # Scale the hand about the wrist (kp[0]) before retargeting. DexTrack scales
    # the scene by ~1.25 (GRAB 4cm cube -> sim 5cm), so the wuji grip aperture
    # (retargeted from the unscaled 4cm human) must be scaled to match the 5cm
    # object, else the (too-small) grip pushes fingers into the bigger cube.
    if args.scale != 1.0:
        kp = kp[:, 0:1, :] + args.scale * (kp - kp[:, 0:1, :])
        print(f"  scaled keypoints about wrist by {args.scale}")
    r = Retargeter.from_yaml(args.config, hand_side="right")
    r.reset()

    qpos = np.zeros((T, r.num_joints), dtype=np.float32)
    for t in range(T):
        qpos[t] = r.retarget(kp[t], apply_filter=True)

    # joint limits from the same URDF the optimizer used
    lo = r.optimizer.robot.joint_limits[:, 0]
    hi = r.optimizer.robot.joint_limits[:, 1]
    dof_names = r.optimizer.robot.dof_joint_names

    # smoothness: per-frame joint delta
    dq = np.abs(np.diff(qpos, axis=0))
    # limit compliance
    below = (qpos < lo[None] - 1e-3).sum()
    above = (qpos > hi[None] + 1e-3).sum()

    print(f"[retarget_keypoints_to_wuji] {args.kp}")
    print(f"  frames               : {T}   num_joints: {r.num_joints}")
    print(f"  dof order (pinocchio): {dof_names}")
    print(f"  qpos range           : [{qpos.min():.3f}, {qpos.max():.3f}]")
    print(f"  max |dq| per frame   : {dq.max():.4f} rad   mean: {dq.mean():.4f}  (smooth if max small)")
    print(f"  limit violations     : below_lo={below}  above_hi={above}  (want 0)")
    print(f"  any NaN              : {np.isnan(qpos).any()}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, qpos)
        # also persist the dof order so downstream (Isaac Gym fly URDF) can build the permutation
        np.save(args.out.replace(".npy", "_dofnames.npy"), np.array(dof_names))
        print(f"  saved -> {args.out}  shape={qpos.shape}")
        print(f"  saved -> {args.out.replace('.npy','_dofnames.npy')}  (pinocchio dof order)")


if __name__ == "__main__":
    main()
