"""B5/B6: assemble wuji kinematic reference npy (DexTrack fly format).

Strategy (see docs/wuji_integration_plan.md):
- object_transl / object_rot_quat : reuse the existing allegro reference
  (already in DexTrack frame, 300 frames).
- global 6-DOF                    : reuse allegro reference cols [0:6]
  (same wrist motion, same fly floating-base convention as our wuji fly URDF).
- wuji finger 20-DOF             : from B4 (qpos20 at GRAB's 776 frames),
  time-resampled to the same 300 frames via object-pose time alignment.
  Finger->isaac DOF order is identity (Phase A used the right_-prefixed URDF).

Time alignment: the object is the shared signal. We model the allegro ref as a
linear crop [c0,c1] of GRAB resampled to 300, and fit (c0,c1) by matching the
object angular-speed profile (frame-invariant under world-frame change).

Run in dextrack env.
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def ang_speed_from_aa(aa):       # (T,3) axis-angle -> (T-1,) rotation speed
    Rs = R.from_rotvec(aa)
    return (Rs[:-1].inv() * Rs[1:]).magnitude()


def ang_speed_from_quat(q):      # (T,4) quat -> (T-1,)
    Rs = R.from_quat(q)
    return (Rs[:-1].inv() * Rs[1:]).magnitude()


def resample(sig, n):            # linear-resample sig (T,...) to length n
    T = len(sig)
    xs = np.linspace(0, T - 1, n)
    lo = np.floor(xs).astype(int); hi = np.minimum(lo + 1, T - 1); w = (xs - lo)
    wshape = (n,) + (1,) * (sig.ndim - 1)
    return sig[lo] * (1 - w).reshape(wshape) + sig[hi] * w.reshape(wshape)


def fit_crop(grab_speed, ref_speed):
    """Search crop [c0,c1] of grab_speed whose resample-to-len(ref) best matches ref_speed."""
    n = len(ref_speed)
    rn = (ref_speed - ref_speed.mean()) / (ref_speed.std() + 1e-9)
    G = len(grab_speed)
    best = (1e9, None, None)
    # coarse then fine
    for c0 in range(0, G - n, 4):
        for c1 in range(c0 + n, G, 4):
            g = resample(grab_speed[c0:c1], n)
            gn = (g - g.mean()) / (g.std() + 1e-9)
            err = np.mean((gn - rn) ** 2)
            if err < best[0]:
                best = (err, c0, c1)
    # fine around best
    _, bc0, bc1 = best
    for c0 in range(max(0, bc0 - 4), bc0 + 5):
        for c1 in range(bc1 - 4, min(G, bc1 + 5)):
            if c1 - c0 < n:
                continue
            g = resample(grab_speed[c0:c1], n)
            gn = (g - g.mean()) / (g.std() + 1e-9)
            err = np.mean((gn - rn) ** 2)
            if err < best[0]:
                best = (err, c0, c1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-npz", default="GRAB/unzipped/grab/s2/cubesmall_inspect_1.npz")
    ap.add_argument("--allegro-ref", default="isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
    ap.add_argument("--wuji-qpos", default="wuji_pipeline/out/s2_cubesmall_inspect_1_wuji_qpos20.npy")
    ap.add_argument("--out", default="isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
    args = ap.parse_args()

    # --- load ---
    g_obj = np.load(args.grab_npz, allow_pickle=True)["object"].item()["params"]
    grab_aa = np.asarray(g_obj["global_orient"])                 # (776,3)
    ref = np.load(args.allegro_ref, allow_pickle=True).item()
    ref_q = ref["object_rot_quat"]                               # (300,4)
    N = ref_q.shape[0]
    wuji_q20 = np.load(args.wuji_qpos)                           # (776,20)

    # --- time align via object angular speed ---
    gs = ang_speed_from_aa(grab_aa)
    rs = ang_speed_from_quat(ref_q)
    err, c0, c1 = fit_crop(gs, rs)
    print(f"[align] best crop GRAB[{c0}:{c1}] -> {N} frames   norm-MSE={err:.4f}")
    # quality: correlation of speed profiles
    g_al = resample(gs[c0:c1], len(rs))
    corr = np.corrcoef(g_al, rs)[0, 1]
    print(f"[align] object angular-speed corr after align: {corr:.4f}  (1.0=perfect)")

    # --- map ref frame i (0..N-1) -> GRAB fractional index, resample wuji fingers ---
    grab_idx = np.linspace(c0, c1 - 1, N)
    lo = np.floor(grab_idx).astype(int); hi = np.minimum(lo + 1, len(wuji_q20) - 1); w = grab_idx - lo
    wuji_fingers_300 = wuji_q20[lo] * (1 - w)[:, None] + wuji_q20[hi] * w[:, None]   # (300,20)

    # --- assemble (300,26) = [allegro global 6, wuji finger 20] ---
    global6 = ref["robot_delta_states_weights_np"][:, :6]        # (300,6)
    robot_states = np.concatenate([global6, wuji_fingers_300], axis=1).astype(np.float32)  # (300,26)

    out = {
        "object_transl": ref["object_transl"].astype(np.float32),
        "object_rot_quat": ref["object_rot_quat"].astype(np.float32),
        "robot_delta_states_weights_np": robot_states,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, out)
    print(f"[assemble] robot_states {robot_states.shape} = [global6 + wuji_finger20]")
    print(f"[assemble] finger range [{wuji_fingers_300.min():.3f}, {wuji_fingers_300.max():.3f}]")
    print(f"[assemble] saved -> {args.out}")


if __name__ == "__main__":
    main()
