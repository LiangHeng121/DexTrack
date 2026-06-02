"""B3: GRAB rhand MANO params -> per-frame 21 hand keypoints (MANO joint order).

Reads a raw GRAB sequence .npz (e.g. s2/cubesmall_inspect_1.npz), runs MANO
forward (manopth, use_pca=False) on the right-hand params, and emits a
(T, 21, 3) keypoint array in MANO joint order. These keypoints are the input
to the wuji retargeter (which re-centers on the wrist, so absolute world
placement here is irrelevant for finger retargeting).

Output keypoints are MANO's 21-joint order:
    0 wrist
    1..4   index  (MCP, PIP, DIP, TIP)
    5..8   middle
    9..12  pinky
    13..16 ring
    17..20 thumb
(Reordering to MediaPipe order happens in the next step, B3b.)

Usage:
    python wuji_pipeline/grab_to_mano_keypoints.py \
        --grab-npz GRAB/unzipped/grab/s2/cubesmall_inspect_1.npz \
        --out      wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy
"""
import argparse
import os

import numpy as np
import torch

from manopth.manolayer import ManoLayer

MANO_ROOT = os.path.join(os.path.dirname(__file__), "..", "retargeting", "manopth", "mano", "models")


def grab_rhand_to_keypoints(grab_npz, flat_hand_mean=True, batch=256):
    d = np.load(grab_npz, allow_pickle=True)
    rh = d["rhand"].item()["params"]
    global_orient = np.asarray(rh["global_orient"], dtype=np.float32)  # (T,3)
    fullpose = np.asarray(rh["fullpose"], dtype=np.float32)            # (T,45) 15 joints axis-angle
    transl = np.asarray(rh["transl"], dtype=np.float32)               # (T,3)
    T = global_orient.shape[0]

    # manopth (use_pca=False): th_pose_coeffs = [global(3), fullpose(45)] = 48
    pose48 = np.concatenate([global_orient, fullpose], axis=1)        # (T,48)

    ml = ManoLayer(
        mano_root=MANO_ROOT,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        side="right",
    )
    betas = torch.zeros(1, 10)  # NOTE: GRAB uses a personalized vtemp (s2_rhand.ply);
                                # betas=0 is a first-pass approximation (validate visually, B7).

    kp = np.zeros((T, 21, 3), dtype=np.float32)
    for s in range(0, T, batch):
        e = min(s + batch, T)
        p = torch.from_numpy(pose48[s:e])
        b = betas.expand(e - s, -1)
        verts, joints = ml(p, b)            # manopth returns mm-scale joints (T,21,3)
        j = joints.detach().numpy() / 1000.0  # mm -> meters
        # manopth applies global_orient (root rot) but not translation -> add it
        kp[s:e] = j + transl[s:e, None, :]
    return kp, dict(n_frames=int(T), obj_name=str(d["obj_name"]),
                    framerate=float(d["framerate"]))


def sanity_check(kp):
    """Rigid-hand sanity: bone lengths should be ~constant over time; hand span reasonable."""
    # MANO bone pairs (parent->child) for the 21-joint layout
    bones = [(0, 1), (1, 2), (2, 3), (3, 4),       # index
             (0, 5), (5, 6), (6, 7), (7, 8),       # middle
             (0, 9), (9, 10), (10, 11), (11, 12),  # pinky
             (0, 13), (13, 14), (14, 15), (15, 16),# ring
             (0, 17), (17, 18), (18, 19), (19, 20)]# thumb
    lens = np.stack([np.linalg.norm(kp[:, a] - kp[:, b], axis=-1) for a, b in bones], axis=1)
    span = np.linalg.norm(kp.max(axis=1) - kp.min(axis=1), axis=-1)  # per-frame bbox diag
    print(f"  frames                : {kp.shape[0]}")
    print(f"  hand bbox diag (m)    : mean={span.mean():.3f}  min={span.min():.3f}  max={span.max():.3f}  (expect ~0.10-0.20)")
    print(f"  bone length std/mean  : {(lens.std(0) / (lens.mean(0) + 1e-9)).max():.4f}  (rigid hand -> should be ~0, <0.02 good)")
    print(f"  any NaN               : {np.isnan(kp).any()}")
    # wrist motion magnitude
    wrist_disp = np.linalg.norm(np.diff(kp[:, 0], axis=0), axis=-1)
    print(f"  wrist per-frame disp  : mean={wrist_disp.mean()*1000:.1f}mm max={wrist_disp.max()*1000:.1f}mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flat-hand-mean", type=lambda s: s.lower() != "false", default=True)
    args = ap.parse_args()

    kp, meta = grab_rhand_to_keypoints(args.grab_npz, flat_hand_mean=args.flat_hand_mean)
    print(f"[grab_to_mano_keypoints] {args.grab_npz}")
    print(f"  meta: {meta}")
    sanity_check(kp)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, kp)
        print(f"  saved -> {args.out}  shape={kp.shape}")


if __name__ == "__main__":
    main()
