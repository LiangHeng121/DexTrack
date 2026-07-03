"""Measure FPOS_v1 reference penetration with the SAME WH110 mujoco query
that hocap_export uses for TopoRetarget, so the two are directly comparable.

Reuses retargeter.runtime.hand (apple mesh injected) + _per_finger_phi_mm.
FPOS robot_delta_states_weights_np = [6 wrist(tx,ty,tz,rx,ry,rz) + 20 finger]
in the SAME wrist6dof convention as the WH110 scene -> load directly.

Self-check: FK fingertips (MuJoCo world) vs FPOS link_key_to_link_pos
(FPOS canonical world). Match => wrist convention consistent => phi valid.

Run with wuji_retarget_pen_min pixi env python.
"""
import argparse
import numpy as np

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
from hand_retarget.io.hocap import materialize_hocap_scene
from hand_retarget.io.hocap_export import _per_finger_phi_mm
from scipy.spatial.transform import Rotation as R

PROJECT = "/data/home/liangheng/DexTrack/wuji_retarget_pen_min"
HOCAP_CONFIG = f"{PROJECT}/config/hocap.yaml"
SCENE = f"{PROJECT}/assets/scenes/single_hand_obj.xml"
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
TIP_BODIES = [f"right_finger{i}_tip_link" for i in range(1, 6)]


def summarize(phi, tag):
    print(f"\n=== {tag}  per-finger phi_mm (mm, neg=penetration) ===")
    for i, fn in enumerate(FINGERS):
        col = phi[:, i]; fin = col[np.isfinite(col)]
        print(f"  {fn:7s} min={fin.min():7.2f}  p50={np.median(fin):7.2f}  frames_pen(<0)={int((col < 0).sum())}/{len(col)}")
    allp = phi[np.isfinite(phi)]
    print(f"  >> MAX PENETRATION over all fingers/frames: {-allp.min():.3f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpos", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--topo-npz", default=None, help="TopoRetarget motion.npz for side-by-side")
    ap.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw")
    args = ap.parse_args()

    # Build runtime + inject apple mesh (same model hocap_export uses)
    scene_xml = materialize_hocap_scene(SCENE, hand_mjcf_dir=None)
    config = HandRetargetConfig.from_yaml(str(scene_xml and HOCAP_CONFIG),
                                          mjcf_path=str(scene_xml),
                                          overrides={"model.hand_side": "right"})
    rt = InteractionMeshHandRetargeter(config).runtime
    hand = rt.hand
    hand.inject_object_mesh(args.mesh, "right")
    assert hand.has_object, "apple mesh not injected"

    fp = np.load(args.fpos, allow_pickle=True).item()
    qpos26 = fp["robot_delta_states_weights_np"].astype(np.float64)  # (T,26)
    obj_t = fp["object_transl"].astype(np.float64)                   # (T,3)
    obj_q = fp["object_rot_quat"].astype(np.float64)                 # (T,4)
    link_pos = fp.get("link_key_to_link_pos", None)
    T = len(qpos26)
    if args.quat_order == "wxyz":
        obj_q = obj_q[:, [1, 2, 3, 0]]
    print(f"[fpos] {args.fpos}\n  frames={T} qpos26 range=[{qpos26.min():.3f},{qpos26.max():.3f}] nan={np.isnan(qpos26).any()}")

    # --- self-check: FK fingertip vs stored FPOS fingertip positions ---
    if link_pos is not None:
        errs = []
        for t in range(0, T, max(1, T // 30)):
            hand.forward(qpos26[t])
            for i, b in enumerate(TIP_BODIES):
                if b in link_pos:
                    errs.append(np.linalg.norm(hand.get_body_pos(b) - link_pos[b][t]))
        errs = np.array(errs)
        print(f"  [self-check] FK tip vs FPOS link pos: mean={errs.mean()*1000:.2f}mm max={errs.max()*1000:.2f}mm "
              f"(<~5mm => wrist convention consistent, phi valid)")

    # --- FPOS penetration via identical query ---
    phi = np.full((T, 5), np.inf)
    for t in range(T):
        hand.set_object_pose(obj_t[t], obj_q[t])   # xyzw
        hand.forward(qpos26[t])
        phi[t] = _per_finger_phi_mm(hand)
    summarize(phi, f"FPOS_v1 (quat={args.quat_order})")

    if args.topo_npz:
        d = np.load(args.topo_npz, allow_pickle=True)
        summarize(d["phi_mm_r"], "TopoRetarget")


if __name__ == "__main__":
    main()
