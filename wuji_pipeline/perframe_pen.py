"""Per-frame max-finger penetration (mm) for FPOS and TOPO of one tag, via the
SAME WH110 mujoco query. Writes phi_fpos.npy / phi_topo.npy (each (T,) = max
penetration mm per frame, 0 if no penetration). Run in wuji_retarget_pen_min env.
"""
import argparse, numpy as np
from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
from hand_retarget.io.hocap import materialize_hocap_scene
from hand_retarget.io.hocap_export import _per_finger_phi_mm

PROJECT = "/data/home/liangheng/DexTrack/wuji_retarget_pen_min"


def perframe(hand, npy):
    d = np.load(npy, allow_pickle=True).item()
    q = d["robot_delta_states_weights_np"].astype(np.float64)
    ot = d["object_transl"].astype(np.float64); oq = d["object_rot_quat"].astype(np.float64)
    T = len(q); out = np.zeros(T)
    for t in range(T):
        hand.set_object_pose(ot[t], oq[t]); hand.forward(q[t])
        phi = _per_finger_phi_mm(hand); phi = phi[np.isfinite(phi)]
        out[t] = max(0.0, -phi.min()) if len(phi) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpos", required=True); ap.add_argument("--topo", required=True)
    ap.add_argument("--mesh", required=True); ap.add_argument("--scale", type=float, default=1.25)
    ap.add_argument("--out-fpos", required=True); ap.add_argument("--out-topo", required=True)
    args = ap.parse_args()
    scene = materialize_hocap_scene(f"{PROJECT}/assets/scenes/single_hand_obj.xml", hand_mjcf_dir=None)
    cfg = HandRetargetConfig.from_yaml(str(scene and f"{PROJECT}/config/hocap.yaml"),
                                       mjcf_path=str(scene), overrides={"model.hand_side": "right"})
    hand = InteractionMeshHandRetargeter(cfg).runtime.hand
    import trimesh, tempfile, os
    m = trimesh.load(args.mesh, process=False, force="mesh"); m.apply_scale(args.scale)
    tf = tempfile.NamedTemporaryFile(suffix=".stl", delete=False); m.export(tf.name)
    hand.inject_object_mesh(tf.name, "right"); os.unlink(tf.name)
    pf = perframe(hand, args.fpos); pt = perframe(hand, args.topo)
    np.save(args.out_fpos, pf); np.save(args.out_topo, pt)
    print(f"FPOS pen max={pf.max():.2f} mean={pf.mean():.2f} | TOPO pen max={pt.max():.2f} mean={pt.mean():.2f}")


if __name__ == "__main__":
    main()
