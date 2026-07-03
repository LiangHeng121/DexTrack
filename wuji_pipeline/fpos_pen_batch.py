"""Measure FPOS max penetration for a list of tags with the SAME WH110 mujoco query
the retargeter uses for TOPO (so pen numbers are directly comparable). Injects each
object's GRAB mesh x1.25 (= training/canonical object). One JSON line per tag.
"""
import argparse, json
import numpy as np
from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
from hand_retarget.io.hocap import materialize_hocap_scene
from hand_retarget.io.hocap_export import _per_finger_phi_mm

PROJECT = "/data/home/liangheng/DexTrack/wuji_retarget_pen_min"
SCENE = f"{PROJECT}/assets/scenes/single_hand_obj.xml"
ROOT = "/data/home/liangheng/DexTrack"
FPOSD = f"{ROOT}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data"
MESHDIR = f"{ROOT}/GRAB/unzipped/tools/object_meshes/contact_meshes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="file, one tag per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.25)
    args = ap.parse_args()

    scene_xml = materialize_hocap_scene(SCENE, hand_mjcf_dir=None)
    config = HandRetargetConfig.from_yaml(str(scene_xml and f"{PROJECT}/config/hocap.yaml"),
                                          mjcf_path=str(scene_xml), overrides={"model.hand_side": "right"})
    hand = InteractionMeshHandRetargeter(config).runtime.hand

    import trimesh, tempfile, os
    tags = [t.strip() for t in open(args.tags) if t.strip()]
    cur_obj = None
    fout = open(args.out, "w")
    for tag in tags:
        obj = tag[len("ori_grab_"):].split("_", 1)[1].split("_")[0]
        if obj != cur_obj:
            m = trimesh.load(f"{MESHDIR}/{obj}.ply", process=False, force="mesh")
            m.apply_scale(args.scale)
            tf = tempfile.NamedTemporaryFile(suffix=".stl", delete=False); m.export(tf.name)
            hand.inject_object_mesh(tf.name, "right"); os.unlink(tf.name); cur_obj = obj
        fp = np.load(f"{FPOSD}/wuji_passive_active_info_{tag}_nf_300.npy", allow_pickle=True).item()
        q = fp["robot_delta_states_weights_np"].astype(np.float64)
        ot = fp["object_transl"].astype(np.float64); oq = fp["object_rot_quat"].astype(np.float64)  # xyzw
        phi = np.full((len(q), 5), np.inf)
        for t in range(len(q)):
            hand.set_object_pose(ot[t], oq[t]); hand.forward(q[t]); phi[t] = _per_finger_phi_mm(hand)
        fin = phi[np.isfinite(phi)]
        rec = {"tag": tag, "fpos_pen_max_mm": round(float(-fin.min()), 3)}
        print(json.dumps(rec)); fout.write(json.dumps(rec) + "\n"); fout.flush()
    fout.close()


if __name__ == "__main__":
    main()
