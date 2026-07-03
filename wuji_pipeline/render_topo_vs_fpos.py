"""Side-by-side KINEMATIC replay: LEFT=TopoRetarget | RIGHT=FPOS_v1 (old),
same seq (apple s2 lift), same apple mesh, same camera. No physics -> penetration
is shown faithfully. Apple is rendered semi-transparent so fingers-inside-apple
is visible.

Both panels share the WH110 hand model (wrist6dof injected + apple mocap mesh)
built by the retargeter runtime -- the model this PoC already validated DOF-1:1.

LEFT  : hand=retarget_sequence(clip) qpos26 ; apple = GRAB obj pose in palm frame
        (exactly the geometry _per_finger_phi_mm measured -> ~1.35mm max pen).
RIGHT : hand=FPOS robot_delta_states_weights_np (26) ; apple=FPOS obj pose
        (FPOS canonical frame; ~15.7mm max pen).

Frames differ in world origin, but the free camera tracks each panel's apple
(lookat=apple pos), so both are centered & same scale. Run: pixi env python,
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
from hand_retarget.io.hocap import load_hocap_clip, materialize_hocap_scene, resolve_hocap_clip_paths
from hand_retarget.io.hocap_export import _per_finger_phi_mm

PROJ = "/data/home/liangheng/DexTrack/wuji_retarget_pen_min"
SCENE = f"{PROJ}/assets/scenes/single_hand_obj.xml"
CFG = f"{PROJ}/config/hocap.yaml"
HOCAP_DIR = "/tmp/hocap_poc/grab_hocap"
CLIP = "grab_s2_apple_lift"
MESH = "/tmp/hocap_poc/grab_hocap/assets/apple/mesh_med.stl"
TOPO_NPZ = "/tmp/hocap_poc/export/motions/grab_s2_apple_lift/motion.npz"
FPOS = "/data/home/liangheng/DexTrack/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_ori_grab_s2_apple_lift_nf_300.npy"
OUT = "/data/home/liangheng/DexTrack/mjlab_topo_vs_fpos_apple_s2.mp4"

H, W = 540, 540
AZ, EL, DIST = 130.0, -18.0, 0.30
FPS = 30

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def build():
    scene_xml = materialize_hocap_scene(SCENE, hand_mjcf_dir=None)
    cfg = HandRetargetConfig.from_yaml(CFG, mjcf_path=str(scene_xml), overrides={"model.hand_side": "right"})
    rt = InteractionMeshHandRetargeter(cfg)
    clip = load_hocap_clip(resolve_hocap_clip_paths(CLIP, HOCAP_DIR))["right"]
    qpos_topo = rt.retarget_sequence(clip)                       # (T,26) also injects apple mesh
    hand = rt.runtime.hand
    assert hand.has_object and hand.nq == qpos_topo.shape[1] == 26, (hand.nq, qpos_topo.shape)

    # apple pose in palm frame (matches penetration-measurement geometry)
    pb = clip.playback
    lmw = pb.landmarks_world; owp = pb.object_position_world
    oqw = pb.object_quaternion_world_xyzw; wqw = pb.wrist_quaternion_world_xyzw
    T = len(qpos_topo)
    ap_pos_t = np.zeros((T, 3)); ap_q_t = np.zeros((T, 4))       # quat xyzw for set_object_pose
    for t in range(T):
        Ral = R.from_quat(wqw[t]).as_matrix()
        ap_pos_t[t] = (owp[t] - lmw[t, 0]) @ Ral
        R_obj_palm = Ral.T @ R.from_quat(oqw[t]).as_matrix()
        ap_q_t[t] = R.from_matrix(R_obj_palm).as_quat()

    # FPOS
    fp = np.load(FPOS, allow_pickle=True).item()
    qpos_fpos = fp["robot_delta_states_weights_np"].astype(np.float64)
    ap_pos_f = fp["object_transl"].astype(np.float64)
    ap_q_f = fp["object_rot_quat"].astype(np.float64)            # xyzw (validated)

    # per-frame min phi (mm) for both, using the collision model
    dphi = np.load(TOPO_NPZ, allow_pickle=True)["phi_mm_r"]
    minphi_topo = np.array([np.nanmin(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in dphi])
    minphi_fpos = np.full(T, np.nan)
    for t in range(T):
        hand.set_object_pose(ap_pos_f[t], ap_q_f[t]); hand.forward(qpos_fpos[t])
        pf = _per_finger_phi_mm(hand)
        minphi_fpos[t] = np.nanmin(pf[np.isfinite(pf)]) if np.isfinite(pf).any() else np.nan

    # visual tweaks on the shared model
    model = hand.model
    oid = hand._obj_geom_id
    model.geom_rgba[oid] = [0.90, 0.32, 0.28, 0.55]             # translucent apple -> see penetration
    model.geom_group[oid] = 0                                    # ensure apple is in a visible group
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if fid >= 0:
        model.geom_rgba[fid][3] = 0.0
    return (hand, model,
            dict(topo=(qpos_topo, ap_pos_t, ap_q_t, minphi_topo),
                 fpos=(qpos_fpos, ap_pos_f, ap_q_f, minphi_fpos)), T)


def render_frame(renderer, hand, model, qpos, ap_pos, ap_q, cam, opt):
    hand.set_object_pose(ap_pos, ap_q)
    hand.data.qpos[:26] = qpos
    mujoco.mj_forward(model, hand.data)
    palm = hand.get_body_pos("right_palm_link")
    cam.lookat[:] = 0.5 * (palm + ap_pos)                        # frame both hand and apple
    renderer.update_scene(hand.data, cam, scene_option=opt)
    return renderer.render().copy()


def annotate(img, title, color, minphi, t, T):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, im.width, 40], fill=(0, 0, 0, 150))
    dr.text((10, 6), title, font=FONT, fill=color)
    pen = -minphi if (minphi is not None and np.isfinite(minphi)) else 0.0
    dr.rectangle([0, im.height - 34, im.width, im.height], fill=(0, 0, 0, 150))
    dr.text((10, im.height - 30), f"max finger penetration: {max(pen,0):.2f} mm", font=FONT_S, fill=(255, 230, 120))
    dr.text((im.width - 120, im.height - 30), f"frame {t}/{T}", font=FONT_S, fill=(200, 200, 200))
    return np.asarray(im)


def main():
    preview = "--preview" in sys.argv
    hand, model, seq, T = build()
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = AZ, EL, DIST
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1                                         # show all geom groups
    opt.sitegroup[:] = 0                                         # hide fingertip site axes

    qL, pL, qLq, phiL = seq["topo"]
    qR, pR, qRq, phiR = seq["fpos"]

    if preview:
        for t in [0, T // 2, int(np.nanargmin(phiR))]:
            l = annotate(render_frame(renderer, hand, model, qL[t], pL[t], qLq[t], cam, opt), "TopoRetarget", (120, 255, 140), phiL[t], t, T)
            r = annotate(render_frame(renderer, hand, model, qR[t], pR[t], qRq[t], cam, opt), "FPOS_v1 (old)", (255, 150, 150), phiR[t], t, T)
            Image.fromarray(np.hstack([l, r])).save(f"/tmp/hocap_poc/preview_{t}.png")
            print(f"preview frame {t}: topo pen={-phiL[t]:.2f}mm fpos pen={-phiR[t]:.2f}mm -> /tmp/hocap_poc/preview_{t}.png")
        return

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2*W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for t in range(T):
        l = annotate(render_frame(renderer, hand, model, qL[t], pL[t], qLq[t], cam, opt), "TopoRetarget", (120, 255, 140), phiL[t], t, T)
        r = annotate(render_frame(renderer, hand, model, qR[t], pR[t], qRq[t], cam, opt), "FPOS_v1 (old)", (255, 150, 150), phiR[t], t, T)
        ff.stdin.write(np.hstack([l, r]).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    print(f"WROTE {OUT}  ({T} frames, {2*W}x{H}@{FPS}fps)")
    print(f"  topo max pen over seq: {-np.nanmin(phiL):.2f}mm   fpos max pen: {-np.nanmin(phiR):.2f}mm")


if __name__ == "__main__":
    main()
