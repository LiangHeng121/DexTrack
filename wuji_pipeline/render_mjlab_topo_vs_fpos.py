"""Aligned side-by-side (mjlab fly hand + x1.25 apple): LEFT=TopoRetarget | RIGHT=FPOS_v1.

Same coordinate system + same fixed camera + pixel-comparable:
  Both references are transformed into the OBJECT frame -> apple is pinned at the
  world ORIGIN with identity orientation for BOTH panels, and each hand is posed
  by its own hand-relative-to-object pose. Object absolute trajectory is dropped
  (irrelevant); only the hand-object relative geometry (=> penetration) is shown.

Rendered with the actual training fly-hand MJCF (right_fly_mjlab.xml) + the GRAB
apple mesh x1.25 (grab_object_cfg convention), kinematic replay (teleport, no
physics) so penetration is exposed. Runs in the wuji_retarget_pen_min pixi env
(mujoco + trimesh + hand_retarget for the FPOS phi labels). MUJOCO_GL=egl, GPU4.

qpos26 = [wrist tx,ty,tz, wrist rx,ry,rz (intrinsic XYZ), finger1..5 joint1..4].
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

DEX = "/data/home/liangheng/DexTrack"
HAND = f"{DEX}/wuji-mjlab/src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_fly_mjlab.xml"
APPLE_PLY = f"{DEX}/GRAB/unzipped/tools/object_meshes/contact_meshes/apple.ply"
SCALE = 1.25
TOPO_NPZ = "/tmp/hocap_poc/export_125/motions/grab_s2_apple_lift/motion.npz"
MESH125 = "/tmp/hocap_poc/grab_hocap_125/assets/apple/mesh_med.stl"
FPOS = f"{DEX}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_ori_grab_s2_apple_lift_nf_300.npy"
OUT = f"{DEX}/mjlab_topo_vs_fpos_apple_s2_aligned.mp4"

W, H, FPS = 560, 560, 30
CAM = dict(lookat=[0.0, 0.0, 0.0], distance=0.34, azimuth=140, elevation=-14)
try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    Fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except Exception:
    F = Fs = ImageFont.load_default()


def q26_objframe(t_wrist, R_wrist, t_obj, R_obj, fingers):
    """Hand-in-object-frame qpos26 (object pinned at origin+identity)."""
    T = len(fingers)
    out = np.zeros((T, 26))
    for t in range(T):
        Rrel = R_obj[t].T @ R_wrist[t]
        out[t, :3] = R_obj[t].T @ (t_wrist[t] - t_obj[t])
        out[t, 3:6] = R.from_matrix(Rrel).as_euler("XYZ")
        out[t, 6:] = fingers[t]
    return out


def load_topo():
    d = np.load(TOPO_NPZ, allow_pickle=True)
    t_w = d["retargeted_wrist_t_r"].astype(np.float64)
    R_w = R.from_quat(d["retargeted_wrist_q_r"][:, [1, 2, 3, 0]]).as_matrix()
    t_o = d["object_t"][:, 0, :].astype(np.float64)
    R_o = R.from_quat(d["object_q"][:, 0, [1, 2, 3, 0]]).as_matrix()
    fing = d["retargeted_qpos_r"].astype(np.float64)
    phi = d["phi_mm_r"]
    minphi = np.array([np.nanmin(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan for r in phi])
    return q26_objframe(t_w, R_w, t_o, R_o, fing), minphi


def load_fpos():
    fp = np.load(FPOS, allow_pickle=True).item()
    q = fp["robot_delta_states_weights_np"].astype(np.float64)
    t_w = q[:, :3]
    R_w = R.from_euler("XYZ", q[:, 3:6]).as_matrix()
    t_o = fp["object_transl"].astype(np.float64)
    R_o = R.from_quat(fp["object_rot_quat"]).as_matrix()   # xyzw
    q26 = q26_objframe(t_w, R_w, t_o, R_o, q[:, 6:26])
    # phi vs x1.25 apple, using the same WH110 collision query as TopoRetarget
    from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
    from hand_retarget.io.hocap import materialize_hocap_scene
    from hand_retarget.io.hocap_export import _per_finger_phi_mm
    P = "/data/home/liangheng/DexTrack/wuji_retarget_pen_min"
    sx = materialize_hocap_scene(f"{P}/assets/scenes/single_hand_obj.xml", hand_mjcf_dir=None)
    cfg = HandRetargetConfig.from_yaml(f"{P}/config/hocap.yaml", mjcf_path=str(sx), overrides={"model.hand_side": "right"})
    hand = InteractionMeshHandRetargeter(cfg).runtime.hand
    hand.inject_object_mesh(MESH125, "right")
    minphi = np.full(len(q), np.nan)
    for t in range(len(q)):
        hand.set_object_pose(t_o[t], fp["object_rot_quat"][t]); hand.forward(q[t])
        pf = _per_finger_phi_mm(hand)
        minphi[t] = np.nanmin(pf[np.isfinite(pf)]) if np.isfinite(pf).any() else np.nan
    return q26, minphi


def build_render_model():
    spec = mujoco.MjSpec.from_file(HAND)
    spec.visual.global_.offwidth = W
    spec.visual.global_.offheight = H
    md = (pathlib.Path(HAND).parent / spec.meshdir).resolve()
    spec.assets = {f"{spec.meshdir}/{f.name}": f.read_bytes() for f in md.glob("*.STL")}
    wb = spec.worldbody
    wb.add_light(pos=[0.2, 0.2, 0.6], dir=[-1, -1, -2])
    wb.add_light(pos=[-0.2, -0.2, 0.6], dir=[1, 1, -2])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], pos=[0, 0, -0.6], rgba=[0.5, 0.5, 0.55, 1])
    # apple: inline mesh (x1.25 about origin) pinned at world origin, translucent, visible group
    tm = trimesh.load(APPLE_PLY, force="mesh"); tm.apply_scale(SCALE)
    mesh = spec.add_mesh(); mesh.name = "apple"
    mesh.uservert = tm.vertices.astype(float).flatten().tolist()
    mesh.userface = tm.faces.astype(int).flatten().tolist()
    ob = wb.add_body(name="apple", pos=[0, 0, 0])
    g = ob.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = "apple"
    g.rgba = [0.90, 0.32, 0.28, 0.55]; g.group = 2; g.contype = 0; g.conaffinity = 0
    model = spec.compile()
    return model, mujoco.MjData(model)


def annotate(img, title, color, minphi, t, T):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, im.width, 40], fill=(0, 0, 0, 150)); dr.text((10, 6), title, font=F, fill=color)
    pen = max(-minphi, 0.0) if (minphi is not None and np.isfinite(minphi)) else 0.0
    dr.rectangle([0, im.height - 34, im.width, im.height], fill=(0, 0, 0, 150))
    dr.text((10, im.height - 30), f"max finger penetration: {pen:.2f} mm", font=Fs, fill=(255, 230, 120))
    dr.text((im.width - 118, im.height - 30), f"frame {t}/{T}", font=Fs, fill=(200, 200, 200))
    return np.asarray(im)


def main():
    preview = "--preview" in sys.argv
    qL, phiL = load_topo()
    qR, phiR = load_fpos()
    T = min(len(qL), len(qR))
    model, data = build_render_model()
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM["lookat"]; cam.distance = CAM["distance"]; cam.azimuth = CAM["azimuth"]; cam.elevation = CAM["elevation"]
    opt = mujoco.MjvOption(); opt.sitegroup[:] = 0

    def frame(q):
        data.qpos[:26] = q; mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        return renderer.render().copy()

    if preview:
        for t in [0, T // 2, int(np.nanargmin(phiR))]:
            l = annotate(frame(qL[t]), "TopoRetarget", (120, 255, 140), phiL[t], t, T)
            r = annotate(frame(qR[t]), "FPOS_v1 (old)", (255, 150, 150), phiR[t], t, T)
            Image.fromarray(np.hstack([l, r])).save(f"/tmp/hocap_poc/aligned_{t}.png")
            print(f"preview {t}: topo pen={max(-phiL[t],0):.2f}mm fpos pen={max(-phiR[t],0):.2f}mm -> /tmp/hocap_poc/aligned_{t}.png")
        return

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2*W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for t in range(T):
        l = annotate(frame(qL[t]), "TopoRetarget", (120, 255, 140), phiL[t], t, T)
        r = annotate(frame(qR[t]), "FPOS_v1 (old)", (255, 150, 150), phiR[t], t, T)
        ff.stdin.write(np.hstack([l, r]).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    print(f"WROTE {OUT} ({T} frames {2*W}x{H}@{FPS})")
    print(f"  topo max pen {max(-np.nanmin(phiL),0):.2f}mm | fpos max pen {max(-np.nanmin(phiR),0):.2f}mm")


if __name__ == "__main__":
    main()
