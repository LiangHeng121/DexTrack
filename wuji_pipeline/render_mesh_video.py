"""Mesh-based offscreen render of a wuji sim rollout (pyrender + EGL), headless.

Loads the actual wuji link meshes + a cube + a ground plane, FKs the hand at each
rollout frame (shadow_hand_dof_pos) and places the cube at its real sim pose
(object_pose), renders with pyrender offscreen (EGL). Camera follows the palm.

Usage: python render_mesh_video.py <rollout.npy> <out.mp4> [env] [ref]
  ref=1 -> render the reference npy (robot_delta_states_weights_np) instead.
"""
import os, sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh, pyrender
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rot
import imageio.v2 as imageio

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "wuji_pipeline/out/wuji_mesh.mp4"
ENV = int(sys.argv[3]) if len(sys.argv) > 3 else 1
IS_REF = len(sys.argv) > 4 and sys.argv[4] == "1"
FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
MD = "assets/wuji_hand_description/meshes/right"
STRIDE = 2; W = H = 480

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])

# link name -> (frame id, pyrender mesh)
link_names = ["right_palm_link"] + \
    [f"right_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)] + \
    [f"right_finger{f}_tip_link" for f in range(1, 6)]
hand_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.85, 0.87, 0.9, 1.0], roughnessFactor=0.6)
links = []
for ln in link_names:
    tm = trimesh.load(f"{MD}/{ln}.STL", process=False)
    links.append((m.getFrameId(ln), pyrender.Mesh.from_trimesh(tm, material=hand_mat)))

cube_tm = trimesh.creation.box([0.05, 0.05, 0.05])
cube_mesh = pyrender.Mesh.from_trimesh(cube_tm, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.9, 0.5, 0.2, 1.0]))
ground = pyrender.Mesh.from_trimesh(trimesh.creation.box([1.0, 1.0, 0.002]),
                                    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.5, 0.5, 0.55, 1.0]))

# load motion
if IS_REF:
    ref = np.load(SRC, allow_pickle=True).item()
    hand = ref["robot_delta_states_weights_np"]; opos = ref["object_transl"]; oqt = ref["object_rot_quat"]
else:
    d = np.load(SRC, allow_pickle=True).item(); ts = sorted([k for k in d.keys() if isinstance(k, int)])
    hand = np.array([d[t]["shadow_hand_dof_pos"][ENV] for t in ts])
    opos = np.array([d[t]["object_pose"][ENV, :3] for t in ts])
    oqt = np.array([d[t]["object_pose"][ENV, 3:7] for t in ts])
Tn = len(hand)

def lookat(eye, tgt, up=np.array([0, 0, 1.0])):
    f = (tgt - eye); f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M

cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0)
renderer = pyrender.OffscreenRenderer(W, H)

frames = []
for k in range(0, Tn, STRIDE):
    qq = np.zeros(m.nq); qq[r2p] = hand[k]
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    sc = pyrender.Scene(bg_color=[0.1, 0.1, 0.12, 1.0], ambient_light=[0.3, 0.3, 0.3])
    for fid, msh in links:
        T = np.eye(4); P = data.oMf[fid]; T[:3, :3] = P.rotation; T[:3, 3] = P.translation
        sc.add(msh, pose=T)
    Tc = np.eye(4); Tc[:3, :3] = Rot.from_quat(oqt[k]).as_matrix(); Tc[:3, 3] = opos[k]
    sc.add(cube_mesh, pose=Tc)
    Tg = np.eye(4); Tg[2, 3] = 0.0; sc.add(ground, pose=Tg)
    palm = data.oMf[m.getFrameId("right_palm_link")].translation
    if IS_REF:
        eye = palm + np.array([0.28, -0.28, 0.12]); tgt = palm   # follow hand (object held)
    else:
        tgt = np.array([opos[0, 0], opos[0, 1], 0.22])           # fixed view: cube + lift path
        eye = tgt + np.array([0.5, -0.5, 0.35])
    cp = lookat(eye, tgt)
    sc.add(cam, pose=cp); sc.add(light, pose=cp)
    color, _ = renderer.render(sc)
    frames.append(color.copy())

imageio.mimsave(OUT, frames, fps=20, codec="libx264")
print(f"saved {len(frames)} frames -> {OUT}")
