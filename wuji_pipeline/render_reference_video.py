"""Render the wuji reference (hand skeleton + cube) to an MP4, headless.

FK the hand at each reference qpos (pinocchio), draw each finger as a polyline
through its joint links to the tip, draw the 5cm cube as a wireframe at the
object pose, and write frames to an MP4 (matplotlib Agg + imageio-ffmpeg).
"""
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
REF = "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
OUT = "wuji_pipeline/out/wuji_cubesmall_reference.mp4"
STRIDE = 2
HALF = 0.025  # cube half-edge (5cm)

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
palm_id = m.getFrameId("right_palm_link")
# per finger: link1..4 + tip frame ids
finger_chains = []
for f in range(1, 6):
    ids = [m.getFrameId(f"right_finger{f}_link{k}") for k in range(1, 5)]
    ids.append(m.getFrameId(f"right_finger{f}_tip_link"))
    finger_chains.append(ids)

ref = np.load(REF, allow_pickle=True).item()
q = ref["robot_delta_states_weights_np"]; obj = ref["object_transl"]; oq = ref["object_rot_quat"]
T = len(q)

# cube wireframe edges (in cube frame)
c = HALF * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
frames = []
for t in range(0, T, STRIDE):
    qq = np.zeros(m.nq); qq[r2p] = q[t]
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    palm = data.oMf[palm_id].translation
    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
    # fingers
    for fi, ids in enumerate(finger_chains):
        pts = np.array([palm] + [data.oMf[i].translation for i in ids])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=colors[fi], ms=3, lw=1.5)
        ax.scatter(*pts[-1], color=colors[fi], s=30)  # tip
    ax.scatter(*palm, color="k", s=40)
    # cube
    Rb = Rot.from_quat(oq[t]).as_matrix()
    cw = (Rb @ c.T).T + obj[t]
    for a, b in edges:
        ax.plot(*zip(cw[a], cw[b]), color="dimgray", lw=1.2)
    ax.set_title(f"frame {t}/{T}   obj z={obj[t,2]:.3f}")
    # fixed view centered on object
    ctr = obj[t]; R = 0.12
    ax.set_xlim(ctr[0]-R, ctr[0]+R); ax.set_ylim(ctr[1]-R, ctr[1]+R); ax.set_zlim(ctr[2]-R, ctr[2]+R)
    ax.view_init(elev=18, azim=t * 0.6)
    ax.set_box_aspect((1, 1, 1))
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    frames.append(img.copy())
    plt.close(fig)

imageio.mimsave(OUT, frames, fps=20, codec="libx264")
print(f"saved {len(frames)} frames -> {OUT}")
