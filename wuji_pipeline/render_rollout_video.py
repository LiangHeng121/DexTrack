"""Render an actual sim rollout (hand + physics object) to MP4, headless.

Reads a saved ts_to_hand_obj rollout (shadow_hand_dof_pos = actual sim hand DOF,
object_pose = actual sim object pose) for one env, FKs the hand, draws the cube
at its real sim pose. Camera follows the palm so you can see whether the cube
stays with the hand (gripped) or is left behind / falls (not gripped).
"""
import sys
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROLLOUT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "wuji_pipeline/out/wuji_rollout.mp4"
ENV = int(sys.argv[3]) if len(sys.argv) > 3 else 1
FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
STRIDE = 2; HALF = 0.025

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
palm_id = m.getFrameId("right_palm_link")
chains = [[m.getFrameId(f"right_finger{f}_link{k}") for k in range(1, 5)] +
          [m.getFrameId(f"right_finger{f}_tip_link")] for f in range(1, 6)]

d = np.load(ROLLOUT, allow_pickle=True).item()
ts = sorted([k for k in d.keys() if isinstance(k, int)])
hand = np.array([d[t]["shadow_hand_dof_pos"][ENV] for t in ts])   # (T,26) actual sim DOF
opose = np.array([d[t]["object_pose"][ENV] for t in ts])          # (T,7)

c = HALF * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

frames = []
for k in range(0, len(ts), STRIDE):
    qq = np.zeros(m.nq); qq[r2p] = hand[k]
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    palm = data.oMf[palm_id].translation
    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
    for fi, ids in enumerate(chains):
        pts = np.array([palm] + [data.oMf[i].translation for i in ids])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color=colors[fi], ms=3, lw=1.5)
        ax.scatter(*pts[-1], color=colors[fi], s=30)
    ax.scatter(*palm, color="k", s=40)
    op = opose[k, :3]; Rb = Rot.from_quat(opose[k, 3:7]).as_matrix()
    cw = (Rb @ c.T).T + op
    for a, b in edges:
        ax.plot(*zip(cw[a], cw[b]), color="dimgray", lw=1.4)
    ax.set_title(f"sim step {ts[k]}  obj z={op[2]:.3f}  palm z={palm[2]:.3f}")
    ctr = palm; R = 0.13   # camera follows the palm
    ax.set_xlim(ctr[0]-R, ctr[0]+R); ax.set_ylim(ctr[1]-R, ctr[1]+R); ax.set_zlim(ctr[2]-R, ctr[2]+R)
    ax.view_init(elev=18, azim=k * 0.6); ax.set_box_aspect((1, 1, 1))
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    frames.append(img.copy()); plt.close(fig)

imageio.mimsave(OUT, frames, fps=20, codec="libx264")
print(f"saved {len(frames)} frames -> {OUT}  (env {ENV})")
