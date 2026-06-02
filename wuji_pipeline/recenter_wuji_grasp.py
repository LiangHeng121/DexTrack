"""B9: re-center the wuji grasp so the fingertip-cluster centroid coincides with
the object center during the contact phase.

Root cause of the failed grasp: B8's Kabsch alignment to (scaled) MANO fingertips
left the object ~0.041m off the wuji fingertip-cluster centroid, so the fingers
sit beside the object instead of around it. A pure translation correction that
centers the cluster on the object brings 4/5 fingertips to the surface.

Applied as a world-frame translation added to the global DOF (WRJ0x/y/z), ramped
by object lift height so the approach phase (object on table) is untouched and the
correction is full during the lift/grasp.

Run in wuji-retarget env (pinocchio).
"""
import numpy as np
import pinocchio as pin

FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
REF = "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
R_OBJ = 0.027  # cubesmall approx radius

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
tip_ids = [m.getFrameId(f"right_finger{i}_tip_link") for i in range(1, 6)]

ref = np.load(REF, allow_pickle=True).item()
q = ref["robot_delta_states_weights_np"].copy()
obj = ref["object_transl"]
T = len(q)

def fk_tips(qrow):
    qq = np.zeros(m.nq); qq[r2p] = qrow
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    return np.array([data.oMf[fid].translation for fid in tip_ids])  # (5,3)

# per-frame cluster centroid + centering correction
cent = np.array([fk_tips(q[t]).mean(0) for t in range(T)])    # (T,3)
corr = obj - cent                                              # (T,3) world-frame shift

# ramp by object lift height: 0 on table, 1 when lifted
z = obj[:, 2]; z0 = np.percentile(z, 5)
w = np.clip((z - z0) / (0.06), 0.0, 1.0)                       # full once lifted ~6cm
# light smoothing of weight
k = 5; w = np.convolve(w, np.ones(k)/k, mode="same")

q[:, :3] = q[:, :3] + w[:, None] * corr

# ---- verify contact after recenter ----
g = slice(120, 280)
D = np.zeros((T, 5))
for t in range(T):
    P = fk_tips(q[t])
    D[t] = np.linalg.norm(P - obj[t], axis=1)
surf = D[g].mean(0) - R_OBJ
print("[recenter] per-finger dist to surface (grasp, m): thumb/index/middle/ring/pinky")
print("  ", np.round(surf, 3))
print(f"  mean closest-3 to surface: {np.sort(surf)[:3].mean():.3f}   fingers touching(<1cm): {(D[g]-R_OBJ<0.01).sum(1).mean():.2f}/5")
cent2 = np.array([fk_tips(q[t]).mean(0) for t in range(T)])
print(f"  object<->cluster-centroid offset (grasp): {np.linalg.norm((cent2-obj)[g],axis=1).mean():.3f}m  (was ~0.041)")

# save recentered states as a plain array (numpy-version-safe); dextrack reassembles
np.save("/tmp/wuji_recentered_states.npy", q.astype(np.float32))
print("[recenter] saved recentered states -> /tmp/wuji_recentered_states.npy  (assemble in dextrack env)")
