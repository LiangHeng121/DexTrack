"""B10: remove finger-object penetration from the wuji reference.

kinematics_only test proved: the hand follows the reference fine, but the
reference fingertips penetrate the object (thumb ~1.8cm inside), so the physics
contact solver violently ejects the (62g) object (flies 5m). No penetration ->
no explosion (like allegro's loose reference), and the policy can close the grip.

Fix: per-frame, per-finger damped-least-squares IK. For any fingertip inside the
object (dist < R + clearance), nudge that finger's joints so the tip moves out to
exactly (R + clearance) along the radial direction. Non-penetrating fingers and
the global 6-DOF are left untouched, so the grasp pose is minimally changed.

Run in wuji-retarget env (pinocchio). Saves plain array -> assemble in dextrack.
"""
import numpy as np
import pinocchio as pin

FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
REF = "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
R_OBJ = 0.027
CLEAR = 0.004      # keep fingertip ~4mm outside the surface (no penetration)
ITERS = 15
STEP = 0.6
DAMP = 1e-3

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
tip_ids = [m.getFrameId(f"right_finger{i}_tip_link") for i in range(1, 6)]
# pinocchio q-indices of each finger's 4 joints
fjq = [r2p[6 + f * 4: 6 + f * 4 + 4] for f in range(5)]
lo = m.lowerPositionLimit; hi = m.upperPositionLimit

ref = np.load(REF, allow_pickle=True).item()
q_all = ref["robot_delta_states_weights_np"].copy()
obj = ref["object_transl"]
T = len(q_all)

def tips_of(qq):
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    return np.array([data.oMf[fid].translation for fid in tip_ids])

n_fixed = 0
for t in range(T):
    qq = np.zeros(m.nq); qq[r2p] = q_all[t]
    for it in range(ITERS):
        P = tips_of(qq)
        moved = False
        for f in range(5):
            dvec = P[f] - obj[t]; dist = np.linalg.norm(dvec)
            if dist < R_OBJ + CLEAR - 1e-4:
                target = obj[t] + (R_OBJ + CLEAR) * dvec / (dist + 1e-9)
                J = pin.computeFrameJacobian(m, data, qq, tip_ids[f], pin.LOCAL_WORLD_ALIGNED)[:3]
                Jf = J[:, fjq[f]]                       # 3x4 for this finger
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + DAMP * np.eye(3), target - P[f])
                qq[fjq[f]] = np.clip(qq[fjq[f]] + STEP * dq, lo[fjq[f]], hi[fjq[f]])
                moved = True
        if not moved:
            break
    if it > 0:
        n_fixed += 1
    q_all[t, 6:] = qq[r2p[6:]]   # write back finger joints (global unchanged)

# verify
g = slice(120, 280); D = np.zeros((T, 5))
for t in range(T):
    qq = np.zeros(m.nq); qq[r2p] = q_all[t]; P = tips_of(qq)
    D[t] = np.linalg.norm(P - obj[t], axis=1)
surf = D[g].mean(0) - R_OBJ
print(f"[depenetrate] frames adjusted: {n_fixed}/{T}  clearance={CLEAR*1000:.0f}mm")
print(f"[depenetrate] per-finger dist to surface (grasp, m): {np.round(surf,3)}  (all >=0 = no penetration)")
print(f"[depenetrate] min any-finger surface dist over grasp: {(D[g]-R_OBJ).min():.4f} m  (>=0 good)")
np.save("/tmp/wuji_depen_states.npy", q_all.astype(np.float32))
print("[depenetrate] saved -> /tmp/wuji_depen_states.npy  (assemble in dextrack)")
