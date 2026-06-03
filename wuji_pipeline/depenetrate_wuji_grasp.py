"""B10 (mesh-aware): remove finger-MESH-object penetration from the wuji reference.

Why mesh-aware: the wuji fingertip COLLISION mesh extends ~0.75-1.4cm beyond the
`right_fingerN_tip_link` frame origin (the joint/bone). Earlier de-penetration used
the link origin, so the actual mesh surface still penetrated ~1cm -> the physics
still ejects/jams the object. This version uses the actual fingertip mesh: per
frame, the mesh vertex closest to the object is the contact point, and we nudge
the finger so that contact point sits at (R + clearance) outside the object.

This is object-independent in spirit (it's the hand's real geometry) and general:
any object, any contact point on the tip mesh.

Run in wuji-retarget env (pinocchio). Saves plain array -> assemble in dextrack.
"""
import numpy as np
import pinocchio as pin
import trimesh

FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
MESHDIR = "assets/wuji_hand_description/meshes/right"
REF = "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
R_OBJ = 0.027
CLEAR = 0.003      # keep the actual mesh surface ~3mm outside the object
ITERS = 20
STEP = 0.5
DAMP = 1e-3
NSUB = 120         # subsample tip mesh verts for speed

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
tip_ids = [m.getFrameId(f"right_finger{i}_tip_link") for i in range(1, 6)]
fjq = [r2p[6 + f * 4: 6 + f * 4 + 4] for f in range(5)]
lo = m.lowerPositionLimit; hi = m.upperPositionLimit

# tip mesh vertices in each tip-link frame (subsampled)
tipverts = []
for i in range(1, 6):
    v = np.asarray(trimesh.load(f"{MESHDIR}/right_finger{i}_tip_link.STL", process=False).vertices)
    idx = np.linspace(0, len(v) - 1, min(NSUB, len(v))).astype(int)
    tipverts.append(v[idx])

def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

ref = np.load(REF, allow_pickle=True).item()
q_all = ref["robot_delta_states_weights_np"].copy()
obj = ref["object_transl"]
T = len(q_all)

n_fixed = 0
for t in range(T):
    qq = np.zeros(m.nq); qq[r2p] = q_all[t]
    any_it = 0
    for it in range(ITERS):
        pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
        moved = False
        for f in range(5):
            Tf = data.oMf[tip_ids[f]]
            vw = (Tf.rotation @ tipverts[f].T).T + Tf.translation     # mesh verts -> world
            dd = np.linalg.norm(vw - obj[t], axis=1)
            k = dd.argmin(); c = vw[k]; dc = dd[k]                    # closest mesh point
            if dc < R_OBJ + CLEAR - 1e-4:
                tgt = obj[t] + (R_OBJ + CLEAR) * (c - obj[t]) / (dc + 1e-9)
                J = pin.computeFrameJacobian(m, data, qq, tip_ids[f], pin.LOCAL_WORLD_ALIGNED)
                r = c - Tf.translation
                Jp = J[:3] - skew(r) @ J[3:6]                          # Jacobian of contact point
                Jf = Jp[:, fjq[f]]
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + DAMP * np.eye(3), tgt - c)
                qq[fjq[f]] = np.clip(qq[fjq[f]] + STEP * dq, lo[fjq[f]], hi[fjq[f]])
                moved = True
        any_it = it
        if not moved:
            break
    if any_it > 0:
        n_fixed += 1
    q_all[t, 6:] = qq[r2p[6:]]

# verify (mesh surface to object)
g = slice(120, 280); mind = np.zeros(T)
perfin = np.zeros((T, 5))
for t in range(T):
    qq = np.zeros(m.nq); qq[r2p] = q_all[t]
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    for f in range(5):
        Tf = data.oMf[tip_ids[f]]
        vw = (Tf.rotation @ tipverts[f].T).T + Tf.translation
        perfin[t, f] = np.linalg.norm(vw - obj[t], axis=1).min() - R_OBJ
    mind[t] = perfin[t].min()
print(f"[depenetrate-mesh] frames adjusted: {n_fixed}/{T}  clearance={CLEAR*1000:.0f}mm")
print(f"[depenetrate-mesh] per-finger MESH-surface dist (grasp,m): {np.round(perfin[g].mean(0),3)}")
print(f"[depenetrate-mesh] min mesh-surface dist over grasp: {mind[g].min():.4f} m  (>=0 = no mesh penetration)")
np.save("/tmp/wuji_depen_states.npy", q_all.astype(np.float32))
print("[depenetrate-mesh] saved -> /tmp/wuji_depen_states.npy")
