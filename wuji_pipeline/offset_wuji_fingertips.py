"""Pure-hand fingertip offset (object-independent).

The retargeting matched the wuji `tip_link` ORIGIN (bone/joint) to the MANO
fingertip, but the actual contact surface is the fingertip mesh, offset ~0.6cm
from the origin (toward the distal pad). Fix: retract each finger along its own
distal axis so the fingertip-mesh centroid lands where the tip_link origin
currently is. This is a fixed per-finger hand-geometry offset (o_f = mesh
centroid in the tip_link frame); it does NOT use the object at all, so it is
consistent for any object pose/shape (no orientation-dependent behavior).

Run in wuji-retarget env (pinocchio + trimesh). Saves plain array.
"""
import argparse
import numpy as np
import pinocchio as pin
import trimesh

_ap = argparse.ArgumentParser()
_ap.add_argument("--ref", default="isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
_ap.add_argument("--out", default="/tmp/wuji_offset_states.npy", help="if .npy dict path, writes full ref dict (object from --ref + offset fingers); legacy default writes q_all only")
_a = _ap.parse_args()
FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
MESHDIR = "assets/wuji_hand_description/meshes/right"
REF = _a.ref
ITERS = 20
STEP = 0.5
DAMP = 1e-3

m = pin.buildModelFromUrdf(FLY); data = m.createData()
jqi = {m.names[j]: m.joints[j].idx_q for j in range(1, m.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
tip_ids = [m.getFrameId(f"right_finger{i}_tip_link") for i in range(1, 6)]
fjq = [r2p[6 + f * 4: 6 + f * 4 + 4] for f in range(5)]
lo = m.lowerPositionLimit; hi = m.upperPositionLimit

# per-finger offset vector o_f = (palmar direction) * (pad extent), in the tip_link frame.
# Palmar direction = direction the tip moves when the finger flexes (kinematic, object-
# independent). Pad extent = how far the mesh sticks out on the palmar side. Retracting
# the finger so this pad point reaches the bone position pulls the actual contact pad
# (not the bone) onto the matched surface -- the correct "pad->back" offset.
o_f = []
for i in range(5):
    pin.forwardKinematics(m, data, np.zeros(m.nq)); pin.updateFramePlacements(m, data)
    R0 = data.oMf[tip_ids[i]].rotation.copy(); p0 = data.oMf[tip_ids[i]].translation.copy()
    qf = np.zeros(m.nq)
    for j in fjq[i][1:]:
        qf[j] = 0.3
    pin.forwardKinematics(m, data, qf); pin.updateFramePlacements(m, data)
    palmar = R0.T @ (data.oMf[tip_ids[i]].translation - p0)
    palmar /= np.linalg.norm(palmar)
    v = np.asarray(trimesh.load(f"{MESHDIR}/right_finger{i+1}_tip_link.STL", process=False).vertices)
    pad_extent = (v @ palmar).max()
    o_f.append(palmar * pad_extent)
o_f = np.array(o_f)
print(f"[offset] per-finger pad offset |o_f| (m): {np.round(np.linalg.norm(o_f, axis=1), 4)}")

def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])

ref = np.load(REF, allow_pickle=True).item()
q_all = ref["robot_delta_states_weights_np"].copy()
T = len(q_all)

for t in range(T):
    qq = np.zeros(m.nq); qq[r2p] = q_all[t]
    pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
    p0 = [data.oMf[tip_ids[f]].translation.copy() for f in range(5)]   # original bone positions
    for it in range(ITERS):
        pin.forwardKinematics(m, data, qq); pin.updateFramePlacements(m, data)
        for f in range(5):
            Tf = data.oMf[tip_ids[f]]
            c = Tf.translation + Tf.rotation @ o_f[f]          # mesh centroid in world
            err = p0[f] - c                                    # move centroid back to bone pos
            if np.linalg.norm(err) < 1e-4:
                continue
            J = pin.computeFrameJacobian(m, data, qq, tip_ids[f], pin.LOCAL_WORLD_ALIGNED)
            r = c - Tf.translation
            Jp = J[:3] - skew(r) @ J[3:6]
            Jf = Jp[:, fjq[f]]
            dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + DAMP * np.eye(3), err)
            qq[fjq[f]] = np.clip(qq[fjq[f]] + STEP * dq, lo[fjq[f]], hi[fjq[f]])
    q_all[t, 6:] = qq[r2p[6:]]

if _a.out.endswith(".npy") and _a.out != "/tmp/wuji_offset_states.npy":
    out = dict(object_transl=ref["object_transl"], object_rot_quat=ref["object_rot_quat"],
               robot_delta_states_weights_np=q_all.astype(np.float32))
    import os
    os.makedirs(os.path.dirname(_a.out), exist_ok=True)
    np.save(_a.out, out)
    print(f"[offset] applied pure-hand offset to all {T} frames -> {_a.out} (full ref dict)")
else:
    np.save("/tmp/wuji_offset_states.npy", q_all.astype(np.float32))
    print(f"[offset] applied pure-hand offset to all {T} frames -> /tmp/wuji_offset_states.npy")
