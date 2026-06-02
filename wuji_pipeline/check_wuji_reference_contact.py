"""B7 (headless numeric): does the assembled wuji reference put the fingers near
the object? FK the wuji fly URDF at each reference frame (q26 = global6+finger20)
with pinocchio, get the 5 fingertip world positions, compare to object_transl.

If fingertips stay far from the object during the grasp phase, the reused-allegro
global placement is wrong and needs calibration.

Run in wuji-retarget env (pinocchio).
"""
import numpy as np
import pinocchio as pin

FLY_URDF = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
REF = "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy"
TIP_FRAMES = [f"right_finger{i}_tip_link" for i in range(1, 6)]

model = pin.buildModelFromUrdf(FLY_URDF)
data = model.createData()
print(f"pinocchio model nq={model.nq} (expect 26)")
# pinocchio joint order may differ from URDF order -> build name->q index map
joint_q_index = {}
for jid in range(1, model.njoints):
    jname = model.names[jid]
    joint_q_index[jname] = model.joints[jid].idx_q

URDF_DOF_ORDER = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
                 [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
# reference column k corresponds to URDF_DOF_ORDER[k]; map to pinocchio q index
ref2pin = np.array([joint_q_index[n] for n in URDF_DOF_ORDER])

ref = np.load(REF, allow_pickle=True).item()
qpos = ref["robot_delta_states_weights_np"]      # (300,26) in URDF_DOF_ORDER
obj = ref["object_transl"]                        # (300,3)
T = qpos.shape[0]
tip_ids = [model.getFrameId(n) for n in TIP_FRAMES]

dists = np.zeros((T, 5))
tip_world = np.zeros((T, 5, 3))
for t in range(T):
    q = np.zeros(model.nq)
    q[ref2pin] = qpos[t]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    for k, fid in enumerate(tip_ids):
        p = data.oMf[fid].translation
        tip_world[t, k] = p
        dists[t, k] = np.linalg.norm(p - obj[t])

# grasp phase ~ when object moves (from B5 the ref motion is ~frame 44-298)
mind = dists.min(axis=1)   # closest fingertip per frame
print(f"\nfingertip -> object-center distance (m), over {T} frames:")
print(f"  closest-tip dist:   mean={mind.mean():.3f}  min={mind.min():.3f}  max={mind.max():.3f}")
print(f"  per-finger mean:    {np.round(dists.mean(0),3)}")
print(f"  grasp phase (f120-280) closest-tip mean: {mind[120:280].mean():.3f}")
print(f"\n  cubesmall is ~0.05m; good grasp => closest tip within ~0.03-0.08m of center.")
print(f"  hand bbox (tip spread) mean: {np.linalg.norm(tip_world.max(1)-tip_world.min(1),axis=1).mean():.3f}")
# compare: object center to palm? sanity that hand is on the right side
print(f"  object center mean pos: {obj.mean(0).round(3)}   tip cloud mean pos: {tip_world.reshape(-1,3).mean(0).round(3)}")
