"""Part A (step 1/2) for reward #4 (fingertip-position tracking): FK each wuji
reference q26 (global6 + finger20) through the wuji fly URDF -> per-frame world
positions of palm + 5 fingertips. Saved as plain-array .npz (cross-numpy-version
safe; the dict-pickle .npy of numpy 2.x can't load in the dextrack numpy 1.24).

Step 2 (assemble_fpos_reference.py, run in dextrack) merges these npz into the
original reference npy -> link_key_to_link_pos, which the task loader reads.

Run THIS in wuji-retarget env (pinocchio)."""
import os, glob, numpy as np, pinocchio as pin

ROOT = "/home/liangh/DexTrack"
FLY_URDF = os.path.join(ROOT, "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf")
SRC = os.path.join(ROOT, "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data")
NPZ = os.path.join(ROOT, "wuji_pipeline/out/fpos_npz")   # step-1 output (plain arrays)

PALM_FRAME = "right_palm_link"
TIP_FRAMES = [f"right_finger{i}_tip_link" for i in range(1, 6)]   # 1=thumb..5=pinky
LINK_FRAMES = [PALM_FRAME] + TIP_FRAMES

model = pin.buildModelFromUrdf(FLY_URDF)
data = model.createData()
assert model.nq == 26, f"expected nq=26, got {model.nq}"

joint_q_index = {model.names[jid]: model.joints[jid].idx_q for jid in range(1, model.njoints)}
URDF_DOF_ORDER = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
                 [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
ref2pin = np.array([joint_q_index[n] for n in URDF_DOF_ORDER])
frame_ids = {n: model.getFrameId(n) for n in LINK_FRAMES}


def fk_link_positions(qpos):
    """qpos: (T,26) in URDF_DOF_ORDER -> {link_name: (T,3)}."""
    T = qpos.shape[0]
    out = {n: np.zeros((T, 3), dtype=np.float32) for n in LINK_FRAMES}
    for t in range(T):
        q = np.zeros(model.nq)
        q[ref2pin] = qpos[t]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for n in LINK_FRAMES:
            out[n][t] = data.oMf[frame_ids[n]].translation
    return out


def main():
    os.makedirs(NPZ, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.npy")))
    print(f"{len(files)} reference files in {SRC}")
    sanity = None
    for i, f in enumerate(files):
        ref = np.load(f, allow_pickle=True).item()
        qpos = ref["robot_delta_states_weights_np"]            # (T,26)
        link_pos = fk_link_positions(qpos)                     # {name:(T,3)}
        out_path = os.path.join(NPZ, os.path.basename(f).replace(".npy", ".npz"))
        np.savez(out_path, **link_pos)                         # plain arrays, version-safe
        if "s2_cubesmall_inspect_1" in f:                      # sanity vs check_wuji_reference_contact
            obj = ref["object_transl"]
            tips = np.stack([link_pos[n] for n in TIP_FRAMES], axis=1)  # (T,5,3)
            d = np.linalg.norm(tips - obj[:, None, :], axis=-1).min(axis=1)
            sanity = (os.path.basename(f), float(d.min()), float(d[44:].mean()))
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} T={qpos.shape[0]}")
    print(f"\nwrote {len(files)} npz to {NPZ}")
    if sanity:
        print(f"sanity ({sanity[0]}): min fingertip->obj dist={sanity[1]*100:.1f}cm, "
              f"grasp-phase mean={sanity[2]*100:.1f}cm (should match check_wuji_reference_contact)")


if __name__ == "__main__":
    main()
