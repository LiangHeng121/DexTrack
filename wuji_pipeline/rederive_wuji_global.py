"""B8: re-derive wuji global 6-DOF so fingertips actually reach the object.

Principle: B4 retargeting already matched wuji fingertips to MANO fingertips in
the wrist frame. So placing the wuji palm at the MANO wrist pose (in DexTrack
frame) lands wuji fingertips on the human fingertips == on the object.

Steps:
  1. Recover GRAB->DexTrack similarity (s, R_world, t) from the object:
     R_world from object orientation (R_ref @ R_grab^-1, frame-invariant, averaged),
     s,t from object positions by least squares.
  2. Transform MANO fingertips (B3a, GRAB frame) -> DexTrack frame target Q (300,5,3).
  3. Per-frame Kabsch: align wuji fingertips (FK, palm frame) to Q -> palm SE(3)
     -> fly URDF global 6-DOF [WRJ0 x/y/z + rx/ry/rz].
  4. Reassemble reference npy + inline contact re-check.

Run in wuji-retarget env (pinocchio).
"""
import argparse
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--grab", default="GRAB/unzipped/grab/s2/cubesmall_inspect_1.npz")
ap.add_argument("--allegro", default="isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
ap.add_argument("--kp21", default="wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy")
ap.add_argument("--out", default="isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
ap.add_argument("--c0", type=int, default=200, help="GRAB crop start (= assemble fit_crop c0)")
ap.add_argument("--c1", type=int, default=499, help="GRAB crop end (= assemble fit_crop c1)")
_a = ap.parse_args()
GRAB, ALLEGRO, KP21, OUT, C0, C1 = _a.grab, _a.allegro, _a.kp21, _a.out, _a.c0, _a.c1
FLY = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"
TIP_KP = [4, 8, 12, 16, 20]  # MediaPipe-order fingertips in kp21 (thumb,index,middle,ring,pinky)


def resample(sig, n):
    T = len(sig); xs = np.linspace(0, T - 1, n)
    lo = np.floor(xs).astype(int); hi = np.minimum(lo + 1, T - 1); w = xs - lo
    ws = (n,) + (1,) * (sig.ndim - 1)
    return sig[lo] * (1 - w).reshape(ws) + sig[hi] * w.reshape(ws)


def kabsch(P, Q):
    """rigid R,t with Q ~= P@R.T + t."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


# ---- load ----
gnpz = np.load(GRAB, allow_pickle=True)["object"].item()["params"]
grab_ot = np.asarray(gnpz["transl"]); grab_oo = np.asarray(gnpz["global_orient"])  # (776,3)
allg = np.load(ALLEGRO, allow_pickle=True).item()
ref_ot = allg["object_transl"]; ref_oq = allg["object_rot_quat"]                   # (300,*)
kp21 = np.load(KP21)                                                               # (776,21,3)
N = ref_ot.shape[0]

# crop+resample GRAB object & mano tips to the 300 ref frames
g_ot = resample(grab_ot[C0:C1], N)                          # (300,3)
g_oR = Rot.from_rotvec(resample(grab_oo[C0:C1], N))         # (300,) rotation
mano_tip = resample(kp21[C0:C1][:, TIP_KP], N)             # (300,5,3) GRAB frame
mano_wrist = resample(kp21[C0:C1][:, 0], N)                # (300,3)

# ---- 1. similarity GRAB->DexTrack via Umeyama on object positions ----
# (object-orientation gives wrong R because DexTrack object canonical frame differs;
#  positions are unaffected, so fit s,R,t from the 300 object-position correspondences.)
def umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    Sc = src - mu_s; Dc = dst - mu_d
    Sigma = (Dc.T @ Sc) / len(src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t

s, Rw_mat, t = umeyama(g_ot, ref_ot)
Rw = Rot.from_matrix(Rw_mat)
resid = np.linalg.norm(s * (g_ot @ Rw_mat.T) + t - ref_ot, axis=1)
print(f"[similarity] Umeyama scale s={s:.4f}  object-fit residual mean={resid.mean()*1000:.1f}mm (small=good)")

# ---- 2. MANO fingertips -> DexTrack frame ----
def to_dex(p):  # p:(...,3) GRAB frame -> DexTrack
    return s * Rw.apply(p.reshape(-1, 3)).reshape(p.shape) + t
Q_tip = to_dex(mano_tip)                                   # (300,5,3) target tips

# ---- 3. FK wuji fingertips (palm frame, global=0) ----
model = pin.buildModelFromUrdf(FLY); data = model.createData()
jqi = {model.names[j]: model.joints[j].idx_q for j in range(1, model.njoints)}
order = ["WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz"] + \
        [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
r2p = np.array([jqi[n] for n in order])
tip_ids = [model.getFrameId(f"right_finger{i}_tip_link") for i in range(1, 6)]

fingers20 = allg_finger = np.load(OUT, allow_pickle=True).item()["robot_delta_states_weights_np"][:, 6:26]  # (300,20)

global6 = np.zeros((N, 6), dtype=np.float32)
P_tipframe = np.zeros((N, 5, 3))
for tt in range(N):
    q = np.zeros(model.nq); q[r2p[6:]] = fingers20[tt]      # global=0, only fingers
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    P = np.array([data.oMf[fid].translation for fid in tip_ids])  # (5,3) palm frame
    P_tipframe[tt] = P
    R_g, t_g = kabsch(P, Q_tip[tt])                        # palm pose in DexTrack
    rx, ry, rz = Rot.from_matrix(R_g).as_euler("XYZ")
    global6[tt] = [t_g[0], t_g[1], t_g[2], rx, ry, rz]

# ---- 4. reassemble + inline contact check ----
robot_states = np.concatenate([global6, fingers20], axis=1).astype(np.float32)
out = dict(object_transl=ref_ot.astype(np.float32),
           object_rot_quat=ref_oq.astype(np.float32),
           robot_delta_states_weights_np=robot_states)
np.save(OUT, out)

# verify: FK full q (global+finger) tip distance to object center
dists = np.zeros((N, 5))
for tt in range(N):
    q = np.zeros(model.nq); q[r2p] = robot_states[tt]
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    for k, fid in enumerate(tip_ids):
        dists[tt, k] = np.linalg.norm(data.oMf[fid].translation - ref_ot[tt])
mind = dists.min(1)
print(f"[rederive] saved -> {OUT}")
print(f"[verify] closest-tip -> object center: mean={mind.mean():.3f} min={mind.min():.3f} max={mind.max():.3f}")
print(f"[verify]   (was 0.080 with allegro global; cubesmall radius ~0.03)")
print(f"[verify] grasp phase f120-280 closest-tip mean: {mind[120:280].mean():.3f}")
