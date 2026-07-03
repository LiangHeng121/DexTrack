"""Per-sequence: fit crop [c0,c1] (object angular-speed match vs reduced_300 allegro ref)
and GRAB->canonical Umeyama similarity {s,R,t}. Emits JSON + canon_sim.npy.

Identical fit_crop to assemble_wuji_reference.py / generate_contact_guidance_grab2.py
and identical umeyama to rederive_wuji_global.py (so results reproduce the PoC exactly).
"""
import argparse, json, os
import numpy as np
from scipy.spatial.transform import Rotation as R


def ang_speed_aa(aa):
    Rs = R.from_rotvec(aa); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def ang_speed_quat(q):
    Rs = R.from_quat(q); return (Rs[:-1].inv() * Rs[1:]).magnitude()
def resample(sig, n):
    T = len(sig); xs = np.linspace(0, T - 1, n)
    lo = np.floor(xs).astype(int); hi = np.minimum(lo + 1, T - 1); w = xs - lo
    ws = (n,) + (1,) * (sig.ndim - 1)
    return sig[lo] * (1 - w).reshape(ws) + sig[hi] * w.reshape(ws)
def fit_crop(gsp, rsp):
    n = len(rsp); rn = (rsp - rsp.mean()) / (rsp.std() + 1e-9); G = len(gsp); best = (1e9, None, None)
    for c0 in range(0, G - n, 4):
        for c1 in range(c0 + n, G, 4):
            g = resample(gsp[c0:c1], n); gn = (g - g.mean()) / (g.std() + 1e-9)
            e = np.mean((gn - rn) ** 2)
            if e < best[0]: best = (e, c0, c1)
    _, b0, b1 = best
    for c0 in range(max(0, b0 - 4), b0 + 5):
        for c1 in range(b1 - 4, min(G, b1 + 5)):
            if c1 - c0 < n: continue
            g = resample(gsp[c0:c1], n); gn = (g - g.mean()) / (g.std() + 1e-9)
            e = np.mean((gn - rn) ** 2)
            if e < best[0]: best = (e, c0, c1)
    return best
def umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    Sc = src - mu_s; Dc = dst - mu_d
    Sigma = (Dc.T @ Sc) / len(src)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    Rm = U @ S @ Vt
    var_s = (Sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * Rm @ mu_s
    return s, Rm, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-npz", required=True)
    ap.add_argument("--allegro-ref", required=True, help="reduced_300 passive_active_info_*.npy")
    ap.add_argument("--out-sim", required=True)
    ap.add_argument("--nframes", type=int, default=300)
    args = ap.parse_args()

    g = np.load(args.grab_npz, allow_pickle=True)["object"].item()["params"]
    grab_aa = np.asarray(g["global_orient"]); grab_ot = np.asarray(g["transl"])
    ref = np.load(args.allegro_ref, allow_pickle=True).item()
    ref_ot = ref["object_transl"]; ref_q = ref["object_rot_quat"]; N = args.nframes

    err, c0, c1 = fit_crop(ang_speed_aa(grab_aa), ang_speed_quat(ref_q))
    g_al = resample(ang_speed_aa(grab_aa)[c0:c1], len(ang_speed_quat(ref_q)))
    corr = float(np.corrcoef(g_al, ang_speed_quat(ref_q))[0, 1])

    g_ot = resample(grab_ot[c0:c1], N)
    s, Rm, t = umeyama(g_ot, ref_ot)
    resid = np.linalg.norm(s * (g_ot @ Rm.T) + t - ref_ot, axis=1)

    info = {"c0": int(c0), "c1": int(c1), "crop_normMSE": float(err), "obj_angspeed_corr": corr,
            "umeyama_s": float(s), "umeyama_resid_mm": float(resid.mean() * 1000)}
    print(json.dumps(info))
    np.save(args.out_sim, {"s": float(s), "R": Rm, "t": t, "c0": int(c0), "c1": int(c1),
                           "crop_normMSE": float(err), "obj_angspeed_corr": corr,
                           "umeyama_resid_mm": float(resid.mean() * 1000)})


if __name__ == "__main__":
    main()
