"""Part A (step 2/2) for reward #4: merge the FK fingertip npz (from
add_link_pos_to_reference.py, run in wuji-retarget) into the original wuji
reference npy -> add `link_key_to_link_pos`, re-save with THIS env's numpy so the
dext-rack training (numpy 1.24) can load the dict pickle.

Run in dextrack env (numpy 1.24)."""
import os, glob, numpy as np

ROOT = "/home/liangh/DexTrack"
SRC = os.path.join(ROOT, "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data")
NPZ = os.path.join(ROOT, "wuji_pipeline/out/fpos_npz")
DST = os.path.join(ROOT, "isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data")


def main():
    print("numpy", np.__version__)
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.npy")))
    n = 0
    for f in files:
        npz_path = os.path.join(NPZ, os.path.basename(f).replace(".npy", ".npz"))
        if not os.path.exists(npz_path):
            print("  MISSING npz, skip:", os.path.basename(f)); continue
        ref = np.load(f, allow_pickle=True).item()            # original (1.24-loadable)
        z = np.load(npz_path)
        ref["link_key_to_link_pos"] = {k: z[k].astype(np.float32) for k in z.files}
        np.save(os.path.join(DST, os.path.basename(f)), ref, allow_pickle=True)
        n += 1
    print(f"assembled {n} files -> {DST}")
    # verify a sample round-trips in this numpy
    s = os.path.join(DST, "wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy")
    d = np.load(s, allow_pickle=True).item()
    lk = d["link_key_to_link_pos"]
    print("verify load OK; link keys:", list(lk.keys()))
    print("  shapes:", {k: v.shape for k, v in lk.items()})


if __name__ == "__main__":
    main()
