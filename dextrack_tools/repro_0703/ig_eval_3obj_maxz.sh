#!/usr/bin/env bash
cd /home/liangh/DexTrack/isaacgymenvs
source /home/liangh/miniconda3/etc/profile.d/conda.sh
conda activate dextrack
RES=/tmp/eval_3obj_results.txt; : > $RES
OBJ3=logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_3obj/wuji_cup_apple_cubesmall/20260615_091354/ckpt/best_ep_3691_rew_166.17.pth

extract() {  # seq
  python - "$1" <<'PY'
import sys, glob, os, numpy as np
seq = sys.argv[1]
ds = sorted(glob.glob(f"logs_test/wuji_eval3/{seq}/*/ts_to_hand_obj_obs*.npy"), key=os.path.getmtime)
if not ds:
    print(f"{seq} NO_NPY"); sys.exit()
arr = np.load(ds[-1], allow_pickle=True).item()
ts = sorted(k for k in arr if isinstance(k, int))
objz = np.stack([arr[t]['object_pose'][:, 2] for t in ts])
refz = np.stack([arr[t]['goal_pose_ref_np'][:, 2] for t in ts])
mz, rk = objz.max(0), refz.max(0)
ok = '1' if np.median(mz) >= np.median(rk)-0.05 else '0'
open("/tmp/eval_3obj_results.txt","a").write(
    f"{seq} mz={np.median(mz):.3f} ref={np.median(rk):.3f} {ok}\n")
print(f"{seq} -> {ok}")
PY
}
# 用独立 log_path 前缀避免和之前的wuji_eval冲突: sed 脚本改 SCRIPT_STEM
run() { SCRIPT_STEM=wuji_eval3 bash /tmp/wuji_eval_test.sh 3 "$1" "$OBJ3" True > /tmp/e3_$1.log 2>&1; extract "$1"; }

for s in s1 s2 s5 s6 s8 s10; do run ori_grab_${s}_cubesmall_lift; done
for s in s1 s3 s4 s5 s6 s8 s9 s10; do run ori_grab_${s}_cup_lift; done
for s in s1 s2 s3 s4 s6 s7 s8 s9; do run ori_grab_${s}_apple_lift; done
echo "=E3_DONE="
