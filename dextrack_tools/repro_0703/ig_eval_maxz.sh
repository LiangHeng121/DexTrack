#!/usr/bin/env bash
cd /home/liangh/DexTrack/isaacgymenvs
source /home/liangh/miniconda3/etc/profile.d/conda.sh
conda activate dextrack
RES=/tmp/eval_results.txt; : > $RES
CUBE=logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle/wuji_cubesmall_pinall3/20260615_002635/ckpt/best_ep_2816_rew_244.60.pth
APPLE=logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_apple/wuji_apple/20260615_033134/ckpt/best_ep_4598_rew_179.26.pth
CUP=logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_cup/wuji_cup/20260615_004253/ckpt/best_ep_3107_rew_165.26.pth
OBJ3=logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_3obj/wuji_cup_apple_cubesmall/20260615_091354/ckpt/best_ep_3691_rew_166.17.pth

extract() {  # label seq
  python - "$1" "$2" <<'PY'
import sys, glob, os, numpy as np
label, seq = sys.argv[1], sys.argv[2]
ds = sorted(glob.glob(f"logs_test/wuji_eval/{seq}/*/ts_to_hand_obj_obs*.npy"), key=os.path.getmtime)
if not ds:
    print(f"{label} {seq} NO_NPY"); sys.exit()
arr = np.load(ds[-1], allow_pickle=True).item()
ts = sorted(k for k in arr if isinstance(k, int))
objz = np.stack([arr[t]['object_pose'][:, 2] for t in ts])
refz = np.stack([arr[t]['goal_pose_ref_np'][:, 2] for t in ts])
mz, rk = objz.max(0), refz.max(0)
ok = '✓' if np.median(mz) >= np.median(rk)-0.05 else '✗'
open("/tmp/eval_results.txt","a").write(
    f"{label} {seq} mz={np.median(mz):.3f} ref={np.median(rk):.3f} lift%={(mz>=rk-0.05).mean()*100:.0f} {ok}\n")
print(f"{label} {seq} -> {ok}")
PY
}
run() { bash /tmp/wuji_eval_test.sh 3 "$2" "$1" True > /tmp/eval_seq_$2.log 2>&1; extract "$3" "$2"; }

for s in s1 s2 s5 s6 s8 s10; do run "$CUBE" ori_grab_${s}_cubesmall_lift cube; done
for s in s1 s3 s4 s5 s6 s8 s9 s10; do run "$CUP" ori_grab_${s}_cup_lift cup; done
for s in s1 s2 s3 s4 s6 s7 s8 s9; do run "$APPLE" ori_grab_${s}_apple_lift apple; done
for s in ori_grab_s2_cubesmall_lift ori_grab_s8_cup_lift ori_grab_s2_apple_lift; do run "$OBJ3" "$s" 3obj; done
echo "=CAMPAIGN_DONE="
