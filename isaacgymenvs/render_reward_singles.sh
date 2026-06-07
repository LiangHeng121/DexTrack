#!/bin/bash
# test + render the 3 finished single-seq reward-switch policies (#1, #2+#3, #4).
# switches don't change env dynamics/obs (reward-only), so plain test reproduces behavior.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
OUT=/home/liangh/DexTrack/render_videos/reward_exps; mkdir -p "$OUT"
SEQ=ori_grab_s2_cubesmall_inspect_1

run_one(){
  name=$1; gpu=$2; ckpt=$3
  echo "===== $name (gpu=$gpu) ====="
  mk=/tmp/mk_$name; touch "$mk"
  numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh $gpu $SEQ "$ckpt" True > /tmp/test_$name.log 2>&1
  echo "  $name test reward: $(grep -oE 'reward: [-0-9.]+' /tmp/test_$name.log | tail -1)"
  R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | head -1)
  if [ -z "$R" ]; then echo "  !! $name no rollout: $(grep -iE 'error|traceback|no such' /tmp/test_$name.log | grep -ivE 'nullptr' | tail -1 | cut -c1-90)"; return; fi
  env=$(python -c "import numpy as np;d=np.load('$R',allow_pickle=True).item();ts=sorted([k for k in d if isinstance(k,int)]);op=np.array([d[t]['object_pose'][:,:3] for t in ts]);z=op[:,:,2].max(0);print(int(np.argmax(z[1:]))+1)" 2>/dev/null); [ -z "$env" ] && env=1
  CUDA_VISIBLE_DEVICES=$gpu python wuji_isaacgym_playback.py --src "$R" --env $env --hand wuji --obj_code $SEQ --gpu 0 --out "$OUT/${name}.mp4" 2>&1 | grep -E "saved|Error|Traceback" | tail -1
}

B1=logs/grab_single_wuji_fp1/ori_grab_s2_cubesmall_inspect_1/20260606_101143/ckpt/best_ep_943_rew_-54.26.pth
B23=logs/grab_single_wuji_relax23/ori_grab_s2_cubesmall_inspect_1/20260606_095846/ckpt/best_ep_980_rew_48.12.pth
B4=logs/grab_single_wuji_fp4/ori_grab_s2_cubesmall_inspect_1/20260606_104821/ckpt/best_ep_997_rew_121.89.pth

run_one n1_boostfinger  2 "$B1"  &
run_one n23_relaxpalm5  3 "$B23" &
run_one n4_fingerpos    6 "$B4"  &
wait
echo "ALL_SINGLES_RENDERED"
ls -la "$OUT"/*.mp4 2>/dev/null