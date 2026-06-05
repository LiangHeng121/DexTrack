#!/bin/bash
# Test + render a policy video for each completed/in-progress training.
# ckpts pre-copied to /tmp/vizckpt/<name>.pth. Sequential on GPU6. Continue on error.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
OUT=/home/liangh/DexTrack/render_videos/all_policies; mkdir -p "$OUT"
GPU=6
run_one(){
  name=$1; script=$2; seq=$3; hand=$4; wdd=$5
  echo "===== $name (seq=$seq hand=$hand) ====="
  touch /tmp/viz_marker
  WUJI_DATA_DIR="$wdd" numEnvs=100 bash scripts/$script $GPU $seq /tmp/vizckpt/$name.pth True > /tmp/viz_$name.log 2>&1
  echo "  test reward: $(grep -oE 'reward: [-0-9.]+' /tmp/viz_$name.log | tail -1)"
  R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer /tmp/viz_marker 2>/dev/null | head -1)
  if [ -z "$R" ]; then echo "  !! no rollout (test failed?) -- $(grep -iE 'error|no such|traceback' /tmp/viz_$name.log|grep -iv nullptr|tail -1|cut -c1-80)"; return; fi
  env=$(python -c "import numpy as np;d=np.load('$R',allow_pickle=True).item();ts=sorted([k for k in d if isinstance(k,int)]);op=np.array([d[t]['object_pose'][:,:3] for t in ts]);z=op[:,:,2].max(0);print(int(np.argmax(z[1:]))+1)" 2>/dev/null)
  [ -z "$env" ] && env=1
  CUDA_VISIBLE_DEVICES=$GPU python wuji_isaacgym_playback.py --src "$R" --env $env --hand $hand --obj_code $seq --gpu 0 --out "$OUT/${name}.mp4" 2>&1 | grep -E "saved|Error|Traceback" | tail -1
}
WV1=./data/GRAB_Tracking_PK_WUJI_v1/data
run_one wuji_flute_offset_single  run_tracking_headless_grab_single_wuji_test.sh  ori_grab_s2_flute_pass_1       wuji    ./data/GRAB_Tracking_PK_WUJI_OFFSET_v1/data
run_one allegro_cubesmall_multi   run_tracking_headless_grab_multiple_test.sh     ori_grab_s2_cubesmall_inspect_1 allegro ""
run_one allegro_flute_multi       run_tracking_headless_grab_multiple_test.sh     ori_grab_s2_flute_pass_1        allegro ""
run_one allegro_combined_multi    run_tracking_headless_grab_multiple_test.sh     ori_grab_s2_cubesmall_inspect_1 allegro ""
run_one wuji_cubesmall_multi      run_tracking_headless_grab_multiple_wuji_test.sh ori_grab_s2_cubesmall_inspect_1 wuji   "$WV1"
run_one wuji_flute_multi          run_tracking_headless_grab_multiple_wuji_test.sh ori_grab_s2_flute_pass_1        wuji   "$WV1"
run_one wuji_combined_multi       run_tracking_headless_grab_multiple_wuji_test.sh ori_grab_s2_cubesmall_inspect_1 wuji   "$WV1"
echo "ALL_RENDER_DONE"
