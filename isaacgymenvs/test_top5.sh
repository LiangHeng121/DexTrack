#!/bin/bash
# Test + render the top-5 runs from the progress table. Each tested WITH its training-time
# dynamics switches (CLIP_DOF / HAND_EMA) so the policy's compensation matches; fair@0.22 + lift%.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data; GPU=1
PINALL3="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0"
PLFP="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3"
printf "%-22s %-9s %-7s %-8s %-9s\n" exp fair@.22 lift% wristCm fingerRad | tee /tmp/top5.txt

# name | type(single/multi) | seq | logdir | dyn-switches
run_one(){
  local name=$1 type=$2 seq=$3 logdir=$4 dyn=$5
  local ck=$(ls -t logs/$logdir/*/ckpt/best_ep_*.pth 2>/dev/null | head -1)
  [ -z "$ck" ] && { printf "%-22s %s\n" "$name" "NO CKPT"; return; }
  cp -f "$ck" /tmp/t5_$name.pth
  local script=run_tracking_headless_grab_single_wuji_test.sh
  [ "$type" = multi ] && script=run_tracking_headless_grab_multiple_wuji_test.sh
  local mk=/tmp/mk_t5_$name; touch "$mk"; sleep 1
  env $dyn PALM_GRIP_THRES=0.22 WUJI_DATA_DIR=$FPOS numEnvs=100 SCRIPT_STEM=t5_$name \
    bash scripts/$script $GPU $seq /tmp/t5_$name.pth True > /tmp/t5_$name.log 2>&1
  local rew=$(grep -oE 'reward: [-0-9.]+' /tmp/t5_$name.log | tail -1 | grep -oE '[-0-9.]+')
  local R=$(find logs_test/t5_$name -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
  [ -z "$R" ] && R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
  cp -f "$R" /tmp/R5_$name.npy 2>/dev/null
  read lift wrist fe pf env <<< "$(python3 -c "
import numpy as np
d=np.load('/tmp/R5_$name.npy',allow_pickle=True).item();ts=sorted([t for t in d if isinstance(t,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,:3] for t in ts])
q=np.array([d[t]['shadow_hand_dof_pos'] for t in ts]);ref=np.array([d[t]['next_ref_np'] for t in ts])
z=op[:,:,2]-op[0,:,2];gz=gp[:,:,2]-gp[0,:,2];pk=gz.mean(1).argmax()
wrist=np.linalg.norm(q[...,:3]-ref[...,:3],axis=-1).mean()*100;fe=np.abs(q[...,6:26]-ref[...,6:26]).sum(-1).mean()
e=np.abs(q[...,6:26]-ref[...,6:26]);pf='/'.join(f'{e[...,i*4:(i+1)*4].sum(-1).mean():.1f}' for i in range(5))
held=[n for n in range(z.shape[1]) if z[pk,n]>0.10]
print(f'{(z[pk]>0.10).mean()*100:.0f} {wrist:.2f} {fe:.2f} {pf} {held[len(held)//2] if held else 1}')" 2>/dev/null)"
  printf "%-22s %-9s %-7s %-8s %-9s %s\n" "$name" "${rew:-?}" "${lift:-?}%" "${wrist:-?}" "${fe:-?}" "${pf:-?}" | tee -a /tmp/top5.txt
  CUDA_VISIBLE_DEVICES=$GPU python wuji_isaacgym_playback.py --src /tmp/R5_$name.npy --env ${env:-1} --hand wuji \
    --obj_code $seq --gpu 0 --out render_videos/smooth_test/top5_${name}_env${env}.mp4 2>&1 | grep -oE "saved.*mp4" | tail -1
}

run_one cgclip_flute   single ori_grab_s2_flute_pass_1        grab_single_wuji_pinall3_cgclip_flute/ori_grab_s2_flute_pass_1        "$PINALL3 CLIP_DOF=1"
run_one pinall3_full   single ori_grab_s2_cubesmall_inspect_1 grab_single_wuji_pinall3_full/ori_grab_s2_cubesmall_inspect_1         "$PINALL3 CLIP_DOF=1 HAND_EMA_COEF=0.6"
run_one cgclip_multi   multi  ori_grab_s2_cubesmall_inspect_1 grab_multiple_wuji_pinall3_cgclip/wuji_cubesmall_pinall3              "$PINALL3 CLIP_DOF=1"
run_one plfp_multi     multi  ori_grab_s2_cubesmall_inspect_1 grab_multiple_wuji_plfp/wuji_cubesmall_plfp                           "$PLFP"
run_one pinall3_multi  multi  ori_grab_s2_cubesmall_inspect_1 grab_multiple_wuji_pinall3/wuji_cubesmall_pinall3                     "$PINALL3"
echo DONE
