#!/bin/bash
# Test the 3 done smoothing singles (ep1000) @ fair0.22: reward + held-lift% (at ref peak) +
# wrist/finger err + per-finger. Sequential on GPU6 (avoid rollout collision). FPOS data.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data; GPU=6
declare -A SEQ=(
  [pinall_sm]=ori_grab_s2_cubesmall_inspect_1
  [pinallsm_flute]=ori_grab_s2_flute_pass_1
  [plfpsm_flute]=ori_grab_s2_flute_pass_1 )
declare -A STEM=(
  [pinall_sm]=grab_single_wuji_pinall_sm
  [pinallsm_flute]=grab_single_wuji_pinallsm_flute
  [plfpsm_flute]=grab_single_wuji_plfpsm_flute )
ORDER=(pinall_sm pinallsm_flute plfpsm_flute)
printf "%-16s %-9s %-7s %-8s %-9s %-26s\n" exp fair@.22 lift% wristCm fingerRad per-finger | tee /tmp/sm.txt
for k in "${ORDER[@]}"; do
  ck=$(ls -t logs/${STEM[$k]}/*/*/ckpt/best_ep_*.pth 2>/dev/null | grep 021715 | head -1)
  [ -z "$ck" ] && ck=$(ls -t logs/${STEM[$k]}/*/*/ckpt/best_ep_*.pth 2>/dev/null | head -1)
  cp -f "$ck" /tmp/sm_$k.pth
  mk=/tmp/mk_sm; touch "$mk"; sleep 1
  PALM_GRIP_THRES=0.22 WUJI_DATA_DIR=$FPOS numEnvs=100 \
    bash scripts/run_tracking_headless_grab_single_wuji_test.sh $GPU ${SEQ[$k]} /tmp/sm_$k.pth True > /tmp/sm_$k.log 2>&1
  rew=$(grep -oE 'reward: [-0-9.]+' /tmp/sm_$k.log | tail -1 | grep -oE '[-0-9.]+')
  R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
  cp -f "$R" /tmp/Rsm_$k.npy 2>/dev/null
  read lift wrist fe pf <<< "$(python -c "
import numpy as np
d=np.load('/tmp/Rsm_$k.npy',allow_pickle=True).item();ts=sorted([t for t in d if isinstance(t,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,:3] for t in ts])
q=np.array([d[t]['shadow_hand_dof_pos'] for t in ts]);ref=np.array([d[t]['next_ref_np'] for t in ts])
pk=gp[:,:,2].mean(1).argmax();z=op[:,:,2]
held=int(((z[pk]-z[0])>0.10).mean()*100)
wrist=np.linalg.norm(q[...,:3]-ref[...,:3],axis=-1).mean()*100
fe=np.abs(q[...,6:26]-ref[...,6:26]).sum(-1).mean()
e=np.abs(q[...,6:26]-ref[...,6:26]);pf='/'.join(f'{e[...,i*4:(i+1)*4].sum(-1).mean():.1f}' for i in range(5))
print(f'{held} {wrist:.2f} {fe:.2f} {pf}')" 2>/dev/null)"
  printf "%-16s %-9s %-7s %-8s %-9s %-26s\n" "$k" "${rew:-?}" "${lift:-?}%" "${wrist:-?}" "${fe:-?}" "${pf:-?}" | tee -a /tmp/sm.txt
done
echo "DONE"
