#!/bin/bash
# Quantitative comparison of the palm-back-line single policies: fair reward (@0.22) +
# sustained lift% (held at reference peak) + wrist/finger error + per-finger. GPU6, sequential.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
SEQ=ori_grab_s2_cubesmall_inspect_1; GPU=6
declare -A D=(
  [palm-lock]="grab_single_wuji_palmlock/ori_grab_s2_cubesmall_inspect_1/20260607_135358"
  [plfp]="grab_single_wuji_plfp/ori_grab_s2_cubesmall_inspect_1/20260607_161928"
  [pinall2]="grab_single_wuji_pinall2/ori_grab_s2_cubesmall_inspect_1/20260607_154422"
  [pinall3]="grab_single_wuji_pinall3/ori_grab_s2_cubesmall_inspect_1/20260607_162950"
  [pinall]="grab_single_wuji_pinall/ori_grab_s2_cubesmall_inspect_1/20260607_143758"
)
ORDER=(palm-lock plfp pinall2 pinall3 pinall)
printf "%-10s %-9s %-8s %-8s %-9s %-26s\n" exp fair@.22 lift% wristCm fingerRad per-finger | tee /tmp/cmp.txt
for k in "${ORDER[@]}"; do
  ck=$(ls -t logs/${D[$k]}/ckpt/best_ep_*.pth 2>/dev/null | head -1); cp -f "$ck" /tmp/cmp_$k.pth
  mk=/tmp/mk_cmp; touch "$mk"; sleep 1
  PALM_GRIP_THRES=0.22 numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh $GPU $SEQ /tmp/cmp_$k.pth True > /tmp/cmp_$k.log 2>&1
  rew=$(grep -oE 'reward: [-0-9.]+' /tmp/cmp_$k.log | tail -1 | grep -oE '[-0-9.]+')
  R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1); cp "$R" /tmp/Rcmp_$k.npy
  read lift wrist fe pf <<< "$(python -c "
import numpy as np
d=np.load('/tmp/Rcmp_$k.npy',allow_pickle=True).item();ts=sorted([t for t in d if isinstance(t,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,:3] for t in ts])
q=np.array([d[t]['shadow_hand_dof_pos'] for t in ts]);ref=np.array([d[t]['next_ref_np'] for t in ts])
pk=gp[:,:,2].mean(1).argmax();z=op[:,:,2]
held=int(((z[pk]-z[0])>0.10).mean()*100)
wrist=np.linalg.norm(q[...,:3]-ref[...,:3],axis=-1).mean()*100
fe=np.abs(q[...,6:26]-ref[...,6:26]).sum(-1).mean()
e=np.abs(q[...,6:26]-ref[...,6:26]);pf='/'.join(f'{e[...,i*4:(i+1)*4].sum(-1).mean():.1f}' for i in range(5))
print(f'{held} {wrist:.2f} {fe:.2f} {pf}')" 2>/dev/null)"
  printf "%-10s %-9s %-8s %-8s %-9s %-26s\n" "$k" "${rew:-?}" "${lift:-?}%" "${wrist:-?}" "${fe:-?}" "${pf:-?}" | tee -a /tmp/cmp.txt
done
echo "DONE"