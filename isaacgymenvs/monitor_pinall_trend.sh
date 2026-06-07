#!/bin/bash
# High-freq trend monitor for pinall single. Poll every 60s; on each NEW best ckpt,
# test (PALM_GRIP_THRES=0.22 fair flag) -> fair reward, and from the rollout compute
# lift% / finger-err / wrist-err. Append a compact trend line. GPU6.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
SEQ=ori_grab_s2_cubesmall_inspect_1
RUN=logs/grab_single_wuji_pinall2/ori_grab_s2_cubesmall_inspect_1/20260607_154422
OUT=/tmp/pinall2_trend.log; GPU=6; seen=""
printf "%-7s %-6s %-9s %-7s %-9s %-8s\n" time ep fairRew lift% fingerErr wristErr >> "$OUT"
for i in $(seq 1 400); do
  ck=$(ls -t "$RUN"/ckpt/best_ep_*.pth 2>/dev/null | head -1)
  epr=$(basename "$ck" 2>/dev/null | grep -oE "ep_[0-9]+" | grep -oE "[0-9]+")
  if [ -n "$ck" ] && [ "$epr" != "$seen" ] && cp -f "$ck" /tmp/pintrend_ck.pth 2>/dev/null; then
    seen="$epr"; mk=/tmp/mk_pt; touch "$mk"; sleep 1
    PALM_GRIP_THRES=0.22 numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh $GPU $SEQ /tmp/pintrend_ck.pth True > /tmp/test_pt.log 2>&1
    rew=$(grep -oE 'reward: [-0-9.]+' /tmp/test_pt.log | tail -1 | grep -oE '[-0-9.]+')
    R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
    read lift fe gt <<< "$(python -c "
import numpy as np
d=np.load('$R',allow_pickle=True).item();ts=sorted([k for k in d if isinstance(k,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts]);q=np.array([d[t]['shadow_hand_dof_pos'] for t in ts]);ref=np.array([d[t]['next_ref_np'] for t in ts])
z=op[:,:,2];L=z.max(0)-z[0]
print(f'{int((L>0.05).mean()*100)} {np.abs(q[...,6:26]-ref[...,6:26]).sum(-1).mean():.2f} {np.linalg.norm(q[...,:3]-ref[...,:3],axis=-1).mean()*100:.1f}')" 2>/dev/null)"
    printf "%-7s %-6s %-9s %-7s %-9s %-8s\n" "$(date +%H:%M)" "$epr" "${rew:-?}" "${lift:-?}%" "${fe:-?}rad" "${gt:-?}cm" >> "$OUT"
  fi
  cur=$(grep -oE "epoch: [0-9]+/1000" "$RUN/screen.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  [ -n "$cur" ] && [ "$cur" -ge 1000 ] && { echo "[done] ep1000" >> "$OUT"; break; }
  sleep 60
done
