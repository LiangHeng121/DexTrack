#!/bin/bash
# Every 5 min: test palm-lock single's CURRENT best ckpt with NO switches (-> ORIGINAL
# reward) + lift%, append to /tmp/palmlock_orig.log. Copies the ckpt first (trainer
# deletes the old best). Uses GPU6. Stops when single hits ep1000.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
SEQ=ori_grab_s2_cubesmall_inspect_1
RUN=logs/grab_single_wuji_pinall/ori_grab_s2_cubesmall_inspect_1/20260607_143758
OUT=/tmp/pinall_orig.log; GPU=6; seen=""
for i in $(seq 1 120); do
  ck=$(ls -t "$RUN"/ckpt/best_ep_*.pth 2>/dev/null | head -1)
  epr=$(basename "$ck" 2>/dev/null | grep -oE "ep_[0-9]+_rew_[-0-9.]+")
  if [ -n "$ck" ] && [ "$epr" != "$seen" ] && cp -f "$ck" /tmp/plo_ckpt.pth 2>/dev/null; then
    seen="$epr"; mk=/tmp/mk_plo; touch "$mk"; sleep 1
    PALM_GRIP_THRES=0.22 numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh $GPU $SEQ /tmp/plo_ckpt.pth True > /tmp/test_plo.log 2>&1
    orig=$(grep -oE 'reward: [-0-9.]+' /tmp/test_plo.log | tail -1 | grep -oE '[-0-9.]+')
    R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
    lift=$(python -c "import numpy as np;d=np.load('$R',allow_pickle=True).item();ts=sorted([k for k in d if isinstance(k,int)]);op=np.array([d[t]['object_pose'][:,:3] for t in ts]);z=op[:,:,2];L=z.max(0)-z[0];print(f'{int((L>0.05).mean()*100)}%% mean={L.mean()*100:.1f}cm')" 2>/dev/null)
    echo "[$(date +%H:%M)] $epr | FAIR_reward(flag@0.22)=${orig:-FAIL} | lift=${lift:-?}" >> "$OUT"
  fi
  cur=$(grep -oE "epoch: [0-9]+/1000" "$RUN/screen.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  [ -n "$cur" ] && [ "$cur" -ge 1000 ] && { echo "[done] ep1000" >> "$OUT"; break; }
  sleep 300
done
