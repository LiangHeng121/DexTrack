#!/bin/bash
# Render several envs of the #2+#3 SINGLE policy (same seq, diff inits) + 1 plain baseline.
# Tests run SEQUENTIALLY and each rollout is copied out immediately (the two share the same
# logs_test/<seq> dir, so concurrent runs would clobber each other's rollout npy).
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
OUT=/home/liangh/DexTrack/render_videos/reward_exps; mkdir -p "$OUT"
SEQ=ori_grab_s2_cubesmall_inspect_1
B23=logs/grab_single_wuji_relax23/ori_grab_s2_cubesmall_inspect_1/20260606_095846/ckpt/best_ep_980_rew_48.12.pth
BBASE=ckpts/wuji_cubesmall_inspect_best.pth

test_capture(){ # name gpu ckpt dest_npy
  name=$1; gpu=$2; ckpt=$3; dest=$4; mk=/tmp/mk_$name; touch "$mk"; sleep 1
  numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh $gpu $SEQ "$ckpt" True > /tmp/test_$name.log 2>&1
  echo "  $name test reward: $(grep -oE 'reward: [-0-9.]+' /tmp/test_$name.log | tail -1)"
  R=$(find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer "$mk" 2>/dev/null | xargs -r ls -t | head -1)
  if [ -z "$R" ]; then echo "  !! $name no rollout"; return 1; fi
  cp "$R" "$dest"; echo "  $name rollout -> $dest"
}

echo "=== sequential tests ==="
test_capture m23s 2 "$B23"   /tmp/R_m23s.npy
test_capture base 2 "$BBASE" /tmp/R_base.npy

PICKS=$(python -c "
import numpy as np
d=np.load('/tmp/R_m23s.npy',allow_pickle=True).item()
ts=sorted([k for k in d if isinstance(k,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts])
z=op[:,:,2].max(0); o=np.argsort(z); N=len(z)
print(' '.join(str(int(o[i])) for i in [0, N//3, 2*N//3, N-1]))
" 2>/dev/null)
echo "PICKS (#2+#3 worst..best lift env)=$PICKS"
Bz=$(python -c "
import numpy as np
d=np.load('/tmp/R_base.npy',allow_pickle=True).item()
ts=sorted([k for k in d if isinstance(k,int)])
op=np.array([d[t]['object_pose'][:,:3] for t in ts]); z=op[:,:,2].max(0)
print(int(np.argmax(z[1:]))+1)" 2>/dev/null); [ -z "$Bz" ] && Bz=1

rm -f "$OUT"/m23s_env*.mp4 "$OUT"/base_single_*.mp4   # clear the bad ones
i=0; gpus=(2 3 6)
for e in $PICKS; do
  g=${gpus[$((i%3))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g python wuji_isaacgym_playback.py --src /tmp/R_m23s.npy --env $e --hand wuji --obj_code $SEQ --gpu 0 --out "$OUT/m23s_env${e}.mp4" 2>&1 | grep -E "saved|Error" | tail -1 &
done
CUDA_VISIBLE_DEVICES=6 python wuji_isaacgym_playback.py --src /tmp/R_base.npy --env $Bz --hand wuji --obj_code $SEQ --gpu 0 --out "$OUT/base_single_env${Bz}.mp4" 2>&1 | grep -E "saved|Error" | tail -1 &
wait
echo "ALL_23SINGLE_RENDERED"
# lift stats for context
python -c "
import numpy as np
for nm,f in [('#2+#3','/tmp/R_m23s.npy'),('baseline','/tmp/R_base.npy')]:
    d=np.load(f,allow_pickle=True).item(); ts=sorted([k for k in d if isinstance(k,int)])
    op=np.array([d[t]['object_pose'][:,:3] for t in ts]); z=op[:,:,2]
    lift=z.max(0)-z[0]; print(f'{nm}: lift(max-init) mean={lift.mean()*100:.1f}cm  >5cm envs={int((lift>0.05).sum())}/100  finalZ mean={z[-1].mean()*100:.1f}cm')
"
ls -la "$OUT"/m23s_*.mp4 "$OUT"/base_single_*.mp4 2>/dev/null