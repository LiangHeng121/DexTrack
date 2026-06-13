#!/bin/bash
# Full-trajectory sweep of pinall3_multi (all 26 cubesmall trajs): fair@0.22 + held@peak lift%.
# Parallel across GPU 1 & 2 (which have ~40GB headroom under the single training runs).
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
P3="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0"
ck=$(ls -t logs/grab_multiple_wuji_pinall3/wuji_cubesmall_pinall3/*/ckpt/best_ep_*.pth | head -1); cp -f "$ck" /tmp/p3all.pth
echo "ckpt: $(basename $ck)"
TRAJS=$(python3 -c "import numpy as np;d=np.load('../assets/inst_tag_list_obj_cubesmall_pinall3.npy',allow_pickle=True).item();print(' '.join(k.replace('_nf_300','') for k in d))")
rm -f /tmp/p3a_*.res

test_one(){ local traj=$1 gpu=$2 stem=p3a_$1
  local mk=/tmp/mk_$stem; touch "$mk"; sleep 1
  env $P3 PALM_GRIP_THRES=0.22 WUJI_DATA_DIR=$FPOS numEnvs=100 SCRIPT_STEM=$stem \
    bash scripts/run_tracking_headless_grab_multiple_wuji_test.sh $gpu $traj /tmp/p3all.pth True > /tmp/p3alog_$traj.log 2>&1
  local rew=$(grep -oE 'reward: [-0-9.]+' /tmp/p3alog_$traj.log | tail -1 | grep -oE '[-0-9.]+')
  local R=$(ls -t logs_test/$stem/$traj/*/ts_to_hand_obj_obs_reset_1.npy 2>/dev/null | head -1)
  local lift=$(python3 -c "
import numpy as np
d=np.load('$R',allow_pickle=True).item();ts=sorted([t for t in d if isinstance(t,int)])
op=np.array([d[t]['object_pose'][:,2] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,2] for t in ts])
z=op-op[0];gz=gp-gp[0];pk=gz.mean(1).argmax();print(int((z[pk]>0.10).mean()*100))" 2>/dev/null)
  echo "$traj ${rew:-NA} ${lift:-NA}" > /tmp/p3a_$traj.res
}

i=0
for traj in $TRAJS; do
  gpu=$([ $((i%2)) -eq 0 ] && echo 1 || echo 2)
  test_one "$traj" "$gpu" &
  i=$((i+1)); (( i % 2 == 0 )) && wait
done
wait

echo "============ pinall3_multi 全 26 轨迹 ============"
python3 -c "
import glob
rs=[open(f).read().split() for f in glob.glob('/tmp/p3a_*.res')]
fa=[float(r[1]) for r in rs if r[1]!='NA']; li=[float(r[2]) for r in rs if r[2]!='NA']
print(f'均fair={sum(fa)/len(fa):.1f}  均举升%={sum(li)/len(li):.1f}  举起(>50%)={sum(1 for x in li if x>50)}/{len(li)}')
print('--- 逐轨迹 (fair / lift%) ---')
for r in sorted(rs, key=lambda r: -float(r[2]) if r[2]!='NA' else 0):
    print(f'{r[0].replace(\"ori_grab_\",\"\"):28s} {r[1]:>8} {r[2]:>4}%')"
echo DONE
