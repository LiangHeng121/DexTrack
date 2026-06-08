#!/bin/bash
# Full-task eval of the 3 cubesmall-multi generalists across ALL 26 training trajectories.
# Per policy: mean fair@0.22 + mean lift% (held at ref-peak) + #trajs lifted(>50%). Parallel on GPU 2/3/6.
source /home/liangh/miniconda3/etc/profile.d/conda.sh; conda activate dextrack 2>/dev/null
cd /home/liangh/DexTrack/isaacgymenvs
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
GPUS=(2 3 6)
declare -A CFG=(
 [plfp]="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3"
 [pinall3]="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0"
 [pinall]="RELAX_PALM=1 FIX_FINGER5=1 BOOST_FINGERPOSE=1 FINGERPOSE_COEF=0.3 FINGER_POS_REW=1 FINGER_POS_COEF=1.5 PALM_POS_REW=1 PALM_POS_COEF=2.0 GLB_TRANS_COEF=1.0 GLB_ROT_COEF=0.2" )
declare -A DIR=(
 [plfp]=grab_multiple_wuji_plfp/wuji_cubesmall_plfp
 [pinall3]=grab_multiple_wuji_pinall3/wuji_cubesmall_pinall3
 [pinall]=grab_multiple_wuji_pinall/wuji_cubesmall_pinall )
TRAJS=$(python -c "import numpy as np;d=np.load('../assets/inst_tag_list_obj_cubesmall_pinall.npy',allow_pickle=True).item();print(' '.join(k.replace('_nf_300','') for k in d))")
rm -f /tmp/ftres_*.res

test_one(){ local pol=$1 traj=$2 gpu=$3 stem=ftest_$1
  env ${CFG[$pol]} PALM_GRIP_THRES=0.22 WUJI_DATA_DIR=$FPOS numEnvs=100 SCRIPT_STEM=$stem \
    bash scripts/run_tracking_headless_grab_multiple_wuji_test.sh $gpu $traj /tmp/ft_$pol.pth True > /tmp/ftlog_${pol}_${traj}.log 2>&1
  local rew=$(grep -oE 'reward: [-0-9.]+' /tmp/ftlog_${pol}_${traj}.log | tail -1 | grep -oE '[-0-9.]+')
  local R=$(ls -t logs_test/$stem/$traj/*/*/ts_to_hand_obj_obs_reset_1.npy 2>/dev/null | head -1)
  local lift=$(python -c "
import numpy as np
d=np.load('$R',allow_pickle=True).item();ts=sorted([t for t in d if isinstance(t,int)])
op=np.array([d[t]['object_pose'][:,2] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,2] for t in ts])
z=op-op[0];gz=gp-gp[0];pk=gz.mean(1).argmax();print(int((z[pk]>0.10).mean()*100))" 2>/dev/null)
  echo "$traj ${rew:-NA} ${lift:-NA}" > /tmp/ftres_${pol}_${traj}.res
}

for pol in plfp pinall3 pinall; do
  ck=$(ls -t logs/${DIR[$pol]}/*/ckpt/best_ep_*.pth 2>/dev/null | head -1); cp -f "$ck" /tmp/ft_$pol.pth
  echo "### $pol  ckpt=$(basename $ck)"
  i=0
  for traj in $TRAJS; do
    test_one "$pol" "$traj" "${GPUS[$((i%3))]}" &
    i=$((i+1)); (( i % 3 == 0 )) && wait
  done
  wait
done

echo "============ FULL-TASK SUMMARY (26 trajs) ============"
printf "%-9s %-10s %-10s %-12s\n" policy meanFair meanLift% lifted(>50%)
for pol in plfp pinall3 pinall; do
  python -c "
import glob
rs=[open(f).read().split() for f in glob.glob('/tmp/ftres_${pol}_*.res')]
fa=[float(r[1]) for r in rs if r[1]!='NA']; li=[float(r[2]) for r in rs if r[2]!='NA']
n=len(rs); mf=sum(fa)/len(fa) if fa else 0; ml=sum(li)/len(li) if li else 0; lifted=sum(1 for x in li if x>50)
print(f'{\"$pol\":<9} {mf:<10.2f} {ml:<10.1f} {lifted}/{n}')"
done
echo "--- per-traj lift% (plfp / pinall3 / pinall) ---"
python -c "
import glob,os
trajs=sorted(set(os.path.basename(f).split('_',2)[2][:-4] for f in glob.glob('/tmp/ftres_*.res')))
def g(p,t):
  try: return open(f'/tmp/ftres_{p}_{t}.res').read().split()[2]
  except: return '-'
for t in trajs: print(f'{t.replace(\"ori_grab_\",\"\"):28s} {g(\"plfp\",t):>4} {g(\"pinall3\",t):>4} {g(\"pinall\",t):>4}')"
echo DONE
