#!/bin/bash
# 用 multiobj generalist best ckpt, 对 9 物体各跑一条代表测试(3 GPU 并行), 测每物体跟踪.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
HE="HAND_EMA_COEF=0.4 WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data CONTACT_SUBDIR=contact_grab2 numEnvs=100"
declare -A REP=( [apple]=ori_grab_s1_apple_lift [duck]=ori_grab_s1_duck_inspect_1 [cup]=ori_grab_s10_cup_lift \
  [alarmclock]=ori_grab_s10_alarmclock_lift_Retake [mouse]=ori_grab_s10_mouse_lift [phone]=ori_grab_s10_phone_lift \
  [train]=ori_grab_s10_train_lift [elephant]=ori_grab_s10_elephant_inspect_1 [cubesmall]=ori_grab_s10_cubesmall_inspect_1 )
GPUS=(0 6 7); i=0
run_one(){
  local obj=$1 seq=$2 gpu=$3
  env $HE SCRIPT_STEM=moeval_$obj bash scripts/run_tracking_headless_grab_multiple_wuji_test.sh $gpu $seq /tmp/ck_multiobj.pth True > /tmp/moeval_$obj.log 2>&1
}
for obj in apple duck cup alarmclock mouse phone train elephant cubesmall; do
  gpu=${GPUS[$((i%3))]}; i=$((i+1))
  run_one $obj ${REP[$obj]} $gpu &
  # 每3个一批等一下(每GPU最多3个排队)
  [ $((i%3)) -eq 0 ] && wait
done
wait
echo "[moeval] 全部测试完成" | tee -a /tmp/moeval_done.log
