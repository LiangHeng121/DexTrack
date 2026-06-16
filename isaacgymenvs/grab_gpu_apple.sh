#!/bin/bash
# 双抢卡起 apple specialist (两个都要, 不是二选一):
#   A = 40000 env, 需某卡 free>=72GB 且 util<30  -> SCRIPT_STEM ..._spec40k
#   B = 20000 env, 需某卡 free>=40GB 且 util<30  -> SCRIPT_STEM ..._spec20k
# 两个必须落在不同 GPU; 各起一次; 两个都起完才退出. 每2秒轮询.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
R='RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=8 CONTACT_SUBDIR=contact_grab2 HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5 IDLE_HOLD_DOF=1 IDLE_HOLD_COEF=0.25 IDLE_HOLD_BETA=3.0 WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data MAX_EPOCHS=5000'

launch(){  # $1=gpu  $2=numEnvs  $3=stem
  echo "[grab] $(date) GPU$1 起 apple numEnvs=$2 stem=$3"
  setsid bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack; cd /home/liangh/DexTrack/isaacgymenvs; env $R numEnvs=$2 minibatch_size=$2 SCRIPT_STEM=$3 cuda_idx=$1 bash scripts/run_tracking_headless_grab_multiple_wuji.sh $1 '' ../assets/inst_tag_list_obj_apple.npy > /tmp/${3}.log 2>&1" < /dev/null > /dev/null 2>&1 &
}

doneA=0; doneB=0; usedgpu=-1
echo "[grab] $(date) 双抢: A=40000(>=72GB)  B=20000(>=40GB)  不同卡"
while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  if [ $doneA -eq 0 ]; then
    for r in "${rows[@]}"; do set -- $r
      if [ "$1" != "$usedgpu" ] && [ "$2" -ge 72000 ] && [ "$3" -lt 30 ]; then
        launch $1 40000 grab_multiple_wuji_pinall3_apple_spec40k; doneA=1; usedgpu=$1; break
      fi
    done
  fi
  if [ $doneB -eq 0 ]; then
    for r in "${rows[@]}"; do set -- $r
      if [ "$1" != "$usedgpu" ] && [ "$2" -ge 40000 ] && [ "$3" -lt 30 ]; then
        launch $1 20000 grab_multiple_wuji_pinall3_apple_spec20k; doneB=1; usedgpu=$1; break
      fi
    done
  fi
  [ $doneA -eq 1 ] && [ $doneB -eq 1 ] && { echo "[grab] $(date) A+B 都已起, 退出"; exit 0; }
  sleep 2
done
