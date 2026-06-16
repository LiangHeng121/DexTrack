#!/bin/bash
# 盯真空卡(used<8GB util<30)起 cup+apple+cubesmall 三物体 generalist 对照组.
# 配置 = beta8_noidle (和 specialist 一致), 40000 env, 纯多任务RL. 起完退出.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
R='RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_SUBDIR=contact_grab2 CONTACT_BETA=8 HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5 WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data'
STEM=grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_3obj
LIST=../assets/inst_tag_list_obj_cup_apple_cubesmall.npy
echo "[grab] $(date) 盯真空卡起 3obj generalist (beta8_noidle, 40000)"
while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits|tr -d ' '|tr ',' ' ')
  for r in "${rows[@]}"; do set -- $r
    if [ "$2" -lt 8000 ] && [ "$3" -lt 30 ]; then
      echo "[grab] $(date) GPU$1 起 3obj generalist"
      setsid bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack; cd /home/liangh/DexTrack/isaacgymenvs; env $R numEnvs=40000 minibatch_size=40000 SCRIPT_STEM=$STEM cuda_idx=$1 bash scripts/run_tracking_headless_grab_multiple_wuji.sh $1 '' $LIST > /tmp/${STEM}.log 2>&1" < /dev/null > /dev/null 2>&1 &
      echo "[grab] $(date) 已起, 退出"; exit 0
    fi
  done
  sleep 2
done
