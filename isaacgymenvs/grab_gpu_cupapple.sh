#!/bin/bash
# 盯"真空卡"(used<8GB 且 util<30)起 cup + apple specialist, beta8_noidle 配方, 40000 env.
# 各落不同空卡; 都起完退出. 每2秒轮询. (不抢桐轩/andrew 的半占卡)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
R='RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_SUBDIR=contact_grab2 CONTACT_BETA=8 HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5 WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data MAX_EPOCHS=5000'
launch(){ # $1=gpu $2=obj
  local stem=grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle_$2
  echo "[grab] $(date) GPU$1 起 $2 specialist (beta8_noidle, 40000)"
  setsid bash -c "source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack; cd /home/liangh/DexTrack/isaacgymenvs; env $R numEnvs=40000 minibatch_size=40000 SCRIPT_STEM=$stem cuda_idx=$1 bash scripts/run_tracking_headless_grab_multiple_wuji.sh $1 '' ../assets/inst_tag_list_obj_$2.npy > /tmp/${stem}.log 2>&1" < /dev/null > /dev/null 2>&1 &
}
done_cup=0; done_apple=0; usedgpu=-1
echo "[grab] $(date) 盯真空卡(used<8GB util<30)起 cup+apple beta8_noidle 40000"
while true; do
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits|tr -d ' '|tr ',' ' ')
  for r in "${rows[@]}"; do set -- $r
    [ "$1" = "$usedgpu" ] && continue
    if [ "$2" -lt 8000 ] && [ "$3" -lt 30 ]; then
      if [ $done_cup -eq 0 ]; then launch $1 cup; done_cup=1; usedgpu=$1; break
      elif [ $done_apple -eq 0 ]; then launch $1 apple; done_apple=1; break; fi
    fi
  done
  [ $done_cup -eq 1 ] && [ $done_apple -eq 1 ] && { echo "[grab] $(date) cup+apple 都已起, 退出"; exit 0; }
  sleep 2
done
