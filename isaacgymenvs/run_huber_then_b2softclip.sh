#!/bin/bash
# 等 huber single(GPU0)跑完 -> 接续训 cgsmooth_B2_softclip single(无huber, ep3000, GPU0)。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs

echo "[seq] $(date) 等 huber single 跑完..."
while pgrep -f "log_path=./logs/grab_single_wuji_pinall3_cgsmooth_B2_softclip_huber" >/dev/null 2>&1; do sleep 60; done
echo "[seq] $(date) huber 结束, GPU0 释放, 启动 cgsmooth_B2_softclip single ep3000"

env RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 \
CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=30 CONTACT_SUBDIR=contact_grab2 \
HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 \
SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5 \
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data MAX_EPOCHS=3000 \
SCRIPT_STEM=grab_single_wuji_pinall3_cgsmooth_B2_softclip cuda_idx=0 \
bash scripts/run_tracking_headless_grab_single_wuji.sh 0 ori_grab_s2_cubesmall_inspect_1 \
> /tmp/single_b2softclip.log 2>&1
echo "[seq] $(date) cgsmooth_B2_softclip 训练结束"

# wandb 重命名
python3 - << 'PYEOF'
import wandb
api=wandb.Api()
for r in list(api.runs("liangheng-peking-university/dextrack", order="-created_at"))[:40]:
    if r.group=="repro_grab_single_wuji_pinall3_cgsmooth_B2_softclip" and r.name!="wuji_pinall3_cgsmooth_B2_softclip_single":
        r.name="wuji_pinall3_cgsmooth_B2_softclip_single"; r.update(); print("[seq] wandb ->", r.name); break
PYEOF
echo "[seq] $(date) 全部完成"
