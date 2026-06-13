#!/bin/bash
# smooth 拆分消融: 接续跑 EMA-only(等 ACTRATE-only 跑完释放 GPU0 后再起,避免同卡 22000-env OOM)。
# ACTRATE-only 已在 GPU0 运行(SCRIPT_STEM=grab_single_wuji_pinall3_cg_actrateonly)。
# 用法: setsid bash run_smooth_ablation_seq.sh < /dev/null > /tmp/ablation_seq.log 2>&1 &
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs

COMMON="RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 \
CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=30 \
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data MAX_EPOCHS=1000"

echo "[seq] $(date) 等 ACTRATE-only 跑完..."
while pgrep -f "log_path=./logs/grab_single_wuji_pinall3_cg_actrateonly" >/dev/null 2>&1; do
  sleep 60
done
echo "[seq] $(date) ACTRATE-only 已结束, GPU0 释放, 启动 EMA-only"

env $COMMON HAND_EMA_COEF=0.4 \
  SCRIPT_STEM=grab_single_wuji_pinall3_cg_emaonly cuda_idx=0 \
  bash scripts/run_tracking_headless_grab_single_wuji.sh 0 ori_grab_s2_cubesmall_inspect_1 \
  > /tmp/run_cg_emaonly.log 2>&1
echo "[seq] $(date) EMA-only 结束"

# wandb 重命名 (默认名 -> 规范名)
python3 - << 'PYEOF'
import wandb
api=wandb.Api()
for r in list(api.runs("liangheng-peking-university/dextrack", order="-created_at"))[:40]:
    if r.group=="repro_grab_single_wuji_pinall3_cg_emaonly" and r.name!="wuji_pinall3_cg_emaonly":
        r.name="wuji_pinall3_cg_emaonly"; r.update(); print("[seq] wandb renamed ->", r.name); break
PYEOF
echo "[seq] $(date) 全部完成"
