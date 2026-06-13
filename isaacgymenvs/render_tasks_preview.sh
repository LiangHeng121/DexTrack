#!/bin/bash
# 等 tasks 重定向完成 -> 渲染每个 (物体×任务) 代表轨迹, 按 <物体>_<任务> 命名。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangh/miniconda3/envs/dextrack/lib
DATA=data/GRAB_Tracking_PK_WUJI_TEST/data
OUT=../render_videos/retarget_preview/tasks; mkdir -p "$OUT"

echo "[render] 等 tasks 重定向完成..."
while ! grep -q "batch done" /tmp/retarget_tasks.log 2>/dev/null; do sleep 20; done
echo "[render] 重定向完成, 开始渲染"

# 从清单读 tag
TAGS=$(python3 -c "import numpy as np; print(' '.join(np.load('/tmp/pick_tasks_list.npy',allow_pickle=True).item().keys()))")
for tag in $TAGS; do
  ref="$DATA/wuji_passive_active_info_${tag}.npy"
  objcode="${tag%_nf_300}"
  # 文件名 = 物体_任务 (去 subject + 末尾 index/Retake)
  name=$(echo "$objcode" | sed -E 's#ori_grab_s[0-9]+_##; s#_[0-9]+(_Retake)?$##; s#_Retake$##')
  if [ ! -f "$ref" ]; then echo "[render] 缺 $tag, 跳过"; continue; fi
  echo "[render] $name ..."
  python wuji_isaacgym_playback.py --src "$ref" --ref --hand wuji --obj_code "$objcode" --gpu 0 \
    --cam_scale 0.5 --cam_follow hand --cam_smooth 6 \
    --out "$OUT/${name}.mp4" 2>&1 | grep -E "saved|Error" | tail -1
done
echo "[render] tasks 全部完成 -> $OUT"
ls "$OUT"/*.mp4 | wc -l
