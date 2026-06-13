#!/bin/bash
# round2: 等 pick10b 重定向完成 -> 渲染 10 个物体的 wuji 参考预览(统一相机)。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangh/miniconda3/envs/dextrack/lib
DATA=data/GRAB_Tracking_PK_WUJI_TEST/data
OUT=../render_videos/retarget_preview; mkdir -p "$OUT"

echo "[render] 等 round2 重定向完成..."
while ! grep -q "batch done" /tmp/retarget_pick10b.log 2>/dev/null; do sleep 20; done
echo "[render] 重定向完成, 开始渲染"

TAGS="ori_grab_s10_mouse_lift_nf_300 ori_grab_s1_camera_lift_nf_300 \
ori_grab_s10_alarmclock_lift_Retake_nf_300 ori_grab_s10_binoculars_lift_nf_300 \
ori_grab_s10_mug_lift_nf_300 ori_grab_s10_cup_lift_nf_300 ori_grab_s10_bowl_lift_nf_300 \
ori_grab_s10_train_lift_nf_300 ori_grab_s10_phone_lift_nf_300 ori_grab_s10_stamp_lift_nf_300"

for tag in $TAGS; do
  ref="$DATA/wuji_passive_active_info_${tag}.npy"
  objcode="${tag%_nf_300}"
  obj=$(echo "$objcode" | sed -E 's#ori_grab_s[0-9]+_##; s#_lift.*##')
  if [ ! -f "$ref" ]; then echo "[render] 缺 $tag, 跳过"; continue; fi
  echo "[render] $obj ..."
  python wuji_isaacgym_playback.py --src "$ref" --ref --hand wuji --obj_code "$objcode" --gpu 0 \
    --cam_scale 0.5 --cam_follow hand --cam_smooth 6 \
    --out "$OUT/${obj}_lift.mp4" 2>&1 | grep -E "saved|Error" | tail -1
done
echo "[render] round2 全部完成 -> $OUT"
