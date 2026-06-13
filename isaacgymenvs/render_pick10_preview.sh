#!/bin/bash
# 等 pick10 重定向完成 -> 渲染 10 个物体的 wuji 参考预览(统一相机)。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangh/miniconda3/envs/dextrack/lib
DATA=data/GRAB_Tracking_PK_WUJI_TEST/data
OUT=../render_videos/retarget_preview; mkdir -p "$OUT"

echo "[render] 等重定向完成..."
while ! grep -q "batch done" /tmp/retarget_pick10.log 2>/dev/null; do sleep 20; done
echo "[render] 重定向完成, 开始渲染"

# tag 列表 (apple 用已生成的 s2)
TAGS="ori_grab_s2_apple_lift_nf_300 \
ori_grab_s10_cubemedium_lift_nf_300 ori_grab_s10_cubelarge_lift_nf_300 \
ori_grab_s10_spherelarge_lift_nf_300 ori_grab_s1_duck_lift_nf_300 \
ori_grab_s1_piggybank_lift_nf_300 ori_grab_s10_stanfordbunny_lift_nf_300 \
ori_grab_s1_elephant_lift_nf_300 ori_grab_s1_pyramidlarge_lift_nf_300 \
ori_grab_s10_gamecontroller_lift_nf_300"

for tag in $TAGS; do
  ref="$DATA/wuji_passive_active_info_${tag}.npy"
  objcode="${tag%_nf_300}"
  obj=$(echo "$objcode" | sed -E 's#ori_grab_s[0-9]+_##; s#_lift##')
  if [ ! -f "$ref" ]; then echo "[render] 缺 $tag, 跳过"; continue; fi
  echo "[render] $obj ..."
  python wuji_isaacgym_playback.py --src "$ref" --ref --hand wuji --obj_code "$objcode" --gpu 0 \
    --cam_scale 0.5 --cam_follow hand --cam_smooth 6 \
    --out "$OUT/${obj}_lift.mp4" 2>&1 | grep -E "saved|Error" | tail -1
done
echo "[render] 全部完成 -> $OUT"
ls -la "$OUT"/*.mp4
