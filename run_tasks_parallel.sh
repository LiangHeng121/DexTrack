#!/bin/bash
# 并行: 8路重定向(跳过已存在) -> 等全部 npy -> 3路渲染。完成写 /tmp/render_tasks.log。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
ROOT=/home/liangh/DexTrack; cd $ROOT
LIST=/tmp/pick_tasks_list.npy
OUTD=isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TEST/data
NR=8   # 重定向并行路数
NV=3   # 渲染并行路数

# 1) 切 NR 份 chunk list
python3 -c "
import numpy as np
tags=sorted(np.load('$LIST',allow_pickle=True).item().keys())
NR=$NR
for i in range(NR):
    sub={t:1 for t in tags[i::NR]}
    np.save(f'/tmp/tasks_chunk_{i}.npy', sub)
print('split', len(tags), 'tags ->', NR, 'chunks')
"
# 2) 启 NR 路并行重定向
echo '[par] 启动 '$NR' 路重定向...'
pids=()
for i in $(seq 0 $((NR-1))); do
  python wuji_pipeline/batch_retarget_multitask.py --list /tmp/tasks_chunk_$i.npy --out-dir $OUTD \
     > /tmp/retarget_chunk_$i.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo '[par] 重定向全部完成'
grep -hc "OK" /tmp/retarget_chunk_*.log 2>/dev/null | paste -sd+ | bc | xargs echo "  共 OK:"
grep -h "FAIL\|skip(no" /tmp/retarget_chunk_*.log 2>/dev/null | head

# 3) NV 路并行渲染
cd $ROOT/isaacgymenvs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangh/miniconda3/envs/dextrack/lib
OUT=../render_videos/retarget_preview/tasks; mkdir -p "$OUT"
TAGS=$(python3 -c "import numpy as np; print(' '.join(sorted(np.load('$LIST',allow_pickle=True).item().keys())))")
render_worker(){
  for tag in $@; do
    ref="$OUTD/wuji_passive_active_info_${tag}.npy"; ref="$ROOT/$ref"
    objcode="${tag%_nf_300}"
    name=$(echo "$objcode" | sed -E 's#ori_grab_s[0-9]+_##; s#_[0-9]+(_Retake)?$##; s#_Retake$##')
    [ -f "$ref" ] || { echo "[render] 缺 $tag"; continue; }
    python wuji_isaacgym_playback.py --src "$ref" --ref --hand wuji --obj_code "$objcode" --gpu 0 \
      --cam_scale 0.5 --cam_follow hand --cam_smooth 6 --out "$OUT/${name}.mp4" >/dev/null 2>&1 \
      && echo "[render] $name ok" || echo "[render] $name ERR"
  done
}
echo '[par] 启动 '$NV' 路渲染...'
arr=($TAGS); rpids=()
for v in $(seq 0 $((NV-1))); do
  sub=(); j=$v
  while [ $j -lt ${#arr[@]} ]; do sub+=("${arr[$j]}"); j=$((j+NV)); done
  render_worker "${sub[@]}" &
  rpids+=($!)
done
wait "${rpids[@]}"
echo "tasks 全部完成 -> $OUT" | tee -a /tmp/render_tasks.log
ls "$OUT"/*.mp4 | wc -l
