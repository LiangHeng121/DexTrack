#!/bin/bash
# 并行重定向: run_retarget_parallel.sh <list.npy> <out-dir> <N路>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
# 限每进程单线程, 避免 16 路 × 满核 BLAS/nlopt 把 CPU 抢崩(load 956)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
ROOT=/home/liangh/DexTrack; cd $ROOT
LIST=$1; OUTD=$2; NR=${3:-16}
python3 -c "
import numpy as np
tags=sorted(np.load('$LIST',allow_pickle=True).item().keys()); NR=$NR
for i in range(NR): np.save(f'/tmp/rt_chunk_{i}.npy', {t:1 for t in tags[i::NR]})
print('split',len(tags),'->',NR,'chunks')
"
echo "[par] 启动 $NR 路重定向 -> $OUTD"
pids=()
for i in $(seq 0 $((NR-1))); do
  python wuji_pipeline/batch_retarget_multitask.py --list /tmp/rt_chunk_$i.npy --out-dir $OUTD \
     > /tmp/rt_chunk_$i.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
ok=$(grep -h "OK (crop" /tmp/rt_chunk_*.log 2>/dev/null | wc -l)
fail=$(grep -h "FAIL" /tmp/rt_chunk_*.log 2>/dev/null | wc -l)
skipn=$(grep -h "skip(no" /tmp/rt_chunk_*.log 2>/dev/null | wc -l)
echo "[par] 重定向完成: OK=$ok FAIL=$fail skip_no_data=$skipn" | tee -a /tmp/retarget_multiobj.log
grep -h "FAIL\|skip(no" /tmp/rt_chunk_*.log 2>/dev/null | head
