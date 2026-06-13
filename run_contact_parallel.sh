#!/bin/bash
# 并行生成 B2 接触: 238 tag, 单线程 × 16 路。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/liangh/DexTrack/isaacgymenvs
LIST=/home/liangh/DexTrack/assets/inst_tag_list_obj_multiobj10.npy
NV=16
# tag 去掉 _nf_300
TAGS=$(python3 -c "import numpy as np; print('\n'.join(t[:-len('_nf_300')] for t in np.load('$LIST',allow_pickle=True).item().keys()))")
worker(){
  for t in "$@"; do
    python wuji_pipeline/generate_contact_guidance_grab2.py --traj "$t" --save >/dev/null 2>&1 \
      && echo "ok $t" || echo "ERR $t"
  done
}
mapfile -t arr <<< "$TAGS"
echo "[contact] $((${#arr[@]})) tags, $NV 路"
pids=()
for v in $(seq 0 $((NV-1))); do
  sub=(); j=$v; while [ $j -lt ${#arr[@]} ]; do sub+=("${arr[$j]}"); j=$((j+NV)); done
  worker "${sub[@]}" > /tmp/contact_w$v.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
ok=$(cat /tmp/contact_w*.log 2>/dev/null | grep -c "^ok ")
err=$(cat /tmp/contact_w*.log 2>/dev/null | grep -c "^ERR ")
echo "[contact] 完成: ok=$ok err=$err" | tee -a /tmp/contact_multiobj.log
cat /tmp/contact_w*.log 2>/dev/null | grep "^ERR " | head
