#!/bin/bash
# 并行生成新物体的 TOPO 数据 + gate验收. 用法: batch_topo_scale.sh [并行度]
ROOT=/data/home/liangheng/DexTrack
RETPY=$ROOT/wuji_retarget_pen_min/.pixi/envs/default/bin/python
TOPO=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1
FPOSD=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1
REJ=$TOPO/rejected; mkdir -p "$REJ/data" "$REJ/contact_grab2" /tmp/topo_gen/res
PAR=${1:-12}
# 新物体(不含已有cube/cup/apple)
TAGS=$(ls $FPOSD/data/ | grep -iE "_(duck|elephant|mouse|phone|train|alarmclock|flute)_" | sed -E 's/wuji_passive_active_info_(.*)_nf_300.npy/\1/' | sort)
export ROOT RETPY TOPO FPOSD REJ
do_one(){
  TAG=$1
  if ! bash $ROOT/wuji_pipeline/run_topo_seq.sh "$TAG" > /tmp/topo_gen/$TAG.log 2>&1; then
    echo "{\"tag\":\"$TAG\",\"PASS\":false,\"reason\":\"pipeline_error\"}" > /tmp/topo_gen/res/$TAG.json
    echo "[$TAG] PIPELINE ERROR"; return
  fi
  J=$($RETPY $ROOT/wuji_pipeline/gate_check.py \
    --topo $TOPO/data/wuji_passive_active_info_${TAG}_nf_300.npy \
    --fpos $FPOSD/data/wuji_passive_active_info_${TAG}_nf_300.npy \
    --motion /tmp/topo_gen/$TAG/export/motions/$TAG/motion.npz \
    --contact-topo $TOPO/contact_grab2/${TAG}_contact.npy \
    --contact-fpos $FPOSD/contact_grab2/${TAG}_contact.npy \
    --sim /tmp/topo_gen/$TAG/sim.npy --tag "$TAG" 2>/dev/null | tail -1)
  echo "$J" > /tmp/topo_gen/res/$TAG.json
  if ! echo "$J" | grep -q '"PASS": true'; then
    mv -f $TOPO/data/wuji_passive_active_info_${TAG}_nf_300.npy "$REJ/data/" 2>/dev/null
    mv -f $TOPO/contact_grab2/${TAG}_contact.npy "$REJ/contact_grab2/" 2>/dev/null
    echo "[$TAG] GATE FAIL"
  else echo "[$TAG] PASS"; fi
}
export -f do_one
echo "$TAGS" | wc -l | xargs echo "总序列:"
echo "$TAGS" | xargs -P $PAR -I{} bash -c 'do_one "$@"' _ {}
echo "===== DONE ====="
