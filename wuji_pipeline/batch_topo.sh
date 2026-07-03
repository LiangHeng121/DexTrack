#!/bin/bash
# Batch-regenerate all cube/cup/apple TOPO refs + B2 contact, gate each, keep only passers.
# Failures: outputs moved to rejected/, reason logged. Results -> /tmp/topo_gen/results.jsonl
ROOT=/data/home/liangheng/DexTrack
RETPY=$ROOT/wuji_retarget_pen_min/.pixi/envs/default/bin/python
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$ROOT/wuji_retarget_pen_min/src:$PYTHONPATH
TOPO=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1
FPOSD=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1
RES=/tmp/topo_gen/results.jsonl
REJ=$TOPO/rejected
mkdir -p "$REJ/data" "$REJ/contact_grab2"
: > "$RES"

TAGS=$(ls $FPOSD/data/ | grep -iE "_(cubesmall|cup|apple)_" | sed -E 's/wuji_passive_active_info_(.*)_nf_300.npy/\1/' | sort)
N=$(echo "$TAGS" | wc -l); i=0
for TAG in $TAGS; do
  i=$((i+1)); echo "########## [$i/$N] $TAG ##########"
  if ! bash $ROOT/wuji_pipeline/run_topo_seq.sh "$TAG" > /tmp/topo_gen/$TAG.log 2>&1; then
    echo "{\"tag\":\"$TAG\",\"PASS\":false,\"reason\":\"pipeline_error\"}" >> "$RES"
    echo "  PIPELINE ERROR (see /tmp/topo_gen/$TAG.log)"; tail -3 /tmp/topo_gen/$TAG.log; continue
  fi
  J=$($RETPY $ROOT/wuji_pipeline/gate_check.py \
    --topo $TOPO/data/wuji_passive_active_info_${TAG}_nf_300.npy \
    --fpos $FPOSD/data/wuji_passive_active_info_${TAG}_nf_300.npy \
    --motion /tmp/topo_gen/$TAG/export/motions/$TAG/motion.npz \
    --contact-topo $TOPO/contact_grab2/${TAG}_contact.npy \
    --contact-fpos $FPOSD/contact_grab2/${TAG}_contact.npy \
    --sim /tmp/topo_gen/$TAG/sim.npy --tag "$TAG" 2>/dev/null | tail -1)
  echo "$J" >> "$RES"; echo "  $J"
  if ! echo "$J" | grep -q '"PASS": true'; then
    echo "  GATE FAIL -> moving outputs to rejected/"
    mv -f $TOPO/data/wuji_passive_active_info_${TAG}_nf_300.npy "$REJ/data/" 2>/dev/null
    mv -f $TOPO/contact_grab2/${TAG}_contact.npy "$REJ/contact_grab2/" 2>/dev/null
  fi
done
echo "===== BATCH DONE. results -> $RES ====="
