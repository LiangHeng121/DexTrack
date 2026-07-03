#!/bin/bash
# Regenerate ONE sequence's TopoRetarget training reference + B2 contact.
# Usage: run_topo_seq.sh <tag>          tag = ori_grab_s2_apple_lift  (no wuji_ / _nf_300)
# Env: runs entirely in the wuji_retarget_pen_min pixi env python (CPU SQP retarget).
set -e
ROOT=/data/home/liangheng/DexTrack
RETPY=$ROOT/wuji_retarget_pen_min/.pixi/envs/default/bin/python
export CUDA_VISIBLE_DEVICES=""      # CPU-only: do not touch training GPUs
export PYTHONPATH=$ROOT/wuji_retarget_pen_min/src:$PYTHONPATH

TAG=$1
REST=${TAG#ori_grab_}                       # s2_apple_lift
SUBJ=$(echo "$REST" | grep -oE '^s[0-9]+')  # s2
SEQ=${REST#${SUBJ}_}                         # apple_lift
OBJ=$(echo "$SEQ" | cut -d_ -f1)             # apple / cubesmall / cup

GRAB=$ROOT/GRAB/unzipped/grab/$SUBJ/$SEQ.npz
ALLEGRO=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_${TAG}_nf_300.npy
FPOS=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_${TAG}_nf_300.npy
KP=$ROOT/wuji_pipeline/out/${REST}_mano_kp21.npy
MESH=$ROOT/GRAB/unzipped/tools/object_meshes/contact_meshes/$OBJ.ply

OUTDATA=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1/data
OUTCONT=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1/contact_grab2
WORK=/tmp/topo_gen/$TAG
mkdir -p "$OUTDATA" "$OUTCONT" "$WORK"

for p in "$GRAB" "$ALLEGRO" "$FPOS" "$KP" "$MESH"; do
  [ -f "$p" ] || { echo "FATAL missing $p"; exit 3; }
done

echo "=== [$TAG] obj=$OBJ  step1 crop+canon ==="
$RETPY $ROOT/wuji_pipeline/fit_crop_canon.py --grab-npz "$GRAB" --allegro-ref "$ALLEGRO" --out-sim "$WORK/sim.npy"
C0=$($RETPY -c "import numpy as np;print(np.load('$WORK/sim.npy',allow_pickle=True).item()['c0'])")
C1=$($RETPY -c "import numpy as np;print(np.load('$WORK/sim.npy',allow_pickle=True).item()['c1'])")
GEND=$((C1-1))                               # linspace(c0, c1-1, 300)

echo "=== [$TAG] step2 grab_to_hocap  crop[$C0:$C1] ==="
$RETPY $ROOT/wuji_pipeline/grab_to_hocap.py \
  --grab-npz "$GRAB" --kp "$KP" --obj-mesh "$MESH" --obj-name "$OBJ" \
  --out-dir "$WORK/hocap" --clip "$TAG" --nframes 300 --mesh-scale 1.25 \
  --grab-idx-start "$C0" --grab-idx-end "$GEND" \
  --canon-sim "$WORK/sim.npy" --object-ref-npy "$FPOS"

echo "=== [$TAG] step3 hocap_export (SQP retarget) ==="
$RETPY -m hand_retarget.io.hocap_export --clip "$TAG" \
  --hocap-dir "$WORK/hocap" --out-dir "$WORK/export" --skip-viz --overwrite

MOTION=$WORK/export/motions/$TAG/motion.npz
[ -f "$MOTION" ] || { echo "FATAL no motion.npz"; exit 4; }

echo "=== [$TAG] step4 assemble train ref ==="
$RETPY $ROOT/wuji_pipeline/assemble_topo_train_ref.py \
  --motion "$MOTION" --object-ref-npy "$FPOS" \
  --out "$OUTDATA/wuji_passive_active_info_${TAG}_nf_300.npy"

echo "=== [$TAG] step5 B2 contact (TOPO link_key) ==="
cd $ROOT/isaacgymenvs
$RETPY $ROOT/isaacgymenvs/wuji_pipeline/generate_contact_guidance_grab2.py \
  --traj "$TAG" \
  --grab_root $ROOT/GRAB/unzipped/grab \
  --allegro_ref_dir $ROOT/isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data \
  --fpos_dir "$OUTDATA" \
  --grab_mesh_dir $ROOT/GRAB/unzipped/tools/object_meshes/contact_meshes \
  --mesh_scale 1.25 --out_dir "$OUTCONT" --save

echo "=== [$TAG] DONE ==="
