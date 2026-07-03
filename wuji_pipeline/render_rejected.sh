#!/bin/bash
set -e
ROOT=/data/home/liangheng/DexTrack
RETPY=$ROOT/wuji_retarget_pen_min/.pixi/envs/default/bin/python
MJPY=$ROOT/wuji-mjlab/.pixi/envs/default/bin/python
FPOSD=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1
TOPOR=$ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1/rejected
MESHDIR=$ROOT/GRAB/unzipped/tools/object_meshes/contact_meshes
WORK=/tmp/topo_render; mkdir -p $WORK

declare -A TASKMAP=( [cubesmall]=WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact
                     [cup]=WujiHand_Tracking_CupMulti_CGSmooth_Contact
                     [apple]=WujiHand_Tracking_AppleMulti_CGSmooth_Contact )

for TAG in "$@"; do
  REST=${TAG#ori_grab_}; SEQ=${REST#s*_}; OBJ=$(echo "$SEQ" | cut -d_ -f1)
  TASK=${TASKMAP[$OBJ]}
  FPOS=$FPOSD/data/wuji_passive_active_info_${TAG}_nf_300.npy
  TOPO=$TOPOR/data/wuji_passive_active_info_${TAG}_nf_300.npy
  CONTACT=$FPOSD/contact_grab2/${TAG}_contact.npy
  MESH=$MESHDIR/$OBJ.ply
  PF=$WORK/${TAG}_phi_fpos.npy; PT=$WORK/${TAG}_phi_topo.npy
  OUT=$ROOT/mjlab_topo_vs_fpos_${TAG}_rejected.mp4
  echo "########## $TAG  obj=$OBJ task=$TASK ##########"
  echo "-- per-frame penetration --"
  $RETPY $ROOT/wuji_pipeline/perframe_pen.py --fpos "$FPOS" --topo "$TOPO" --mesh "$MESH" \
    --out-fpos "$PF" --out-topo "$PT" 2>/dev/null
  echo "-- render --"
  cd $ROOT/wuji-mjlab
  TAG=$TAG TASK=$TASK OBJ=$OBJ FPOS_NPY=$FPOS TOPO_NPY=$TOPO CONTACT_NPY=$CONTACT \
    PHI_FPOS=$PF PHI_TOPO=$PT OUT=$OUT \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 \
    $MJPY $ROOT/wuji_pipeline/render_refstyle_param.py 2>&1 | grep -E "loaded|frames|WROTE|Error|Traceback" | tail -8
done
