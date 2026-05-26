#!/bin/bash
# 把单次 rollout + 所需的 URDF / mesh 打包给 Mac 端渲染用。
# Usage:
#   bash render/pack_for_mac.sh <ROLLOUT_NPY_PATH> <SEQ_NAME> [OUT_TARBALL]
# Example:
#   bash render/pack_for_mac.sh \
#     isaacgymenvs/logs/.../ts_to_hand_obj_obs_reset_1.npy \
#     ori_grab_s2_cubesmall_inspect_1
set -euo pipefail

ROLLOUT_NPY="$1"
SEQ_NAME="$2"
OUT_TARBALL="${3:-/tmp/dextrack_render_${SEQ_NAME}.tar.gz}"

REPO=/mnt/beegfs/heng/DexTrack
URDF_FLY=allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf
OBJ_MESH_DIR=meshdatav3_scaled/sem/${SEQ_NAME}
REF_NPY=isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_${SEQ_NAME}_nf_300.npy

STAGE=$(mktemp -d)
trap "rm -rf $STAGE" EXIT

mkdir -p "$STAGE/rollout"
mkdir -p "$STAGE/assets"

cp "$ROLLOUT_NPY"                       "$STAGE/rollout/ts_to_hand_obj_obs_reset_1.npy"
cp -r "$REPO/assets/allegro_hand_description"  "$STAGE/assets/"
mkdir -p "$STAGE/assets/$(dirname ${OBJ_MESH_DIR})"
cp -r "$REPO/assets/${OBJ_MESH_DIR}"           "$STAGE/assets/${OBJ_MESH_DIR}"
[ -f "$REPO/$REF_NPY" ] && cp "$REPO/$REF_NPY" "$STAGE/rollout/reference.npy" || true

cat > "$STAGE/meta.txt" <<EOF
seq_name=${SEQ_NAME}
urdf=assets/${URDF_FLY}
obj_mesh=assets/${OBJ_MESH_DIR}/${SEQ_NAME}.obj
rollout=rollout/ts_to_hand_obj_obs_reset_1.npy
reference=rollout/reference.npy
EOF

tar -czhf "$OUT_TARBALL" -C "$STAGE" .

echo
echo "Packed: $OUT_TARBALL"
echo "Size:   $(du -h $OUT_TARBALL | cut -f1)"
echo
echo "On Mac:"
echo "  scp ${USER:-$(whoami)}@$(hostname):${OUT_TARBALL} ~/Downloads/"
echo "  mkdir -p ~/dextrack_render && tar -xzf ~/Downloads/$(basename $OUT_TARBALL) -C ~/dextrack_render"
