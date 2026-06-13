#!/bin/bash
# 把 report_videos/ 增量同步到 Google Drive。配置好 rclone remote 后，每次更新只需跑这个脚本。
# 用法: bash report_videos/sync_to_gdrive.sh
set -e
REMOTE="gdrive"                       # rclone remote 名（见下方一次性配置）
DEST="DexTrack_report"                # Google Drive 上的目标文件夹
SRC="$(cd "$(dirname "$0")" && pwd)"  # = report_videos/ 绝对路径

if ! rclone listremotes | grep -q "^${REMOTE}:"; then
  echo "!! 还没配置 rclone remote '${REMOTE}'，先做一次性授权（见脚本顶部注释 / 问 Claude）"; exit 1
fi

echo ">> 同步 $SRC  ->  ${REMOTE}:${DEST}"
# sync = 让云端镜像本地（云端多出的同名旧文件会被覆盖/删除以保持一致）
# 想“只增不删”改成 copy
rclone sync "$SRC" "${REMOTE}:${DEST}" \
  --exclude "sync_to_gdrive.sh" \
  --progress --transfers 8 --checkers 16
echo ">> 完成。云端文件列表："
rclone lsf -R "${REMOTE}:${DEST}"
