#!/usr/bin/env bash
# ============================================================================
# DexTrack + wuji-mjlab 一键迁移安装(新服务器)
#
#   curl -L <这个脚本的raw地址> -o migrate_install.sh   # 或从 HF 数据集下载
#   DEXTRACK_ROOT=/path/to/DexTrack  bash migrate_install.sh
#
# 前置:① git(+GitHub 访问) ② HF 登录(私有数据集): `hf auth login` 或 export HF_TOKEN=...
#       ③ curl ④ rsync ⑤ GPU + CUDA(训练时). conda/isaacgym 仅 Isaac Gym 侧需要,mjlab 不需要.
# 做什么:克隆两个仓库 + 从 HF 拉数据(~95MB)和 ckpt(~44GB) + 装 pixi 环境 + 路径自适配.
# ============================================================================
set -euo pipefail

DEXTRACK_ROOT="${DEXTRACK_ROOT:-/home/liangh/DexTrack}"
HF_REPO="liang12121/dextrack-wuji-mjlab-assets"
# DexTrack 主仓 SSH(需 SSH key);无 key 改用 https://github.com/LiangHeng121/DexTrack.git
DEXTRACK_GIT="${DEXTRACK_GIT:-git@github.com:LiangHeng121/DexTrack.git}"
WUJI_GIT="https://github.com/LiangHeng121/wuji-mjlab.git"
WUJI_BRANCH="dextrack-tracking"
DEFAULT_ROOT="/home/liangh/DexTrack"   # 代码里硬编码的原始路径

echo "==> DEXTRACK_ROOT=$DEXTRACK_ROOT"

# 1. 克隆 DexTrack 主仓
if [ ! -d "$DEXTRACK_ROOT/.git" ]; then
  echo "==> 克隆 DexTrack -> $DEXTRACK_ROOT"
  git clone "$DEXTRACK_GIT" "$DEXTRACK_ROOT"
else echo "==> DexTrack 已存在,跳过克隆"; fi
cd "$DEXTRACK_ROOT"

# 2. 克隆 wuji-mjlab fork(dextrack-tracking 分支,这才是 mjlab 工作所在)
if [ ! -d "$DEXTRACK_ROOT/wuji-mjlab/.git" ]; then
  echo "==> 克隆 wuji-mjlab($WUJI_BRANCH)"
  git clone -b "$WUJI_BRANCH" "$WUJI_GIT" "$DEXTRACK_ROOT/wuji-mjlab"
else echo "==> wuji-mjlab 已存在,跳过克隆"; fi

# 3. HF 登录检查(数据集私有)
if ! hf auth whoami >/dev/null 2>&1; then
  echo "!! 需要 HF 登录: 运行 'hf auth login' 或 export HF_TOKEN=hf_xxx 后重跑"; exit 1
fi

# 4. 从 HF 拉数据 + ckpt,rsync 进 DEXTRACK_ROOT(HF 仓结构镜像了 root 相对路径)
STAGE="$(mktemp -d)"
echo "==> 从 HF 下载 assets(~95MB 数据 + ~44GB ckpt,较久)..."
hf download "$HF_REPO" --repo-type dataset --local-dir "$STAGE"
echo "==> rsync 数据 -> $DEXTRACK_ROOT"
rsync -a "$STAGE/isaacgymenvs" "$STAGE/assets" "$STAGE/GRAB" "$DEXTRACK_ROOT/"
[ -d "$STAGE/wuji-mjlab/logs" ] && rsync -a "$STAGE/wuji-mjlab/logs" "$DEXTRACK_ROOT/wuji-mjlab/"
rm -rf "$STAGE"

# 5. 路径自适配:代码硬编码 /home/liangh/DexTrack(env_cfgs.py + grab_object_cfg.py)
if [ "$DEXTRACK_ROOT" != "$DEFAULT_ROOT" ]; then
  echo "==> 路径改写 $DEFAULT_ROOT -> $DEXTRACK_ROOT (wuji-mjlab/src)"
  grep -rl "$DEFAULT_ROOT" "$DEXTRACK_ROOT/wuji-mjlab/src" 2>/dev/null \
    | xargs -r sed -i "s|$DEFAULT_ROOT|$DEXTRACK_ROOT|g"
fi

# 6. 装 pixi + 环境(wuji-mjlab 用 pixi,不要 pip 装进 pixi env,要加包改 pixi.toml)
if ! command -v pixi >/dev/null 2>&1; then
  echo "==> 安装 pixi"; curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
echo "==> pixi install(建 wuji-mjlab 环境,含 mujoco-warp + 内嵌 rsl_rl)"
cd "$DEXTRACK_ROOT/wuji-mjlab" && pixi install

# 7. 验证
echo "==> 验证已注册的 contact 任务:"
pixi run list-envs 2>/dev/null | grep -iE "Contact" | head
cat <<EOF

============================================================
✅ 安装完成. 关键路径:
  代码        : $DEXTRACK_ROOT/wuji-mjlab(分支 $WUJI_BRANCH)
  数据        : $DEXTRACK_ROOT/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1
  ckpt        : $DEXTRACK_ROOT/wuji-mjlab/logs/rsl_rl/wuji_tracking
  迁移交接 doc : $DEXTRACK_ROOT/docs/MIGRATION_HANDOFF.md  ← 先读这个!

训练示例(3obj 多物体 generalist,当前最优路线):
  cd $DEXTRACK_ROOT/wuji-mjlab
  CUDA_VISIBLE_DEVICES=0 pixi run train --task WujiHand_Tracking_3Obj_CGSmooth_Contact --env.scene.num-envs 8000

从 ckpt 续训:加 --agent.resume True --agent.load-run <run目录名> --agent.load-checkpoint model_X.pt
============================================================
EOF
