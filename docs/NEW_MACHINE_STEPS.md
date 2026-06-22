# 新机器迁移 —— 一步一步

> 配套 `docs/MIGRATION_HANDOFF.md`。人只做最少 bootstrap，其余交给 Claude Code。
> ⚠️ **下载数据前先确认旧机的 HF 上传已全部完成**（否则拿到不全的数据）。

## A. 人手动跑（4 步 bootstrap）

```bash
# 0. 设根目录(非 /home/liangh 就改;代码硬编码此路径,安装脚本会 sed 自适配)
export DEXTRACK_ROOT=/home/liangh/DexTrack

# 1. 装 hf CLI(git 一般已有)
pip install -U "huggingface_hub[cli,hf_xet]"

# 2. HF 登录(token 见迁移交接对话 / HF 账号 Settings→Access Tokens;公开仓不放 token)
hf auth login --token <你的_HF_TOKEN>

# 3. 克隆主仓(公开仓,无需 auth)
git clone https://github.com/LiangHeng121/DexTrack.git "$DEXTRACK_ROOT"

# 4. 在仓里启动 Claude Code
cd "$DEXTRACK_ROOT" && claude
```

## B. 进 Claude Code 后,第一句话粘给它

```
这是从旧机迁移的项目(wuji 手 × mjlab 动作跟踪)。先完整读 docs/MIGRATION_HANDOFF.md
和它引用的 docs/mjlab_migration_plan.md 末节。然后执行迁移安装:
bash migrate_install.sh (克隆 wuji-mjlab fork、从私有 HF 下载 ~135GB 数据+ckpt、
pixi install、路径自适配),再 hf download liang12121/dextrack-wuji-mjlab-assets
SETUP_SECRETS.md --repo-type dataset --local-dir /tmp 按里面命令配 wandb/gh,
最后验证(pixi run list-envs | grep Contact、数据/ckpt 目录在)。
⚠️ 下载这步等用户确认"旧机上传完成"再跑。验证后按 doc §5 TODO 继续(扩多物体
generalist)。结论/纪律严格照 doc:接触门控、多物体>专项、apple 不加 kp、判据用
max_z、全崩先 df -h、精确 PID 杀进程、不 pip 进 pixi、不把 wuji-mjlab 提交进主仓。
```

## C. 校验(安装后,人或 Claude 都可)

```bash
cd "$DEXTRACK_ROOT/wuji-mjlab"
pixi run list-envs | grep -i Contact                                  # 4 个 *_CGSmooth_Contact
ls logs/rsl_rl/wuji_tracking | wc -l                                  # ckpt run 目录 (~53)
ls ../isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/data | head     # WUJI 参考数据
# 冒烟训练(Ctrl-C 停):
CUDA_VISIBLE_DEVICES=0 pixi run train --task WujiHand_Tracking_3Obj_CGSmooth_Contact --env.scene.num-envs 8000
```

## 备注
- HF 数据集(私有):`liang12121/dextrack-wuji-mjlab-assets`,镜像 DEXTRACK_ROOT 相对路径。
- 凭证全在私有 HF 的 `SETUP_SECRETS.md`(wandb/GitHub token + 用法),公开仓不放 token。
- `migrate_install.sh` 幂等,下载可重跑(断点续传)。CUDA 驱动需机器本身具备。
