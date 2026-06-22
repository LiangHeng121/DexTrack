# 新机器迁移 —— 一步一步

> 配套 `docs/MIGRATION_HANDOFF.md`。人只做最少 bootstrap，其余交给 Claude Code。
> ⚠️ **下载数据前先确认旧机的 HF 上传已全部完成**（否则拿到不全的数据）。

## A. 人手动跑（4 步 bootstrap）

```bash
# 0. 设根目录(非 /home/liangh 就改;代码硬编码此路径,安装脚本会 sed 自适配)
export DEXTRACK_ROOT=/home/liangh/DexTrack

# 1. 装 hf CLI(git 一般已有)
#    若系统 python 报 externally-managed-environment(PEP 668,Debian/Ubuntu 常见),
#    用独立 venv(推荐,不碰系统 python):
python3 -m venv ~/.venvs/hf
~/.venvs/hf/bin/pip install -U "huggingface_hub[cli,hf_xet]"
export PATH="$HOME/.venvs/hf/bin:$PATH"
echo 'export PATH="$HOME/.venvs/hf/bin:$PATH"' >> ~/.bashrc
#    (或一行图快: pip install --break-system-packages -U "huggingface_hub[cli,hf_xet]")

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

## D. 环境迁移

**mjlab/pixi(活跃工作,必需)** —— 无需迁移,`migrate_install.sh` 里的 `pixi install` 从
`pixi.lock` 精确重建。✅ 自动完成。

**conda(Isaac Gym 侧,可选;只做 mjlab 不需要)** —— 用 conda-pack 打包好放在 HF,解包即用:
```bash
# dextrack (py3.8, Isaac Gym 主环境)
hf download liang12121/dextrack-wuji-mjlab-assets conda_packs/dextrack_env.tar.gz --repo-type dataset --local-dir /tmp
mkdir -p ~/miniconda3/envs/dextrack
tar -xzf /tmp/conda_packs/dextrack_env.tar.gz -C ~/miniconda3/envs/dextrack
~/miniconda3/envs/dextrack/bin/conda-unpack          # 修复绝对路径(必做)
# wuji-retarget (py3.10, GRAB→wuji 重定向)
hf download liang12121/dextrack-wuji-mjlab-assets conda_packs/wuji-retarget_env.tar.gz --repo-type dataset --local-dir /tmp
mkdir -p ~/miniconda3/envs/wuji-retarget
tar -xzf /tmp/conda_packs/wuji-retarget_env.tar.gz -C ~/miniconda3/envs/wuji-retarget
~/miniconda3/envs/wuji-retarget/bin/conda-unpack
```
- `isaacgym/` 源码随 HF 一起下到 `$DEXTRACK_ROOT/isaacgym`(dextrack env 的 `-e` 安装指向它,通常无需重装)。
- **libpython3.8 钩子**:`conda activate dextrack` 若报 `libpython3.8.so` 缺失,见 `MIGRATION_HANDOFF.md` §3.1(加 conda activate.d 钩子把 `$CONDA_PREFIX/lib` 进 `LD_LIBRARY_PATH`)。
- conda-pack 是相对当前用户路径打的;新机 miniconda 路径不同也没事(`conda-unpack` 会修)。

## 备注
- HF 数据集(私有):`liang12121/dextrack-wuji-mjlab-assets`,镜像 DEXTRACK_ROOT 相对路径。
- 凭证全在私有 HF 的 `SETUP_SECRETS.md`(wandb/GitHub token + 用法),公开仓不放 token。
- `migrate_install.sh` 幂等,下载可重跑(断点续传)。CUDA 驱动需机器本身具备。
