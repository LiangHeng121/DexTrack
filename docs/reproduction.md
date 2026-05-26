# DexTrack 复现进度与步骤

本文档维护从环境就绪到完整复现的全部步骤、命令与进度。每个阶段独立可执行，按从易到难顺序排列。

> **更新方式**：完成一个阶段后在「进度表」勾选 + 记录 reward / checkpoint 路径。遇到新坑加到「常见问题」。

---

## 进度表

| 阶段 | 内容 | 状态 | reward | 备注 |
|---|---|---|---|---|
| 0 | 环境 + 数据齐备 | ☐ | — | wh4 + 本地 Linux |
| 1 | cubesmall_inspect 单序列 | ☐ | / | |
| 2 | 与预训练对比 | ☐ | / | |
| 3a | duck_inspect | ☐ | / | |
| 3b | flute_pass | ☐ | / | |
| 4a | TACO shovel `taco_20231104_169` | ☐ | / | |
| 4b | TACO ladle `taco_20231104_186` | ☐ | / | |
| 4c | TACO soap `taco_20231103_073` | ☐ | / | |
| 5a | LEAP+Franka elephant | ☐ | / | |
| 5b | LEAP+Franka hammer (sample 6) | ☐ | / | |
| 5c | LEAP+Franka watch | ☐ | / | |
| 6a | Generalist: duck 全部轨迹 | ☐ | / | |
| 6b | Generalist: GRAB s2..s10 训练集 | ☐ | / | |

---

## 阶段 0：环境就绪

### wh4 服务器（训练）

```bash
ssh wh4
conda activate dextrack
cd /mnt/beegfs/heng/DexTrack/isaacgymenvs

# 检查清单
nvidia-smi                                                       # GPU 在
python -c "import isaacgym, torch; print(torch.cuda.is_available())"  # True
ls data/GRAB_Tracking_PK_reduced_300/data/*.npy | head -3       # 数据在
ls ../assets/meshdatav3_scaled/sem/ | head -3                   # mesh 在
ls ckpts/s2_cubesmall_inspect_ckpt.pth                          # 评估 ckpt 在
```

### 本地 Linux（评估 + 可视化）

```bash
cd ~/code/DexTrack/isaacgymenvs
nvidia-smi
echo $DISPLAY                                                    # 非空（可视化用）
python -c "import isaacgym, torch; print(torch.cuda.is_available())"
```

---

## 通用运行模板：screen

所有训练 run 都遵循这个模式：

```bash
# 1. 开命名 screen
screen -S <run_name>

# 2. 进 screen 后：
conda activate dextrack
cd /mnt/beegfs/heng/DexTrack/isaacgymenvs

# 3. 启动训练
bash scripts/<script>.sh <gpu> <seq>

# 4. Ctrl+a d 离开 screen，任务继续跑
```

**所有训练脚本自带三件事**（不用手动加任何 export / tee）：

1. **Wandb 自动同步**：脚本顶部预设 `WANDB_ACTIVATE=true`、`WANDB_PROJECT=dextrack`、`WANDB_ENTITY=liangheng-peking-university`、`WANDB_GROUP=repro_<script_stem>`（按脚本名自动分组）。
2. **三层目录：脚本 / 序列 / 时间戳**，同序列多次 run 自然聚在一起：
   ```
   ./logs/<script_stem>/<seq>/<YYYYMMDD_HHMMSS>/
       screen.log
       ckpt/
         best_ep_152_rew_158.21.pth      # 历史最佳，新 best 写入时会自动删旧的
         last_ep_200_rew_135.42.pth      # 周期性快照（每 200 epoch 一份）
         last_ep_400_rew_148.78.pth
         ...
         logs.txt                        # 每次刷新 best 时追加一行 "epoch: X, mean_rewards: Y"
       summaries/
   ```
   例如：
   - `run_tracking_headless_grab_single.sh 0 ori_grab_s2_cubesmall_inspect_1` →
     `./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/20260526_044012/`
   - `run_tracking_headless_taco_single.sh 3 taco_20231104_169` →
     `./logs/taco_single/taco_20231104_169/20260526_044012/`
3. **Screen log 固定叫 `screen.log`**（目录本身已是唯一）。脚本启动会打印 `[run] run dir: ...` 和 `[run] screen log: ...`。

**手类型 / 控制策略 在 `<script_stem>` 那一层区分**：
- `grab_single/` → Allegro 悬浮手 + 累积残差
- `grab_single_ctlv2/` → Allegro + 相对位置控制
- `grab_single_wfranka/` → LEAP + Franka
- `grab_single_syntraj_wfranka/` → LEAP+Franka 合成 reorient 轨迹
- `grab_multiple*/` → 多轨迹 generalist
- `taco_single/` → TACO 数据集 + Allegro

所以同一 seq 用不同手训练，会落到不同 `script_stem` 目录，绝不冲突。

**机制**：
- bash 把 `seq` 和 `TS` 拆成两层塞进 `log_path`
- `train_pool_2.py` 给 rl_games 传 `full_experiment_name="."` → experiment 目录 = `log_path` 直接，没有多余子层
- `A2CSupervisedAgent.__init__` 把 `nn_dir` 改为 `experiment_dir/ckpt`，所以保存目录叫 `ckpt/` 而不是 `nn/`
- `A2CSupervisedAgent.train()` 重写 ckpt 保存命名：
  - **best**：`best_ep_<X>_rew_<Y>.pth`，写入新 best 时自动删旧的（始终只有一个）
  - **last（周期）**：`last_ep_<X>_rew_<Y>.pth`，每 `save_frequency=200` epoch 一份，累加保留
  - **max_epochs 兜底**：`last_ep_<max>_rew_<Y>.pth` 或 `last_frame_<X>_rew_<Y>.pth`

**临时覆盖默认值**（不动脚本）：
```bash
WANDB_GROUP=ablation_lr bash scripts/<script>.sh 0 <seq>     # 改 wandb group
WANDB_ACTIVATE=false bash scripts/<script>.sh 0 <seq>        # 整个关掉 wandb
```

**screen 操作**：
- 列会话：`screen -ls`
- 回会话：`screen -r <run_name>`
- 杀会话：`screen -X -S <run_name> quit`
- 在会话内离开：`Ctrl+a` 然后 `d`

---

## 阶段 1：单序列复现 (cubesmall_inspect)

**目标**：epoch 50 reward > 150（README 声称）

```bash
screen -S train_cube
conda activate dextrack
cd /mnt/beegfs/heng/DexTrack/isaacgymenvs

bash scripts/run_tracking_headless_grab_single.sh 0 ori_grab_s2_cubesmall_inspect_1
```

启动后会打印 run 目录和 screen log 路径，类似：
```
[run] run dir:    ./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/20260526_044012
[run] screen log: ./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/20260526_044012/screen.log
```

**期望收敛曲线**：

| epoch | reward |
|---|---|
| 0-10 | 几十 |
| 20-40 | 50-100 |
| 50-100 | 150-200 |
| 200+ | 200+ |

**Checkpoint 路径**：
```bash
# 最新一次 cubesmall run 的 best（评估用这个）
ls -t ./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/*/ckpt/best_ep_*.pth | head -1

# 所有周期性快照（恢复训练用）
ls -t ./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/*/ckpt/last_ep_*.pth | head
```

---

## 阶段 2：与预训练对比

训到 reward 平稳后，做评估对比：

```bash
# 你训的 best checkpoint
BEST=$(ls -t ./logs/grab_single/ori_grab_s2_cubesmall_inspect_1/*/ckpt/best_ep_*.pth | head -1)

bash scripts/run_tracking_headless_grab_single_test.sh \
    0 ori_grab_s2_cubesmall_inspect_1 "$BEST" True

# 预训练 baseline
bash scripts/run_tracking_headless_grab_single_test.sh \
    0 ori_grab_s2_cubesmall_inspect_1 ./ckpts/s2_cubesmall_inspect_ckpt.pth True
```

**判定**：reward 差距 < 20 算复现成功。

---

## 阶段 3：并行复现剩余 GRAB demo

8 张 A100 并行（每张卡一个 run）。每个 screen 一个序列：

```bash
# GPU 1 - duck
screen -S train_duck -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_grab_single.sh 1 ori_grab_s2_duck_inspect_1
"

# GPU 2 - flute
screen -S train_flute -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_grab_single.sh 2 ori_grab_s2_flute_pass_1
"
```

⚠️ **一张卡只能跑一个**（默认 `numEnvs=22000`，A100 80GB 单 run 已占满）。

---

## 阶段 4：TACO 单序列

```bash
screen -S train_taco1 -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_taco_single.sh 3 taco_20231104_169
"

screen -S train_taco2 -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_taco_single.sh 4 taco_20231104_186
"

screen -S train_taco3 -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_taco_single.sh 5 taco_20231103_073
"
```

TACO 收敛比 GRAB 慢，预期 200-300 epoch 才看到好 reward。

---

## 阶段 5：LEAP + Franka

LEAP+Franka 是更难的双臂任务，收敛慢，单条 4-12 小时起。

```bash
# Elephant inspect
screen -S train_eleph -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_grab_single_wfranka.sh 0 ori_grab_s2_elephant_inspect_1
"

# Hammer (synthesized reorientation, sample 6)
screen -S train_hammer -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_grab_single_syntraj_wfranka.sh 1 ori_grab_s2_hammer_use_2 6
"

# Watch set
screen -S train_watch -dm bash -c "
conda activate dextrack && cd /mnt/beegfs/heng/DexTrack/isaacgymenvs &&
bash scripts/run_tracking_headless_grab_single_wfranka.sh 2 ori_grab_s1_watch_set_2
"
```

---

## 阶段 6：Generalist (多轨迹)

**最重，预计 1-3 天**。

```bash
# 6a: 单物体 (duck) 所有轨迹
screen -S train_gen_duck
conda activate dextrack
cd /mnt/beegfs/heng/DexTrack/isaacgymenvs
bash scripts/run_tracking_headless_grab_multiple.sh 0 '' ../assets/inst_tag_list_obj_duck.npy

# 6b: GRAB 全训练集 (s2..s10)
screen -S train_gen_all
conda activate dextrack
cd /mnt/beegfs/heng/DexTrack/isaacgymenvs
bash scripts/run_tracking_headless_grab_multiple.sh 0 '' ''
```

> **注意**：论文 Table 中 generalist 的最佳 number 依赖未释出的 *specialist-generalist 迭代训练循环* 和 *同伦优化*。纯按现有脚本跑不到论文 SOTA。

---

## 监控

```bash
# 所有 screen 会话
screen -ls

# 跟某个 run
screen -r train_cube

# 看 GPU
watch -n 2 nvidia-smi

# 看具体 run 的 log（路径在脚本启动时打印过）
tail -f logs/<script_stem>/<seq>/<TS>/screen.log | grep -E "reward|epoch"

# 最新一次训练的 log
tail -f $(ls -t logs/*/*/*/screen.log | head -1)
```

---

## 训练中断恢复

脚本里把 `export checkpoint=...` 改成最新 ckpt 路径再重跑：

```bash
LATEST=$(ls -t ./logs/<script_stem>/<seq>/*/ckpt/last_ep_*.pth | head -1)
echo "$LATEST"
# 编辑对应 script，把 export checkpoint='' 改成 export checkpoint=$LATEST
```

---

## 常见问题

### Q: 训练几个 epoch 后 reward 没动 / 是负数
最常见是数据路径错了导致 mocap reference 没加载。看 log 里：
```
==> Loading mocap reference information from ...
```
路径不对就说明 `tracking_save_info_fn` / `tracking_data_sv_root` 写错了，回 [CLAUDE.md](../CLAUDE.md) 的 "脚本路径覆盖" 部分。

### Q: OOM
默认 `numEnvs=22000` 假设 A100 80GB。卡更小就改脚本：
```bash
export numEnvs=8000      # 或更小
export minibatch_size=8000
```
注意两者要同步降。

### Q: `ValueError: Cannot parse the dataset type from obj_type: ...`
序列名前缀错了。代码靠前缀判断数据集：
- GRAB 序列必须以 `ori_grab_` 开头（例如 `ori_grab_s2_cubesmall_inspect_1`、`ori_grab_s1_watch_set_2`）
- TACO 序列必须以 `taco_` 开头（例如 `taco_20231104_169`）

README 里 wfranka watch 的训练命令写的是 `s1_watch_set_2`，**这是 README 的笔误**，正确写法是 `ori_grab_s1_watch_set_2`。doc 里阶段 5 命令已经修正。

### Q: Checkpoint 太多，硬盘吃紧
脚本默认 `save_frequency=200`，每 200 epoch 存一次。训练完手动清旧的：
```bash
find ./logs -name "last_*.pth" -mtime +7 -delete
```

---

## 关键文件

- `isaacgymenvs/train_pool_2.py` — 真正的训练入口（不是 train.py）
- `isaacgymenvs/tasks/allegro_hand_tracking_generalist.py` — 唯一存活的任务类
- `isaacgymenvs/scripts/run_tracking_headless_*.sh` — 启动脚本（很多 export 重复，最后一次生效）
- `isaacgymenvs/cfg/` — Hydra 配置，但绝大多数被脚本 CLI 覆盖

详见 [CLAUDE.md](../CLAUDE.md)。
