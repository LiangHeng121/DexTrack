# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目简介

DexTrack 是一个基于 NVIDIA Isaac Gym 的强化学习系统，用于灵巧机器人手—物体操作跟踪。它训练策略来模仿运动学参考轨迹（这些轨迹由 GRAB/TACO 数据集中的人—物交互重定向而来），支持两种本体：
- **悬浮 Allegro 手 (fly Allegro hand)** —— 带 6 个全局自由度的 Allegro 手，无机械臂。
- **LEAP 手 + Franka Panda 机械臂**（由 `w_franka` 标志控制）。

两种动作空间（"控制策略"）：
- **累积残差** + 运动学偏置 (`use_kinematics_bias_wdelta=True`) —— DexTrack 原始动作空间。
- **相对位置** 目标 (`use_relative_control=True`) —— 对应脚本中的 `_ctlv2` 变体。

两种训练模式：
- **单轨迹训练** —— 为单条参考序列训练一个策略。
- **多轨迹训练（通才策略 generalist）** —— 一个策略跟踪多条序列，可选择 RL 与监督/IL 损失并用。

## 环境准备

仓库中**不包含**外部数据，需按 `README.md` 的 "Data" 一节从 OneDrive 下载（多个大 `.zip`）：
- `isaacgymenvs/data/{GRAB_Tracking_PK_reduced_300, GRAB_Tracking_PK_LEAP_OFFSET_..., modified_kinematics_data_leap_wfranka_v15urdf, TACO_Tracking_PK_reduced}/`
- `assets/{datasetv4.1, meshdatav3_scaled, rsc/objs}/`
- 可选的预训练 checkpoint 放在 `isaacgymenvs/ckpts/`

安装（Python 3.8）：
```bash
conda create -n dextrack python=3.8.0 && conda activate dextrack
cd isaacgym/python && pip install -e .        # Isaac Gym Preview 4 (需另外从 NVIDIA 官网下载)
pip install whls/torch_cluster-1.6.3+pt24cu121-cp38-cp38-linux_x86_64.whl
pip install rl_games transforms3d matplotlib omegaconf hydra-core trimesh mujoco tqdm
```

**重要**：很多 shell 脚本里写死了 `/cephfs/...` 路径（原作者共享文件系统的路径），这些值会在同一脚本的后面被本地路径（`./data/...` / `../assets/...`）**再次覆盖**。排查 "file not found" 时，请从头到尾通读脚本——**最后一次 `export` 才生效**，真正起作用的设置就在 `python ...` 调用之前那一段。

## 训练与评估

**所有命令必须在 `isaacgymenvs/` 目录下运行**（脚本使用 `../assets/...` 这种相对路径）。脚本按位置传参；`<GPU_ID>` 替换为 GPU 索引（仅支持单 GPU），`<SEQ_NAME>` / `<TAG>` 替换为序列名，例如 `ori_grab_s2_cubesmall_inspect_1` 或 `taco_20231104_169`。

单轨迹 (fly Allegro)：
```bash
bash scripts/run_tracking_headless_grab_single.sh        <GPU_ID> <SEQ_NAME>                     # 训练，累积残差
bash scripts/run_tracking_headless_grab_single_ctlv2.sh  <GPU_ID> <SEQ_NAME>                     # 训练，相对位置
bash scripts/run_tracking_headless_grab_single_test.sh   <GPU_ID> <SEQ_NAME> <CKPT> <HEADLESS>   # 评估 (HEADLESS=False 需要显示设备)
bash scripts/run_tracking_headless_taco_single.sh        <GPU_ID> <TAG>                          # TACO 数据集
```

多轨迹 (fly Allegro)：
```bash
bash scripts/run_tracking_headless_grab_multiple.sh  <GPU_ID> <SUBJ_NM> <SEQ_TAG_LIST>
# SEQ_TAG_LIST 是列出轨迹的 .npy 文件，例如 ../assets/inst_tag_list_obj_duck.npy
# SUBJ_NM 和 SEQ_TAG_LIST 都为空时表示 "GRAB 训练集所有 subject s2..s10"。
```

LEAP + Franka：
```bash
bash scripts/run_tracking_headless_grab_single_wfranka.sh         <GPU_ID> <TAG>
bash scripts/run_tracking_headless_grab_single_syntraj_wfranka.sh <GPU_ID> <TAG> <SAMPLE_ID>   # SAMPLE_ID 取 [0,99]，对应合成的 in-hand 重定向轨迹
bash scripts/run_tracking_headless_grab_multiple_wfranka.sh       <GPU_ID> <SEQ_TAG_LIST>
```

Checkpoint 保存在脚本中 `log_path` 指定的目录（默认 `./logs/...`）；训练日志/run 输出到 `./runs/`（或脚本覆盖后的 `log_path`）。

## 代码架构

整体在 Isaac Gym 之上分三层：

**任务 / 环境层** —— `isaacgymenvs/tasks/allegro_hand_tracking_generalist.py`（约 14.7k 行）是当前唯一在用的任务类。`tasks/__init__.py` 里的 `isaacgym_task_map` 几乎全被注释掉了——`AllegroHandTrackingGeneralist` 一个类同时处理：单/多轨迹训练、Allegro / Allegro+Franka / LEAP+Franka 三种本体。所有行为开关（动作空间、观测类型、奖励权重、监督损失、teacher distillation、forecasting、vision 等）都由 trainer 以 kwargs 形式传进来。需要理解环境行为时，**以这个文件为准**，不要只看 cfg YAML 的默认值。

**Trainer / Orchestrator 层** —— `isaacgymenvs/train_pool_2.py`（约 1.5k 行，约 250 个 argparse 参数）是 `scripts/` 下**所有**脚本真正调用的入口。它的工作流程：
1. 解析一个超大的扁平 argparse 命名空间（每个行为开关都是一个 CLI flag）。
2. 根据 `--launch_type`（`trajectory`、`trajectory_baseline_search`、`object_type`）枚举 "data tag"（轨迹 / 物体实例）。
3. 对每个 tag 构造训练 run 名称，把 flags 覆盖到 Hydra 配置上，调用 `launch_one_process(...)` 创建 Isaac Gym 环境 + rl_games trainer。"pool" 这个名字是历史遗留——单实例训练时其实只跑一个进程。

`train.py` 是 IsaacGymEnvs 上游标准的 Hydra 入口；`train_2.py` 是一个精简变体。**但脚本都不调用它们**，全部走 `train_pool_2.py`。

**RL 算法层** —— `isaacgymenvs/learning/` 里是基于 rl_games 扩展的自定义 agent，加入了本项目的若干技巧：`a2c_supervised*.py`（PPO + 来自 teacher / 优化轨迹的行为克隆损失）、`a2c_dagger_continuous.py`（DAgger）、`a2c_fromsupervised.py`（从监督预训练 warm-start）、`a2c_supervised_wplanning*.py`（带 forecasting head）。监督损失系数 `--supervised_loss_coef` 控制 IL/RL 混合比例，这是论文中 "specialist-generalist 迭代训练" 思想的核心。`amp_*` 系列是 IsaacGymEnvs 上游遗留下来的，**未使用**。

**配置层** —— `isaacgymenvs/cfg/` 走 Hydra 结构（`config.yaml` + `task/*.yaml` + `train/*.yaml`）。但 shell 脚本几乎把所有重要参数都当 CLI override 传进来，所以 YAML 主要提供默认值和结构；**不要假设 YAML 里的值还生效**，要先看脚本是否覆盖。

**测试入口** —— `test_pool.py` 与 `test_generalist_pool.py` 是 `train_pool_2.py` 的镜像版本，只不过运行在推理模式，由 `*_test.sh` 脚本调用。

## 修改前需要知道的事

- **累积残差动作空间 (`use_kinematics_bias_wdelta=True`) 与相对位置动作空间 (`use_relative_control=True`) 互斥**，分别走任务文件中不同的代码路径。一些奖励项和观测类型只对其中一种有意义（例如 `pure_state_wref_wdelta` vs `pure_state_wref`）。脚本里的搭配是正确的——**复制脚本时要保留这种搭配**。
- **仓库没有测试套件、没有 lint 配置、没有 CI**。验证手段就是跑几个 epoch 的训练脚本观察奖励曲线。
- **论文中提到的两个组件未在此仓库释出**（README 已说明）：单轨迹跟踪的同伦优化 (homotopy optimization)，以及 specialist↔generalist 迭代训练循环。监督损失相关代码 (`a2c_supervised*`) 和 teacher 相关 flag（`--use_teacher_model`、`--teacher_model_path`、`--teacher_index_to_weights`）是构建那个（未释出的）迭代循环的零部件。
- **手写 shell 脚本里同一变量会被反复覆盖**。许多脚本中同一个变量会有多行 `export FOO=...`——**只有 `python` 调用前最后一次赋值才有效**。修改行为时找最后一次赋值，而不是第一次。
- **`.gitignore` 全局排除 `*.npy`**，包括 `assets/` 下的。脚本中引用的轨迹 `.npy` 和 `inst_tag_list_*.npy` 文件预期在本地存在但不入 git。
