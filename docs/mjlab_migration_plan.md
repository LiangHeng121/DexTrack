# mjlab 迁移评估 + 最小跟踪任务计划（2026-06-08，调研记录，回头再迁）

记录"是否/如何把 DexTrack 从 Isaac Gym 迁到 mjlab(MuJoCo)"的调研结论 + 最小单序列 cubesmall 跟踪任务的搭建清单。**结论:技术上该迁(MuJoCo 物理直击我们的根因),但是平台级工作量;先做一个有界的最小原型验证再决定全面迁移。**

## 1. 模拟器/框架关系(先理清两个层次)

| 层次 | 选项 | 说明 |
|---|---|---|
| **物理引擎** | **PhysX** | NVIDIA,游戏物理出身,接触/关节限位对操作不够精(我们踩的反关节软限位、抓握脆弱) |
| | **MuJoCo** | DeepMind,专为机器人接触操作,接触准、**关节限位是硬的**;`mujoco-warp` 是它的 GPU 版(几千 env 并行) |
| **RL 框架** | **Isaac Gym** | NVIDIA 老框架,引擎=PhysX,**已停更(py3.8)。DexTrack 现在用的就是这个** |
| | **IsaacLab** | Isaac Gym 接班人,引擎仍 PhysX,manager-based 架构,现代但重(Omniverse) |
| | **mjlab** | **把 IsaacLab 那套搬到 MuJoCo 上**(mujoco-warp),轻量、现代。`wuji-mjlab` 用它 |

类比:`mjlab : IsaacLab = MuJoCo : PhysX`(架构同,引擎换)。

**我们现在**:Isaac Gym Preview 4 + PhysX + rl_games(`a2c_supervised*`),conda `dextrack` / py3.8。

## 2. wuji-mjlab 是什么(`/home/liangh/DexTrack/wuji-mjlab`)

wuji 厂商官方栈:wuji-mjlab(任务+部署)→ mjlab==1.3.0 → MuJoCo + mujoco-warp;RL=rsl-rl。现做**手内 cube 重定向**(WujiHand_Reorient),但**框架通用**(文档有"Adding a new task"),**tracking 任务可搭**。带完整 **sim2real**(ONNX→真手 + 视觉 + ZMQ)。现代栈(py3.11/torch2.7/CUDA12.8)。**pixi-only 安装**(非 conda/pip)。

## 3. 为什么值得迁(技术理由)

- **MuJoCo 物理直击我们一直在打的根因**:反关节超界(PhysX 软限位→MuJoCo 硬限位,**CLIP_DOF hack 不需要**)、抓握脆弱、flute 抓不住——多是 **PhysX 的锅**,MuJoCo 很可能让它们自然消失,reward 工程大幅简化(连 contact guidance 都可能少需要)。
- **Isaac Gym 是死路**(停更、py3.8);mjlab 活跃维护。
- **sim2real 现成**——对正是这只 wuji 手有可跑真机闭环。
- wuji 手已在 MJCF 建好(调好接触参数 + 解剖分组 DR);生态完整(`wuji-retargeting` 重定向 + mjlab + sim2real)。

**代价**:换引擎+框架=把跟踪管线在 mjlab 重建;RL 栈不同(rsl-rl vs rl_games);规模存疑(IsaacGym 跑 40000 env 多任务,reorient 只 8192,mujoco-warp 大规模多轨迹速度待测);pixi 环境、数据管线重接。

## 4. 最小单序列 cubesmall 跟踪 —— 搭建清单(回头照此做)

**范围**:只跑单条 cubesmall 跟踪,**不要** multi/generalist、监督蒸馏(a2c_supervised)、250 flag、TACO/franka/leap、forecasting/teacher/vision、那堆 reward 开关。

**① 机器人模型(MJCF)—— 小改,大部分复用**
- ✅ wuji 手 MJCF 已有:`wuji-mjlab/src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml`(20 指 DOF);
- ⚠️ **加 6-DOF fly 基座**:reorient 把手腕固定(只转 cube),我们要手腕能动(才能举)。根部加 3 平移+3 旋转关节,对应 DexTrack 的 `WRJ0x/y/z/rx/ry/rz`;
- ✅ cube 物体已有:`assets/objects/inhand_object/xmls/cube.xml`,缩到 cubesmall(5cm)。

**② mjlab 任务(主要工作量,新建 `tasks/tracking/`,manager-based 项)**
- **action term**:kinematics-bias(目标 = 参考 qpos[t] + 残差·scale);
- **obs terms**:手 qpos/qvel + 物体位姿 + 参考下一帧;
- **reward terms**(只搬核心):跟踪(qpos→参考)、物体跟踪(obj→参考 obj)、指-物距离、抓握 bonus/举升;
- **command term**:加载 cubesmall 参考(q26 + 物体位姿,300 帧),按步推进、暴露 ref[t];
- **reset/termination**:复位到参考帧 0、定长 300。

**③ 数据 —— 现成**:cubesmall 参考 npy(`data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy`,含 q26 `robot_delta_states_weights_np` + 物体位姿 + FPOS 指尖)塞进 command term。⚠️ **q26 的 DOF 顺序要对上 MJCF 关节顺序**。

**④ RL**:rsl-rl PPO,复制 reorient 的配置改超参。

**模板**:`wuji-mjlab/src/wuji_mjlab/tasks/reorient/`(`reorient_env_cfg.py` 工厂、`reorient_terms.py` 所有 MDP 项、`config/wuji_hand/` 薄绑定层、`mdp/` 观测/命令/动作)。照此结构建 `tracking/`。

## 5. 要盯的坑

- **DOF 顺序**:q26 ↔ MJCF 关节(和我们做 contact 生成时一样,Isaac Gym 侧是 6 全局[3 trans+3 rot] + finger1..5×joint1..4);
- **fly 基座建模**:3 滑+3 铰对上 `WRJ0*`(MuJoCo 里也可用 free joint,但 free 是 quat,为对上 q26 的 6 显式 DOF 用 3 slide+3 hinge);
- **四元数约定**:物体 `object_rot_quat` 是 **xyzw**(Isaac Gym 侧确认过);
- **mujoco-warp 在目标 env 规模下的速度**;
- **pixi 环境**:这台机是 Isaac Gym 的 conda 栈,mjlab 要另起 pixi 环境(可能有摩擦)。

## 6. 工作量估计(我来做)

- **第1段:能跑的骨架**(env 加载/随机策略 step/不崩)≈ **半天-1天**(大头是学 mjlab API + pixi 装环境 + MJCF fly 基座 + DOF 对齐);
- **第2段:真能跟踪/举起的策略** ≈ **再 3-5 天**(reward 配平 + MuJoCo 接触/执行器调参 + 训练墙钟 + 可能的 API/物理惊喜);
- **合计 ~一周内**出一个验证过的单 cubesmall 跟踪原型。
- **最大不确定**:没用过 mjlab 的学习曲线 + MuJoCo 物理调参,真动手前估不准。

**决策点**:做完第1段(骨架 + 摸清 API)后重估第2段;若 MuJoCo 下抓握天然稳、反关节自动不犯,就是迁移值得的决定性证据。

## 7. 相关
- `wuji-mjlab/`(本地 clone)、`wuji-mjlab/README.md`、`tasks/reorient/`(模板);
- `wuji-retargeting`(厂商重定向,数据管线可能复用);
- 当前 Isaac Gym 侧:`docs/cubesmall_single_multi.md`(本轮所有实验 + reward 开关 + 物理调查)。
