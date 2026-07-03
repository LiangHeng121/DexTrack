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

---

## 8. 执行进度表(2026-06-16 起,边做边更新)

**决策**:第1段先做骨架(A) → 决策点重估 → 再做 PPO(B)。pixi 走 B(先查现状再装)。
**状态图例**:⬜ 未开始 / 🔵 进行中 / ✅ 完成 / ⚠️ 卡住或有坑 / ⏸️ 暂缓

### 第1段:能跑的骨架(目标 ~半天集中干活,+ pixi 下载)

| # | 任务 | 验收标准 | 状态 | 备注 |
|---|---|---|---|---|
| 0 | pixi 现状检查(只读) | 报告 pixi/env 装没装、装它要动什么 | ✅ | 干净:pixi 没装、`.pixi/` 不存在;有 `pixi.lock`(可复现);驱动575/CUDA12.9 够新;磁盘 490G 够 |
| 0b | 装 pixi 环境 | `pixi install` 成功、`pixi run list-envs` 干净 | ✅ | pixi 0.70.2 装好(~/.pixi/bin);`pixi install` exit0,`.pixi/` 7.6G;`pixi run list-envs` 列出14任务 import干净。⚠️运行用 `cd wuji-mjlab && ~/.pixi/bin/pixi run ...`;别 pip install 进去 |
| 1 | MJCF 加 6-DOF fly 基座 | MuJoCo compile 通过;DOF 顺序 = q26 全局6 + finger1..5×j1..4 | ✅ | 新文件 `mjcf/right_fly_mjlab.xml`(不动原版);palm body 加 6 joint(slide x/y/z + hinge x/y/z)+ 6 position 执行器(kp/kv 占位待调)。验证:compile OK,nq=26,joint/actuator 顺序 `q26 完全一致`;FK 抽查 palm=q26[:3] err=0.0m,指尖跨帧位移0.31m。⚠️ 执行器 kp/kv 是占位值,步骤2/5 调 |
| 2 | 新建 `tasks/tracking/` 骨架 | 照搬 reorient/G1-tracking:command(读 npy 推进 ref[t])+ kinematics-bias action + 最简 obs + reset/term + `register_mjlab_task` | ✅ | 建好 13 文件 `src/wuji_mjlab/tasks/tracking/`:`HandObjectMotionCommand`(读我们 dict-npy、q26 列重排到 robot 关节序、reset 写手关节+物体 freejoint 到参考帧0)、`KinematicsBiasAction`(target=ref_qpos[t]+act·scale)、obs(command+joint+object pos/quat/err+last_action=114维)、reward(joint/obj_pos/obj_ori 全 exp 核 + action_rate)、time_out。注册 `WujiHand_Tracking_Cubesmall`(list-envs #15)。控制 dt=0.0166(sim 0.0083×decim2)≈GRAB60Hz |
| 3 | 骨架验收 | 零/随机策略 step 不崩 + 运动学回放渲染肉眼确认 q26→MJCF 映射对 | ✅ | **两半都过**。(a)运动学回放:`mjlab_fly_cubesmall_replay.mp4`,手型正确+抓握+cube坐标=参考。(b)smoke test(`/tmp/smoke_tracking.py`,8env GPU0):305步零+随机动作不崩、reward全程有限、reset时err_joint0.03/err_obj0.00、time_out step301触发全env truncate后重置。**第1段决策点已到——可进第2段训练** |

### 第2段:真能跟踪/举起的策略(目标 1–3 天,训练墙钟为大头)

| # | 任务 | 验收标准 | 状态 | 备注 |
|---|---|---|---|---|
| 4 | 接 rsl-rl + 最朴素 reward | 单 cubesmall 起训不崩 | ✅ | 2026-06-17 启动 `WujiHand_Tracking_Cubesmall` 4096env GPU0(detached,wandb run 8jm7c7l9)。**吞吐 ~1.4s/iter(≈7万env步/s),10k iter≈3.9h**。reward 爬升(iter10→12: 1.69→2.37,object_pos 0.21→0.28),ep长度满300无早终止。**MuJoCo+RL 能学,迁移可行性已证**。reward 只搬核心,无 softclip/idle/contact-guide |
| 5 | 训出能跟踪/举起 | 物体 z 高度贴合参考、手 qpos 跟上 ref | ✅ | **2026-06-17 ★★ 成功!策略抓起+举升+跟踪 cube**:最终配方 = **finger kp×8 + action_scale 0.1**。iter250 策略物体 max z=0.402m(参考峰0.40)、z跟踪误差均6mm/最大20mm,全程抓起→举到40cm→放下紧贴参考。视频 `mjlab_cubesmall_SUCCESS_lift.mp4`。**两步修复**见下。 ----- **抓握突破:夹持力(kp)太弱是真因**。诊断:并排 ref-vs-policy 视频显示手姿态/高度完美贴参考、手指能接触(强闭合移动cube),但 wuji 指 kp(~0.2-0.69,厂商轻柔reorient调)太弱、撑不住62.5g→举升时滑脱。**finger kp×8/kv×3 后开环(零残差)就把cube举到0.39m(参考峰0.40)**。base kp不变(本就<1cm)。已重训。⚠️后段(t200+)仍略滑,看策略残差能否补/或kp再调;sim2real真实增益待回头调。 ----- 早前进展:① reward 从我的 exp 朴素版换成**照搬 DexTrack base + pinall3**(FINGER_POS/PALM_POS 稠密跟踪参考指尖/掌世界位置 + RELAX_PALM + FIX_FINGER5);② **逐行核对发现并修两个 bug**:finger_obj coef 0.5→0.3、**env_origins 大 bug(参考多加 env_origins→多env训练物体偏移出手→抓握flag永远0→base run卡死真因)**。修复后 22000 训练 **reward 掉头上升(-2.6→-1.1)、object_inplace_bonus 0.018→0.12 增长 = 学会抓握**。⚠️ env 坐标系:mjlab 这里 sim 位置是 env-local,参考别加 env_origins |
| 6 | 物理结论复盘 | 抓握是否天然稳/反关节是否自动不犯/圆物体能否抓 | 🔵 | **单 cubesmall 跟踪在 mjlab/MuJoCo 跑通**(抓起+举升+跟踪,误差mm级)=迁移核心可行性已证。**调通的关键不是「MuJoCo 天然抓得好」,而是两处工程**:① finger kp 要够(厂商 reorient 增益太弱举不起);② action_scale 要小(残差太大策略漂离能举的参考)。反关节/圆物体/多序列待后续。下一步:训到收敛看稳定性、再扩 apple/cup 等(importer 已就绪) |

### ★ 通用物体 importer(2026-06-17)
- **`tasks/tracking/grab_object_cfg.py: get_grab_object_cfg(name)`**:GRAB `contact_meshes/<name>.ply` → free-floating `EntityCfg`(视觉全网格 + 碰撞 geom)。真实网格全在 `GRAB/unzipped/tools/object_meshes/contact_meshes/*.ply`(全套:cubesmall/apple/cup/duck/bowl/...)。
- **★ 凸性是关键坑**:MuJoCo mesh 碰撞=凸包。凸物体(cubesmall/apple)单凸包;凹物体(cup/bowl/duck/flute)用 **CoACD 凸分解**成多凸块(实测 cup→24块、duck→7块),否则手指进不去凹陷。`pixi add coacd` 已装进环境(pixi.toml/lock)。
- 网格 **×1.25**(GRAB→sim 约定,和 contact 管线一致)对齐参考位姿;接触参数仿 cube.xml(solref/solimp/friction、priority=0 让手 priority=1 赢);⚠️ MjSpec `solimp` 要 5 维(cube.xml 写3维是简写)。
- **★ 质量(踩坑已修)**:DexTrack 用统一 `rigid_obj_density=500`(非真实质量;参考就在此物理下生成,保持一致)。两个错法:① 凸分解每块算密度→重叠重复累加(cup 222g vs 真96.6g,+68%);② 整 mesh 让 MuJoCo 自动算→用**凸包体积**(空心 cup 虚高到603g)。**正解**:trimesh 算真实网格 volume/moment_inertia → 显式写 body.mass/inertia,所有 geom 密度0。现在精确:cubesmall 62.5g/duck 217g/cup 96.6g。
- **★ CPU mujoco vs mujoco-warp(踩坑)**:**CPU 经典 mujoco 独立渲染脚本对多块 CoACD 碰撞不稳**(duck 开环被弹飞 >100m)。**真实 env(mujoco-warp)稳定**(duck 开环 obj z 0.06 无爆)。→ 复杂物体物理渲染**必须用 env 渲染** `dextrack_tools/render_env_physics.py`(和训练同物理),别用 CPU 独立脚本。训练无风险。
- **已把真实 cubesmall 网格接入 tracking env**(替掉带 dex 贴图的盒子);smoke test 用真网格通过。duck 视觉+物理渲染均正确。

### ★ 训练前物理验证(2026-06-17,带物理跟踪参考)
- **工具**:`dextrack_tools/render_physics_env_reference.py`(真实 env 零残差步进=纯跟踪参考,物体真实重力/接触)。⚠️ **别用裸 `add_geom` 盒子**做物理渲染:缺 cube.xml 接触调参→接触过刚,物体被弹飞(实测冲到 z=3.15m)。真实 env 用 `get_inhand_object_cfg`(cube.xml 调好的 solref/solimp/priority)稳定。
- **结论**:开环跟踪参考,**MuJoCo 物理下手空举、cube 留桌上**(物体 z 实际 max 0.032 vs 参考 0.408,误差均0.19)。**符合预期**:参考是纯运动学轨迹不含抓握力,这正是 RL 残差要学的;和 DexTrack「kinematics_only 不抬物体」一致。**物理本身是稳的**(无爆炸)。
- 两对照视频:`mjlab_fly_cubesmall_replay.mp4`(纯运动学=理想)、`mjlab_fly_cubesmall_PHYSICS_reference.mp4`(带物理零残差=开环抓不住)。
- ⚠️ 渲染时见过 cosmetic `Warp CUDA error 2 out of memory (wp_cuda_device_get_memory_info)`——是显存信息查询失败,不影响运行(300帧正常出),训练时留意。
- **wandb**:已登录(`liangheng`,走 ~/.netrc),ppo logger=wandb,训练直接上 wandb。

### ★★ 已验证成功配置(2026-06-17,可复现基线 —— 万一后续改动失败回到这里)
**结果**:策略抓起+举升+跟踪 cubesmall,物体 max z 0.402m(参考0.40)、z误差均6mm,iter250 即达成。视频 `mjlab_cubesmall_SUCCESS_lift.mp4`。
**配置**(wuji-mjlab fork `dextrack-tracking` 分支):
- 任务:`WujiHand_Tracking_Cubesmall`,22000 env(显存~60GB,GPU0)
- **动作**:`mode="offset"`,`action_scale=0.1`(统一残差,target=ref_qpos[t]+action·0.1)
- **手指执行器 kp×8 / kv×3**(`right_fly_mjlab.xml`;否则夹不住)
- **reward**:pinall3 base(hand_pose0.5 + finger_obj0.3 + object_pos1.0 + bonus1.0 + finger_pos1.0 + palm_pos1.0,RELAX_PALM+FIX_FINGER5,无 cgsmooth/B2/softclip/idle)
- **obs**:简化 114 维(`obs_mode="simple"`)= command(ref_qpos)+joint_pos_rel+joint_vel_rel+object_pos/quat/err+last_action
- 物体:真实 cubesmall 网格×1.25,密度500(62.5g)
- sim:nconmax96/njmax512,timestep0.0083,decimation2
- **关键 commit**:env_origins修复`27c6272`、质量`5deb5f2`、pinall3`b8a41fa`、kp×8`edbcd5a`、action_scale0.1`49f190f`、wdelta开关`6850749`
- **复现**:用 obs_mode="simple" + mode="offset" + action_scale 0.1(开关切回即可)。

### ★★ full obs(DexTrack-faithful) 也成功(2026-06-17)
- 默认任务换成 **full obs 499维**(含256物体latent) + **scale_rewards_by_dt=False**(reward量级对齐DexTrack,~205)后:策略 iter300 物体 max z **0.408m**(参考0.40)、z误差均 **4mm**(比 simple obs 的6mm更准)、训练稳定无发散。视频 `mjlab_cubesmall_SUCCESS_fullobs.mp4`。
- 三任务:#15 `Cubesmall`(full obs+offset,默认,reward~百量级)、#16 `Cubesmall_Simple`(simple obs+offset+dt-scaled,精确复现首成功)、#17 `Cubesmall_Wdelta`(full obs+DexTrack wdelta动作)。
- **对齐 DexTrack 进度(除reward补丁外)**:动作(wdelta可选#17)✅、obs(full 499维含latent)✅、reward量级✅、kp 为MuJoCo侧改动(TODO核实)。

### ★★ wdelta(DexTrack动作)成功并设为默认(2026-06-17)
- DexTrack 原版 `use_kinematics_bias_wdelta`(累积残差,分组scale)直接搬:手指 scale=20 在 MuJoCo 下**太激进**→确定性策略学成 bang-bang(action‖Δ‖~3/步,手抖0.17/步,不举max z 0.028,reward -42)。
- **降手指 dof_speed_scale 20→5** 后:cube举到 **0.407m、z误差2mm(最准)、抖动0.062/步、reward +221**。视频 `mjlab_cubesmall_SUCCESS_wdelta.mp4`。
- **已设为默认** #15 `WujiHand_Tracking_Cubesmall`=wdelta;offset 降为 `_Offset`;`_Simple`保留首成功配置。
- 教训:DexTrack 动作的 scale 不能直接迁(MuJoCo 接触/积分不同),需调(像 kp×8)。bang-bang 也和缺 ACTION_RATE 惩罚有关(若要原值20需配 cgsmooth)。
- **对齐 DexTrack 现状(除reward补丁)**:动作机制(wdelta)✅、scale(手指调5,非原20)~、obs(full499含latent)✅、reward量级✅;kp×8 + 手指scale5 是 MuJoCo 侧必要适配(TODO核实现实性)。

### ★ kp 消融结论(2026-06-17)
- **开环消融**(零残差,任务直接放参考目标): kp×1 max z 0.048(不举)、×2 0.269(勉强)、×3 0.341、×5 0.392(z误差0.015最优)、×8 0.392。摩擦替代不了 kp(×1+friction1.5/3.0 仍 0.034 失败,弱夹持→法向力≈0→摩擦力≈0)。
- **★ 关键发现:训练策略下,原厂 kp×1 也能完美举起!** 任务 `WujiHand_Tracking_Cubesmall_OrigKp`(finger_kp_scale=1.0 + wdelta,其余默认),iter250 rollout **max z=0.409(参考0.405)、z 误差均 0.002m**,和 kp×8 一模一样;训练 Mean reward 240、object_inplace_bonus 58(≈kp×8)。**原因**:wdelta 残差会累积、能把手指目标设到远超参考的闭合量,即使 kp 小,大目标误差 kp·(target−q) 仍产生足够力(受 forcerange 真实电机上限约束内)。开环失败只因用了零残差=精确参考目标,没多闭合。
- **含义**:kp 增大对**训练成功非必需**——策略能自补偿。kp×8 仅加速/稳定开环行为,不是「走捷径」(forcerange 没改、力仍受真实电机上限)。
- **待用户定**:默认 kp 是否从 ×8 降?选项:×1(最贴厂商 sim2real,训练已证可行)/ ×5(开环最优,折中)/ 保持 ×8(当前默认,已验证)。我未自动改默认。仍需核对真手 wuji 控制器 kp/带宽规格作 sim2real 参照。

### ★ 三档 reward 定量对比(2026-06-17, kp×1 + wdelta, 各 500 iter)
忠实移植 cgsmooth/B2/softclip 三补丁后,对比原始/pinall3/cgsmooth_B2_softclip(任务 `WujiHand_Tracking_Cubesmall_Cmp_{Original,Pinall3,CGSmooth}`,只 reward 不同)。离线 eval `dextrack_tools/eval_reward_cmp.py`(64env×300步,fair=canonical base config无关指标)。

| 指标 | 原始(base) | pinall3 | cgsmooth_B2_softclip |
|---|---|---|---|
| **fair reward** | −27.9 | **237.6** | 213.4 (−10%) |
| max_z(m)/举升 | 0.042 ❌不举 | 0.408 ✓ | 0.409 ✓ |
| obj 跟踪(mm) | 188.6 | **2.0** | 3.0 |
| 抖动 jitter | 0.021 | **0.018** | 0.027 |
| 指误差(rad) | 0.365 | **0.206** | 0.720 |
| palm 误差(cm) | 12.2 | **0.46** | 0.72 |
| 接触距离(cm) | 10.9 | 1.14 | **0.86** |

- **★ 原始 base reward 在 kp×1 下举不起来**(max_z 4cm、fair −28、物体偏 19cm)。没有稠密 FINGER_POS/PALM_POS 跟踪,策略陷在"贴着地面物体悬停"局部最优、不敢提交举升。**这正是 pinall3 存在的理由**(稠密指尖/掌位置跟踪 bootstrap 抓握)。注:训练时 bonus 18 是地面接触刷的门控 bonus,非真举起。
- **pinall3 = 500iter 最优**:举升 0.408、2mm 物体跟踪、指误差最低 0.206、palm 0.46cm、抖动最低。复刻 DexTrack "pinall 赢家"。
- **cgsmooth_B2_softclip**:也举起、fair 213(−10%,符合 DexTrack "平滑换约 11% fair")、**接触距离最紧 0.86cm → contact_guide(B2)确实把手指拉到接触点**。**但** 500iter 下指误差/抖动反而更高(0.72/0.027)——9 项 reward 比 6/4 项收敛慢、欠训;HAND_EMA 滞后也松了手指(DexTrack 同观察 finger 1.74→2.14)。**DexTrack 的抖动↓28% 是 ep1000 才显现**,500iter 看不到平滑收益。
- **结论**:① 原始 reward 不可用(不举);② pinall3 是当前最稳选择;③ cgsmooth 的 B2 接触收紧已验证有效,但平滑收益需训到 ~1000-1500iter 才显现(待选做)。

### ★ fair reward 进 wandb/tb + 多序列就绪(2026-06-17)
- **fair_reward_metric**(rewards.py):config 无关的 canonical fair(0.6/0.1/0.1,grip0.22,palm2.0,4指,无补丁)每步写 `extras['log']['fair_reward']`,返回 0 不影响训练。三档共享同一条 `Episode/fair_reward` 实时曲线(tb 默认;agent cfg `logger="wandb"` 切 wandb)。已加到所有 reward_mode。
- **三档策略视频**:`mjlab_cubesmall_cmp_{Original,Pinall3,CGSmooth}.mp4`(Original 0.042 不举/Pinall3 0.408/CGSmooth 0.409)。渲染工具 `dextrack_tools/render_policy.py`。
- **多序列 generalist 就绪**:commands.py 扩展为多序列(stacked S×T,per-env `env_seq` 随机分配、reset 重抽),env_cfgs `wuji_hand_cubesmall_multi_tracking_env_cfg`(**23 个 cubesmall 序列 s1-s10 inspect/lift/pass,排除 3 个 offhand**),per-seq latent+contact。注册 `WujiHand_Tracking_CubesmallMulti_{Original,Pinall3,CGSmooth}`。smoke 过(obs 499、term 5/7/10、fair 记录、接触加载)。
- **待 launch(用户腾 3 卡后,每卡一档并行)**:
  ```
  CUDA_VISIBLE_DEVICES=<g> pixi run train --task WujiHand_Tracking_CubesmallMulti_<R> --env.scene.num-envs 22000 --agent.max-iterations <N>
  ```
  (R=Original/Pinall3/CGSmooth)。22000env 单档 ~62GB 适配一张 A100。

### ★ DexTrack PPO 对齐审计(2026-06-17)
对照 DexTrack wuji 多序列脚本 + `cfg/train/HumanoidPPOSupervised.yaml`,把 ppo.py 全部对齐(reward/模拟器之外):
| 参数 | DexTrack | 旧 | 已改为 |
|---|---|---|---|
| 网络 | net_type v4 `[8192,4096,2048,1024,512,256,128]` | (512,256,128) | **v4**(actor+critic) |
| learning_rate | 5e-4 adaptive(kl 0.008) | 1e-4 fixed | **5e-4 adaptive desired_kl0.008** |
| horizon_length | 32 | 24 | **32** |
| mini_epochs | 5 | 4 | **5** |
| critic_coef | 4 | 0.5 | **4** |
| clip_value | True | False | **True** |
| entropy_coef | 0.0 | 0.001 | **0.0** |
| gamma/lam/grad_norm/e_clip/minibatch数 | 0.99/0.95/1.0/0.2/32 | 同 | ✓ 本就对齐 |

- **★ 单卡最大 env 数 = 26000(模拟器差异,非那两个数)**:OOM 信息显示 `PyTorch only 102MB allocated`——**76.97GB 几乎全是 mujoco-warp 每个 world 的固定状态**(qpos/qvel/xpos/contact 等,~2MB/env),不随 nconmax/njmax 走。把 96/512→64/256 砍小后 40000 仍在 **warp CUDA graph 创建 OOM**(78GB)。瓶颈是 warp 的 graph-launch 瞬时峰值(≈稳态+15GB)。实测:22000=55GB稳、**26000=70GB稳(可跑最大)**、30000=62GB稳态但 iter0 后 graph_launch 峰值 OOM、40000=graph创建 OOM。**DexTrack 单卡能 40000 是因为 PhysX 每 env 比 mujoco-warp 省 2-3×** —— 这是"只换模拟器"的硬代价,不是参数没调对。→ env 26000(而非40000)是被 mujoco-warp 单卡内存所迫的唯一额外偏差;nconmax/njmax 保持原 96/512(砍小救不了40000且未验证物理)。
- **fair reward 进 wandb ✓**:`fair_reward_metric`(返回0不训练)每步写 `extras['log']['fair_reward']`→ logger 记 `Mean episode fair_reward`;agent cfg `logger="wandb"` 已开,实测 wandb run 生成、fair 曲线在记。
- **三档多序列长跑(2026-06-17)**:`WujiHand_Tracking_CubesmallMulti_{Original,Pinall3,CGSmooth}` **各 27000 env**(单卡折中:26000=70GB稳/28000=77GB太贴边/27000=73GB留~7GB)+ net v4 + 对齐PPO,max10000,GPU3/4/6 各一档并行。fair_reward 进 wandb(config无关,三档横比)。起步 fair~-0.6(多序列pinall3早期)。

### ★★ 三档多序列泛化对比(2026-06-18, iter~1700 中间态)
全 23 条 cubesmall 序列(无offhand)逐条离线 rollout,每条 episode-累加 `object_inplace_bonus`(满分300,统一定义 grip0.22/5指,三档可比)。工具 `dextrack_tools/eval_bonus_per_seq.py`(`ALL_SEQS=1`)。

**汇总(满分300):**
| policy | 均值 bonus | 最佳序列数 | 失败(<50%) |
|---|---|---|---|
| Original | 213.5 | 4/23 | 5 |
| Pinall3 | 257.8 | 8/23 | 4 |
| **CGSmooth** | **279.2** | **11/23** | **0 ✅** |

**结论:CGSmooth 是最稳的 generalist** —— 均值最高、最佳序列最多、**零失败**(23条全≥65%);Pinall3 失败4条、Original 失败5条。稠密跟踪项(pinall3)是多序列泛化的关键(Original 在高举升序列频繁失败);contact_guide(B2)+平滑(CGSmooth)进一步把别档失败的难序列救活。

**全 23 序列 bonus(hold%):**
| 序列 | Original | Pinall3 | CGSmooth |
|---|---|---|---|
| s10 inspect_1 | 49(16%) | 283(94%) | 285(95%) |
| s10 lift | 189(63%) | 290(97%) | 289(96%) |
| s10 pass_1 | 285(95%) | 288(96%) | 284(95%) |
| s1 inspect_1 | 82(27%) | 80(27%) | **284(95%)** |
| s1 lift | 221(74%) | 289(96%) | 290(97%) |
| s1 pass_1 | 292(97%) | 289(96%) | 290(97%) |
| s2 inspect_1 | 290(96%) | 288(96%) | 289(96%) |
| s2 lift | 265(88%) | 294(98%) | 294(98%) |
| s2 pass_1 | 286(95%) | 281(94%) | 283(94%) |
| s3 inspect_1 | 286(95%) | 284(95%) | 285(95%) |
| s4 pass_1 | 278(93%) | 275(91%) | 279(93%) |
| s5 inspect_1 | 58(19%) | 291(97%) | 287(96%) |
| s5 lift | 181(60%) | 292(97%) | 289(96%) |
| s5 pass_1 | 112(37%) | 139(46%) | **277(92%)** |
| s6 inspect_1 | 158(53%) | 283(94%) | 285(95%) |
| s6 lift | 180(60%) | 286(95%) | 285(95%) |
| s6 pass_1 | 283(94%) | 287(96%) | 286(95%) |
| s7 pass_1 | 284(95%) | 285(95%) | 284(95%) |
| s8 inspect_1 | 286(95%) | 290(97%) | 290(97%) |
| s8 lift | 226(75%) | 259(86%) | **293(98%)** |
| s8 pass_1 | 131(44%) | 139(46%) | 195(65%) |
| s9 inspect_1 | 289(96%) | 289(96%) | 288(96%) |
| s9 pass_1 | 204(68%) | 150(50%) | 209(70%) |

**对比视频**(三宫格 左→右 = Original|Pinall3|CGSmooth,工具 `dextrack_tools/render_compare.py`):
- `mjlab_cmp_ori_grab_s1_cubesmall_inspect_1.mp4` —— ★Ori&Pin 都抓不住(27%),仅 CG 成功(95%)
- `mjlab_cmp_ori_grab_s5_cubesmall_pass_1.mp4` —— Ori&Pin 失败(37/46%),CG 成功(92%)
- `mjlab_cmp_ori_grab_s5_cubesmall_inspect_1.mp4` —— Original 失败(19%),Pin/CG 举起
- `mjlab_cmp_ori_grab_s9_cubesmall_pass_1.mp4` —— Pinall3 失败(50%),Ori/CG ~70%
- `mjlab_cmp_ori_grab_s8_cubesmall_pass_1.mp4` —— 三档都难(放手),CG 最高(65 vs 46/44)
- `mjlab_cmp_ori_grab_s8_cubesmall_lift.mp4` —— 都举起,CG 最干净(98% vs 86/75)
- 单档多序列: `mjlab_cgsmooth_<seq>.mp4`(CG 在6条上 z误差1-6mm); `render_multi_seqs.py` 锁定任意序列渲染。

⚠️ 以上均为 **iter~1700 中间态**(三档仍在跑/会继续涨);待收敛重测终态 + 量化指误差/抖动(CGSmooth 平滑优势未量化)。

### ★★ 接触门控 + 多物体 generalist(2026-06-21,关键结果)
**动机**:距离 flag(finger_dist≤0.6 且 palm_dist≤0.22)在低位参考帧会对**趴地板悬停**误触发 → 策略趴地刷 fair(cup fair 112 却不举)。改用**真实接触门控**:`n_finger_contacts(读 mujoco-warp 接触对,≥2 指真碰物体)` 整体替换 `flag==2`,门控 `object_pos_tracking` + `object_inplace_bonus`(distance shaping/finger_obj 不变,仍给接近梯度)。reward_mode 加 `_contact` 后缀触发(`rewards._grasp_gate`)。**fair_reward_metric 也改成接触门控**(commit `69d4e4d`,去掉"没接触白给 bonus")。

**6 个 run 从头训**(全 `cgsmooth_b2_softclip_contact`,kp×1,save_interval250):cubesmall/cup/apple 多序列 + 3obj + cup/apple 单序列(`only_seq`)。

**判据修正**:用 **max_z vs 参考峰 ref**(`✓`=mz≥ref−0.05)判"跟上",**不用绝对 0.2 阈值**——很多 lift 参考本身只举到 0.1-0.13,绝对阈值会把"参考低但跟得准"误判为失败。

**成功率(逐条 lift 序列离线 rollout,max_z 对比 ref):**
| run | 物体 | n/m | 备注 |
|---|---|---|---|
| cubesmall 多 | cube | **6/6** | 全中 |
| cup 多 | cup | **6/8** | s8/s9 没举 |
| apple 多 | apple | **0/8**(+1部分) | 全趴地 |
| **3obj** | cube | **6/6** | |
| **3obj** | cup | **7/8** | 仅 s9 失败(比 cup专项还多1) |
| **3obj** | apple | **3/8**(+1) | s2/s4/s9 举起(专项0/8) |
| cup 单(s8) | cup | 1/1 | 0.243/0.28 |
| apple 单(s2) | apple | 0/1 | 0.064,没举 |

**★ 关键结论:3obj 多物体 generalist 在每个物体上都 ≥ 对应单物体专项**(cube 6/6=6/6、**cup 7/8>6/8**、**apple 3/8>0/8**)。多物体迁移逼真:学会抓轻 cube/cup 的抓法迁移到重 apple。**apple 用 kp×1 完全能举(3obj 证明),不需加 kp**;apple 专项失败不是抓力/训练量问题(它总样本 3.2B > 3obj 2.56B 反而更多)——是**缺多物体迁移**。→ **主推多物体 generalist 路线,放弃单物体/单序列专项。**

**iter≠训练量(回答"3obj 为何训这么快")**:3obj 8000env→13s/iter→跑到 9999;专项 20-24k env→28-33s/iter→才 ~4500。但**总样本**(iter×env×32):3obj 2.56B < cup 3.28B < apple 3.18B < cube 3.59B。env 数当初按填满显存设,非学习最优(8000env iter 快但每步样本少梯度噪)。

**⚠ eval/render 工具 bug(已修,影响 3obj 数据)**:锁序列的 `_resample_command` monkeypatch **只写激活物体、忘 park 非激活的**→3obj 里 cup/apple 堆在手边(x≈0.06/0.17 而非 x=12/14)→污染:cube"抛飞 0.89"实为**撞旁边物体弹飞**。修复(park 非激活,`_park[j]` pos+单位quat)后 cube 干净 **6/6 跟踪不再抛飞**,apple s2 从 0.152→**0.397**。**训练不受影响**(真 `_resample_command` 会 park);单物体 eval 不受影响(只 1 物体)。

**⚠ 磁盘满根因(已修,曾误判外部kill)**:net-v4 ckpt ~0.2-1.2GB × `save_interval=50` × 多 run → `logs/rsl_rl/wuji_tracking` 攒到 **630GB 填满 3.6T 盘** → 训练写不进 ckpt → **所有 job 同时崩**(看似定点外部 kill)。修:`save_interval 50→250` + 清中间 ckpt(每 run 留最新,一次释放 640GB)。**再遇全崩先 `df -h`**。

**视频**(每帧标 z;`/home/liangh/DexTrack/mjlab_ct_*` / `mjlab_3obj_*FIXED`):
- `ct_cup_s6_GOOD`(cup 举 0.47) vs `ct_apple_s2_FAIL`(apple 专项趴地) —— 同 kp×1,cup 举得起 apple 举不起。
- `3obj_apple_s2_GOOD`(generalist 把 324g apple 举到 0.40) —— 对比专项 0.064,迁移威力。
- `3obj_cube_FIXED`(park 后 cube 干净跟踪) —— 对照污染版的抛飞。

### ★★ env/kp 消融实验(2026-06-24 启动,3 个并行)
回答三个隔离问题:apple 专项失败是 env 数太多还是缺多物体迁移?加 kp 对 apple 有用吗?3obj 能否上更大 env?**严格对齐**接触门控基线,每个实验只动一个变量。

**mjlab contact 单卡 env 上限(权威值,用户记录;doc 此前缺失)**:Isaac Gym 侧(distance 门控)全部 40000;mjlab 侧(contact 门控)受 mujoco-warp 单卡显存所限各不同——
| 物体 | Isaac Gym(distance) | mjlab(contact) |
|---|---|---|
| cubesmall | 40000 | 24000 |
| apple | 40000 | 22000 |
| cup | 40000 | 20000 |
| 3obj | 40000 | 8000(待测能否更大) |

**三实验(全 cgsmooth_b2_softclip_contact + wdelta + obs full499 + net v4 + 对齐 PPO + max10000 + save250,下表只列差异变量)**:
| 实验 | task | env | kp | GPU | 研究问题 / 对照 |
|---|---|---|---|---|---|
| ① apple env | `AppleMulti_CGSmooth_Contact` | 8000 | 1 | 3 | apple 用 3obj 同 env(8000) 是否比专项 22000 更好?隔离"env 数 vs 多物体迁移"。vs 原专项(22000,0/8) + 3obj 里 apple(8000,3/8) |
| ② apple kp | `AppleMulti_CGSmooth_Contact_Kp8`(新注册) | 22000 | 8 | 4 | 加 kp(×8) 对 apple 是否有用?对齐原专项 22000,唯一变量 kp。vs 原专项(kp1,0/8) |
| ③ 3obj env | `3Obj_CGSmooth_Contact` | smoke 测上限 | 1 | 5 | 3obj 能否上更大 env(现仅 8000)?vs 现 3obj(8000,16/22) |

- **对齐核实**:doc 唯一精确训练命令 = 3obj 8000env 那条(HANDOFF L17,**训练不带 MUJOCO_GL**),实验①③直接基于它;非命令行参数全走 `__init__.py`/`ppo.py` 注册默认自动对齐。apple 专项精确 env 数 doc/ckpt/tfevents **均未存**,靠用户记录的 22000(实验②对齐它,我反推的 23000 是错的)。
- 实验② 新注册 task `WujiHand_Tracking_AppleMulti_CGSmooth_Contact_Kp8`:仅 `finger_kp_scale` 1→8,其余照搬 apple contact 专项。
- **判据**(沿用):max_z vs 参考峰(`✓`=mz≥ref−0.05),逐 apple lift 序列离线 rollout。⚠ `eval_bonus_per_seq.py` 的 3obj 锁序列 park 修复待核实(只影响实验③评估,①②单物体不受影响)。
- 启动命令(各 setsid detach,日志 `/tmp/wuji_exp{1,2}_*.log` / `/tmp/wuji_smoke_3obj_*.log`):`CUDA_VISIBLE_DEVICES=<g> pixi run train --task <T> --env.scene.num-envs <N> --agent.run-name <R>`。

**✅ Q1 结果(2026-06-26,apple env 消融已评估)**:exp1 apple 8000 训完(iter9999),逐 apple-lift 8序列 max_z(判据 mz≥ref−0.05;工具 `dextrack_tools/eval_apple_maxz.py`,自校验复现 doc 的 3obj park 值如 s2=0.397):
  | ckpt | n/8 | 说明 |
  |---|---|---|
  | apple 8000 kp1 (exp1) | **0/8** | 全趴地 max_z≈0.063(=apple 静置地面半径) |
  | apple 22000 kp1 (原专项) | **0/8** | 复现 doc |
  | 3obj park | **3/8**(s2/s4/s9) | max_z 紧贴 ref |
  - **结论:env 数不是 apple 失败原因,缺多物体迁移才是**。apple 8000=22000=0/8,env 22000→8000 毫无改善(都卡趴地悬停局部最优);只有 3obj generalist 举起 apple。

**✅ Q2 结果(2026-06-26,apple kp 消融已评估)**:exp2 apple kp8 22000(model_5000,收敛)逐序列 max_z(工具 `eval_apple_kp8.py`):
  | 配置 | n/8 | 说明 |
  |---|---|---|
  | apple kp1 8000(exp1) | 0/8 | |
  | apple kp1 22000(原专项) | 0/8 | |
  | apple **kp8** 22000(exp2) | **1/8** | 仅 s4(0.063→0.378),边际非零效应 |
  | **3obj generalist kp1** | **3/8** | |
  - **Q1+Q2 合并结论**:apple 专项失败**既非 env 数**(8000=22000=0/8)**也非抓力**(kp8 仅 1/8);**多物体迁移是唯一有效路径**(3obj 3/8 ≥ 所有专项变体)。完整印证 doc"apple 不用加 kp、主推多物体 generalist"。**Q3**(swap,3obj 更大 env)待 swap 训练收敛评估。

### ★★ 调研:能否不 load 3 物体(2026-06-25,异构 env 可行性)
动机:3obj 每 env 都 load 全 3 物体、park 2 个,显存×3(单卡 env 8000 vs 单物体 22000)+ eval 污染。问:能否每 env 只 load 它需要的 1 物体?

**核心结论**:完全省掉 3 物体 mesh 顶点显存**不行**(`mesh_vert` 是全局共享池,无 per-world 批维,3 套 mesh 必须都编进编译后的单一 MjModel)。**但**挖出两个关键事实:
1. **代码注释 `env_cfgs.py:151` "mujoco-warp can't swap mesh per env" 是错的**:mujoco-warp 的 `geom_dataid: array("*","ngeom")` **有 `"*"` 批维**,碰撞 kernel 按 `worldid % geom_dataid.shape[0]` 索引(`collision_convex.py:82,343`),配合 mjlab `expand_model_fields`(把字段第0维 tile 到 nworld)**可以 per-world 换 mesh**(同构拓扑前提)。团队没用上。(`geom_type` 无批维=类型全 world 共享不可变;`geom_size/geom_friction` 也有 `"*"`)
2. **3obj 跌到 8000 env 的主因不是 mesh 显存,是每 env 3 个 object body 撑大 contact/constraint buffer**(`nconmax=384/njmax=1536`,注释 "observed ncon≥248")。

**四路径**(评估见下,⭐=最佳性价比):
- **(c) per-world geom_dataid swap** ⭐:单个 object geom slot + 每 world 指向当前序列 mesh,每 env 只 1 object body 参与物理(不再 park)→削 contact buffer(真正主因)+消 eval 污染→**env 上限拉回**。中等工程量。**障碍**:cup 用 CoACD 多凸包(多 geom) vs cube/apple 单 hull,geom 数不同构,须先统一拓扑(geom 数量是 per-world 不可变同构约束)。**收益(拉回多少 env)未实测,需 PoC**。
- (a) 多 mujoco-warp 实例 + 共享 actor-critic 喂一个 rsl_rl runner:唯一真·每 env 只 1 物体,显存线性。工程量最大(写 MultiVecEnv 适配器拼 batch、对齐 reset/CUDA graph;`rl/runner.py` 是 rsl_rl 薄封装,假设单 VecEnv)。
- (b) primitive 近似:损失抓握保真度(cup 凹/apple 圆),与 GRAB 参考对不上,**不推荐**。运行时换 mesh 顶点机制上不可行。
- (d) specialist→distillation:团队既定正道,根本绕过(pure multi-obj RL 本就 fair~-30 难训,`env_cfgs.py:152`)。低成本。

**IsaacGym 为何能每 env 不同物体**(`allegro_hand_tracking_generalist.py:5316-5467`):PhysX `create_env`+`create_actor` 每 env 独立场景图、按 `i % len(object_list)` 只实例化那一个物体;mujoco-warp 是「单编译 Model × nworld 份 Data」SIMD,要求同构拓扑。引擎根本架构差异,单实例内学不来。

**推荐**:短期走 (d)(既定路线,且 3obj 8000 已 generalist≥专项);若坚持单实例多物体 RL 则上 (c);真要省到底才上 (a)。**待办:修正 `env_cfgs.py:150-151` 错误注释**。

**PoC 实现进展(2026-06-25,选定"先 c 再 d")**:
- ✅ **拓扑同构已验证**(推翻 agent 的"cup 不同构"):cube/cup/apple 各 = 1 visual mesh + 1 collision hull = **2 geom 完全同构**(cube is_convex→单hull;cup/apple `_CONVEX_HULL_OBJS` 强制单hull,多凸包会NaN)。→ 单一 object body slot 可容三物体,**无拓扑障碍**。
- ✅ **碰撞风险排除**:`geom_rbound`/`geom_aabb` 都是 `"*"` per-world 批字段,broadphase 按 `worldid%shape` 索引(`collision_driver.py:294/298/395`)→ 换 mesh 同时写 rbound/aabb,broadphase 不漏检。
- **完整 per-world 字段清单**(expand+reset 按 env_obj 写):`geom_dataid`(narrowphase顶点)+`geom_rbound`+`geom_aabb`(broadphase)+`body_mass`+`body_inertia`+`body_ipos`+`body_iquat`(质量惯量)。全是 per-world 批字段(mujoco-warp 为 DR 设计)。
- **接入**:mjlab `expand_model_fields`+`requires_model_fields`(`envs/mdp/dr/geom.py`/`body.py` 现成参考)。
- **方案**:新 spec(单body+多mesh池)+新 env_cfg(nconmax/njmax 回落单物体96/512)+commands swap模式(替 park)+新 task `..._Swap`(不动现有3obj作对照)。

**✅✅ PoC 成功(2026-06-25,fork 实现,4文件+257行未commit)**:
- **里程碑1核心机制✓**:探针 `scratchpad/swap_probe.py` nworld=4 设 cube/cup/apple/cube,各自正确落地(cube 0.025=半边长/cup 0.068/apple 0.056),无 NaN/穿地 → **mujoco-warp 碰撞尊重运行时 swap 的 geom_dataid**。
- **里程碑2集成✓**:新 task `WujiHand_Tracking_3Obj_CGSmooth_Contact_Swap`(list-envs #17,现有 #16 park 版未动作对照)训练 3 iter 无 NaN/OOM,`contact_guide`/`object_inplace_bonus` 上升 → 接触门控在 swap mesh 上正确工作。
- **★ env 上限拉回 2.75×**:swap 版 **≥22000 env 跑通**(GPU5 峰值 55GB) vs park 版硬上限 **8000**。证实 3-body contact buffer 膨胀(nconmax384/njmax1536)才是 8000 瓶颈,swap 回落单物体 buffer→恢复单物体 22000 上限。**实验③("3obj 更大 env")由此解锁**。
- **⚠ 待修(正式 swap-vs-park 对比前必做)**:`body_subtreemass`/`body_invweight0`/`dof_invweight0` 未 per-world 重算(停在 cube 值)→apple 用 cube solver 权重,接触阻抗略偏(smoke `unstable` 终止 0.09 可能相关);需走 event-manager `RecomputeLevel.set_const` 按物体重算。稳态吞吐(22000 vs park 8000@13s/iter)未正式测。
- 代码:`grab_object_cfg.py`(_obj_phys/_build_swap_spec/get_grab_multiobj_swap_cfg)、`commands.py`(swap模式_setup_swap/_write_swap/_resample分支)、`env_cfgs.py`(wuji_hand_3obj_swap_tracking_env_cfg)、`__init__.py`(注册)。未 commit。

**✅ invweight 修复 + 正式 swap 训练启动(2026-06-25)**:
- **invweight per-world 重算**(commands.py +3处):expand 列表加 `body_subtreemass`/`body_invweight0`/`dof_invweight0`,`_write_swap` 末尾调 `sim.recompute_constants(RecomputeLevel.set_const)` 从 per-world 质量/惯量重算 solver 常量。探针验证:修复前三物体全 cube 值(bug),修复后 apple subtreemass 0.0625→0.324、invweight0 16→3.08(最重→invweight最低,物理正确)。smoke `unstable` 终止 0.09→**0.000**。
- **四训练并行**(GPU3/4 = exp1/2 apple消融; GPU5/6 = swap):
  | run | env | GPU | iter-time | 用途 |
  |---|---|---|---|---|
  | swap 22000 | 22000 | 5 | 45.8s(含warmup) | 实验③解锁:3obj 更大 env 是否更好 generalist |
  | swap 8000 | 8000 | 6 | 21.8s | 机制等价对照:应≈park8000(16/22),验证 swap 不引偏差 |
- **⚠ 吞吐开销**:`recompute_constants` 每 reset 全 world 重算→swap 比 park 慢 ~1.7×(swap8000 21.8s vs park8000 13s)。可优化为"仅 env_obj 变化时重算"。
- **★ 样本量**:swap22000 每 iter 样本=8000的2.75×,达 park8000×10000=2.56B 等量样本只需 **~3636 iter**(不必跑满10000,否则~127h)。评估对比按等样本(~3600 iter)或等 wall-clock 取点。

**❌ swap 训练发现 bug(2026-06-25,已停诊断)**:swap 跑到 iter~267 时 mean reward 暴跌(swap8000 -8.7→**-1180**、swap22000 **-2064**)、**手部跟踪崩溃**(finger_pos_tracking **-150** vs apple专项 -5.18、hand_pose -61 vs -7、error_joint_pos **0.77→2.01 发散**),**但物体跟踪正常**(object_pos -0.5/contact_guide +7/inplace_bonus 在涨)。→ **物体 swap 对、手崩**。**根因假设**:invweight 修复的 `recompute_constants(set_const)` 全模型重算误伤手 26 dof 的 `dof_invweight0`/`body_invweight0` → 手 actuator 失稳跟不上参考。已停两 swap run(精确 PID,GPU5/6 释放,exp1/2 未动),fork 诊断。

**诊断纠正(fork) + 复核(2026-06-25)**:
- **① 原"发散"主要是指标误读**:`Episode_Reward` 是逐 episode 求和(×~300步,实测 mean episode length 302),早期大负正常;应比 **park 3obj**(fork 实测 iter8 也 -1338)而非 apple 专项(-5.18,单物体 iter1238 收敛值)。协调者当时比错基线 + 用 Episode_Reward 绝对值判"发散",**过急**。
- **② 但 `recompute_constants(set_const)` 全模型重算是真隐患**(每 reset 重算全 nv dof invweight 含手26dof + 覆写 qpos + 全 kinematics),已修:物体是独立运动学树→invweight 与手无关→改为 reset **只写物体 dof/body invweight 索引**、删 recompute。验证:手 invweight untouched(maxdiff 2.4e-4 噪声)、物体 invweight 正确(cube16/cup10.4/apple3.2)、unstable 0.000<park0.031。
- **③ 判 swap 健康用 `error_joint_pos`(均值,会回落)+unstable,别用 Episode_Reward 绝对值**(sum,早期必大负,park 同样)。
- **⚠ 收敛仍待验证**:修复版重启 swap8000(GPU5)/22000(GPU6),但 iter3-13 早期 error_joint_pos 仍在上升(2.2/3.7)、未见回落,swap22000 unstable 还 0.094。fork 的"≡park"基于 iter8 对照+smoke,**非收敛**。需训数百 iter 看 error_joint_pos 回落到 park 水平 + max_z 才算数。
- **教训**:① 比基线要同任务同阶段(park 3obj,非 apple 专项);② Episode_Reward 是 sum 不能看绝对值;③ "PoC/smoke 通过"≠长期训练健康。

**❌❌ swap 性能失败(2026-06-26,致命,已止损)**:实测稳态 iter-time — **park8000 13s / swap8000 53s(4×) / swap22000 129s(10×)**,开销随 env 数**超线性恶化**。swap22000 空跑 8 天才 5828 iter。吞吐(env-step/s):park8000 19692 / swap8000 4830 / swap22000 5457 → **swap22000 吞吐仅 park8000 的 28%**;达等样本 2.56B:park8000 **36h** vs swap22000 **130h**(跑满10000 iter 需 15 天)。**swap 拉高 env 上限(8000→22000)但吞吐崩盘,完全负收益**。
- **根因**:per-world mesh swap(每 reset 写 geom_dataid/rbound/aabb + `expand_model_fields` 把字段扩 nworld 份)是 mujoco-warp 慢路径,失去"所有 world 共享同一 geom"的优化,每 world 独立索引 → 随 env 数恶化。机制固有,非调参能救。
- **结论:swap 机制可行(碰撞正确/能拉 env 上限)但性能完全不实用,放弃用它做正式训练**。已停 swap22000(止损);swap8000 已训完(iter10000)留作 Q3 机制等价对照(纯验证价值)。**正式路线回 park8000(36h/16-22 已验证)或 specialist→distillation(路径d)**。
- **监控教训**:盯长训要算 `iter-time × 总iter = 墙钟`,不只看 iter/reward——swap22000 129s/it 从头就在,应第一次查就算出 15 天而立即止损,而非空跑 8 天占 GPU。

**❌ Q3 结果:swap 双重失败(2026-06-26)**:3obj 全22序列 max_z 三方(park8000/swap8000/swap22000):
| 物体 | park8000 | swap8000 | swap22000 |
|---|---|---|---|
| cube | 6/6 | 6/6 | 6/6 |
| cup | 7/8 | 7/8 | 7/8 |
| apple | **3/8** | **0/8** | **0/8** |
| TOTAL | **16/22** | 13/22 | 13/22 |
- **swap 机制物理正确,apple 退化是训练问题(非机制)** —— 交叉诊断纠正(协调者一度误判"机制物理退化"):cube/cup 三方等价(6/6,7/8,max_z 差≤1mm)。apple swap 0/8 vs park 3/8。**交叉诊断(park策略 × swap env)**:把 park 策略装进 swap env 评 apple → **2/8**(s3 apple 动 36.5cm、s4/s9 举到 ref)→ **swap env 物理完全支持 apple 抓举**;而 swap-训练策略让 apple 全程静止(z_range 0.005–0.009,没碰)→ **swap 训练陷入"无视 apple"局部最优**。**根因**:swap 三物体共用一个 body slot,物体身份仅靠 256-d latent(信号弱),对最重 apple → 策略抓 cube/cup 无视 apple;park 每物体独立 body,信号强,学到 apple 3/8。
- **Q3 答案:"3obj 更大 env"此路不通**:swap22000(更大env)=13/22 < park8000=16/22(swap 更大反而更差+10×慢+apple退化);park 更大撞显存墙。**park8000 (16/22)=单卡 3obj generalist 最优**;真要更大 env 只能多卡 DDP。
- **swap 定论(修正)**:机制**物理正确**+解锁 env 上限(8000→22000),但两个问题 —— ①**性能 10×慢(硬伤,per-world 字段慢路径,即使修好②也不实用)** ②as-trained 在 apple 退化(**训练条件化弱,非机制**;物体身份仅靠 256-d latent 信号不足)。**Q3"更大 env 增益"暂无法在 swap 上干净回答**(apple 训练失败掩盖 env 信号)。修复方向:加强物体条件化(喂几何/尺寸特征而非仅 latent)、apple 课程、或走 distillation。**实用性被 10×慢否决**;swap 的价值=证明机制可行 + 揭示物体条件化信号强度对 generalist 关键(park 独立 body 强 vs swap 弱 latent)。

**★ 三问最终结论(2026-06-26)**:apple 举起的唯一有效路径 = 多物体迁移(park 3obj)。① env 数不是原因(8000=22000=0/8) ② kp 边际(kp8=1/8) ③ 更大 env 单卡不可行(park 显存墙;swap 机制解锁 22000 但 10×慢+训练在 apple 退化,env 增益无法干净评估)。**park8000 16/22 是单卡最优,扩物体走 distillation**。

### ★ TODO(待办)
- **扩多物体 generalist**(主线):3obj(cube+cup+apple)已验证 generalist≥专项;下一步加更多物体 / 让 3obj 多训(现仅 2.56B 样本就 16/22)。**apple 不用加 kp**。
- **cgsmooth 训更久**(可选):单序列三档都只 500iter;cgsmooth 9 项 reward 欠训,延到 1000-1500iter 才能量化 HAND_EMA/action_rate 的抖动↓收益。
- **多卡 40000 env**(可选):若要真正对齐 DexTrack 的 40000+v4,需上多卡(DDP);单卡 mujoco-warp 上限 ~22000。
- **目标对齐 DexTrack**:除 reward 外尽量一致(只换模拟器)。动作 wdelta✅、obs full499✅、三档 reward 补丁✅(本次)、kp 为 MuJoCo 侧必要改动(见 kp 消融)。

### git / 协作基建(2026-06-16 搭好)
- **fork**:`https://github.com/LiangHeng121/wuji-mjlab`(公开 fork 自 `wuji-technology/wuji-mjlab`)。
- **remote**:`origin`=厂商(拉上游 `git pull origin main`)、`mine`=我们的 fork(推改动)。
- **分支**:所有 tracking 改动在 `dextrack-tracking`(已推 `mine`),不动 `main`。
- **工具**:gh CLI 在 `~/.local/bin/gh`(已 `gh auth login` 登 LiangHeng121,token 含 repo/workflow)。
- **原则**:tracking 任务尽量全用新文件(`tasks/tracking/` + 新 fly-base XML),少改厂商原文件 → 上游可干净 merge。`.pixi/`(7.6G 环境)gitignore,绑这个目录别乱挪。

### 已核实的关键事实(开工前对照真实文件确认)
- **q26 布局**:`[3 平移, 3 旋转, th/ff/mf/rf/lf ×4]`(读 npy 确认)。
- **★ fly 基座结构已钉死**(来源:Isaac Gym 侧 `assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf` 前54行):基座 = **串联链 3 prismatic(WRJ0x/y/z, 轴 x/y/z) 然后 3 revolute(WRJ0rx/ry/rz, 轴 x/y/z)**,全在 origin,child 链到 `right_palm_link`。**q26[0:6] 按位置 1:1 喂 `[WRJ0x,y,z,rx,ry,rz]`,旋转就是这条 x→y→z 嵌套链(不用纠结 euler 序,MJCF 照抄链顺序即可)**。限位:prismatic ±1m/effort100/damping10,revolute ±3.14/effort1000/damping1。MuJoCo 做法:在 palm body 上按此序挂 6 个 joint(slide x/y/z + hinge x/y/z),第一个最靠 parent。
- **★★ 旋转约定 gotcha(踩过,已解决)**:q26[3:6] 喂关节链产生的是 **intrinsic XYZ**(`Rx·Ry·Rz`,= scipy `'XYZ'`,= MuJoCo 单body多joint链实测)。⚠️ Isaac Gym 代码里的 `quat_from_euler_xyz()`(torch_jit_utils)其实是 **extrinsic xyz = intrinsic ZYX**(`Rz·Ry·Rx`),和关节链**差最多179°,不是同一个旋转**——**别拿它当 MuJoCo 手腕朝向的 ground truth**。真正 ground truth = URDF/关节链(intrinsic),已照抄。验证靠**数据手—物一致性**:用我的 intrinsic 链,指尖到物体中心最近距离中位 0.025m(cubesmall 半边 0.025m,指尖正贴 cube 表面=正确抓握);若约定反了指尖会甩出几十 cm。
- **✅ pixi env 渲染(GL)可用**:headless EGL 渲染要**同时**设 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`(只设前者会 EGLError)。退出时 `Exception ignored in Renderer.__del__` 的 EGLError 是无害清理报错,可忽略。系统有 libEGL_nvidia.so.575。朝向验证另用了几何法(指尖-物体距离)双保险。
- **★ 渲染 group 坑(踩过)**:wuji MJCF 有全局 `<default><geom group="3">`(碰撞组,渲染器默认隐藏)。`MjSpec.add_geom` 加物体若不显式设 group,会继承 group3 → **物体全程不渲染(看着像没生成)**。修复:`cube_geom.group = 2`(可见组)。手能显示是因其视觉 mesh 显式标 `group="1"`。
- **MJCF 关节顺序**:finger1→5 = 拇/食/中/环/小,joint1→4,与 q26 指块对齐;tip site `right_fingerN_tip` 已存在可直接用。
- **数据真实路径**:`isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/`(原 §4③ 少了 `isaacgymenvs/` 前缀)。
- **物体**:原型先用 reorient 的 dex-cube box 缩放(非真 GRAB cubesmall mesh);若抓握对几何敏感再换。
- **mjlab 动作模板**:`JointPositionOffsetEMAAction`(`target=default_pos+action·scale`+EMA+warmup)≈ 我们的累积残差,把 `default_pos` 换成 `ref_qpos[t]` 即可。
- **reorient PPO 规模**:8192 env × 5000 iter;单序列 tracking 可少。注册走 `register_mjlab_task` + `WujiOnPolicyRunner`。
- **★ mjlab 自带可参考任务**(`pixi run list-envs` 发现):`Mjlab-Tracking-Flat-Unitree-G1`(人形逐帧轨迹跟踪 = command-term 范式,比 reorient 更贴 tracking)、`Mjlab-Lift-Cube-Yam`(举升 reward 范式)。建 command/reward term 时优先参考这俩。

### ★ specialist→generalist 在线蒸馏(可 scale 架构, 2026-07-03 规格)
**目标**:DexTrack 的 specialist→generalist,但**避开"多物体单实例"**(park/swap 都 scale 不了几百物体)。
**核心架构 = 多单物体实例 + 共享 student**:
- 每物体一个**纯单物体**仿真实例(mujoco-warp 快路径、能上大 env、无 park/swap 开销);student 是一个共享权重网络,在 N 个实例各跑 rollout,各自被对应老师标注,经验在 batch 维拼接喂一个 PPO+BC 更新。
- 多物体性在**网络**里(一个 net 吃所有物体经验,靠 256-d obj latent 区分),不在仿真里 → scale 到几百物体 = 每轮抽 k 个物体各开单实例轮转(天然 curriculum),不往一个仿真塞几百 mesh。
- DAgger 风格(student 探索的 state、老师在线标注),非离线 BC。
**关键代码事实**(已核实):
- rsl_rl 是 **in-tree 可编辑源码** `src/wuji_rl_libs/rsl_rl/rsl_rl/`(非只读包!),learn 循环 `runners/on_policy_runner.py:79-98`(act→step→process_env_step),PPO update `algorithms/ppo.py:215-306`(surrogate/value loss)。**BC 损失直接加在 ppo.py:update**。
- VecEnv 契约 = `mjlab/rl/vecenv_wrapper.py`:num_envs/num_actions/device/max_episode_length/get_observations()→TensorDict/reset()/step(actions)→(obs,rew,dones,extras)。
**三块实现**:① MultiVecEnv 适配器(新文件~150行):持 N 个单物体 RslRlVecEnvWrapper,对外一个 VecEnv,batch 维 cat,暴露 env_teacher_id 路由张量。② 老师标注+BC(改 in-tree rsl_rl~80行):runner 载 3 冻结 specialist,rollout 存 teacher_actions,ppo update 加 `bc_loss=supervised_coef×MSE(student_mean,teacher)`,**默认 coef=0(开关)**。③ 注册 distillation task(小):3 单物体 env_cfg + teacher ckpt 路径 + coef。
**第一步验证**:3 老师(cube 6/6+cup 6/8+apple 0/8 全纳入),student 从零,验收=管线正确(student 复现老师水平)。**注意天花板**:现有 specialist 都 ≤ 3obj(16/22),纯蒸馏不指望超基线;近期价值=跑通 BC/DAgger 基建,等 apple 消融出好老师再战。

### ★ 待办:多实例纯 PPO 3obj(coef=0, MultiVecEnv 跑通后)
MultiVecEnv 适配器是共用地基:配 BC 损失=蒸馏(需老师);**supervised_coef=0=纯 PPO 多实例 generalist(无老师)**,即 park 3obj 的正统升级——3 个单物体实例(cube/cup/apple)喂一个共享 student,每实例快路径、能上各自单卡上限(24000/20000/22000)、无 park/swap 开销、无 eval 污染。
**待办(蒸馏管线跑通 + 有空闲卡后)**:跑 coef=0 多实例 3obj,对比 park 3obj(16/22),验证多实例(网络层迁移)能否 ≥ park(同环境迁移)。若 ≥,则 park/swap 都可退休,"多物体训练"统一走"多单物体实例+共享网络",直接 scale 到几百物体(抽 k 个物体轮转)。
**未知**:park 的 apple 3/8 靠"同环境多物体迁移",多实例靠"网络层迁移",是否等效需实测。

### 当前进行中(2026-07-03,会话末快照)
- **apple 消融 9 实验**:5 起训(GPU5=MassCur/6=R123/0=R1/1=R2/7=R3,8000env/10000iter,~32h),队列脚本 PID 3906539 盯着腾卡续 RSI→ET→Noise→Friction。日志 /tmp/wuji_ap_*.log。**smoke early signal:R1 ungated 8/8 vs gated 0/8(逃生门=病灶)**。全开关默认关、现有 task 未变。**未 commit**。
- **蒸馏管线**:fork 在 GPU3 实现 MultiVecEnv+BC(改 in-tree rsl_rl,规格见上上节)。**未 commit**。
- 待评估:apple 消融 ~5000iter 中途 max_z(工具 eval_apple_maxz.py)。
