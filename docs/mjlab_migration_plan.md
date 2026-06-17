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

### ★ TODO(待办)
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
