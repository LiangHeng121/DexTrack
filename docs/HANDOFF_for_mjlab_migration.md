# DexTrack → mjlab 迁移 · 完整交接文档

> 给负责 mjlab 迁移的新 Claude Code 会话。本文档汇总了截至 2026-06-16 的**所有持久 memory** + 当前 Isaac Gym 侧 DexTrack 工作的全部关键发现、reward 设计、数据布局、操作坑、当前实验状态。
> 迁移计划本体见 **`docs/mjlab_migration_plan.md`**(最小单序列 cubesmall 跟踪原型清单);本文是它的知识背景补充。

---

## 0. TL;DR — 迁移最该知道的

- **为什么迁**:我们一直在打的物理根因(反关节超界、抓握脆弱、flute/圆物体抓不住)大多是 **PhysX 的锅**。MuJoCo 硬关节限位 + 准接触很可能让它们自然消失,**reward 工程大幅简化**(CLIP_DOF hack、可能连 contact guidance 都少需要)。Isaac Gym 已停更(py3.8)。
- **目标栈**:mjlab(MuJoCo + mujoco-warp)+ rsl-rl,对齐 wuji 厂商仓库 `/home/liangh/DexTrack/wuji-mjlab`(它做手内 reorient + sim2real,框架通用可搭 tracking)。
- **本体**:悬浮 wuji 5 指手(20 指 DOF + 6 全局 DOF fly 基座)。当前 Isaac Gym 侧任务文件 `isaacgymenvs/tasks/allegro_hand_tracking_generalist.py`(~14.7k 行,唯一在用的任务类)。
- **任务**:模仿 GRAB/TACO 重定向的运动学参考轨迹(手 q26 + 物体位姿,300 帧),让策略物理跟踪。
- **迁移时复用**:wuji MJCF(`wuji-mjlab/.../wuji_hand/mjcf/right_mjlab.xml`)、cube.xml、我们的参考 npy + contact 数据。**新建** manager-based tracking 任务(action/obs/reward/command/reset),模板 `wuji-mjlab/src/wuji_mjlab/tasks/reorient/`。

---

## 1. 服务器 / 环境(memory: dextrack-server-setup)

- **路径**:`/home/liangh/DexTrack/isaacgymenvs`(所有命令在此目录跑,脚本用 `../assets/` 相对路径)。
- **硬件**:单机 **8× A100-SXM4-80GB**。
- **环境**:conda env `dextrack`(torch 2.4.1+cu121,py3.8)。GRAB 1269 npy、TACO/LEAP-Franka 数据、mesh、预训练 ckpt 全在本地。
- **阻断性修复**:isaacgym import 报 `libpython3.8.so.1.0` 缺失。已加 conda 激活钩子 `~/miniconda3/envs/dextrack/etc/conda/activate.d/ld_library_path.sh` 把 `$CONDA_PREFIX/lib` 加进 `LD_LIBRARY_PATH`,`conda activate dextrack` 后自动生效。
- **mjlab 侧**:`wuji-mjlab` 用 **pixi**(非 conda/pip),py3.11/torch2.7/CUDA12.8,要另起环境。
- ⚠️ **GPU 共享(重要)**:这台机是多用户共享,**`tongxuantian` 和 `andrew` 是别的 Linux 用户,他们的进程无权 kill**(`/proc/PID` 不可写)。`andrew` 在 GPU0 常驻 ~17GB。抢卡只能等整张空出来。
- ⚠️ **不要 commit**:`spider/`(441MB,andrew 的)、`wuji-mjlab/`(厂商 clone)。

---

## 2. 当前 Isaac Gym 架构(迁移要照搬的行为以此为准)

三层(`docs/CLAUDE.md` 有详述):
- **任务层** `tasks/allegro_hand_tracking_generalist.py` —— 一个类处理单/多轨迹、Allegro/wuji/LEAP+Franka 三本体。所有行为开关由 trainer 以 kwargs 传入。**理解环境行为以此文件为准,别只看 cfg YAML 默认值。**
- **Trainer** `train_pool_2.py`(~250 argparse flag)—— `scripts/` 下所有脚本真正调用的入口(不是 `train.py`)。
- **RL** `learning/a2c_supervised*.py` —— rl_games + 行为克隆损失(specialist→generalist 蒸馏用)。`--supervised_loss_coef` 控 IL/RL 混合。

**两种动作空间(互斥)**:累积残差 `use_kinematics_bias_wdelta=True`(DexTrack 原始,wuji 全用这个)vs 相对位置 `use_relative_control=True`。wuji 脚本里 `useRelativeControl=True` 是历史遗留 flag,真正生效的是 `use_kinematics_bias_wdelta`。

---

## 3. ★ Reward 设计(迁移最该带走的核心)

### 3.1 配方代际(cubesmall 多序列 fair 实测,fair = 配置无关的纯跟踪指标)

| 代际 | 含义 | fair 末50均 / 峰 |
|---|---|---|
| `cgsmooth`(=cg_smooth) | 基线 + HAND_EMA 平滑,无接触引导 | **105 / 125**(最高!) |
| `cgsmooth_B2`(=cg_smooth_grabct2) | + 接触引导(B2 真值 flag) | 98 / 121 |
| `cgsmooth_softclip`(=cg_smooth_softclip) | + 软关节限位 | 89 / 111 |
| **`cgsmooth_softclip_B2`**(=cg_smooth_grabct2_softclip) | 接触 + softclip | **88 / 110(已收敛,推荐)** |
| `..._beta8_noidle` | 上 + 接触 beta=8(少吸附) | 74 / 97(还在训,同 iter 反而最快) |
| `..._beta30_hold025` | 上 + idle_hold 0.25(beta=30) | 38(慢,idle 拖累) |
| `..._beta8_idleXX` / `idlehold` | idle 一族 | **负(全崩)** |

**★★ 关键结论(迁移直接用)**:
1. **cubesmall 多序列 fair 在大多数配置上收敛到 ~80-105;接触/softclip/beta8/idle 这些改动对 fair 提升不大甚至略降。** 它们的价值在 **fair 测不到的地方**:抗抖、防反关节、少吸附、闲指自然。
2. **idle 一族明显伤 fair**(慢启动,见 3.4)。
3. **当前推荐多序列配置 = `cgsmooth_softclip_B2`**:fair 最高(88)、已收敛、配方最干净(接触引导 + softclip,无 idle/beta 旋钮)。用户判断"一点吸附没毛病",所以 beta8(减吸附)的卖点不成立。
4. **MuJoCo 迁移后这些大概率重新洗牌** —— softclip(防反关节)和 contact guidance 很多是补 PhysX 的;MuJoCo 硬限位 + 准接触下可能不需要。**迁移时先搭最朴素的 tracking reward(无这些补丁),看 MuJoCo 是否天然不犯反关节/抓得稳,再决定加哪些。**

### 3.2 ★ 推荐多序列配置 `cgsmooth_softclip_B2` 完整配方(从其实际 run 抓取)
即 `cg_smooth_grabct2_softclip`,= **pinall3 + 接触引导(beta=30)+ HAND_EMA + ACTION_RATE + softclip,无 idle、无 beta8**。和 3.3 的"最佳单序列配方"只差两点:**CONTACT_BETA=30(默认,不是 8)** 且 **无 IDLE_HOLD**。

env 开关:
```
# pinall3 reward 开关
RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0
# 接触引导(B2 真值 flag,beta=30 默认)
CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=30 CONTACT_SUBDIR=contact_grab2
# 平滑(cgsmooth = HAND_EMA + ACTION_RATE)+ 软限位(softclip)
HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5
# 无 IDLE_HOLD(idle 关)
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
```
关键 Hydra/cmdline flag(任务文件读这些,非 env):
```
task.env.numEnvs=40000  train.params.config.minibatch_size=40000  train.params.config.max_epochs=10000
task.env.use_kinematics_bias_wdelta=True   # 累积残差(wuji 全用这个)
task.env.observationType=pure_state_wref_wdelta   task.env.use_fingertips=True
task.env.hand_pose_guidance_glb_trans_coef=0.6  glb_rot_coef=0.1  fingerpose_coef=0.1
task.env.rew_finger_obj_dist_coef=0.3  rew_delta_hand_pose_coef=0.5  rew_obj_pose_coef=1.0
task.env.glb_trans_vel_scale=0.5  glb_rot_vel_scale=0.5  dofSpeedScale=20  rigid_obj_density=500  dt=0.0166
task.env.episodeLength=1000  num_frames=300  rew_smoothness_coef=0.0   # 内置 smoothness 关,平滑靠 HAND_EMA+ACTION_RATE
train.params.config.supervised_loss_coef=0.0   # 纯 RL(无蒸馏)
task.env.target_inst_tag_list_fn=../assets/inst_tag_list_obj_cubesmall_pinall3.npy   # 多序列轨迹集
```
实测 resolved 打印(来自该 run screen.log,供核对):`[REWARD SWITCH] relax_palm=True palm_grip_thres=0.22 palm_dist_rew_w=0.0 fix_finger5=True fingerpose_coef=0.1 finger_pos_rew=True finger_pos_coef=1.0 palm_pos_rew=True palm_pos_coef=1.0 glb_trans_coef=0.6 glb_rot_coef=0.1` · `[CONTACT_GUIDE] contact_guide=True coef=1.0 beta=30.0` + `contact source dir=...FPOS.../contact_grab2` · `[HAND_EMA] add_hand_targets_smooth=True coef=0.4` · `[ACTION_RATE] coef=0.0005` · 无 `[IDLE_HOLD]`(关)。
> ⚠️ SOFT_LIMIT_COEF=0.5 是按后续 B2_softclip 默认填的,该老 run 的 softclip 精确 coef 若要严格复现,查启动脚本 `scripts/run_tracking_headless_grab_multiple_wuji.sh`(SOFT_LIMIT 段)。

### 3.3 "最佳单序列配方"(pinall3,Isaac Gym 侧调到的)
全套 env 开关(供迁移时理解 reward 结构,不是逐字照搬):
```
RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0
CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=8 CONTACT_SUBDIR=contact_grab2
HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5
IDLE_HOLD_DOF=1 IDLE_HOLD_COEF=0.25 IDLE_HOLD_BETA=3.0
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
```
核心 reward 项(迁移要在 mjlab 里搭的):
- **跟踪**:手 qpos → 参考 qpos(delta_qpos)、物体位姿 → 参考物体位姿。
- **指-物距离** + **抓握 bonus / 举升**。
- **contact guidance**(接触指拉向真值接触点):`coef·exp(-beta·d)`,归一化除以接触指数。
- **HAND_EMA + ACTION_RATE**(抗抖,= "cgsmooth")。
- **SOFT_LIMIT**(软关节限位,防反关节,= "softclip";MuJoCo 硬限位下可能不需要)。
- **idle_hold**(闲指保持,见 3.4,可选)。

### 3.4 ★ idle_hold(闲指保持)+ **有界奖励通用教训**
**问题**:cube 是 thumb+index+middle 三指捏握,**ring/pinky 不接触、约束最弱 → 乱甩**。
**最终实现**(bounded exp reward):
```
reward += idle_hold_coef · Σ_{f=th..lf} (1 − contact_flag_f) · exp(−beta · ‖delta_qpos[该指4关节]‖₂)
```
- `(1−flag)` 只压闲指(接触指交给 contact guidance);delta_qpos = 达成DOF − 参考DOF;关节角(DOF)版比指尖版好(指尖欠定 4 关节)。
- **coef 扫描:0.25 是甜点**(fair 184.9 最高 + 闲指够紧);0.5 闲指最紧但慢启动严重;≥1 崩。
- **闲指收紧实测(multi 跨6序列均,度)**:beta30_hold025 ring5.8/pinky5.5(最紧,idle_hold 有效)> cgsmooth_softclip_B2 ring10.6/pinky12.2 ≈ beta8 ring13/pinky12。

**★★ 通用教训(迁移务必带走)**:**接触/姿态类引导项,用有界奖励 `exp(−β·d)`,绝不用无界距离惩罚 `‖d‖`/huber。** 无界惩罚早期偏差大(~1.7 rad)就远超主跟踪 reward(~0.3)→ 压垮学习(coef 0.25 都崩)。这条对 contact guide(本就 exp)、idle_hold(改成 exp 才work)、失败的 huber 实验都成立。

### 3.5 cg-beta(接触吸附)
contact guide = `coef·exp(-beta·d)`,接触点处拉力梯度 = beta。**beta=8 比 beta=30 柔(少吸附)**:接触指尖到接触点 beta8 2-3.8cm vs beta30 1-2.8cm。但用户最终判断"一点吸附没毛病",beta8 的减吸附价值有限。MuJoCo 准接触下吸附行为会变,重测。

### 3.6 ★ fair reward 指标(评估必读)
- **fair = 配置无关的纯跟踪奖励**:任务文件里对每步额外调一次 `compute_hand_reward_tracking`,用**固定系数**(0.6/0.1/0.1、flag@0.22),**不含** contact_guide / idle_hold / action_rate / softclip。这样不同 reward 配置可横向比"跟踪到底多准"。
- **读法**:本地 tensorboard `logs/<run>/.../summaries/events.out.tfevents.*`,tag `reward_fair/iter`(及 `rewards/iter` 原始 reward)。**必须** `EventAccumulator(f, size_guidance={'scalars':0})` 否则降采样。
- ⚠️ **别信 wandb**(API 截断/滞后 ~1240 epoch,据此画图结论会完全错)。memory: wandb-truncates-use-tensorboard。

---

## 4. ★ 数据布局(迁移直接复用)

- **参考轨迹(FPOS 版,当前用)**:`data/GRAB_Tracking_PK_WUJI_FPOS_v1/data/wuji_passive_active_info_<seq>_nf_300.npy`。含手 **q26**(`robot_delta_states_weights_np`,= 6 全局[3 平移+3 旋转] + 5 指×4 关节)+ 物体位姿 + **FPOS 指尖位置**(`ref_fingertip_pos`/`ref_palm_pos`,reward 开关 #4/#5 用)。
  - ⚠️ **必须 `WUJI_DATA_DIR=$FPOS`**,否则回退到非 FPOS 数据 → `ref_fingertip_pos/ref_palm_pos=None` → FINGER_POS/PALM_POS reward **静默 no-op**(flag 仍 print True)。memory: restart-config-diff-discipline。
- **contact 数据**:`data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact_grab2/`(B2 版),每序列 `pos (T,5,3)` + `flag (T,5)`,5 = [thumb,index,middle,ring,pinky]。
- **DOF 顺序(迁移对齐 MJCF 关键)**:q26 = 6 全局[3 trans + 3 rot] + finger1..5 × joint1..4。指关节块 [:,6:] 顺序 = [th,ff,mf,rf,lf]×4(ring=DOF[18:22],pinky=DOF[22:26])。
- **四元数约定**:物体 `object_rot_quat` 是 **xyzw**(Isaac Gym 侧确认)。
- **轨迹列表**:`assets/inst_tag_list_obj_<obj>.npy`(dict,key=traj tag)。已建:cup(31)、apple(27)、cubesmall(26,含 3 条 offhand)、cup_apple_cubesmall(84,合并)。⚠️ `.gitignore` 全局排 `*.npy`,不入 git。

### 4.1 contact guidance 来源(A/B/B2,memory: grab-contact-guidance-plan)
- **真值来源**:GRAB 原始 `GRAB/unzipped/grab/s*/<obj>_<action>.npz`,`contact['object']` = (帧×顶点) int8,值 = 碰该顶点的 SMPL-X 部位ID(0=无)。标签映射在 `GRAB/tools/utils.py:166 contact_ids`(Index41-43/Middle44-46/Pinky47-49/Ring50-52/Thumb53-55/掌22)。
- **A**(`generate_contact_guidance.py`):wuji 重定向指尖投影,继承重定向误差。
- **B**(质心):**有 bug**,质心飘到接触面中心(cube 大面差 2cm),反而落后 A。弃用。
- **B2**(`generate_contact_guidance_grab2.py` → `contact_grab2/`,**当前用**):真值 flag + 点取「人接触片里离 wuji 指尖最近顶点」≈A 点。**接触"点"用 A 投影就够,真值唯一价值在"flag(时机/哪根指)"**,只对 flute 类细长物体(A 几何阈值过检 30-40%)有用。
- ⚠️ **B2 mesh bug**:用仿真 mesh 取点时,bowl 仿真 mesh 被抽稀(顶点数≠GRAB ply)→ 越界/取错点。修复:改用 GRAB 原 mesh `GRAB/unzipped/tools/object_meshes/contact_meshes/{obj}.ply × 1.25`(同序同数 + assert)。
- SPIDER(andrew 的)确认用人手(MANO)接触逐指 flag + 可达点,与 B2 一致。

---

## 5. ★ 物理 / 物体发现(MuJoCo 迁移会验证的)

- **反关节超界**:PhysX 软限位 → 大拇指根关节等会超界。靠 SOFT_LIMIT(softclip)/CLIP_DOF hack 补。**MuJoCo 硬限位预期自然解决。**
- **抓握脆弱、flute 抓不住**:PhysX 接触不准。**MuJoCo 预期改善。**
- **★ 圆物体(apple)抓不起来**:深挖确认(用对的指标 —— 物体 z 高度时间曲线,**别用 max−start 会被末尾尖峰骗**):
  - apple multi specialist fair 卡 ~0(eat 8 条轨迹"举到嘴边"全程不抬物体、拖垮 24 条平均;lift/pass 能抬)。
  - **单序列 apple lift 能训出来**(fair 83,最终抬举 0.36m 贴合参考 0.38m)→ 说明 apple 可学,multi 失败是 **eat/多任务干扰**,不是手抓不住。
  - **eat 在 fly 手上没意义**(无嘴/身体参考),retarget 出来就难。建议 apple specialist 去 eat 留 lift+pass。
  - ⚠️ **kinematics_only 参考本来就不抬物体**(手按参考走但物体是自由物理,replay 抓不稳就留地上)——这是正常的,**别误判成"参考也没举"**。要看物体该有的轨迹用 `goal_pose_ref_np`(demo 目标,不带物理)。
- **MuJoCo 迁移的决定性验证**:在 MuJoCo 下跑单 cubesmall + apple lift,若抓握天然稳、反关节自动不犯、圆物体能抓 → 就是迁移值得的硬证据。

---

## 6. 多物体 generalist + specialist→蒸馏(DexTrack 真方法)

- **纯多任务 RL 学不好**(已验证):3obj 对照(cup+apple+cubesmall)fair 卡 -30;旧 10 物体 multiobj10 fair ~-7 平台。物体 latent 特征弱(256维,物体间 cosine 0.74-0.91)+ 冻结。
- **DexTrack 真方法(README 确认,未释出)= specialist→generalist 迭代蒸馏**:`a2c_supervised.py`(A2CSupervisedAgent extends A2CAgent = 同 PPO)。多 teacher 经 `teacher_index_to_weights`(每物体一个 specialist),BC loss = ‖student_mu − teacher_mu‖ + PPO loss,`supervised_loss_coef` 控权重。
- **当前计划**:训 cup/apple/cubesmall 单物体 specialist(纯 RL,supervised_loss_coef=0)→ 收敛后当 teacher 蒸馏进 3 物体 generalist,对比 3obj 纯 RL 对照。
- **迁移注意**:mjlab/rsl-rl 没有这套蒸馏,要重搭(或先只做单/多任务 RL,蒸馏后做)。

---

## 7. ★ 渲染管线 + 坑

- **playback** `wuji_isaacgym_playback.py`:`--src <rollout.npy> --env N --hand wuji --obj_code <seq> --gpu 0 --out x.mp4`。UP_AXIS_Z、gravity=0 纯运动学回放 rollout 的 `object_pose` + `shadow_hand_dof_pos`。
- **rollout npy 结构**(每 timestep 一个 dict,key=int 帧号):`shadow_hand_dof_pos`(达成 DOF,N×26)、`next_ref_np`(参考 DOF)、`goal_pose_ref_np`(参考物体位姿,N×7)、`object_pose`(达成物体,N×7)、`shadow_hand_dof_tars`、`actions`。**测量/可视化都从这里读。**
- **报告格式**:`report_videos/<config>_multi/cmp_<seq>_<success|fail>.{mp4,gif}`(Reference (kinematic) | Policy 并排)+ `train_reward_curve.png`;`<config>_single/`:`<name>_policy.mp4` + `ref_physics_vs_policy.mp4` + 曲线。生成脚本模板 `gen_report_b2softclip.sh`。
- ⚠️ **相机坑(踩了好几轮)**:
  - **展示抖动**:用**默认固定相机**(`--env 50`,不加 `--cam_follow`/`--cam_smooth`)。cam_follow hand 跟着手移动会把抖动/抬升**吸收掉**(看着不抖/没举)。memory: render-jitter-default-camera。
  - **抽帧时机**:抬举可能在轨迹末段(frame 250-295),抽 15/45/75% 会**错过**。先看高度曲线定时机再抽帧。
  - 报告对比用 `--cam_scale 0.5 --cam_follow hand --cam_smooth 21`(贴近手、稳,适合看抓握/手型,但不适合看绝对抬升)。
  - **判断抬没抬用物体 z 高度时间曲线**(matplotlib,不依赖 3D 相机),最可靠。中文字体用 `/nix/store/.../ghostscript-.../DroidSansFallback.ttf`。
- **allegro --ref 渲染**(wuji 不受影响,memory: allegro-ref-render-gotcha):需 DOF 块重排 + 关节限位 clamp,否则手型(大拇指)错。
- **kinematics_only 开环参考**(物体掉落对比):用真实 env `kinematics_only=True` 跑 test 生成 rollout 再渲染,别在 playback 手写 PD。⚠️ **多序列 test 脚本曾写死 `kinematics_only=False`**(L195,已改成尊重环境变量);单序列脚本 L69 `${kinematics_only:-False}` 一直正常。memory: multi-test-kinematics-only-hardcoded。

---

## 8. ★ 操作纪律(避免事故)

- **杀进程用 PID/pgid,不用宽 `pkill -f`**:train_pool_2.py 命令行极长含通用子串(`grab_train_test_setting=True` 含 `_test`、`tracking_ori_grab`、`cubesmall`),宽模式会误杀训练。**曾一秒误杀 5 个训练 run。** 杀前 `ps -eo pid,cmd|grep` 肉眼核对没有 train_pool_2.py 再杀。⚠️ 还有个坑:`pkill -f grab_gpu_apple.sh` 会撞上**自己这条命令行**(含该字样)自杀。memory: kill-pattern-matches-training。
- **train_pool 会 fork 子 `python train.py` 才是真正占 GPU 的**:杀训练要杀全(launcher + tee + train.py + 一堆 worker 子进程)。可靠做法:遍历 `/proc/*/environ` 按 `SCRIPT_STEM=` 反查全部 PID。
- **重启 run 要 diff 全量 config**,别从 `/proc/PID/environ` grep 子集重建(会漏 env var,如 WUJI_DATA_DIR → reward 静默 no-op)。memory: restart-config-diff-discipline。
- **setsid 启动训练**(脱离 harness 会话),`> /tmp/x.log 2>&1 < /dev/null &`。
- **抢卡防被占**:整张卡空出来的瞬间别人(tongxuan 的 auto-grabber)会抢;launch-before-kill 无空窗。但 grabber 会抢**任何**空卡,**做任何腾卡操作前先停 grabber**。
- **GPU 占用约 ~68GB / 40000-env 多序列**(单序列 22000 env ~38GB)。
- 训练速率慢:40000-env 多序列 ~50 秒/iter(~1.2 iter/min)。
- **doc 主动更新**(用户明确要求):每次启动实验/出结果**主动**更新实验记录 doc,别等提醒。腾 GPU **优先停 multi 不停 single**。memory: proactive-doc-updates。

---

## 9. 当前实验状态(2026-06-16 交接时,Isaac Gym 侧)

**正在跑(可能在你接手时已变,先 `nvidia-smi` + 遍历 environ 核对):**
- GPU7:`beta8_noidle` 多序列 cubesmall(iter ~1500,fair ~75 平台)。
- GPU5:`beta30_hold025` 多序列 cubesmall(iter ~1600,fair ~38,慢)。
- GPU4:`cup` specialist(beta8_noidle,40000)。
- GPU6:`apple` specialist(beta8_noidle,40000,fair 卡 ~0 因 eat)。
- GPU3:`3obj` generalist 对照(cup+apple+cubesmall 纯多任务 RL,fair ~-30)。
- 单序列 `apple_lift`(GPU0,已 ep1000,fair 83,**已验证 apple 可学**)。

**未决定的事**:
- 多序列最终配置:倾向 **cgsmooth_softclip_B2**(fair 88 已收敛、最简);用户想等 beta8_noidle/beta30_hold025 训到同 step(~2341)再公平对照,但预估**还要 9-12 小时(过夜)**,且 beta8 已平台 ~75 大概率追不上 88。
- apple specialist 去 eat 重训(留 lift+pass)未做。
- specialist→蒸馏 Phase 2 未起。

**对 mjlab 迁移的启示**:上面这些 reward 调参/specialist 之争**很多是在补 PhysX 的坑**。迁到 MuJoCo 后建议**从最朴素 tracking reward 重新开始**(跟踪 + 物体跟踪 + 指物距离 + 抓握/举升),先看 MuJoCo 物理下抓握/反关节/圆物体是否天然就好,再决定要不要加 contact guidance / softclip / idle_hold 这些补丁。**别盲目照搬 Isaac Gym 这套 reward 工程**——它的复杂度有相当部分是 PhysX 逼出来的。

---

## 10. 相关文件索引

- 迁移计划本体:`docs/mjlab_migration_plan.md`(最小原型清单 + 工作量估计)。
- 实验记录:`docs/cubesmall_single_multi.md`(reward 开关 + 物理调查 + §7.6 实验矩阵)。
- reward 调参:`docs/wuji_reward_tuning.md`(idle_hold + cg-beta)。
- 重定向 + 可视化:`docs/wuji_retargeting_and_visualization.md`。
- 多物体 generalist:`docs/wuji_multiobj_generalist.md`。
- contact guidance:`docs/grab_contact_guidance_plan.md`。
- wuji 集成:`docs/wuji_integration_plan.md`。
- 任务文件:`isaacgymenvs/tasks/allegro_hand_tracking_generalist.py`。
- 入口:`isaacgymenvs/train_pool_2.py` / `test_pool.py` / `test_generalist_pool.py`。
- 渲染:`isaacgymenvs/wuji_isaacgym_playback.py`、`gen_report_b2softclip.sh`。
- mjlab 模板:`wuji-mjlab/src/wuji_mjlab/tasks/reorient/`。
- 持久 memory:`~/.claude/projects/-home-liangh-DexTrack/memory/`(本文已汇总,但原文有更多细节)。
