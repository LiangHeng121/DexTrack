# 接入 wuji 手复现 cubesmall_inspect —— 详细计划

> 目标：把全新本体 **wuji 手（fly 悬浮手形态）** 接入 DexTrack，跑通
> `ori_grab_s2_cubesmall_inspect_1` 单序列跟踪任务，对齐 `docs/reproduction.md` 阶段 1 的复现标准。
>
> 决策已定：① 本体 = fly 悬浮手（手掌 + 6 全局 DOF，无机械臂）；② GRAB→wuji 重定向参考**从头生成**；③ 控制策略沿用 `grab_single`（累积残差 `use_kinematics_bias_wdelta=True`）。
>
> **更新方式**：完成一步勾选 `[x]`，遇坑记到末尾「踩坑记录」。

---

## 0. 现状盘点（已确认）

| 维度 | Allegro | LEAP | **wuji（待接入）** |
|---|---|---|---|
| 手指数 / 手指 DOF | 4 / 16 | 4 / 16 | **5 / 20**（finger1..5 × joint1..4） |
| 指尖 link | 4 | 4 | **5**（`fingerN_tip_link`） |
| fly 总 DOF | 22 | 22 | **26**（20 + 6 全局） |
| Isaac Gym fly URDF | ✅ | ✅ | ❌ 要造 |
| GRAB→手 重定向参考数据 | ✅ | ✅ | ❌ 没有 |
| 任务代码 `hand_type` 分支 | ✅ | ✅ | ❌ 要加 |

**wuji 原始资产**：`wuji/mujoco-sim/wuji_hand_description/`
- `urdf/right.urdf`（25 个 revolute 关节里 20 个是手指 DOF，5 个 `fingerN_tip_fixed` 是固定指尖）
- `mjcf/right.xml`、`meshes/right/*.STL`

**参考 npy 目标格式**（以 LEAP 为样板，`leap_passive_active_info_*_nf_300.npy` 是个 dict）：
- `object_transl` (T,3)、`object_rot_quat` (T,4) —— 物体参考轨迹（与手无关，可直接复用 GRAB 原始物体轨迹）
- `robot_delta_states_weights_np` (T, N) —— **手的逐帧关节状态**，N = 6 全局 + 手指 DOF（wuji = 26）
- `link_key_to_link_pos` —— 各 link 关键点位置（dict）

**关键路径参考**：
- 任务文件：`isaacgymenvs/tasks/allegro_hand_tracking_generalist.py`
- URDF 选择分支：约 `4615`(`shadow_hand_asset_file`)、`1867`/`4272`(link/body 名)
- fingertip body_names 分支：约 `1405-1416`
- 参考数据文件名/路径分支：约 `1049-1080`（`leap_passive_active_info_{obj}`）
- 重定向管线：`retargeting/`（`exp_runner_stage_1_grasp*.py` + `grab_data_utils_grasp*.py` + `manopth` + `assets/{allegro,leap}_mano_corres.npy`）

---

## 阶段 A —— 资产准备：wuji 做成 Isaac Gym 可加载的悬浮手 ✅ 已完成

- [x] **A1** 拷贝 `wuji/wuji-description/hand/body/{urdf,meshes}` → `assets/wuji_hand_description/`（用带 `right_` 前缀的官方 retargeting URDF，使手指 DOF 顺序与 B4 重定向输出一致）。26 个 STL。
- [x] **A2** mesh 路径：URDF 内 `../meshes/right/*.STL`，拷到 `assets/wuji_hand_description/{urdf,meshes}/` 后相对关系成立。
- [x] **A3** 造 fly URDF `assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf`（脚本 `wuji_pipeline/build_wuji_fly_urdf.py`）：把 6-DOF 浮动基链（`WRJ0x/y/z` prismatic + `WRJ0rx/ry/rz` revolute，照抄 allegro fly_v2 的 mass/limit/dynamics）注入到 `right_palm_link` 前，WRJ0rz child = `right_palm_link`；去掉 `<mujoco>` 标签。
- [x] **A4** Isaac Gym 加载验证（`wuji_pipeline/check_wuji_fly_isaacgym.py`）：**DOF=26**，顺序 `[WRJ0x,y,z,rx,ry,rz, right_finger1..5_joint1..4]`，5 个 `right_fingerN_tip_link` body 全在，palm 在，限位正确，无报错。
- [x] **A5 DOF/link 命名表（锁定）**：
  - DOF[0:6] 全局 = `WRJ0x,WRJ0y,WRJ0z`(平移,±1) `WRJ0rx,WRJ0ry,WRJ0rz`(旋转,±3.14)
  - DOF[6:26] 手指 = `right_finger{1..5}_joint{1..4}`（顺序同 B4 retargeting pinocchio 顺序）
  - 指尖 body = `right_finger{1..5}_tip_link`；palm body = `right_palm_link`

**→ B2 置换表当场解决（恒等）**：`isaac_dof[6:26] = retarget_qpos20[0:20]`（无需重排）；`isaac_dof[0:6] = 全局6-DOF`。浮动基照抄 allegro → 全局约定与 allegro 一致 → **B5 可直接复用 allegro 全局 6-DOF**（仅余 wuji/allegro palm 原点固定偏移待标定/由 RL 吸收）。

---

## 阶段 B —— 生成 wuji 的运动学参考（复用官方 wuji-retargeting，大幅简化）

> **重大简化**：官方仓库 `wuji/wuji-retargeting/` 自带手指重定向器 `Retargeter.retarget(kp21)->qpos20`
> （21 点 MediaPipe 关键点 → 20 维手指关节角，pinocchio FK + nlopt 优化，纯 CPU）。官方明确支持
> "把任意数据转成 21 点格式即可复用算法，无需改算法"。GRAB 用 MANO 手，MANO 关键点可直接重排成这 21 点。
> 所以**不再需要自建 wuji FK / mano_corres / 优化器**——把 GRAB 当成一个「自定义离线输入设备」即可。
> 模板用 `example/config/retarget_manus_right.yaml`（MANUS 手套 = 离线外部骨架，最贴近我们的场景）。

- [x] **B-pre（原始 GRAB MANO 序列）—— 已就绪**。已下载 `GRAB/dataset/grab__s2.zip` 等，用官方脚本解压：
  `cd GRAB && python grab/unzip_grab.py --grab-path dataset --extract-path ./unzipped`
  → `GRAB/unzipped/grab/s2/cubesmall_inspect_1.npz`（776 帧 @120fps）+ `GRAB/unzipped/tools/object_meshes/contact_meshes/cubesmall.ply`。
  **npz 关键字段**（`rhand` 是 dict）：
  - `rhand['params']['global_orient']` (776,3)、`['transl']` (776,3) —— 手腕全局位姿（→ B5 的全局 6-DOF）
  - `rhand['params']['fullpose']` (776,45) —— 15 关节完整 axis-angle，MANO 前向直接用这个（别用 PCA 的 `hand_pose`(24)）
  - `rhand['vtemp']` = `tools/subject_meshes/male/s2_rhand.ply` —— s2 个性化手型；**本地已有 `retargeting/assets/s2_rhand.ply`**
  - `object['params']['transl']`/`['global_orient']` (776,3)；`object_mesh: cubesmall.ply`
  ✅ 本地齐备：MANO 模型 `retargeting/manopth/mano/models/MANO_RIGHT.pkl`、vtemp `retargeting/assets/s2_rhand.ply`、物体轨迹（npz 或复用现成 allegro npy）。
  ⚠️ **唯一缺**：MANO 前向 Python 依赖 —— `manopth`(已 clone 在 `retargeting/manopth/`，需 `pip install -e`)、`chumpy`（MANO_RIGHT.pkl 是 chumpy pickle）；torch 已在 env。
- [x] **B0 env —— 已完成**。新建 conda env `wuji-retarget`（python 3.10），装 `pin(4.0.0) nlopt scipy pyyaml numpy`，`pip install -e wuji/wuji-retargeting --no-deps`（跳过只给手套用的 wuji-sdk）。两份 wuji URDF 已对比：`wuji/wuji-description/hand/body/urdf/right.urdf`（官方 retargeting 用，关节带 `right_` 前缀）与 `wuji/mujoco-sim/.../right.urdf`（无前缀）结构一致。
- [x] **B1 URDF 来源 —— 已完成**。`wuji_retargeting/wuji-description/` 子模块没 init，但有本地 clone。symlink 解决：`ln -sfn /home/liangh/DexTrack/wuji/wuji-description wuji/wuji-retargeting/wuji_retargeting/wuji-description`，`base.py:195` 的 `_PACKAGE_ROOT/wuji-description/hand/body/urdf/right.urdf` 即可解析。
- [x] **B4 逐帧重定向 —— 已完成**。脚本 `wuji_pipeline/retarget_keypoints_to_wuji.py`（跑在 wuji-retarget env）：21 关键点序列 → 官方 `Retargeter.retarget()`（warm-start + lp 滤波）→ `qpos(20)`。776 帧 7.8s，**限位违规 0、平滑（均值 dq 0.0034 rad）、无 NaN**。输出 `wuji_pipeline/out/s2_cubesmall_inspect_1_wuji_qpos20.npy` + `_dofnames.npy`。
- [x] **B2 锁定关节顺序 —— 已完成（恒等）**。retargeter 输出 `qpos(20)` 的 pinocchio 顺序（`_dofnames.npy`）= `right_finger1..5_joint1..4`，与阶段 A 的 Isaac Gym fly URDF 手指 DOF（索引 6-25）顺序**完全一致**（因 fly URDF 用了同一份带 `right_` 前缀的 URDF）。故 `isaac_dof[6:26]=retarget_qpos20[0:20]`，无需置换表。
- [x] **B3a MANO→21 关键点 —— 已完成并验证**。脚本 `wuji_pipeline/grab_to_mano_keypoints.py`：读 GRAB npz 的 `global_orient`+`fullpose`(45)+`transl`，manopth `ManoLayer(use_pca=False)` 前向直接得 **21 关节**（已含 5 指尖，无需手动取顶点），mm→m，加 transl。输出 `wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy` (776,21,3)。合理性校验通过：手 bbox 0.17m、骨长基本刚性（指尖 4% 属正常）、无 NaN、手腕 4.9mm/帧平滑。依赖已装：`chumpy`(已 patch numpy1.24 import)、`manopth`(editable)、`opencv-python-headless`。
  - ⚠️ 待 B7 视觉验证的近似：① `flat_hand_mean` 约定（默认 True，若与 GRAB 反了则手指整体偏 mean pose）；② betas=0 代替 vtemp `s2_rhand.ply` 个性化手型。
  - MANO 21 关节顺序：0 wrist；1-4 index；5-8 middle；9-12 pinky；13-16 ring；17-20 thumb。
- [x] **B3b 顺序确认 + 时序重采样 —— 已完成**。① 顺序：**manopth 输出的 21 关节顺序已是 MediaPipe 顺序**（manolayer.py:260 reorder 后 = 0 wrist，1-4 thumb，5-8 index，9-12 middle，13-16 ring，17-20 pinky），**无需手指重排**，B3a 输出直接喂 retargeter。② 重采样：776→300 在 B5/B8 用物体信号对齐实现（窗 GRAB[200:499]→300），不需复刻 DexTrack 离线抽帧逻辑。

> **B5/B6 策略（已定，绕开 DexTrack 未释出的离线裁剪）**：不复刻 DexTrack 的 start_idx/window/scale 离线预处理（README 明说部分未释出，反推脆弱）。改用三点：① 手指角坐标系无关（B4 已产出）；② 全局手腕轨迹与手型无关 → 复用现成 allegro/leap 参考的 object 轨迹 + 全局 6-DOF（已在正确 DexTrack 系、已 300 帧）；③ 用两边共享的「物体位姿」信号做时间对齐，得到 `300→776` 帧映射，把 wuji 手指角从 776 重采样到 300。
> 已定量发现：allegro ref 是 GRAB 的裁剪+重定心+缩放(~1.19)，物体运动窗 GRAB 244–536 ↔ ref 44–298，反推裁剪≈GRAB[193:538]→300（仅作 sanity，不直接用，最终用物体信号对齐）。

- [x] **B5 时间对齐 + 全局位姿 —— 已完成**。脚本 `wuji_pipeline/assemble_wuji_reference.py`：用 GRAB 原始物体角速度(776) 与 allegro ref 角速度(300) 对齐（坐标系无关），拟合裁剪窗 = **GRAB[200:499]→300**，**corr=1.0000 / norm-MSE=0**（完美，严格证明时间映射精确）。全局 6-DOF 复用 allegro ref 前 6 维。
- [x] **B6 组装 npy —— 已完成**。输出 `isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_ori_grab_s2_cubesmall_inspect_1_nf_300.npy`，3 字段（fly 格式，无需 link_key_to_link_pos）：`object_transl`/`object_rot_quat` 复用 allegro；`robot_delta_states_weights_np` (300,**26**) = [allegro 全局6] + [wuji 手指20，按精确 map 从776重采样，手指顺序恒等]。手指值在限位内。
- [x] **B7 校验（headless 数值）—— 已做，发现真问题**。脚本 `wuji_pipeline/check_wuji_reference_contact.py`（pinocchio FK fly URDF 算指尖→物体距离）。结果：复用 allegro 全局时，最近指尖离物体中心 **mean 0.080m**（cubesmall 半径~0.03，即离表面~5cm，**手指没够到**），主要 Y 偏 ~9cm。最优世界系常量平移只能降到 0.040m 且 max 0.104/std 0.016（**朝向也飘**）→ 必须重推全局，不能沿用 allegro。
- [x] **B8 重推全局 6-DOF —— 已完成并验证**。脚本 `wuji_pipeline/rederive_wuji_global.py`：① **Umeyama**（物体位置对应，不用朝向——DexTrack 物体 canonical 系不同会污染朝向法）恢复 GRAB→DexTrack 相似变换 s=1.2503、残差 **1.5mm**；② MANO 5 指尖→DexTrack 系=目标 Q；③ 每帧 Kabsch 对齐 wuji 指尖(FK 掌系)↔Q→全局 SE(3)→WRJ0 xyz+rxryrz(欧拉XYZ)；④ 重组装。**复验：最近指尖→物体中心 0.080→0.013m，抓握期 0.010m（落到物体上）✅**。参考文件已就绪可训练。
  - 备注：可选 Isaac Gym kinematics_only 渲染做一次肉眼确认（数值接触已强证据，非必须）。

**产出**：`wuji_passive_active_info_..._cubesmall_inspect_1_nf_300.npy` ✅ + 置换表(恒等) ✅ + 可视化确认（B7 待做）。

**剩余风险**：① 关节顺序（B2 已锁，恒等）✅；② MANO↔MediaPipe 重排（B3a 确认 manopth 已是 MediaPipe 序）✅；③ **全局手掌摆位是否让 wuji 手指够到物体（B7 待验证，是当前最大未知）**。重定向算法本身没碰。

---

## 阶段 C —— 任务代码接入 `hand_type == 'wuji'`

> 在 `tasks/allegro_hand_tracking_generalist.py` 里，凡是 `hand_type == 'allegro' / 'leap'` 的分支都要补 `'wuji'`。下面按代码区域列。

**好消息：obs/action 维度全是动态算的**（`num_obs` 用 `num_fingertips`/`nn_hand_dof`，`numObservations`/`numActions` 在 init 动态赋值，`num_shadow_hand_dofs` 读 URDF=26）→ 5 指 + 26 DOF 自动适配，rl_games 网络跟着变。`w_finger_pos_rew`/`hand_specific_randomizations` 默认 False → 4 指奖励/kinematics_chain 不走。真正写死的是名字 + `nn_hand_dof` + 关节重排。

`tasks/allegro_hand_tracking_generalist.py` 实改 8 处（均已完成 ✅）：
- [x] **C1** URDF 选择（~4615）：wuji → `wuji_hand_description/urdf/wuji_hand_right_fly.urdf`
- [x] **C2** fingertips（~1404）：wuji 5 个 `right_fingerN_tip_link`（`num_fingertips=len()` 自动=5）
- [x] **C3** hand_center / palm（~1426）：wuji → `right_palm_link`
- [x] **C4** body_names（~5090）：wuji 5 指 dict（palm+thumb..pinky）
- [x] **C5 `nn_hand_dof`（~1378，关键写死维度）**：wuji=26（allegro/leap=22）
- [x] **C6 关节重排 `joint_idxes_ordering`（~888，关键）**：wuji=恒等 `range(26)`（参考已是 sim DOF 顺序）；allegro 是 22 元素置换会把 26 截成 22
- [x] **C7** finger 奖励分支（~6018/6218）+ load_kinematics_chain（~1867）：加 wuji 分支防 raise（默认不走）

**另需改 2 个文件**（原计划漏了，已完成 ✅）：
- [x] **C8 `train_pool_2.py`**：① 轨迹枚举前缀（~1204）加 `wuji_`；② 参考文件名构造（~644）加 wuji 分支 `wuji_passive_active_info_{tag}.npy`。否则 `tot_tracking_data: []` / `raise ValueError`。
- [x] **C9 `learning/a2c_supervised.py`（~274）**：`nn_act_dims` wuji=26（gt_act buffer = nn_act_dims+1=27，匹配 env 给的 quat 版 27=3+4+20）。否则 experience buffer 23 vs 27 崩。

**产出**：✅ env 用 `hand_type=wuji` 构造、加载 wuji 参考、26-DOF reset、RL 端到端训练（冒烟 16 envs 跑到 epoch 24 无报错）。

**踩坑提醒（已记录到末尾）**：参考 npy 必须用 numpy 1.x 重存（pipeline 在 wuji-retarget env numpy 2.x 保存的 dict-npy，dextrack numpy 1.24 报 `No module named numpy._core`）。

---

## 阶段 D —— 脚本 + 冒烟测试

- [x] **D1 脚本 —— 已完成**。`scripts/run_tracking_headless_grab_single_wuji.sh`（复制 grab_single 改）：`hand_type=wuji`、`tracking_save_info_fn`/`tracking_data_sv_root`=`./data/GRAB_Tracking_PK_WUJI_v1/data`、`tracking_info_st_tag=wuji_passive_active_info_`；`numEnvs`/`minibatch_size` 改成 `${...:-22000}` 可覆盖（冒烟用小值）。SCRIPT_STEM 自动=`grab_single_wuji`，日志落 `logs/grab_single_wuji/`。
- [x] **D2/D3 冒烟 —— 已通过**。`numEnvs=16 bash scripts/run_tracking_headless_grab_single_wuji.sh 1 ori_grab_s2_cubesmall_inspect_1`：env 构建、`num_shadow_hand_dofs=26`、wuji 参考加载、RL 训练到 epoch 24 无维度/加载错。reward 早期负值（16 envs 噪声大，需正式 22000 envs 判断收敛）。
- [ ] **D4 正式训练**：空闲 GPU 上 `bash scripts/run_tracking_headless_grab_single_wuji.sh <gpu> ori_grab_s2_cubesmall_inspect_1`（默认 22000 envs），看 reward 是否从负往正爬。reward 不动→查参考/维度（回 B/C）。

**产出**：✅ `run_tracking_headless_grab_single_wuji.sh` + 冒烟通过；正式训练待跑。

---

## 阶段 E —— 对齐复现标准 & 收尾

- [ ] **E1** 正式训练到 reward 平稳，记录 best ckpt 路径与 reward。
- [x] **E2 测试脚本 —— 已建好并验证**。`scripts/run_tracking_headless_grab_single_wuji_test.sh`（复制 test 脚本改 4 个 wuji 数据路径变量；用法 `<gpu> ori_grab_s2_cubesmall_inspect_1 <CKPT> <HEADLESS>`）。它走的也是 train_pool_2.py（已含 wuji 修复），无需再改代码。实测用 best_ep_193 ckpt：加载成功、num_shadow_hand_dofs=26、跑完 298 帧、reward 4.29、保存 rollout 到 `logs_test/grab_single_wuji/`。（reward 低因 ckpt 早，脚本本身 OK。）
- [ ] **E3** 在 `docs/reproduction.md` 进度表加一行 wuji cubesmall，记录结果。
- [ ] **E4** 决定 wuji 资产/参考数据/脚本是否入 git（注意 `*.npy` 被 .gitignore 全局排除；mesh STL 体积）。

---

## 关键决策点 / 待办澄清

1. **wuji DOF 顺序**（阶段 A3）一旦定下，阶段 B 参考数据列顺序、阶段 C obs/action 维度必须全程一致 —— 这是最容易出错、最难 debug 的地方，建议用一张置换表锁死（B2）。
2. **阶段 B 已大幅简化**：复用官方 `wuji-retargeting` 的 `retarget(kp21)->qpos20`，不再自建优化器，只需写 GRAB→21点 适配器 + 全局位姿提取 + npy 组装。主要风险转移到「关节顺序置换」和「MANO↔MediaPipe 关键点重排」，都能用官方 tuning_tool 三层骨架快速验证。
3. 5 指手 vs DexTrack 处处假设 4 指 —— 阶段 C 的隐式维度假设要逐个揪出来（现在是整个项目剩余工作量的大头）。

---

## 踩坑记录

（开工后在此追加）
