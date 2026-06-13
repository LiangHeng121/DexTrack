# Wuji 重定向 + 可视化 操作文档

本文件记录两件可复现的事：
1. **重定向**：把 GRAB 人手运动转成 wuji 手的参考轨迹（policy 跟踪的目标）。
2. **可视化**：在**真实 Isaac Gym** 里渲染参考/策略的抓取视频。

相关文档：[wuji_integration_plan.md](wuji_integration_plan.md)（接入分阶段计划）、[systematic_training_plan.md](systematic_training_plan.md)（3×2 训练矩阵）。

所有命令从仓库根目录 `/home/liangh/DexTrack/` 运行。涉及两个 conda 环境：
- **`dextrack`**（numpy 1.24, torch, isaacgym, manopth）—— 跑 MANO 前向、assemble、训练、测试、Isaac Gym 渲染。
- **`wuji-retarget`**（py3.10, numpy 2.x, pinocchio 4.0, nlopt）—— 跑官方 wuji 重定向、rederive、offset、离线 mesh 渲染。

> ⚠️ 跨环境陷阱：`wuji-retarget`(numpy2.x) 存的 dict-npy 在 `dextrack`(numpy1.24) 加载会报 `No module named numpy._core`。需要时在 dextrack 用 npz 中转重存。

---

## 一、重定向管线（GRAB → wuji 参考）

脚本都在 `wuji_pipeline/`。链路：**0 关键点 → 1 retarget → 2 assemble → 3 rederive → 4 offset**。其中只有第 1 步是优化（nlopt），其余是几何/重采样。

| 步 | 脚本 | 输入 → 输出 | env |
|---|---|---|---|
| **0** | `grab_to_mano_keypoints.py` | GRAB npz → 21 MediaPipe 关键点（manopth 前向，`--vtemp` 用受试者真实手型）| dextrack |
| **1 retarget** | `retarget_keypoints_to_wuji.py` | kp21 → 20 关节角（nlopt，**`--scale 1.25`**）| wuji-retarget |
| **2 assemble** | `assemble_wuji_reference.py` | qpos20 重采样到 300 帧 + 接全局手腕轨迹 → 参考 npy | dextrack |
| **3 rederive** | `rederive_wuji_global.py` | Kabsch 重算手腕 6-DOF，原地覆盖参考的 global | wuji-retarget |
| **4 offset** | `offset_wuji_fingertips.py` | IK 回缩手指（指腹外扩）→ `/tmp/wuji_offset_states.npy`，再并回参考 | wuji-retarget |

### 输出路径
- 第 0、1 步中间产物在 `wuji_pipeline/out/`：
  - `s2_cubesmall_inspect_1_mano_kp21.npy`（21 关键点）
  - `s2_cubesmall_inspect_1_wuji_qpos20.npy`（20 关节角）+ `..._wuji_qpos20_dofnames.npy`（列顺序图例：`right_finger1..5_joint1..4`，纯元数据，保证角度对号入座）
- 第 2、3、4 步都作用在**同一个最终参考文件**：
  `isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data/wuji_passive_active_info_<SEQ>_nf_300.npy`
  （2 创建 → 3 改 global → 4 换成回缩手指）

### 完整命令（以 cubesmall 为例）
```bash
cd /home/liangh/DexTrack
CT=/home/liangh/miniconda3/etc/profile.d/conda.sh

# 0  (dextrack)
source $CT; conda activate dextrack
python wuji_pipeline/grab_to_mano_keypoints.py \
    --npz GRAB/unzipped/grab/s2/cubesmall_inspect_1.npz \
    --vtemp retargeting/assets/s2_rhand.ply \
    --out wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy

# 1 retarget  (wuji-retarget) —— 唯一优化步, scale 1.25
conda activate wuji-retarget
python wuji_pipeline/retarget_keypoints_to_wuji.py \
    --kp wuji_pipeline/out/s2_cubesmall_inspect_1_mano_kp21.npy \
    --config wuji/wuji-retargeting/example/config/retarget_manus_right.yaml \
    --out wuji_pipeline/out/s2_cubesmall_inspect_1_wuji_qpos20.npy \
    --scale 1.25

# 2 assemble  (dextrack) —— 只重采样
conda activate dextrack
python wuji_pipeline/assemble_wuji_reference.py     # 默认 out = 参考 npy

# 3 rederive global  (wuji-retarget)
conda activate wuji-retarget
python wuji_pipeline/rederive_wuji_global.py

# 4 offset  (wuji-retarget) —— 仅 offset 版需要
python wuji_pipeline/offset_wuji_fingertips.py      # 之后把 /tmp/wuji_offset_states.npy 手指并回参考
```

### 两个关键处理（不要混淆）
- **scale 1.25**（第 1 步）：DexTrack sim 把物体放大了（GRAB 4cm → sim 5cm），人手关键点也必须放大 1.25× 才能让 wuji 抓握开口匹配 5cm 物体。**这是 wuji 抓得起来的根因**，缺了它手指会被挤进物体把它顶飞（reward 卡 ~20）。
- **fingertip offset / 指腹外扩**（第 4 步，~1.1cm）：重定向把 `tip_link` 原点（骨头）对到了物体表面，但真正接触的是指腹 mesh（在原点外 ~1.1cm）。offset 沿**指腹→指背**方向回缩手指，让指腹接触点落到表面、避免训练时指尖扎进物体。**可选**步骤。

### 一键批量（推荐，省去手动切 env）

`wuji_pipeline/batch_retarget_multitask.py` 把上面 0→3 步（不含 offset）按序列编排，**内部用 `conda run` 自动切 dextrack/wuji-retarget，从仓库根任何 env 运行即可**。逐条 ~73 秒，跳过已存在。

```bash
cd /home/liangh/DexTrack
# --list = 0维 ndarray 包 dict，keys=tag。可用现成清单或自建
python3 -c "import numpy as np; np.save('/tmp/l.npy', {'ori_grab_s2_apple_lift_nf_300':1})"
python wuji_pipeline/batch_retarget_multitask.py --list /tmp/l.npy \
    --out-dir isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data --scale 1.25
```
产物即 no-offset 参考（3 字段：`object_transl/object_rot_quat/robot_delta_states_weights_np`）。质量自检见脚本打印：限位违规应=0、`mean dq~0.003`、无 NaN、物体拟合残差 ~2mm。

### FPOS：补 `link_key_to_link_pos`（训练 + 接触生成必需）

重定向产物**缺指尖位置**。`finger_pos` 奖励与接触生成都要它。两步（SRC/DST 路径在脚本内硬编码：WUJI_v1 → WUJI_FPOS_v1，按目录批处理、跳过已存在）：
```bash
cd /home/liangh/DexTrack
conda run -n wuji-retarget python wuji_pipeline/add_link_pos_to_reference.py  # FK出palm+5指尖世界位置 -> wuji_pipeline/out/fpos_npz/
conda run -n dextrack       python wuji_pipeline/assemble_fpos_reference.py    # 合并 -> FPOS_v1/data 加 link_key_to_link_pos
```
> **FPOS ≠ offset**：FPOS 只在轨迹上**加一个 `link_key_to_link_pos` 字段**（三个基础字段与 WUJI_v1 逐字节相同，无任何 offset）；offset（上节）是单独把手指回缩 ~1.1cm 的变体。训练实际用 `WUJI_FPOS_v1`。

---

## 二、offset / no-offset 与命名约定

**默认（无后缀）= 不带 offset（no-offset）；带 offset 的加 `_offset` 后缀。**

| | 默认 = **no-offset** | **offset**（加后缀，保留）|
|---|---|---|
| 参考数据 | `data/GRAB_Tracking_PK_WUJI_v1/data/` | `data/GRAB_Tracking_PK_WUJI_OFFSET_v1/data/` |
| 训练日志 | `logs/grab_single_wuji/` | `logs/grab_single_wuji_offset/` |
| 稳定 ckpt | `ckpts/wuji_cubesmall_inspect_best.pth` | `ckpts/wuji_cubesmall_inspect_offset_best.pth`（ep957 rew181）|
| 视频 | `render_videos/cubesmall_wuji_*.mp4` | `render_videos/cubesmall_wuji_*_offset.mp4` |

- **no-offset 参考** = scale1.25 + 跳过第 4 步。手指比 offset 版多弯 ~0.115 rad，global 完全相同（offset 不改 global）。
- 重建 no-offset 参考（不重 retarget）：重跑第 2 步 assemble 拿 scale-only 手指，配上当前参考的 global：
  ```bash
  conda activate dextrack
  python wuji_pipeline/assemble_wuji_reference.py --out /tmp/assemble_scaleonly.npy
  # 然后: no_offset = {当前参考的 object + global6, assemble 的 fingers20}
  ```

### 训练脚本怎么切换 offset/no-offset
`scripts/run_tracking_headless_grab_single_wuji.sh` 支持环境变量覆盖（用独立变量名 `WUJI_DATA_DIR` 避开脚本第 143 行的 cephfs 硬覆盖；`SCRIPT_STEM` 改了第 3 行 `:-` 尊重 env）：
```bash
cd isaacgymenvs && conda activate dextrack
# 默认 = no-offset
bash scripts/run_tracking_headless_grab_single_wuji.sh <GPU> ori_grab_s2_cubesmall_inspect_1
# offset 版
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_OFFSET_v1/data SCRIPT_STEM=grab_single_wuji_offset \
    bash scripts/run_tracking_headless_grab_single_wuji.sh <GPU> ori_grab_s2_cubesmall_inspect_1
```

---

## 二·补、接触引导生成（contact / contact_grab / contact_grab2 = A / B / B2）

三个脚本在 `isaacgymenvs/wuji_pipeline/`，**都从 `isaacgymenvs/` 运行、dextrack env、逐 traj、加 `--save`**。
输入要 FPOS 参考（读 `link_key_to_link_pos`）+ 物体 mesh（`../assets/rsc/objs/meshes/{traj}.obj`）。三版**只是接触点/flag 来源不同**，奖励端代码一致（`exp(-beta·d)`，beta=30）。来龙去脉见 [grab_contact_guidance_plan.md](grab_contact_guidance_plan.md)。

```bash
cd /home/liangh/DexTrack/isaacgymenvs && conda activate dextrack
T=ori_grab_s2_cubesmall_inspect_1
python wuji_pipeline/generate_contact_guidance.py       --traj $T --save   # A -> contact/
python wuji_pipeline/generate_contact_guidance_grab.py  --traj $T --save   # B -> contact_grab/   ⚠有bug
python wuji_pipeline/generate_contact_guidance_grab2.py --traj $T --save   # B2-> contact_grab2/  (真值,推荐)
```

| 版本 | 脚本 | 接触点 | flag | 说明 |
|---|---|---|---|---|
| **A**（默认） | `generate_contact_guidance.py` | 重定向 wuji 指尖**投影到物体面**最近点 | 几何阈值 `dist<1.2cm` | 纯几何、零标注；继承重定向误差；flute 类细长物体过检 30-40% |
| **B** | `generate_contact_guidance_grab.py` | GRAB 真值接触片**质心** | GRAB 真值 | ⚠ **有 bug**（宽接触面质心飘 ~2cm），勿用 |
| **B2**（推荐真值版） | `generate_contact_guidance_grab2.py` | GRAB 真值片内**离 wuji 指尖最近顶点**（≈A 点） | GRAB 真值 | 修 B 的质心 artifact；真值唯一价值=修 flute 类过检 flag |

- 输出 `{out_dir}/{tag}_contact.npy`：`contact_flag(300,5)`、`contact_pos_local(300,5,3)`（物体局部系，RL 用活物体位姿变换回世界系）。列序 `[拇,食,中,无名,小]`。
- B/B2 额外用到 GRAB 真值接触（`GRAB/unzipped/grab/`）+ allegro 原参考做帧对齐（与重定向 assemble 同一套角速度匹配，`grab_idx[i]=round(linspace(c0,c1-1,300))`）+ 仿真 mesh（= GRAB ply×1.25 同顶点序）。
- 默认 `out_dir` 都在 `data/GRAB_Tracking_PK_WUJI_FPOS_v1/<subdir>`，可 `--out_dir` 覆盖；A 用 `--data_dir/--mesh_dir/--thresh`，B/B2 用 `--grab_root/--allegro_ref_dir/--sim_mesh_dir`。

**训练侧选哪版** —— 环境变量 `CONTACT_SUBDIR`（task `allegro_hand_tracking_generalist.py:4103`，从 `tracking_save_info_fn` 同级找接触目录）：
```bash
CONTACT_SUBDIR=contact        # A（默认）
CONTACT_SUBDIR=contact_grab   # B
CONTACT_SUBDIR=contact_grab2  # B2
```

> **新物体端到端**：① `batch_retarget_multitask.py` 重定向 → ② FPOS 两步 → ③ `generate_contact_guidance_grab2.py --traj <tag> --save` → ④ 渲染检查（下节）→ ⑤ 把 tag 加进 `assets/inst_tag_list_obj_*.npy` 训练。

---

## 三、可视化（真实 Isaac Gym 渲染）

核心脚本：`isaacgymenvs/wuji_isaacgym_playback.py`。它在**真实 Isaac Gym** 里加载真 URDF mesh + 物体 mesh + 地面 + 光照，逐帧把手 DOF / 物体位姿设成已存的 rollout 状态，用**无头相机传感器**（GPU Vulkan，不需要 X/xvfb）渲染 → MP4。两种本体通用。

> 为什么用相机传感器而不是 train.py 的 `capture_video`：后者靠 xvfb 虚拟屏 + viewer，本机无 sudo、软件 GL 跑 Isaac Gym viewer 不可靠。相机传感器才是官方无头渲染。

### 渲染器参数
```bash
conda activate dextrack
pip install imageio-ffmpeg     # 仅首次, 写 MP4 需要
python wuji_isaacgym_playback.py \
    --src <rollout.npy>           # test 存的 ts_to_hand_obj_obs_reset_*.npy；或参考 npy(配 --ref)
    --hand wuji|allegro           # 选手 URDF（默认 wuji）
    --env <idx>                   # rollout 里渲哪个 env（默认 1）
    --ref                         # src 是参考 npy 而非 rollout
    --gpu <id> --out <mp4>
```
关键点：`shadow_hand_dof_pos` 是 sim/asset DOF 顺序，渲染器直接 `ds["pos"][:]=hand[t]`，allegro(22)/wuji(26) 通用，无需重排。

可选相机参数：`--cam_scale <1拉近>` `--cam_follow {mid,hand}` `--cam_smooth <帧窗>`；接触叠加 `--contact <B版npy>`(实心○) `--contact_proj <A版npy>`(空心◇)。
> ⚠ **判断"策略抖不抖"必须用默认相机**（不传任何 cam 参数）。`--cam_follow hand` 会跟随手腕抵消整体运动、`--cam_smooth` 平滑相机，两者都会把手的抖动一起吸收掉，看着就不抖了。逐帧像素 diff 也会被远近/框选 confound，判抖以眼睛看默认相机视频为准。

### 三种视频怎么生成

**A. 策略抓取**（policy 学到的真实抓取）
```bash
cd isaacgymenvs && conda activate dextrack
# 1) 跑策略测试 (kinematics_only=False, 策略真控制)
numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh \
    <GPU> ori_grab_s2_cubesmall_inspect_1 $PWD/ckpts/wuji_cubesmall_inspect_best.pth True
# 2) 选举得最高/跟踪最好的 env, 渲染
python wuji_isaacgym_playback.py --src logs_test/.../ts_to_hand_obj_obs_reset_1.npy \
    --env <best> --hand wuji --gpu <id> --out render_videos/cubesmall_wuji_policy.mp4
```
> ckpt 训练时会删旧 best，测试前先 `cp` 到稳定路径再用绝对路径传入。

**B. 参考轨迹 + 物理**（光放参考、无主动捏合，物体走真实物理）
```bash
# kinematics_only=True: 手按参考运动, 物体自由物理
kinematics_only=True numEnvs=100 bash scripts/run_tracking_headless_grab_single_wuji_test.sh \
    <GPU> ori_grab_s2_cubesmall_inspect_1 $PWD/ckpts/wuji_cubesmall_inspect_best.pth True
python wuji_isaacgym_playback.py --src logs_test/.../ts_to_hand_obj_obs_reset_1.npy \
    --env <idx> --hand wuji --gpu <id> --out render_videos/cubesmall_wuji_reference_physics.mp4
```

**C. allegro 同理**，`--hand allegro` + allegro 的 test 脚本 `run_tracking_headless_grab_single_test.sh`（已支持 `kinematics_only`、`numEnvs` 覆盖）。

### 选 env 的判据
分析 rollout 里每个 env 的 `object_pose` z 轨迹：策略版选「物体跟踪误差最小」或「举得最高」的 env；kinematics_only 版选「峰值 z 接近中位数」的代表 env（避开 env0 的 reset glitch）。

### 视频成品
都收在仓库根 `render_videos/`（见上「命名约定」表）。健康检查：随便取一帧，非黑像素应 ~73%（720×720，手+物体+地面+天空都在）；纯黑说明相机没渲到。

### 离线 mesh 渲染（备选，不需要 Isaac Gym）
`wuji_pipeline/render_mesh_video.py`（wuji-retarget env，pyrender+EGL，真实 STL mesh）+ `render_reference_video.py`/`render_rollout_video.py`（matplotlib 骨架）。这些用 pinocchio FK 离线重建，不如 Isaac Gym 渲染真实，留作备选。

---

## 四、当前已产出（cubesmall）

| 视频 | 本体/版本 | 结果 |
|---|---|---|
| `cubesmall_allegro_policy.mp4` | allegro 策略 (rew216) | 100% 举起 0.41m |
| `cubesmall_wuji_policy_offset.mp4` | wuji offset 策略 (ep957 rew181) | 100% 举起 0.41m |
| `cubesmall_wuji_policy.mp4` | wuji no-offset 策略 | ⏳ 训练中(GPU2) |
| `cubesmall_{allegro,wuji}_reference_physics.mp4` | 参考+物理 (no-offset) | 物体留地(无主动捏合) |
| `cubesmall_wuji_reference_physics_offset.mp4` | 参考+物理 (offset) | 物体留地 |

---

## 五、重定向质量 QC（"一跳一跳"的判定）

逐帧 IK 多解，**快速交互瞬间可能放大手指运动** → 渲染看着一跳一跳。判定基线 = 训练成功的 cubesmall：
手指逐帧跳（20 关节角度差之和）`max 6.5–9.7°`、中位 `2–6°`；全局 rot `max ~10°`。

诊断（逐帧分解到 物体/全局6DOF/手指20DOF，并与**同轨迹 allegro 原参考**比对时间剖面）：
- **中位 ≈ cubesmall + 与 allegro 快帧重合** → 真实人手快动作，正常。
- **某段连续 `max ≫ 15°`**（如 stamp 32°、mouse 21°，而 allegro 同帧仅 ~12°）→ 重定向在该段**过放大 ~2.5×**（小物体、手指挤、IK 解摆动），多见于 stamp/mouse 这类需精细手指操作的小物体。
- 四元数符号翻转计数高（如 phone 34 次）= 物体定向不稳（近对称物体），渲染不受影响但属信号。

**建议**：选物体优先 `max 手指跳 ≤ ~13°` 的紧凑好抓物体（cube/sphere/apple/duck/elephant/gamecontroller…）；小而 fiddly 的（stamp/mouse）避开，或加时间平滑/限速后处理削峰。
诊断脚本思路：对每个 `wuji_passive_active_info_*.npy` 取 `robot_delta_states_weights_np`，`np.degrees(np.abs(np.diff(rs[:,6:],axis=0)).sum(1))` 看 max/中位/坏帧数。
