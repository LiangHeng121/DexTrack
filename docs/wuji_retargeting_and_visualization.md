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
