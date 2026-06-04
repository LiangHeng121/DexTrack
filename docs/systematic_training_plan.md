# 系统化训练计划：单/多任务 × Allegro/Wuji（3×2 矩阵）

本文件定义当前阶段的训练任务矩阵与可视化目标。目标：对 **3 种任务设置** × **2 种本体**共 **6 个组合**各训练一个跟踪策略，并对每个都产出真实 Isaac Gym 渲染视频。

## 任务矩阵

| 任务设置 | 序列 | Allegro 手 | Wuji 手 |
|---|---|---|---|
| **单序列 A** | `ori_grab_s2_cubesmall_inspect_1` | ✅ 已复现 rew **219.65** | ✅ 已训 rew **181.93**（ep957，1000ep 跑完）|
| **单序列 B** | `ori_grab_s2_flute_pass_1` | 🏃 **训练中**（本轮启动）| ⏳ 待办（需先做 wuji 重定向参考）|
| **多任务（generalist）** | 见下「多任务范围」 | ⏳ 待训 | ⏳ 待办（需多序列 wuji 重定向）|

> Wuji 的 reward 还要继续慢慢调，单序列 B 和多任务的 wuji 版本等 reward 调好、重定向管线复用后再做。本轮先把 **Allegro 的剩余两个（flute 单序列 + 多任务）**训上。

## 本体与控制设置（保持一致才可比）

- **Allegro fly**：4 指 16 DOF + 6 全局 = 22 DOF。脚本 `run_tracking_headless_grab_single.sh`（累积残差）。
- **Wuji fly**：5 指 20 DOF + 6 全局 = 26 DOF。脚本 `run_tracking_headless_grab_single_wuji.sh`。
- 两者都用累积残差动作空间（`use_kinematics_bias_wdelta=True`），单序列 state-based。

## 各组合命令

### Allegro 单序列（本轮启动 flute）
```bash
cd isaacgymenvs && conda activate dextrack
# cubesmall（已复现）
bash scripts/run_tracking_headless_grab_single.sh <GPU> ori_grab_s2_cubesmall_inspect_1
# flute（本轮）
bash scripts/run_tracking_headless_grab_single.sh <GPU> ori_grab_s2_flute_pass_1
```
日志：`./logs/grab_single/<seq>/<时间戳>/`，ckpt 在该目录 `ckpt/best_ep_*.pth`。

### Allegro 多任务（generalist）
```bash
bash scripts/run_tracking_headless_grab_multiple.sh <GPU> <SUBJ_NM> <SEQ_TAG_LIST>
# SUBJ_NM 与 SEQ_TAG_LIST 都为空 = GRAB 训练集所有 subject s2..s10（完整 generalist，重）
# 或给 .npy 实例列表（如 ../assets/inst_tag_list_obj_duck.npy）限定子集
```

### Wuji 单序列（已有 cubesmall）
```bash
bash scripts/run_tracking_headless_grab_single_wuji.sh <GPU> ori_grab_s2_cubesmall_inspect_1
```

## 可视化（6 个都要）

统一用真实 Isaac Gym 相机渲染器 `isaacgymenvs/wuji_isaacgym_playback.py`（无头相机传感器，走 GPU Vulkan）：
1. 跑测试脚本得到 rollout（`logs_test/.../ts_to_hand_obj_obs_reset_1.npy`）；
2. `python wuji_isaacgym_playback.py --src <rollout> --env <好的env> --gpu <id> --out <mp4>`。

**待办**：当前 `wuji_isaacgym_playback.py` 写死了 wuji fly URDF + 26-DOF 顺序。要支持 Allegro，需把**手 URDF 路径 + DOF 顺序 + 指尖 link 名**参数化（Allegro fly URDF = `allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf`，22 DOF）。多任务可视化时每个被跟踪物体各渲一段。

## 多任务范围（待确认）

generalist 的序列集合需定，直接影响工作量（尤其 wuji 要逐序列重定向）。候选：
- **(a) 小集合**：cubesmall + flute + 少数几条（便于 wuji 也能重定向、3×2 严格可比）。
- **(b) 单物体多序列**：如所有 cubesmall 或所有 flute 序列。
- **(c) 完整 generalist**：GRAB s2..s10 全量（最接近论文，但 wuji 重定向成本极高）。

→ 默认建议 (a)，待用户确认后再定 wuji 侧。

## 当前状态（2026-06-03）

- Allegro cubesmall：✅ rew 219（复现完成）
- Allegro flute：🏃 本轮启动训练
- Allegro 多任务：⏳ 待启动（范围待定）
- Wuji cubesmall：✅ rew 181.93（ep957）；稳定 ckpt `isaacgymenvs/ckpts/wuji_cubesmall_inspect_best.pth`
- Wuji flute / 多任务：⏳ 等 wuji reward 调好 + 重定向复用

相关文档：[reproduction.md](reproduction.md)（allegro 复现细节）、[wuji_integration_plan.md](wuji_integration_plan.md)（wuji 接入与重定向管线）。
