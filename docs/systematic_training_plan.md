# 系统化训练计划：单/多任务 × Allegro/Wuji（3×2 矩阵）

本文件定义当前阶段的训练任务矩阵与可视化目标。目标：对 **3 种任务设置** × **2 种本体**共 **6 个组合**各训练一个跟踪策略，并对每个都产出真实 Isaac Gym 渲染视频。

## 任务矩阵（进度截至 2026-06-04）

| 任务设置 | 序列 | Allegro 手 | Wuji 手 |
|---|---|---|---|
| **单序列 A** | `ori_grab_s2_cubesmall_inspect_1` | ✅ rew **219**（复现）+ 视频 | ✅ offset 版 rew **181.93**（ep957）+ 视频；no-offset 版 🏃 训练中（默认）|
| **单序列 B** | `ori_grab_s2_flute_pass_1` | 🏃 GPU1 ep635 rew44 | 🏃 GPU6（no-offset，真实手型）|
| **多任务（generalist）** | 见下「多任务范围」 | 🏃 cubesmall/flute/合并 3 个 | 🏃 GPU7 合并 44 条（real per-subject vtemp）|

> Wuji reward 还在慢慢调（当前主推 no-offset 版，见下「offset/no-offset」）。Allegro 侧三种任务都已铺开训练。

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
# $2=SUBJ_NM(留空) $3=tag列表.npy。空列表='' = 完整 s2..s10(~1269条,很重)
```
本轮用的 per-object 列表（`{tag:1}` 字典，0维 object 数组，在 `assets/`）：
| 列表 | 序列数 | 文件 |
|---|---|---|
| cubesmall | 26 | `assets/inst_tag_list_obj_cubesmall.npy` |
| flute | 18 | `assets/inst_tag_list_obj_flute.npy` |
| cubesmall+flute（合并跨物体）| 44 | `assets/inst_tag_list_obj_cubesmall_flute.npy` |
| duck（README 现成例子）| 23 | `assets/inst_tag_list_obj_duck.npy` |

> 列表生成：扫 `data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_*_<obj>_*.npy`，去前后缀得 tag，存成 `{tag:1}` 字典。
> **多任务 = 纯 RL generalist**：脚本默认 `supervised_loss_coef=0`（BC 蒸馏关闭，论文那套 specialist 预优化轨迹未释出、本地无），所以是干净的纯 RL 多轨迹跟踪。`max_epochs=10000`，比单序列(1000)久很多；每条轨迹平均只分到 `numEnvs/序列数` 个 env（如 40000/26≈1500），收敛慢。

### Wuji 单序列（cubesmall）
```bash
# 默认 = no-offset 版
bash scripts/run_tracking_headless_grab_single_wuji.sh <GPU> ori_grab_s2_cubesmall_inspect_1
# offset 版（加后缀指向独立目录）
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_OFFSET_v1/data SCRIPT_STEM=grab_single_wuji_offset \
    bash scripts/run_tracking_headless_grab_single_wuji.sh <GPU> ori_grab_s2_cubesmall_inspect_1
```

## offset / no-offset（默认 = no-offset）

wuji 参考有两版（区别仅最后一步「指尖外扩 offset」，详见 [wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)）。**默认无后缀 = no-offset；带 offset 的加 `_offset` 后缀**：

| | 默认 = no-offset | offset（保留）|
|---|---|---|
| 数据 | `data/GRAB_Tracking_PK_WUJI_v1/` | `data/GRAB_Tracking_PK_WUJI_OFFSET_v1/` |
| 日志 | `logs/grab_single_wuji/` | `logs/grab_single_wuji_offset/` |
| ckpt | `ckpts/wuji_cubesmall_inspect_best.pth` | `ckpts/wuji_cubesmall_inspect_offset_best.pth`（ep957 rew181）|
| 视频 | `render_videos/cubesmall_wuji_*.mp4` | `render_videos/cubesmall_wuji_*_offset.mp4` |

## 多任务范围（已定 = cubesmall + flute）

用 **per-object generalist**：分别训 cubesmall(26)、flute(18)，再加一个合并跨物体(44)。理由：和单任务同物体可比；对 wuji 可行（以后只需重定向这两个物体的几十条，而非完整 s2–s10 的上千条）。完整 generalist(s2–s10) 是论文级，allegro 可行但 wuji 不现实，暂不做。

## 可视化（真实 Isaac Gym 渲染）

渲染器 `isaacgymenvs/wuji_isaacgym_playback.py` 已**泛化**（`--hand allegro|wuji`，无头相机传感器，allegro 22DOF / wuji 26DOF 通用）。流程：跑 test → 拿 rollout → 渲染。三种视频（策略抓取 / 参考+物理 / 离线 mesh）的完整命令见 [wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。视频统一收在仓库根 `render_videos/`。

已产出（cubesmall）：`cubesmall_allegro_policy.mp4`、`cubesmall_wuji_policy_offset.mp4`、`cubesmall_{allegro,wuji}_reference_physics.mp4`、`cubesmall_wuji_reference_physics_offset.mp4`。待产出：no-offset wuji 策略、flute、多任务各序列。

## 当前状态（2026-06-04，**7 个训练并行 — 3×2 矩阵全格铺开**）

| GPU | 任务 | 序列数 | 进度 |
|---|---|---|---|
| 1 | allegro flute 单序列 | 1 | ep635/1000 rew44（阈值~100，上升中）|
| 2 | wuji cubesmall 单序列（**no-offset，默认**）| 1 | ep387/1000 rew166（offset 版终 181）|
| 3 | allegro cubesmall 多任务 | 26 | ep113/10000 |
| 4 | allegro flute 多任务 | 18 | ep104/10000 |
| 5 | allegro cubesmall+flute 合并 generalist | 44 | ep105/10000 |
| 6 | wuji flute 单序列（no-offset，真实手型）| 1 | ep153/1000 rew-51（早期，flute 难）|
| 7 | **wuji cubesmall+flute 合并 generalist** | 44 | ep5/10000 |

已完成：allegro cubesmall rew219、wuji cubesmall offset rew181（ep957，归档为 `_offset`）。

**wuji 多任务数据已就绪**：44 条 cubesmall+flute 序列全部重定向（no-offset，**真实 per-subject 手型** vtemp，s1–s10 全覆盖），用 `wuji_pipeline/batch_retarget_multitask.py` 批量生成。需要 GRAB `Subject Shape Templates`（male+female）提供各 subject 手型。

可视化：`render_videos/` 已有 cubesmall（policy+reference）、flute（reference），及 `reference_samples/`（抽样 wuji+allegro 参考回放）。渲染器 `wuji_isaacgym_playback.py` 支持 `--hand allegro|wuji` + `--ref`。

相关文档：[reproduction.md](reproduction.md)（allegro 复现）、[wuji_integration_plan.md](wuji_integration_plan.md)（wuji 接入）、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)（重定向+可视化操作）。
