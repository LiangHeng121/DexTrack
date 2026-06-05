# 系统化训练计划：单/多任务 × Allegro/Wuji（3×2 矩阵）

本文件定义当前阶段的训练任务矩阵与可视化目标。目标：对 **3 种任务设置** × **2 种本体**共 **6 个组合**各训练一个跟踪策略，并对每个都产出真实 Isaac Gym 渲染视频。

## 任务矩阵（进度截至 2026-06-04）

| 任务设置 | 序列 | Allegro 手 | Wuji 手 |
|---|---|---|---|
| **单序列 A** | `ori_grab_s2_cubesmall_inspect_1` | ✅ rew **219** + 视频 | ✅ offset **181.93** + 视频；✅ **no-offset 175.47**（默认）+ 视频 |
| **单序列 B** | `ori_grab_s2_flute_pass_1` | ✅ rew **43.55** + 视频（flute 难）| ❌ no-offset **-42** + offset **-29** 都失败（flute 对 wuji 太难）|
| **多任务（generalist）** | cubesmall(26)/flute(18)/合并(44) | 🏃 3 个训练中(best 133/165/160) | 🏃 3 个训练中(best 53/-31/-28，flute 拖累) |

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

## 已完成（单序列）

| 任务 | best | test | 视频 | 备注 |
|---|---|---|---|---|
| allegro cubesmall | 219 | 216 | `cubesmall_allegro_policy.mp4` | ✅ |
| allegro flute | 43.55 | 39 | `flute_allegro_policy.mp4` | 弱（阈值~100 未达）|
| wuji cubesmall offset | 181.93 | 176 | `cubesmall_wuji_policy_offset.mp4` | ✅ |
| wuji cubesmall **no-offset**（默认）| 175.47 | 172 | `cubesmall_wuji_policy.mp4` | ✅ 100% 举起 |
| wuji flute no-offset | **-42** | -46 | `flute_wuji_policy.mp4` | ❌ 失败（0% 举起）|
| wuji flute offset | **-29.46** | — | — | ❌ 失败（offset 也没救活）|

ckpt：`ckpts/{s2_cubesmall_inspect_ckpt, allegro_flute_pass_best, wuji_cubesmall_inspect_best(no-offset默认), wuji_cubesmall_inspect_offset_best}.pth`。
**结论：flute_pass 对 wuji 太难**（细杆 2cm，pass 动作只 43% 帧接触）——no-offset/offset 两版单序列都训不起来（rew 负）。allegro flute 也只 43（阈值未达），flute 本身难。

## 训练中（2026-06-04，多任务 6 个并行，~ep2300–3300/10000）

| wandb name | 任务 | 序列 | epoch | best(mean rew) |
|---|---|---|---|---|
| `allegro_cubesmall_multi` | allegro cube | 26 | 3317 | **132.81** |
| `allegro_flute_multi` | allegro flute | 18 | 3006 | **165.24** |
| `allegro_combined_multi` | allegro 合并 | 44 | 3208 | **159.88** |
| `wuji_cubesmall_multi` | wuji cube | 26 | 2678 | 53.38 |
| `wuji_flute_multi` | wuji flute | 18 | 2330 | -30.76 |
| `wuji_combined_multi` | wuji 合并 | 44 | 2788 | -28.41 |

- wandb 名已全唯一（`<hand>_<obj>_multi`），不靠 group 区分（API 改名 + 脚本 log_path 中段改 `<hand>_<obj>`）。
- **观察**：allegro 三个多任务都健康（132–165）；wuji cubesmall 多任务正（53），但 **wuji flute / 合并仍负**（被 flute 难度拖累，和单序列一致）。
- GPU6 空闲（wuji flute offset 单训完）。

**wuji 多任务数据**：44 条全部重定向（no-offset，**真实 per-subject 手型** vtemp，s1–s10），`wuji_pipeline/batch_retarget_multitask.py` 批量生成；手型来自 GRAB `Subject Shape Templates`（male+female）。

**稳定性教训**（多任务 ckpt 581MB/个）：磁盘满会崩进程 + 写出**损坏 ckpt**（截断）+ 卡死幸存进程 → `save_frequency` 已 200→500；resume 优先用完整的 `best_ep`（损坏的总是正在写的 `last_ep`）；并发 resume 要错开时间戳/SCRIPT_STEM 防 run-dir 撞车。看门狗 `watchdog_trainings.sh` 轮询 epoch 停滞报警。

可视化：`render_videos/` 有 cubesmall/flute 的 policy + reference_physics，及 `reference_samples/`（抽样真实物理参考，物体留地=参考无主动捏合）。渲染器 `wuji_isaacgym_playback.py` 支持 `--hand allegro|wuji`、`--ref`(纯回放,物体粘参考)、默认(读 rollout,真实物理)。

相关文档：[reproduction.md](reproduction.md)（allegro 复现）、[wuji_integration_plan.md](wuji_integration_plan.md)（wuji 接入）、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)（重定向+可视化操作）。
