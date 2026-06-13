# wuji 多物体 generalist —— 配置与流程

第一个 wuji（5 指悬浮手）多物体通才策略的物体/任务配置。
重定向/接触/可视化的流水线见 [wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。

## 配置（2026-06-12 定）

**10 物体 · 29 个(物体×任务) · 238 条轨迹**（GRAB，逐 subject/index 全收）。

| 物体 | 选中任务（轨迹数） | 小计 |
|---|---|---|
| alarmclock | lift(9) · pass(7) | 16 |
| apple | eat(9) · lift(8) · pass(10) | 27 |
| bowl | drink(17) · lift(5) · pass(7) | 29 |
| cup | drink(15) · lift(8) · pour(8) | 31 |
| duck | inspect(9) · lift(5) · pass(7) | 21 |
| elephant | inspect(9) · lift(5) · pass(9) | 23 |
| mouse | lift(7) · pass(9) · use(10) | 26 |
| phone | call(8) · lift(5) · pass(9) | 22 |
| train | lift(7) · pass(4) · play(9) | 20 |
| cubesmall | inspect(8) · pass(9) · lift(6) | 23 |
| **合计** | | **238** |

### 筛选决策
- **物体**：从两轮可视化里挑形状紧凑、好控制、重定向干净的（手指逐帧跳 ≤ ~13°，对标 cubesmall 基线）。**排除 gamecontroller**（整体去掉）。
- **任务**：**去掉所有 `offhand`**（物体静置、与抓取关系弱，雷同）；去掉 **alarmclock_see**、**cup_pass**。保留 lift + 各物体特色动作（eat/drink/pour/inspect/play/use/call/pass）。
- **加入 cubesmall**（已有大量单物体经验）的 inspect/pass/lift 作锚点物体。

### 物体形状多样性（泛化覆盖）
方块(cubesmall) · 球/圆胖(apple) · 容器(bowl/cup) · 紧凑玩具(duck/elephant) · 紧凑电子(mouse/phone/alarmclock) · 长条玩具(train)。
> 重定向质量备注：mouse 有一段 ~21°/帧的快速手指运动（人手真实操作，但重定向略放大），其余物体都在 cubesmall 基线内。详见重定向 doc §五 QC。

### tag 清单
`assets/inst_tag_list_obj_multiobj10.npy`（dict，keys=238 个 `ori_grab_..._nf_300` tag）。

---

## 准备 → 训练流程

1. **重定向** 全 238 条 → `data/GRAB_Tracking_PK_WUJI_v1/data/`
   `python wuji_pipeline/batch_retarget_multitask.py --list assets/inst_tag_list_obj_multiobj10.npy --out-dir isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data`（并行加速见下）
2. **FPOS** 补 link 位置 → `WUJI_FPOS_v1`
   `conda run -n wuji-retarget python wuji_pipeline/add_link_pos_to_reference.py`
   `conda run -n dextrack python wuji_pipeline/assemble_fpos_reference.py`
3. **接触(B2)** 逐 tag → `contact_grab2/`
   `for t in <238 tags>: python wuji_pipeline/generate_contact_guidance_grab2.py --traj $t --save`
4. **训练** wuji 多物体 generalist
   `bash scripts/run_tracking_headless_grab_multiple_wuji.sh <GPU> "" assets/inst_tag_list_obj_multiobj10.npy`
   配方沿用当前最优（pinall3 + cg_smooth，可加 softclip）；`CONTACT_SUBDIR=contact_grab2`、`WUJI_DATA_DIR=$FPOS`、`HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005`。

并行重定向（128 核）：切 N 份 chunk list 并发跑 `batch_retarget_multitask.py`，~238 条十几分钟可完成。

## 状态
- [x] 重定向 238（16 路并行单线程，~15 分钟，0 fail）
- [x] FPOS（link 位置合并，259 文件）
- [x] 接触 B2 238/238（见下 bowl 修复）
- [x] 起训练 GPU5（2026-06-12）：配方 **cg_smooth + softclip(coef0.5) + B2**，CONTACT_SUBDIR=contact_grab2，numEnvs=40000，wandb `wuji_pinall3_cgsmooth_softclip_B2_multiobj9`。停了 GPU5 baseline cg_smooth(ep4845)腾卡。

### ⚠ bowl 被自动排除 → 实际 9 物体 / 209 轨迹
训练 inst 来源 = DexTrack 优化轨迹字典 `grab_inst_tag_to_opt_res` ∩ 目标清单（task `allegro_hand_tracking_generalist.py`）。**bowl 的 29 条不在优化字典里**（其余 9 物体都在）→ 训练只收 209。bowl 同时还有 B2 mesh 抽稀问题（见上）——这个物体在数据集里就是有坑。**当前多物体 generalist = 9 物体（apple/duck/cup/alarmclock/mouse/phone/train/elephant/cubesmall）/ 209 轨迹。** 若要纳入 bowl 需补它的优化轨迹字典条目（DexTrack 同伦优化未释出，较难）。

### B2 mesh bug 修复（2026-06-12）
原 `generate_contact_guidance_grab2.py` 用**仿真 mesh** `assets/rsc/objs/meshes/{traj}.obj` 按 GRAB 接触索引取点，假设"仿真 mesh = GRAB ply×1.25 同顶点序"。**bowl 的仿真 mesh 被抽稀(25107≠GRAB 25119)** → 索引越界(9 条报错)或静默取错点(其余 bowl)。10 物体审计：只有 bowl 不一致。
**修复**：改用 **GRAB 原 mesh** `GRAB/unzipped/tools/object_meshes/contact_meshes/{obj}.ply × 1.25`（接触数组本就对它索引，同序同数，加 `assert verts==contact_verts`）。对未抽稀物体输出逐字节不变（cubesmall 点差 5.6e-9）。bowl 29 条已重生成。同理 B 版若用也需此修复。
