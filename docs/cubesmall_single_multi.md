# cubesmall 单/多任务 专题

聚焦 `cubesmall` 物体的单序列 + 多任务（generalist）训练：现状、reward 拆解、下一步尝试。

> **⚠️ 重要更正（2026-06-06）**：本文档早期版本基于 `compute_hand_reward_tracking_twostages` 分析，但实际训练 `use_twostage_rew=False`，走的是 **`compute_hand_reward_tracking`（12731 行）**。两者结构不同。以下均已更正为**活跃函数**的真实情况。

## 1. 现状

### 单序列（`ori_grab_s2_cubesmall_inspect_1`）
| 本体/变体 | best | test | 视频 | 评价 |
|---|---|---|---|---|
| allegro | 219 | 216 | `cubesmall_allegro_policy.mp4` | ✅ |
| wuji offset | 181.93 | 176 | `cubesmall_wuji_policy_offset.mp4` | ✅ |
| wuji no-offset（默认）| 175.47 | 172 | `cubesmall_wuji_policy.mp4` | ✅ 100% 举起 |

### 多任务（26 条 cubesmall）
| 本体 | best | s2_inspect 单测 |
|---|---|---|
| allegro | 132 | 167 |
| wuji | 53 | -42 ⚠️ generalist 远不如 specialist |

### 26 条参考质量
参考忠实（指尖都够到立方体）。接触帧占比：`inspect`/`lift` 70–89%，`pass`/`offhand` 12–33%（松手动作难）。视频在 `render_videos/cubesmall_refs/`。

## 2. 当前 reward（活跃函数 `compute_hand_reward_tracking`，`goal_cond=False`, `train_free_hand=False`）

```
reward = -0.5·delta_value                          # ① 逐帧 手姿态跟踪（手腕 + 手指，已开启）
       - 0.3·(finger_dist + 2·palm_dist)           # ② 指尖/手掌 离物体远 的惩罚
       + goal_hand_rew                             # ③ 握住时 = -2·‖物体-目标位姿‖
       + bonus                                     # ④ 握住且很近 = 1/(1+10·goal_dist)
       # + hand_up  ← 在活跃函数里被注释，没有！
```
系数：`rew_delta_hand_pose_coef=0.5`、`rew_finger_obj_dist_coef=0.3`、`hand_pose_guidance_glb_trans/rot/fingerpose = 0.6/0.1/0.1`。

### 实测拆解（wuji no-offset，最佳 env，episodic~175）
| 项 | 每步均值 | 说明 |
|---|---|---|
| ④ **bonus 物体到位** | **+0.941** | **绝对主导** |
| ① delta_value 逐帧手姿态 | **-0.218** | 手腕+手指，**本来就在跟踪**（占 bonus 约 23%）|
| ② finger/palm 离物体惩罚 | -0.154 | |
| ③ goal_hand 物体跟踪 | -0.006 | goal_dist 均 3mm（物体跟得极准）|
| ⑤ hand_up | **无** | 活跃函数里没有；举起靠 ③④（参考物体轨迹本身上升到 0.4m）|

合计 ≈ +0.56/步 × 300 ≈ 169 ≈ 175。**bonus 主导，但逐帧手姿态跟踪(①)是个实打实的项(-0.218)。**

## 3. 每项具体怎么算的（通俗 + 公式，活跃函数 `compute_hand_reward_tracking`）

先认识几个"输入量"（每帧、每个 env 都有）：
- `object_pos` / `object_handle_pos`：方块**当前**在 sim 里的位置（xyz，从刚体状态读）。
- `target_pos`：方块**这一帧该在**的位置 = 人演示的物体轨迹（参考 npy 的 `object_transl[t]`，按 `progress_buf` 逐帧取）。
- `right_hand_pos`：手掌位置；`ff/mf/rf/th_pos`：食/中/无名/拇 4 个指尖位置——都是 `rigid_body_states[:, idx, 0:3]`，**Isaac Gym 直接给的当前世界坐标，不用 FK**。
- `delta_qpos`：= **当前手 26 维 qpos − 这一帧的参考手 qpos**（`self.delta_qpos = shadow_hand_dof_pos − cur_hand_qpos_ref`，逐帧）。前 6 = 全局（3平移+3旋转），6:26 = 20 个手指角。
- `flag`（是否握住）= `(4指尖到物体距离之和 ≤ 0.12×num_fingers)` **且** `(手掌到物体 ≤ right_hand_dist_thres)`，两条件都满足 → `flag==2`。

### ① delta_value（逐帧手姿态偏离，含手指）
```
# delta_qpos 是传进来的逐帧偏差，函数内不再覆盖
delta_hand_pos_value = ‖delta_qpos[:, 0:3]‖₁     # 全局平移 x,y,z 的绝对差之和
delta_hand_rot_value = ‖delta_qpos[:, 3:6]‖₁     # 全局旋转 3 维的绝对差之和
delta_qpos_value     = ‖delta_qpos[:, 6:26]‖₁    # 20 个手指角 每个|当前-参考|相加
delta_value = 0.6·delta_hand_pos_value + 0.1·delta_hand_rot_value + 0.1·delta_qpos_value
① = -0.5 · delta_value
```
**通俗**：当前手和**这一帧参考手**差多少——手腕位置（权 0.6）、手腕朝向（权 0.1）、20 个手指角（权 0.1）全部算，加权求和再 ×(-0.5)。**逐帧、含手指，和"像不像人"直接相关；本来就开着，只是手指那 0.1 偏小。** 实测约 -0.218/步。

### ② finger/palm 离物体惩罚
```
right_hand_dist        = ‖object_pos − right_hand_pos‖      # 手掌到方块中心（≥0.5m 截断到 0.5）
right_hand_finger_dist = Σ_{食,中,无名,拇} ‖object_pos − 指尖‖  # 4 指尖到方块中心 之和（≥0.6×num_fingers 截断）
② = -0.3 · (right_hand_finger_dist + 2·right_hand_dist)
```
**通俗**：量 4 个指尖、手掌到**方块中心的直线距离**，加起来（手掌权重翻倍），越远扣越多 → 逼手**贴到方块上**。只看"近不近物体"，不看姿势对不对。注意 finger_dist **只用了 4 指（漏小指）**，详见 §4.5。实测约 -0.154/步。

### ③ goal_hand（物体跟踪，只在握住时）
```
goal_dist = ‖target_pos − object_pos‖                       # 方块 实际 vs 这帧该在的位置
③ = 握住(flag==2) ? (-2·goal_dist) : 0
```
**通俗**：**只有手握住方块时**，才看"方块离它这一帧该在的位置差多少"，差越大扣越多；没握住 → 这项 0 分。这就是"按人的物体轨迹搬运"的约束（也包含了把方块抬到 0.4m——因为参考物体轨迹本身就上升）。实测约 -0.006/步（物体跟得极准，goal_dist 均 3mm）。

### ④ bonus（物体到位，主分来源）
```
④ = (握住 且 goal_dist ≤ 0.05m) ? 1/(1+10·goal_dist) : 0
```
**通俗**：握住 **且** 方块离目标 5cm 以内时，给一个**大正奖励**——完美贴合时≈1，稍微飘开就快速衰减。实测约 +0.94/步，是分数的大头。

### ⑤ hand_up —— **活跃函数里没有**（被注释）
物体能举起来不是靠 hand_up，而是靠 ③④：参考物体轨迹本身上升到 0.4m，策略跟踪它就把方块带上去了。

## 4. 诊断（回答两个"为什么"）

- **为什么手和人手差别大**：reward **有**逐帧手姿态跟踪(①)，但**手指系数只有 0.1**，相对 bonus(+0.94) 太弱，策略没动力把那 ~10°（0.18 rad）手指偏差压更小。→ 不是"没奖励"，是"奖励太弱"。
- **为什么多任务 cubesmall 学不好**：① 物体中心 reward，26 条各分 ~1500 env 数据稀；② `pass/offhand` 低接触序列信号差；③ wuji 5 指更难协调；④ 见下 #3 的 finger_dist bug 让抓握判定对 wuji 偏松。

## 4.5 finger_dist 对 wuji 算不对（真 bug）

`right_hand_finger_dist` = **拇+食+中+无名 4 个指尖**到物体距离之和（task 6413+ 行直接读 sim 当前指尖）。**第5指 pinky（right_finger5）被注释掉了**（6438 行）。但阈值用 `num_fingers=5`（grip flag 阈值 0.12×5=0.6、clamp 0.6×5=3.0）。
- 后果：**小指完全没算**；**抓握 flag 阈值(0.6,按5指)对着4指之和比 → 太松，容易误判"已握住"**（从而解锁 bonus）。
- allegro 4 指、num_fingers=4 一致，没问题；只有 wuji 漏了第5指又用了5的阈值。
- 修：把第5指加进 finger_dist（或阈值改成实际的4）。

## 5. 接下来尝试（已按更正调整）

1. **★ 调大手指姿态系数** —— ✅ **已实现，开关 `BOOST_FINGERPOSE=1`**（`hand_pose_guidance_fingerpose_coef` 0.1→0.5，可用 `FINGERPOSE_COEF` 调值）。直接强化已有的逐帧手指跟踪（×5），预期让手更像人。详见 §7 实现模块。
2. **palm/flag 适配长手指**（wuji 手指长、手掌该离物体远点）—— ✅ **已实现，开关 `RELAX_PALM=1`**（放宽抓握手掌阈值 0.12→0.22 + 去掉 palm 惩罚）。详见 [§7 实现模块](#7-实现模块reward-开关-2-relax_palm--3-fix_finger5)。
3. **修 finger_dist 第5指**（见 4.5）—— ✅ **已实现，开关 `FIX_FINGER5=1`**（finger_dist 补 wuji `pinky`，对齐 5 指阈值）。详见 [§7 实现模块](#7-实现模块reward-开关-2-relax_palm--3-fix_finger5)。
4. **指尖位置跟踪（任务空间）** —— ✅ **已实现，开关 `FINGER_POS_REW=1`**（+ `FINGER_POS_COEF` 默1.0）：活跃函数加附加项 `-coef·Σ_{5指}‖sim指尖 − 参考指尖‖`。关键不是"换 MANO 目标"，而是**惩罚算在指尖空间而非关节空间**。参考指尖原本 wuji 数据没有（只有 q26）→ 离线 FK 重定向 qpos 生成（见 §7.2/§7.5）。详见 §7 实现模块。

**#1/#2/#3/#4 均已实现并在跑**（见 §7.6）。下一步看四组实验收敛结果对比、择优组合。

## 6. 已知坑（分析层面）

- **改 reward 要改 `compute_hand_reward_tracking`（12716），不是 `_twostages`**（后者 `use_twostage_rew=True` 才用，当前 False 没用到）。我早期把开关加到了 twostages，全部无效 → 一定要按 §7 的方式 grep print 确认改动落在活跃函数。
- **bonus 主导**：reward 大头是 ④ bonus（物体到位），逐帧手姿态(①)虽在跟踪但系数弱 → "手不像人"是奖励太弱不是没奖励（见 §4）。

---

## 7. 实现模块：reward 开关 (#1 BOOST_FINGERPOSE / #2 RELAX_PALM / #3 FIX_FINGER5 / #4 FINGER_POS_REW)

> 本节自成一块，集中 #1/#2/#3/#4 的全部代码实现、设计、坑与验证。改动**未提交**；默认全关 = 原行为完全不变。四个开关互相独立，可单独或组合开。

### 7.1 开关一览（env 变量门控）

| 开关 | env | 作用 | 默认(关) | 开 |
|---|---|---|---|---|
| #1 | `BOOST_FINGERPOSE=1` | 调大逐帧手指姿态跟踪系数（×5，让手更像人） | `fingerpose_coef=0.1` | `0.5`（可用 `FINGERPOSE_COEF` 调值） |
| #2 | `RELAX_PALM=1` | 放宽抓握 flag 的手掌阈值 + 去掉 palm 距离惩罚 | `palm_grip_thres=0.12, palm_dist_rew_w=2.0` | `0.22, 0.0` |
| #3 | `FIX_FINGER5=1` | finger_dist 补第5指(wuji `pinky`=right_finger5) | `fix_finger5=False`（仅4指） | 含5指 |
| #4 | `FINGER_POS_REW=1` | 指尖空间跟踪附加项 `-coef·Σ_{5指}‖sim − 参考指尖‖` | `finger_pos_coef=0`（无此项） | `1.0`（`FINGER_POS_COEF`；需 FPOS 数据）|

### 7.2 改的位置（活跃函数 `compute_hand_reward_tracking` 12716-13070 + task）

- **#1**：__init__ 里 `BOOST_FINGERPOSE=1` 时覆盖 `self.hand_pose_guidance_fingerpose_coef`（0.1→`FINGERPOSE_COEF`，默 0.5）。该系数本就作为参数传进活跃函数（`delta_value = 0.6·glb_trans + 0.1·glb_rot + coef·fingerpose`），**无需改函数签名**。
- **#2/#3 签名**：活跃函数加 4 个**带默认值**参数 `palm_grip_thres=0.12, palm_dist_rew_w=2.0, fix_finger5=False, right_hand_lf_pos: Optional[Tensor]=None`。
- **#2**：3 处 `right_hand_dist <= 0.12`（12904/12956/13009）→ `<= palm_grip_thres`；13023 palm 惩罚 `2.0·right_hand_dist` → `palm_dist_rew_w·right_hand_dist`。
- **#3**：task 指尖提取处（原 `'little'` 注释块）→ `if 'pinky' in hand_body_idx_dict` 取 `self.right_hand_lf_pos` 否则 `None`；活跃函数 finger_dist 计算后、clamp 前加 `if fix_finger5: if lf is not None: finger_dist += ‖obj-lf‖`（5指之和对阈值 0.6×5 一致）。
- **#4（数据）**：wuji 参考 npy 原本只有 q26，无指尖位置 → 两步离线生成（不覆盖原数据）：① `wuji_pipeline/add_link_pos_to_reference.py`（wuji-retarget 环境）pinocchio FK q26 → palm+finger1..5 世界坐标，存纯数组 npz；② `wuji_pipeline/assemble_fpos_reference.py`（dextrack 环境）合并进原 npy 的 `link_key_to_link_pos`、用 numpy 1.24 重存到 `data/GRAB_Tracking_PK_WUJI_FPOS_v1/`。**坑**：numpy 2.x 存的 dict-pickle npy 在 1.24 加载报 `numpy._core`，故走纯数组 npz 中转。
- **#4（reward）**：活跃函数加 3 个**带默认值**参数 `finger_pos_rew=False, finger_pos_coef=0.0, ref_fingertip_pos: Optional[Tensor]=None`；在 reward 算完后、return 前加 `if finger_pos_rew: reward += -coef·Σ‖sim指尖−ref‖`（th/ff/mf/rf/lf 对 ref finger1/2/3/4/5）。task 在 compute_reward 里从 `self.tot_key_to_tot_link_pos`（数据有 `link_key_to_link_pos` 时由 4022-4025 自动填充）按 `env_inst_idxes`+逐帧建 `self.ref_fingertip_pos`(N×5×3)，传进 5965 调用点。**用新开关 `finger_pos_rew`，不碰 `w_finger_pos_rew`**（后者会在 5860 转去 rbpos 路径）。启动需 `WUJI_DATA_DIR` 指向 FPOS 目录。
- **__init__**：统一在一个开关块读 env → `self.boost_fingerpose/relax_palm/fix_finger5/finger_pos_rew/finger_pos_coef/palm_grip_thres/palm_dist_rew_w`，并 `print [REWARD SWITCH] ...`。
- **调用点**：仅活跃路径 5965 传 `self.*`（#2/#3 的 4 个 + #4 的 3 个新参数；#1 的 coef 走原有 `self.hand_pose_guidance_fingerpose_coef` 参数）。

### 7.3 低风险设计

- **带默认值参数** → 只有 5965 调用点传真值；5718（会条件指向 `_taco`）及其它调用点省略=默认=原行为不变，无需改它们。
- **按行号精确改**：`_taco`/`_rbpos`/`_twostages`/`_warm` 是近重复函数，`right_hand_dist <= 0.12`、palm 惩罚等字符串全文几十处 → 只能限定 12716-13070 按行号改，禁止全局 replace_all。
- **TorchScript 细化**：`Optional[Tensor]` 只在直接 `if x is not None:` 时细化，不能 `if a and x is not None:` → 用嵌套 if。

### 7.4 验证（务必做，吸取上次 inert 教训）

`py_compile` ✓ + 最小 JIT 冒烟测试 ✓（torch 2.4.1 支持 None 默认值 + 嵌套细化 + 多参数 print + `int(tensor[0])`）。**启动后 grep 训练日志，两条都出现且值对才算生效**（jit 内 `REWARD_ACTIVE` 已含 `fingerpose_coef=`，可一并确认 #1 提升值真传进活跃函数）：
1. `[REWARD SWITCH] relax_palm=... palm_grip_thres=... palm_dist_rew_w=... fix_finger5=... boost_fingerpose=... fingerpose_coef=...`（__init__）
2. `REWARD_ACTIVE compute_hand_reward_tracking palm_grip_thres= ... palm_dist_rew_w= ... fix_finger5= ... fingerpose_coef= ...`（**jit 内部，`progress_buf[0]==1` 触发**）
   - #1 跑应见 `boost_fingerpose=True fingerpose_coef=0.5`，且 `relax_palm=False ... fix_finger5=False`。
   - #2+#3 跑应见 `palm_grip_thres=0.22 palm_dist_rew_w=0.0 fix_finger5=True fingerpose_coef=0.1`。
   - #4 跑还要看 `REWARD_ACTIVE ... finger_pos_rew= True finger_pos_coef= 1.`，**且 task 端 `[FINGER_POS] ref_fingertip_pos=(N, 5, 3) tot_link_keys=[...finger1..5]`**（确认 FPOS 参考指尖真接进 reward，非 None）。已实测：单 (22000,5,3)、多 (40000,5,3)。

### 7.5 启动

```bash
# --- #1（BOOST_FINGERPOSE）单独跑：单 GPU2 / 多 GPU4 ---
BOOST_FINGERPOSE=1 SCRIPT_STEM=grab_single_wuji_fp1 \
  bash scripts/run_tracking_headless_grab_single_wuji.sh 2 ori_grab_s2_cubesmall_inspect_1
BOOST_FINGERPOSE=1 SCRIPT_STEM=grab_multiple_wuji_fp1 \
  bash scripts/run_tracking_headless_grab_multiple_wuji.sh 4 '' ../assets/inst_tag_list_obj_cubesmall_fp1.npy

# --- #2+#3（RELAX_PALM+FIX_FINGER5）合并跑：单 GPU6 / 多 GPU7 ---
RELAX_PALM=1 FIX_FINGER5=1 SCRIPT_STEM=grab_single_wuji_relax23 \
  bash scripts/run_tracking_headless_grab_single_wuji.sh 6 ori_grab_s2_cubesmall_inspect_1
RELAX_PALM=1 FIX_FINGER5=1 SCRIPT_STEM=grab_multiple_wuji_relax23 \
  bash scripts/run_tracking_headless_grab_multiple_wuji.sh 7 '' ../assets/inst_tag_list_obj_cubesmall_relax23.npy

# --- #4（FINGER_POS_REW）单独跑：单 GPU3 / 多 GPU5；需先生成 FPOS 数据 + WUJI_DATA_DIR 指向它 ---
# 数据：conda activate wuji-retarget && python wuji_pipeline/add_link_pos_to_reference.py
#       conda activate dextrack      && python wuji_pipeline/assemble_fpos_reference.py
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
FINGER_POS_REW=1 WUJI_DATA_DIR=$FPOS SCRIPT_STEM=grab_single_wuji_fp4 \
  bash scripts/run_tracking_headless_grab_single_wuji.sh 3 ori_grab_s2_cubesmall_inspect_1
FINGER_POS_REW=1 WUJI_DATA_DIR=$FPOS SCRIPT_STEM=grab_multiple_wuji_fp4 \
  bash scripts/run_tracking_headless_grab_multiple_wuji.sh 5 '' ../assets/inst_tag_list_obj_cubesmall_fp4.npy
```
> 多任务用改名的 tag 列表副本（`..._fp1.npy` / `..._relax23.npy` / `..._fp4.npy`）得唯一 RUN_MID；wandb 名启动后用 API 重命名为下表。

### 7.6 当前在跑（2026-06-06，wuji no-offset，未提交代码）

| 实验 | log 路径 | wandb 名 |
|---|---|---|
| #1 单 | `logs/grab_single_wuji_fp1/ori_grab_s2_cubesmall_inspect_1/20260606_101143` | `wuji_cubesmall_single_1` |
| #1 多 | `logs/grab_multiple_wuji_fp1/wuji_cubesmall_fp1/20260606_101143` | `wuji_cubesmall_multi_1` |
| #2+#3 单 | `logs/grab_single_wuji_relax23/ori_grab_s2_cubesmall_inspect_1/20260606_095846` | `wuji_cubesmall_single_2_3` |
| #2+#3 多 | `logs/grab_multiple_wuji_relax23/wuji_cubesmall_relax23/20260606_095846` | `wuji_cubesmall_multi_2_3` |
| #4 单 | `logs/grab_single_wuji_fp4/ori_grab_s2_cubesmall_inspect_1/20260606_104821` | `wuji_cubesmall_single_4` |
| #4 多 | `logs/grab_multiple_wuji_fp4/wuji_cubesmall_fp4/20260606_104821` | `wuji_cubesmall_multi_4` |

> 注：#4 启动抢占了会话前的两个旧 allegro 多任务（cubesmall ep4341 best137 @GPU3、combined ep4194 best161 @GPU5），均可从 `best_ep` resume。

---

相关：[systematic_training_plan.md](systematic_training_plan.md)、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。
