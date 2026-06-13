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

## 7. 实现模块：reward 开关 (#1 BOOST_FINGERPOSE / #2 RELAX_PALM / #3 FIX_FINGER5 / #4 FINGER_POS_REW / #5 PALM_POS_REW)

> 本节自成一块，集中所有 reward 开关的代码实现、设计、坑与验证。改动**未提交**；默认全关 = 原行为完全不变。开关互相独立，可单独或组合开。

### 7.1 开关一览（env 变量门控）

| 开关 | env | 作用 | 默认(关) | 开 |
|---|---|---|---|---|
| #1 | `BOOST_FINGERPOSE=1` | 调大逐帧手指姿态跟踪系数（×5，让手更像人） | `fingerpose_coef=0.1` | `0.5`（可用 `FINGERPOSE_COEF` 调值） |
| #2 | `RELAX_PALM=1` | 放宽抓握 flag 的手掌阈值 + 去掉 palm 距离惩罚 | `palm_grip_thres=0.12, palm_dist_rew_w=2.0` | `0.22, 0.0` |
| #3 | `FIX_FINGER5=1` | finger_dist 补第5指(wuji `pinky`=right_finger5) | `fix_finger5=False`（仅4指） | 含5指 |
| #4 | `FINGER_POS_REW=1` | 指尖空间跟踪附加项 `-coef·Σ_{5指}‖sim − 参考指尖‖` | `finger_pos_coef=0`（无此项） | `1.0`（`FINGER_POS_COEF`；需 FPOS 数据）|
| #5 | `PALM_POS_REW=1` | **task空间 palm 位置跟踪** `-coef·‖sim_palm − 参考palm‖`（把 palm 钉在自然位、反贴握） | `palm_pos_coef=0`（无此项） | `PALM_POS_COEF`（代码默认 2.0，**本轮实验用 5.0**；需 FPOS 数据）|
| #5附 | `GLB_TRANS_COEF` / `GLB_ROT_COEF` | 覆盖手腕平移/旋转跟踪系数（钉手腕） | `0.6 / 0.1` | 任意（**本轮用 2.0 / 0.3**）|

> **为什么 #4/#5 需要 `WUJI_DATA_DIR=FPOS`**：参考指尖/palm 位置来自数据的 `link_key_to_link_pos` 字段，**只有 FPOS 数据(`GRAB_Tracking_PK_WUJI_FPOS_v1`)有**。FPOS = no-offset 原数据 **+** 离线 FK 加的 link 位置；q26/物体轨迹与 no-offset **完全一致(0 差异)**，训练等价，只是多了参考。不指向 FPOS → `ref_palm_pos/ref_fingertip_pos=None` → 该项空操作。

### 7.2 改的位置（活跃函数 `compute_hand_reward_tracking` 12716-13070 + task）

- **#1**：__init__ 里 `BOOST_FINGERPOSE=1` 时覆盖 `self.hand_pose_guidance_fingerpose_coef`（0.1→`FINGERPOSE_COEF`，默 0.5）。该系数本就作为参数传进活跃函数（`delta_value = 0.6·glb_trans + 0.1·glb_rot + coef·fingerpose`），**无需改函数签名**。
- **#2/#3 签名**：活跃函数加 4 个**带默认值**参数 `palm_grip_thres=0.12, palm_dist_rew_w=2.0, fix_finger5=False, right_hand_lf_pos: Optional[Tensor]=None`。
- **#2**：3 处 `right_hand_dist <= 0.12`（12904/12956/13009）→ `<= palm_grip_thres`；13023 palm 惩罚 `2.0·right_hand_dist` → `palm_dist_rew_w·right_hand_dist`。
- **#3**：task 指尖提取处（原 `'little'` 注释块）→ `if 'pinky' in hand_body_idx_dict` 取 `self.right_hand_lf_pos` 否则 `None`；活跃函数 finger_dist 计算后、clamp 前加 `if fix_finger5: if lf is not None: finger_dist += ‖obj-lf‖`（5指之和对阈值 0.6×5 一致）。
- **#4（数据）**：wuji 参考 npy 原本只有 q26，无指尖位置 → 两步离线生成（不覆盖原数据）：① `wuji_pipeline/add_link_pos_to_reference.py`（wuji-retarget 环境）pinocchio FK q26 → palm+finger1..5 世界坐标，存纯数组 npz；② `wuji_pipeline/assemble_fpos_reference.py`（dextrack 环境）合并进原 npy 的 `link_key_to_link_pos`、用 numpy 1.24 重存到 `data/GRAB_Tracking_PK_WUJI_FPOS_v1/`。**坑**：numpy 2.x 存的 dict-pickle npy 在 1.24 加载报 `numpy._core`，故走纯数组 npz 中转。
- **#4（reward）**：活跃函数加 3 个**带默认值**参数 `finger_pos_rew=False, finger_pos_coef=0.0, ref_fingertip_pos: Optional[Tensor]=None`；在 reward 算完后、return 前加 `if finger_pos_rew: reward += -coef·Σ‖sim指尖−ref‖`（th/ff/mf/rf/lf 对 ref finger1/2/3/4/5）。task 在 compute_reward 里从 `self.tot_key_to_tot_link_pos`（数据有 `link_key_to_link_pos` 时由 4022-4025 自动填充）按 `env_inst_idxes`+逐帧建 `self.ref_fingertip_pos`(N×5×3)，传进 5965 调用点。**用新开关 `finger_pos_rew`，不碰 `w_finger_pos_rew`**（后者会在 5860 转去 rbpos 路径）。启动需 `WUJI_DATA_DIR` 指向 FPOS 目录。
- **#5（reward）**：活跃函数加 3 个**带默认值**参数 `palm_pos_rew=False, palm_pos_coef=0.0, ref_palm_pos: Optional[Tensor]=None`；reward 算完后加 `if palm_pos_rew: reward += -coef·‖right_hand_pos − ref_palm_pos‖`（right_hand_pos=sim palm=`right_palm_link`）。task 在 compute_reward 里和 #4 共用一个构建块从 `tot_key_to_tot_link_pos['right_palm_link']` 逐帧建 `self.ref_palm_pos`(N×3)。`GLB_TRANS_COEF`/`GLB_ROT_COEF` 在 __init__ 直接覆盖 `self.hand_pose_guidance_glb_trans/rot_coef`（本就是活跃函数参数，无需改签名）。需 `WUJI_DATA_DIR` 指向 FPOS。
- **__init__**：统一在一个开关块读 env → `self.boost_fingerpose/relax_palm/fix_finger5/finger_pos_rew/palm_pos_rew/palm_pos_coef/...`，并 `print [REWARD SWITCH] ...`。
- **调用点**：仅活跃路径 5965 传 `self.*`（#2/#3 的 4 个 + #4 的 3 个 + #5 的 3 个新参数；#1/#5附 的 coef 走原有 `glb_trans/rot/fingerpose_coef` 参数）。

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

### 7.5b #5 启动（palm-lock，2026-06-07）

```bash
FPOS=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
# #2+#3 + #5(palm位置 coef5 + 手腕trans/rot 2.0/0.3)，#4不加；单 GPU2 / 多 GPU3
RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3 \
  WUJI_DATA_DIR=$FPOS SCRIPT_STEM=grab_single_wuji_palmlock \
  bash scripts/run_tracking_headless_grab_single_wuji.sh 2 ori_grab_s2_cubesmall_inspect_1
RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3 \
  WUJI_DATA_DIR=$FPOS SCRIPT_STEM=grab_multiple_wuji_palmlock \
  bash scripts/run_tracking_headless_grab_multiple_wuji.sh 3 '' ../assets/inst_tag_list_obj_cubesmall_palmlock.npy
```

### 7.6 实验记录（wuji cubesmall，未提交代码；完整配置 + 结果）

| 实验 | 完整配置（env 开关，均 + `WUJI_DATA_DIR=FPOS` 若含 #4/#5） | wandb / log stem | 结果 |
|---|---|---|---|
| #1 | `BOOST_FINGERPOSE=1 FINGERPOSE_COEF=0.5`(×5) | `..._1` / `grab_*_wuji_fp1` | 单 test(原始)**-7** ❌(×5过猛);**多已停**(弃) |
| #2+#3 | `RELAX_PALM=1 FIX_FINGER5=1` | `..._2_3` / `grab_*_wuji_relax23` | 单 test(原始)**-78**、举 1% ❌(放宽flag→退化);多在跑 best24 |
| #4 | `FINGER_POS_REW=1 FINGER_POS_COEF=1.0` | `..._4` / `grab_*_wuji_fp4` | 单 test(原始)**178**、举 **100%** ✅(但仍贴握);多在跑 |
| **#5 palm-lock** ★赢家 | `RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3` | `..._palmlock` / `grab_*_wuji_palmlock` | **★ ep464:fair@0.22=147、98% 持举、palm 0.6cm、手指 3.5rad**(从 ep312 的 88/90% 一路爬,还在涨)。**palm 完美在后 + 稳举 + 手指相对最收**——目前 naturalness+举升最优。强 palm 钉 + **不**碰手指=配方 |
| **pinall**(钉手指) | `...PALM_POS_COEF=2.0 GLB 1.0/0.2 FINGER_POS_COEF=1.5 BOOST_FINGERPOSE=1 FINGERPOSE_COEF=0.3` | `..._pinall` | ep331:手指被压到 ~4°/关节(贴参考)但**举 1-2%** ❌——**钉手指=反方向**(拽回脆参考,拆了鲁棒握) |
| **pinall2**(松手指) | `...PALM_POS_COEF=2.0 GLB 1.0/0.2 FINGER_POS_COEF=1.0` fingerpose=0.1 | `..._pinall2` | ep92:**99% 持举**(学得快,ep52 就举)、fair 15↑。但 **palm 较前(1.45cm)、手指更飞(5.2rad)** ——松手指**没让手指更贴参考,反而更飞**;palm 弱钉(2.0)也没 palm-lock 后 |
| **plfp**(强palm+轻#4) | `RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3 FINGER_POS_REW=1 FINGER_POS_COEF=1.0` fingerpose=0.1 | `..._plfp` | 🏃 2026-06-07 单 GPU5:**palm-lock 的强 palm + pinall2 的轻 #4**,试"palm 很后 + #4 把指尖拉得比 palm-lock 的 3.5 更贴参考 + 还能举" |
| **pinall3**(弱palm) | `RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=1.0 GLB_TRANS_COEF=0.6 GLB_ROT_COEF=0.1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0` fingerpose=0.1 | `..._pinall3` | 🏃 2026-06-07 单 GPU3(停 palm-lock 多腾):pinall2 手指参数 + **palm 都砍小**(钉1.0、手腕回默认0.6/0.1),探 palm 钉强度轴的弱端 |

> **palm 钉强度轴(单序列对比)**:palm-lock 5.0 → pinall2 2.0 → pinall3 1.0(均 #4=1.0,palm-lock 除外无#4)。趋势:**palm 钉越强 → palm 越后、手指反而越收**(5.0→palm0.6/手指3.5;2.0→palm1.45/手指5.2;1.0=pinall3 待测)。若坐实,**强钉(palm-lock)= naturalness 最优**,这轴到头。plfp 另探"强palm+轻#4"。

### ★★ 最终定量对比(全部 ep1000 跑完,2026-06-08,统一 fair@0.22)★★

| 实验 | 手指跟踪 | fair@0.22 | 持举% | 手腕cm | **手指rad** | 结论 |
|---|---|---|---|---|---|---|
| **pinall ⭐赢家** | fingerpose0.3 + #4=1.5(最重) | **201** | 98% | 0.98 | **1.74**(最低,≈5°/关节) | **手指≈参考 + palm后 + 稳举,全达成** |
| plfp | fingerpose0.1 + #4=1.0 | 191 | 99% | 1.33 | 3.04 | 手指偏松 |
| pinall2 | fingerpose0.1 + #4=1.0 | 184 | 99% | 1.31 | 3.59 | 手指最松 |
| palm-lock | 无(默认0.1) | 180 | 99% | 0.81 | 3.29 | palm最后但手指越握越偏(到5.9) |
| pinall3 | fingerpose0.1 + #4=1.0 弱palm | 180 | 99% | 1.29 | 3.21 | 手指偏松 |

> **★ 重大更正**:**pinall(重钉手指)才是赢家**——手指误差 1.74rad(全场最低、最贴参考)+ palm 在后(0.98cm)+ 98% 稳举 + fair 201(最高)。逐帧:物体跟参考 corr=1.00、手指全程 1.5-1.9rad(palm-lock 是 2.4→5.9 越握越偏)。
> **§8.3/§8.5 的"手指≈参考 与 能举 不可兼得"结论是错的**——那是基于早期(ep200)pinall 还卡在 1% 时下的。**重手指惩罚收敛慢(ep200 才 1%,但 ep1000 拿到 98%+手指1.74)**,最终逼策略找到了"贴参考又夹得住"的握法。**用户最初"钉手指让它贴参考"的直觉正确,只是需要训满。**
> 视频:`render_videos/reward_exps/FINAL_pinall_WIN_env77.mp4`(手指贴参考+palm后+稳举)vs `FINAL_palmlock_env79.mp4`(palm后但手指越握越偏)+ `FINAL_plfp/pinall2/pinall3`(5路全)。

### 8.6 抖动 → 加平滑（2026-06-08）

视频里手特别抖,因为**两个平滑机制默认全关**:
- **平滑奖励 `rew_smoothness_coef=0`**:活跃函数 12874 行 `smoothness_rew = -coef·‖prev_dof_vel − cur_dof_vel‖`(惩罚加速度突变/急动)。脚本写死 0.000。
- **动作 EMA `add_hand_targets_smooth=False`**:对 DOF 目标低通 `目标=(1-c)·新+c·上帧`(12337行),默认关。**⚠️ 纠正(2026-06-08):这行低通只在 `if self.not_use_kine_bias:` 分支里(12329),而 wuji 用 `use_kinematics_bias_wdelta=True`(`not_use_kine_bias=False`)→ 该分支不走 → 直接开此 flag 对 wuji 无效**,要用必须先把低通搬/复制到 kinematics-bias 的目标计算处。原"治法②翻开即压抖"不成立。
- 动作="参考+残差",残差逐帧自由振荡。

> **★ 诊断纠正(2026-06-08):抖动是策略内在残差震荡,不是 test 噪声 ★**
> 原以为 test 的 `whether_randomize_obs` 噪声让它抖。**实测错**:`whether_randomize_obs` 的噪声注入在 `apply_randomizations()` 里,只在 `if self.randomize:` 调用,而 test `task.randomize=False` → **噪声从不注入**。证据:`NO_RAND_OBS_ACT=1` 关随机化 vs 不关,rollout **md5 完全相同**、reward 都 201.20。这个确定性无噪声 rollout 仍 `|Δqpos|/frame=0.385`、**beyondRef=0.357**(扣掉参考运动后剩的)→ 抖动是**残差本身在震荡**,内在。
> 诊断开关:`NO_RAND_OBS_ACT=1`(task `__init__` 加,默认关,强制关 obs/act 随机化;test 里其实是空设但保留)。

**🏃 实验:pinall + 训练EMA(2026-06-08,GPU2,cubesmall 单)**:把手指目标 EMA 接进了 kinematics-bias 路径(pre_physics_step ~12540,prev_targets 更新前对 `cur_targets[:,6:]` 低通,gate=`add_hand_targets_smooth`,env 开关 `HAND_EMA_COEF`)。config = 老 pinall(201/98%)+ `HAND_EMA_COEF=0.4`,**不带 jerk**(rew_smoothness=0)。验证:`[HAND_EMA] applied ... on kinematics-bias path` 确实每步执行(不同于 cfg 自带那行只在 not_use_kine_bias 分支)。wandb `wuji_pinall_ema`。
> **★ 结果(ep1000,test 也带 HAND_EMA_COEF=0.4——策略在带 EMA 动力学下学的,test 必须同口径):EMA 成功,第一个奏效的抗抖方案 ★**
> | | fair@.22 | 举升% | \|Δqpos\|(抖) | beyondRef | fingerErr |
> |---|---|---|---|---|---|
> | pinall_EMA(0.4) | 179.3 | **93%** | **0.277** | **0.252** | 2.14 |
> | 老 pinall(无平滑) | 201.2 | 98% | 0.385 | 0.357 | 1.74 |
> | pinall_sm(jerk) | 2.5 | 1% | — | — | — |
>
> **抖动 ↓28%**(0.385→0.277)**且抓握保住**(举升 93%,非 jerk 的 1%)。代价温和:fair −11%、举升 −5pt、指误差 1.74→2.14(EMA 滞后让手指略松)。**坐实分析:EMA=滤波非惩罚→不毁抓握**(对比 jerk 罚到参考加速→脱抓)。coef 0.4 可调:更大=更平滑但更滞后/代价更高。视频 `render_videos/smooth_test/pinall_EMA_coef0.4_*.mp4`。

**🏃 无平滑重跑(2026-06-08)**:平滑批次毁了抓握,故 pinall/plfp 的 flute单 + cubesmall多 **去掉 REW_SMOOTH 重跑**(= cubesmall 单能 work 的那版,jerk 关、无 EMA、FPOS)。wandb:`wuji_pinall_flute`(G3)、`wuji_plfp_flute`(G4)、`wuji_pinall_multi`(G5)、`wuji_plfp_multi`(G7)。多任务用 tag 列表副本 `..._pinall.npy`/`..._plfp.npy`。config 已 diff 确认 = sm 版减 REW_SMOOTH(rew_smoothness=0、HAND_EMA=0)。
> **flute 单结果(~ep850,fair@.22):pinall -34.79 / plfp -39.16,均 0% 举升 → 失败**。手贴参考(wrist 0.2-0.5cm、finger 1.3-1.7)但抓不住 2cm 细杆。**flute 失败与平滑无关,是物体几何难**(和一贯 wuji flute -42 一致,无平滑也没救)。视频 `pinall_flute_env1/3_floor.mp4`/`plfp_flute_env1/3_floor.mp4`(手贴参考、flute 留地;env0 是小弹动离群勿用)。
>
> **flute 新方向(2026-06-08):参考举不起来是预期的——kinematics-bias 下 RL 靠残差修;flute 0% = RL 没学出"夹住细杆"的残差。挡着残差的是:#4/#5 把手钉在(不夹的)参考上 + palm-back 把掌推远。** 两个尝试:
> - **`wuji_pinall3_flute`**(G2):pinall3 轻钉(relax_palm + finger/palm 各 1.0)——比 pinall 的 1.5/2.0 松,但仍钉。
> - **`wuji_flute_loose`**(G6):**放手让残差去夹**——去 relax_palm(palm 须到 0.12 贴杆)+ **关 #4/#5**(手指自由)+ **接触奖励 RFOD=0.6**(默 0.3,驱动手指碰杆帮探索)。直觉:cubesmall 要"钉紧",flute 要"松开+驱动接触",需求相反。
> - 新增 env 开关 `RFOD`(单序列脚本 120 行 `rew_finger_obj_dist_coef="${RFOD:-0.3}"`)。
> - **结果(ep1000,fair@.22):pinall3_flute -41.59/0%、flute_loose -293.47/0%(手腕飞 13.36cm)→ 两个都失败,松开版更糟。** 去掉 #4/#5 位置锚后手不去夹杆而是到处飘(13cm 误差);位置跟踪至少把手拴在物体附近。**flute 确认死路**:pinall/plfp/pinall3/loose 四配置全 0%,根因是 **wuji 长手指 × 2cm 细杆几何不匹配**,reward shaping 救不了。视频 `pinall3_flute_env1.mp4`/`flute_loose_env1.mp4`。
> cubesmall 多任务 ep350/10000 太早,待 ep600+ 再测。
> **★ cubesmall 多任务中途测(ep368,在 cubesmall_inspect_1 轨迹评估 generalist):plfp_multi fair+14.37/举升96% ✅,pinall_multi fair-85.80/举升0% ❌——和单序列排名反转 ★** 单序列 pinall(重#4 1.5)最优,但多任务上 **plfp(强palm5.0+轻#4 1.0)收敛/泛化快得多**;pinall 重#4 钉手指在 26 条轨迹上约束过多、收敛慢。pinall_multi 尚早或可追上。视频 `plfp_multi_ep368_env50.mp4`(举起)/`pinall_multi_ep368_env1.mp4`(floor)。
> **多轨迹 sweep(ep368,plfp vs pinall,举升%)**:s2_inspect 96/0、s1_lift 19/0、s5_pass 2/0、s8_offhand 0/0 → plfp 全面 > pinall,但 plfp 也只先学会 inspect 类(早期不均匀)。
> **★ 三方消融(2026-06-08):加跑 `wuji_pinall3_multi`(G4,pinall3=轻#4 1.0+弱palm 1.0)拆解 plfp 为何行 ★** pinall_multi(重#4 1.5)0% / plfp_multi(轻#4+强palm5.0)96% / pinall3_multi(轻#4+弱palm)待训。pinall_multi 未停(用户要它跑完);pinall3_multi 用 plfp_flute 跑完腾出的 G4。tag 列表副本 `..._pinall3.npy`。
> **★ 消融结论(ep490,inspect_1 举升%):pinall_multi(重#4 1.5)0% / pinall3_multi(轻#4 1.0+弱palm1.0)93% / plfp_multi(轻#4+强palm5.0)96% → 关键是"轻 #4",palm 强弱次要 ★** 重 #4 把手指钉在不夹物体的参考位置、在多样轨迹上罚死夹取所需的大残差→0%;轻 #4(≤1.0)松开手指→残差能去夹→93-96%。**正好解释单→多排名反转**:重#4 单序列能 work(残差小贴紧好),摊到多任务就拖死。**实操:多任务用轻#4(plfp/pinall3),别用 pinall 重#4。** s1_lift 两者 ep490 仍 0%(早期没学会该轨迹,inspect 类先成)。视频 `pinall3_multi_ep490_env*.mp4`。
> **全任务 sweep(26 轨迹,mid-training:plfp ep903/pinall3 ep506/pinall ep891,fair@.22 + 举升%)**:plfp 均fair0.4/均举升17.8%/举起5/26;pinall3 -4.9/12.7%/3/26;pinall -53.0/8.9%/2/26。**plfp>pinall3>pinall 一致**。逐轨迹:三者都会 s2_inspect(99/93/94)、s10_pass(99/99/96);plfp/pinall3 会而 pinall 不会 s7_pass(96/97/0);只 plfp 会 s2_pass/s9_inspect。**大多数轨迹(几乎全部 lift/offhand 动作)三者仍 0%** → generalist 一簇一簇学、远未收敛,这是"reward 抬升慢"的真相(26 条只学会少数)。reward 曲线图 `render_videos/multi_reward_curves.png`、vs baseline `multi_vs_baseline.png`(同期 plfp≈baseline,baseline 训到 ep8090 峰值~66;pinall 重#4 明显垫底)。脚本 `fulltask_test.sh`。
> **★ pinall3_multi 全 26 轨迹(收敛后,ckpt ep1851 rew65.19,2026-06-09,fair@.22 + 举升%)★**:**均 fair=36.7 / 均举升=36.8% / 举起(>50%)=10/26**(对比 ep506 时 3/26 → 训到 ep1851 翻到 10/26,确实在一簇簇学)。**清晰二分:inspect/pass 类几乎全成、lift/offhand 类全挂**。成功(10):s9/s2/s10/s7 的 inspect/pass→99%,s5_inspect93%,s9_pass89%,s3_inspect81%,s6_inspect78%。失败(16):**所有 6 条 `lift` + 所有 3 条 `offhand` 一律 0%**,部分 pass(s1/s4/s5/s6/s8)0–7%。**关键现象:fair 高 ≠ 能举**——s5_lift fair=95.8、s10_lift=107、s8_lift=63 却 0% 举升;因为 fair 主要计手位+物位跟踪,前半段贴参考就拿分,只是最后抬不起来。→ generalist 对"需要真正抬起物体"的动作(lift/offhand)仍未学会,inspect/pass(更多是贴近+移动、抬升幅度小)先收敛。脚本 `sweep_pinall3_multi.sh`。

**★ 平滑方式调研(借鉴 wuji-mjlab/spider/wuji-retargeting,2026-06-09)★**:
- **wuji-mjlab(厂商自己在这只手上的 RL)= `action-EMA(α0.5) + action_rate(1阶+2阶,-1.0) + torque(-24)` 组合**;
- wuji-retargeting = 输出 `LPFilter` 低通;spider = reference 侧滤波+样条 + rollout 滑动平均;
- **关键借鉴 = `action_rate` penalty**:`-coef·[‖aₜ−aₜ₋₁‖² + ‖aₜ−2aₜ₋₁+aₜ₋₂‖²]`,**在残差(策略输出)上、不碰 dof_vel** → 不像 jerk 罚参考加速,**不毁抓握**,且厂商验证过。
- 已实现开关 `ACTION_RATE` + `ACTION_RATE_COEF`(残差历史 prev/prev_prev + jit 1阶+2阶项 + BREAKDOWN `actionRate`)。**坑:coef 0.01 太高**(actionRate=-1.02 压过整个 reward),raw≈100,**降到 0.0005**(actionRate≈-0.053,占 reward ~7%,合理且随策略变平滑自减)。
- **全栈实验** `wuji_pinall3_full`(G2,pinall3+cg+clip+EMA0.6+action_rate0.0005,cubesmall 单):看严重抖动能否压下且不掉抓握。

**★ top-5 测试结论(2026-06-09,逐帧抖动 |Δqpos|)★**:pinall3=0.351、**cg=0.658(翻倍!cg 是抖动元凶——拽手指碰接触点)**、**clip=12.5(爆抖!CLIP_DOF 硬钳 dof_pos+set_dof_state 每步 → 限位 chattering,实现有缺陷)**、full=0.309(EMA0.6 把 cg+clip 的抖都压平,但 fair 179→64、手指 4.14→9.07,过头)。lift:cg/clip/full 都 99%(cubesmall),**但 cgclip_flute 仍 0%(contact guidance 没救 flute,fair 升到 13 但举不起)**。教训:**clip 别用(chattering)、EMA0.6 太狠、cg 价值存疑(增抖不增益)**。
**🏃 cg+smooth 批(2026-06-09,smooth=EMA0.4+action_rate0.0005,无clip)**:`wuji_pinall3_cg_smooth_single`(G1)、`_smooth_flute`(G2,max_epochs=3000)、`_smooth_multi`(G5)、`wuji_pinall3_cg_multi`(G7,cg-only对照)。脚本 `max_epochs/episodeLength` 改成 `${MAX_EPOCHS}/${EPISODE_LENGTH}` 可 env 覆盖。停了 pinall_multi(垫底)/plfp_multi(最优,已刻画)。

**★★ flute 重大修正(2026-06-10):flute 不是死路,之前 0% 是指标误导 ★★**
- **指标 bug**:之前判 flute 0% 用的是"**峰值位移帧**物体跟踪<5cm"。pass 动作峰值在最远最高的极点(1.24m),策略恰好在那丢杆 → 报 0%。但**前中段(前 ~140-160 帧)抓得很稳**(误差<3cm)。**正确指标看全程跟随/持握时长,不能只看峰值帧**(lift/offhand/pass 大动作类都受此 bug 影响,之前的成功率可能都偏低)。
- **实情(s2_flute_pass_1)**:两个手都**抓住 flute 并带过大半 pass**——allegro 物体移动中位 0.71m/持握 5.4s、wuji_cg_smooth 0.54m/4.6s、cgclip 0.53m/4.8s(参考 1.24m)。只在 pass 极端伸展点撑不住(wuji 长手指+细杆够不到)。**有 bonus**(step1 实测 0.97),持握那段一直拿。
- **fair@0.22 排名**:cgclip **+2.24**(唯一正,最好)> cg_smooth −10.54 > pinall3(无cg) −41.59。**cg 是主要功臣**(两个 cg 变体都远超 pinall3 基线;flute 是杆状,contact guidance 帮手指找接触点特别有用)。clip 大概率中性(cubesmall multi 已证 clip 无正作用),cgclip>cg_smooth 更可能是 smooth 在 flute 大动作上有害。**起 `wuji_pinall3_cg_flute_single`(G2,纯 pinall3+cg,无 clip/smooth)一锤定音**;停了 `cg_smooth_flute`。
- **flag-morphology(关键!)**:`palm_grip_thres`(握住 flag)gate 的不只是 bonus,还有 `goal_hand_rew = where(flag==2, −2·goal_dist, 0)` 的**物体位置惩罚**。**wuji 长手指 → 握住时手掌离物体 0.12~0.22,必须 flag@0.22 才被认成握住、才拿 bonus**(@0.12 时 wuji bonus 全丢 → fair −115.69);allegro 短手指天生 @0.12。**所以 RELAX_PALM(0.12→0.22)对 wuji 是必需的,且跨手 fair 不可直接比**(@0.22 对 allegro 偏苛:它丢杆的帧 palm 落在 0.12~0.22 被判"握住"→ 吃 −2·goal_dist 狠罚 → allegro fair@0.22 −49.81，但 native@0.12 是 +39)。
- 视频:`report_videos/{allegro_flute,cg_smooth_flute,cgclip_flute}/ref_vs_policy.gif` + `flute_4panel/`(4 格同视角;给 playback 加了 `--cam_eye/--cam_target` 固定相机)。**渲染参考别用 `ls *flute_pass_1*` 通配**(匹配 s1/s2/s6/s9/s10 多被试,head 取错),要显式完整文件名。

**治法**:抗抖动需训练侧手段,但**别用 jerk-on-dof_vel**(§8.6 已证毁抓握——它连参考的抓-举加速一起罚)。只约束残差的方向(均需先确认在 kinematics-bias 路径上生效,见上 EMA 的坑):① **action-rate penalty** `‖residualₜ−residualₜ₋₁‖`(最对症,需加项);② **训练期 EMA** `add_hand_targets_smooth`(机制最干净=不罚只滤,但**代码只在 `not_use_kine_bias` 分支,wuji 路径要先把低通搬过去**——非零代码);③ 调大 **action-L2** `actionPenaltyScale`(现 -0.0002 → 更大,残差更小,需先验证它在活跃路径上确实生效)。`add_torque/work/global_motion_penalty` 同理需防过罚抓握。

**`rew_smoothness_coef` 脚本里写死 0.000 且前面多处覆盖** → 单序列脚本 440 行已改成 `${REW_SMOOTH:-0.000}`(独立变量名,避开前面的硬覆盖),用 `REW_SMOOTH=0.002` 传入。

**当前 jerk**:无平滑的 pinall `||Δdof_vel||`≈10.4/步 → 加 0.002 平滑惩罚 ≈ -6.3/集,**仅占 reward(~200)的 3%**,温和、主要压抖不动主行为(策略会把 jerk 压下去赚回这 6 分)。

**平滑批次(🏃 2026-06-08,均 REW_SMOOTH=0.002 + FPOS)**:
| 实验 | 配置 | 任务 | GPU | wandb |
|---|---|---|---|---|
| pinall_sm | pinall 全套 | cubesmall 单 | 2 | `wuji_pinall_sm` |
| pinallsm_flute | pinall 全套 | **flute 单** | 3 | `wuji_pinallsm_flute` |
| plfpsm_flute | plfp 全套 | **flute 单** | 4 | `wuji_plfpsm_flute` |
| pinallsm_multi | pinall 全套 | **cubesmall 多** | 5 | `wuji_pinallsm_multi` |
| plfpsm_multi | plfp 全套 | **cubesmall 多** | 7 | `wuji_plfpsm_multi` |

> 脚本平滑覆盖:单序列脚本 440 行、多任务脚本 610 行均改 `${REW_SMOOTH:-0.000}`(独立变量避开前面硬覆盖)。flute 难(2cm 细杆,wuji 此前 -42 失败),这批看新配置 + 平滑能否改善。

> **★ 结果:平滑(REW_SMOOTH=0.002)把抓握搞坏了(2026-06-08,3 个单序列 ep1000 测完)★**
> | exp | fair@.22 | 真持举% | fingerRad | per-finger | 备注 |
> |---|---|---|---|---|---|
> | pinall_sm (cube) | **2.51** | **1%** | 1.07 | 0.1/0.1/0.2/0.2/0.3 | 手紧贴参考但物体留地 |
> | pinallsm_flute | -30.25 | 0% | 1.06 | — | flute 仍失败 |
> | plfpsm_flute | -36.41 | 0% | 1.47 | — | flute 仍失败 |
>
> **对照(同测试设置,无平滑老 pinall):fair=201.20、持举 98%、物体升 0.378≈参考** → 证明不是测试问题。唯一差异 `REW_SMOOTH=0.002`。
> **机理**:rollout 里物体**每帧升幅≈0.000**(参考升到 0.38),手指误差却很小(贴参考)→ 手平滑跟随参考运动学在空中做举的动作,但**没建立抓握/物体留地**。jerk 惩罚(哪怕 0.002)阻止了脆弱抓握所需的快速捏合(呼应 §8.5 抓握脆弱+§8 长手指放大关节误差)。之前"0.002 温和占 3% 不伤主行为"的估计**错误**。
> 视频:`render_videos/smooth_test/`(pinall_sm_smooth_FAIL vs pinall_NOsmooth_OK 并排)。
> **结论**:**REW_SMOOTH 这条路在此任务上是死路**(至少 0.002 级别)。抗抖动改走 EMA 动作滤波(`add_hand_targets_smooth`,test 期低通、不改训练 reward),或干脆接受抖动(老 pinall 201/98% 可用)。pinallsm_multi/plfpsm_multi 多任务带同样平滑→大概率同样坏,建议停。

> **打分尺度（重要）**：test 默认=原始 reward(flag@0.12)对 palm-back 不公平(见 §8.4);公平加 `PALM_GRIP_THRES=0.22`(独立开关,只放宽阈值)。**最可靠是举升%**——且用**"参考峰值帧仍举着"**(真持举),不要用 `z.max>5cm`(会把碰飞算进去,虚高:palm-lock 98% vs 真持举 90%)。
> **reward 分解工具**：活跃函数末尾 `REWARD_BREAKDOWN` print(帧 1/150 快照;临时改成每步 gate 可累加成整集绝对值)。实测 palm-lock 尺度整集:bonus +85,负的几乎全在手指(fingerJoint -111、fingerPos -64),palm 几乎不花钱(palmPos -2)。
> **监控**：`monitor_pinall_trend.sh`(高频/GPU6,每 60s 测 best 的 fairRew/举升%/fingerErr/wristErr)。
> **当前结论(2026-06-07)**:**palm-lock(强 palm 钉 5.0 + 不碰手指)是最优**——palm 最后、手指相对最收、98% 稳举。"钉手指"(pinall)反害、"松手指"(pinall2)更飞;palm 弱钉不如强钉。plfp 在试"强palm+轻#4"能否再把手指拉近一点。**"手指≈原轨迹+能举"仍受 §8.5 物理坎制约**。
> 视频:`render_videos/reward_exps/` —— palm-lock(palm后+稳举)、pinall(自然手但滑掉)、pinall2(举到顶但手指飞+早松)。
> 注:#1/#2+#3/#4 多已停;各 single + palm-lock多 + baseline多 在跑。

### 8.7 wandb 里加"原始/公平 reward"监控（2026-06-08）

各 reward-switch 配置的训练 reward 不可比(开关不同 → 量纲不同)。于是在**训练进程里每步多算一次"公平 reward"**作为配置无关的任务指标,直接进 wandb 曲线。

- **口径(永远固定)**:`flag@0.22` + 原始系数(`glb_trans=0.6/glb_rot=0.1/fingerpose=0.1`)+ palm 惩罚 `2.0` + 4 指 + **无** #4/#5 位置项 + **smoothness=0**(这样平滑/非平滑 run 也可比)。= §8.4 公平打分、= 测试脚本 `PALM_GRIP_THRES=0.22` 那套。
- **实现**:`allegro_hand_tracking_generalist.py` 活跃 reward 调用点(~6007)后,用同状态、上述固定参数**再调一次** `compute_reward_func`,只取 `[0]`(jit 函数无副作用,丢弃的 reset/successes 无影响);按 episode 累加 → `self.reward_fair_mean`(对标测试的 episodic 180-201)。`a2c_supervised.py` 日志块(~3731)加 `add_scalar('reward_fair/{step,iter,time}', ...)`。
- **开关**:`LOG_FAIR_REWARD`(默认 1;设 0 完全去掉这次多算)。`early_terminate=False` → episode 定长,fair 累加与测试同口径。
- **注意**:wandb 这条是**训练中随机策略**上算的(带探索噪声),会系统性**低于**doc 表里的最优-ckpt eval(180-201),随收敛靠近——看趋势/相对高低,绝对值仍以 eval 为准。
- **开销**:每步多一次 jit reward(纯张量),实测单序列 fps total 仍 ~82-85k,**无可见变慢**。
- **验证**:`[REWARD SWITCH] ... log_fair_reward=True` ✓;TB 标量 `reward_fair/iter` 已写(epoch2 = -93.6,vs 训练 reward -359 → 确实剥掉了 #4/#5 惩罚)✓。
- **代价**:5 个平滑 run(§8.6 表)为加此监控**于 ep~120 重启**(从头跑,丢 ~1h);现已带新代码重新在 GPU 2/3/4/5/7 上跑。
- **坑(已修)**:第一次重启**漏传 `WUJI_DATA_DIR=$FPOS`**(从 `/proc/environ` 扒配置时只 grep 了 reward 开关名,漏了数据目录)→ 默认回退到非-FPOS 的 `WUJI_v1` → `ref_*=None` → **#4/#5 静默变空操作**,reward 结构变样、wandb 曲线与原 run 完全对不上。**教训**:重启 wuji #4/#5 的 run 必须带 `WUJI_DATA_DIR=$FPOS`;验证用 `[FINGER_POS]` print(在 run 目录 `screen.log`,显示 `ref_fingertip_pos=(N,5,3) ref_palm_pos=(N,3)` 才算生效,`=None` 即漏了)。已带 FPOS 重启修正。

---

## 8. 抓握物理调查：为什么"palm 总是贴物体"（2026-06-07，关键结论）

**问题**：wuji 手指长，用户希望 palm 离物体远(自然)、靠长手指夹、策略只微调；但所有能举起来的策略(baseline/#4)都把 palm 贴近物体、手指过弯，和参考差很大。逐层查清如下。

### 8.1 测量与工具
- **参考 palm 距离**：用 FPOS 数据的 `right_palm_link` FK 位置 vs `object_transl`，**参考 palm 离物体 ~14cm**（抓握期）；而原始抓握 flag 要求 palm ≤ **0.12m=12cm** → **参考姿态本身都不满足原始 flag**，原始 reward 逼策略把 palm 拉到比参考还近。
- **真物理测试** `isaacgymenvs/reference_physics_test.py`：开重力、物体自由刚体、手按精确参考 qpos PD 驱动。结论：**no-offset 参考抓握完全有效**（物体跟参考 corr=1.00，抬到 41cm 再放下）。注意 `wuji_isaacgym_playback.py` 是**纯运动学**(无重力+物体焊死)，那种"reference"视频不能用来判断物理抓握。
- **裕度测试**（给参考手指加高斯扰动）：**0.03rad(~2°)手指误差就在抬升中途滑掉，0.08rad(≈RL策略实际精度)完全抓不住**。摩擦 1.5→8 都救不了（几何问题非打滑）。

### 8.2 三个变体的跟踪实测（同 env，qpos 空间）
| 变体 | 手腕跟踪误差 | 手指 qpos 误差 | 举升 | 解读 |
|---|---|---|---|---|
| #4 | 5.0cm | 3.36rad | 100% | 举得起=偏离参考"贴+弯"造裕度 |
| #2+#3 | 3.2cm | 1.65rad | 1% | 忠实参考=palm 自然但**掉物体** |
| baseline | 5.9cm | 3.60rad | 100% | 同 #4，靠贴握 |

### 8.3 结论
1. **"palm 贴"是物理逼出来的功能性补偿**：参考的"指尖轻捏"自然抓握裕度极低(0.03rad)，RL 精度(~0.08rad)够不着 → 策略只能偏离到"贴握+过弯"的厚裕度包络抓握才举得起。**reward 不是根因**。
2. **便宜的招全失败**：摩擦(几何无效)、offset 参考(指尖外扩→更差，精确都抓不起)、naive 均匀多弯(把精确抓握也搞掉)。
3. **长手指是结构性原因**：0.08rad 关节误差 × ~15cm 指长 × 4 节 ≈ 指尖偏 1cm+ → 脱离 5cm 小方块。**小方块是长手指最难的情形**。
4. **(d) 尺寸假设无法验证**：重定向只对 cubesmall 把指尖放到表面；cubemedium 勉强(+0.6cm)、cubelarge 差 7.6cm 完全没合上 → 得不到有效的大物体抓握，无法对比尺寸。
5. **真正的解需要 grasp synthesis / 更好的重定向**(palm 后 + 手指鲁棒夹持，裕度 > 0.08rad)，但用户认为不可 scale。
6. **#5 palm-lock 实测(ep147)**：纯训练侧——锁 palm 在自然位 + 加强手腕跟踪。结果:**手腕跟踪误差钉到 0.3cm**(baseline 贴握 5.9cm)→ **palm 完美在后,达成了"自然 palm"这半边**;但手指误差 2.65rad、**举升仅 1%**(没夹住、掉)。**证明:palm 能钉住,但光钉 palm 举不起,抓握在手指**。→ **pinall**(在 palm-lock 基础上把手指也钉紧:#4 指尖位置 coef1.5 + #1 温和 fingerpose0.3,同时 PALM_POS_COEF 5→2 留夹持余地)。

### 8.4 打分尺度:原始 reward 对 palm-back 不公平

原始 reward 的 bonus(大头,+0.94/步)被"握住 flag"门控,要 **palm ≤ 0.12m**。palm-back 策略(palm ~14cm)即使**接近阶段物体贴着参考**也拿不到 bonus(goal_dist = 物体 vs 逐帧参考物体位置,接近阶段本应小)→ 白扣 ~+几十。实测 palm-lock:原始(flag@0.12)**-89** vs 公平(flag@0.22)**-21**,差的 68 分纯属阈值偏差。**故 palm-back 策略要用 `PALM_GRIP_THRES=0.22` 公平打分(独立开关,只放宽阈值);最可靠仍是举升%**(与开关无关)。

### 8.5 palm-lock 突破 + "手指自然 ↔ 能举"的硬权衡（关键，2026-06-07）

**palm-lock 在 ep312 突然学会举了**(之前 ep147 还 1%):
- **90% 持举**(参考峰值帧物体仍在 35.7cm,参考 40.8,corr=0.95),**palm 完美在后**(手腕跟踪 0.7cm,98% env <1cm)。
- **怎么做到的**:**钉 palm + 放开手指** → 手指**自己摸出一个鲁棒握**(误差 ~3.8rad,**不是**参考那个脆的轻捏),palm 保持自然在后。**所以:别约束手指,让它自由去找夹得住的握法。**

**反证(pinall)**:在 palm-lock 上**加手指跟踪**(把手指往参考拽),手指压到 ~1.5rad(贴参考)→ **举 1%**。**钉手指 = 反方向**,拆了那个鲁棒握。

**量化"手指自然 ↔ 能举"是互斥的**(换算到每关节):

| | 手指误差(/关节) | 举升 |
|---|---|---|
| pinall(钉手指) | **~4°/关节**(≈参考) | 1% ❌ |
| palm-lock(放手指) | **~11°/关节** | 90% ✅ |

→ **越贴参考越夹不住**:举起 5cm 小方块,手指物理上**必须**比人手多弯 ~11°/关节去做个更实的握。**"手指≈原轨迹 + 能举"在当前(脆)参考下不可兼得**——要兼得只能把参考抓握做 robust(grasp synthesis)。pinall2 在试中间点(松手指惩罚),但大概率跨不过这道物理坎。

**补充**:
- **小指**不是最自由的(各指关节误差:拇0.61/食0.89/中0.63/无名0.92/小0.76 rad);小指**看着**自由是因为最长、指尖摆动明显,关节其实居中。`finger_dist` 在 `FIX_FINGER5=1` 时**有**带小指(活跃函数 12790),但 5cm 小方块没小指的位置,拉不过去。
- **视频对比**:`render_videos/reward_exps/palmlock_ep312_clean_env92.mp4`(palm 后 + 真举到 41cm)vs `pinall_ep181_env92.mp4`(手指更自然但夹起一点就**滑掉、空手举**)——一眼看清权衡。

相关工具：`reference_physics_test.py`（真物理抓握/裕度测试）、`wuji_pipeline/add_link_pos_to_reference.py`+`assemble_fpos_reference.py`（FPOS 参考 link 位置）、`monitor_pinall_trend.sh`（高频测 fair reward+举升+手指误差）。视频在 `render_videos/reward_exps/`。

### 8.8 反关节 / dof 限位 clip（师兄建议改进#1，2026-06-08）

**调查**:wuji URDF(`wuji_hand_description/urdf/wuji_hand_right_fly.urdf`)手指限位 —— joint1≈[-0.16,1.6]、**joint2=±0.37(侧摆 abduction)**、**joint3/joint4 lower≈-0.46~-0.48(-27°,远端指节反掰空间)**。任务在 4926 行直接用 URDF 限位做 **target 钳位**(4812-4815 有注释掉的"i>5→append(0.0)"clip hack,前人想过没启用)。

**关键发现(用 Isaac Gym 实测 dof 顺序+限位、再统计 26 轨迹参考 + rollout):**
- dof 顺序 = 6 全局 + finger1..5×joint1..4(idx6-25),sim 限位 = URDF;
- **参考本身就用满 URDF 范围**(joint2 到 ±0.37、joint3/4 多条轨迹 refMIN 顶到 -0.47)→ **不能 clip 到 0**(会夹坏参考);
- **策略冲出 URDF 硬界**:rollout polMIN 到 -1~-2.5(限位 -0.47),但**核实是瞬态尖峰**(joint3/4 不在 >0.5%帧的持续超界名单);**唯一持续超界是 joint2**(侧摆,20-78%帧坐在界外 0.01-0.03rad)。→ 反关节实情=**偶发瞬态弹动(像抽搐)+ joint2 轻微过张**,非手一直反掰。target 钳位管不到(是接触力把软限位顶穿)。

**改进 = 硬化 URDF 限位(不是设0)**:独立开关 `CLIP_DOF`(默认关)。`compute_observations` 刷新后把 hand dof_pos 钳到 `[lower,upper]`、钳处速度归零、`set_dof_state_tensor_indexed` 写回 sim。验证 `[CLIP_DOF] clip_dof_limits=True` + `active: clamped N violations`(每步在跑;set_dof_state 每步写回未写崩训练)。

**实验**:`wuji_pinall3_clip`(G2,pinall3+CLIP_DOF,cubesmall 单)vs **已有 pinall3**(179.63/99%)。看点:clip 是否**不掉举升**的前提下让手更干净(rollout 全程不超界 + 渲染无反关节弹动 + 顺带压抖)。测时硬验:对照无clip rollout 冲到 -2.5,clip 版应全程 ≤ URDF 界。

### 8.8b 反关节复盘 + 软惩罚（2026-06-10,改用 soft penalty 替代硬 clip）

**复盘 clip 有没有用(cgclip vs cg 多序列,fair 同 epoch 隔离):** 早期(ep<400)cgclip 略快,**ep500 起 cg(无clip)反超并领先**(ep700-800:cg 51.8 vs cgclip 40.6),cg 斜率还更陡;clip 每步 set_dof_state **慢 ~25%**。→ **clip 对 reward 无正作用**,绝对值高只是训得久。

**但 clip 不是"空操作"(修正之前结论):** test 时 cgclip `clamped 0 violations`(收敛策略已学乖、不超界),但**无 clip 的策略确实冲出 URDF**:`p3msw`(pinall3)joint3 到 −0.567(轻微);**`cg-only` −1.28(9% env)、`cg_smooth` −2.19(49% env、各 ~8/300 帧)**。即 **contact_guide 把手指往接触点卷→引反关节,smooth(EMA+action_rate)放大一大截**;clip 训练时 clamp、策略因此学会不超界。但越界是**短暂尖刺**(0.14% 帧),且**不影响任务 reward**(cg_smooth 照样 95-99% 跟踪)。越界遍布各 joint(joint2 外展下界 50-75% 帧 marginal、joint1/3/4 幅度大)。

**结论:clip 有效(阻越界)但没用(越界不损 reward)+ 代价大(慢+单序列 chatter 12.5)。** 渲染/真机在乎"手指反折"才需治 → **选软惩罚,不用硬 clip**(避开 chatter+慢,且教策略自己避开、可泛化)。

**实现 `SOFT_LIMIT`(默认关,task __init__ + compute_reward Python侧):** 对**achieved 手指 qpos**(`shadow_hand_dof_pos[:,6:]`,**20 指节全做、跳 6 个全局**)做 **quadratic hinge 双向**惩罚:`−coef·Σ_j[relu(lo−q)²+relu(q−hi)²]`,用各 joint URDF [lower,upper]。`SOFT_LIMIT_COEF` 可调。**加在 fair 调用之后 → fair 保持干净可比**。quadratic 自动分级:marginal 越界(joint2 0.01)≈0、严重(1-2rad)狠罚。参考贴在界上(−0.47)→ relu=0 不被罚,符合"保留参考用反关节"。step1 打印 `[SOFT_LIMIT] active mean_pen max_pen` 验证量级。

**实验**:`wuji_pinall3_cg_smooth_softclip`(=pinall3+cg+smooth+SOFT_LIMIT coef0.5,G1,26 cubesmall,FPOS,wandb `wuji_pinall3_cg_smooth_softclip_multi`)。启动验证:GPU1、所有开关 active、`mean_pen=0.0153 max_pen=7.91`(平均可忽略、只狠罚严重越界 env,不压垮)。脚本 `run_tracking_headless_grab_multiple_wuji.sh` 第287行 cuda_idx 改成 `${cuda_idx:-2}` 可 env 覆盖(注意 `$1` 同时给 GPU 和 st_idx,st_idx 对 tag-list 运行无关)。

**★★ test 结论(2026-06-10,softclip 划算):** 匹配 epoch test(cg_smooth `last_ep_1000` vs softclip `best_ep_936`,3 轨迹 s2_inspect/s2_pass/s10_pass,HAND_EMA=0.4,fair@0.22)：

| | fair@0.22 均 | 反关节 env 峰值>0.5 占比 | max 越界深度 |
|---|---|---|---|
| cg_smooth ep1000 | **92.3** | **36%** | 3.07 |
| softclip ep936 | **88.3** | **5%** | 1.43 |

**softclip:fair 仅 −4(噪声内,s2_pass 还 109.8>106.5 反超),反关节 env 占比 36%→5%(7× 少)、最深 3.07→1.43。最严重的 s2_inspect:94%→13%。** → **近乎零代价换掉绝大部分反关节,coef=0.5 合适。** 反关节深度 = `Σ_{20指节}relu(lower−q)` 从 rollout `shadow_hand_dof_pos[:,6:26]` 算,wuji dof 下限见 `/tmp/wuji_dof_limits.npy`(joint3/4≈−0.47)。

**坑(重要):wandb 严重滞后训练 ~1240 epoch**(reward_fair 只同步到 ep687,本地实际 ep1927)。**fair 全程曲线要读本地 tensorboard**(`logs/<run>/.../summaries/events*`,`reward_fair/iter` 完整 n=1932),`event_accumulator` + `size_guidance={'scalars':0}` 不降采样;**别信 wandb summary/history(截断)**。真·fair 曲线对比图 `wuji_pipeline/out/softclip_vs_nosoftclip_fair_REAL.png`:softclip 收敛慢但向 cg_smooth/cg 带收拢(ep930 差 −14 训练态、但 test 态只 −4);cg_smooth≈cg 高 epoch 略胜。test 柱状图 `softclip_test_fair_vs_reversejoint.png`。

### 8.9 contact guidance（借鉴 SPIDER，师兄改进#2,2026-06-08）

**来源**:SPIDER(facebookresearch,arXiv 2511.09484,物理引导重定向)的 contact guidance —— `exp(-β·dist)` 正奖励,把机器人指尖吸引到**人手在物体上的接触点**(从接触检测来),逐指、flag 门控。补 DexTrack 最大短板:**只跟踪 qpos、无"该接触物体哪儿"信号 → 抓握脆弱、flute 失败**。

**数据(自洽生成,GRAB 无现成接触标注)**:`wuji_pipeline/generate_contact_guidance.py` —— 几何版 detect_contact:每帧把物体 mesh 按位姿变换、算每指尖到表面最近距离 → `<阈值(1.2cm)` 为接触。输出 `contact_flag(T,5)` + `contact_pos_local(T,5,3)`(物体系接触点)+ `contact_dist`,存 `<FPOS>/contact/<traj>_contact.npy`。**坑**:trimesh.closest_point 需 rtree(没装)→ 改 scipy cKDTree 查最近顶点(mesh 密,≈表面);quat 约定**固定 xyzw**(任务 5105/4640 行确认,别自动判)。验证:cubesmall=拇指+食指+中指三指接触(85-88%帧,frame37→末持握);**flute 也有有效接触目标**(f1/f2/f3 碰杆 frame19-135)→ 有望救 flute。

**reward 实现**(独立开关 `CONTACT_GUIDE`,默认关;`CONTACT_COEF`默1.0、`CONTACT_BETA`默30):
- 加载(__init__ ~4083,按 data_inst_tag_list 顺序,**注意 tag 是 tuple 取[0]**)→ `tot_contact_pos_local/flag`;
- compute_reward(~5984)按 progress 取每帧接触点,**用活物体位姿变换到世界系**(`object_pos + quat_apply(object_rot, contact_pos_local)`,随物体举起而动);
- jit reward 项:`+coef · mean_{接触指}[ exp(-β·‖sim指尖−目标接触点‖) ]`,flag 门控;指序 `[th,ff,mf,rf,lf]=ref finger1..5`(和 #4 一致)。BREAKDOWN 加 `contactGuide`。
- 验证:`loaded contact data (1,300,5,3)` + `ref_contact_world=(N,5,3) flag_mean=0.52` + `contactGuide=0.019`(非0,随训练涨)。

**实验**:`wuji_pinall3_cg`(G3,pinall3 + CONTACT_GUIDE,cubesmall 单)vs 已有 pinall3(179/99%)。看点:抓握更稳/更实(指尖真在接触点)+ 可能救 flute(下一步生成 flute 接触 + 试)。

**组合实验(pinall3 + CG + CLIP,2026-06-09)**:批量生成全 27 条接触(26 cubesmall + flute,`generate_contact_guidance.py --save`,xyzw 固定)后起两个:
- `wuji_pinall3_cgclip_multi`(G6,cubesmall 多,加载 22 条 train-split 接触);
- `wuji_pinall3_cgclip_flute`(G1,flute 单,停掉已收敛的 baseline 腾卡)——**contact guidance 救 flute 的关键检验**(显式驱动指尖碰杆 + clip 防反关节)。
两者 clip+cg 均验证生效(contactGuide 非0、接触加载无 MISSING)。注:`add_contact_conditions` 旧基建用 `<traj>_contact_flag.npy` 命名,我们的新数据是 `<traj>_contact.npy`(独立);loading 在 __init__ ~4083 按 data_inst_tag_list 顺序(tag 是 tuple,取 [0])。

### 8.10 GRAB 真值接触(B 版)+ cg_smooth 对照实验（2026-06-10）

A 版接触(8.9)用 wuji 重定向指尖投影到表面,继承重定向误差。**B 版改用 GRAB 原数据真值接触**(`contact['object']` 标注的人手真实接触顶点),生成器 `wuji_pipeline/generate_contact_guidance_grab.py`,全 27 条已生成到 `data/.../contact_grab/`。三处对齐全解决(标签=GRAB tools/utils.py contact_ids、网格=仿真 obj×1.25 同序、帧=重跑 assemble_wuji_reference.fit_crop,corr=1.0;唯 s8_lift corr=0.877)。A-vs-B 实测:flag 97-100%同,接触点差 cubesmall 拇指1.9cm / flute A过检flag 30-40%。**完整记录见 [docs/grab_contact_guidance_plan.md]**;接触点叠在渲染视频里的对比已上 Drive(grab_contact_guidance/)。

**启动 3 个 cg_smooth+B(`CONTACT_SUBDIR=contact_grab` 新开关切换,task.py:4097):**
- **G3 多序列** `..._cg_smooth_grabct`(22 insts,ep10000)→ 对照 **G5 `cg_smooth`(A 版)**,看真值接触对 fair + 反关节有没有用(核心检验)。
- **G4 cubesmall_inspect 单 / G6 flute_pass 单**(cg_smooth+B,ep3000)→ 看 B 两种修正模式。
- 停掉腾卡:pinall3_full / pinall3 baseline / pinall3_cgclip(均已证无用)。日志确认 `contact source dir=.../contact_grab`。

---

相关：[systematic_training_plan.md](systematic_training_plan.md)、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。
