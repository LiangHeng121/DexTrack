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
| **#5 palm-lock** | `RELAX_PALM=1 FIX_FINGER5=1 PALM_POS_REW=1 PALM_POS_COEF=5.0 GLB_TRANS_COEF=2.0 GLB_ROT_COEF=0.3` | `..._palmlock` / `grab_*_wuji_palmlock` | **★ ep312 突破**:**90% 持举 + palm 完美在后**(手腕 0.7cm,98%env<1cm)!手指自己找到鲁棒握(~11°/关节,非参考脆捏)。fair@0.22=88。**达成"palm 后 + 举起"**(见 §8.5)|
| **pinall** | `...PALM_POS_COEF=2.0 GLB 1.0/0.2 FINGER_POS_COEF=1.5 BOOST_FINGERPOSE=1 FINGERPOSE_COEF=0.3` | `..._pinall` | ep200:手指被压到 ~4°/关节(贴参考)但**举 1%** ❌——**钉手指=反方向**(把手指拽回脆参考,拆了鲁棒握)|
| **pinall2** | `...PALM_POS_COEF=2.0 GLB 1.0/0.2 FINGER_POS_COEF=1.0`（fingerpose 不 boost=0.1）| `..._pinall2` | 🏃 2026-06-07 单 GPU7:松手指惩罚(关节0.1+指尖1.0),找"能举+手指尽量贴参考"的折中点 |

> **打分尺度（重要）**：test 默认=原始 reward(flag@0.12)对 palm-back 不公平(见 §8.4);公平加 `PALM_GRIP_THRES=0.22`(独立开关,只放宽阈值)。**最可靠是举升%**——且用**"参考峰值帧仍举着"**(真持举),不要用 `z.max>5cm`(会把碰飞算进去,虚高:palm-lock 98% vs 真持举 90%)。
> **reward 分解工具**：活跃函数末尾 `REWARD_BREAKDOWN` print(帧 1/150 快照;临时改成每步 gate 可累加成整集绝对值)。实测 palm-lock 尺度整集:bonus +85,负的几乎全在手指(fingerJoint -111、fingerPos -64),palm 几乎不花钱(palmPos -2)。
> **监控**：`monitor_pinall_trend.sh`(高频/GPU6,每 60s 测 best 的 fairRew/举升%/fingerErr/wristErr → `/tmp/pinall2_trend.log`)。
> 注:#1/#2+#3 多已停(弃);pinall2 停了 #2+#3 多腾 GPU7。palm-lock 单/多、pinall 单、#4 多、baseline 多都在跑。

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

---

相关：[systematic_training_plan.md](systematic_training_plan.md)、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。
