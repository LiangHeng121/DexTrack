# cubesmall 单/多任务 专题

聚焦 `cubesmall` 物体的单序列 + 多任务（generalist）训练：现状、reward 拆解、下一步尝试。

## 1. 现状（2026-06）

### 单序列（都在 `ori_grab_s2_cubesmall_inspect_1`）
| 本体/变体 | best rew | test | 视频 | 评价 |
|---|---|---|---|---|
| allegro | 219 | 216 | `cubesmall_allegro_policy.mp4` | ✅ 100% 举起 |
| wuji offset | 181.93 | 176 | `cubesmall_wuji_policy_offset.mp4` | ✅ |
| wuji no-offset（默认）| 175.47 | 172 | `cubesmall_wuji_policy.mp4` | ✅ 100% 举起 |

### 多任务（26 条 cubesmall 序列，s1–s10）
| 本体 | best(mean rew) | s2_inspect 单测 | 视频 |
|---|---|---|---|
| allegro | 132 | **167** | `all_policies/allegro_cubesmall_multi.mp4` |
| wuji | 53 | **-42** ⚠️ | `all_policies/wuji_cubesmall_multi.mp4` |

> **wuji 多任务在 s2_inspect 上 -42，但单任务同序列 172** —— generalist 远不如 specialist。

### 26 条参考质量（指尖→立方体中心，cube 半边 2.5cm）
参考**本身忠实**（指尖都够到立方体，min 0.9–2.9cm）。但**接触帧占比按动作差异大**：
- `inspect`/`lift`：70–89%（握得久，好跟踪）
- `pass`/`offhand`：12–33%（要松手/递出，手大半时间离物体，难）

26 条参考回放视频在 `render_videos/cubesmall_refs/`。

## 2. 当前 reward 组成（`goal_cond=False`，实际用的）

代码：`compute_hand_reward_tracking_twostages`（`tasks/allegro_hand_tracking_generalist.py:14242`，`@torch.jit.script`）。

```
reward = -0.5·delta_value                                # ① 手腕(全局6DOF)偏离"单个抓握帧"姿态
       - 0.3·(finger_obj_dist + 2·palm_obj_dist)         # ② 指尖/手掌 离物体远 的惩罚
       + goal_hand_rew   = (握住时) -2·‖物体-目标位姿‖    # ③ 握住时才奖励物体跟踪
       + bonus           = (握住且很近) 1/(1+10·goal_dist)# ④ 物体到位的大奖励
       + hand_up         = (物体抬起后) 0.2 / 0.1·a_z     # ⑤ 抬升
```
系数：`rew_delta_hand_pose_coef=0.5`、`rew_finger_obj_dist_coef=0.3`、`hand_pose_guidance_glb_trans/rot/fingerpose=0.6/0.1/0.1`。

### 实测拆解（wuji no-offset，跟踪最好的 env，episodic~172）
| 项 | 每步均值 | 占比/说明 |
|---|---|---|
| ④ **bonus（物体到位）** | **+0.94** | **绝对主导**——握住且物体贴合目标 |
| ⑤ hand_up（抬升）| +0.12 | 抬起>0.1m 帧占 62% |
| ② finger/palm 离物体惩罚 | -0.15 | finger_dist 均 0.30、palm 均 0.11m |
| ③ goal_hand（物体跟踪）| -0.006 | goal_dist 均 **3mm**（物体跟得极准），握住 97% 帧 |
| ① delta_value（手腕姿态）| ~-0.02 | 很小 |

**结论：reward 几乎全靠"握住 + 把物体送到目标轨迹 + 抬起"（④+⑤）。手的姿态像不像人，reward 基本不管。**

## 3. 通俗解释这几个 reward 到底是什么

- **① delta_value（手腕姿态）**：手腕（全局平移+旋转）和"一个固定抓握姿势"差多少。**注意**：原本还有"手指角度"那一项，但代码里被**丢掉了**（`delta_value = delta_glb_value`，14356 行）——所以**手指弯成什么样完全不罚**。
- **② finger/palm 离物体惩罚**：5 指指尖、手掌离立方体越远扣越多。逼着手**贴到物体上**（但不管贴的姿势对不对）。
- **③ goal_hand（物体跟踪）**：**只有当手握住时**，按"立方体离它该在的位置（逐帧人演示的物体轨迹）差多少"给惩罚。差越小越好。
- **④ bonus（物体到位）**：握住 + 立方体很贴合目标（<5cm）时给的**大正奖励**（最高 ~1）。这是分数主来源。
- **⑤ hand_up（抬升）**：立方体被抬起来后，鼓励继续上抬/保持高度。

一句话：**现在的 reward 是"物体中心"的——把方块抓住、按人的物体轨迹搬运、举起来就给高分；至于手指/手腕摆得像不像人，没有奖励项管。**

## 3.5 每项具体怎么算的（通俗 + 公式）

先认识几个"输入量"（每帧、每个 env 都有）：
- `object_pos`：方块**当前**在 sim 里的位置（xyz）。
- `target_pos`：方块**这一帧该在**的位置 = 人演示的物体轨迹（参考 npy 的 `object_transl[t]`）。
- `right_hand_pos`：手掌位置；`ff/mf/rf/th_pos`：4 个指尖位置（食/中/无名/拇，由手的关节角 FK 算出）。
- `hand_pose`：手当前 26 个数（6 全局 + 20 手指角）。
- `grasping_frame_hand_pose`：参考里"抓握那一刻"的**单帧**手姿态（26 个数，固定不变）。
- `flag`（是否握住）= `(4指尖到方块距离之和 ≤ 0.12×指数)` **且** `(手掌到方块 ≤ ~0.12m)` 两个条件都满足 → `flag==2`。

### ① delta_value（手腕姿态偏离）
```
diff = hand_pose - grasping_frame_hand_pose      # 当前手 - 抓握帧手 (26维)
Δpos = |diff[0:3]|求和                            # 全局平移 x,y,z 的绝对差之和
Δrot = |diff[3:6]|求和                            # 全局旋转 3 维的绝对差之和
Δfinger = |diff[6:26]|求和                        # 手指 20 角的差 —— 算了但被丢弃!
delta_value = 0.6·Δpos + 0.1·Δrot                # 手指项没进来
① = -0.5 · delta_value
```
**通俗**：手腕的位置和朝向，离"那个抓握姿势"差多少，加权求和再取负。**手指弯成啥样算出来了却没用**（所以不约束手指像不像人）。注意目标是**一个固定帧**，不是逐帧跟人。

### ② finger/palm 离物体惩罚
```
palm_dist = ‖object_pos - right_hand_pos‖         # 手掌到方块中心 (超 0.5m 截断)
finger_dist = Σ_{ff,mf,rf,th} ‖object_pos - 指尖‖  # 4 指尖到方块中心 之和 (超 0.6×指数 截断)
② = -0.3 · (finger_dist + 2·palm_dist)
```
**通俗**：量 4 个指尖、手掌到方块中心的**直线距离**，加起来（手掌权重翻倍），越远扣越多。逼手**贴到方块上**——但只看"近不近物体"，不看"姿势对不对"。

### ③ goal_hand（物体跟踪，只在握住时）
```
goal_dist = ‖target_pos - object_pos‖             # 方块 实际 vs 该在的位置
③ = 握住(flag==2) ? (-2·goal_dist) : 0
```
**通俗**：**只有手握住方块时**，才看"方块离它这一帧该在的位置差多少"，差越大扣越多。没握住 → 这项 0 分。这就是"按人的物体轨迹搬运"的约束。

### ④ bonus（物体到位，主分来源）
```
④ = (握住 且 goal_dist ≤ 0.05m) ? 1/(1+10·goal_dist) : 0
```
**通俗**：握住 **且** 方块离目标 5cm 以内时，给一个**大正奖励**——完美贴合时≈1，稍微飘开就快速衰减。实测它贡献 ~+0.94/步，是分数的大头。

### ⑤ hand_up（抬升）
```
lowest = object_pos[2]                            # 方块高度
hand_up = (lowest ≥ 阈值1 且 握住) ? 0.1·a_z : 0   # a_z = 动作里的上抬量
hand_up = (lowest ≥ 阈值2 且 握住) ? 0.2 : 上一行  # 抬够高直接给 0.2
```
**通俗**：方块被抬过阈值1（且握着）就按"上抬动作"给奖励、鼓励继续往上；抬过阈值2直接给固定 +0.2。鼓励**举起来并保持**。

## 4. 诊断（回答两个"为什么"）

- **为什么手和人手差别大**：reward **没有逐帧"手姿态跟踪"项**（手指项被丢、手腕项只对单帧、且系数小）。手姿态只靠 kinematics-bias（动作=参考+残差）软约束，策略可自由偏离去找"能搬动物体"的任意抓法。
- **为什么多任务 cubesmall 学不好**：① reward 是物体中心，26 条各分 ~1500 env（单任务 22000）数据稀；② `pass/offhand` 低接触序列信号差；③ wuji 5 指更难协调，且**没有"模仿人手"的强锚**帮 generalist bootstrap。

## 5. 接下来尝试（按优先级）

1. **★ 逐帧"手指+手腕姿态跟踪"**：把被丢的 `手指角度差` 项加回来，且目标从"单个抓握帧"改成**逐帧参考**。给策略强模仿锚——预期最能改善"像人手" + 帮 generalist 收敛。改动小（reward 函数）。
2. **指尖位置跟踪 vs MANO**：奖励 wuji 5 指尖匹配**逐帧人手指尖位置**（重定向到人手指尖仅 ~9mm，目标可靠）。把手钉在演示上。
3. **接触一致性奖励**：只在参考"该接触"的帧奖励接触、"该松手"的帧奖励松开——比一刀切的 finger-dist 惩罚更适合 pass/offhand。
4. **多任务专属**：先去掉 pass/offhand 低接触序列（或降权）让 generalist 先学好 inspect/lift；或多任务也加 #1 的模仿锚。

**建议先做 #1**（逐帧手姿态跟踪），开一个 cubesmall 单任务对照训练验证"像人手"+reward 是否提升，再推广到多任务。

相关：[systematic_training_plan.md](systematic_training_plan.md)、[wuji_retargeting_and_visualization.md](wuji_retargeting_and_visualization.md)。
