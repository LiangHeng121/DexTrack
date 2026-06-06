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
