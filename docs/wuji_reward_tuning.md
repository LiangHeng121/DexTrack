# wuji 奖励调参:闲指保持(idle_hold)+ 接触吸附(cg-beta)

在 B2_softclip 配方(pinall3 + contact_guide(B2) + HAND_EMA + ACTION_RATE + softclip)基础上,解决两个"看着不自然"的问题。单序列 cubesmall_inspect、ep1000 验证,fair 基准 = B2_softclip **180.6**。

---

## 一、idle_hold:闲指保持姿态

**问题**:cube 是 thumb+index+middle 三指捏握,**ring/pinky 不接触(contact_flag 0%)、约束最弱 → 乱甩**(实测偏离参考:ring 40°、pinky **68°**;参考里它们几乎不动)。contact_guide 只拉接触指,管不到闲指。

**思路演进(每步都有教训):**

| 版本 | 做法 | 问题 |
|---|---|---|
| v1 指尖位置(world) | `-coef·‖tip−ref_tip‖` | 3D 指尖 ← 4 关节**欠定**:tip 对了但中间关节摆歪 → 不自然(无名指尤其);且 world 系依赖手掌、和抓取耦合 |
| v2 DOF(关节角) | `-coef·‖Δq_finger‖` | 直接约束整根手指姿态、与手掌解耦。**但无界距离惩罚**:Δq 早期 ~1.7 rad,penalty(coef1→1.7)远超单步跟踪 reward(~0.3)→ **像 huber 一样压垮学习**(coef 0.25 都崩) |
| **v3 有界 exp 奖励(最终)** | `+coef·(1−flag)·exp(−β·‖Δq_finger‖)` | 贴参考(Δ→0)给≈coef 正奖励、远了≈0 **不爆**。RL 自会去刷这份 bonus → 把闲指保持在参考姿态 |

**最终实现**(task.py,env 变量):
- `IDLE_HOLD_COEF`(>0 开启)、`IDLE_HOLD_DOF=1`(关节角,推荐;0=指尖)、`IDLE_HOLD_BETA=3`(exp 衰减,rad)。
- 公式(逐指求和,fair 不含此项):
  ```
  idle_hold_value = Σ_{f=th..lf} (1 − contact_flag_f) · exp(−beta · ‖delta_qpos[该指4关节]‖₂)
  reward += idle_hold_coef · idle_hold_value
  ```
- `(1−flag)` 自动屏蔽接触指(交给 contact_guide),只压闲指 → 与接触引导互补。`delta_qpos = 达成DOF − 参考DOF`。

**coef 扫描结果(都 ep1000,闲指=ring/pinky):**
| coef | ring | pinky | fair峰 | 备注 |
|---|---|---|---|---|
| 0(基线) | 40° | 68° | 180.6 | 乱甩 |
| 0.1 | — | — | 181.9 | |
| **0.25** | 23° | 24° | **184.9** | fair 最高、闲指够好 |
| **0.5** | **5°** | **15°** | 180.5 | 闲指最自然(≈参考)、fair≈基线 |
| 1.0 / 2.0 | — | — | 崩(负) | bonus 太大、退化解 |

**结论**:coef 必须低(≤0.5);**有界 exp 几乎零 fair 代价**。coef=0.5 闲指最自然,coef=0.25 fair 略高。渲染 `report_videos/idlehold_check/`。

---

## 二、cg-beta:减接触吸附

**问题**:接触指"太吸附到接触点、不自然"。cg 奖励 `coef·exp(−beta·d)`,**接触点处拉力梯度 = beta**;beta=30 太陡 → 手指猛吸上去。

**调法**:beta 往下扫(coef 保持 1.0)。
| beta | 接触指尖→接触点距离(cm) | fair峰 |
|---|---|---|
| 30(基线) | th1.2 / ff2.8 / mf1.0 | 180.6 |
| **8** | th2.2 / ff3.8 / mf2.1(**更松**) | **186.0** |
| 15 | — | ~140(单序列没收敛好,异常) |

**结论**:**beta=8 接触指更松(吸附轻)、fair 反而更高(186)** —— 拉力缓和、策略不和它较劲。渲染 `report_videos/cgbeta_check/`。

---

## 三、最终合并配方

**B2_softclip + cg beta=8 + idle_hold(DOF exp,coef=0.5,beta=3)**。单/多序列在跑(`..._beta8_idle05_single/multi`)。完整 env:
```
RELAX_PALM=1 FIX_FINGER5=1 FINGER_POS_REW=1 FINGER_POS_COEF=1.0 PALM_POS_REW=1 PALM_POS_COEF=1.0 \
CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=8 CONTACT_SUBDIR=contact_grab2 \
HAND_EMA_COEF=0.4 ACTION_RATE=1 ACTION_RATE_COEF=0.0005 SOFT_LIMIT=1 SOFT_LIMIT_COEF=0.5 \
IDLE_HOLD_DOF=1 IDLE_HOLD_COEF=0.5 IDLE_HOLD_BETA=3.0 \
WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data
```

**通用教训**:接触/姿态类引导项,**用有界奖励 `exp(−β·d)`,别用无界距离惩罚 `‖d‖` / huber**(后者早期偏差大就压过主跟踪 reward → 崩)。这条对 cg(本就 exp)、idle_hold(改成 exp)、以及之前失败的 huber 都成立。
