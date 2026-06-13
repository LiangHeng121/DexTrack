# GRAB 真值接触 → contact guidance（待办，先按下不表）

> 状态：**仅记录，未实现**。等 pinall3_cg / cg_smooth 实验结论清楚后再决定要不要做。
> 关联：当前的 contact guidance 实现见 `isaacgymenvs/wuji_pipeline/generate_contact_guidance.py`（A 版，post-retarget）。

## 动机

当前 contact guidance（cg / cg_smooth 实验在用的）的接触点来自 **wuji 重定向后的指尖**：
- `generate_contact_guidance.py` 读 FPOS 的 `link_key_to_link_pos`（= 对 wuji q26 参考做离线 FK 得到的 wuji 五指尖），逐帧投到物体网格最近表面点（<1.2cm 判接触）。
- **缺点**：接触点继承重定向误差。wuji 手指长、和人手比例不同，重定向可能把指尖放偏 → 检测出的"接触点"也偏。等于在引导策略去贴合"重定向认为的接触"，而非真实人手抓握接触。

## 关键发现：GRAB 原始数据集全在本地，且有真值接触标注

- 路径：`/home/liangh/DexTrack/GRAB/unzipped/grab/s1..s10/<obj>_<action>.npz`
- 每序列 npz keys：`gender, sbj_id, framerate, obj_name, body, lhand, rhand, object, table, contact, n_comps, n_frames, motion_intent`
- `rhand` = {params: global_orient/hand_pose/transl/fullpose, vtemp}（MANO 右手）
- `object` = {params, object_mesh}
- **`contact` = {body, object, threshold}** ← 真值接触：
  - `contact['object']`：**(n_frames × 53890 物体顶点) int8**，值 = 碰到该物体顶点的 **SMPL-X 身体部位 ID**（0=无接触）。例 s8/cubesmall_inspect_1：唯一值 `[0,41,42,43,46,52,55]`，880 帧、43% 帧有接触。
  - `contact['body']`：(n_frames × 10475 身体顶点) int8。
  - threshold = 2e-05。
- 即 GRAB 直接给了"每帧、每个物体顶点、被哪根手指接触"，比 MANO FK 更权威，**完全不依赖重定向**。

## 方案（与现有 jit reward 项同构，只换接触点来源）

对每帧、每根手指：取它接触的那些物体顶点 → 物体局部坐标下求接触区域（质心/点集）→ reward 引导 wuji 对应指尖贴该点（用实时物体位姿变换）。jit 项结构不变，只是 `ref_contact_local` 的来源从"wuji 指尖投影"换成"GRAB 真值"。

## 优点（相对 A 版）

1. **脱离重定向误差**：接触点是人真握住时的物理接触。
2. **接触 flag 是真值**：哪根指/哪几帧接触，GRAB 实测，不靠几何阈值猜。
3. 全本地，不用 cephfs，不用自己跑 MANO FK。

## 三个要对齐的工程点（实现前需确认）

1. **SMPL-X 标签 → wuji 五指映射**：`[41,42,43,46,52,55]` 这些 part ID 对到 wuji finger1–5（th/ff/mf/rf/lf）。查 SMPL-X 右手分段表。（先确认到底是不是 5 指 + palm，标签数对不对得上。）
2. **物体网格/坐标对齐**：GRAB object_mesh（53890 顶点）vs 仿真 `meshdatav3_scaled/<obj>/coacd` 网格——可能缩放/规范位姿不同。接触顶点的物体局部坐标要映射到**仿真物体局部系**（同一物体，确认 scale/canonical 对齐）。
3. **帧对齐**：GRAB 880 帧 @ 原帧率 vs 训练数据 nf_300（降采样）。接触要按**和重定向相同的降采样**对齐到 300 帧。

## 没解决的核心取舍（仍在）

几何不匹配：即便接触点是人手真值，wuji 长手指碰同一点可能需要别扭姿态。缓解：**用真值当 flag + 较软的位置目标**（甚至只用 flag 决定"哪根指/哪帧该接触"，位置仍允许 wuji 自己的几何）。但总比错的 wuji 投影目标强。

## 下一步（真要做时）

先做"可行性确认"（只读数据、不改代码）：① SMPL-X 标签确属 5 指；② GRAB cubesmall 网格和仿真网格能对齐（scale/canonical）；③ 帧降采样规则。确认能对齐后，照 `generate_contact_guidance.py` 改一个 GRAB 真值版生成器，输出同样的 `<traj>_contact.npy`（contact_flag + contact_pos_local），reward 端不用动。

## ★ 可行性确认结果（2026-06-10，只读数据，3/4 干净，1 个待解）

| 对齐点 | 结论 |
|---|---|
| **接触数据** | ✅ `contact['object']` (T×53890 int8) 直接索引 GRAB 网格顶点；object_mesh = `tools/object_meshes/contact_meshes/<obj>.ply`（本地有），object.params = transl(axis-angle global_orient) per帧 |
| **标签→手指** | ✅ **权威映射在 `GRAB/tools/utils.py:166` `contact_ids`**（不是 base-40，那是早先猜错的）：R_Index=41/42/43、R_Middle=44/45/46、R_Pinky=47/48/49、R_Ring=50/51/52、R_Thumb=53/54/55、R_Hand(掌)=22。→ reward 列 [th,ff,mf,rf,lf]=[{53,54,55},{41,42,43},{44,45,46},{50,51,52},{47,48,49}]，palm 22 丢弃。nearest-wuji 标定印证（55→thumb、43→index、46→middle 全对上）。 |
| **物体网格** | ✅ **trivial**：GRAB cubesmall.ply 与仿真 `meshdatav3_scaled/.../<seq>.obj` **都是 53890 顶点、都中心在原点、仿真=GRAB×1.25 均匀缩放、同朝向** → 同一网格只差 scale → **接触索引 i 直接对应仿真网格顶点 i**，接触点搬过来无需变换 |
| **帧对齐** | ✅ **已解决（确定性，可完美复现）**：转 wuji 数据时就做过，规则在 `wuji_pipeline/assemble_wuji_reference.py`。见下。 |

**帧对齐（已解决）**：wuji 300 帧不是均匀降采样，而是 `assemble_wuji_reference.py:fit_crop()`（L43-68）用**物体角速度曲线匹配**在 GRAB 776 帧里找最优裁剪段 `[c0,c1]`，再 `grab_idx = np.linspace(c0, c1-1, 300)`（L98）把每个 wuji 帧映射回一个 GRAB 分数帧。`[c0,c1]` 没存盘只 print，但**完全确定性**——输入只是 GRAB `global_orient` 角速度 + allegro ref `object_rot_quat` 角速度（`isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_<seq>_nf_300.npy`），**都不依赖 wuji 中间文件**。接触生成器里**重跑同一 `fit_crop` → 同一 grab_idx**，每 wuji 帧取 round(grab_idx[i]) 的**最近** GRAB 帧（接触标签离散、不能插值）的接触即可完美对齐。

**结论：4/4 全通，可行。** 网格/标签/数据/帧对齐全解决。

## ★★ 已实现并验证（2026-06-10）：`isaacgymenvs/wuji_pipeline/generate_contact_guidance_grab.py`（B 版）

生成器写好并在 s2_cubesmall_inspect_1 上跑通。流程：① 读 GRAB `contact['object']` + `global_orient`；② 重跑 `fit_crop` 得 crop GRAB[200:499]→300（**normMSE=0.0000、角速度 corr=1.0000，帧对齐完美**）；③ 每 wuji 帧取 round(grab_idx) 的最近 GRAB 帧，按 contact_ids 分 5 指 → object-local 接触点 = 仿真顶点 sv[被接触idx]（已验证 = GRAB ply×1.25 同序，无需变换）取质心；④ 存格式同 A 版。用法：`python wuji_pipeline/generate_contact_guidance_grab.py --traj <traj> --save`，输出到 `data/.../contact_grab/`。

**A 版(wuji 投影) vs B 版(GRAB 真值) 实测对比（s2_cubesmall_inspect_1）：**
- **flag（哪根指/何时接触）97–100% 一致**：拇/食/中接触、无名/小指从不接触，两版判断几乎相同 → 真值**不改接触时序**。
- **接触点位置差异（都接触的帧）**：拇指 **1.9cm**、食指 **1.4cm**、中指 0.3cm。cubesmall 半边长 2.5cm，所以**拇指 1.9cm = 放到立方体明显不同的区域** → 这就是重定向误差：wuji 长拇指投影点 ≠ 人手拇指真实接触点。**B 版价值 = 把拇指/食指目标纠正 1–2cm**（中指本就准）。
- 可视化 `wuji_pipeline/out/contact_AB_compare_s2_cubesmall_inspect_1.png`：每指 A(×)/B(○)簇靠近但偏移，拇指/食指可见、中指重合。

**flute s2_flute_pass_1 也跑了 B（验证不同物体）**：网格同序 ×1.25（25629 顶点，误差0），帧对齐完美。A-vs-B **差异轴和 cubesmall 不同**：
- 点位差更小（拇1.3/食0.7/中0.9cm）；
- 但 **flag 差更大：A 比 B 多检 30–40% 接触帧**（中指 A 111f vs B 77f、拇 A 62f vs B 45f）。原因：A 的几何阈值 <1.2cm 在**细长笛**上过检——指尖悬在表面附近就误判接触；B 真值只在人真碰到时算。

**★ 批量生成完成（2026-06-10）**：多任务集全部 **27 条**(s1–s10 cubesmall 各 inspect/lift/pass/offhand + s2 flute_pass)已生成 B，存 `data/GRAB_Tracking_PK_WUJI_FPOS_v1/contact_grab/`。**26/27 帧对齐完美 corr=1.0**；唯一例外 **s8_cubesmall_lift corr=0.877**（fit_crop 角速度匹配不完美，该序列旋转曲线平缓/有歧义，接触时序可能略偏——用前先单独看一眼）。

**切换开关（已加）**：`task.py:4097` 原硬编码 `'contact'`，现加 `CONTACT_SUBDIR` 环境变量。跑 cg+B 实验只需 `export CONTACT_SUBDIR=contact_grab`（默认 `contact`=A 版，不动）。cg 其他 flag(CONTACT_GUIDE/COEF/BETA)不变。

**★ 已启动对照实验（2026-06-10）：cg_smooth + B（真值接触）3 个 run**
- **G3 多序列** `grab_multiple_wuji_pinall3_cg_smooth_grabct`（22 insts，MAX_EPOCHS 默认 10000）—— 直接对照 **G5 `cg_smooth`（A 版，同 cg_smooth 配置但用 wuji 投影接触）**，看真值接触对 fair + 反关节有没有用。
- **G4 单序列 cubesmall_inspect** + **G6 单序列 flute_pass**（cg_smooth+B，MAX_EPOCHS=3000）—— 分别看 B 的两种修正模式（cube=点位、flute=flag）。
- 配置 = `PINALL3 + CONTACT_GUIDE=1 CONTACT_COEF=1.0 CONTACT_BETA=30 + HAND_EMA_COEF=0.4 + ACTION_RATE=1 ACTION_RATE_COEF=0.0005 + CONTACT_SUBDIR=contact_grab + WUJI_DATA_DIR=$FPOS`。日志确认 `contact source dir=.../contact_grab`、`loaded contact data 22/1 insts`。
- 为腾卡停掉：pinall3_full、pinall3 baseline、pinall3_cgclip（均已证无用）。**重点看 G3(B) vs G5(A) 的 fair 曲线 + 反关节 env%。**

**结论（值不值得换）**：B 修正两类"启发式误差"，哪类为主取决于物体——① **接触点位置**（cubesmall 拇指 1.9cm，wuji 长指投影偏）；② **接触 flag/时序**（flute：A 几何阈值过检 30–40%）。两类都是 A 版（重定向投影 + 几何阈值）的固有误差，B 用真值修掉。**差异整体温和**（点 1–2cm、flag 90–97% 一致），是精修不是革命；但**原理更对、生成零成本（reward 端不动）**。建议：① 批量给多任务集生成 B；② 空 GPU 时跑一组对照（cg+B vs cg+A，看 fair 上升速度 + 反关节 env%）——尤其验证"B 的真实接触点是否减轻 cg 把拇指拽进反关节"（cg 反关节根源是把指尖往接触点卷，接触点更准也许卷得更合理）。

---

## 另一条 TODO：调现有 cg 的参数（加速 reward 但不引反关节）2026-06-10

**实测的 cg 现状（多序列 fair）：**
- **价值 = 加速早期 reward**：ep100–1000 cg/cgclip/cg_smooth 的 fair 都明显快于 pinall3（contact_guide 助攻早期收敛）。但**不抬上限**——pinall3 慢但给够 epoch 反超（pinall3 best fair 101 > cgclip 86 > cg 79 ≈ cg_smooth 78，但 pinall3 训到 ep2973、cg 系才 ~ep970）。clip/smooth 对 fair 无正贡献。
- **代价 = 引反关节**：contact_guide 把手指往接触点用力卷 → 卷过头进反关节。**cg 自己就会**（s10_pass 冲到 −2.61、s2_inspect −1.28、s6_pass −1.33；越界出 URDF −0.47 一大截）。**smooth 是帮凶放大频率**（s2_inspect <−1.0 的 env 占比 9%→49%）。根源是 **cg 本身**，不是 smooth。

**调参目标 = 两者兼得（快 + 不反关节）。两个轴：**
1. **治反关节**：上 `SOFT_LIMIT` 软惩罚（已实现，20 指节双向 quadratic，见 [docs/cubesmall_single_multi.md §8.8b]）。**cg-only 也该上**（不只 cg_smooth）。优于硬 CLIP_DOF（clip chatter 12.5、慢 25%、对 reward 无益）。
2. **更好地"引导"（保住/加强加速）**：现 cg 用 `exp(−beta·dist)`、beta=30 → 有效范围仅 ~3cm，**只奖励"已贴近"、远处梯度≈0 不引导**（这也是 cg 在已解序列中性、难序列帮不上的原因）。要它真从远处把手指引到接触点：**调小 beta（如 10–15，范围 ~7–10cm）**，或**改成 SPIDER 的线性罚 −‖tip−ref‖**（远处仍有恒定梯度，从零引导）。coef 也可调。

**注意**：beta 调小让 cg 更"引导"可能**加剧反关节**（更强地把手指往接触点拽）→ 必须**和 SOFT_LIMIT 一起调**，找"加速够 + 反关节被压住"的平衡点。建议网格：beta∈{15,30} × {exp, linear} × SOFT_LIMIT_COEF∈{0.3,0.5,1.0}，看 fair 上升速度 vs 反关节<−1.0 env%。

**当前在跑**：`wuji_pinall3_cg_smooth_softclip`（pinall3+cg+smooth+SOFT_LIMIT 0.5，G1）——先验证软惩罚能把反关节压下去且不掉 fair,再做上面的 beta/linear 网格。

## ★★★ 重大修正（2026-06-10）：质心 B 有 bug → B2（真值flag + 片内最近顶点）

**实测 B(质心) multi 在 cubesmall 上反而落后 A ~24 fair(ep778)。** 深挖：
- B 的"点"= 接触片**质心**;cube 整个面被人拇指碰(2399 顶点),质心飘到面中心、离 wuji 指尖 ~2cm。**那 2cm 是质心 artifact,不是人碰了不同地方** —— wuji 指尖就落在人接触片**里面**(到片内最近被碰顶点仅 0.4cm)。
- A 的投影点本来就 ≈ 人接触区内离指尖最近点(差 <1mm)。所以 cube 上 B<A = 质心把拇指目标拽到面中心、和 reference 抓握冲突,**不是真值接触有害**。
- 接触分"点"+"flag":**点**——A 投影已够准,真值无增量;**flag**——cube 上 A 也准(97-100%,只漏起始几帧),只有 **flute 类细长物体 A 几何阈值(<1.2cm)过检 30-40%**(中指过检 34 帧)真值 flag 才有用。
- **→ 真值接触的唯一真实价值 = 修 flute 类物体的过检 flag。**

**B2(正确实现)= 真值 flag + 点取「人接触片里离 wuji 指尖最近的顶点」**(≈A 投影点,对齐 SPIDER)。生成器 `wuji_pipeline/generate_contact_guidance_grab2.py` → `contact_grab2/`,27 条已生成。校验:B2点 vs A点 ≤0.13cm(cube)/≤0.51cm(flute),B2 flag==真值 300/300。

**B2 对照已启动**(`CONTACT_SUBDIR=contact_grab2`):G3 `_multi_B2`、G4 `_cube_B2`、G6 `_fluit_B2`(单 ep1000)。预期 cube 上 B2≈A、flute 上 B2>A → 干净隔离"真值价值=修 flag"。

**SPIDER 确认用人手(MANO)接触**(`contact_left/right` from trajectory_keypoints.npz 逐指,ik.py:231),点用"指尖↔物体site实时贴合"可达点(mjwp_eq.py:639)不是固定人手点 —— B2 设计与之一致。质心 B(第一版)= 我们自己的错误,A(wuji投影)反而一直够好,差距只在 flag。
