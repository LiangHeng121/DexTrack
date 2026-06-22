# MIGRATION HANDOFF —— 新服务器先读这一篇

> 这是从旧 8×A100 机器(`nebula101`, `/home/liangh/DexTrack`)迁移过来的交接文档。
> 浓缩了之前所有 memory + 已完成的工作 + TODO + 操作纪律。**先通读本篇，再看引用的细节 doc。**
> 当前活跃工作线 = **wuji 手 × mjlab(MuJoCo) 动作跟踪**，不是 Isaac Gym 侧。

---

## 0. 一分钟上手

```bash
# 一键安装(克隆两仓 + 拉 HF 数据/ckpt + 装 pixi 环境 + 路径自适配)
DEXTRACK_ROOT=/home/liangh/DexTrack bash migrate_install.sh   # 见仓库根 migrate_install.sh
# 验证
cd $DEXTRACK_ROOT/wuji-mjlab && pixi run list-envs | grep -i Contact
# 训练(当前最优路线:3 物体 generalist)
CUDA_VISIBLE_DEVICES=0 pixi run train --task WujiHand_Tracking_3Obj_CGSmooth_Contact --env.scene.num-envs 8000
```
- **代码**:`wuji-mjlab/`(自己的 fork,分支 `dextrack-tracking`;**不要把它提交进 DexTrack 主仓**,是独立 git 仓)。
- **数据/ckpt 在 HF**:`liang12121/dextrack-wuji-mjlab-assets`(私有 dataset,需 `hf auth login`)。
- **🔑 凭证(HF/wandb/GitHub token)**:**不在本仓**(DexTrack/wuji-mjlab 都是公开仓,token 不能进)。token 全部放在**私有 HF 数据集里的 `SETUP_SECRETS.md`**。接入步骤:
  1. 先用 **HF token** 登录(bootstrap;token 见迁移交接对话,或 HF 账号 Settings→Access Tokens):`hf auth login`
  2. 下载凭证:`hf download liang12121/dextrack-wuji-mjlab-assets SETUP_SECRETS.md --repo-type dataset --local-dir .`
  3. 按 `SETUP_SECRETS.md` 里命令配好 wandb / GitHub(`migrate_install.sh` 也会顺带拉下来)。
  > token 若担心泄露,迁移后在 HF/wandb/GitHub 各**轮换(regenerate)**一次。
- 渲染需 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。
- **不要 pip 装进 pixi env**(要加包改 `pixi.toml`)。

---

## 1. 当前最重要的结论(2026-06-21,本轮核心成果)

详见 `docs/mjlab_migration_plan.md` §"★★ 接触门控 + 多物体 generalist"。要点:

1. **接触门控(contact-gating)是抓握 reward 的正解**。距离 flag(`finger_dist≤0.6 且 palm_dist≤0.22`)会被"趴地板悬停"误触发→策略趴地刷 fair 不举物体。改用**真实接触**(`n_finger_contacts≥2`,读 mujoco-warp 接触对)替换 `flag==2` 门控 `object_pos_tracking`+`object_inplace_bonus`。reward_mode 加 `_contact` 后缀触发;`fair_reward_metric` 也已改成接触门控。

2. **多物体 generalist > 单物体专项**(成功率 max_z vs 参考峰):
   | run | cube | cup | apple |
   |---|---|---|---|
   | 单物体专项 | 6/6 | 6/8 | **0/8**(全趴地) |
   | **3obj generalist** | 6/6 | **7/8** | **3/8** |
   - 3obj 在每个物体上 ≥ 对应专项。**apple 用 kp×1 完全能举**(3obj 证明,s2 举到 0.40);apple 专项失败是缺多物体迁移(学轻 cube/cup 抓法迁移到重 apple),**不是抓力不够,不用加 kp**。
   - → **主推多物体 generalist,放弃单物体/单序列专项**。

3. **判据只信 max_z(离线 rollout 量物体高度) vs 参考峰**,不信 fair——fair(任何门控)会被趴地刷分,且聚合 fair 会把"哪个物体/序列失败"抹平。**不要用绝对阈值 0.2**(很多参考本身只举到 0.1)。

4. **eval/render 锁序列时必须 park 非激活物体**(3obj):monkeypatch `_resample_command` 只写激活物体会让另两个堆在手边→污染(cube 假"抛飞")。park 用 `_park[j]` 的 pos + 单位四元数。训练侧的真 `_resample_command` 本就 park,不受影响。

---

## 2. 仓库 / 数据 / ckpt 位置

- **DexTrack 主仓**:`git@github.com:LiangHeng121/DexTrack.git`(分支 main)。含本文档、`docs/`、Isaac Gym 侧代码、`wuji_pipeline/` 重定向管线、对比视频 `mjlab_*.mp4`。
- **wuji-mjlab fork**:`https://github.com/LiangHeng121/wuji-mjlab.git` 分支 `dextrack-tracking`(remote `mine`);上游 `origin`=`wuji-technology/wuji-mjlab`。所有 tracking 改动在这分支。
- **HF dataset** `liang12121/dextrack-wuji-mjlab-assets`(私有):镜像了 DEXTRACK_ROOT 相对路径——
  - `isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1/`(WUJI 重定向参考 + `contact_grab2/` B2 接触)
  - `assets/obj_type_to_obj_feat.npy`(256-d 物体 latent)
  - `GRAB/unzipped/tools/object_meshes/contact_meshes/`(GRAB 物体 mesh,grab_object_cfg ×1.25 加载)
  - `wuji-mjlab/logs/rsl_rl/wuji_tracking/<run>/`(每 run 的**最新+最好 ckpt** + tb events)
- **代码硬编码 `/home/liangh/DexTrack`** 共 2 文件(`env_cfgs.py`、`grab_object_cfg.py`);若 DEXTRACK_ROOT 不同,安装脚本会 sed 改写。
- Isaac Gym 侧的大数据(`assets/meshdatav3_scaled` 3.5G、`rsc/objs` 5.9G、GRAB/TACO 等)**没上 HF**,需要时按 `README.md` 的 Data 节从 OneDrive 重新下;mjlab 工作用不到。

---

## 3. 所有 memory(按主题归并)

### 3.1 环境 / 服务器
- **旧机环境**(`[[dextrack-server-setup]]`):8×A100,conda env `dextrack`(torch2.4.1+cu121,py3.8)。Isaac Gym import 报 `libpython3.8.so` → 已加 conda activate 钩子把 `$CONDA_PREFIX/lib` 进 `LD_LIBRARY_PATH`。**新机若跑 Isaac Gym 侧需重做此钩子**。mjlab 侧用 pixi(py3.11),无此问题。
- **wuji 手本体**(`[[wuji-hand-integration]]`):fly 悬浮手,**5指×4关节 + 6全局DOF = 26 DOF**(异于 Allegro/LEAP 的 4指16DOF,代码处处假设4指)。重定向链路:GRAB MANO→21关键点(manopth)→`wuji-retargeting` Retargeter→qpos。细节 `docs/wuji_integration_plan.md`。

### 3.2 数据 / 接触 guidance
- **GRAB 真值接触 B2**(`[[grab-contact-guidance-plan]]`):接触点来源演进 A(指尖投影)→B(质心,有 bug 飘 2cm)→**B2(真值 flag + 片内离指尖最近顶点≈A点)**。生成器 `wuji_pipeline/generate_contact_guidance_grab2.py`→`contact_grab2/`。真值的唯一价值=修 flute 类细长物体的过检 flag;cube 上 A 就够准。**B2 mesh bug**:必须用 GRAB 原 ply×1.25(不是被抽稀的仿真 mesh)。细节 `docs/grab_contact_guidance_plan.md`。
- **多物体 generalist 配置**(Isaac Gym 侧):`docs/wuji_multiobj_generalist.md`(9物体/209轨迹,bowl 因不在优化字典被排除)。

### 3.3 训练 / 评估踩坑
- **磁盘满=全崩**(`[[recurring-crash-was-disk-full]]`):mjlab 多 run 同时死,**真因是磁盘 100% 满**(net-v4 ckpt 0.2-1.2GB × save_interval 太密),不是外部 kill。**再遇全崩先 `df -h .`**。已改 `save_interval 50→250` + 清理脚本(每 run 留最高 iter)。
- **wandb 截断→用 tensorboard**(`[[wandb-truncates-use-tensorboard]]`):fair/reward 全程曲线读本地 tb(`Episode/fair_reward` 标签,`event_accumulator` 必须 `size_guidance={'scalars':0}` 否则降采样)。wandb 滞后/截断,别信。
- **kill 误杀训练**(`[[kill-pattern-matches-training]]`):清进程**只用 PID/pgid**,宽泛 `pkill -f`/`grep -E` 会撞训练 cmdline(含 `_test`/任务名)误杀。曾一秒 kill 5 个训练。
- **restart 配置 diff 纪律**(`[[restart-config-diff-discipline]]`):重启 run 别从 `/proc/PID/environ` grep 子集重建(会丢 env var);用 doc 里记的完整命令。
- **多序列 test 的 kinematics_only 写死**(`[[multi-test-kinematics-only-hardcoded]]`,Isaac Gym 侧):`..._multiple_test.sh` L195 曾写死 `kinematics_only=False`;单/多 cubesmall 跟踪用**同一个 raw 重定向**参考。

### 3.4 渲染
- **展示抖动用默认相机**(`[[render-jitter-default-camera]]`):判断策略抖不抖必须用固定默认相机;`--cam_follow hand`/`--cam_smooth` 会把手的抖动吸收掉。判断以眼睛看默认相机视频为准,不靠像素 diff。
- **allegro --ref 渲染**(`[[allegro-ref-render-gotcha]]`,Isaac Gym 侧):需 DOF 块重排 + 关节限位 clamp,否则大拇指手型错;**wuji 不受影响**(identity 映射)。

### 3.5 工作方式(用户偏好)
- **主动更新 doc**(`[[proactive-doc-updates]]`):每次启动实验/拿到结果,**主动**更新对应实验 doc(别等用户提醒,用户会烦)。释放 GPU 时停 multi 不停 single。

---

## 4. 已完成

- ✅ mjlab 单/多序列 cubesmall 跟踪跑通(wdelta 动作 + full obs499 + DexTrack PPO 对齐 net-v4)——`docs/mjlab_migration_plan.md`。
- ✅ 三档 reward(original/pinall3/cgsmooth_b2_softclip)多序列对比:CGSmooth 最稳(cubesmall 23序列 0 失败)。
- ✅ **接触门控**实现 + cup/apple/cube/3obj 多序列 + cup/apple 单序列从头训。
- ✅ **接触门控完整 eval**:cube 6/6、cup 6/8、3obj 16/22;**3obj generalist ≥ 每个专项,apple 可举**。
- ✅ 修复 eval/render park bug、磁盘满根因(save_interval→250 + 清理)。
- ✅ 对比视频(`mjlab_ct_*`、`mjlab_3obj_*FIXED`)、fair 改接触门控。
- ✅ 迁移:两仓 push、数据+ckpt 上 HF、安装脚本、本 doc。

## 5. TODO

- **[主线] 扩多物体 generalist**:3obj 已验证 generalist≥专项;下一步 ① 加更多物体(参考 Isaac Gym 侧 9物体清单)② 让 3obj 多训(现仅 2.56B 样本就 16/22,env 8000 可加大)。**apple 不用加 kp**。
- **[关注] 3obj 收敛后重测 max_z**(用 park 修复版 eval);cup s9 / apple s1/s6/s7/s8 仍未举,看是否随训练解决。
- **[可选] cgsmooth 训更久**量化 HAND_EMA/action_rate 抖动收益(单序列只 500iter 时欠训)。
- **[可选] 多卡 40000 env**:对齐 DexTrack 需多卡 DDP(单卡 mujoco-warp 上限 ~22000-26000)。
- **[背景] specialist→generalist 蒸馏**:DexTrack 真方法,未释出;多物体迁移已部分体现其价值。

## 6. 关键操作纪律(务必遵守)

1. 清进程**只用精确 PID/pgid**,不用宽泛 `pkill -f`/`grep`(会误杀训练)。
2. GPU 共享:其他用户(tongxuantian/andrew 等)的进程**不能动**;只动自己的。
3. mjlab run 全崩**先 `df -h`**(磁盘满,不是外部 kill)。
4. **不要 pip 装进 pixi env**(改 `pixi.toml`);**不要把 wuji-mjlab/ 提交进 DexTrack 主仓**(独立 git 仓)。
5. 训练是 setsid detach 的,**直接 grep 日志**(`/tmp/wuji_*.log`),别 tail 子 transcript。
6. 渲染 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`;eval/render 3obj 锁序列要 park 非激活物体。
7. 判据看 **max_z vs 参考峰**,不是 fair。

## 7. doc 索引

| doc | 内容 |
|---|---|
| `docs/mjlab_migration_plan.md` | ★ mjlab 实验日志(本轮接触门控/多物体结果在最后) |
| `docs/HANDOFF_for_mjlab_migration.md` | Isaac Gym→mjlab 迁移交接(数据布局/物理发现/fair 指标) |
| `docs/cubesmall_single_multi.md` | Isaac Gym 侧 cubesmall reward 调优(palm-lock/抖动/contact guidance) |
| `docs/grab_contact_guidance_plan.md` | GRAB 真值接触 A/B/B2 完整记录 |
| `docs/wuji_integration_plan.md` | wuji 手接入 DexTrack 分阶段计划 |
| `docs/wuji_multiobj_generalist.md` | Isaac Gym 侧多物体 generalist 配置 |
| `docs/wuji_reward_tuning.md` | reward 开关调优 |
| `migrate_install.sh` | 一键迁移安装脚本 |
