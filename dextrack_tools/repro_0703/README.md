# 复现 0703 对比（IG vs mjlab / 专项 vs 3obj）

结果表在 `docs/0703.md`。三类指标：**max_z 成功率**（eval）、**fair**（tb）、**收敛步数/时间**（tb）。

## 需要的东西（都已在 HF / 仓库）
- **IG ckpt + tb**：`isaacgymenvs/logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle*/`（HF `isaacgymenvs/logs`，migrate_install 已拉）。
- **mjlab ckpt + tb**：`wuji-mjlab/logs/rsl_rl/wuji_tracking/2026-06-21*_CGSmooth_Contact`（HF `wuji-mjlab/logs`）。
- **conda `dextrack` 环境**（跑 IG eval 用）：HF `conda_packs/dextrack_env.tar.gz`，解包见 `docs/NEW_MACHINE_STEPS.md §D`。
- **pixi 环境**（跑 mjlab eval + 读 tb）：`wuji-mjlab` 里 `pixi install`。

## 4 个 IG best ckpt
```
cubesmall: isaacgymenvs/logs/..._beta8_noidle/wuji_cubesmall_pinall3/20260615_002635/ckpt/best_ep_2816_rew_244.60.pth
apple:     ..._beta8_noidle_apple/wuji_apple/20260615_033134/ckpt/best_ep_4598_rew_179.26.pth
cup:       ..._beta8_noidle_cup/wuji_cup/20260615_004253/ckpt/best_ep_3107_rew_165.26.pth
3obj:      ..._beta8_noidle_3obj/wuji_cup_apple_cubesmall/20260615_091354/ckpt/best_ep_3691_rew_166.17.pth
```
（4 个 mjlab contact run 用 `logs/rsl_rl/wuji_tracking` 里最新 ckpt，脚本自动取。）

## 复现步骤

### A. max_z 成功率
**IG**（在 `isaacgymenvs/`，先 `conda activate dextrack`）：脚本按序列跑 `run_tracking_headless_grab_multiple_wuji_test.sh`（numEnvs=100，存 `logs_test/.../ts_to_hand_obj_obs_*.npy`），从中取物体 z 峰 vs 参考峰。
```bash
# 先建 cuda_idx 覆盖的临时脚本(原脚本 cuda_idx=2 写死;GPU 换成空卡, 如 3):
sed 's/^export cuda_idx=2/export cuda_idx=3/' scripts/run_tracking_headless_grab_multiple_wuji_test.sh > /tmp/wuji_eval_test.sh
bash dextrack_tools/repro_0703/ig_eval_maxz.sh          # cube/cup/apple 专项 -> /tmp/eval_results.txt
bash dextrack_tools/repro_0703/ig_eval_3obj_maxz.sh     # 3obj 在 22 条 -> /tmp/eval_3obj_results.txt
```
（脚本里 ckpt 路径、conda 路径、GPU 号按新机改。）

**mjlab**（在 `wuji-mjlab/`）：
```bash
pixi run python ../dextrack_tools/repro_0703/mjlab_eval_maxz.py WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact cubesmall
pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_CupMulti_CGSmooth_Contact cup
pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_AppleMulti_CGSmooth_Contact apple
# 表2的 3obj 在各物体序列上:
pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_3Obj_CGSmooth_Contact cubesmall
pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_3Obj_CGSmooth_Contact cup
pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_3Obj_CGSmooth_Contact apple
```

### B. fair + 收敛步数/时间（tb）
```bash
# 每个 run 目录跑一次(IG env=40000; mjlab env: cube24000/cup20000/apple22000/3obj8000):
python dextrack_tools/repro_0703/read_metrics_tb.py <IG_run目录> 40000 reward_fair/iter
python dextrack_tools/repro_0703/read_metrics_tb.py <mjlab_run目录> 24000 Episode/fair_reward
```

## 判据 / 口径
- ✓ = 物体 max_z ≥ 参考峰 − 0.05。IG eval 用 100env 取中位、mjlab 用 1env（同一判据）。
- **fair 不可跨模拟器比绝对值**（IG 距离门控 reward_fair vs mjlab 接触门控 Episode/fair_reward）。
- 收敛 = 到 90% 峰值 fair 的点；收敛时间 = 中位单iter时间 × 收敛iter（剔除停机间隙）。
- 序列集：cubesmall s1/s2/s5/s6/s8/s10_lift；cup s1/s3/s4/s5/s6/s8/s9/s10_lift；apple s1/s2/s3/s4/s6/s7/s8/s9_lift。
