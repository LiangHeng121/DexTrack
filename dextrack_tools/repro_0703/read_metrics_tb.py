"""从 tb event 读 fair + 收敛(到90%峰值) + 步数 + 时间 (0703 表1/表2 的 fair/收敛/时间列)。
需要 tensorboard(在 pixi 或 dextrack conda 里都有)。
跑法: python read_metrics_tb.py <run目录(含 events.out 或 summaries/)> <env数> [fair标签]
  IG fair标签=reward_fair/iter ; mjlab fair标签=Episode/fair_reward (默认自动找含'fair')
IG run 例: isaacgymenvs/logs/grab_multiple_wuji_pinall3_cgsmooth_B2_softclip_beta8_noidle/wuji_cubesmall_pinall3/20260615_002635  40000
mjlab run 例: wuji-mjlab/logs/rsl_rl/wuji_tracking/<...CubesmallMulti_CGSmooth_Contact>  24000
horizon=32 (两边一致)。env-steps = iter × env × 32。收敛时间 = 中位单iter时间 × 收敛iter。
"""
import sys, glob
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as EA

pat, env = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else None
fs = glob.glob(pat + "/events.out*") or glob.glob(pat + "/summaries/events.out*")
ea = EA.EventAccumulator(fs[0], size_guidance={"scalars": 0}); ea.Reload()
tags = ea.Tags()["scalars"]
t = tag if tag and tag in tags else next(
    x for x in tags if "fair" in x.lower() and "metric" not in x.lower())
sc = ea.Scalars(t)
mx = max(sc, key=lambda s: s.value); iters = sc[-1].step
dts = np.diff([s.wall_time for s in sc]); dts = dts[dts > 0]
itdt = np.median(dts) / (sc[1].step - sc[0].step)     # 秒/iter (中位, 剔除停机间隙)
conv = next((s.step for s in sc if s.value >= 0.9 * mx.value), iters)  # 到90%峰值
print(f"tag={t}")
print(f"fair峰={mx.value:.1f} (final={sc[-1].value:.1f})")
print(f"总iters={iters}  总steps={iters*env*32/1e9:.1f}G")
print(f"收敛iter={conv}  收敛steps={conv*env*32/1e9:.1f}G  收敛时间={itdt*conv/3600:.1f}h  ({itdt:.0f}s/iter)")
