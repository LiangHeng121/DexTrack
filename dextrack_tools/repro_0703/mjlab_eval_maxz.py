"""mjlab 侧 max_z 成功率 eval (0703 对比 表1的 mjlab 列 + 表2)。
每序列: 锁定该序列, num_envs=1, 300步 rollout, 量物体 root z 的 max vs 参考峰。
多物体(3obj)必须 park 非激活物体(否则堆手边污染)。判据 ✓ = max_z >= 参考峰-0.05。

跑法(在 wuji-mjlab 目录):
  pixi run python ../dextrack_tools/repro_0703/mjlab_eval_maxz.py <TASK> <object> [seqs...]
例:
  pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_CubesmallMulti_CGSmooth_Contact cubesmall
  pixi run python .../mjlab_eval_maxz.py WujiHand_Tracking_3Obj_CGSmooth_Contact cup   # 3obj 在 cup 序列上
不给 seqs 则自动用该 object 的所有 *_lift 序列。
"""
import sys, glob, types
from dataclasses import asdict
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from wuji_mjlab.tasks.tracking.config.wuji_hand.env_cfgs import _object_sequences
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK, OBJ = sys.argv[1], sys.argv[2]
SEQS = sys.argv[3:] or [s for s in _object_sequences(OBJ) if "lift" in s]


def latest(task):
    tag = task.replace("WujiHand_Tracking_", "")
    for wd in sorted(glob.glob(f"logs/rsl_rl/wuji_tracking/*{tag}/model_*.pt"),
                     key=lambda x: int(x.split("_")[-1][:-3]), reverse=True):
        try:
            torch.load(wd, map_location="cpu"); return wd
        except Exception:
            continue


def lk(i):  # 锁序列 + park 非激活物体(多物体)
    def _r(self, ids):
        self.time_steps[ids] = 0; self.env_seq[ids] = i
        if hasattr(self, "env_obj"):
            self.env_obj[ids] = self.seq_obj[i]
        s, t = self.env_seq[ids], self.time_steps[ids]
        jp = torch.clip(self._ref_qpos[s, t].clone(),
                        self.robot.data.soft_joint_pos_limits[ids][:, :, 0],
                        self.robot.data.soft_joint_pos_limits[ids][:, :, 1])
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        self.robot.reset(env_ids=ids)
        if hasattr(self, "objs") and len(self.objs) > 1:
            act = int(self.seq_obj[i])
            for j, o in enumerate(self.objs):
                if j == act:
                    root = torch.cat([self._ref_obj_pos[s, t], self._ref_obj_quat[s, t],
                                      torch.zeros(len(ids), 6, device=self.device)], dim=-1)
                else:
                    pos = self._park[j].flatten()[:3]
                    root = torch.cat([pos, torch.tensor([1., 0, 0, 0], device=self.device),
                                      torch.zeros(6, device=self.device)]).unsqueeze(0)
                o.write_root_state_to_sim(root, env_ids=ids); o.reset(env_ids=ids)
        else:
            self.obj.write_root_state_to_sim(
                torch.cat([self._ref_obj_pos[s, t], self._ref_obj_quat[s, t],
                           torch.zeros(len(ids), 6, device=self.device)], dim=-1), env_ids=ids)
            self.obj.reset(env_ids=ids)
    return _r


wd = latest(TASK)
ec, ac = prepare_task_cfgs(TASK, [], play=True); ec.scene.num_envs = 1
base = ManagerBasedRlEnv(cfg=ec, device="cuda:0")
env = RslRlVecEnvWrapper(base, clip_actions=ac.clip_actions)
r = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(ac), device="cuda:0")
r.load(wd, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
pol = r.get_inference_policy(device="cuda:0")
cmd = base.command_manager.get_term("motion")
NAMES = [p.split("info_")[1].split("_nf_300")[0] for p in cmd.cfg.motion_files]
ok = 0; rows = []
for sq in SEQS:
    if sq not in NAMES:
        continue
    cmd._resample_command = types.MethodType(lk(NAMES.index(sq)), cmd)
    base.reset(); obs = env.get_observations(); mz = rz = 0
    for _ in range(300):
        obs, _, _, _ = env.step(pol(obs))
        mz = max(mz, float(cmd.obj_pos[0, 2])); rz = max(rz, float(cmd.ref_obj_pos[0, 2]))
    good = mz >= rz - 0.05; ok += good
    rows.append(f"  {'✓' if good else '✗'} {sq:<28} max_z={mz:.3f} ref={rz:.2f}")
print(f"{TASK} on {OBJ}: {ok}/{len(rows)} (ckpt {wd.split('model_')[-1][:-3]}it)")
print("\n".join(rows))
