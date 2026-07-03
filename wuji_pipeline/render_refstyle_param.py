"""Parameterized cmp3 refstyle render: LEFT=FPOS_v1 | RIGHT=TopoRetarget, for one tag.
Config via env vars: TAG, TASK, FPOS_NPY, TOPO_NPY, CONTACT_NPY, PHI_FPOS, PHI_TOPO, OUT.
Run from wuji-mjlab cwd, wuji-mjlab pixi env, MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4.
"""
from __future__ import annotations
import os
import imageio, numpy as np, torch
from PIL import Image, ImageDraw
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TAG = os.environ["TAG"]; TASK = os.environ["TASK"]
FPOS = os.environ["FPOS_NPY"]; TOPO = os.environ["TOPO_NPY"]; CONTACT = os.environ["CONTACT_NPY"]
PEN = {0: np.load(os.environ["PHI_FPOS"]), 1: np.load(os.environ["PHI_TOPO"])}
OUT = os.environ["OUT"]
OBJZLABEL = os.environ.get("OBJ", "obj")
PANEL = {0: ("FPOS_v1 (old)", (255, 150, 150)), 1: ("TopoRetarget", (120, 255, 140))}
NSTEP = 300; DEV = "cuda:0"


def label(frame, title, color, z, pen):
    img = Image.fromarray(np.ascontiguousarray(frame)); d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
    d.text((6, 3), f"{title}   [{TAG}]", fill=color)
    d.text((6, 24), f"{OBJZLABEL} z = {z:.2f} m    max finger penetration = {pen:.1f} mm", fill=(255, 230, 120))
    return np.asarray(img)


def main():
    env_cfg, _ = prepare_task_cfgs(TASK, [], play=True)
    env_cfg.scene.num_envs = 1
    m = env_cfg.commands["motion"]
    m.motion_files = (FPOS, TOPO)
    m.contact_files = (CONTACT, CONTACT)
    m.seq_object_idx = (0, 0)
    base = ManagerBasedRlEnv(cfg=env_cfg, device=DEV, render_mode="rgb_array")
    cmd = base.command_manager.get_term("motion")
    robot, obj = cmd.robot, cmd.obj
    ids = torch.tensor([0], device=DEV)
    Tref = cmd._ref_obj_pos.shape[1]
    print(f"loaded {cmd._ref_qpos.shape[0]} seqs, Tref={Tref}", flush=True)
    base.reset()
    cols = {}
    for idx in (0, 1):
        fr = []
        for t in range(NSTEP):
            tt = min(t, Tref - 1)
            jp = cmd._ref_qpos[idx, tt].unsqueeze(0)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
            root = torch.cat([cmd._ref_obj_pos[idx, tt], cmd._ref_obj_quat[idx, tt],
                              torch.zeros(6, device=DEV)]).unsqueeze(0)
            obj.write_root_state_to_sim(root, env_ids=ids)
            base.sim.forward()
            img = base.render()
            if img is not None:
                z = float(cmd._ref_obj_pos[idx, tt, 2])
                fr.append(label(np.asarray(img), PANEL[idx][0], PANEL[idx][1], z, float(PEN[idx][tt])))
        cols[idx] = fr
        print(f"  {PANEL[idx][0]}: {len(fr)} frames", flush=True)
    n = min(len(cols[0]), len(cols[1]))
    h = max(cols[0][0].shape[0], cols[1][0].shape[0])
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    combined = [np.concatenate([cols[0][i], sep, cols[1][i]], axis=1) for i in range(n)]
    imageio.mimwrite(OUT, combined, fps=30, quality=8)
    print(f"WROTE {OUT}  ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
