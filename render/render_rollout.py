"""
DexTrack rollout offline renderer.

Reads `ts_to_hand_obj_obs_reset_1.npy` produced by Isaac Gym eval, plus the
Allegro hand URDF and the object mesh, and outputs an mp4 / gif.

Usage:
    python render_rollout.py \
        --rollout rollout/ts_to_hand_obj_obs_reset_1.npy \
        --urdf assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf \
        --obj-mesh assets/meshdatav3_scaled/sem/ori_grab_s2_cubesmall_inspect_1/ori_grab_s2_cubesmall_inspect_1.obj \
        --env-idx 0 \
        --out cubesmall.mp4

Optional: --reference rollout/reference.npy to overlay the kinematic GT trajectory.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from PIL import Image
import imageio.v2 as imageio

os.environ.setdefault("PYOPENGL_PLATFORM", "egl" if os.uname().sysname == "Linux" else "")
import pyrender


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])


def pose_to_T(pose7):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def load_rollout(path, env_idx):
    raw = np.load(path, allow_pickle=True).item()
    frame_ids = sorted(k for k in raw.keys() if isinstance(k, int))
    hand_qpos = np.stack([raw[t]["shadow_hand_dof_pos"][env_idx] for t in frame_ids])
    obj_pose  = np.stack([raw[t]["object_pose"][env_idx]         for t in frame_ids])
    ref_obj   = np.stack([raw[t]["goal_pose_ref_np"][env_idx]    for t in frame_ids])
    return hand_qpos, obj_pose, ref_obj


def make_scene(urdf_path, obj_mesh_path, ref_mesh=False):
    robot = yourdfpy.URDF.load(urdf_path, build_scene_graph=True,
                               load_meshes=True, build_collision_scene_graph=False)
    scene = pyrender.Scene(bg_color=[0.95, 0.95, 0.97, 1.0],
                           ambient_light=[0.3, 0.3, 0.3])

    link_nodes = {}
    for link_name, link in robot.link_map.items():
        if not link.visuals:
            continue
        for vis in link.visuals:
            if vis.geometry.mesh is None:
                continue
            mesh_fp = robot._filename_handler(vis.geometry.mesh.filename)
            tm = trimesh.load(mesh_fp, force="mesh")
            if vis.geometry.mesh.scale is not None:
                tm.apply_scale(vis.geometry.mesh.scale)
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.55, 0.55, 0.6, 1.0],
                metallicFactor=0.1, roughnessFactor=0.7)
            pm = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)
            node = scene.add(pm, pose=np.eye(4))
            link_nodes.setdefault(link_name, []).append((node, vis.origin if vis.origin is not None else np.eye(4)))

    obj_tm = trimesh.load(obj_mesh_path, force="mesh")
    obj_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.35, 0.25, 1.0],
        metallicFactor=0.05, roughnessFactor=0.7)
    obj_node = scene.add(pyrender.Mesh.from_trimesh(obj_tm, material=obj_mat, smooth=False))

    ref_node = None
    if ref_mesh:
        ref_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.2, 0.7, 0.3, 0.4], alphaMode="BLEND",
            metallicFactor=0.0, roughnessFactor=0.9)
        ref_node = scene.add(pyrender.Mesh.from_trimesh(obj_tm.copy(), material=ref_mat, smooth=False))

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    L = np.eye(4); L[:3, 3] = [0.4, 0.4, 1.2]
    scene.add(light, pose=L)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = np.array([
        [ 1.0, 0.0, 0.0, 0.05],
        [ 0.0, 0.3, -0.95, -0.55],
        [ 0.0, 0.95, 0.3,  0.55],
        [ 0.0, 0.0, 0.0,   1.0],
    ])
    scene.add(cam, pose=cam_pose)

    return robot, scene, link_nodes, obj_node, ref_node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--obj-mesh", required=True)
    ap.add_argument("--reference", default=None,
                    help="Pass to overlay the GT kinematic object trajectory (green ghost).")
    ap.add_argument("--env-idx", type=int, default=0)
    ap.add_argument("--out", default="rollout.mp4")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="Render every Nth frame.")
    args = ap.parse_args()

    print(f"[load] rollout: {args.rollout}")
    hand_qpos, obj_pose, ref_obj = load_rollout(args.rollout, args.env_idx)
    n_frames = hand_qpos.shape[0]
    print(f"[load] {n_frames} frames, hand_dof={hand_qpos.shape[1]}")

    robot, scene, link_nodes, obj_node, ref_node = make_scene(
        args.urdf, args.obj_mesh, ref_mesh=(args.reference is not None))

    joint_order = list(robot.actuated_joint_names)
    print(f"[urdf] {len(joint_order)} actuated joints: {joint_order[:6]} ...")
    if len(joint_order) != hand_qpos.shape[1]:
        print(f"[warn] URDF joints ({len(joint_order)}) != rollout dof ({hand_qpos.shape[1]}). "
              "Assuming order in rollout matches URDF's actuated_joint_names ordering.")

    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    frames_out = []
    for t in range(0, n_frames, args.stride):
        cfg = {jn: float(hand_qpos[t, i]) for i, jn in enumerate(joint_order)}
        robot.update_cfg(cfg)
        fk = robot.scene.graph.get
        for link_name, entries in link_nodes.items():
            T_link = robot.get_transform(link_name)
            for node, vis_origin in entries:
                scene.set_pose(node, T_link @ vis_origin)

        scene.set_pose(obj_node, pose_to_T(obj_pose[t]))
        if ref_node is not None:
            scene.set_pose(ref_node, pose_to_T(ref_obj[t]))

        color, _ = renderer.render(scene)
        frames_out.append(color)

    renderer.delete()

    print(f"[save] {args.out} ({len(frames_out)} frames @ {args.fps} fps)")
    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames_out, fps=args.fps)
    else:
        imageio.mimwrite(args.out, frames_out, fps=args.fps, quality=8)
    print("[done]")


if __name__ == "__main__":
    main()
