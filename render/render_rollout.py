"""
DexTrack rollout offline renderer (v2: explicit Isaac Gym DOF order + sanity check).

Reads `ts_to_hand_obj_obs_reset_1.npy` produced by Isaac Gym eval, plus the
Allegro hand URDF and the object mesh, and outputs an mp4 / gif.

Usage:
    python render_rollout.py \
        --rollout rollout/ts_to_hand_obj_obs_reset_1.npy \
        --urdf assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf \
        --obj-mesh assets/meshdatav3_scaled/sem/<seq>/coacd/<seq>.obj \
        --out out.mp4
"""
import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
import imageio.v2 as imageio

if os.uname().sysname == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender


ISAACGYM_DOF_ORDER = [
    "WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz",
    "joint_0",  "joint_1",  "joint_2",  "joint_3",
    "joint_12", "joint_13", "joint_14", "joint_15",
    "joint_4",  "joint_5",  "joint_6",  "joint_7",
    "joint_8",  "joint_9",  "joint_10", "joint_11",
]


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
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
            origin = vis.origin if vis.origin is not None else np.eye(4)
            link_nodes.setdefault(link_name, []).append((node, origin))

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
    L = np.eye(4)
    L[:3, 3] = [0.4, 0.4, 1.2]
    scene.add(light, pose=L)

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    cam_pose = np.array([
        [1.0, 0.0,  0.0,   0.05],
        [0.0, 0.3, -0.95, -0.55],
        [0.0, 0.95, 0.3,   0.55],
        [0.0, 0.0,  0.0,   1.0],
    ])
    scene.add(cam, pose=cam_pose)

    return robot, scene, link_nodes, obj_node, ref_node


def apply_pose(robot, scene, link_nodes, hand_q22):
    cfg = {jn: float(hand_q22[i]) for i, jn in enumerate(ISAACGYM_DOF_ORDER)}
    robot.update_cfg(cfg)
    for link_name, entries in link_nodes.items():
        T_link = robot.get_transform(link_name)
        for node, vis_origin in entries:
            scene.set_pose(node, T_link @ vis_origin)


def sanity_check(robot, hand_qpos, obj_pose, env_idx):
    """Print palm and object world positions at frame 0 and check distance."""
    cfg = {jn: float(hand_qpos[0, i]) for i, jn in enumerate(ISAACGYM_DOF_ORDER)}
    robot.update_cfg(cfg)
    candidates = ["palm_link", "link_palm_rz", "link_palm_z"]
    palm_T = None
    palm_name = None
    for c in candidates:
        try:
            palm_T = robot.get_transform(c)
            palm_name = c
            break
        except Exception:
            continue
    if palm_T is None:
        print("[check] could not find palm link, available links:",
              list(robot.link_map.keys())[:10])
        return
    palm_xyz = palm_T[:3, 3]
    obj_xyz = obj_pose[0, :3]
    dist = float(np.linalg.norm(palm_xyz - obj_xyz))
    print(f"[check] frame 0 env {env_idx}: palm({palm_name})={palm_xyz}, obj={obj_xyz}, dist={dist:.4f} m")
    if dist > 0.3:
        print(f"[warn] palm-object distance > 30 cm at frame 0. Coord/joint order likely wrong.")


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
    ap.add_argument("--quat-wxyz", action="store_true",
                    help="Interpret object quaternion as (w,x,y,z) instead of default (x,y,z,w).")
    args = ap.parse_args()

    print(f"[load] rollout: {args.rollout}")
    hand_qpos, obj_pose, ref_obj = load_rollout(args.rollout, args.env_idx)
    n_frames = hand_qpos.shape[0]
    print(f"[load] {n_frames} frames, hand_dof={hand_qpos.shape[1]}")

    if args.quat_wxyz:
        obj_pose = obj_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        ref_obj = ref_obj[:, [0, 1, 2, 4, 5, 6, 3]]

    robot, scene, link_nodes, obj_node, ref_node = make_scene(
        args.urdf, args.obj_mesh, ref_mesh=(args.reference is not None))

    urdf_joint_names = list(robot.actuated_joint_names)
    missing = [n for n in ISAACGYM_DOF_ORDER if n not in urdf_joint_names]
    if missing:
        print(f"[error] URDF missing joints: {missing}")
        return
    print(f"[urdf] {len(urdf_joint_names)} actuated joints found, all 22 IG DOFs present.")

    sanity_check(robot, hand_qpos, obj_pose, args.env_idx)

    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    frames_out = []
    for t in range(0, n_frames, args.stride):
        apply_pose(robot, scene, link_nodes, hand_qpos[t])
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
