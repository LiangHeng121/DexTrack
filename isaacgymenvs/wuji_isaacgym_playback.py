"""Render a wuji grasp INSIDE Isaac Gym (real URDF meshes, object mesh, ground,
lighting) via a headless camera sensor -> MP4. Run from isaacgymenvs/.

Kinematic playback: each frame we set the hand DOF + object root pose to the
saved states and render with Isaac Gym's own camera (not an offline reconstruction).

Usage (dextrack env):
  python wuji_isaacgym_playback.py --src <npy> --out out.mp4 [--ref] [--env 1] [--gpu 0]
    --ref : src is a reference npy (robot_delta_states_weights_np + object_transl/rot_quat)
            else src is a sim rollout (shadow_hand_dof_pos + object_pose), use --env idx
"""
import argparse, os
import numpy as np
from isaacgym import gymapi
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="wuji_pipeline/out/wuji_isaacgym.mp4")
ap.add_argument("--ref", action="store_true")
ap.add_argument("--env", type=int, default=1)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--obj_code", default="ori_grab_s2_cubesmall_inspect_1")
ap.add_argument("--hand", default="wuji", choices=["wuji", "allegro"])
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--W", type=int, default=720)
ap.add_argument("--H", type=int, default=720)
args = ap.parse_args()

HAND_URDF = {
    "wuji": "wuji_hand_description/urdf/wuji_hand_right_fly.urdf",
    "allegro": "allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf",
}[args.hand]

# ---- load motion ----
if args.ref:
    d = np.load(args.src, allow_pickle=True).item()
    hand = np.asarray(d["robot_delta_states_weights_np"], dtype=np.float32)   # (T,26)
    opos = np.asarray(d["object_transl"], dtype=np.float32)                   # (T,3)
    oqt = np.asarray(d["object_rot_quat"], dtype=np.float32)                  # (T,4) xyzw
else:
    d = np.load(args.src, allow_pickle=True).item()
    ts = sorted([k for k in d.keys() if isinstance(k, int)])
    hand = np.array([d[t]["shadow_hand_dof_pos"][args.env] for t in ts], dtype=np.float32)
    opos = np.array([d[t]["object_pose"][args.env, :3] for t in ts], dtype=np.float32)
    oqt = np.array([d[t]["object_pose"][args.env, 3:7] for t in ts], dtype=np.float32)
T = len(hand)
print(f"frames={T}  hand={hand.shape}  obj={opos.shape}")

# ---- sim ----
gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 60.0
sp.substeps = 1
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, 0.0)           # no gravity: pure kinematic playback
sp.use_gpu_pipeline = False
sp.physx.use_gpu = True
sim = gym.create_sim(args.gpu, args.gpu, gymapi.SIM_PHYSX, sp)

pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, pp)

asset_root = "../assets"
ho = gymapi.AssetOptions(); ho.fix_base_link = True; ho.disable_gravity = True
hand_asset = gym.load_asset(sim, asset_root, HAND_URDF, ho)
oo = gymapi.AssetOptions(); oo.fix_base_link = True; oo.disable_gravity = True; oo.use_mesh_materials = True
obj_file = f"meshdatav3_scaled/sem/{args.obj_code}/coacd/coacd_1_vis.urdf"
if not os.path.exists(os.path.join(asset_root, obj_file)):
    obj_file = f"meshdatav3_scaled/sem/{args.obj_code}/coacd/coacd_1.urdf"
obj_asset = gym.load_asset(sim, asset_root, obj_file, oo)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
pose = gymapi.Transform(); pose.p = gymapi.Vec3(0, 0, 0)
hand_actor = gym.create_actor(env, hand_asset, pose, "hand", 0, 0)
obj_actor = gym.create_actor(env, obj_asset, pose, "obj", 0, 0)

# shadow_hand_dof_pos is saved in raw sim/asset DOF order, so set DOF directly.
ndof = gym.get_asset_dof_count(hand_asset)
assert hand.shape[1] == ndof, f"rollout dof {hand.shape[1]} != {args.hand} asset dof {ndof}"
print(f"{args.hand} asset dof = {ndof}")

# drive DOF stiffly so the hand holds the set pose during the 1 physics step
dp = gym.get_actor_dof_properties(env, hand_actor)
dp["driveMode"][:] = gymapi.DOF_MODE_POS
dp["stiffness"][:] = 1e6
dp["damping"][:] = 1e3
gym.set_actor_dof_properties(env, hand_actor, dp)

# camera
cp = gymapi.CameraProperties(); cp.width = args.W; cp.height = args.H; cp.enable_tensors = False
cam = gym.create_camera_sensor(env, cp)
# lighting
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, -1, 1))

obj_rb = gym.get_actor_rigid_body_count(env, hand_actor)  # object root body index in env

frames = []
for t in range(0, T, args.stride):
    # set hand DOF
    ds = np.zeros(ndof, dtype=gymapi.DofState.dtype)
    ds["pos"][:] = hand[t]
    gym.set_actor_dof_states(env, hand_actor, ds, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, hand_actor, ds["pos"].astype(np.float32))
    # set object root pose
    rs = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_ALL)
    rs["pose"]["p"][0] = (opos[t, 0], opos[t, 1], opos[t, 2])
    rs["pose"]["r"][0] = (oqt[t, 0], oqt[t, 1], oqt[t, 2], oqt[t, 3])
    rs["vel"]["linear"][0] = (0, 0, 0); rs["vel"]["angular"][0] = (0, 0, 0)
    gym.set_actor_rigid_body_states(env, obj_actor, rs, gymapi.STATE_ALL)

    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    # camera follows the hand base (WRJ0x/y/z) + object midpoint
    c = 0.5 * (hand[t, :3] + opos[t])
    eye = gymapi.Vec3(c[0] + 0.45, c[1] - 0.45, c[2] + 0.30)
    gym.set_camera_location(cam, env, eye, gymapi.Vec3(c[0], c[1], c[2]))

    img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
    img = img.reshape(args.H, args.W, 4)[:, :, :3].copy()
    frames.append(img)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
imageio.mimsave(args.out, frames, fps=20, codec="libx264")
print(f"saved {len(frames)} frames -> {args.out}")
gym.destroy_sim(sim)
