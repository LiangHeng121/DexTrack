"""Definitive physics test: is the NO-OFFSET wuji reference grasp force-closure?
Drive the hand DOF to the EXACT reference qpos each frame (stiff PD), let the cube
be a FREE rigid body under gravity (only its initial pose is set). If the grasp holds,
the cube rises with the hand following the reference object trajectory; if not, it
stays on the ground while the hand lifts empty.

Run from isaacgymenvs/ (dextrack env)."""
import argparse, os, numpy as np
from isaacgym import gymapi

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="reference npy (robot_delta_states_weights_np + object_transl/rot_quat)")
ap.add_argument("--obj_code", default="ori_grab_s2_cubesmall_inspect_1")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default="")
ap.add_argument("--stiffness", type=float, default=1e4)
ap.add_argument("--damping", type=float, default=2e2)
ap.add_argument("--friction", type=float, default=1.5)
args = ap.parse_args()

d = np.load(args.ref, allow_pickle=True).item()
hand = np.asarray(d["robot_delta_states_weights_np"], dtype=np.float32)   # (T,26)
opos = np.asarray(d["object_transl"], dtype=np.float32)
oqt = np.asarray(d["object_rot_quat"], dtype=np.float32)
T = len(hand)

gym = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 1.0/60.0; sp.substeps = 2; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)          # GRAVITY ON
sp.use_gpu_pipeline = False; sp.physx.use_gpu = True
sp.physx.contact_offset = 0.002; sp.physx.rest_offset = 0.0
sim = gym.create_sim(args.gpu, args.gpu, gymapi.SIM_PHYSX, sp)
pp = gymapi.PlaneParams(); pp.normal = gymapi.Vec3(0,0,1); gym.add_ground(sim, pp)

asset_root = "../assets"
ho = gymapi.AssetOptions(); ho.fix_base_link = True; ho.disable_gravity = True
hand_asset = gym.load_asset(sim, asset_root, "wuji_hand_description/urdf/wuji_hand_right_fly.urdf", ho)
oo = gymapi.AssetOptions(); oo.fix_base_link = False; oo.disable_gravity = False; oo.use_mesh_materials = True  # FREE cube
obj_file = f"meshdatav3_scaled/sem/{args.obj_code}/coacd/coacd_1_vis.urdf"
if not os.path.exists(os.path.join(asset_root, obj_file)):
    obj_file = f"meshdatav3_scaled/sem/{args.obj_code}/coacd/coacd_1.urdf"
obj_asset = gym.load_asset(sim, asset_root, obj_file, oo)

env = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)
pose = gymapi.Transform(); pose.p = gymapi.Vec3(0,0,0)
hand_actor = gym.create_actor(env, hand_asset, pose, "hand", 0, 0)
opose = gymapi.Transform(); opose.p = gymapi.Vec3(float(opos[0,0]), float(opos[0,1]), float(opos[0,2]))
opose.r = gymapi.Quat(float(oqt[0,0]), float(oqt[0,1]), float(oqt[0,2]), float(oqt[0,3]))
obj_actor = gym.create_actor(env, obj_asset, opose, "obj", 0, 0)

# friction on hand + object
for a in (hand_actor, obj_actor):
    sh = gym.get_actor_rigid_shape_properties(env, a)
    for s in sh: s.friction = args.friction
    gym.set_actor_rigid_shape_properties(env, a, sh)

ndof = gym.get_asset_dof_count(hand_asset)
dp = gym.get_actor_dof_properties(env, hand_actor)
dp["driveMode"][:] = gymapi.DOF_MODE_POS; dp["stiffness"][:] = args.stiffness; dp["damping"][:] = args.damping
gym.set_actor_dof_properties(env, hand_actor, dp)
# init hand at reference frame 0
ds = np.zeros(ndof, dtype=gymapi.DofState.dtype); ds["pos"][:] = hand[0]
gym.set_actor_dof_states(env, hand_actor, ds, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env, hand_actor, hand[0].astype(np.float32))

frames = []; cam = None
if args.out:
    cp = gymapi.CameraProperties(); cp.width = 720; cp.height = 720; cp.enable_tensors = False
    cam = gym.create_camera_sensor(env, cp)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8,0.8,0.8), gymapi.Vec3(0.8,0.8,0.8), gymapi.Vec3(1,-1,1))

obj_z = np.zeros(T)
for t in range(T):
    gym.set_actor_dof_position_targets(env, hand_actor, hand[t].astype(np.float32))
    gym.simulate(sim); gym.fetch_results(sim, True)
    rs = gym.get_actor_rigid_body_states(env, obj_actor, gymapi.STATE_POS)
    obj_z[t] = rs["pose"]["p"][0][2]
    if cam is not None:
        gym.step_graphics(sim); gym.render_all_camera_sensors(sim)
        c = 0.5*(hand[t,:3] + np.array([rs["pose"]["p"][0][0], rs["pose"]["p"][0][1], obj_z[t]]))
        gym.set_camera_location(cam, env, gymapi.Vec3(c[0]+0.45, c[1]-0.45, c[2]+0.30), gymapi.Vec3(*c))
        img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR).reshape(720,720,4)[:,:,:3].copy()
        frames.append(img)

ref_z = opos[:,2]
print(f"=== physics test: drive hand to EXACT reference, free cube under gravity ===")
print(f"reference cube z: init={ref_z[0]*100:.1f}cm  max={ref_z.max()*100:.1f}cm  final={ref_z[-1]*100:.1f}cm")
print(f"PHYSICS   cube z: init={obj_z[0]*100:.1f}cm  max={obj_z.max()*100:.1f}cm  final={obj_z[-1]*100:.1f}cm")
print(f"lift achieved (physics max-init) = {(obj_z.max()-obj_z[0])*100:.1f}cm   (reference lifts {(ref_z.max()-ref_z[0])*100:.1f}cm)")
peak = ref_z.argmax()
print(f"at reference peak frame {peak}: ref z={ref_z[peak]*100:.1f}cm  physics z={obj_z[peak]*100:.1f}cm  -> "
      + ("HELD (grasp works)" if obj_z[peak] > ref_z[0]+0.05 else "DROPPED (not force-closure)"))
if args.out and frames:
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=20, codec="libx264"); print(f"saved video -> {args.out}")
gym.destroy_sim(sim)
