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
ap.add_argument("--cam_scale", type=float, default=1.0, help="<1 拉近相机让手更大, >1 拉远")
ap.add_argument("--cam_follow", default="mid", choices=["mid", "hand"],
                help="相机跟随 手+物中点(mid) 还是 只跟手(hand,物体掉落/远离时构图更稳)")
ap.add_argument("--cam_smooth", type=int, default=0,
                help="相机跟随中心做滑动平均的窗口(帧),抗抖动;0=不平滑,建议15~25(奇数)")
ap.add_argument("--cam_eye", default="", help="固定相机eye 'x,y,z'(给了就用固定相机,忽略cam_follow)")
ap.add_argument("--cam_target", default="", help="固定相机target 'x,y,z'")
ap.add_argument("--cam_center_src", default="", help="外部 (T,3) npy 作相机跟随中心(两面板共用同一中心 -> 视角一致)")
# 接触点叠加: 把 contact npy 的 object-local 接触点用物体当前位姿变到世界,投影到像素画圆。
ap.add_argument("--contact", default="", help="GRAB真值(B版) contact npy, 画实心圆(○)")
ap.add_argument("--contact_proj", default="", help="wuji投影(A版) contact npy, 画空心圆(◇)叠加对比")
ap.add_argument("--marker_radius", type=int, default=9, help="接触点标记半径(像素)")
args = ap.parse_args()
# 五指固定配色 (BGR? no, RGB) 拇/食/中/无名/小
FINGER_RGB = [(220,40,40),(40,180,40),(40,90,220),(240,150,20),(150,40,200)]
FINGER_NM = ["thumb","index","middle","ring","pinky"]
# 注: "参考开环+物理"(物体掉落对比)不在此脚本做, 用真实 env 的 kinematics_only=True 跑 test
#     生成 rollout 后, 像普通 rollout 一样在这里渲染即可(手型/动力学才与训练一致)。

HAND_URDF = {
    "wuji": "wuji_hand_description/urdf/wuji_hand_right_fly.urdf",
    "allegro": "allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf",
}[args.hand]

# ---- load motion ----
if args.ref:
    d = np.load(args.src, allow_pickle=True).item()
    hand = np.asarray(d["robot_delta_states_weights_np"], dtype=np.float32)   # (T,ndof) 文件DOF顺序
    opos = np.asarray(d["object_transl"], dtype=np.float32)                   # (T,3)
    oqt = np.asarray(d["object_rot_quat"], dtype=np.float32)                  # (T,4) xyzw
    # 文件里的手指DOF顺序 != Isaac Gym asset 顺序; env(task.py joint_idxes_ordering)用 argsort 重排。
    # wuji 是 identity, allegro 需要重排, 否则手指接错(手型看着不对)。
    if args.hand == "allegro" and hand.shape[1] == 22:
        joint_idxes_ordering = list(range(10)) + [i + 14 for i in range(8)] + [10, 11, 12, 13]
        hand = hand[:, np.argsort(joint_idxes_ordering)]
else:
    d = np.load(args.src, allow_pickle=True).item()
    ts = sorted([k for k in d.keys() if isinstance(k, int)])
    hand = np.array([d[t]["shadow_hand_dof_pos"][args.env] for t in ts], dtype=np.float32)
    opos = np.array([d[t]["object_pose"][args.env, :3] for t in ts], dtype=np.float32)
    oqt = np.array([d[t]["object_pose"][args.env, 3:7] for t in ts], dtype=np.float32)
T = len(hand)
print(f"frames={T}  hand={hand.shape}  obj={opos.shape}")

# ---- contact overlay data ----
def _load_contact(fn):
    if not fn: return None
    c = np.load(fn, allow_pickle=True).item()
    print(f"contact: {fn}  flag{c['contact_flag'].shape} src={c.get('source','proj')}")
    return (np.asarray(c["contact_pos_local"], np.float32), np.asarray(c["contact_flag"], np.float32))
contactB = _load_contact(args.contact)        # GRAB truth (solid)
contactA = _load_contact(args.contact_proj)   # wuji projection (hollow)

def _quat_apply(q_xyzw, v):  # rotate v (3,) by quat xyzw
    x, y, z, w = q_xyzw
    qv = np.array([x, y, z]); t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)

def _contact_world(contact, t, opos_t, oquat_t):
    # returns list of (finger_idx, world_pos(3,)) for fingers in contact at frame t
    cp, fl = contact; tt = min(t, cp.shape[0] - 1); out = []
    for f in range(cp.shape[1]):
        if fl[tt, f] > 0:
            out.append((f, opos_t + _quat_apply(oquat_t, cp[tt, f])))
    return out

def _project(pts_w, view, proj, W, H):
    # IsaacGym: row-vector convention, world->view->clip. returns list of (px,py,depth)
    res = []
    for p in pts_w:
        ph = np.array([p[0], p[1], p[2], 1.0], np.float64)
        clip = ph @ view @ proj
        if clip[3] == 0: res.append(None); continue
        ndc = clip[:3] / clip[3]
        px = (ndc[0] * 0.5 + 0.5) * W
        py = (1.0 - (ndc[1] * 0.5 + 0.5)) * H
        res.append((px, py, clip[3]))
    return res

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

# clamp qpos to asset 关节限位 —— env 一直对 targets 做 tensor_clamp; 文件参考值可能超限
# (尤其 allegro 大拇指根关节 ~0.71 > 上限 ~0.47), 不 clamp 会比真实手型多弯一截。
_lo = np.asarray(dp["lower"], dtype=np.float32); _hi = np.asarray(dp["upper"], dtype=np.float32)
if np.all(_hi >= _lo) and np.any(_hi > _lo):
    hand = np.clip(hand, _lo, _hi)

# camera
cp = gymapi.CameraProperties(); cp.width = args.W; cp.height = args.H; cp.enable_tensors = False
cam = gym.create_camera_sensor(env, cp)
# lighting
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, -1, 1))

obj_rb = gym.get_actor_rigid_body_count(env, hand_actor)  # object root body index in env

# 预计算相机跟随中心(可选滑动平均抗抖): 手抖时相机不再逐帧抖, 只跟低频大动作
if args.cam_center_src:
    cam_centers = np.load(args.cam_center_src).astype(np.float32)   # 共用外部中心(两面板一致视角)
    assert cam_centers.shape[0] >= T, f"cam_center_src {cam_centers.shape} < frames {T}"
    cam_centers = cam_centers[:T].copy()
else:
    cam_centers = hand[:, :3].copy() if args.cam_follow == "hand" else 0.5 * (hand[:, :3] + opos)
if args.cam_smooth and args.cam_smooth > 1:
    w = int(args.cam_smooth); pad = w // 2
    padded = np.pad(cam_centers, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(w, dtype=np.float32) / w
    cam_centers = np.stack([np.convolve(padded[:, k], ker, mode="valid")[: cam_centers.shape[0]] for k in range(3)], axis=1)

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

    if args.cam_eye and args.cam_target:
        # 固定相机(同视角对比用)
        _e = [float(x) for x in args.cam_eye.split(",")]
        _g = [float(x) for x in args.cam_target.split(",")]
        gym.set_camera_location(cam, env, gymapi.Vec3(_e[0], _e[1], _e[2]), gymapi.Vec3(_g[0], _g[1], _g[2]))
    else:
        # camera follows the (optionally smoothed) center
        c = cam_centers[t]
        s = args.cam_scale
        eye = gymapi.Vec3(c[0] + 0.45 * s, c[1] - 0.45 * s, c[2] + 0.30 * s)
        gym.set_camera_location(cam, env, eye, gymapi.Vec3(c[0], c[1], c[2]))

    img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
    img = img.reshape(args.H, args.W, 4)[:, :, :3].copy()

    # ---- overlay contact markers (project object-local contact -> world -> pixel) ----
    if contactB is not None or contactA is not None:
        from PIL import Image, ImageDraw
        view = np.asarray(gym.get_camera_view_matrix(sim, env, cam))
        projm = np.asarray(gym.get_camera_proj_matrix(sim, env, cam))
        pil = Image.fromarray(img); dr = ImageDraw.Draw(pil); r = args.marker_radius
        for contact, solid in [(contactB, True), (contactA, False)]:
            if contact is None: continue
            cw = _contact_world(contact, t, opos[t], oqt[t])
            if not cw: continue
            fidx = [f for f, _ in cw]; pix = _project([p for _, p in cw], view, projm, args.W, args.H)
            for f, pr in zip(fidx, pix):
                if pr is None or pr[2] <= 0: continue
                px, py = pr[0], pr[1]; col = FINGER_RGB[f]
                if solid:  # B truth: filled circle
                    dr.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(255, 255, 255), width=2)
                else:      # A proj: hollow diamond
                    dr.polygon([(px, py - r - 2), (px + r + 2, py), (px, py + r + 2), (px - r - 2, py)], outline=col, width=3)
        img = np.asarray(pil)
    frames.append(img)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
imageio.mimsave(args.out, frames, fps=20, codec="libx264")
print(f"saved {len(frames)} frames -> {args.out}")
gym.destroy_sim(sim)
