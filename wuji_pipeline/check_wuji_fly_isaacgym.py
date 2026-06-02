"""A4: load the wuji fly URDF in Isaac Gym and report DOF count/order/limits.

Run from isaacgymenvs/ (asset paths are relative to ../assets).
    conda activate dextrack
    cd isaacgymenvs && python ../wuji_pipeline/check_wuji_fly_isaacgym.py
"""
from isaacgym import gymapi
import numpy as np

ASSET_ROOT = "../assets"
ASSET_FILE = "wuji_hand_description/urdf/wuji_hand_right_fly.urdf"

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1 / 60.0
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)  # graphics=-1 headless

opts = gymapi.AssetOptions()
opts.fix_base_link = True
opts.disable_gravity = True
opts.collapse_fixed_joints = False  # keep tip_fixed frames as bodies
asset = gym.load_asset(sim, ASSET_ROOT, ASSET_FILE, opts)

ndof = gym.get_asset_dof_count(asset)
nbody = gym.get_asset_rigid_body_count(asset)
dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)
body_names = gym.get_asset_rigid_body_names(asset)

print(f"=== wuji fly URDF loaded ===")
print(f"DOF count : {ndof}  (expect 26 = 6 global + 20 finger)")
print(f"body count: {nbody}")
print(f"\nDOF order / type / limits:")
for i, n in enumerate(dof_names):
    t = "prism" if dof_props["hasLimits"][i] and dof_props['driveMode'][i] is not None else ""
    lo, hi = dof_props["lower"][i], dof_props["upper"][i]
    print(f"  [{i:2d}] {n:24s} lower={lo:+.3f} upper={hi:+.3f}")

tips = [b for b in body_names if "tip_link" in b]
print(f"\nfingertip bodies ({len(tips)}): {tips}")
print(f"palm in bodies: {'right_palm_link' in body_names}")

gym.destroy_sim(sim)
print("\nOK")
