"""A3: build the Isaac Gym 'fly' (floating) wuji hand URDF.

Takes assets/wuji_hand_description/urdf/right.urdf (20 finger DOF, root link
`right_palm_link`) and injects a 6-DOF floating base in front of the palm,
mirroring allegro_hand_description_right_fly_v2.urdf exactly (WRJ0x/y/z prismatic
+ WRJ0rx/ry/rz revolute). Resulting DOF order (Isaac Gym):
    [WRJ0x, WRJ0y, WRJ0z, WRJ0rx, WRJ0ry, WRJ0rz,
     right_finger1_joint1..4, ... right_finger5_joint1..4]  = 6 + 20 = 26
The finger order matches the wuji-retargeting pinocchio order (same URDF), so
the retarget->isaac finger permutation is identity.

Also strips the <mujoco> compiler tag (Isaac Gym doesn't need it).
"""
import os
import re

SRC = "assets/wuji_hand_description/urdf/right.urdf"
DST = "assets/wuji_hand_description/urdf/wuji_hand_right_fly.urdf"

# Floating-base block: massless-ish dummy links + 6 joints. Masses/limits/dynamics
# copied from allegro_hand_description_right_fly_v2.urdf. WRJ0rz child = right_palm_link.
FLOATING_BASE = """
  <!-- ===== Floating base: 6 global DOF (mirrors allegro fly_v2) ===== -->
  <link name="hand_root">
    <origin xyz="0 0 0"/>
  </link>
  <link name="link_palm_x"><inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00208916" ixy="-3.63457e-06" ixz="0.000223277" iyy="0.00182848" iyz="-1.75634e-05" izz="0.000482459"/></inertial></link>
  <link name="link_palm_y"><inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00208916" ixy="-3.63457e-06" ixz="0.000223277" iyy="0.00182848" iyz="-1.75634e-05" izz="0.000482459"/></inertial></link>
  <link name="link_palm_z"><inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00208916" ixy="-3.63457e-06" ixz="0.000223277" iyy="0.00182848" iyz="-1.75634e-05" izz="0.000482459"/></inertial></link>
  <link name="link_palm_rx"><inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00208916" ixy="-3.63457e-06" ixz="0.000223277" iyy="0.00182848" iyz="-1.75634e-05" izz="0.000482459"/></inertial></link>
  <link name="link_palm_ry"><inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00208916" ixy="-3.63457e-06" ixz="0.000223277" iyy="0.00182848" iyz="-1.75634e-05" izz="0.000482459"/></inertial></link>

  <joint name="WRJ0x" type="prismatic">
    <parent link="hand_root"/><child link="link_palm_x"/><origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit effort="100.0" lower="-1" upper="1" velocity="7"/><dynamics damping="10.0" friction="0.0001"/>
  </joint>
  <joint name="WRJ0y" type="prismatic">
    <parent link="link_palm_x"/><child link="link_palm_y"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit effort="100.0" lower="-1" upper="1" velocity="7"/><dynamics damping="10.0" friction="0.0001"/>
  </joint>
  <joint name="WRJ0z" type="prismatic">
    <parent link="link_palm_y"/><child link="link_palm_z"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit effort="100.0" lower="-1" upper="1" velocity="7"/><dynamics damping="10.0" friction="0.0001"/>
  </joint>
  <joint name="WRJ0rx" type="revolute">
    <parent link="link_palm_z"/><child link="link_palm_rx"/><origin xyz="0 0 0"/><axis xyz="1 0 0"/>
    <limit effort="1000" lower="-3.14" upper="3.14" velocity="7"/><dynamics damping="1.0" friction="0.0001"/>
  </joint>
  <joint name="WRJ0ry" type="revolute">
    <parent link="link_palm_rx"/><child link="link_palm_ry"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit effort="1000" lower="-3.14" upper="3.14" velocity="7"/><dynamics damping="1.0" friction="0.0001"/>
  </joint>
  <joint name="WRJ0rz" type="revolute">
    <parent link="link_palm_ry"/><child link="right_palm_link"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit effort="1000" lower="-3.14" upper="3.14" velocity="7"/><dynamics damping="1.0" friction="0.0001"/>
  </joint>

"""


def main():
    with open(SRC) as f:
        txt = f.read()

    # strip <mujoco ...> ... </mujoco> (or self-closing) block
    txt = re.sub(r"\s*<mujoco>.*?</mujoco>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"\s*<mujoco\b[^>]*/>", "", txt)

    # insert floating base right before the right_palm_link <link> definition
    m = re.search(r'<link\s+name="right_palm_link">', txt)
    if not m:
        raise RuntimeError("could not find right_palm_link link definition")
    txt = txt[:m.start()] + FLOATING_BASE.lstrip("\n") + "  " + txt[m.start():]

    with open(DST, "w") as f:
        f.write(txt)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
