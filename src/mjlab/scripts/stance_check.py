# stance_check.py
import mujoco
from mjlab.asset_zoo.robots.caltech_c1.c1_constants import C1_XML

m = mujoco.MjModel.from_xml_path(str(C1_XML))
d = mujoco.MjData(m)


def show(L_hip, L_knee, R_hip, R_knee):
  d.qpos[:] = 0.0
  for name, v in [
    ("L_hip_joint", L_hip),
    ("L_knee_joint", L_knee),
    ("R_hip_joint", R_hip),
    ("R_knee_joint", R_knee),
  ]:
    d.qpos[m.joint(name).qposadr[0]] = v
  mujoco.mj_forward(m, d)
  base = m.body("base_link").id
  L = d.site_xpos[m.site("left_foot_contact").id]
  R = d.site_xpos[m.site("right_foot_contact").id]
  print(f"foot x: L {L[0]:+.3f}  R {R[0]:+.3f}   COM x: {d.subtree_com[base][0]:+.3f}")
  print(f"foot height mismatch: {abs(L[2] - R[2]) * 1000:.1f} mm")
  print(f"keyframe z: {d.xpos[base][2] - min(L[2], R[2]) + 0.001:.4f}")


show(-0.498, 0.8, -0.498, 0.8)  # current
show(-0.55, 0.60, -0.10, 0.63)  # staggered candidate
