import mujoco
from PIL import Image

from mjlab.asset_zoo.robots.caltech_c1.c1_constants import get_c1_robot_cfg
from mjlab.entity import Entity

# Build the model through mjlab so actuators and the init_state keyframe are included.
model = Entity(get_c1_robot_cfg()).spec.compile()
model.vis.global_.offwidth = 1600
model.vis.global_.offheight = 1600
model.vis.scale.jointlength = 0.3
model.vis.scale.jointwidth = 0.3

data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
mujoco.mj_forward(model, data)


# Side view of the sagittal plane.
cam = mujoco.MjvCamera()
cam.lookat[:] = data.xpos[model.body("base_link").id] - [0, 0, 0.35]
cam.distance = 2.2
cam.azimuth = 130
cam.elevation = -15

opt = mujoco.MjvOption()
opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = (
  True  # draw joint axes; set False for a plain render
)

with mujoco.Renderer(model, height=1600, width=1600) as renderer:
  renderer.update_scene(data, camera=cam, scene_option=opt)
  Image.fromarray(renderer.render()).save("c1_render.png")

print(dict(zip([model.joint(i).name for i in range(model.njnt)], data.qpos)))
