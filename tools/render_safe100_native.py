"""GIF of Safe100's CBF policy in its own mjlab / MuJoCo-Warp env (13 cm stairs, filter off): state recorded from the
GPU simulation, frames rendered with MuJoCo's renderer on the same compiled model.

  docker run --rm --gpus '"device=1"' -v $PWD:/workspace/g1_stairs g1-mjlab tools/render_safe100_native.py
"""
import os
import sys

import imageio.v2 as iio
import mujoco
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party/Safe100Humanoid"))
import mjlab.tasks  # noqa: E402,F401
import onnxruntime as ort  # noqa: E402
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

cfg = load_env_cfg("Unitree-G1-Stairs-CBF", play=True)
cfg.scene.num_envs = 1
cfg.scene.terrain.terrain_generator.sub_terrains["forward_stairs"].step_height_range = (0.13, 0.13)
cfg.scene.terrain.terrain_generator.num_rows = 1
cfg.actions["joint_pos"].enabled = False
env = ManagerBasedRlEnv(cfg, device="cuda:0")
sess = ort.InferenceSession(os.path.join(REPO, "third_party/Safe100Humanoid/results/models/cbf/policy.onnx"))
obs, _ = env.reset()
m = env.sim.mj_model
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 544, 960)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
cam.distance, cam.azimuth, cam.elevation = 3.6, 90.0, -12.0
frames = []
for k in range(int(10.0 / env.step_dt)):
    a = sess.run(None, {"obs": obs["actor"].cpu().numpy().astype(np.float32)})[0]
    obs, *_ = env.step(torch.as_tensor(a, device="cuda:0"))
    if k % 2 == 0:
        d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        frames.append(r.render())
out = os.path.join(REPO, "output/media/safe100_native.mp4")
iio.mimsave(out, frames, fps=25)
print("wrote", out, len(frames))
