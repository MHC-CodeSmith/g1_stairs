"""Kinematic preview of a dance reference (strut or groove) (upper body only, robot pinned standing) -> MP4 + contact sheet.

  docker run --rm -u 1000:1000 -v $PWD:/work -w /work --entrypoint python g1-stairs:latest g1_strut/preview_dance.py
"""
import os
import sys

import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dance  # noqa: E402
import groove  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--choreo", choices=["strut", "groove"], default="strut")
ap.add_argument("--azimuth", type=float, default=150.0)
args = ap.parse_args()
choreo = dance if args.choreo == "strut" else groove

MJCF = "/opt/TienKung-Lab/legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml"
model = mujoco.MjModel.from_xml_path(MJCF)
model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
data = mujoco.MjData(model)
adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in dance.JOINTS]
legs = {"hip_pitch": -0.2, "knee": 0.42, "ankle_pitch": -0.23}
tab = choreo.table()
r = mujoco.Renderer(model, 720, 960)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.9]
cam.distance, cam.azimuth, cam.elevation = 2.4, args.azimuth, -10
os.makedirs("output", exist_ok=True)
period = 0.8 if choreo is dance else groove.PERIOD
frames, fps = [], 30
for i in range(int(2 * period * fps)):
    ph = (i / fps / period) % 1.0
    data.qpos[:] = 0
    data.qpos[0:7] = [0, 0, 0.793, 1, 0, 0, 0]
    for side in ("left", "right"):
        for j, v in legs.items():
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}_joint")]] = v
    data.qpos[adr] = choreo.at_phase(ph, tab)
    mujoco.mj_forward(model, data)
    r.update_scene(data, camera=cam)
    frames.append(r.render())
r.close()
name = f"output/{args.choreo}_reference"
imageio.mimsave(f"{name}.mp4", frames, fps=fps, macro_block_size=8)
if choreo is dance:
    pick = [int(p * period * fps) for p in (0.0, 0.12, 0.25, 0.37, 0.5, 0.62, 0.75, 0.87)]
else:  # middle of each beat's hold
    pick = [int((k + groove.HOLD / 2) / groove.BEATS * period * fps) for k in range(groove.BEATS)]
sheet = np.concatenate([np.concatenate([frames[k][::2, ::2] for k in pick[:4]], 1),
                        np.concatenate([frames[k][::2, ::2] for k in pick[4:]], 1)], 0)
imageio.imwrite(f"{name}.png", sheet)
print("phases shown:", [round(k / fps / period, 2) for k in pick])
