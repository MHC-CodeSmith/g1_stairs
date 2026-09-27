"""Side-by-side sheet: source clip frames (top) vs the retargeted G1 upper body, rendered kinematically (bottom).

  docker run --rm -e MUJOCO_GL=egl --runtime nvidia -v $PWD:/w -w /w --entrypoint python g1-stairs:latest \
      tools/compare_clip.py reference/bully_maguire.gif reference/bully_g1.npz output/bully_compare
"""
import sys

import imageio.v2 as iio
import mujoco
import numpy as np

clip, ref, out = sys.argv[1:4]
frames = [f[..., :3] for f in iio.get_reader(clip)]
r = np.load(ref)
q, dt, src_dt = r["q"], float(r["dt"]), float(r["source_dt"])
m = mujoco.MjModel.from_xml_path("/opt/TienKung-Lab/legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml")
m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
d = mujoco.MjData(m)
adr = [m.jnt_qposadr[m.joint(str(j)).id] for j in r["joints"]]
legs = {"hip_pitch": -0.2, "knee": 0.42, "ankle_pitch": -0.23}
ren = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.95]; cam.distance, cam.azimuth, cam.elevation = 2.2, 180, -5  # facing the robot, like the clip


def render(k):
    d.qpos[:] = 0; d.qpos[0:7] = [0, 0, 0.793, 1, 0, 0, 0]
    for s in ("left", "right"):
        for j, v in legs.items():
            d.qpos[m.jnt_qposadr[m.joint(f"{s}_{j}_joint").id]] = v
    d.qpos[adr] = q[k]
    mujoco.mj_forward(m, d)
    ren.update_scene(d, camera=cam)
    return ren.render()


pick = np.linspace(0, len(frames) - 1, 10).astype(int)
cols = []
for i in pick:
    top = frames[i]
    bot = render(int(round(i * src_dt / dt)) % len(q))
    h = 240
    top = top[np.linspace(0, top.shape[0] - 1, h).astype(int)][:, np.linspace(0, top.shape[1] - 1, 320).astype(int)]
    bot = bot[::2, ::2]
    cols.append(np.concatenate([top, bot], 0))
iio.imwrite(out + ".png", np.concatenate([np.concatenate(cols[:5], 1), np.concatenate(cols[5:], 1)], 0))
w = iio.get_writer(out + ".mp4", fps=int(round(1 / dt)), codec="libx264", quality=8, macro_block_size=8)
for k in range(len(q)):
    w.append_data(render(k))
w.close()
ren.close()
print("wrote", out + ".png", out + ".mp4")
