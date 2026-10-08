import os, sys
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command, REPO, STAIRS
from arena.wbmpc_centroidal import CentroidalMpc, STANDING_POSE, STANDING_BASE_Z

CAM = {"cam_distance": 3.6, "azimuth": 100.0, "elevation": -5.0}

def gif(mp4, out, span):
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    a, b = span
    vf = ("fps=10,scale=360:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48:stats_mode=diff[p];"
          "[s1][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", mp4, "-vf", vf,
                    "-loop", "0", out], check=True)

os.makedirs(os.path.join(REPO, "output/media"), exist_ok=True)
os.makedirs(os.path.join(REPO, "docs/media"), exist_ok=True)

arena = Arena(terrain="stairs", render=True)
pol = CentroidalMpc()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

FPS = 25
dt = arena.model.opt.timestep
seconds = 9.0
n = int(seconds / dt)
save_every = max(1, round(1.0 / (dt * FPS)))

RAMP_START, RAMP_END, VX_TARGET = 1.0, 4.0, 0.3
frames = []
fall_t = None
for k in range(n):
    t = k * dt
    if t < RAMP_START:
        vx = 0.0
    elif t < RAMP_END:
        vx = VX_TARGET * (t - RAMP_START) / (RAMP_END - RAMP_START)
    else:
        vx = VX_TARGET
    cmd = Command(vx=vx, height=STANDING_BASE_Z)
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, cmd), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % save_every == 0:
        frames.append(arena.render(**CAM))
    arena.step_physics(1)

mp4 = os.path.join(REPO, "output/media", "wbmpc_centroidal_stairs_attempt.mp4")
import imageio.v2 as iio
iio.mimsave(mp4, frames, fps=FPS, macro_block_size=8)
gif(mp4, os.path.join(REPO, "docs/media", "wbmpc_centroidal_stairs_attempt.gif"), (0.0, len(frames)/FPS))
print("fell" if fall_t is not None else "no fall", fall_t, flush=True)
import os as _os
_os._exit(0)
