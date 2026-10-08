import os, sys
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command, REPO
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

CAM = {"cam_distance": 3.2, "azimuth": 100.0, "elevation": -5.0}

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

arena = Arena(terrain="flat", render=True)
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

FPS = 25
dt = arena.model.opt.timestep
seconds = 11.0
n = int(seconds / dt)
# save exactly FPS frames per sim-second so the mp4 plays back in real time
save_every = max(1, round(1.0 / (dt * FPS)))

frames = []
fall_t = None
last_trigger = -1.0
PERIOD = 0.8
for k in range(n):
    t = k * dt
    if t >= 2.0 and t - last_trigger >= PERIOD:
        pol._wm.trigger_walk()
        last_trigger = t
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % save_every == 0:
        frames.append(arena.render(**CAM))
    arena.step_physics(1)

print(f"frames saved: {len(frames)}, expected mp4 duration: {len(frames)/FPS:.2f}s (should be ~{seconds}s)", flush=True)

mp4 = os.path.join(REPO, "output/media", "labrob_walk_trigger2.mp4")
import imageio.v2 as iio
iio.mimsave(mp4, frames, fps=FPS, macro_block_size=8)
gif(mp4, os.path.join(REPO, "docs/media", "labrob_walk_trigger.gif"), (0.0, len(frames)/FPS))
print("fell" if fall_t is not None else "no fall", fall_t, flush=True)
import os as _os
_os._exit(0)
