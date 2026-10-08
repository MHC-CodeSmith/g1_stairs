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
arena.tau_rate_limit = 500.0  # Nm/s - prevents the WBC's torque-command blow-up from launching the
                               # robot; does not fix the underlying fall, just makes it look like real
                               # physics instead of a numerical explosion. See docs/ARENA_REPORT.md.
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

FPS = 25
dt = arena.model.opt.timestep
seconds = 13.0
n = int(seconds / dt)
save_every = max(1, round(1.0 / (dt * FPS)))

frames = []
fall_t = None
triggered = False
for k in range(n):
    t = k * dt
    if not triggered and t >= 2.0:
        pol._wm.trigger_walk()
        triggered = True
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % save_every == 0:
        frames.append(arena.render(**CAM))
    arena.step_physics(1)

mp4 = os.path.join(REPO, "output/media", "labrob_walk_trigger4.mp4")
import imageio.v2 as iio
iio.mimsave(mp4, frames, fps=FPS, macro_block_size=8)
gif(mp4, os.path.join(REPO, "docs/media", "labrob_walk_trigger.gif"), (0.0, len(frames)/FPS))
print("fell" if fall_t is not None else "no fall", fall_t, flush=True)
import os as _os
_os._exit(0)
