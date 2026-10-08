import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command, REPO, STAIRS
from arena.g1manip import G1ManipWalker

CAM = {"cam_distance": 2.6, "azimuth": 100.0, "elevation": -10.0}

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
pol = G1ManipWalker()
init = dict(zip(pol.joints, pol.default))
arena.reset(base_z=0.76, joint_pos=init)
st = arena.state()
pol.reset(st)
arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)

frames = []
fall_t = None
seconds = 8.0
FPS = 25
save_every = max(1, round(1.0 / (arena.model.opt.timestep * FPS)))
for k in range(int(seconds / arena.model.opt.timestep)):
    st = arena.state()
    v = 0.0 if st.t < 1.0 else 0.5
    arena.set_targets(pol.joints, pol.act(st, Command(v)), pol.kp, pol.kd)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % save_every == 0:
        frames.append(arena.render(**CAM))
    arena.step_physics(1)

mp4 = os.path.join(REPO, "output/media", "g1manip_walker_stairs.mp4")
import imageio.v2 as iio
iio.mimsave(mp4, frames, fps=FPS, macro_block_size=8)
gif(mp4, os.path.join(REPO, "docs/media", "g1manip_walker_stairs.gif"), (0.0, len(frames) / FPS))
print("fell" if fall_t is not None else "no fall", fall_t, flush=True)
