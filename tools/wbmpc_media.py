import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command, REPO  # noqa: E402
from arena.wbmpc import WbMpc, STANDING_POSE, STANDING_BASE_Z  # noqa: E402

CAM = {"cam_distance": 2.2, "azimuth": 100.0, "elevation": -10.0}


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
pol = WbMpc()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)
arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
arena.set_external_torque(pol.joints, pol.tau_ext)

frames = []
fall_t = None
seconds = 8.0
for k in range(int(seconds / arena.model.opt.timestep)):
    if k % 1 == 0:
        st = arena.state()
        arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
        arena.set_external_torque(pol.joints, pol.tau_ext)
        if fall_t is None and arena.fallen(st):
            fall_t = st.t
        if k % 10 == 0:
            frames.append(arena.render(**CAM))
    arena.step_physics(1)

mp4 = os.path.join(REPO, "output/media", "wbmpc_standing.mp4")
import imageio.v2 as iio
iio.mimsave(mp4, frames, fps=25, macro_block_size=8)
gif(mp4, os.path.join(REPO, "docs/media", "wbmpc_standing.gif"), (0.0, seconds))
print("fell" if fall_t is not None else "no fall", fall_t)
final = arena.state()
print("final height:", final.height, "base_pos:", list(final.base_pos))
import os as _os
_os._exit(0)
