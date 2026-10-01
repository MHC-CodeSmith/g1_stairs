import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.grail import stair_clip_names, track_on_stairs  # noqa: E402
from arena.world import REPO  # noqa: E402

CAM = {"cam_distance": 3.2, "azimuth": 90.0, "elevation": -8.0}


def gif(mp4, out, span):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    a, b = span
    b = min(b, a + 10.0)
    vf = ("fps=10,scale=360:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48:stats_mode=diff[p];"
          "[s1][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", mp4, "-vf", vf,
                    "-loop", "0", out], check=True)


os.makedirs(os.path.join(REPO, "output/media"), exist_ok=True)
os.makedirs(os.path.join(REPO, "docs/media"), exist_ok=True)
name0 = stair_clip_names()[0]
for p, span in [("sonic_tracking", (0.0, 8.0)), ("grail_terrain", (0.0, 7.0))]:
    mp4 = os.path.join(REPO, "output/media", f"gstairs_boxes_{p}.mp4")
    r = track_on_stairs(p, name0, video=mp4, cam=CAM, boxes=True)
    gif(mp4, os.path.join(REPO, "docs/media", f"gstairs_boxes_{p}.gif"), span)
    print(p, "fell" if r["fell"] else "no fall", r.get("fall_t"))
