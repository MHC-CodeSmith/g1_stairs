"""Record the arena runs shown in docs/ARENA_REPORT.md and convert them to GIFs in docs/media/.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/make_media.py --workers 14 [--only name,...]

Each job re-runs a scored test with rendering on (same code path as the scorecard), writes an MP4 to output/media/ and
a GIF (360 px wide, 10 fps, 48-colour palette, at most 10 s) to docs/media/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import REPO  # noqa: E402

SIDE = {"cam_distance": 3.6, "azimuth": 90.0, "elevation": -12.0}
FRONT = {"cam_distance": 3.0, "azimuth": 150.0, "elevation": -15.0}
LOCO = ["g1_body", "g1dwaq_stairs", "agile_vel_height", "gr00t_wbc", "sonic", "holosoma_fastsac", "holosoma_ppo",
        "unitree_rl_lab", "safe100_cbf", "safe100_nominal", "mujoco_playground", "unitree_rl_gym",
        "g1_walk37_baseline", "g1_walk37_robust"]


def jobs():
    J = []
    for p in LOCO:
        J.append((f"stairs_{p}", "run", dict(policy_name=p, terrain="stairs", schedule="stairs", seconds=14, cam=SIDE),
                  (1.0, 13.0)))
    for p in ["g1_body", "g1dwaq_stairs", "agile_vel_height", "holosoma_fastsac", "sonic", "safe100_cbf"]:
        J.append((f"hb_{p}", "hb", dict(policy=p), (0.5, 12.5)))
    for p in ["g1_body", "gr00t_wbc", "agile_vel_height", "sonic"]:
        J.append((f"crouch_{p}", "run", dict(policy_name=p, terrain="flat", schedule="crouch", seconds=14, cam=FRONT),
                  (1.5, 13.5)))
    for p in ["g1_body", "g1dwaq_stairs", "sonic", "agile_vel_height"]:
        J.append((f"push_{p}", "run", dict(policy_name=p, terrain="flat", schedule="push", seconds=25, cam=FRONT),
                  (8.0, 24.0)))
    for p in ["g1_body", "holosoma_fastsac", "sonic", "agile_vel_height"]:
        J.append((f"rough_{p}", "run", dict(policy_name=p, terrain="rough", schedule="walk", seconds=20, cam=FRONT),
                  (2.0, 14.0)))
    for p in ["g1_body", "gr00t_wbc", "agile_vel_height", "unitree_rl_gym"]:
        J.append((f"arms_{p}", "run", dict(policy_name=p, terrain="flat", schedule="walk", seconds=12, arms=True,
                                           cam=FRONT), (1.0, 11.0)))
    for p in ["sonic_tracking", "gmt", "twist"]:
        J.append((f"track_dance_{p}", "track", dict(policy=p, clip="gmt_dance"), (0.0, 12.0)))
        J.append((f"track_kick_{p}", "track", dict(policy=p, clip="gmt_kick_walk"), (0.0, 6.0)))
    for p in ["unitree_dance_102", "unitree_gangnam_style"]:
        J.append((f"track_{p}", "track", dict(policy=p, clip=None), (0.0, 12.0)))
    for r in (0.12, 0.18, 0.22):
        for p in ["g1_body", "g1dwaq_stairs", "safe100_cbf"]:
            J.append((f"sweep{int(r * 100)}_{p}", "sweep", dict(policy=p, rise=r), (1.0, 13.0)))
    J.append(("walk_sonic_planner", "run", dict(policy_name="sonic", terrain="flat", schedule="walk", seconds=20,
                                                cam=FRONT), (1.0, 19.0)))
    return J


def gif(mp4, out, span):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    a, b = span
    b = min(b, a + 10.0)
    vf = ("fps=10,scale=360:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48:stats_mode=diff[p];"
          "[s1][p]paletteuse=dither=none:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(a), "-t", str(b - a), "-i", mp4, "-vf", vf, "-loop", "0",
                    out], check=True)


def work(job):
    name, kind, kw, span = job
    mp4 = os.path.join(REPO, "output/media", name + ".mp4")
    try:
        if kind == "run":
            from arena.run import run
            run(video=mp4, **kw)
        elif kind == "hb":
            from arena.stairs_bench import episode
            episode(kw["policy"], "hb_stairs", 0.8, 13.0, hb=True, video=mp4, cam=SIDE)
        elif kind == "sweep":
            from arena.stairs_bench import episode
            episode(kw["policy"], f"steps:{kw['rise']}:0.3:8", 0.5, 14.0, video=mp4, cam=SIDE)
        elif kind == "track":
            from arena.track import load_clip, track
            clip = load_clip(kw["clip"]) if kw["clip"] else None
            track(kw["policy"], clip, video=mp4, cam=FRONT)
        gif(mp4, os.path.join(REPO, "docs/media", name + ".gif"), span)
        return name, os.path.getsize(os.path.join(REPO, "docs/media", name + ".gif")) / 1e6
    except Exception as e:
        return name, repr(e)[:200]


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", default=None)
    ap.add_argument("--exclude", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(REPO, "output/media"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "docs/media"), exist_ok=True)
    J = jobs()
    if a.only:
        J = [j for j in J if any(o in j[0] for o in a.only.split(","))]
    if a.exclude:
        J = [j for j in J if not any(o in j[0] for o in a.exclude.split(","))]
    print(len(J), "clips", flush=True)
    with Pool(a.workers) as pool:
        for name, r in pool.imap_unordered(work, J):
            print(name, f"{r:.2f} MB" if isinstance(r, float) else r, flush=True)
