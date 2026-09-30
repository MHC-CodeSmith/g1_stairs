"""Every tracker on GRAIL's three released stair clips, on a stair height field rebuilt from the reference's own
footfalls (arena.grail.terrain_from_reference). Negative result: all trackers fall at the first step-down.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/grail_stairs_attempt.py
"""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from arena.grail import stair_clip_names, track_on_stairs  # noqa: E402

POL = ["grail_terrain", "sonic_tracking", "gmt", "twist"]


def job(a):
    p, n = a
    try:
        r = track_on_stairs(p, n)
        return {k: r[k] for k in ("policy", "clip", "fell", "fall_t", "joint_err", "root_xy_err", "yaw_err")}
    except Exception as e:
        return {"policy": p, "clip": n[-30:], "error": repr(e)[:200]}


if __name__ == "__main__":
    J = [(p, n) for n in stair_clip_names() for p in POL]
    with Pool(12) as pool:
        res = pool.map(job, J)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json.dump(res, open(os.path.join(root, "output/grail_stairs_attempt.json"), "w"), indent=1, default=float)
    for r in res:
        print(r)
