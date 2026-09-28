"""Run one policy on one terrain with a command schedule; returns metrics and (optionally) a video.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.run --policy holosoma_fastsac --terrain flat
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from arena.policies import REGISTRY, make
from arena.world import STAIRS, Arena, Command

# (start s, Command) - piecewise constant
SCHEDULES = {
    "walk": [(0, Command()), (2, Command(vx=0.5)), (7, Command(vx=1.0)), (11, Command(vx=0.5, wz=0.5)),
             (15, Command(vx=0.0, vy=0.3)), (18, Command())],
    "stairs": [(0, Command()), (1.5, Command(vx=0.6))],
    "stand": [(0, Command())],
}


def cmd_at(schedule, t):
    c = schedule[0][1]
    for t0, cc in schedule:
        if t >= t0:
            c = cc
    return c


def run(policy_name, terrain="flat", schedule="walk", seconds=20.0, video=None, push=None, seed=0,
        heading_hold=True, verbose=False):
    arena = Arena(terrain, seed=seed, render=video is not None)
    pol = make(policy_name)
    init = {}
    if hasattr(pol, "default"):
        init = dict(zip(pol.joints, np.asarray(pol.default, float)))
    arena.reset(base_z=0.80, joint_pos=init)
    if push:
        arena.push = push
    st = arena.state()
    pol.reset(st)
    arena.set_targets(pol.joints, np.array([init.get(j, arena.hold_q[arena.joint_names.index(j)]) for j in pol.joints]),
                      pol.kp, pol.kd)
    dec = int(round(pol.control_dt / arena.model.opt.timestep))
    sched = SCHEDULES[schedule]
    frames, log = [], []
    fall_t = None
    for k in range(int(seconds / arena.model.opt.timestep)):
        if k % dec == 0:
            st = arena.state()
            cmd = cmd_at(sched, st.t)
            if heading_hold and cmd.wz == 0 and schedule == "stairs":   # keep straight on the stairs
                yaw = math.atan2(2 * (st.base_quat[0] * st.base_quat[3] + st.base_quat[1] * st.base_quat[2]),
                                 1 - 2 * (st.base_quat[2] ** 2 + st.base_quat[3] ** 2))
                cmd = Command(cmd.vx, float(np.clip(-1.0 * st.base_pos[1], -0.3, 0.3)), float(np.clip(-2 * yaw, -1, 1)),
                              cmd.height, cmd.rpy)
            arena.set_targets(pol.joints, pol.act(st, cmd), pol.kp, pol.kd)
            log.append((st.t, *st.base_pos, st.height, *st.lin_vel_b, st.ang_vel_b[2], cmd.vx, cmd.vy, cmd.wz))
            if fall_t is None and arena.fallen(st):
                fall_t = st.t
            if fall_t is not None and st.t > fall_t + 1.0:
                break
            if video is not None and k % (dec * 2) == 0:
                frames.append(arena.render())
        arena.step_physics(1)
    L = np.array(log)
    # columns: t, x, y, z, height, vx_b, vy_b, vz_b, wz_b, cmd vx, cmd vy, cmd wz
    t, vx, vy, wz = L[:, 0], L[:, 5], L[:, 6], L[:, 8]
    cvx, cvy, cwz = L[:, 9], L[:, 10], L[:, 11]
    steady = np.zeros(len(t), bool)
    for i, (t0, _) in enumerate(sched):          # skip the first 1.5 s after each command change
        t1 = sched[i + 1][0] if i + 1 < len(sched) else 1e9
        steady |= (t >= t0 + 1.5) & (t < t1)
    ok = steady & (np.arange(len(t)) < (np.searchsorted(t, fall_t) if fall_t else len(t)))
    res = {"policy": policy_name, "terrain": terrain, "schedule": schedule, "fell": fall_t is not None,
           "fall_t": fall_t, "dist_x": float(L[-1, 1] - L[0, 1]), "final_height": float(L[-1, 4]),
           "vx_err": float(np.mean(np.abs(vx[ok] - cvx[ok]))) if ok.any() else float("nan"),
           "vy_err": float(np.mean(np.abs(vy[ok] - cvy[ok]))) if ok.any() else float("nan"),
           "wz_err": float(np.mean(np.abs(wz[ok] - cwz[ok]))) if ok.any() else float("nan"),
           "max_z": float(L[:, 3].max())}
    if terrain == "stairs":
        res["reached_top"] = bool(L[:, 3].max() > STAIRS["top"] + 0.55)
        res["crossed"] = bool(res["reached_top"] and not res["fell"] and L[-1, 1] > STAIRS["end"])
    if video:
        import imageio.v2 as iio
        iio.mimsave(video, frames, fps=int(round(1 / (pol.control_dt * 2))), macro_block_size=8)
    if verbose:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="all")
    ap.add_argument("--terrain", default="flat")
    ap.add_argument("--schedule", default="walk")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    names = list(REGISTRY) if a.policy == "all" else a.policy.split(",")
    for n in names:
        try:
            run(n, a.terrain, a.schedule, a.seconds, video=a.video, verbose=True)
        except Exception as e:  # keep going through the list
            print({"policy": n, "error": repr(e)[:300]})
