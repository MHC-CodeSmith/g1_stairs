"""Who climbs stairs best: every locomotion policy in the arena on three stair protocols.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.stairs_bench --workers 14

1. humanoidbench  HumanoidBench's `stair` task geometry (4 pyramids of 5 x 0.18 m steps with 0.6 m treads, walls) and
                  its reward (humanoid_bench/envs/basic_locomotion_envs.py ClimbingUpwards, dm_control tolerances),
                  20 s at 50 Hz = 1000 steps, so the return is at most 1000. The G1 quantities follow HumanoidBench's G1:
                  head = torso_link frame + 0.5 m along its z, feet = ankle roll links, COM velocity of the pelvis
                  subtree, actuator forces of the 29 body joints. Command: 0.8 m/s forward (1 m/s gives full `move`
                  reward), heading held. HumanoidBench's own baselines are for the H1 and train on this reward; the
                  policies here never saw it, so this only reuses its course and its score.
2. sweep          8 steps up / platform / 8 down with 0.30 m treads and rises 0.08 .. 0.24 m; 0.5 m/s for 30 s.
                  Score: the tallest rise the policy crosses (up, over, down, still standing).
3. safe100        Safe100Humanoid's evaluation staircase: 6 steps of 0.13 m with 0.35 m treads, first riser 0.6 m
                  ahead; the command drops to zero on the top platform, as its navigator does after the last target.
                  Success = standing on the top platform after 12 s; 16 seeds of
                  initial pose noise (x, y +-5/8 cm, yaw +-0.08 rad, as its reset event).

Writes output/stairs_bench.json and docs/figures/stairs_*.png (arena/report_figures.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from arena.world import REPO, Arena, Command  # noqa: E402

SWEEP = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]


def tolerance(x, lo, hi, margin, sigmoid="gaussian", value_at_margin=0.1):
    """dm_control.utils.rewards.tolerance for scalars/arrays (gaussian, linear, quadratic)."""
    x = np.asarray(x, float)
    inb = (lo <= x) & (x <= hi)
    if margin == 0:
        return np.where(inb, 1.0, 0.0)
    d = np.where(x < lo, lo - x, x - hi) / margin
    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_margin))
        v = np.exp(-0.5 * (d * scale) ** 2)
    elif sigmoid == "linear":
        scale = 1 - value_at_margin
        sx = d * scale
        v = np.where(np.abs(sx) < 1, 1 - sx, 0.0)
    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_margin)
        sx = d * scale
        v = np.where(np.abs(sx) < 1, 1 - sx ** 2, 0.0)
    else:
        raise ValueError(sigmoid)
    return np.where(inb, 1.0, v)


class HBReward:
    def __init__(self, arena):
        m = arena.model
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.torso, self.lf, self.rf, self.pelvis = bid("torso_link"), bid("left_ankle_roll_link"), \
            bid("right_ankle_roll_link"), bid("pelvis")
        from arena.policies import MJ29
        self.acts = [arena.act_of_joint[arena.joint_names.index(j)] for j in MJ29]

    def __call__(self, arena):
        m, d = arena.model, arena.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        R = d.xmat[self.torso].reshape(3, 3)
        head = d.xpos[self.torso][2] + R[2, 2] * 0.5
        standing = tolerance(head - d.xpos[self.lf][2], 1.2, np.inf, 0.45) * \
            tolerance(head - d.xpos[self.rf][2], 1.2, np.inf, 0.45)
        upright = tolerance(R[2, 2], 0.5, np.inf, 1.9, "linear", 0.0)
        small = (4 + tolerance(d.actuator_force[self.acts], 0, 0, 10, "quadratic", 0.0).mean()) / 5
        move = tolerance(d.subtree_linvel[self.pelvis][0], 1.0, np.inf, 1.0, "linear", 0.0)
        move = (5 * move + 1) / 6
        return float(standing * upright * small * move), float(R[2, 2])


def episode(policy, terrain, vx, seconds, seed=0, noise=None, hb=False, video=None, x_end=None, top_x=None,
            top_z=None, stop_x=None, cam=None):
    from arena.policies import make
    arena = Arena(terrain, seed=seed, render=video is not None)
    pol = make(policy)
    init = dict(zip(pol.joints, np.asarray(pol.default, float))) if hasattr(pol, "default") else {}
    x = y = yaw = 0.0
    if noise is not None:
        rng = np.random.default_rng(noise)
        x, y, yaw = rng.uniform(-0.05, 0.05), rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
    arena.reset(base_z=0.80, joint_pos=init, x=x, y=y, yaw=yaw)
    st = arena.state()
    pol.reset(st)
    arena.set_targets(pol.joints, np.array([init.get(j, arena.hold_q[arena.joint_names.index(j)]) for j in pol.joints]),
                      pol.kp, pol.kd)
    dec = int(round(pol.control_dt / arena.model.opt.timestep))
    rew = HBReward(arena) if hb else None
    ret, max_z, fall_t, frames = 0.0, 0.0, None, []
    n = int(round(seconds / pol.control_dt))
    for k in range(n):
        st = arena.state()
        t = st.t
        q = st.base_quat
        heading = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        v = 0.0 if t < 1.0 or (stop_x is not None and st.base_pos[0] > stop_x) else vx
        cmd = Command(v, float(np.clip(-1.0 * st.base_pos[1], -0.3, 0.3)) if v else 0.0,
                      float(np.clip(-2 * heading, -1, 1)))
        arena.set_targets(pol.joints, pol.act(st, cmd), pol.kp, pol.kd)
        if hasattr(pol, "tau_ext"):   # torque-output adapters (e.g. labrob's WBC/IS-MPC), on top of PD
            arena.set_external_torque(pol.joints, pol.tau_ext)
        max_z = max(max_z, float(st.base_pos[2]))
        if fall_t is None and arena.fallen(st):
            fall_t = t
        if video is not None and k % 2 == 0:
            frames.append(arena.render(**(cam or {})))
        arena.step_physics(dec)
        if hb:
            r, upz = rew(arena)
            ret += r
            if upz < 0.1:          # HumanoidBench termination
                fall_t = fall_t or arena.t
                break
        elif fall_t is not None and arena.t > fall_t + 1.0:
            break
    st = arena.state()
    out = {"policy": policy, "terrain": terrain, "fell": fall_t is not None, "fall_t": fall_t,
           "x": float(st.base_pos[0]), "climbed": max_z - 0.80}
    if hb:
        out["hb_return"] = ret
    if x_end is not None:
        out["crossed"] = fall_t is None and st.base_pos[0] > x_end
    if top_x is not None:
        out["on_top"] = fall_t is None and top_x[0] < st.base_pos[0] < top_x[1] and st.base_pos[2] > top_z
    if video:
        import imageio.v2 as iio
        iio.mimsave(video, frames, fps=25, macro_block_size=8)
    return out


def jobs(policies):
    J = []
    for p in policies:
        J.append(("humanoidbench", p, None))
        for r in SWEEP:
            J.append(("sweep", p, r))
        for s in range(16):
            J.append(("safe100", p, s))
    return J


def work(job):
    kind, p, a = job
    try:
        if kind == "humanoidbench":
            r = episode(p, "hb_stairs", 0.8, 20.0, hb=True)
        elif kind == "sweep":
            run, n = 0.30, 8
            r = episode(p, f"steps:{a}:{run}:{n}", 0.5, 30.0, x_end=0.6 + 2 * run * n + 1.2 + 0.3)
            r["rise"] = a
        else:
            top0 = 0.6 + 6 * 0.35
            r = episode(p, "steps:0.13:0.35:6", 0.5, 12.0, noise=a, x_end=None, top_x=(top0, top0 + 1.2),
                        top_z=0.78 + 0.13 * 6 - 0.1, stop_x=top0 + 0.5)
            r["seed"] = a
        r["kind"] = kind
        return r
    except Exception as e:
        return {"kind": kind, "policy": p, "arg": a, "error": repr(e)[:300]}


def summary(res):
    by = {}
    for r in res:
        if "error" in r:
            continue
        s = by.setdefault(r["policy"], {"sweep": {}, "safe100": []})
        if r["kind"] == "humanoidbench":
            s["hb_return"], s["hb_climbed"], s["hb_fell"], s["hb_x"] = r["hb_return"], r["climbed"], r["fell"], r["x"]
        elif r["kind"] == "sweep":
            s["sweep"][r["rise"]] = bool(r["crossed"])
        else:
            s["safe100"].append(bool(r["on_top"]))
    for s in by.values():
        ok = [h for h, c in sorted(s["sweep"].items()) if c]
        s["max_rise"] = max(ok) if ok else 0.0
        s["safe100_success"] = float(np.mean(s["safe100"])) if s["safe100"] else float("nan")
    return by


if __name__ == "__main__":
    from multiprocessing import Pool

    from arena.policies import REGISTRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    pols = a.only.split(",") if a.only else list(REGISTRY)
    J = jobs(pols)
    print(f"{len(J)} runs", flush=True)
    res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, J)):
            res.append(r)
            if "error" in r:
                print("ERROR", r, flush=True)
    path = os.path.join(REPO, "output/stairs_bench.json")
    if a.only and os.path.exists(path):
        keep = [r for r in json.load(open(path)) if r.get("policy") not in pols]
        res = keep + res
    json.dump(res, open(path, "w"), indent=1, default=float)
    for p, s in sorted(summary(res).items(), key=lambda kv: -kv[1].get("hb_return", 0)):
        print(f"{p:22s} HB return {s.get('hb_return', float('nan')):7.1f}  climbed {s.get('hb_climbed', 0):.2f} m  "
              f"max rise {s['max_rise']:.2f} m  safe100 {s['safe100_success']:.0%}", flush=True)
