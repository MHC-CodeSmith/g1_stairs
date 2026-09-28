"""Motion-tracking test: play a reference clip through each tracker and measure how closely the robot follows it.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.track --policy sonic_tracking,gmt,twist --clip all

Clips (all re-anchored to start at the origin facing +x): SONIC's sample walk (40 s) and GMT's example motions.
Metrics over the frames before any fall: mean absolute error of the 23 joints every tracker controls (wrists excluded)
[rad], root xy error [m] (mean and final), root height error [m], heading error [rad] (mean and final).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from arena.policies import MJ29, TRACKERS, make_tracker
from arena.trackers import GMT_ROOT, IDX23, Clip
from arena.world import TP, Arena, Command


def clips():
    out = {"sonic_walk": os.path.join(TP, "hf/GEAR-SONIC/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl")}
    for p in sorted(glob.glob(os.path.join(GMT_ROOT, "assets/motions/*.pkl"))):
        out["gmt_" + os.path.basename(p)[:-4]] = p
    return out


def load_clip(name):
    p = clips()[name]
    return (Clip.from_sonic_pkl(p) if name.startswith("sonic") else Clip.from_gmt_pkl(p)).anchored()


def track(policy_name, clip: Clip, video=None, terrain="flat", verbose=False, lookahead=2.0):
    pol = make_tracker(policy_name, clip)
    arena = Arena(terrain, render=video is not None)
    # reference sampled at 50 Hz
    T = int((clip.seconds - lookahead) * 50)          # stop before the trackers' look-ahead runs off the clip
    t = np.arange(T) / 50.0
    x = np.clip(t * clip.fps, 0, len(clip.dof) - 1)
    f0 = np.floor(x).astype(int)
    f1 = np.minimum(f0 + 1, len(clip.dof) - 1)
    w = (x - f0)[:, None]
    ref_q = (1 - w) * clip.dof[f0] + w * clip.dof[f1]
    ref_p = (1 - w) * clip.root_pos[f0] + w * clip.root_pos[f1]
    x_, y_, z_, w_ = clip.root_rot[0]
    arena.reset(base_z=float(ref_p[0, 2]) + 0.03, joint_pos=dict(zip(MJ29, ref_q[0])),
                yaw=float(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))))
    st = arena.state()
    pol.reset(st)
    arena.set_targets(pol.joints, np.array([ref_q[0][MJ29.index(j)] for j in pol.joints]), pol.kp, pol.kd)
    dec = int(round(pol.control_dt / arena.model.opt.timestep))
    ref_r = clip.root_rot[np.rint(x).astype(int)]
    ref_yaw = np.arctan2(2 * (ref_r[:, 3] * ref_r[:, 2] + ref_r[:, 0] * ref_r[:, 1]),
                         1 - 2 * (ref_r[:, 1] ** 2 + ref_r[:, 2] ** 2))
    jerr, perr, herr, yerr, frames, fall_t = [], [], [], [], [], None
    for k in range(T):
        st = arena.state()
        jerr.append(np.mean(np.abs(st.get(st.q, MJ29)[IDX23] - ref_q[k][IDX23])))
        perr.append(np.linalg.norm(st.base_pos[:2] - ref_p[k, :2]))
        herr.append(abs(st.base_pos[2] - ref_p[k, 2]))
        q = st.base_quat
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        yerr.append(abs((yaw - ref_yaw[k] + np.pi) % (2 * np.pi) - np.pi))
        if arena.fallen(st):
            fall_t = st.t
            break
        arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
        if video is not None and k % 2 == 0:
            frames.append(arena.render())
        arena.step_physics(dec)
    res = {"policy": policy_name, "clip": clip.name, "seconds": round(T / 50, 1), "fell": fall_t is not None,
           "fall_t": fall_t, "joint_err": float(np.mean(jerr)), "root_xy_err": float(np.mean(perr)),
           "root_xy_err_final": float(perr[-1]), "root_z_err": float(np.mean(herr)),
           "yaw_err": float(np.mean(yerr)), "yaw_err_final": float(yerr[-1])}
    if video:
        import imageio.v2 as iio
        iio.mimsave(video, frames, fps=25, macro_block_size=8)
    if verbose:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=",".join(TRACKERS))
    ap.add_argument("--clip", default="sonic_walk", help="clip name, comma list, or 'all'")
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    names = list(clips()) if a.clip == "all" else a.clip.split(",")
    for c in names:
        clip = load_clip(c)
        for n in a.policy.split(","):
            try:
                track(n, clip, a.video, verbose=True)
            except Exception as e:
                print({"policy": n, "clip": c, "error": repr(e)[:300]}, flush=True)
