"""luckyrobots/g1-manipulation-challenge: RL walker (legs/waist/standing/walking/turning) + a right-arm
reacher, trained for a tabletop manipulation scene. Only the walker is used here - the reacher is a
pure arm-IK-style policy for picking up a block, out of scope for a locomotion benchmark, and the arena's
own `arm_wave` disturbance mechanism already covers "something else is moving the arms" for policies that
don't control them.

Lifted directly from g1-manipulation-challenge's run.py (the upstream repo's own standalone MuJoCo demo,
no training code needed): obs = concat(lin_vel_b, ang_vel_b, proj_gravity_b, joint_pos - default,
joint_vel, last_action, cmd[vx,vy,wz]) (99-dim), action -> target = default + action * action_scales,
exactly the offset+scale PD-target pattern every other adapter here uses. model_config.json's
obs_mean/obs_std fields are present but unused by run.py itself (probably baked into the ONNX graph
already) - not applied here either, matching upstream.
"""
from __future__ import annotations

import json
import os

import numpy as np

from arena.policies import ArenaPolicy, _onnx
from arena.world import TP

ROOT = os.path.join(TP, "g1-manipulation-challenge")
with open(os.path.join(ROOT, "model_config.json")) as f:
    _CONFIG = json.load(f)

_JOINT_NAMES = _CONFIG["joint_names"]
_DEFAULT = np.array([_CONFIG["default_joint_pos"][n] for n in _JOINT_NAMES], np.float32)
_SCALES = np.array([_CONFIG["action_scales"][n] for n in _JOINT_NAMES], np.float32)


def _pd_gains(names):
    # g1-manipulation-challenge/run.py's G1Controller._compute_pd_gains, verbatim constants
    S5020, D5020 = 14.2506, 0.9072
    S7520_14, D7520_14 = 40.1792, 2.5579
    S7520_22, D7520_22 = 99.0984, 6.3088
    S4010, D4010 = 16.7783, 1.0681
    kp, kd = np.zeros(len(names)), np.zeros(len(names))
    for i, n in enumerate(names):
        if "elbow" in n or "shoulder" in n or "wrist_roll" in n:
            kp[i], kd[i] = S5020, D5020
        elif "hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint":
            kp[i], kd[i] = S7520_14, D7520_14
        elif "hip_roll" in n or "knee" in n:
            kp[i], kd[i] = S7520_22, D7520_22
        elif "wrist_pitch" in n or "wrist_yaw" in n:
            kp[i], kd[i] = S4010, D4010
        elif "ankle" in n or n in ("waist_pitch_joint", "waist_roll_joint"):
            kp[i], kd[i] = S5020 * 2, D5020 * 2
        else:
            kp[i], kd[i] = S5020, D5020
    return kp, kd


class G1ManipWalker(ArenaPolicy):
    name, joints = "g1manip_walker", _JOINT_NAMES
    control_dt = 0.005 * 4  # 200 Hz physics (run.py sets model.opt.timestep=0.005), decimation=4
    uses = "vx vy wz"
    default = _DEFAULT

    def __init__(self):
        self.net, self.net_in = _onnx(os.path.join(ROOT, "walker.onnx"))
        self.kp, self.kd = _pd_gains(self.joints)

    def reset(self, st):
        self.last_action = np.zeros(len(self.joints), np.float32)

    def obs(self, st, cmd):
        joint_pos = st.get(st.q, self.joints) - self.default
        joint_vel = st.get(st.qd, self.joints)
        c = np.array([cmd.vx, cmd.vy, cmd.wz], np.float32)
        return np.concatenate([st.lin_vel_b, st.ang_vel_b, st.gravity_b, joint_pos, joint_vel,
                               self.last_action, c]).astype(np.float32)

    def act(self, st, cmd):
        action = self.net.run(None, {self.net_in: self.obs(st, cmd)[None]})[0][0]
        self.last_action = action.copy()
        return self.default + action * _SCALES
