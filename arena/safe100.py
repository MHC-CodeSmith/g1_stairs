"""lzqw/Safe100Humanoid stair policies (CBF-trained and nominal) in the arena.

From its mjlab task `Unitree-G1-Stairs-{CBF,Nominal}` (src/tasks/stairs_cbf/config.py on top of
src/tasks/velocity/velocity_env_cfg.py, src/assets/robots/unitree_g1/g1_constants.py) and mjlab 1.2.0's observation
manager:

- Actor (405): per term, 5 frames oldest first (mjlab CircularBuffer, filled with the first frame after a reset),
  terms in order: gyro at imu_in_pelvis 3, projected gravity 3, command [vx, vy, wz] 3, gait phase [sin, cos] of
  (t mod 0.6)/0.6 (zero when |command| < 0.1) 2, q - HOME keyframe 29, qd 29, last raw action 12. Joints in MuJoCo
  order. No noise at play time.
- Action: 12 leg joints (MuJoCo order), q_target = HOME + 0.25 * effort_limit / stiffness * raw. The rest of the body
  holds the HOME keyframe. PD gains: stiffness = armature (2 pi 10 Hz)^2, damping = 2 * 2 * armature * 2 pi 10 Hz,
  armature from the two-stage planetary rotor inertias; waist roll/pitch and ankles use twice the 5020 values.
- The runtime CBF filter (which needs the stair riser positions) is not run here: Safe100 reports the same 95.3%
  success with the filter on and off, and the arena policies get no terrain knowledge either.
"""
from __future__ import annotations

import os
import re

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

ROOT = os.path.join(TP, "Safe100Humanoid/results/models")


def _refl(rotor, gear):
    return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


ARM = {"5020": _refl((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 1 + 56 / 16)),
       "7520_14": _refl((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
       "7520_22": _refl((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
       "4010": _refl((0.068e-4, 0.0, 0.0), (1, 5, 5))}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
GROUPS = [  # (patterns, motor, multiplier) as G1_ARTICULATION
    ((r".*_elbow_joint", r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint",
      r".*_wrist_roll_joint"), "5020", 1),
    ((r".*_hip_pitch_joint", r".*_hip_yaw_joint", r"waist_yaw_joint"), "7520_14", 1),
    ((r".*_hip_roll_joint", r".*_knee_joint"), "7520_22", 1),
    ((r".*_wrist_pitch_joint", r".*_wrist_yaw_joint"), "4010", 1),
    ((r"waist_pitch_joint", r"waist_roll_joint"), "5020", 2),
    ((r".*_ankle_pitch_joint", r".*_ankle_roll_joint"), "5020", 2),
]
HOME = {r".*_hip_pitch_joint": -0.1, r".*_knee_joint": 0.3, r".*_ankle_pitch_joint": -0.2,
        r".*_shoulder_pitch_joint": 0.35, r".*_elbow_joint": 0.87, r"left_shoulder_roll_joint": 0.18,
        r"right_shoulder_roll_joint": -0.18}
W = 10 * 2.0 * 3.1415926535


def _group(name):
    for pats, motor, mult in GROUPS:
        if any(re.fullmatch(p, name) for p in pats):
            return motor, mult
    raise KeyError(name)


def _home(name):
    for p, v in HOME.items():
        if re.fullmatch(p, name):
            return v
    return 0.0


class Safe100(ArenaPolicy):
    uses = "vx vy wz"
    joints = MJ29
    LEGS = MJ29[:12]

    def __init__(self, variant="cbf"):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(os.path.join(ROOT, variant, "policy.onnx"), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.name = f"safe100_{variant}"
        kp, kd, scale = [], [], []
        for n in MJ29:
            m, k = _group(n)
            a = ARM[m] * k
            kp.append(a * W * W)
            kd.append(2 * 2.0 * a * W)
            scale.append(0.25 * EFFORT[m] * k / (a * W * W))
        self.kp, self.kd = np.array(kp), np.array(kd)
        self.scale = np.array(scale)[:12]
        self.default = np.array([_home(n) for n in MJ29])

    def reset(self, st):
        self.last = np.zeros(12)
        self.hist = None
        self.t0 = st.t
        self.k = 0

    def obs(self, st, cmd):
        q, qd = st.get(st.q, MJ29), st.get(st.qd, MJ29)
        c = cmd.vec3()
        ph = ((self.k * self.control_dt) % 0.6) / 0.6
        phase = np.zeros(2) if np.linalg.norm(c) < 0.1 else np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
        terms = [st.ang_vel_b, st.gravity_b, c, phase, q - self.default, qd, self.last]
        if self.hist is None:
            self.hist = [[t] * 5 for t in terms]
        else:
            self.hist = [h[1:] + [t] for h, t in zip(self.hist, terms)]
        return np.concatenate([np.concatenate(h) for h in self.hist]).astype(np.float32)

    def act(self, st, cmd):
        raw = self.sess.run(None, {self.inp: self.obs(st, cmd)[None]})[0][0].astype(float)
        self.last = raw
        self.k += 1
        tgt = self.default.copy()
        tgt[:12] = self.default[:12] + self.scale * raw
        return tgt
