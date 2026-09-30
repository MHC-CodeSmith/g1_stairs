"""unitreerobotics/unitree_rl_lab G1 29-DoF policies in the arena: the velocity walker (v0) and two BeyondMimic-style
dance trackers (dance_102, gangnam_style).

Rebuilt from the C++ deploy stack (deploy/include/isaaclab/{manager,envs/mdp}, deploy/include/unitree_articulation.h,
deploy/robots/g1_29dof/src/State_Mimic.cpp and include/State_Mimic.h) and each policy's params/deploy.yaml:

- Joint arrays are in IsaacLab order; `joint_ids_map[i]` is the SDK/MuJoCo motor of IsaacLab joint i. `stiffness` and
  `damping` in deploy.yaml are applied per motor (SDK order); `default_joint_pos`, action scale/offset in IsaacLab order.
- Observation terms are scaled, then kept in per-term deques of `history_length` frames (oldest first) that `reset()`
  fills with the first frame, and concatenated term by term.
- velocity_commands are clipped to the command ranges in deploy.yaml (the joystick's range).
- Action: q_target = raw * scale + offset, no clip; `last_action` is the raw action.
- Mimic: the motion CSV (60 fps: root pos, root quat xyzw, 29 joints in SDK order) is sampled with the loader's own
  rounding (index = round(t / duration * (N - 1)), blend possibly negative), joint velocities by forward difference.
  The anchor orientation compares the torso frame (pelvis quat * Rz(waist yaw) Rx(roll) Ry(pitch)) of robot and
  reference after aligning their initial yaw.
"""
from __future__ import annotations

import os

import numpy as np
import yaml

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

ROOT = os.path.join(TP, "unitree_rl_lab/deploy/robots/g1_29dof/config/policy")
MIMIC = {"dance_102": ("G1_Take_102.bvh_60hz.csv", 2.8, 24.0),
         "gangnam_style": ("G1_gangnam_style_V01.bvh_60hz.csv", 5.2, 25.5)}   # config.yaml time_start/time_end


def _onnx(path):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


# ----------------------------------------------------------------------------------- quaternions (w, x, y, z)
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmat(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_quat(axis, a):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.sin(a / 2)
    return np.array([np.cos(a / 2), *v])


def yaw_quat(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return axis_quat("z", yaw)


def slerp_eigen(q0, q1, t):
    """Eigen::Quaternion::slerp (shortest path, linear fallback near identity; t may be outside [0, 1])."""
    d = float(np.dot(q0, q1))
    ad = abs(d)
    if ad >= 1 - 1e-6:
        s0, s1 = 1 - t, t
    else:
        th = np.arccos(ad)
        s0, s1 = np.sin((1 - t) * th) / np.sin(th), np.sin(t * th) / np.sin(th)
    if d < 0:
        s1 = -s1
    return s0 * np.asarray(q0) + s1 * np.asarray(q1)


class _UnitreeLab(ArenaPolicy):
    joints = MJ29               # targets returned in SDK order

    def _load(self, d):
        cfg = yaml.safe_load(open(os.path.join(d, "params/deploy.yaml")))
        self.cfg = cfg
        self.ids = np.array(cfg["joint_ids_map"], int)                    # IsaacLab i -> SDK motor
        self.kp = np.array(cfg["stiffness"], float)                        # per SDK motor
        self.kd = np.array(cfg["damping"], float)
        self.default_lab = np.array(cfg["default_joint_pos"], float)
        a = cfg["actions"]["JointPositionAction"]
        self.scale, self.offset = np.array(a["scale"], float), np.array(a["offset"], float)
        assert a["clip"] is None
        self.terms = cfg["observations"]
        self.sess, self.inp = _onnx(os.path.join(d, "exported/policy.onnx"))
        self.default = np.zeros(29)
        self.default[self.ids] = self.offset                               # standing targets, SDK order

    def _lab(self, st):
        q, qd = st.get(st.q, MJ29), st.get(st.qd, MJ29)
        return q[self.ids], qd[self.ids]

    def _compute(self, raw_terms, first):
        """ObservationManager::compute_group: scale, push into per-term deques, concatenate term by term."""
        out = []
        for name, spec in self.terms.items():
            x = np.asarray(raw_terms[name], float)
            if spec.get("clip") is not None:
                x = np.clip(x, *spec["clip"])
            if spec.get("scale") is not None:
                x = x * np.asarray(spec["scale"], float)
            n = spec.get("history_length", 1)
            if first:
                self.hist[name] = [x] * n
            self.hist[name] = (self.hist[name] + [x])[-n:]
            out.append(np.concatenate(self.hist[name]))
        return np.concatenate(out).astype(np.float32)

    def _decode(self, raw):
        self.last = raw.astype(float)
        tgt = np.zeros(29)
        tgt[self.ids] = raw * self.scale + self.offset
        return tgt


class UnitreeLabVelocity(_UnitreeLab):
    """policy/velocity/v0: 5-frame history of gyro*0.2, gravity, clipped command, q - default, qd*0.05, raw action."""

    name = "unitree_rl_lab"
    uses = "vx vy wz (clipped to vx -0.5..1.0, vy +-0.3, wz +-0.2)"

    def __init__(self):
        self._load(os.path.join(ROOT, "velocity/v0"))
        r = self.cfg["commands"]["base_velocity"]["ranges"]
        self.lo = np.array([r["lin_vel_x"][0], r["lin_vel_y"][0], r["ang_vel_z"][0]])
        self.hi = np.array([r["lin_vel_x"][1], r["lin_vel_y"][1], r["ang_vel_z"][1]])

    def reset(self, st):
        self.last = np.zeros(29)
        self.hist = {}
        self.first = True

    def obs(self, st, cmd):
        q, qd = self._lab(st)
        terms = {"base_ang_vel": st.ang_vel_b, "projected_gravity": st.gravity_b,
                 "velocity_commands": np.clip(cmd.vec3(), self.lo, self.hi),
                 "joint_pos_rel": q - self.default_lab, "joint_vel_rel": qd, "last_action": self.last}
        o = self._compute(terms, self.first)
        self.first = False
        return o

    def act(self, st, cmd):
        raw = self.sess.run(None, {self.inp: self.obs(st, cmd)[None]})[0][0]
        return self._decode(raw)


class MotionCsv:
    """State_Mimic::MotionLoader_ (60 fps CSV: root pos 3, root quat x y z w, 29 joints in SDK order)."""

    def __init__(self, path, fps=60.0):
        d = np.loadtxt(path, delimiter=",")
        self.dt = 1.0 / fps
        self.n = len(d)
        self.duration = self.n * self.dt
        self.pos = d[:, :3]
        self.quat = d[:, [6, 3, 4, 5]]                                    # -> w x y z
        self.dof = d[:, 7:]
        self.vel = np.vstack([(self.dof[1:] - self.dof[:-1]) / self.dt, (self.dof[-1:] - self.dof[-2:-1]) / self.dt])
        self.update(0.0)

    def update(self, t):
        phase = np.clip(t / self.duration, 0.0, 1.0)
        self.i0 = int(np.round(phase * (self.n - 1)))
        self.i1 = min(self.i0 + 1, self.n - 1)
        self.blend = np.round((t - self.i0 * self.dt) / self.dt * 1e5) / 1e5

    def joint_pos(self):
        return self.dof[self.i0] * (1 - self.blend) + self.dof[self.i1] * self.blend

    def joint_vel(self):
        return self.vel[self.i0] * (1 - self.blend) + self.vel[self.i1] * self.blend

    def root_quat(self):
        return slerp_eigen(self.quat[self.i0], self.quat[self.i1], self.blend)

    def to_clip(self, t0, t1, name):
        """The played window as an arena.trackers.Clip (for the tracking metrics)."""
        from arena.trackers import Clip
        a, b = int(round(t0 / self.dt)), int(round(t1 / self.dt))
        return Clip(1.0 / self.dt, self.pos[a:b], self.quat[a:b][:, [1, 2, 3, 0]], self.dof[a:b], name)


def _torso(quat, q_sdk):
    return qmul(qmul(qmul(quat, axis_quat("z", q_sdk[12])), axis_quat("x", q_sdk[13])), axis_quat("y", q_sdk[14]))


class UnitreeLabMimic(_UnitreeLab):
    """policy/mimic/<clip>: tracks its own dance clip from time_start to time_end."""

    uses = "its own dance clip"

    def __init__(self, clip_name="dance_102"):
        d = os.path.join(ROOT, "mimic", clip_name)
        self._load(d)
        f, self.t0, self.t1 = MIMIC[clip_name]
        self.motion = MotionCsv(os.path.join(d, "params", f))
        self.name = f"unitree_{clip_name}"
        self.clip = self.motion.to_clip(self.t0, self.t1, clip_name)

    def reset(self, st):
        self.last = np.zeros(29)
        self.hist = {}
        self.k = 0
        q_sdk = st.get(st.q, MJ29)
        m = self.motion
        m.update(0.0)                  # State_Mimic::enter computes init_quat before motion->reset(time_start)
        ref_yaw = qmat(yaw_quat(m.root_quat()))
        robot_yaw = qmat(yaw_quat(_torso(st.base_quat, q_sdk)))
        self.init_R = robot_yaw @ ref_yaw.T
        self.first = True

    def obs(self, st, cmd=None):
        m = self.motion
        m.update(self.k * self.control_dt + self.t0)
        q, qd = self._lab(st)
        q_sdk = st.get(st.q, MJ29)
        jp, jv = m.joint_pos(), m.joint_vel()
        real = qmat(_torso(st.base_quat, q_sdk))
        ref = qmat(_torso(m.root_quat(), jp))
        rot = ((self.init_R @ ref).T @ real).T
        terms = {"motion_command": np.concatenate([jp[self.ids], jv[self.ids]]),
                 "motion_anchor_ori_b": [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]],
                 "base_ang_vel": st.ang_vel_b, "joint_pos_rel": q - self.default_lab, "joint_vel_rel": qd,
                 "last_action": self.last}
        o = self._compute(terms, self.first)
        self.first = False
        return o

    def act(self, st, cmd=None):
        raw = self.sess.run(None, {self.inp: self.obs(st)[None]})[0][0]
        self.k += 1
        return self._decode(raw)
