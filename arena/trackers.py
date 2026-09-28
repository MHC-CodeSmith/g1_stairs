"""General motion trackers in the arena: GMT and TWIST (SONIC's tracking mode is in arena/sonic.py).

One reference format for every tracker, `Clip`: fps, root position (T,3), root quaternion xyzw (T,4) and the 29 body
joint angles in MuJoCo order (T,29). Each adapter turns the clip into its repo's own motion file and reads it back with
that repo's own MotionLib (velocities, smoothing, interpolation and looping stay theirs); the observation and action
code follows the repo's MuJoCo sim2sim script:

- GMT (humanoid-general-motion-tracking sim2sim.py): 23 joints (no wrists), reference at 20 future steps
  [1, 5, ..., 95] x (root z, roll, pitch, local root velocity, yaw rate, 23 joints), proprioception
  (gyro*0.25, roll, pitch, q - default, qd*0.05 with ankle velocities zeroed, last action) and the 20 previous
  proprioception frames. Action * 0.5 + default.
- TWIST (deploy_real/server_{high_level_motion_lib,low_level_g1_sim}.py): 23 joints + wrist roll driven directly from
  the reference, reference at the next step (root z, roll, pitch, yaw, local root velocity, yaw rate, 23 joints),
  the same proprioception, and the 10 previous (reference + proprioception) frames. Action * 0.5 + default.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from collections import deque

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

GMT_ROOT = os.path.join(TP, "humanoid-general-motion-tracking")
TWIST_ROOT = os.path.join(TP, "TWIST")
J23 = [n for n in MJ29 if "wrist" not in n]
IDX23 = [MJ29.index(n) for n in J23]


class Clip:
    def __init__(self, fps, root_pos, root_rot_xyzw, dof29, name="clip"):
        self.fps, self.name = float(fps), name
        self.root_pos = np.asarray(root_pos, float)
        self.root_rot = np.asarray(root_rot_xyzw, float)
        self.dof = np.asarray(dof29, float)

    @property
    def seconds(self):
        return (len(self.dof) - 1) / self.fps

    @staticmethod
    def from_sonic_pkl(path):
        import joblib
        v = next(iter(joblib.load(path).values()))
        return Clip(v["fps"], v["root_trans_offset"], v["root_rot"], v["dof"], os.path.basename(path)[:-4])

    @staticmethod
    def from_gmt_pkl(path):
        v = pickle.load(open(path, "rb"))
        dof = np.zeros((len(v["dof_pos"]), 29))
        dof[:, IDX23] = v["dof_pos"]
        return Clip(v["fps"], v["root_pos"], v["root_rot"], dof, os.path.basename(path)[:-4])

    def anchored(self):
        """Same motion, starting at x = y = 0 with zero heading (what the robot does at reset)."""
        x, y, z, w = self.root_rot[0]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(-yaw), np.sin(-yaw)
        pos = self.root_pos.copy()
        pos[:, :2] -= pos[0, :2]
        pos[:, :2] = pos[:, :2] @ np.array([[c, -s], [s, c]]).T
        qz = np.array([0, 0, np.sin(-yaw / 2), np.cos(-yaw / 2)])      # xyzw
        rot = np.array([_qmul_xyzw(qz, q) for q in self.root_rot])
        return Clip(self.fps, pos, rot, self.dof, self.name)

    def write_gmt(self, path):
        """GMT / TWIST motion file (23 joints; body positions unused by the trackers, zero-filled)."""
        T = len(self.dof)
        d = {"fps": self.fps, "root_pos": self.root_pos.astype(np.float32),
             "root_rot": self.root_rot.astype(np.float32), "dof_pos": self.dof[:, IDX23].astype(np.float32),
             "local_body_pos": np.zeros((T, 38, 3), np.float32), "link_body_list": ["pelvis"] * 38}
        with open(path, "wb") as f:
            pickle.dump(d, f)
        return path


def _qmul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _euler_xyzw(q):
    x, y, z, w = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _rotate_inverse_xyzw(q, v):
    x, y, z, w = q
    qv = np.array([x, y, z])
    return v * (2 * w * w - 1) - np.cross(qv, v) * w * 2 + qv * np.dot(qv, v) * 2


def _rpy_wxyz(q):
    return np.array(_euler_xyzw([q[1], q[2], q[3], q[0]]))


def _motion_lib(root, module, clip):
    """The repo's own MotionLib on a temporary motion file written from `clip`."""
    import importlib
    import torch  # noqa: F401  (MotionLib uses torch)
    if root not in sys.path:
        sys.path.insert(0, root)
    if "isaacgym" not in sys.modules:   # TWIST's torch_utils star-imports isaacgym.torch_utils
        import types
        from arena import _isaacgym_torch_utils
        sys.modules["isaacgym"] = types.ModuleType("isaacgym")
        sys.modules["isaacgym.torch_utils"] = _isaacgym_torch_utils
    ml = importlib.import_module(module).MotionLib
    path = clip.write_gmt(os.path.join(tempfile.mkdtemp(), f"{clip.name}.pkl"))
    return ml(path, "cpu")


class _Tracker(ArenaPolicy):
    uses = "reference motion"
    default: np.ndarray

    def _frame_ref(self, times):
        import torch
        ids = torch.zeros(len(times), dtype=torch.int)
        out = self.ml.calc_motion_frame(ids, torch.tensor(times, dtype=torch.float))
        return [o.numpy() for o in out[:5]]        # root_pos, root_rot (xyzw), root_vel, root_ang_vel, dof_pos

    def _proprio(self, st):
        q, qd = st.get(st.q, J23), st.get(st.qd, J23)
        qd = qd.copy()
        qd[[4, 5, 10, 11]] = 0.0
        return np.concatenate([st.ang_vel_b * 0.25, _rpy_wxyz(st.base_quat)[:2], q - self.default, qd * 0.05,
                               self.last]).astype(np.float32)


class Gmt(_Tracker):
    name, joints = "gmt", J23
    steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    def __init__(self, clip: Clip):
        import torch
        self.torch = torch
        self.clip = clip
        self.ml = _motion_lib(GMT_ROOT, "utils.motion_lib", clip)
        self.net = torch.jit.load(os.path.join(GMT_ROOT, "assets/pretrained_checkpoints/pretrained.pt"),
                                  map_location="cpu")
        self.kp = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40] * 8, float)
        self.kd = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5] * 8, float)
        self.default = np.array([-0.2, 0, 0, 0.4, -0.2, 0] * 2 + [0, 0, 0] + [0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2])

    def reset(self, st):
        self.last = np.zeros(23, np.float32)
        self.hist = deque([np.zeros(74)] * 20, maxlen=20)
        self.k = 0

    def mimic(self):
        t = [(self.k + s) * self.control_dt for s in self.steps]
        pos, rot, vel, ang, dof = self._frame_ref(t)
        rows = []
        for i in range(len(t)):
            r, p, _ = _euler_xyzw(rot[i])
            rows.append(np.concatenate([[pos[i, 2], r, p], _rotate_inverse_xyzw(rot[i], vel[i]),
                                        [_rotate_inverse_xyzw(rot[i], ang[i])[2]], dof[i]]))
        return np.concatenate(rows)

    def obs(self, st, cmd=None):
        prop = self._proprio(st)
        return np.concatenate([self.mimic(), prop, np.array(self.hist).flatten()]).astype(np.float32), prop

    def act(self, st, cmd=None):
        o, prop = self.obs(st)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(o)[None]).numpy()[0]
        self.last = a.astype(np.float32)
        self.hist.append(prop)
        self.k += 1
        return np.clip(a, -10, 10) * 0.5 + self.default


class Twist(_Tracker):
    name = "twist"
    joints = J23 + ["left_wrist_roll_joint", "right_wrist_roll_joint"]

    def __init__(self, clip: Clip):
        import torch
        self.torch = torch
        self.clip = clip
        self.ml = _motion_lib(os.path.join(TWIST_ROOT, "pose"), "pose.utils.motion_lib_pkl", clip)
        self.net = torch.jit.load(os.path.join(TWIST_ROOT, "assets/twist_general_motion_tracker.pt"), map_location="cpu")
        # server_low_level_g1_sim.py gains are in its 25-joint order (arm = 4 joints + wrist roll)
        kp25 = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40, 40, 40, 40, 20] * 2, float)
        kd25 = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5, 5, 5, 5, 1] * 2, float)
        body = [i for i in range(25) if i not in (19, 24)]
        self.kp = np.concatenate([kp25[body], kp25[[19, 24]]])
        self.kd = np.concatenate([kd25[body], kd25[[19, 24]]])
        self.default = np.array([-0.2, 0, 0, 0.4, -0.2, 0] * 2 + [0, 0, 0] + [0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2])

    def reset(self, st):
        self.last = np.zeros(23, np.float32)
        self.hist = deque([np.zeros(105)] * 10, maxlen=10)
        self.k = 0

    def mimic(self):
        pos, rot, vel, ang, dof = self._frame_ref([(self.k + 1) * self.control_dt])
        r, p, y = _euler_xyzw(rot[0])
        return np.concatenate([[pos[0, 2], r, p, y], _rotate_inverse_xyzw(rot[0], vel[0]),
                               [_rotate_inverse_xyzw(rot[0], ang[0])[2]], dof[0]])   # 31; wrist rolls are 0

    def obs(self, st, cmd=None):
        full = np.concatenate([self.mimic(), self._proprio(st)])
        return np.concatenate([full, np.array(self.hist).flatten()]).astype(np.float32), full

    def act(self, st, cmd=None):
        o, full = self.obs(st)
        self.hist.append(full)
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(o)[None]).numpy()[0]
        self.last = a.astype(np.float32)
        self.k += 1
        return np.concatenate([np.clip(a, -10, 10) * 0.5 + self.default, [0.0, 0.0]])
