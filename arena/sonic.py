"""NVIDIA GEAR-SONIC in the arena: kinematic planner -> motion encoder (64-d token) -> decoder (29 joint targets).

Rebuilt from the C++ ONNX reference deployment in GR00T-WholeBodyControl
(gear_sonic_deploy/src/g1/g1_deploy_onnx_ref: src/g1_deploy_onnx_ref.cpp, include/policy_parameters.hpp,
include/localmotion_kplanner{,_onnx}.hpp, include/input_interface/gamepad.hpp) and HF nvidia/GEAR-SONIC
observation_config.yaml:

- Decoder input (994): token 64 + 10-frame histories, oldest first, of pelvis gyro (30), q - default (290), qd (290),
  last raw actions (290), projected gravity (30). Joint histories are in IsaacLab joint order.
- Encoder input (1762): every encoder term laid out in config order, zeros except the ones the "g1" mode (0) uses:
  mode id, reference joint positions and velocities at 10 future frames 5 apart (IsaacLab order, absolute), and the
  reference root orientation relative to the robot's pelvis at those frames (first two rotation-matrix columns).
- Action: q_target[mj i] = default[i] + a[isaaclab index of i] * 0.25 * effort_limit / kp.
- Planner (planner_sonic.onnx, 27 modes): 4 context frames at 30 Hz -> up to 64 frames of MuJoCo qpos at 30 Hz,
  resampled to 50 Hz; re-run at 10 Hz when the command changes and every 1 s while moving; the new plan is
  cross-faded over 8 frames starting 2 frames ahead.

Velocity commands are mapped the way the gamepad handler does: facing angle integrates the yaw rate, the movement
direction is the command direction rotated by the facing angle, the speed picks slow walk / walk / run. A height
command below 0.75 m selects the squat mode with that height.
"""
from __future__ import annotations

import os

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

HF = os.path.join(TP, "hf/GEAR-SONIC")

# policy_parameters.hpp
ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])   # [mj i] -> isaac index
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])    # [isaac i] -> mj index
_ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR = (["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2 + ["7520_14", "5020", "5020"] +
          ["5020"] * 5 + ["4010"] * 2 + ["5020"] * 5 + ["4010"] * 2)
_W = 10 * 2.0 * 3.1415926535
_DOUBLE = {4, 5, 10, 11, 13, 14}              # ankles, waist roll/pitch: 2x kp and kd
KP = np.array([_ARM[m] * _W * _W * (2 if i in _DOUBLE else 1) for i, m in enumerate(_MOTOR)])
KD = np.array([2.0 * 2.0 * _ARM[m] * _W * (2 if i in _DOUBLE else 1) for i, m in enumerate(_MOTOR)])
ACTION_SCALE = np.array([0.25 * _EFFORT[m] / (_ARM[m] * _W * _W) for m in _MOTOR])
DEFAULT = np.array([-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0, 0, 0] +
                   [0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0])

ENCODER_TERMS = [  # (name, dim) in observation_config.yaml order; total 1762
    ("encoder_mode_4", 4), ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290), ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1), ("motion_anchor_orientation", 6), ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120), ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9), ("vr_3point_local_orn_target", 12), ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60), ("motion_joint_positions_wrists_10frame_step1", 60)]
ENC_OFF, _o = {}, 0
for _n, _d in ENCODER_TERMS:
    ENC_OFF[_n] = _o
    _o += _d
assert _o == 1762

MODES = {"idle": 0, "slow_walk": 1, "walk": 2, "run": 3, "squat": 4, "kneel_two": 5, "kneel_one": 6, "lying": 7,
         "crawl": 8, "idle_boxing": 9, "walk_boxing": 10, "left_punch": 11, "right_punch": 12, "random_punch": 13,
         "elbow_crawl": 14, "left_hook": 15, "right_hook": 16, "forward_jump": 17, "stealth_walk": 18,
         "injured_walk": 19, "ledge_walk": 20, "object_carry": 21, "stealth_walk_2": 22, "happy_dance_walk": 23,
         "zombie_walk": 24, "gun_walk": 25, "scare_walk": 26}
STATIC_MODES = {0, 4, 5, 6, 7, 9}


# ------------------------------------------------------------------------------------ quaternions (w, x, y, z)
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


def heading(q):
    r = qmat(q)
    return np.arctan2(r[1, 0], r[0, 0])


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ------------------------------------------------------------------------------------------------- motion
class Motion:
    """Reference motion at 50 Hz: root pos (T,3), root quat wxyz (T,4), joint pos/vel in IsaacLab order (T,29)."""

    def __init__(self, pos, quat, jpos, jvel=None):
        self.pos, self.quat, self.jpos = np.asarray(pos, float), np.asarray(quat, float), np.asarray(jpos, float)
        if jvel is None:
            jvel = np.zeros_like(self.jpos)
            jvel[:-1] = (self.jpos[1:] - self.jpos[:-1]) * 50.0
            jvel[-1] = jvel[-2]
        self.jvel = np.asarray(jvel, float)

    @property
    def T(self):
        return len(self.jpos)

    @staticmethod
    def from_mujoco_qpos_30hz(qpos):
        """Planner output (N,36) at 30 Hz -> 50 Hz (localmotion_kplanner.hpp ResampleGeneratedSequence50Hz)."""
        n = len(qpos)
        T = int(np.floor(n / 30.0 * 50))
        pos, quat, jp = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 29))
        for f in range(T):
            f30 = f / 50.0 * 30
            f0 = int(np.floor(f30))
            f1 = min(f0 + 1, n - 1)
            w1 = f30 - f0
            pos[f] = (1 - w1) * qpos[f0, :3] + w1 * qpos[f1, :3]
            quat[f] = slerp(qpos[f0, 3:7], qpos[f1, 3:7], w1)
            jp[f] = ((1 - w1) * qpos[f0, 7:] + w1 * qpos[f1, 7:])[MUJOCO_TO_ISAACLAB]
        return Motion(pos, quat, jp)

    @staticmethod
    def from_clip(clip):
        """arena.trackers.Clip (any fps, root quat xyzw, MuJoCo joint order) -> 50 Hz reference."""
        fps, dof, rp = clip.fps, clip.dof, clip.root_pos
        rq = clip.root_rot[:, [3, 0, 1, 2]]
        n = len(dof)
        T = int(np.floor((n - 1) / fps * 50)) + 1
        pos, quat, jp = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 29))
        for f in range(T):
            x = f / 50.0 * fps
            f0 = min(int(np.floor(x)), n - 1)
            f1 = min(f0 + 1, n - 1)
            w1 = x - f0
            pos[f] = (1 - w1) * rp[f0] + w1 * rp[f1]
            quat[f] = slerp(rq[f0], rq[f1], w1)
            jp[f] = ((1 - w1) * dof[f0] + w1 * dof[f1])[MUJOCO_TO_ISAACLAB]
        return Motion(pos, quat, jp)


# ------------------------------------------------------------------------------------------------- policy
class Sonic(ArenaPolicy):
    """GEAR-SONIC driven by velocity/height commands (planner) or by a reference clip (`motion`, tracking)."""

    joints = MJ29
    uses = "vx vy wz height (planner) | reference motion"
    trained_with_hands = False

    def __init__(self, clip=None, seed=1234, style: str | None = None):
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        mk = lambda f: ort.InferenceSession(os.path.join(HF, f), opt, providers=["CPUExecutionProvider"])  # noqa: E731
        self.enc, self.dec = mk("model_encoder.onnx"), mk("model_decoder.onnx")
        self.clip = clip
        self.ref = Motion.from_clip(clip) if clip is not None else None
        self.planner = None if clip is not None else mk("planner_sonic.onnx")
        self.name = "sonic_tracking" if clip is not None else ("sonic" if style is None else f"sonic_{style}")
        self.style = MODES[style] if style else None     # e.g. "happy_dance_walk" replaces slow_walk/walk
        self.seed = seed
        self.kp, self.kd = KP.copy(), KD.copy()
        self.default = DEFAULT.copy()

    # ------------------------------------------------------------------ state history (IsaacLab order)
    def _frame(self, st):
        q, qd = st.get(st.q, MJ29), st.get(st.qd, MJ29)
        return {"w": st.ang_vel_b.copy(), "q": (q - self.default)[MUJOCO_TO_ISAACLAB],
                "qd": qd[MUJOCO_TO_ISAACLAB], "a": self.last.copy(), "g": st.gravity_b.copy()}

    def reset(self, st):
        self.last = np.zeros(29)
        f = self._frame(st)
        self.hist = [f] * 10                        # Isaac Lab history buffers start filled with the first frame
        self.frame = 0
        self.tick = 0
        self.init_base_quat = st.base_quat.copy()
        self.facing = heading(st.base_quat) - heading(self.init_base_quat)   # planner frame = initial heading
        self.last_move = None
        self.replan_clock = 0.0
        self.token = np.zeros(64)
        if self.ref is not None:
            self.motion = self.ref
        else:
            q = st.get(st.q, MJ29)
            ctx = np.zeros((4, 36))
            ctx[:, 2] = 0.788740
            ctx[:, 3] = 1.0
            ctx[:, 7:] = q
            self._plan(ctx, MODES["idle"], -1.0, -1.0, np.zeros(3), np.array([1.0, 0, 0]))
            self.motion = self.new_plan
            self.new_plan = None
        self.init_ref_quat = self.motion.quat[0].copy()

    # ------------------------------------------------------------------ planner
    def _plan(self, ctx, mode, speed, height, move_dir, face_dir):
        feed = {
            "context_mujoco_qpos": ctx[None].astype(np.float32),
            "target_vel": np.array([speed], np.float32), "mode": np.array([mode], np.int64),
            "movement_direction": move_dir[None].astype(np.float32),
            "facing_direction": face_dir[None].astype(np.float32),
            "random_seed": np.array([self.seed], np.int64), "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.array([[1] * 6 + [0] * 5], np.int64),
            "height": np.array([height], np.float32)}
        qpos, n = self.planner.run(None, feed)
        n = int(np.asarray(n).reshape(-1)[0])
        assert 0 < n <= 64 and np.isfinite(qpos[0, :n]).all(), n
        self.new_plan = Motion.from_mujoco_qpos_30hz(qpos[0, :n].astype(float))

    def _context(self, gen_frame):
        """4 frames at 30 Hz from the current reference, starting at gen_frame (50 Hz index)."""
        m, ctx = self.motion, np.zeros((4, 36))
        for n in range(4):
            f50 = (gen_frame / 50.0 + n / 30.0) * 50.0
            f0 = min(int(np.floor(f50)), m.T - 1)
            f1 = min(f0 + 1, m.T - 1)
            w1 = f50 - f0
            ctx[n, 3:7] = slerp(m.quat[f0], m.quat[f1], w1)
            ctx[n, 0:3] = (1 - w1) * m.pos[f0] + w1 * m.pos[f1]
            ctx[n, 7 + MUJOCO_TO_ISAACLAB] = (1 - w1) * m.jpos[f0] + w1 * m.jpos[f1]
        return ctx

    def _movement(self, cmd):
        """Gamepad-style mapping of a body-frame velocity command to planner inputs (planner frame)."""
        v = np.hypot(cmd.vx, cmd.vy)
        face = np.array([np.cos(self.facing), np.sin(self.facing), 0.0])
        if cmd.height is not None and cmd.height < 0.75:
            return MODES["squat"], 0.0, float(cmd.height), np.zeros(3), face
        if v < 0.05:
            return MODES["idle"], -1.0, -1.0, np.zeros(3), face
        a = self.facing + np.arctan2(cmd.vy, cmd.vx)
        mode = self.style if self.style is not None else (1 if v < 0.8 else 2 if v < 1.5 else 3)
        return mode, float(v), -1.0, np.array([np.cos(a), np.sin(a), 0.0]), face

    def _planner_tick(self, cmd):
        """Planner thread body at 10 Hz (g1_deploy_onnx_ref.cpp Planner())."""
        mode, speed, height, mv, face = self._movement(cmd)
        cur = (mode, round(speed, 4), round(height, 4), tuple(np.round(mv, 4)), tuple(np.round(face, 4)))
        self.replan_clock += 0.1
        interval = 0.1 if mode == 3 else 0.2 if mode == 8 else 1.0
        time_to_replan = self.replan_clock >= interval - 1e-9
        if time_to_replan:
            self.replan_clock = 0.0
        last = self.last_move
        if last is None:
            need = True
        elif cur[0] != last[0] or cur[4] != last[4] or cur[2] != last[2]:
            need = True
        else:
            need = mode not in STATIC_MODES and (cur[1] != last[1] or cur[3] != last[3] or (time_to_replan and speed != 0))
        if not need:
            return
        self.last_move = cur
        self.gen_frame = self.frame + 2
        self._plan(self._context(self.gen_frame), mode, speed, height, mv, face)

    def _blend_new_plan(self):
        """CurrentFrameAdvancement(): rebase the reference at the current frame and cross-fade the new plan."""
        old, new, cf, fg = self.motion, self.new_plan, self.frame, self.gen_frame
        self.new_plan = None
        L = fg - cf + new.T
        if L <= 0:
            return
        start = max(0, fg - cf)
        pos, quat, jp, jv = np.zeros((L, 3)), np.zeros((L, 4)), np.zeros((L, 29)), np.zeros((L, 29))
        for f in range(L):
            fo = int(np.clip(f + cf, 0, old.T - 1))
            fn = int(np.clip(f + cf - fg, 0, new.T - 1))
            w = float(np.clip((f - start) / 8.0, 0.0, 1.0))
            jp[f] = (1 - w) * old.jpos[fo] + w * new.jpos[fn]
            jv[f] = (1 - w) * old.jvel[fo] + w * new.jvel[fn]
            pos[f] = (1 - w) * old.pos[fo] + w * new.pos[fn]
            quat[f] = slerp(old.quat[fo], new.quat[fn], w)
        self.motion = Motion(pos, quat, jp, jv)
        self.frame = 0

    # ------------------------------------------------------------------ observations
    def encoder_obs(self, st):
        m, o = self.motion, np.zeros(1762)
        o[ENC_OFF["encoder_mode_4"]] = 0.0                                  # "g1" mode
        apply = qmul(yaw_quat(heading(self.init_base_quat)), yaw_quat(-heading(self.init_ref_quat)))
        for k in range(10):
            f = min(self.frame + 5 * k, m.T - 1)
            o[ENC_OFF["motion_joint_positions_10frame_step5"] + 29 * k:][:29] = m.jpos[f]
            o[ENC_OFF["motion_joint_velocities_10frame_step5"] + 29 * k:][:29] = m.jvel[f]
            r = qmat(qmul(qconj(st.base_quat), qmul(apply, m.quat[f])))
            o[ENC_OFF["motion_anchor_orientation_10frame_step5"] + 6 * k:][:6] = r[:, :2].reshape(-1)
        return o

    def decoder_obs(self):
        h = self.hist
        return np.concatenate([self.token] + [np.concatenate([e[k] for e in h]) for k in ("w", "q", "qd", "a", "g")])

    def obs(self, st, cmd):
        return self.decoder_obs()

    def act(self, st, cmd):
        # 1. log the state (with the previous action) into the history
        self.hist = self.hist[1:] + [self._frame(st)]
        # 2. planner at 10 Hz (runs before the control tick that consumes it)
        if self.planner is not None and self.tick % 5 == 0:
            self._planner_tick(cmd)
            if self.new_plan is not None:
                self._blend_new_plan()
        if self.planner is not None:
            self.facing += cmd.wz * self.control_dt
        # 3. encoder -> token, 4. decoder -> action
        self._enc_obs = self.encoder_obs(st)
        self.token = self.enc.run(None, {"obs_dict": self._enc_obs[None].astype(np.float32)})[0][0].astype(float)
        self._dec_obs = self.decoder_obs()
        a = self.dec.run(None, {"obs_dict": self._dec_obs[None].astype(np.float32)})[0][0].astype(float)
        self.last = a.copy()
        # 5. advance the reference
        self.frame = min(self.frame + 1, self.motion.T - 1)
        self.tick += 1
        return self.default + a[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
