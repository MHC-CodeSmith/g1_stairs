"""Skill adapters: run pretrained policies from different repos on one G1 (29-DoF) interface.

Every skill sees the same `RobotState` (joint arrays in the host env's joint order, looked up by name) and returns
absolute joint position targets for the joints it controls, plus the PD gains it was trained with. Each adapter
rebuilds its policy's native observation (term order, joint order, defaults, scales, history layout) and action
decoding; `gain_equivalent_target` converts a target made for one set of PD gains into the target that produces the
same torque under another set, so a skill can drive, or label, a robot that uses different gains.

Pure torch (no Isaac Lab import): the adapters are unit-tested against the exporters' ONNX files.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

G1_JOINTS_LAB = [  # Isaac Lab articulation order of the G1 29-DoF USD (TienKung-Lab and WBC-AGILE agree)
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


@dataclass
class RobotState:
    """Robot state in the host env's conventions. Joint arrays are (N, J) in `joint_names` order."""

    joint_names: list[str]
    q: torch.Tensor
    qd: torch.Tensor
    ang_vel_b: torch.Tensor        # (N, 3) base angular velocity, body frame
    gravity_b: torch.Tensor        # (N, 3) projected gravity, body frame
    base_height: torch.Tensor      # (N,) pelvis height above the ground under it
    time: torch.Tensor             # (N,) seconds since this skill (or episode) started

    def index(self, names: list[str]) -> torch.Tensor:
        lut = {n: i for i, n in enumerate(self.joint_names)}
        return torch.tensor([lut[n] for n in names], device=self.q.device)


def gain_equivalent_target(q_target, q, qd, kp_from, kd_from, kp_to, kd_to):
    """Target that makes a PD with (kp_to, kd_to) output the torque (kp_from, kd_from) would at this state.

    tau = kp_from (q_target - q) - kd_from qd  =  kp_to (q_new - q) - kd_to qd
    """
    tau = kp_from * (q_target - q) - kd_from * qd
    return q + (tau + kd_to * qd) / kp_to


class History:
    """Per-term observation history, oldest -> newest, filled with the first frame after a reset (Isaac Lab
    CircularBuffer semantics)."""

    def __init__(self, length: int):
        self.length, self.buf = length, None

    def push(self, x: torch.Tensor) -> torch.Tensor:
        if self.buf is None or self.buf.shape[0] != x.shape[0]:
            self.buf = x.unsqueeze(1).repeat(1, self.length, 1)
        else:
            self.buf = torch.cat([self.buf[:, 1:], x.unsqueeze(1)], dim=1)
        return self.buf

    def reset(self, env_ids=None):
        if self.buf is None:
            return
        if env_ids is None:
            self.buf = None
        else:
            self.buf[env_ids] = float("nan")  # refilled on the next push, see _refill


def _refill(h: History, x: torch.Tensor):
    """Push x; envs whose history was reset (NaN) are filled with x in every slot."""
    buf = h.push(x)
    bad = torch.isnan(buf[:, 0, 0])
    if bad.any():
        buf[bad] = x[bad].unsqueeze(1).expand(-1, h.length, -1)
    return buf


class Skill:
    name: str = "skill"
    joints: list[str] = []           # joints this skill drives (its output order)
    kp: torch.Tensor                 # (len(joints),) PD gains it was trained with
    kd: torch.Tensor

    def act(self, state: RobotState, command: torch.Tensor) -> torch.Tensor:
        """Absolute joint position targets (N, len(joints))."""
        raise NotImplementedError

    def reset(self, env_ids=None):
        pass


class AgileVelocityHeight(Skill):
    """NVIDIA WBC-AGILE `Velocity-Height-G1-History-v0` lower-body policy (12 leg joints).

    Command (N, 4): [vx, vy, yaw_rate, pelvis height] (height 0.4..0.72 m). Trained with the upper body held or,
    when standing still, randomly moved; waist roll/pitch fixed. Obs per frame (80): command 4, base ang vel 3,
    projected gravity 3, joint pos - default 29, joint vel * 0.1 (29), last raw action 12; 5 frames, term-major.
    Constants below are read from the LEAPP ONNX export (see tests/test_agile_adapter.py).
    """

    name = "agile_velocity_height"
    joints = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
              "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
              "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"]
    DEFAULT_HEIGHT = 0.72
    _default29 = {"left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1, "left_knee_joint": 0.3,
                  "right_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2}

    def __init__(self, torchscript_path: str, device="cpu", joint_vel_scale: float = 0.1):
        self.device = torch.device(device)
        self.net = torch.jit.load(torchscript_path, map_location=self.device).eval()
        self.default29 = torch.tensor([self._default29.get(n, 0.0) for n in G1_JOINTS_LAB], device=self.device)
        self.default12 = torch.tensor([self._default29.get(n, 0.0) for n in self.joints], device=self.device)
        self.scale = torch.tensor([0.22, 0.22, 0.3475, 0.3475, 0.22, 0.22, 0.1738, 0.1738, 1, 1, 1, 1.0],
                                  device=self.device)
        self.kp = torch.tensor([100, 100, 100, 100, 100, 100, 200, 200, 20, 20, 20, 20.0], device=self.device)
        self.kd = torch.tensor([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 5, 5, 0.2, 0.2, 0.1, 0.1], device=self.device)
        self.joint_vel_scale = joint_vel_scale
        self.hist = {k: History(5) for k in ("cmd", "ang", "grav", "q", "qd", "act")}
        self.last_action = None

    def reset(self, env_ids=None):
        for h in self.hist.values():
            h.reset(env_ids)
        if self.last_action is not None:
            if env_ids is None:
                self.last_action = None
            else:
                self.last_action[env_ids] = 0.0

    def observe(self, state: RobotState, command: torch.Tensor, last_action: torch.Tensor | None = None):
        """Push one frame; returns the flattened (N, 400) policy input. `last_action` overrides the stored one
        (e.g. the raw action equivalent of what another controller executed)."""
        n = state.q.shape[0]
        idx29 = state.index(G1_JOINTS_LAB)
        if last_action is None:
            last_action = self.last_action if self.last_action is not None else torch.zeros(n, 12, device=state.q.device)
        frames = {
            "cmd": command, "ang": state.ang_vel_b, "grav": state.gravity_b,
            "q": state.q[:, idx29] - self.default29, "qd": state.qd[:, idx29] * self.joint_vel_scale,
            "act": torch.clamp(last_action, -10.0, 10.0),
        }
        return torch.cat([_refill(self.hist[k], frames[k]).flatten(1) for k in self.hist], dim=1)

    @torch.no_grad()
    def act(self, state, command, last_action=None):
        x = self.observe(state, command, last_action)
        raw = self.net(x)
        self.last_action = raw
        return self.decode(raw)

    def decode(self, raw):
        return self.default12 + self.scale * torch.clamp(raw, -6.0, 6.0)

    def encode(self, targets12):
        """Raw action that decodes to `targets12` (for feeding an executed target back as 'last action')."""
        return (targets12 - self.default12) / self.scale


class DwaqPolicy(Skill):
    """G1DWAQ_Lab / TienKung-Lab DWAQ policy (ours: stairs, strut, groove, bully fine-tunes). 29 outputs.

    Obs per frame: ang vel 3, gravity 3, command [vx, vy, yaw_rate] 3, joint pos - default 29, joint vel 29,
    last raw action 29, then sin(leg phase) 2, cos(leg phase) 2 (0.8 s gait clock), then optional extras in order:
    "dance_clock" (sin/cos over `dance_period`, groove/bully tasks), "height_cmd" (command[:, 3] - 0.72, g1_body).
    Actor input = [VAE code 19, current obs]; encoder input = 5 frames.
    """

    name = "dwaq"
    joints = G1_JOINTS_LAB
    GAIT_PERIOD, GAIT_OFFSET = 0.8, 0.5
    _defaults = {".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
                 ".*_elbow_joint": 0.87, "left_shoulder_roll_joint": 0.18, "left_shoulder_pitch_joint": 0.35,
                 "right_shoulder_roll_joint": -0.18, "right_shoulder_pitch_joint": 0.35}
    # TienKung-Lab G1_CFG gains
    _kp = {"hip_yaw": 150, "hip_roll": 150, "hip_pitch": 200, "knee": 200, "waist": 200, "ankle": 20,
           "shoulder_pitch": 100, "shoulder_roll": 100, "shoulder_yaw": 50, "elbow": 50, "wrist": 40}
    _kd = {"hip_yaw": 5, "hip_roll": 5, "hip_pitch": 5, "knee": 5, "waist": 5, "ankle": 2,
           "shoulder_pitch": 2, "shoulder_roll": 2, "shoulder_yaw": 2, "elbow": 2, "wrist": 2}

    EXTRA_DIMS = {"dance_clock": 2, "height_cmd": 1}
    NOMINAL_HEIGHT = 0.72

    def __init__(self, checkpoint: str, device="cpu", dance_period: float | None = None, name: str | None = None,
                 extras: list[str] | None = None):
        import re

        self.device = torch.device(device)
        self.name = name or self.name
        sd = torch.load(checkpoint, map_location=self.device, weights_only=False)["model_state_dict"]
        self.num_obs = sd["decoder.4.weight"].shape[0]
        self.dance_period = dance_period
        self.extras = list(extras) if extras is not None else (["dance_clock"] if dance_period else [])
        want = 100 + sum(self.EXTRA_DIMS[e] for e in self.extras)
        assert self.num_obs == want, f"{checkpoint}: {self.num_obs} obs, extras {self.extras} need {want}"

        def mlp(prefix, dims, final_act=False):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2 or final_act:
                    layers.append(torch.nn.ELU())
            net = torch.nn.Sequential(*layers)
            net.load_state_dict({k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")})
            return net.to(self.device).eval()

        a_in = sd["actor.0.weight"].shape[1]
        self.actor = mlp("actor", [a_in, 512, 256, 128, 29])
        self.encoder = mlp("encoder", [sd["encoder.0.weight"].shape[1], 128, 64], final_act=True)  # ELU after 64 too
        self.enc_lat = torch.nn.Linear(64, 16).to(self.device)
        self.enc_lat.load_state_dict({"weight": sd["encode_mean_latent.weight"], "bias": sd["encode_mean_latent.bias"]})
        self.enc_vel = torch.nn.Linear(64, 3).to(self.device)
        self.enc_vel.load_state_dict({"weight": sd["encode_mean_vel.weight"], "bias": sd["encode_mean_vel.bias"]})

        def pick(table, n):
            for k, v in table.items():
                if k in n:
                    return v
            raise KeyError(n)

        def default(n):
            for pat, v in self._defaults.items():
                if re.fullmatch(pat, n):
                    return v
            return 0.0

        self.default29 = torch.tensor([default(n) for n in self.joints], device=self.device)
        self.kp = torch.tensor([float(pick(self._kp, n)) for n in self.joints], device=self.device)
        self.kd = torch.tensor([float(pick(self._kd, n)) for n in self.joints], device=self.device)
        self.hist = History(5)
        self.last_action = None

    def reset(self, env_ids=None):
        self.hist.reset(env_ids)
        if self.last_action is not None:
            if env_ids is None:
                self.last_action = None
            else:
                self.last_action[env_ids] = 0.0

    def frame(self, state: RobotState, command: torch.Tensor, last_action: torch.Tensor | None = None):
        n = state.q.shape[0]
        idx = state.index(self.joints)
        if last_action is None:
            last_action = self.last_action if self.last_action is not None else torch.zeros(n, 29, device=state.q.device)
        t = state.time
        phase_l = (t % self.GAIT_PERIOD) / self.GAIT_PERIOD
        leg = torch.stack([phase_l, (phase_l + self.GAIT_OFFSET) % 1.0], dim=1)
        parts = [state.ang_vel_b, state.gravity_b, command[:, :3], state.q[:, idx] - self.default29,
                 state.qd[:, idx], last_action, torch.sin(2 * math.pi * leg), torch.cos(2 * math.pi * leg)]
        for e in self.extras:
            if e == "dance_clock":
                ph = 2 * math.pi * (t % self.dance_period) / self.dance_period
                parts.append(torch.stack([torch.sin(ph), torch.cos(ph)], dim=1))
            elif e == "height_cmd":
                h = command[:, 3] if command.shape[1] > 3 else torch.full_like(t, self.NOMINAL_HEIGHT)
                parts.append((h - self.NOMINAL_HEIGHT).unsqueeze(1))
        return torch.clamp(torch.cat(parts, dim=1), -100.0, 100.0)

    @torch.no_grad()
    def act(self, state, command, last_action=None):
        obs = self.frame(state, command, last_action)
        hist = _refill(self.hist, obs).flatten(1)
        h = self.encoder(hist)
        code = torch.cat([self.enc_vel(h), self.enc_lat(h)], dim=1)
        raw = torch.clamp(self.actor(torch.cat([code, obs], dim=1)), -100.0, 100.0)
        self.last_action = raw
        return self.default29 + 0.25 * raw

    def encode(self, targets29):
        return (targets29 - self.default29) / 0.25


class ClipUpperBody(Skill):
    """Plays a retargeted clip (tools/retarget_g1.py output) on the upper body, eased in from the current pose."""

    name = "clip"

    def __init__(self, path: str, device="cpu", blend_in: float = 1.0, lead: float = 0.04):
        import numpy as np

        r = np.load(path)
        self.device = torch.device(device)
        self.q = torch.tensor(r["q"], dtype=torch.float32, device=self.device)
        self.dt, self.period = float(r["dt"]), float(r["period"])
        self.joints = [str(j) for j in r["joints"]]
        self.blend_in, self.lead = blend_in, lead
        self.kp = torch.tensor([DwaqPolicy._kp["waist"] if "waist" in n else
                                next(v for k, v in DwaqPolicy._kp.items() if k in n) for n in self.joints],
                               dtype=torch.float32, device=self.device)
        self.kd = torch.tensor([next(v for k, v in DwaqPolicy._kd.items() if k in n) for n in self.joints],
                               dtype=torch.float32, device=self.device)
        self.start = None

    def reset(self, env_ids=None):
        if env_ids is None or self.start is None:
            self.start = None
        else:
            self.start[env_ids] = float("nan")

    def act(self, state, command=None):
        idx = state.index(self.joints)
        if self.start is None:
            self.start = state.q[:, idx].clone()
        bad = torch.isnan(self.start[:, 0])
        if bad.any():
            self.start[bad] = state.q[bad][:, idx]
        k = ((state.time + self.lead) / self.dt).long() % self.q.shape[0]
        alpha = torch.clamp(state.time / self.blend_in, 0.0, 1.0).unsqueeze(1)
        return self.start + alpha * (self.q[k] - self.start)
