"""Adapters: each pretrained policy rebuilt behind one interface for the arena.

    p = make("holosoma_fastsac"); p.reset(state)
    joints, q_target = p.joints, p.act(state, Command(vx=0.5))      # every p.control_dt seconds
    kp, kd = p.kp, p.kd                                              # PD gains the policy was trained with

Each adapter reproduces its repo's own observation (term order, joint order, defaults, scales, clocks, history) and
action decoding, citing the file it follows. `obs(state, cmd)` is exposed separately so tests can compare it with the
repo's own code (arena/check_adapters.py).
"""
from __future__ import annotations

import json
import math
import os
from collections import deque

import numpy as np

from arena.world import TP, Command, State

MJ29 = [  # MuJoCo / Unitree SDK order of the G1 29-DoF body
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LEGS12 = MJ29[:12]


def _onnx(path):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return s, name


class ArenaPolicy:
    name = "policy"
    joints: list[str] = []
    control_dt = 0.02
    kp: np.ndarray
    kd: np.ndarray
    uses = ""            # what the policy needs from the command
    trained_with_hands = False

    def reset(self, st: State):
        pass

    def obs(self, st: State, cmd: Command) -> np.ndarray:
        raise NotImplementedError

    def act(self, st: State, cmd: Command) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- unitree_rl_gym
class RlGym(ArenaPolicy):
    """unitree_rl_gym deploy/deploy_mujoco/deploy_mujoco.py + configs/g1.yaml (legs only, arms not in its model)."""

    name, joints = "unitree_rl_gym", LEGS12
    uses = "vx vy wz"

    def __init__(self):
        import torch
        self.torch = torch
        self.net = torch.jit.load(os.path.join(TP, "unitree_rl_gym/deploy/pre_train/g1/motion.pt"), map_location="cpu")
        self.kp = np.array([100, 100, 100, 150, 40, 40] * 2, float)
        self.kd = np.array([2, 2, 2, 4, 2, 2] * 2, float)
        self.default = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, np.float32)
        self.cmd_scale = np.array([2.0, 2.0, 0.25], np.float32)

    def reset(self, st):
        self.action = np.zeros(12, np.float32)

    def obs(self, st, cmd):
        q, qd = st.get(st.q, self.joints), st.get(st.qd, self.joints)
        phase = (st.t % 0.8) / 0.8
        return np.concatenate([st.ang_vel_b * 0.25, st.gravity_b, cmd.vec3() * self.cmd_scale, q - self.default,
                               qd * 0.05, self.action, [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]]
                              ).astype(np.float32)

    def act(self, st, cmd):
        with self.torch.no_grad():
            self.action = self.net(self.torch.from_numpy(self.obs(st, cmd))[None]).numpy()[0]
        return self.action * 0.25 + self.default


# ------------------------------------------------------------------------------------------ mujoco_playground
def _playground_model():
    import mujoco
    d = os.path.join(TP, "mujoco_playground/mujoco_playground/_src/locomotion/g1/xmls")
    return mujoco.MjModel.from_xml_path(os.path.join(d, "scene_mjx_feetonly_flat_terrain.xml"))


class PlaygroundPolicy(ArenaPolicy):
    """mujoco_playground experimental/sim2sim/play_g1_joystick.py (OnnxController). Gains = the position actuators'
    kp and the joints' damping in its g1_mjx_feetonly.xml."""

    name, joints = "mujoco_playground", MJ29
    uses = "vx vy wz"

    def __init__(self):
        m = _playground_model()
        self.default = np.array(m.keyframe("knees_bent").qpos[7:], np.float32)
        names = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
        assert names == MJ29, names
        self.kp = np.array([m.actuator_gainprm[a, 0] for a in range(m.nu)], float)
        self.kd = np.array([m.dof_damping[m.jnt_dofadr[m.actuator_trnid[a, 0]]] for a in range(m.nu)], float)
        self.sess, self.inp = _onnx(os.path.join(TP, "mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx"))
        self.phase_dt = 2 * np.pi * 1.5 * self.control_dt
        site = m.site("imu_in_pelvis")      # its velocimeter/gyro sit at this site on the pelvis
        assert m.body(site.bodyid).name == "pelvis" and np.allclose(site.quat, [1, 0, 0, 0])
        self.imu_offset = np.array(site.pos)

    def reset(self, st):
        self.last = np.zeros(29, np.float32)
        self.phase = np.array([0.0, np.pi])

    def obs(self, st, cmd):
        q, qd = st.get(st.q, self.joints), st.get(st.qd, self.joints)
        lin = st.lin_vel_b + np.cross(st.ang_vel_b, self.imu_offset)    # velocity of the IMU site
        return np.hstack([lin, st.ang_vel_b, st.gravity_b, cmd.vec3(), q - self.default, qd, self.last,
                          np.cos(self.phase), np.sin(self.phase)]).astype(np.float32)

    def act(self, st, cmd):
        a = self.sess.run(["continuous_actions"], {self.inp: self.obs(st, cmd)[None]})[0][0]
        self.last = a.copy()
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        return a * 0.5 + self.default


# ------------------------------------------------------------------------------------ g1_walk_isaaclab_mujoco
G1WALK_ALIAS = {"torso_joint": "waist_yaw_joint", "elbow_pitch": "elbow", "elbow_roll": "wrist_roll",
                "zero": "hand_thumb_0", "one": "hand_thumb_1", "two": "hand_thumb_2", "three": "hand_index_0",
                "four": "hand_index_1", "five": "hand_middle_0", "six": "hand_middle_1"}


def g1walk_to_mj(name):
    if name == "torso_joint":
        return "waist_yaw_joint"
    side, rest = name.split("_", 1)
    for k, v in G1WALK_ALIAS.items():
        if rest == f"{k}_joint":
            return f"{side}_{v}_joint"
    return name


class G1Walk37(ArenaPolicy):
    """g1_walk_isaaclab_mujoco mujoco/mujoco_eval/{policy,run_grid}.py: 37 joints (old Isaac G1 + Dex3, waist
    roll/pitch and wrist pitch/yaw absent), obs = lin vel, ang vel, gravity, command, q - default, qd, last action."""

    uses = "vx vy wz"
    trained_with_hands = True

    def __init__(self, variant="robust"):
        import torch
        self.torch = torch
        root = os.path.join(TP, "g1_walk_isaaclab_mujoco/mujoco/policies")
        meta = json.load(open(os.path.join(root, "isaac_metadata.json")))
        self.src_names = meta["joint_names"]
        self.joints = [g1walk_to_mj(n) for n in self.src_names]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.scale = float(meta["action_scale"])
        f = {"baseline": "baseline/policy_actor.pt", "robust": "robust/policy_actor_robust.pt"}[variant]
        self.net = torch.jit.load(os.path.join(root, f), map_location="cpu").eval()
        self.name = f"g1_walk37_{variant}"
        kp, kd = [], []
        for n in self.src_names:          # run_grid.pd_gains (on the original names)
            if "ankle" in n:
                kp.append(20.0); kd.append(2.0)
            elif "hip" in n or "knee" in n or "torso" in n:
                kp.append(200.0 if ("pitch" in n or "knee" in n or "torso" in n) else 150.0); kd.append(5.0)
            else:
                kp.append(40.0); kd.append(10.0)
        self.kp, self.kd = np.array(kp), np.array(kd)

    def reset(self, st):
        self.last = np.zeros(37, np.float32)

    def obs(self, st, cmd):
        q, qd = st.get(st.q, self.joints), st.get(st.qd, self.joints)
        return np.concatenate([st.lin_vel_b, st.ang_vel_b, st.gravity_b, cmd.vec3(), q - self.default, qd,
                               self.last]).astype(np.float32)

    def act(self, st, cmd):
        with self.torch.no_grad():
            a = self.net(self.torch.from_numpy(self.obs(st, cmd))[None]).numpy()[0]
        self.last = a.astype(np.float32)
        return self.default + self.scale * a


# ------------------------------------------------------------------------------------ GR00T decoupled WBC
class Gr00tWbc(ArenaPolicy):
    """GR00T-WholeBodyControl decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py + g1_gear_wbc.yaml: legs +
    waist (15), 6-frame history of 86-dim obs, Balance policy when |command| <= 0.05 else Walk. Commands: velocity,
    pelvis height (0.74 default), torso roll/pitch/yaw."""

    name, joints = "gr00t_wbc", MJ29[:15]
    uses = "vx vy wz height rpy"
    trained_with_hands = True   # hands are rigid masses in its g1_gear_wbc.xml

    def __init__(self):
        d = os.path.join(TP, "GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy")
        self.balance = _onnx(os.path.join(d, "GR00T-WholeBodyControl-Balance.onnx"))
        self.walk = _onnx(os.path.join(d, "GR00T-WholeBodyControl-Walk.onnx"))
        self.kp = np.array([150, 150, 150, 200, 40, 40] * 2 + [250, 250, 250], float)
        self.kd = np.array([2, 2, 2, 4, 2, 2] * 2 + [5, 5, 5], float)
        self.default = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0, 0, 0], np.float32)
        self.cmd_scale = np.array([2.0, 2.0, 0.5], np.float32)

    def reset(self, st):
        self.action = np.zeros(15, np.float32)
        self.hist = deque([np.zeros(86, np.float32)] * 6, maxlen=6)

    def single_obs(self, st, cmd):
        q, qd = st.get(st.q, MJ29), st.get(st.qd, MJ29)
        pad = np.zeros(29, np.float32)
        pad[:15] = self.default
        o = np.zeros(86, np.float32)
        o[0:3] = cmd.vec3() * self.cmd_scale
        o[3] = 0.74 if cmd.height is None else cmd.height
        o[4:7] = cmd.rpy
        o[7:10] = st.ang_vel_b * 0.5
        o[10:13] = st.gravity_b
        o[13:42] = q - pad
        o[42:71] = qd * 0.05
        o[71:86] = self.action
        return o

    def obs(self, st, cmd):
        self.hist.append(self.single_obs(st, cmd))
        return np.concatenate(list(self.hist))

    def act(self, st, cmd):
        sess, inp = self.balance if np.linalg.norm(cmd.vec3()) <= 0.05 else self.walk
        self.action = sess.run(None, {inp: self.obs(st, cmd)[None]})[0][0].astype(np.float32)
        return self.action * 0.25 + self.default


# ------------------------------------------------------------------------------------------------- holosoma
class Holosoma(ArenaPolicy):
    """holosoma_inference policies/{base,locomotion}.py with config_values loco-g1-29dof: terms sorted by name and
    scaled, gait phase 1 s (both feet at pi when the command is zero), action * 0.25 + default. kp/kd/dof names come
    from the ONNX metadata."""

    uses = "vx vy wz"

    def __init__(self, variant="fastsac"):
        import onnxruntime as ort
        p = os.path.join(TP, f"holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/{variant}_g1_29dof.onnx")
        self.sess, self.inp = _onnx(p)
        md = self.sess.get_modelmeta().custom_metadata_map
        self.joints = json.loads(md["dof_names"])
        self.kp, self.kd = np.array(json.loads(md["kp"]), float), np.array(json.loads(md["kd"]), float)
        self.default = np.array([-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0, 0, 0] +
                                [0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0], np.float32)
        assert self.joints == MJ29
        self.name = f"holosoma_{variant}"
        self.phase_dt = 2 * np.pi / (50 * 1.0)

    def reset(self, st):
        self.last = np.zeros(29, np.float32)
        self.phase = np.array([0.0, np.pi])
        self.standing = False

    def _update_phase(self, cmd):
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        v = cmd.vec3()
        if np.linalg.norm(v[:2]) < 0.01 and abs(v[2]) < 0.01:
            self.phase = np.array([np.pi, np.pi])
            self.standing = True
        elif self.standing:
            self.phase = np.array([0.0, np.pi])
            self.standing = False

    def obs(self, st, cmd):
        q, qd = st.get(st.q, self.joints), st.get(st.qd, self.joints)
        v = cmd.vec3()
        terms = {  # sorted by name, as BasePolicy.obs_terms_sorted
            "actions": self.last, "base_ang_vel": st.ang_vel_b * 0.25, "command_ang_vel": v[2:3],
            "command_lin_vel": v[:2], "cos_phase": np.cos(self.phase), "dof_pos": q - self.default,
            "dof_vel": qd * 0.05, "projected_gravity": st.gravity_b, "sin_phase": np.sin(self.phase)}
        return np.concatenate([terms[k] for k in sorted(terms)]).astype(np.float32)

    def act(self, st, cmd):
        self._update_phase(cmd)
        self._last_obs = self.obs(st, cmd)
        a = np.clip(self.sess.run(None, {self.inp: self._last_obs[None]})[0][0], -100, 100)
        self.last = a.astype(np.float32)
        return a * 0.25 + self.default


# ------------------------------------------------------------------------- our skills layer (Isaac-trained)
class _SkillWrap(ArenaPolicy):
    """Wraps skills/adapters.py skills (Isaac Lab trained) as arena policies."""

    def _rs(self, st):
        import torch
        from skills.adapters import RobotState
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)[None]  # noqa: E731
        return RobotState(st.names, f(st.q), f(st.qd), f(st.ang_vel_b), f(st.gravity_b),
                          torch.tensor([st.height], dtype=torch.float32), torch.tensor([self.t], dtype=torch.float32))

    def reset(self, st):
        self.skill.reset()
        self.t0 = st.t

    def act(self, st, cmd):
        import torch
        self.t = st.t - self.t0
        c = [cmd.vx, cmd.vy, cmd.wz] + ([cmd.height if cmd.height is not None else 0.72] if self.height_cmd else [])
        out = self.skill.act(self._rs(st), torch.tensor([c], dtype=torch.float32))[0].numpy()
        return out[self._sel] if hasattr(self, "_sel") else out


class Dwaq(_SkillWrap):
    """G1DWAQ_Lab stair policy (TienKung-Lab DWAQ), via skills.adapters.DwaqPolicy."""

    name, uses, height_cmd = "g1dwaq_stairs", "vx vy wz", False

    def __init__(self):
        from skills import registry
        self.skill = registry.load("dwaq_upstream")
        self.joints = list(self.skill.joints)
        self.kp, self.kd = self.skill.kp.numpy().astype(float), self.skill.kd.numpy().astype(float)


class Agile(_SkillWrap):
    """NVIDIA WBC-AGILE Velocity-Height-G1-History-v0, via skills.adapters.AgileVelocityHeight."""

    name, uses, height_cmd = "agile_vel_height", "vx vy wz height", True

    def __init__(self):
        from skills import registry
        self.skill = registry.load("agile_velocity_height")
        self.joints = list(self.skill.joints)
        self.kp, self.kd = self.skill.kp.numpy().astype(float), self.skill.kd.numpy().astype(float)


G1BODY_UPPER = [  # g1_rl/body.py UPPER_JOINTS (that module needs Isaac Lab)
    "waist_yaw_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class G1Body(_SkillWrap):
    """Our distilled g1_body student (teachers G1DWAQ + GR00T WBC), legs + waist pitch; the rest of the upper body is
    left to whatever drives it (held pose, or the arena's arm waving)."""

    name, uses, height_cmd = "g1_body", "vx vy wz height", True

    def __init__(self):
        from skills import registry
        self.skill = registry.load("g1_body_student")
        allj = list(self.skill.joints)
        self._sel = [i for i, j in enumerate(allj) if j not in G1BODY_UPPER]
        self.joints = [allj[i] for i in self._sel]
        self.kp = self.skill.kp.numpy().astype(float)[self._sel]
        self.kd = self.skill.kd.numpy().astype(float)[self._sel]


REGISTRY = {
    "unitree_rl_gym": RlGym,
    "mujoco_playground": PlaygroundPolicy,
    "g1_walk37_baseline": lambda: G1Walk37("baseline"),
    "g1_walk37_robust": lambda: G1Walk37("robust"),
    "gr00t_wbc": Gr00tWbc,
    "holosoma_fastsac": lambda: Holosoma("fastsac"),
    "holosoma_ppo": lambda: Holosoma("ppo"),
    "g1dwaq_stairs": Dwaq,
    "agile_vel_height": Agile,
    "sonic": lambda: _sonic()(),
    "g1_body": G1Body,
    "unitree_rl_lab": lambda: _ul().UnitreeLabVelocity(),
    "safe100_cbf": lambda: _s100()("cbf"),
    "safe100_nominal": lambda: _s100()("nominal"),
    "labrob": lambda: _labrob()(),
    "wbmpc": lambda: _wbmpc()(),
    "romoco": lambda: _romoco()(),
}


def _s100():
    from arena.safe100 import Safe100
    return Safe100

def _labrob():
    from arena.labrob import Labrob
    return Labrob

def _wbmpc():
    from arena.wbmpc import WbMpc
    return WbMpc

def _romoco():
    from arena.romoco import RoMoCo
    return RoMoCo
FIXED_CLIP = ["unitree_dance_102", "unitree_gangnam_style"]   # trackers bound to their own clip: make_fixed(name)


def _ul():
    from arena import unitree_lab
    return unitree_lab


def make_fixed(name) -> ArenaPolicy:
    return _ul().UnitreeLabMimic(name.replace("unitree_", ""))
TRACKERS = ["sonic_tracking", "gmt", "twist", "grail_terrain"]   # take a reference clip: make_tracker(name, clip)


def _sonic():
    from arena.sonic import Sonic
    return Sonic


def make_tracker(name, clip) -> ArenaPolicy:
    from arena.trackers import Gmt, Twist
    if name == "grail_terrain":
        from arena.grail import GrailTerrain
        return GrailTerrain(clip)
    return {"sonic_tracking": lambda: _sonic()(clip=clip), "gmt": lambda: Gmt(clip), "twist": lambda: Twist(clip)}[name]()


def make(name) -> ArenaPolicy:
    return REGISTRY[name]()
