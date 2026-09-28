"""MuJoCo arena: one G1 model, one control loop, any pretrained policy through an adapter (arena/policies.py).

Robot: Unitree's official G1 29-DoF + Dex3-1 hands (43 motors; unitree_rl_gym g1_29dof_with_hand_rev_1_0.xml), with
torque motors, the model's joint torque limits, and rotor armature per motor class (values used by mujoco_playground and
Isaac Lab). Physics 500 Hz; policies run at their own rate (50 Hz for all current adapters) and output joint position
targets that a per-joint PD turns into torques every physics step. Joints no active policy owns are held at a hold pose.

A `native` model path can be given instead, to run a policy on the exact model it shipped with (adapter checks).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(REPO, "third_party")
G1_DEX3_XML = os.path.join(TP, "unitree_rl_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml")
G1_29_XML = os.path.join(TP, "unitree_rl_gym/resources/robots/g1_description/g1_29dof_rev_1_0.xml")

ARMATURE = [  # (substring, armature)  first match wins; Unitree motor reflected inertias
    ("hip_roll", 0.025101925), ("knee", 0.025101925),
    ("hip_pitch", 0.01017752004), ("hip_yaw", 0.01017752004), ("waist_yaw", 0.01017752004),
    ("ankle", 0.00721945), ("waist_roll", 0.00721945), ("waist_pitch", 0.00721945),
    ("wrist_pitch", 0.00425), ("wrist_yaw", 0.00425),
    ("hand_", 0.001),
    ("shoulder", 0.003609725), ("elbow", 0.003609725), ("wrist_roll", 0.003609725),
]
HOLD_POSE = {  # arms relaxed at the sides, elbows bent, hands open
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 0.9, "right_elbow_joint": 0.9,
}
HOLD_GAINS = [  # (substring, kp, kd) for joints held by the arena
    ("hand_", 2.0, 0.05), ("wrist", 20.0, 1.0), ("shoulder", 60.0, 2.0), ("elbow", 40.0, 2.0),
    ("waist", 150.0, 5.0), ("hip", 150.0, 5.0), ("knee", 150.0, 5.0), ("ankle", 30.0, 2.0),
]


def _match(table, name, default=None):
    for key, *vals in table:
        if key in name:
            return vals[0] if len(vals) == 1 else vals
    return default


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float | None = None          # pelvis height above ground [m]; None = policy default
    rpy: tuple = (0.0, 0.0, 0.0)          # torso roll/pitch/yaw offsets (GR00T)

    def vec3(self):
        return np.array([self.vx, self.vy, self.wz], dtype=np.float32)


@dataclass
class State:
    """Robot state at the start of a control step. Joint arrays follow `Arena.joint_names`."""
    t: float
    q: np.ndarray
    qd: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray        # (w, x, y, z) pelvis
    lin_vel_b: np.ndarray        # pelvis linear velocity, pelvis frame
    lin_vel_w: np.ndarray
    ang_vel_b: np.ndarray        # pelvis angular velocity, pelvis frame
    gravity_b: np.ndarray        # projected gravity, pelvis frame
    torso_quat: np.ndarray
    torso_ang_vel_b: np.ndarray
    height: float                # pelvis height above the ground below it
    names: list = field(default_factory=list)

    def get(self, arr, joints):
        idx = [self.names.index(j) for j in joints]
        return arr[idx]


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v * (2 * w * w - 1) - 2 * w * np.cross(qv, v) + 2 * qv * np.dot(qv, v)


def add_terrain(spec: mujoco.MjSpec, kind: str, seed: int = 0):
    """Floor/terrain geoms, materials and light added to `spec`. The robot starts at x=0 facing +x."""
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.32, .34, .38], rgb2=[.26, .28, .32], width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.1)
    spec.add_material(name="step", rgba=[.55, .56, .58, 1])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.6, .75, .9], rgb2=[.1, .15, .2], width=256, height=256)
    wb = spec.worldbody
    light = wb.add_light(pos=[0, -2, 6], dir=[0, .3, -1])
    if hasattr(mujoco, "mjtLightType"):          # MuJoCo >= 3.4
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        light.directional = True
    fr = [1.0, 0.005, 0.0001]
    if kind == "rough":  # 16 x 16 m height field, up to 6 cm relief, smoothed
        rng = np.random.default_rng(seed)
        n = 161
        z = rng.uniform(-1, 1, (n, n))
        for _ in range(2):
            z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
        z = (z - z.min()) / (z.max() - z.min())
        z[74:87, 74:87] = z[74:87, 74:87].mean()   # flat spawn patch at the centre
        spec.add_hfield(name="rough", size=[8, 8, 0.06, 0.1], nrow=n, ncol=n, userdata=z.flatten().tolist())
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="rough",
                    pos=[0, 0, -0.06 * float(z[80, 80])], material="grid", friction=fr)
        return
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", friction=fr)
    if kind == "stairs":  # 10 up (0.15 rise x 0.31 tread), 1.2 m platform at 1.5 m, 10 down; first riser at x=1.2
        x0, rise, run, w = 1.2, 0.15, 0.31, 1.6

        def box(cx, h):
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[run / 2, w / 2, h / 2], pos=[cx, 0, h / 2],
                        material="step", friction=fr)
        for i in range(10):
            box(x0 + run * (i + 0.5), rise * (i + 1))
        top_x = x0 + run * 10
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, w / 2, rise * 5], pos=[top_x + 0.6, 0, rise * 5],
                    material="step", friction=fr)
        for i in range(10):
            box(top_x + 1.2 + run * (i + 0.5), rise * (10 - i))
    elif kind != "flat":
        raise ValueError(kind)


TERRAIN_GROUP = 3


def _terrain_group(spec, before):
    for g in spec.worldbody.geoms:
        if g not in before:
            g.group = TERRAIN_GROUP


STAIRS = {"start": 1.2, "end": 1.2 + 3.1 + 1.2 + 3.1, "top": 1.5}


def state_from(m, d, pelvis, torso, qadr, vadr, names, t) -> State:
    """State from any MjData (also used to feed adapters the states of a repo's own simulation loop).

    After mj_step the derived fields (xquat, cvel, ...) still describe the pre-step state while qpos/qvel are
    post-step; recompute them so every quantity refers to the same instant (as reading qpos/qvel directly does)."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    quat = d.xquat[pelvis].copy()
    vel, velw, tvel = np.zeros(6), np.zeros(6), np.zeros(6)
    X = mujoco.mjtObj.mjOBJ_XBODY    # the body frame (mjOBJ_BODY would be the rotated/offset inertial frame)
    mujoco.mj_objectVelocity(m, d, X, pelvis, vel, 1)    # local orientation
    mujoco.mj_objectVelocity(m, d, X, pelvis, velw, 0)
    if torso >= 0:
        mujoco.mj_objectVelocity(m, d, X, torso, tvel, 1)
    pos = d.xpos[pelvis].copy()
    return State(t=t, q=d.qpos[qadr].copy(), qd=d.qvel[vadr].copy(), base_pos=pos, base_quat=quat,
                 lin_vel_b=vel[3:6].copy(), lin_vel_w=velw[3:6].copy(), ang_vel_b=vel[0:3].copy(),
                 gravity_b=quat_rotate_inverse(quat, np.array([0, 0, -1.0])),
                 torso_quat=d.xquat[torso].copy() if torso >= 0 else quat, torso_ang_vel_b=tvel[0:3].copy(),
                 height=float(pos[2]), names=list(names))


def joint_index(m, names=None):
    """(names, qpos addresses, dof addresses) of the hinge joints, optionally restricted to `names`."""
    all_names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    names = all_names if names is None else names
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    return names, np.array([m.jnt_qposadr[j] for j in jid]), np.array([m.jnt_dofadr[j] for j in jid])


class Arena:
    def __init__(self, terrain="flat", robot_xml=G1_DEX3_XML, dt=0.002, native=False, seed=0,
                 armature=True, render=False, width=960, height=540):
        """native=True: load robot_xml untouched (its own floor, actuators and options) for adapter checks."""
        if native:
            self.model = mujoco.MjModel.from_xml_path(robot_xml)
        else:
            spec = mujoco.MjSpec.from_file(robot_xml)
            for el in list(spec.worldbody.geoms) + list(spec.worldbody.lights):   # the robot file's own floor/light
                spec.delete(el) if hasattr(spec, "delete") else el.delete()     # API moved between MuJoCo versions
            before = list(spec.worldbody.geoms)
            add_terrain(spec, terrain, seed)
            _terrain_group(spec, before)
            spec.option.timestep = dt
            spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model = spec.compile()
            if armature:
                for j in range(1, self.model.njnt):
                    arm = _match(ARMATURE, self.model.joint(j).name)
                    if arm is not None:
                        self.model.dof_armature[self.model.jnt_dofadr[j]] = arm
            self.model.vis.global_.offwidth, self.model.vis.global_.offheight = width, height
        self.native = native
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.act_of_joint = np.full(len(jid), -1)
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                j = m.actuator_trnid[a, 0]
                if j in jid:
                    self.act_of_joint[jid.index(j)] = a
        self.position_actuators = bool(m.nu and m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE)
        lim = np.array([m.jnt_actfrcrange[j] for j in jid])
        limited = np.array([m.jnt_actfrclimited[j] for j in jid]).astype(bool)
        self.tau_limit = np.where(limited, lim[:, 1], 1e6)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.hold_q = np.array([HOLD_POSE.get(n, 0.0) for n in self.joint_names])
        g = np.array([_match(HOLD_GAINS, n, [20.0, 1.0]) for n in self.joint_names], dtype=float)
        self.hold_kp, self.hold_kd = g[:, 0], g[:, 1]
        self.renderer = mujoco.Renderer(m, height, width) if render else None
        self.policies = []
        self.reset()

    # ------------------------------------------------------------------ control
    def reset(self, base_z=0.80, joint_pos: dict | None = None, x=0.0, y=0.0, yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[0:3] = [x, y, base_z]
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        q = self.hold_q.copy()
        for n, v in (joint_pos or {}).items():
            if n in self.joint_names:
                q[self.joint_names.index(n)] = v
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, d)
        self.target = q.copy()
        self.kp, self.kd = self.hold_kp.copy(), self.hold_kd.copy()
        self.t = 0.0
        self.push = None

    def ground_height(self, x, y):
        """Height of terrain below (x, y) from a downward ray (robot geoms excluded)."""
        if self.native:
            return 0.0
        pnt = np.array([x, y, 3.0])
        geomid = np.zeros(1, dtype=np.int32)
        groups = np.zeros(6, dtype=np.uint8)
        groups[TERRAIN_GROUP] = 1
        dist = mujoco.mj_ray(self.model, self.data, pnt, np.array([0, 0, -1.0]), groups, 1, -1, geomid)
        return 3.0 - dist if dist >= 0 else 0.0

    def state(self) -> State:
        st = state_from(self.model, self.data, self.pelvis, self.torso, self.qadr, self.vadr, self.joint_names, self.t)
        st.height = float(st.base_pos[2] - self.ground_height(st.base_pos[0], st.base_pos[1]))
        return st

    def set_targets(self, joints, q_target, kp, kd):
        idx = [self.joint_names.index(j) for j in joints]
        self.target[idx], self.kp[idx], self.kd[idx] = q_target, kp, kd

    def step_physics(self, n: int, qd_target=None):
        d = self.data
        for _ in range(n):
            if self.native and self.position_actuators:
                ok = self.act_of_joint >= 0
                d.ctrl[self.act_of_joint[ok]] = self.target[ok]
            else:
                q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
                tau = self.kp * (self.target - q) - self.kd * qd
                tau = np.clip(tau, -self.tau_limit, self.tau_limit)
                ok = self.act_of_joint >= 0
                d.ctrl[self.act_of_joint[ok]] = tau[ok]
            if self.push is not None:        # one (t0, duration, force xyz) or a list of them
                d.xfrc_applied[self.pelvis, :3] = 0.0
                for t0, dur, force in (self.push if isinstance(self.push, list) else [self.push]):
                    if t0 <= self.t < t0 + dur:
                        d.xfrc_applied[self.pelvis, :3] = force
            mujoco.mj_step(self.model, d)
            self.t += self.model.opt.timestep

    def fallen(self, st: State | None = None):
        st = st or self.state()
        return st.gravity_b[2] > -0.5 or st.height < 0.35

    def render(self, cam_distance=3.0, azimuth=135.0, elevation=-15.0):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.pelvis
        cam.distance, cam.azimuth, cam.elevation = cam_distance, azimuth, elevation
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for grp in (0, 1, 2, TERRAIN_GROUP):
            opt.geomgroup[grp] = 1
        self.renderer.update_scene(self.data, camera=cam, scene_option=opt)
        return self.renderer.render()
