"""GuilhermeAsura/labrob_mujoco_environment (matteogoddi/labrob_mujoco_environment): classical IS-MPC footstep
planning + whole-body QP control (Pinocchio dynamics, HPIPM for the MPC QP, qpOASES for the WBC QP). No RL, no
ROS. Torque-output, unlike every other adapter in this arena (all PD-target): its WalkingManager computes full
joint torques each 2 ms control step from the robot's measured state, so it plugs into Arena.tau_ext rather than
Arena.set_targets's internal PD loop (kp=kd=0 here; see arena/world.py's step_physics).

Driven through bridge/labrob/bridge.cpp (tools/build_labrob_bridge.sh), a pybind11 wrapper around
labrob::WalkingManager::init()/update() - see that file for why isMPCLoopClosed/ZMP_TYPE/etc. are hand-defined
there. WalkingManager::init() loads its URDF from a path relative to cwd
(../robot/g1/g1_description/*.urdf, hardcoded in include/globals.h), so init() here temporarily chdirs into
third_party/labrob_mujoco_environment/build - the one directory that relative path resolves correctly from.

Armatures: labrob's own stock MJCF (robot/g1/g1_mj_description/g1_29dof_rev_1_0.xml) sets no <joint armature=...>
anywhere, i.e. main_sim.cpp's `armatures[name] = mj_model_ptr->dof_armature[...]` is always 0 for every joint in
upstream's own demo. We pass the same all-zero map here rather than this arena's own (nonzero) ARMATURE table, to
match the dynamics the WBC was actually tuned against; this is a known simplification if the two ever need to
match exactly.

Initial pose: WalkingManager assumes it starts from upstream's own standing pose (deeper-squat, base_z=0.725112;
include/globals.h's robot_type::G1.initial_joint_positions) - not this arena's 0.80 m default. Callers should
Arena.reset(base_z=0.725112, joint_pos=STANDING_POSE) before using this policy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

BRIDGE_DIR = os.path.join(os.path.dirname(TP), "bridge", "labrob", "build")
LABROB_BUILD_DIR = os.path.join(TP, "labrob_mujoco_environment", "build")

STANDING_POSE = {   # include/globals.h robot_type::G1.initial_joint_positions
    "left_hip_pitch_joint": -0.44, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.95, "left_ankle_pitch_joint": -0.50, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.44, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.95, "right_ankle_pitch_joint": -0.50, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.07, "left_shoulder_roll_joint": 0.35, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.25, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.07, "right_shoulder_roll_joint": -0.35, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.25, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}
STANDING_BASE_Z = 0.725112


class Labrob(ArenaPolicy):
    name, joints = "labrob", MJ29  # all 29 body joints (labrob's own model has no Dex3 hands)
    control_dt = 1.0 / 500  # G1_CONTROLLER_HZ in include/globals.h; run at full rate, no decimation
    uses = "vx vy wz"
    trained_with_hands = False

    def __init__(self):
        if BRIDGE_DIR not in sys.path:
            sys.path.insert(0, BRIDGE_DIR)
        import labrob_bridge
        self._wm = labrob_bridge.WalkingManager()
        self._wm.set_reactive_standing(True)
        self.kp = np.zeros(len(self.joints))
        self.kd = np.zeros(len(self.joints))
        self.tau_ext = np.zeros(len(self.joints))
        self.default = np.array([STANDING_POSE[n] for n in self.joints])  # arena.run's init-pose convention

    def reset(self, st):
        joint_pos = {n: float(STANDING_POSE.get(n, 0.0)) for n in self.joints}
        armatures = {n: 0.0 for n in self.joints}  # see module docstring
        base_pos = (0.0, 0.0, STANDING_BASE_Z)
        base_quat = (1.0, 0.0, 0.0, 0.0)
        cwd = os.getcwd()
        try:
            os.chdir(LABROB_BUILD_DIR)  # WalkingManager::init() loads its URDF relative to cwd
            ok = self._wm.init(joint_pos, armatures, base_pos, base_quat)
        finally:
            os.chdir(cwd)
        if not ok:
            raise RuntimeError("labrob WalkingManager.init() failed")
        self.tau_ext[:] = 0.0

    def obs(self, st, cmd):
        return st.get(st.q, self.joints)  # no learned observation vector; exposed for arena.check_adapters parity

    def act(self, st, cmd):
        q = st.get(st.q, self.joints)
        qd = st.get(st.qd, self.joints)
        joint_pos = {n: float(v) for n, v in zip(self.joints, q)}
        joint_vel = {n: float(v) for n, v in zip(self.joints, qd)}
        tau = self._wm.update(
            joint_pos, joint_vel,
            tuple(float(v) for v in st.base_pos),
            tuple(float(v) for v in st.base_quat),
            tuple(float(v) for v in st.lin_vel_b),
            tuple(float(v) for v in st.ang_vel_b),
        )
        self.tau_ext = np.array([tau[n] for n in self.joints])
        return q  # inert: kp == kd == 0, so set_targets's PD term contributes nothing
