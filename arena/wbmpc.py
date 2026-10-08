"""manumerous/wb_humanoid_mpc (1X Technologies / Manuel Galliker): whole-body nonlinear MPC (OCS2, SQP
solver) optimizing full-order torque-level dynamics in real time. No RL. Unlike labrob_mujoco_environment
(arena/labrob.py), this one DOES take a velocity command (vx, vy, pelvis height, wz) - see
ProceduralMpcMotionManager::setAndScaleVelocityCommand in bridge/wbmpc/bridge.cpp - so it can run through
the same vx/vy/wz test battery as the RL policies, not just stand still.

Output format differs from every other adapter here: per joint, (q_des, qd_des, kp, kd, feed_forward_effort)
rather than just a PD target. Arena.step_physics computes tau = kp*(target-q) - kd*qd + tau_ext, so this
maps on almost exactly: target=q_des, kp=kp, kd=kd, and tau_ext = kd*qd_des + feed_forward_effort absorbs
the two terms Arena's own PD loop doesn't have (a desired velocity, and feedforward torque).

Driven through bridge/wbmpc/bridge.cpp (tools/build_wbmpc_bridge.sh), a pybind11 wrapper around
WBMpcInterface -> SqpMpc -> WBMpcMrtJointController (see that file for why no rclcpp/ROS2 node is needed
at runtime). The underlying CppAD-generated dynamics/cost libraries are cached under cppad_code_gen/ next
to wherever the process's cwd was on first use - construction is only fast (~1s) when that cache already
exists; a clean cache can take minutes to regenerate. Construction also loads the URDF/task/reference files
by absolute path, but __init__ chdirs into bridge/wbmpc/ first so the cache already built there
(from bridge/wbmpc's own build/smoke test) is found - same category of cwd quirk as labrob's
relative URDF path.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

BRIDGE_DIR = os.path.join(os.path.dirname(TP), "bridge", "wbmpc", "build")
WBMPC_ROOT = os.path.join(TP, "wb_humanoid_mpc")
TASK_FILE = os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info")
REFERENCE_FILE = os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info")
URDF_FILE = os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")
GAIT_FILE = os.path.join(WBMPC_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.info")

STANDING_POSE = {   # config/command/reference.info's defaultJointState
    "left_hip_pitch_joint": -0.05, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.1, "left_ankle_pitch_joint": -0.05, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.05, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.1, "right_ankle_pitch_joint": -0.05, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}
STANDING_BASE_Z = 0.7925


class WbMpc(ArenaPolicy):
    name, joints = "wbmpc", MJ29  # all 29 body joints (6 wrists held fixed by the MPC itself)
    control_dt = 1.0 / 500
    uses = "vx vy wz height"
    trained_with_hands = False

    def __init__(self):
        if BRIDGE_DIR not in sys.path:
            sys.path.insert(0, BRIDGE_DIR)
        import wbmpc_bridge
        cwd = os.getcwd()
        try:
            os.chdir(BRIDGE_DIR)  # CppAdInterface caches cppad_code_gen/ relative to cwd; reuse the one built there
            self._mpc = wbmpc_bridge.WbMpc(TASK_FILE, URDF_FILE, REFERENCE_FILE, GAIT_FILE)
        finally:
            os.chdir(cwd)
        self.kp = np.zeros(len(self.joints))
        self.kd = np.zeros(len(self.joints))
        self.tau_ext = np.zeros(len(self.joints))
        self.default = np.array([STANDING_POSE[n] for n in self.joints])

    def reset(self, st):
        joint_pos = {n: float(STANDING_POSE.get(n, 0.0)) for n in self.joints}
        self._mpc.start(joint_pos, (0.0, 0.0, STANDING_BASE_Z), (1.0, 0.0, 0.0, 0.0))
        self.tau_ext[:] = 0.0

    def obs(self, st, cmd):
        return st.get(st.q, self.joints)  # no learned observation vector; exposed for arena.check_adapters parity

    def act(self, st, cmd):
        height = cmd.height if cmd.height is not None else STANDING_BASE_Z
        self._mpc.set_velocity_command(cmd.vx, cmd.vy, height, cmd.wz)

        q = st.get(st.q, self.joints)
        qd = st.get(st.qd, self.joints)
        joint_pos = {n: float(v) for n, v in zip(self.joints, q)}
        joint_vel = {n: float(v) for n, v in zip(self.joints, qd)}
        action = self._mpc.update(
            joint_pos, joint_vel,
            tuple(float(v) for v in st.base_pos),
            tuple(float(v) for v in st.base_quat),
            tuple(float(v) for v in st.lin_vel_b),
            tuple(float(v) for v in st.ang_vel_b),
            float(st.t),
        )

        q_des = np.array([action[n]["q_des"] for n in self.joints])
        qd_des = np.array([action[n]["qd_des"] for n in self.joints])
        self.kp = np.array([action[n]["kp"] for n in self.joints])
        self.kd = np.array([action[n]["kd"] for n in self.joints])
        ff = np.array([action[n]["ff"] for n in self.joints])
        self.tau_ext = self.kd * qd_des + ff
        return q_des
