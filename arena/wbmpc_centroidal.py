"""wb_humanoid_mpc's OTHER OCS2 formulation: humanoid_centroidal_mpc (center-of-mass momentum +
full kinematics) rather than arena/wbmpc.py's humanoid_wb_mpc (full joint-space dynamics).
GuilhermeAsura/humanoid_repos_eval found the whole-body-dynamics formulation "thrashes instead of
walking" (matching what we found independently: falls within ~2s, unstable even standing past
~1.7s) while this centroidal one "produced the smoothest motion" of every analytical controller
they tested. Same output format and Arena mapping as arena/wbmpc.py.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from arena.policies import MJ29, ArenaPolicy
from arena.world import TP

BRIDGE_DIR = os.path.join(os.path.dirname(TP), "bridge", "wbmpc_centroidal", "build")
CACHE_DIR = os.path.join(os.path.dirname(TP), "bridge", "wbmpc_centroidal")  # cppad_code_gen/ lives here
WBMPC_ROOT = os.path.join(TP, "wb_humanoid_mpc")
TASK_FILE = os.environ.get("WBMPC_TASK_FILE") or os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.info")
REFERENCE_FILE = os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.info")
URDF_FILE = os.path.join(WBMPC_ROOT, "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")
GAIT_FILE = os.path.join(WBMPC_ROOT, "humanoid_nmpc/humanoid_common_mpc/config/command/gait.info")

STANDING_POSE = {
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
FILTER_ALPHA = float(os.environ.get("WBMPC_ALPHA", "0.4"))


class CentroidalMpc(ArenaPolicy):
    name, joints = "wbmpc_centroidal", MJ29
    control_dt = 1.0 / 500
    uses = "vx vy wz height"
    trained_with_hands = False

    def __init__(self):
        if BRIDGE_DIR not in sys.path:
            sys.path.insert(0, BRIDGE_DIR)
        import wbmpc_centroidal_bridge
        cwd = os.getcwd()
        try:
            os.chdir(CACHE_DIR)  # CppAdInterface caches cppad_code_gen/ relative to cwd
            self._mpc = wbmpc_centroidal_bridge.CentroidalMpc(TASK_FILE, URDF_FILE, REFERENCE_FILE, GAIT_FILE)
        finally:
            os.chdir(cwd)
        self.kp = np.zeros(len(self.joints))
        self.kd = np.zeros(len(self.joints))
        self.tau_ext = np.zeros(len(self.joints))
        self.default = np.array([STANDING_POSE[n] for n in self.joints])

    def reset(self, st):
        joint_pos = {n: float(STANDING_POSE.get(n, 0.0)) for n in self.joints}
        self._mpc.start(joint_pos, (0.0, 0.0, STANDING_BASE_Z), (1.0, 0.0, 0.0, 0.0))
        # CentroidalMpcRobotSim.cpp (the reference this bridge mirrors) polls ready() after start()
        # and waits an extra 200ms "to allow MPC policy to initialize" before ever reading a control
        # action - skipping this left computeJointControlAction() reading from the MRT policy buffer
        # before the background solver thread's first solve ever completed, which left preSolverRun's
        # own initTime bookkeeping on a garbage/subnormal value forever after (confirmed by instrumenting
        # ProceduralMpcMotionManager::preSolverRun directly: initTime printed as ~6.95e-310, and the
        # "don't change gait for 0.2s" gate comparing against it never passed) - this is why the
        # automatic gait-promotion state machine never advanced past "stance" in every test this session.
        while not self._mpc.ready():
            time.sleep(0.1)
        time.sleep(0.2)
        self.tau_ext[:] = 0.0
        self._filt = None

    def obs(self, st, cmd):
        return st.get(st.q, self.joints)

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
        tau = self.kd * qd_des + ff
        # The closed loop MPC -> PD -> robot has a period-2 instability at the solve rate: each new plan
        # makes the joint target jump with alternating sign and ~1.2x growing amplitude (hip pitch jumps
        # +0.018, -0.021, +0.026, ... rad at consecutive solves), which bounces the robot and ends in a
        # yaw spin. A first-order low-pass on the command (alpha per control step) breaks the cycle:
        # 12/12 lockstep runs walked 30 s with alpha 0.1-0.4 vs. ~1/3 without.
        if self._filt is None:
            self._filt = (q_des.copy(), tau.copy())
        qf, tf = self._filt
        qf += FILTER_ALPHA * (q_des - qf)
        tf += FILTER_ALPHA * (tau - tf)
        self.tau_ext = tf.copy()
        return qf.copy()
