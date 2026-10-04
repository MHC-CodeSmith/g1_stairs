"""min-dai/RoMoCo: reduced-order planner (LIP-based) + whole-body task-space-control QP (Clarabel.cpp
solver), legs-only (G1ModelLeg: 6 floating-base DoF + 12 leg joints - arms and waist are not part of
this model at all, unlike every other adapter here except unitree_rl_gym's leg-only policy).

Output format matches wb_humanoid_mpc's: per joint (joint_positions, joint_velocities, joint_kp,
joint_kd, joint_torques_ff), see BipedMotorCommands::SolveFullTorque in romoco_core for the exact
formula this mirrors onto Arena's own PD+tau_ext: target=joint_positions, kp=joint_kp, kd=joint_kd,
tau_ext = joint_kd*joint_velocities + joint_torques_ff.

Driven through bridge/romoco/bridge.cpp (tools/build_romoco_bridge.sh), wrapping
BasicControllerStateMachine::UpdateControl() directly - romoco_ros/ros_controller_node.hpp's
RosControllerNode owns exactly the same objects (BasicControllerStateMachine, OutputBase,
TorqueSolverBase) and only adds a ROS proprioception subscriber/motor-command publisher around this
same call, no rclcpp::init needed here either.

q/dq layout (RobotBasePinocchio::BaseJointOrder + G1ModelLeg::JointIndex): 6 floating-base DoF as
[x, y, z, yaw, pitch, roll] - Euler angles, NOT a quaternion, unlike every other adapter's st.base_quat
- followed by the 12 leg joints in exactly MJ29's LEGS12 order. arena/wbmpc.py's st.base_quat (w,x,y,z)
is converted to this Euler convention in reset()/act() below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from arena.policies import LEGS12, ArenaPolicy
from arena.world import TP

BRIDGE_DIR = os.path.join(os.path.dirname(TP), "bridge", "romoco", "build")
ROMOCO_ROOT = os.path.join(TP, "RoMoCo")
# colcon --symlink-install made these symlinks absolute to /home/docker/RoMoCo/... (the container
# path RoMoCo was built from), not relative - so this only resolves inside that same container,
# not via ROMOCO_ROOT's /workspace/g1_stairs/third_party/RoMoCo alias.
CONFIG_FOLDER = "/home/docker/RoMoCo/install/g1_stack/share/g1_stack/config_18dof"
LOG_PATH = "/tmp/romoco_logs"

STANDING_POSE = {   # config_18dof: z_lb/z_ub 0.6-0.68m; crouched hip/knee/ankle matching that height
    "left_hip_pitch_joint": -0.3, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.6, "left_ankle_pitch_joint": -0.3, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.3, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.6, "right_ankle_pitch_joint": -0.3, "right_ankle_roll_joint": 0.0,
}
STANDING_BASE_Z = 0.65


def _quat_to_eulerZYX(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


def _body_rate_to_eulerZYX_rate(omega_body, yaw, pitch, roll):
    """Inverse of the standard Z(yaw)-Y(pitch)-X(roll) body-rate kinematic relation: given body-frame
    angular velocity and the current Euler angles, returns (yaw_dot, pitch_dot, roll_dot). Needed because
    RobotBasePinocchio's floating-base DoF are these three sequential-revolute generalized coordinates,
    not a quaternion - their rates are NOT equal to body angular velocity component-wise except at zero
    angle/rate (aliasing them directly works for an instant at rest but drifts as the robot moves)."""
    wx, wy, wz = omega_body
    cr, sr = np.cos(roll), np.sin(roll)
    cp = np.cos(pitch)
    cp = cp if abs(cp) > 1e-6 else 1e-6
    pitch_dot = wy * cr - wz * sr
    yaw_dot = (wy * sr + wz * cr) / cp
    roll_dot = wx + yaw_dot * np.sin(pitch)
    return yaw_dot, pitch_dot, roll_dot


class RoMoCo(ArenaPolicy):
    name, joints = "romoco", LEGS12
    control_dt = 1.0 / 500
    uses = "vx vy wz"
    trained_with_hands = False

    def __init__(self):
        if BRIDGE_DIR not in sys.path:
            sys.path.insert(0, BRIDGE_DIR)
        import romoco_bridge
        os.makedirs(LOG_PATH, exist_ok=True)
        self._ctrl = romoco_bridge.RoMoCo(CONFIG_FOLDER, LOG_PATH)
        self.kp = np.zeros(len(self.joints))
        self.kd = np.zeros(len(self.joints))
        self.tau_ext = np.zeros(len(self.joints))
        self.default = np.array([STANDING_POSE[n] for n in self.joints])

    def reset(self, st):
        self.tau_ext[:] = 0.0

    def obs(self, st, cmd):
        return st.get(st.q, self.joints)  # no learned observation vector; exposed for arena.check_adapters parity

    def _q_dq(self, st):
        yaw, pitch, roll = _quat_to_eulerZYX(st.base_quat)
        yaw_dot, pitch_dot, roll_dot = _body_rate_to_eulerZYX_rate(st.ang_vel_b, yaw, pitch, roll)
        legs_q = st.get(st.q, self.joints)
        legs_qd = st.get(st.qd, self.joints)
        q = np.concatenate(([st.base_pos[0], st.base_pos[1], st.base_pos[2], yaw, pitch, roll], legs_q))
        dq = np.concatenate(([st.lin_vel_w[0], st.lin_vel_w[1], st.lin_vel_w[2],
                              yaw_dot, pitch_dot, roll_dot], legs_qd))
        return q, dq

    def act(self, st, cmd):
        q, dq = self._q_dq(st)
        cmd_values = [cmd.vx, cmd.vy, 0.0, 0.0, 0.0, cmd.wz, 0.0, 0.0]
        out = self._ctrl.update(q.tolist(), dq.tolist(), 0, cmd_values)  # mode 0 = Standing

        q_des = np.asarray(out["joint_positions"])
        qd_des = np.asarray(out["joint_velocities"])
        self.kp = np.asarray(out["joint_kp"])
        self.kd = np.asarray(out["joint_kd"])
        ff = np.asarray(out["joint_torques_ff"])
        self.tau_ext = self.kd * qd_des + ff
        return q_des
