#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <array>
#include <memory>
#include <string>

#include "g1_model_leg.hpp"
#include "romoco_state_machine/controller_state_machine.hpp"
#include "romoco_control/torque_solver_base.hpp"
#include "romoco_core/output_base.hpp"
#include "romoco_core/biped_commands.hpp"
#include "romoco_core/biped_motor_commands.hpp"

namespace py = pybind11;
using namespace romoco;

// pybind11 bridge to RoMoCo's BasicControllerStateMachine (reduced-order planner + whole-body
// task-space-control QP, Clarabel.cpp solver), the same plain-C++-class-under-a-ROS2-wrapper
// pattern as bridge/labrob and bridge/wbmpc: romoco_ros/ros_controller_node.hpp's
// RosControllerNode owns exactly these same objects (BasicControllerStateMachine, OutputBase,
// TorqueSolverBase) and its "ROS" part is only a proprioception subscriber/motor-command
// publisher around UpdateControl() - called directly here instead, no rclcpp::init needed.
//
// G1ModelLeg is G1's 18-DoF leg-only model (6 floating-base DoF + 12 leg joints, in exactly
// MJ29's LEGS12 order shifted by the 6 base DoF: see arena/romoco.py), so this adapter is
// leg-only like arena/policies.py's RlGym, not whole-body like labrob/wb_humanoid_mpc.
class PyRoMoCo {
 public:
  PyRoMoCo(const std::string& config_folder, const std::string& log_path)
      : robot_((SimpleTimer::SetExternalTime(0.0), std::make_shared<robot::G1ModelLeg>(config_folder))),
        controller_(config_folder, log_path, robot_) {}

  py::dict update(const std::array<double, 18>& q, const std::array<double, 18>& dq, int mode,
                   const std::array<double, 8>& command_values, double t) {
    SimpleTimer::SetExternalTime(t);
    Eigen::VectorXd qv(18), dqv(18);
    for (int i = 0; i < 18; ++i) {
      qv(i) = q[i];
      dqv(i) = dq[i];
    }
    robot_->UpdateAll(qv, dqv);

    DesiredCommand command;
    command.mode = static_cast<Mode>(mode);
    for (int i = 0; i < 8; ++i) {
      command.values(i) = command_values[i];
    }

    BipedMotorCommands out = controller_.UpdateControl(command, robot_, output_, torque_solver_, qv, dqv);

    py::dict result;
    result["joint_positions"] = out.joint_positions;
    result["joint_velocities"] = out.joint_velocities;
    result["joint_kp"] = out.joint_kp;
    result["joint_kd"] = out.joint_kd;
    result["joint_torques_ff"] = out.joint_torques_ff;
    return result;
  }

  std::vector<double> com_debug(const std::array<double, 18>& q, const std::array<double, 18>& dq) {
    Eigen::VectorXd qv(18), dqv(18);
    for (int i = 0; i < 18; ++i) { qv(i) = q[i]; dqv(i) = dq[i]; }
    robot_->UpdateAll(qv, dqv);
    auto k = robot_->com_kinematics();
    return {k.position.x(), k.position.y(), k.position.z(), k.velocity.x(), k.velocity.y(), k.velocity.z(), robot_->mass()};
  }

 private:
  std::shared_ptr<robot::RobotBasePinocchio> robot_;
  BasicControllerStateMachine controller_;
  std::shared_ptr<OutputBase> output_;
  std::unique_ptr<TorqueSolverBase> torque_solver_;
};

PYBIND11_MODULE(romoco_bridge, m) {
  m.doc() = "pybind11 bridge to RoMoCo's BasicControllerStateMachine (reduced-order planner + TSC-QP)";
  py::class_<PyRoMoCo>(m, "RoMoCo")
      .def(py::init<const std::string&, const std::string&>(), py::arg("config_folder"), py::arg("log_path"))
      .def("com_debug", &PyRoMoCo::com_debug, py::arg("q"), py::arg("dq"))
      .def("update", &PyRoMoCo::update, py::arg("q"), py::arg("dq"), py::arg("mode"), py::arg("command_values"), py::arg("t") = 0.0);
}
