#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <map>
#include <string>

#include <WalkingManager.hpp>
#include <RobotState.hpp>
#include <JointCommand.hpp>
#include <globals.h>

namespace py = pybind11;

// Declared extern in globals.h, defined by whichever main_*.cpp links in upstream; this
// bridge is its own entry point so it must define them too. Matches main_sim.cpp's values
// (closed-loop MPC, no residual observer).
bool isMPCLoopClosed = true;
bool isObserverActive = false;
bool switchWalkingState = false;
bool switchCoopState = false;
int  ZMP_TYPE = 3;
Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(G1_NUM_MOTOR);

// Thin pybind11 wrapper around labrob::WalkingManager (IS-MPC footstep planner +
// whole-body QP WBC), so a Python caller (our own MuJoCo arena) can drive it with
// its own robot state each control step instead of labrob's own main_sim.cpp loop.
// RobotState/JointCommand cross the C++/Python boundary as plain dict/array
// arguments rather than exposing the Eigen-heavy structs directly to Python.
class PyWalkingManager {
 public:
  void set_reactive_standing(bool r) { wm_.setReactiveStanding(r); }
  void set_verbose_coop(bool v) { wm_.setVerboseCoop(v); }

  bool init(const py::dict& joint_pos,
            const py::dict& armatures,
            const std::array<double, 3>& base_pos,
            const std::array<double, 4>& base_quat_wxyz) {
    labrob::RobotState rs = make_state(joint_pos, {}, base_pos, base_quat_wxyz, {0, 0, 0}, {0, 0, 0});
    std::map<std::string, double> arm;
    for (auto item : armatures) {
      arm[py::str(item.first)] = item.second.cast<double>();
    }
    return wm_.init(rs, arm);
  }

  py::dict update(const py::dict& joint_pos,
                   const py::dict& joint_vel,
                   const std::array<double, 3>& base_pos,
                   const std::array<double, 4>& base_quat_wxyz,
                   const std::array<double, 3>& lin_vel,
                   const std::array<double, 3>& ang_vel) {
    labrob::RobotState rs = make_state(joint_pos, joint_vel, base_pos, base_quat_wxyz, lin_vel, ang_vel);
    labrob::JointCommand jc;
    wm_.update(rs, jc);
    py::dict out;
    for (auto& kv : jc) {
      out[py::str(kv.first)] = kv.second;
    }
    return out;
  }

  int64_t controller_frequency() const { return wm_.get_controller_frequency(); }

 private:
  static labrob::RobotState make_state(const py::dict& joint_pos,
                                        const py::dict& joint_vel,
                                        const std::array<double, 3>& base_pos,
                                        const std::array<double, 4>& base_quat_wxyz,
                                        const std::array<double, 3>& lin_vel,
                                        const std::array<double, 3>& ang_vel) {
    labrob::RobotState rs;
    rs.position = Eigen::Vector3d(base_pos[0], base_pos[1], base_pos[2]);
    rs.orientation = Eigen::Quaterniond(base_quat_wxyz[0], base_quat_wxyz[1],
                                         base_quat_wxyz[2], base_quat_wxyz[3]);
    rs.linear_velocity = Eigen::Vector3d(lin_vel[0], lin_vel[1], lin_vel[2]);
    rs.angular_velocity = Eigen::Vector3d(ang_vel[0], ang_vel[1], ang_vel[2]);
    for (auto item : joint_pos) {
      rs.joint_state[py::str(item.first)].pos = item.second.cast<double>();
    }
    for (auto item : joint_vel) {
      rs.joint_state[py::str(item.first)].vel = item.second.cast<double>();
    }
    return rs;
  }

  labrob::WalkingManager wm_;
};

PYBIND11_MODULE(labrob_bridge, m) {
  m.doc() = "pybind11 bridge to labrob_mujoco_environment's WalkingManager (IS-MPC + whole-body QP)";
  py::class_<PyWalkingManager>(m, "WalkingManager")
      .def(py::init<>())
      .def("set_reactive_standing", &PyWalkingManager::set_reactive_standing)
      .def("set_verbose_coop", &PyWalkingManager::set_verbose_coop)
      .def("init", &PyWalkingManager::init,
           py::arg("joint_pos"), py::arg("armatures"), py::arg("base_pos"), py::arg("base_quat_wxyz"))
      .def("update", &PyWalkingManager::update,
           py::arg("joint_pos"), py::arg("joint_vel"), py::arg("base_pos"),
           py::arg("base_quat_wxyz"), py::arg("lin_vel"), py::arg("ang_vel"))
      .def("controller_frequency", &PyWalkingManager::controller_frequency);
}
