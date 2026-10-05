#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <memory>
#include <string>

#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <humanoid_common_mpc/command/WalkingVelocityCommand.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>

#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

namespace py = pybind11;
using namespace ocs2;
using namespace ocs2::humanoid;

// Same pybind11 bridge pattern as bridge/wbmpc, but for wb_humanoid_mpc's OTHER formulation:
// humanoid_centroidal_mpc (center-of-mass momentum + full kinematics) instead of humanoid_wb_mpc
// (full joint-space dynamics). Both ship in the same repo; GuilhermeAsura/humanoid_repos_eval's own
// evaluation found the whole-body-dynamics formulation (bridge/wbmpc) "thrashes instead of walking"
// - matching what we independently found in our own arena (falls within ~2s, unstable even standing
// alone past ~1.7s) - while this centroidal formulation "produced the smoothest motion" of every
// analytical (non-RL) controller they tested. Construction mirrors
// humanoid_centroidal_mpc_ros2/src/CentroidalMpcRobotSim.cpp, again with no rclcpp/ROS2 node needed.
class PyCentroidalMpc {
 public:
  PyCentroidalMpc(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile,
                  const std::string& gaitFile)
      : robotDescription_(urdfFile),
        interface_(taskFile, urdfFile, referenceFile),
        mpc_(interface_.mpcSettings(), interface_.sqpSettings(), interface_.getOptimalControlProblem(),
             interface_.getInitializer()) {
    calcPtr_ = std::make_shared<CentroidalMpcTargetTrajectoriesCalculator>(
        referenceFile, interface_.getMpcRobotModel(), interface_.getPinocchioInterface(),
        interface_.getCentroidalModelInfo(), interface_.mpcSettings().timeHorizon_);
    ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajFunc =
        [this](const vector4_t& v, scalar_t t0, scalar_t t1, const vector_t& x0) {
          return calcPtr_->commandedVelocityToTargetTrajectories(v, t0, x0);
        };
    motionManagerPtr_ = std::make_shared<ProceduralMpcMotionManager>(
        gaitFile, referenceFile, interface_.getSwitchedModelReferenceManagerPtr(), interface_.getMpcRobotModel(),
        targetTrajFunc);
    mpc_.getSolverPtr()->setReferenceManager(interface_.getReferenceManagerPtr());
    mpc_.getSolverPtr()->addSynchronizedModule(motionManagerPtr_);
    controllerPtr_ = std::make_unique<CentroidalMpcMrtJointController>(
        robotDescription_, interface_.modelSettings(), interface_.getMpcRobotModel(), mpc_,
        interface_.getPinocchioInterface(), interface_.mpcSettings().mpcDesiredFrequency_, nullptr);
  }

  void set_velocity_command(double vx, double vy, double height, double wz) {
    motionManagerPtr_->setAndScaleVelocityCommand(WalkingVelocityCommand(vx, vy, height, wz));
  }

  void start(const py::dict& joint_pos, const std::array<double, 3>& base_pos,
             const std::array<double, 4>& base_quat_wxyz) {
    robot::model::RobotState initState = make_state(joint_pos, {}, base_pos, base_quat_wxyz, {0, 0, 0}, {0, 0, 0});
    controllerPtr_->startMpcThread(initState);
  }

  bool ready() const { return controllerPtr_->ready(); }

  py::dict update(const py::dict& joint_pos, const py::dict& joint_vel, const std::array<double, 3>& base_pos,
                   const std::array<double, 4>& base_quat_wxyz, const std::array<double, 3>& lin_vel,
                   const std::array<double, 3>& ang_vel) {
    robot::model::RobotState state = make_state(joint_pos, joint_vel, base_pos, base_quat_wxyz, lin_vel, ang_vel);
    robot::model::RobotJointAction action(robotDescription_);
    controllerPtr_->computeJointControlAction(0.0, state, action);

    py::dict out;
    for (const auto& name : robotDescription_.getJointNames()) {
      const auto& a = *action.at(robotDescription_.getJointIndex(name));
      py::dict j;
      j["q_des"] = a.q_des;
      j["qd_des"] = a.qd_des;
      j["kp"] = a.kp;
      j["kd"] = a.kd;
      j["ff"] = a.feed_forward_effort;
      out[py::str(name)] = j;
    }
    return out;
  }

 private:
  robot::model::RobotState make_state(const py::dict& joint_pos, const py::dict& joint_vel,
                                       const std::array<double, 3>& base_pos,
                                       const std::array<double, 4>& base_quat_wxyz,
                                       const std::array<double, 3>& lin_vel, const std::array<double, 3>& ang_vel) {
    robot::model::RobotState state(robotDescription_, 2);
    state.setConfigurationToZero();
    state.setRootPositionInWorldFrame(Eigen::Vector3d(base_pos[0], base_pos[1], base_pos[2]));
    state.setRootRotationLocalToWorldFrame(
        Eigen::Quaterniond(base_quat_wxyz[0], base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3]));
    state.setRootLinearVelocityInLocalFrame(Eigen::Vector3d(lin_vel[0], lin_vel[1], lin_vel[2]));
    state.setRootAngularVelocityInLocalFrame(Eigen::Vector3d(ang_vel[0], ang_vel[1], ang_vel[2]));
    for (auto item : joint_pos) {
      state.setJointPosition(robotDescription_.getJointIndex(py::str(item.first)), item.second.cast<double>());
    }
    for (auto item : joint_vel) {
      state.setJointVelocity(robotDescription_.getJointIndex(py::str(item.first)), item.second.cast<double>());
    }
    return state;
  }

  robot::model::RobotDescription robotDescription_;
  CentroidalMpcInterface interface_;
  SqpMpc mpc_;
  std::shared_ptr<CentroidalMpcTargetTrajectoriesCalculator> calcPtr_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManagerPtr_;
  std::unique_ptr<CentroidalMpcMrtJointController> controllerPtr_;
};

PYBIND11_MODULE(wbmpc_centroidal_bridge, m) {
  m.doc() = "pybind11 bridge to wb_humanoid_mpc's CentroidalMpcMrtJointController (OCS2 centroidal NMPC, SQP)";
  py::class_<PyCentroidalMpc>(m, "CentroidalMpc")
      .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&>(),
           py::arg("task_file"), py::arg("urdf_file"), py::arg("reference_file"), py::arg("gait_file"))
      .def("set_velocity_command", &PyCentroidalMpc::set_velocity_command, py::arg("vx"), py::arg("vy"),
           py::arg("height"), py::arg("wz"))
      .def("start", &PyCentroidalMpc::start, py::arg("joint_pos"), py::arg("base_pos"), py::arg("base_quat_wxyz"))
      .def("ready", &PyCentroidalMpc::ready)
      .def("update", &PyCentroidalMpc::update, py::arg("joint_pos"), py::arg("joint_vel"), py::arg("base_pos"),
           py::arg("base_quat_wxyz"), py::arg("lin_vel"), py::arg("ang_vel"));
}
