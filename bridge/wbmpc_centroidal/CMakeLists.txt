cmake_minimum_required(VERSION 3.16)
project(wbmpc_centroidal_bridge)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# all of these are ament packages from the wb_humanoid_mpc colcon workspace; source
# install/setup.bash before configuring so find_package can see them via CMAKE_PREFIX_PATH.
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization log log_setup)
find_package(pybind11 REQUIRED)
find_package(humanoid_centroidal_mpc REQUIRED)
find_package(humanoid_common_mpc REQUIRED)
find_package(robot_model REQUIRED)
find_package(robot_core REQUIRED)
find_package(ocs2_sqp REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_core REQUIRED)
find_package(ocs2_oc REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(absl REQUIRED)

pybind11_add_module(wbmpc_centroidal_bridge bridge.cpp)

target_include_directories(wbmpc_centroidal_bridge PRIVATE
    ${humanoid_centroidal_mpc_INCLUDE_DIRS}
    ${humanoid_common_mpc_INCLUDE_DIRS}
    ${robot_model_INCLUDE_DIRS}
    ${robot_core_INCLUDE_DIRS}
    ${ocs2_sqp_INCLUDE_DIRS}
    ${ocs2_mpc_INCLUDE_DIRS}
    ${ocs2_core_INCLUDE_DIRS}
    ${ocs2_oc_INCLUDE_DIRS}
    ${ocs2_pinocchio_interface_INCLUDE_DIRS}
)
target_link_libraries(wbmpc_centroidal_bridge PRIVATE
    ${humanoid_centroidal_mpc_LIBRARIES}
    ${humanoid_common_mpc_LIBRARIES}
    ${robot_model_LIBRARIES}
    ${robot_core_LIBRARIES}
    ${ocs2_sqp_LIBRARIES}
    ${ocs2_mpc_LIBRARIES}
    ${ocs2_core_LIBRARIES}
    ${ocs2_oc_LIBRARIES}
    ${ocs2_pinocchio_interface_LIBRARIES}
    pinocchio::pinocchio
    absl::hash
    absl::raw_hash_set
    absl::flat_hash_map
)
target_compile_definitions(wbmpc_centroidal_bridge PRIVATE
    ${humanoid_centroidal_mpc_DEFINITIONS}
    ${humanoid_common_mpc_DEFINITIONS}
)
