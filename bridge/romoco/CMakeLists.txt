cmake_minimum_required(VERSION 3.16)
project(romoco_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(pybind11 REQUIRED)
find_package(g1_stack REQUIRED)
find_package(romoco_state_machine REQUIRED)
find_package(romoco_control REQUIRED)
find_package(romoco_core REQUIRED)
find_package(romoco_output REQUIRED)
find_package(romoco_utils REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization)

pybind11_add_module(romoco_bridge bridge.cpp)
target_link_libraries(romoco_bridge PRIVATE
    g1_stack::g1_stack
    romoco_state_machine::romoco_state_machine
    romoco_control::romoco_control
    romoco_core::romoco_core
    romoco_output::romoco_output
    romoco_utils::romoco_utils
    pinocchio::pinocchio
)
