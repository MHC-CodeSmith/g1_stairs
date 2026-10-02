cmake_minimum_required(VERSION 3.16)
project(labrob_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# labrob_mujoco_environment is vendored (gitignored) at third_party/; WalkingControllerLibrary.a
# must already be built there (tools/run_labrob_headless.sh does this) before this target links.
set(LABROB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/labrob_mujoco_environment)

find_package(pybind11 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(mujoco REQUIRED)
find_package(OpenMP REQUIRED)

# same manual paths as labrob_mujoco_environment/CMakeLists.txt (neither project installs a
# find_package/config for these three)
set(blasfeo_INCLUDE_DIRS /opt/blasfeo/include)
set(hpipm_INCLUDE_DIRS /opt/hpipm/include)
set(blasfeo_LIBRARIES /opt/blasfeo/lib/libblasfeo.a)
set(hpipm_LIBRARIES /opt/hpipm/lib/libhpipm.a)
set(qpOASES_LIBRARIES /usr/local/lib/libqpOASES.a)
set(qpOASES_INCLUDE_DIRS /usr/local/include)

pybind11_add_module(labrob_bridge bridge.cpp)
add_definitions(-DCOAL_DISABLE_HPP_FCL_WARNINGS)
target_include_directories(labrob_bridge PRIVATE
    ${LABROB_SRC}/include
    ${qpOASES_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
    ${hpipm_INCLUDE_DIRS}
)
target_link_libraries(labrob_bridge PRIVATE
    ${LABROB_SRC}/build/libWalkingControllerLibrary.a
    pinocchio::pinocchio
    ${hpipm_LIBRARIES}
    ${blasfeo_LIBRARIES}
    ${qpOASES_LIBRARIES}
    jsoncpp_lib
    OpenMP::OpenMP_CXX
    mujoco::mujoco
)
