#!/usr/bin/env bash
# Builds bridge/wbmpc (pybind11 module wrapping wb_humanoid_mpc's WBMpcMrtJointController)
# inside the wbmpc container. Needs the colcon workspace built first (PARALLEL_JOBS default 8):
#
#   docker run -d --name wbmpc --network host --ipc host \
#       -v "$PWD:/workspace/g1_stairs" \
#       -v "$PWD/third_party/wb_humanoid_mpc:/wb_humanoid_mpc_ws/src/wb_humanoid_mpc" \
#       wb-humanoid-mpc-base:latest sleep infinity
#   docker exec wbmpc git config --global --add safe.directory /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
#   docker exec wbmpc git -C /wb_humanoid_mpc_ws/src/wb_humanoid_mpc submodule update --init --recursive
#   docker exec wbmpc bash -c 'source /opt/ros/jazzy/setup.bash && cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc && \
#       make build-all PARALLEL_JOBS=8 EXTRA_CMAKE_ARGS=-DCMAKE_POSITION_INDEPENDENT_CODE=ON'
#   docker exec -u root wbmpc bash -c 'apt-get update -qq && apt-get install -y -qq python3-pybind11 pybind11-dev'
#
# (EXTRA_CMAKE_ARGS=-DCMAKE_POSITION_INDEPENDENT_CODE=ON is required: a shared pybind11 .so can't
# link the colcon workspace's static libs otherwise.)
#
# usage: tools/build_wbmpc_bridge.sh
set -euo pipefail

if [ "$(docker inspect -f "{{.State.Running}}" wbmpc 2>/dev/null)" != "true" ]; then
    echo "wbmpc container not running; see this script's header for how to set it up" >&2
    exit 1
fi

docker exec -w /workspace/g1_stairs/bridge/wbmpc wbmpc bash -c "
    source /opt/ros/jazzy/setup.bash && source /wb_humanoid_mpc_ws/install/setup.bash && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=/usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11 >/dev/null && \
    make -j\$(nproc)
"

echo "--- built: bridge/wbmpc/build/wbmpc_bridge.cpython-*.so ---"
