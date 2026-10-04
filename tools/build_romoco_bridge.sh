#!/usr/bin/env bash
# Builds bridge/romoco (pybind11 module wrapping RoMoCo's BasicControllerStateMachine) inside the
# romoco container. Needs the colcon workspace built first, with persistent build/install volumes
# (colcon's build/install dirs are NOT checked out under third_party/RoMoCo - they must survive
# container recreation, unlike a plain bind mount of the source):
#
#   mkdir -p ~/g1_stairs_persist/romoco_{build,install}
#   docker run -d --name romoco --runtime nvidia --network host --ipc host \
#       -v "$PWD:/workspace/g1_stairs" \
#       -v "$PWD/third_party/RoMoCo:/home/docker/RoMoCo" \
#       -v ~/g1_stairs_persist/romoco_build:/home/docker/RoMoCo/build \
#       -v ~/g1_stairs_persist/romoco_install:/home/docker/RoMoCo/install \
#       --user docker -e HOME=/home/docker -w /home/docker/RoMoCo \
#       romoco-base:latest sleep infinity
#   docker exec romoco git config --global --add safe.directory /home/docker/RoMoCo
#   docker exec -d romoco bash -c 'source /opt/ros/humble/setup.bash && \
#       export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH && cd /home/docker/RoMoCo && \
#       colcon build --symlink-install --parallel-workers 8 \
#       --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON'
#   docker exec -u root romoco bash -c 'apt-get update -qq && apt-get install -y -qq python3-pybind11 pybind11-dev'
#
# romoco-base:latest is built from docker/Dockerfile.romoco (our copy of RoMoCo's own
# docker/Dockerfile, patched to pin Pinocchio to v3.9.0 - its own unpinned HEAD clone hits a
# boost::variant/GCC incompatibility building against this environment's Boost).
#
# colcon's --symlink-install makes config-file symlinks under install/ absolute to
# /home/docker/RoMoCo/src/... (the path it was built from) - they only resolve inside a container
# with that exact mount, which arena/romoco.py's CONFIG_FOLDER constant assumes.
#
# usage: tools/build_romoco_bridge.sh
set -euo pipefail

if [ "$(docker inspect -f "{{.State.Running}}" romoco 2>/dev/null)" != "true" ]; then
    echo "romoco container not running; see this script's header for how to set it up" >&2
    exit 1
fi

docker exec -w /workspace/g1_stairs/bridge/romoco romoco bash -c "
    source /opt/ros/humble/setup.bash && source /home/docker/RoMoCo/install/setup.bash && \
    export CMAKE_PREFIX_PATH=/opt/openrobots:\$CMAKE_PREFIX_PATH && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=/usr/lib/python3/dist-packages/pybind11/share/cmake/pybind11 >/dev/null && \
    make -j\$(nproc)
"

echo "--- built: bridge/romoco/build/romoco_bridge.cpython-*.so ---"
