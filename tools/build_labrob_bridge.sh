#!/usr/bin/env bash
# Builds bridge/labrob (pybind11 module wrapping labrob_mujoco_environment's WalkingManager)
# inside the labrob container. Run tools/run_labrob_headless.sh at least once first (it builds
# the labrob container and the -fPIC WalkingControllerLibrary.a this links against).
#
# usage: tools/build_labrob_bridge.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Enables the real (not dead-code) standing->walking coop planner - see
# patches/labrob_enable_coop_walk.patch and docs/ARENA_REPORT.md for why. Idempotent: skips if
# already applied (e.g. a rebuild after the first apply).
if ! grep -q "if (switchWalkingState && false){" "${ROOT_DIR}/third_party/labrob_mujoco_environment/src/WalkingManager.cpp" 2>/dev/null; then
    patch -p1 -d "${ROOT_DIR}/third_party/labrob_mujoco_environment" < "${ROOT_DIR}/patches/labrob_enable_coop_walk.patch"
fi

if [ "$(docker inspect -f "{{.State.Running}}" labrob 2>/dev/null)" != "true" ]; then
    echo "labrob container not running; run tools/run_labrob_headless.sh first" >&2
    exit 1
fi

PYBIND11_CMAKE_DIR=$(docker exec labrob /opt/venvs/base/bin/python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

docker exec -w /workspace/g1_stairs/bridge/labrob labrob bash -c "
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR=${PYBIND11_CMAKE_DIR} \
        -DPYTHON_EXECUTABLE=/opt/venvs/base/bin/python3 >/dev/null && \
    make -j\$(nproc)
"

echo "--- built: bridge/labrob/build/labrob_bridge.cpython-*.so ---"
echo "import from third_party/labrob_mujoco_environment/build (relative URDF/scene paths assume that cwd)"
