#!/usr/bin/env bash
# Builds and runs labrob_mujoco_environment's main_sim standalone demo headlessly (no
# physical/forwarded X server needed): MujocoUI.hpp always opens a GLFW window with no
# offscreen render path, so we give it a virtual one (Xvfb + llvmpipe software GL) inside
# the container itself. Logs land in /tmp/robot_logs inside the container (odom_pos.txt,
# pelvis_rpy.txt, com_position.txt, etc. - one row per control step).
#
# Also used as the "get the labrob container up" step for tools/build_labrob_bridge.sh,
# which needs the same container and the same -fPIC WalkingControllerLibrary.a build.
#
# usage: tools/run_labrob_headless.sh [duration_seconds]
set -euo pipefail

DURATION="${1:-15}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(docker inspect -f "{{.State.Running}}" labrob 2>/dev/null)" != "true" ]; then
    docker rm -f labrob >/dev/null 2>&1 || true
    docker run -d --name labrob --runtime nvidia --network host --ipc host \
        -v "${ROOT_DIR}:/workspace/g1_stairs" \
        -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
        labrob-base:latest sleep infinity
    docker exec -u root labrob bash -c "mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix"
fi

LABROB_DIR=/workspace/g1_stairs/third_party/labrob_mujoco_environment
# -fPIC: also needed by bridge/labrob (a shared .so cannot link a non-PIC static lib), so
# always build it this way rather than having two differently-built copies of the library.
docker exec -w "${LABROB_DIR}" labrob bash -c \
    "mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Wno-dev >/dev/null && make -j\$(nproc)"

docker exec -u root labrob bash -c "pkill -9 -f '[X]vfb' >/dev/null 2>&1; rm -rf /tmp/.X11-unix/* /tmp/.X99-lock /tmp/robot_logs; mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix"

docker exec -w "${LABROB_DIR}/build" labrob bash -c "
    setsid Xvfb :99 -screen 0 1280x1024x24 &
    sleep 2
    (sleep ${DURATION} && kill -INT -\$\$) &
    echo y | DISPLAY=:99 ./main_sim --sim
"

echo "--- logs written to /tmp/robot_logs inside the labrob container ---"
docker exec labrob bash -c "tail -n 3 /tmp/robot_logs/odom_pos.txt /tmp/robot_logs/pelvis_rpy.txt"
