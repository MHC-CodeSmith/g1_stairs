#!/usr/bin/env bash
# Isaac Sim video of a policy on the staircase: 5.1 physics rollout -> 6.0 RTX render -> MP4.
#   g1_strut/make_isaac_video.sh logs/g1_dwaq_strut/<run>/model_2999.pt strut [--task g1_dwaq_strut --vx 0.9]
# Checkpoint paths are relative to the repo (mounted at /workspace/g1_stairs) or absolute inside the container.
set -euo pipefail
cd "$(dirname "$0")/.."
ckpt=$1; name=$2; shift 2
gpu=${GPU:-0}
docker compose run --rm -T -e NVIDIA_VISIBLE_DEVICES=$gpu train \
  g1_strut/rollout_isaac.py --checkpoint "$ckpt" --out "output/$name.npz" "$@" 2>&1 \
  | grep -E "^t=|^ |====|Traceback|Error"
docker compose run --rm -T -e NVIDIA_VISIBLE_DEVICES=$gpu render \
  g1_strut/render_isaac6.py --rollout "output/$name.npz" --frames "output/frames_$name" 2>&1 | grep -E "\[render\]|Traceback|Error"
docker run --rm -v "$PWD:/w" -w /w --entrypoint python g1-stairs:latest -c "
import glob, imageio.v2 as iio
w = iio.get_writer('output/$name.mp4', fps=25, codec='libx264', quality=8, macro_block_size=8)
for f in sorted(glob.glob('output/frames_$name/*.png')): w.append_data(iio.imread(f))
w.close()"
docker run --rm -v "$PWD/output:/o" --entrypoint sh g1-stairs:latest -c "rm -rf /o/frames_$name && chown $(id -u):$(id -g) /o/$name.mp4"
echo "video: output/$name.mp4"
