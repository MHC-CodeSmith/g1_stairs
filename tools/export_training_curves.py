"""TensorBoard scalars of a g1_body run -> output/g1_body_training.json (for tools/report_figures.py).

  docker run --rm -v $PWD:/workspace/g1_stairs --entrypoint /isaac-sim/python.sh g1-isaaclab:latest \
      tools/export_training_curves.py logs/g1_body/2026-09-28_22-14-57
"""
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
e = EventAccumulator(glob.glob(os.path.join(root, run, "events*"))[0], size_guidance={"scalars": 0})
e.Reload()
keys = ["Train/mean_reward", "Train/mean_episode_length", "Body/height_err_crouch", "Body/crouch_frac",
        "Loss/distill", "Loss/distill_coef", "Curriculum/terrain_levels"]
out = {k: [(s.step, s.value) for s in e.Scalars(k)] for k in keys if k in e.Tags()["scalars"]}
json.dump(out, open(os.path.join(root, "output/g1_body_training.json"), "w"))
print({k: len(v) for k, v in out.items()})
