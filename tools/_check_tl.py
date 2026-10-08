import sys
sys.path.insert(0, "/workspace/g1_stairs")
from arena.world import Arena
from arena.labrob import STANDING_POSE, STANDING_BASE_Z
arena = Arena(terrain="flat", render=False)
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
import numpy as np
for n, lim in zip(arena.joint_names, arena.tau_limit):
    if lim < 1e5:
        pass
    else:
        print("UNLIMITED:", n, lim)
print("min/max tau_limit:", arena.tau_limit.min(), arena.tau_limit.max())
