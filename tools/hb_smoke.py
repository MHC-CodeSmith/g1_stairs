"""Smoke test: confirm HumanoidBench's native G1 stair env actually resets and steps (not just registered).

No trained G1 policy exists for this env (see docs/ARENA_REPORT.md), so this is not a benchmark run.

Requires humanoid-bench's own deps (torch, dm_control, gymnasium, mujoco, brax) installed, and MUJOCO_GL=osmesa (or
another headless GL backend) for its always-on offscreen renderer:

  cd third_party/humanoid-bench && pip install -e . gymnasium mujoco
  MUJOCO_GL=osmesa python3 ../../tools/hb_smoke.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import humanoid_bench  # noqa: E402,F401  registers envs

env = gym.make("g1-stair-v0")
obs, _ = env.reset(seed=0)
print("obs shape", obs.shape, "action shape", env.action_space.shape)
tot = 0.0
for i in range(200):
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
    tot += r
    if term or trunc:
        print(f"zero-action rollout fell and terminated at step {i} (expected: no motor torque)")
        break
else:
    print("ran 200 steps without ending")
print("total reward (zero action)", tot)
