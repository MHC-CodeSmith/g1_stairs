"""skills.adapters.Gr00tWbc (batched torch, Isaac side) vs arena.policies.Gr00tWbc (checked against GR00T's own
MuJoCo controller in arena/check_adapters.py) on random robot states and commands, including a reset mid-sequence.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tests/test_gr00t_adapter.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.policies import Gr00tWbc as ArenaGr00t  # noqa: E402
from arena.world import Command, State  # noqa: E402
from skills import registry  # noqa: E402
from skills.adapters import G1_JOINTS_LAB, G1_JOINTS_MJ, RobotState  # noqa: E402

rng = np.random.default_rng(0)
N, STEPS = 4, 30
skill = registry.load("gr00t_wbc")
refs = [ArenaGr00t() for _ in range(N)]
names = G1_JOINTS_LAB                    # the host env uses Isaac Lab order; the adapter must look joints up by name
perm = [G1_JOINTS_MJ.index(n) for n in names]
worst = 0.0
for k in range(STEPS):
    q = rng.normal(0, 0.3, (N, 29))
    qd = rng.normal(0, 1.0, (N, 29))
    ang = rng.normal(0, 0.5, (N, 3))
    g = rng.normal(0, 0.1, (N, 3)) + [0, 0, -1]
    cmd = np.stack([rng.uniform(-0.6, 0.6, N) * (k % 3 != 0), rng.uniform(-0.3, 0.3, N) * (k % 3 != 0),
                    rng.uniform(-0.5, 0.5, N) * (k % 3 != 0), rng.uniform(0.5, 0.74, N)], 1)
    if k == 0 or k == 17:                # reset everything at the start and env 1..2 mid-sequence
        ids = list(range(N)) if k == 0 else [1, 2]
        if k:
            skill.reset(torch.tensor(ids))
        for i in ids:
            refs[i].reset(None)
    st = RobotState(names, torch.tensor(q[:, perm], dtype=torch.float32), torch.tensor(qd[:, perm], dtype=torch.float32),
                    torch.tensor(ang, dtype=torch.float32), torch.tensor(g, dtype=torch.float32),
                    torch.zeros(N), torch.zeros(N))
    out = skill.act(st, torch.tensor(cmd, dtype=torch.float32)).numpy()
    for i in range(N):
        s = State(0, q[i], qd[i], None, None, None, None, ang[i], g[i], None, None, 0.7, list(G1_JOINTS_MJ))
        ref = refs[i].act(s, Command(*cmd[i, :3], height=float(cmd[i, 3])))
        worst = max(worst, float(np.abs(out[i] - ref).max()))
print(f"gr00t_wbc skill vs arena adapter: max |target diff| = {worst:.2e} over {N}x{STEPS} steps")
assert worst < 1e-4
print("PASS")
