import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from arena.world import Arena, Command
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

arena = Arena(terrain="flat", render=False)
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

seconds = 13.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
triggered = False
for k in range(n):
    t = k * dt
    if not triggered and t >= 2.0:
        pol._wm.trigger_walk()
        triggered = True
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    max_tau = float(np.max(np.abs(pol.tau_ext)))
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if 10.5 <= t <= 12.0:
        log.append((round(t,3), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4),
                     round(max_tau,2), round(float(st.lin_vel_w[2]),3)))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 1.0:
        break

print("fell_t", fall_t, flush=True)
for row in log:
    print(row, flush=True)
import os as _os
_os._exit(0)
