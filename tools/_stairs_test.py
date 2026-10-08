import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command, STAIRS
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

arena = Arena(terrain="stairs", render=False)
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

seconds = 14.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
last_trigger = -1.0
PERIOD = 0.8
for k in range(n):
    t = k * dt
    if t >= 2.0 and t - last_trigger >= PERIOD:
        pol._wm.trigger_walk()
        last_trigger = t
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 100 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4), fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {"stairs_start": STAIRS["start"], "fell": fall_t is not None, "fall_t": fall_t,
          "max_x": max(l[2] for l in log), "reached_stairs": max(l[2] for l in log) >= STAIRS["start"],
          "log_tail": log[-14:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
