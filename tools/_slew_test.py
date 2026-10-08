import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 2000.0

arena = Arena(terrain="flat", render=False)
arena.tau_rate_limit = RATE
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

seconds = 16.0
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
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 100 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4), fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {"rate": RATE, "fell": fall_t is not None, "fall_t": fall_t, "max_x": max(l[2] for l in log), "log_tail": log[-10:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
