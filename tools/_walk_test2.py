import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

arena = Arena(terrain="flat", render=False)
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

TRIGGER_T = 2.0
triggered = False
seconds = 8.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
for k in range(n):
    t = k * dt
    if not triggered and t >= TRIGGER_T:
        pol._wm.trigger_walk()
        triggered = True
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 15 == 0:
        log.append((round(t,3), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4)))
    arena.step_physics(1)

result = {"full_log": log}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
