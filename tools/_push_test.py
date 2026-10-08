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
PUSH_T0, PUSH_T1 = 5.0, 5.3
PUSH_FORCE = 50.0  # N, lateral (y)
triggered = False
seconds = 10.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
import mujoco
body_id = mujoco.mj_name2id(arena.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
for k in range(n):
    t = k * dt
    if not triggered and t >= TRIGGER_T:
        pol._wm.trigger_walk()
        triggered = True
        print(f"triggered walk at t={t:.3f}", flush=True)
    arena.data.xfrc_applied[body_id] = 0
    if PUSH_T0 <= t < PUSH_T1:
        arena.data.xfrc_applied[body_id][1] = PUSH_FORCE
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 50 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[1]),4), fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {"fell": fall_t is not None, "fall_t": fall_t, "log_tail": log[-16:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
