import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
from arena.world import Arena, Command
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

FORCE = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
PUSH_T0 = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

arena = Arena(terrain="flat", render=False)
arena.tau_rate_limit = 500.0
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

pelvis_id = mujoco.mj_name2id(arena.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

seconds = 14.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
triggered = False
PUSH_T1 = PUSH_T0 + 0.2
for k in range(n):
    t = k * dt
    if not triggered and t >= 2.0:
        pol._wm.trigger_walk()
        triggered = True
    arena.data.xfrc_applied[pelvis_id] = 0
    if PUSH_T0 <= t < PUSH_T1:
        arena.data.xfrc_applied[pelvis_id][1] = FORCE
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, Command()), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 100 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[1]),4), fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {"force": FORCE, "push_t0": PUSH_T0, "fell": fall_t is not None, "fall_t": fall_t, "log_tail": log[-12:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
