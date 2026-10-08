import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command
from arena.romoco import RoMoCo, STANDING_POSE, STANDING_BASE_Z

VX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1

arena = Arena(terrain="flat", render=False)
pol = RoMoCo()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

seconds = 16.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
import numpy as np
for k in range(n):
    t = k * dt
    st = arena.state()
    q, dq = pol._q_dq(st)
    mode = 1 if t >= 1.0 else 0
    vx = VX if t >= 10.0 else 0.0
    cmd_values = [vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = pol._ctrl.update(q.tolist(), dq.tolist(), mode, cmd_values)
    q_des = np.asarray(out["joint_positions"])
    qd_des = np.asarray(out["joint_velocities"])
    pol.kp = np.asarray(out["joint_kp"]); pol.kd = np.asarray(out["joint_kd"])
    ff = np.asarray(out["joint_torques_ff"])
    pol.tau_ext = pol.kd * qd_des + ff
    arena.set_targets(pol.joints, q_des, pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if k % 100 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4), mode, vx, fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.3:
        break

result = {"vx": VX, "fell": fall_t is not None, "fall_t": fall_t, "max_x": max(l[2] for l in log), "log_tail": log[-14:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
