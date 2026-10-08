import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from arena.world import Arena, Command
from arena.romoco import RoMoCo, STANDING_POSE, STANDING_BASE_Z, _quat_to_eulerZYX

arena = Arena(terrain="flat", render=False)
pol = RoMoCo()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

dt_ctrl = arena.model.opt.timestep
prev_ypr = None

def q_dq_findiff(st):
    global prev_ypr
    yaw, pitch, roll = _quat_to_eulerZYX(st.base_quat)
    if prev_ypr is None:
        yaw_dot = pitch_dot = roll_dot = 0.0
    else:
        py, pp, pr = prev_ypr
        yaw_dot = (yaw - py) / dt_ctrl
        pitch_dot = (pitch - pp) / dt_ctrl
        roll_dot = (roll - pr) / dt_ctrl
    prev_ypr = (yaw, pitch, roll)
    legs_q = st.get(st.q, pol.joints)
    legs_qd = st.get(st.qd, pol.joints)
    q = np.concatenate(([st.base_pos[0], st.base_pos[1], st.base_pos[2], yaw, pitch, roll], legs_q))
    dq = np.concatenate(([st.lin_vel_w[0], st.lin_vel_w[1], st.lin_vel_w[2],
                          yaw_dot, pitch_dot, roll_dot], legs_qd))
    return q, dq

seconds = 16.0
n = int(seconds / dt_ctrl)
fall_t = None
log = []
for k in range(n):
    t = k * dt_ctrl
    st = arena.state()
    q, dq = q_dq_findiff(st)
    mode = 1 if t >= 1.0 else 0
    vx = 0.3 if t >= 10.0 else 0.0
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

result = {"method": "findiff", "fell": fall_t is not None, "fall_t": fall_t,
          "max_x": max(l[2] for l in log), "log_tail": log[-14:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
