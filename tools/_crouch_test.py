import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath("/workspace/g1_stairs/x")))
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

MODE = sys.argv[1] if len(sys.argv) > 1 else "centroidal"

from arena.world import Arena, Command, REPO

if MODE == "centroidal":
    from arena.wbmpc_centroidal import CentroidalMpc as Pol, STANDING_POSE, STANDING_BASE_Z
else:
    from arena.wbmpc import WbMpc as Pol, STANDING_POSE, STANDING_BASE_Z

arena = Arena(terrain="flat", render=False)
pol = Pol()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

schedule = [(0.0, None), (2.0, 0.62), (6.0, 0.52), (10.0, 0.70)]
seconds = 14.0
fall_t = None
cur_h = None
err_052 = []
dt = arena.model.opt.timestep
n = int(seconds / dt)
log = []
for k in range(n):
    t = k * dt
    for (ts, h) in schedule:
        if t >= ts:
            cur_h = h
    cmd = Command(height=cur_h)
    st = arena.state()
    arena.set_targets(pol.joints, pol.act(st, cmd), pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    if fall_t is None and arena.fallen(st):
        fall_t = st.t
    if cur_h == 0.52 and fall_t is None:
        err_052.append(abs(st.base_pos[2] - 0.52))
    if k % 50 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), cur_h, fall_t))
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {
    "mode": MODE,
    "fell": fall_t is not None,
    "fall_t": fall_t,
    "min_err_at_052": min(err_052) if err_052 else None,
    "log_tail": log[-10:],
}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
