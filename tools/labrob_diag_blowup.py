import os, sys, json
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from arena.world import Arena, Command
from arena.labrob import Labrob, STANDING_POSE, STANDING_BASE_Z

PERIOD = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5

arena = Arena(terrain="flat", render=False)
pol = Labrob()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state()
pol.reset(st)

seconds = 20.0
dt = arena.model.opt.timestep
n = int(seconds / dt)
fall_t = None
log = []
last_trigger = -1.0
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
    if k % 50 == 0:
        log.append((round(t,2), round(float(st.base_pos[2]),4), round(float(st.base_pos[0]),4), fall_t))
    if k%500==0 or (7.30<t<7.40 and k%2==0):
        j=pol.joints.index("left_shoulder_roll_joint"); j2=pol.joints.index("left_shoulder_pitch_joint"); j3=pol.joints.index("left_elbow_joint")
        print("L t=%.3f tauSR=%.1f tauSP=%.1f tauEl=%.1f qSR=%.3f qdSR=%.3f z=%.4f"%(t,pol.tau_ext[j],pol.tau_ext[j2],pol.tau_ext[j3],st.q[arena.joint_names.index("left_shoulder_roll_joint")] if False else st.get(st.q,pol.joints)[j],st.get(st.qd,pol.joints)[j],st.base_pos[2]),flush=True)
    arena.step_physics(1)
    if fall_t is not None and t > fall_t + 0.5:
        break

result = {"period": PERIOD, "fell": fall_t is not None, "fall_t": fall_t, "max_x": max(l[2] for l in log), "log_tail": log[-20:]}
print("RESULT_JSON:" + json.dumps(result), flush=True)
os._exit(0)
