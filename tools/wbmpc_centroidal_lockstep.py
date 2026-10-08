import os, sys, json, time
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from arena.world import Arena, Command
from arena.wbmpc_centroidal import CentroidalMpc, STANDING_POSE, STANDING_BASE_Z
VX = float(sys.argv[1]); SPS = int(sys.argv[2])
arena = Arena(terrain="flat", render=False)
pol = CentroidalMpc()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state(); pol.reset(st)
mpc = pol._mpc
mpc.set_lockstep(True)
dt = arena.model.opt.timestep
cnt = SPS   # solve on first step
fall_t=None; maxx=0.0; maxyaw=0.0
for k in range(int(30.0/dt)):
    t=k*dt
    st = arena.state()
    vx = 0.0 if t<2 else min(VX, VX*(t-2)/3.0)
    q_des = pol.act(st, Command(vx=vx, height=STANDING_BASE_Z))
    arena.set_targets(pol.joints, q_des, pol.kp, pol.kd)
    arena.set_external_torque(pol.joints, pol.tau_ext)
    cnt += 1
    if cnt >= SPS:
        cnt = 0
        c0 = mpc.solve_count(); mpc.grant_solve()
        t0 = time.monotonic()
        while mpc.solve_count() == c0 and time.monotonic()-t0 < 5: time.sleep(0.0005)
    w,x,y,z = st.base_quat
    maxyaw = max(maxyaw, abs(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))))
    maxx = max(maxx, float(st.base_pos[0]))
    if fall_t is None and arena.fallen(st): fall_t = st.t
    arena.step_physics(1)
    if fall_t is not None and t > fall_t+0.3: break
print("RESULT_JSON:"+json.dumps({"vx":VX,"steps_per_solve":SPS,"fell":fall_t is not None,"fall_t":fall_t,"max_x":round(maxx,3),"max_yaw_deg":round(maxyaw,1),"sim_t":round(t,1)}), flush=True)
os._exit(0)
