import os, sys, json, time
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from arena.world import Arena, Command
from arena.wbmpc import WbMpc, STANDING_POSE, STANDING_BASE_Z
VX=float(sys.argv[1]); PUSH=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
arena = Arena(terrain="flat", render=False)
if PUSH: arena.push = (10.0, 0.2, (0.0, PUSH, 0.0))
pol = WbMpc()
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE)
st = arena.state(); pol.reset(st)
dt = arena.model.opt.timestep; fall_t=None; maxx=0.0; maxyaw=0.0; w0=time.monotonic()
for k in range(int(30.0/dt)):
    t=k*dt; st=arena.state()
    vx = 0.0 if t<2 else min(VX, VX*(t-2)/3.0)
    q_des = pol.act(st, Command(vx=vx, height=STANDING_BASE_Z))
    arena.set_targets(pol.joints, q_des, pol.kp, pol.kd); arena.set_external_torque(pol.joints, pol.tau_ext)
    w,x,y,z = st.base_quat
    maxyaw=max(maxyaw,abs(np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))); maxx=max(maxx,float(st.base_pos[0]))
    if fall_t is None and arena.fallen(st): fall_t=st.t
    arena.step_physics(1)
    if fall_t is not None and t>fall_t+0.3: break
    d=w0+(k+1)*dt-time.monotonic()
    if d>0: time.sleep(d)
print("RESULT_JSON:"+json.dumps({"mode":"free","vx":VX,"push":PUSH,"fell":fall_t is not None,"fall_t":fall_t,"max_x":round(maxx,3),"max_yaw_deg":round(maxyaw,1)}),flush=True)
os._exit(0)
