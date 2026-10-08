import os, sys, json, time
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from arena.world import Arena, Command
from arena.wbmpc import WbMpc, STANDING_POSE, STANDING_BASE_Z
MODE=sys.argv[1]; ARG=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
terrain = "stairs" if MODE=="stairs" else ("rough" if MODE=="rough" else "flat")
arena = Arena(terrain=terrain, render=False)
VX = 0.3 if MODE=="stairs" else 0.12
if MODE=="push": arena.push = (10.0, 0.2, (0.0, ARG, 0.0))
pol = WbMpc()
x0 = 0.0
arena.reset(base_z=STANDING_BASE_Z, joint_pos=STANDING_POSE, x=x0)
st = arena.state(); pol.reset(st)
dt = arena.model.opt.timestep; T = 40.0 if MODE=="stairs" else 30.0
fall_t=None; maxx=x0; maxz=0.0; minz=9; w0=time.monotonic()
for k in range(int(T/dt)):
    t=k*dt; st=arena.state()
    if MODE=="crouch":
        h = STANDING_BASE_Z if t<2 else (0.62 if t<6 else (0.52 if t<10 else 0.70)); vx=0.0
    else:
        h = STANDING_BASE_Z; vx = 0.0 if t<2 else min(VX, VX*(t-2)/3.0)
    q_des = pol.act(st, Command(vx=vx, height=h))
    arena.set_targets(pol.joints, q_des, pol.kp, pol.kd); arena.set_external_torque(pol.joints, pol.tau_ext)
    maxx=max(maxx,float(st.base_pos[0])); maxz=max(maxz,float(st.base_pos[2]))
    if t>11 or MODE!="crouch": minz=min(minz,float(st.base_pos[2]))
    if fall_t is None and arena.fallen(st): fall_t=st.t
    arena.step_physics(1)
    if fall_t is not None and t>fall_t+0.3: break
    d=w0+(k+1)*dt-time.monotonic()
    if d>0: time.sleep(d)
print("RESULT_JSON:"+json.dumps({"mode":MODE,"arg":ARG,"fell":fall_t is not None,"fall_t":fall_t,"max_x":round(maxx,3),"max_z":round(maxz,3),"z_end":round(float(st.base_pos[2]),3)}),flush=True)
os._exit(0)
