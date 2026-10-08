import os, sys, json, time
sys.path.insert(0, "/workspace/g1_stairs")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from arena.world import Arena, Command
from arena.romoco import RoMoCo, STANDING_POSE, STANDING_BASE_Z
VX=float(sys.argv[1]); PACE=(sys.argv[2]=="1") if len(sys.argv)>2 else True
if os.environ.get("NATIVE"):
    arena = Arena(terrain="flat", render=False, robot_xml="/home/docker/RoMoCo/src/g1_stack/model_files/g1_29_withsensor.xml", native=True)
else:
    arena = Arena(terrain="flat", render=False)
if os.environ.get("DT"): arena.model.opt.timestep = float(os.environ["DT"])
if os.environ.get("DAMP"):
    arena.model.dof_damping[6:] = float(os.environ["DAMP"])
if os.environ.get("POSE") == "crouch":
    STANDING_POSE = dict(STANDING_POSE)
    for side in ("left","right"):
        STANDING_POSE[side+"_knee_joint"]=0.4; STANDING_POSE[side+"_hip_pitch_joint"]=-0.2; STANDING_POSE[side+"_ankle_pitch_joint"]=-0.2
if os.environ.get("UPPER"):
    f=float(os.environ["UPPER"])
    for i,n in enumerate(arena.joint_names):
        if any(k in n for k in ("waist","shoulder","elbow","wrist","hand_")):
            arena.hold_kp[i]*=f; arena.hold_kd[i]*=f**0.5
BASEZ = float(os.environ.get("BASEZ", STANDING_BASE_Z))
pol = RoMoCo()
arena.reset(base_z=BASEZ, joint_pos=STANDING_POSE)
st = arena.state(); pol.reset(st)
dt = arena.model.opt.timestep; fall_t=None; maxx=0.0; w0=time.monotonic()
for k in range(int(30.0/dt)):
    t=k*dt; st=arena.state()
    R0=float(os.environ.get("RAMP0","2")); vx = (0.0 if t<R0 else min(VX, VX*(t-R0)/3.0)) if R0>=0 else VX
    q_des = pol.act(st, Command(vx=vx, height=BASEZ))
    arena.set_targets(pol.joints, q_des, pol.kp, pol.kd); arena.set_external_torque(pol.joints, pol.tau_ext)
    maxx=max(maxx,float(st.base_pos[0]))
    if fall_t is None and arena.fallen(st): fall_t=st.t
    arena.step_physics(1)
    if fall_t is not None and t>fall_t+0.3: break
    if PACE:
        d=w0+(k+1)*dt-time.monotonic()
        if d>0: time.sleep(d)
print("RESULT_JSON:"+json.dumps({"vx":VX,"alpha":os.environ.get("ROMOCO_ALPHA","1.0"),"fell":fall_t is not None,"fall_t":fall_t,"max_x":round(maxx,3)}),flush=True)
os._exit(0)
