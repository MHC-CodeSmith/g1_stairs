"""Safe100 adapter check: run Safe100Humanoid's own mjlab env (its play config, 13 cm stairs) with its ONNX policy,
and at every step compare the actor observation mjlab computes with the one arena.safe100 builds from the same robot
state (gyro sensor, projected gravity, command, joint positions/velocities). Also reports the runs' success as its
evaluate_stairs.py defines it (max progress >= 2.65 m).

  docker run --rm --gpus '"device=1"' -v $PWD:/workspace/g1_stairs g1-mjlab tools/check_safe100.py [cbf|nominal]
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE = os.path.join(REPO, "third_party/Safe100Humanoid")
sys.path.insert(0, SAFE)
sys.path.insert(0, REPO)

import mjlab.tasks  # noqa: E402,F401
import onnxruntime as ort  # noqa: E402
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from arena.safe100 import Safe100  # noqa: E402
from arena.world import Command  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "cbf"
task = {"cbf": "Unitree-G1-Stairs-CBF", "nominal": "Unitree-G1-Stairs-Nominal"}[variant]
N = 16
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = N
st_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["forward_stairs"]
st_cfg.step_height_range = (0.13, 0.13)
cfg.scene.terrain.terrain_generator.num_rows = 1
cfg.scene.terrain.max_init_terrain_level = 0
cfg.episode_length_s = 20.0
cfg.seed = 42
if hasattr(cfg.actions["joint_pos"], "enabled"):
    cfg.actions["joint_pos"].enabled = False          # filter off: the policy alone, as in the arena
env = ManagerBasedRlEnv(cfg, device="cuda:0")
sess = ort.InferenceSession(os.path.join(SAFE, "results/models", variant, "policy.onnx"),
                            providers=["CPUExecutionProvider"])
obs, _ = env.reset()
robot = env.scene["robot"]
names = list(robot.joint_names)
ad = Safe100(variant)
assert names == list(ad.joints), names
gyro = env.scene["robot"].data  # noqa: F841


class S:  # the fields arena.safe100 reads
    pass


def state(i):
    s = S()
    d = robot.data
    s.q = d.joint_pos[i].cpu().numpy().astype(float)
    s.qd = d.joint_vel[i].cpu().numpy().astype(float)
    s.ang_vel_b = env.scene.sensors["robot/imu_ang_vel"].data[i].cpu().numpy().astype(float) \
        if "robot/imu_ang_vel" in env.scene.sensors else d.root_link_ang_vel_b[i].cpu().numpy()
    s.gravity_b = d.projected_gravity_b[i].cpu().numpy().astype(float)
    s.get = lambda arr, joints: arr
    s.t = 0.0
    return s


ad.reset(state(0))
worst, per = 0.0, np.zeros(7)
bounds = np.cumsum([0] + [15, 15, 15, 10, 145, 145, 60])
max_prog = np.zeros(N)
origin = env.scene.env_origins[:, 0].cpu().numpy()
alive = np.ones(N, bool)
with torch.inference_mode():
    for k in range(int(12.0 / env.step_dt)):
        o = obs["actor"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        o = o.cpu().numpy()
        cmd = env.command_manager.get_command("twist")[0].cpu().numpy().astype(float)
        mine = ad.obs(state(0), Command(*cmd))
        a_ref = np.concatenate([sess.run(None, {"obs": o[i:i + 1].astype(np.float32)})[0] for i in range(N)])
        ad.last = a_ref[0].astype(float)
        ad.k += 1
        if alive[0]:
            diff = np.abs(mine - o[0])
            worst = max(worst, float(diff.max()))
            for j in range(7):
                per[j] = max(per[j], diff[bounds[j]:bounds[j + 1]].max())
        obs, _, term, trunc, _ = env.step(torch.as_tensor(a_ref, device="cuda:0"))
        done = (term | trunc).cpu().numpy()
        alive &= ~done
        x = robot.data.root_link_pos_w[:, 0].cpu().numpy() - origin
        max_prog = np.where(alive, np.maximum(max_prog, x), max_prog)
print(f"[safe100_{variant}] max |obs diff| {worst:.2e}  per term " +
      ", ".join(f"{n} {v:.1e}" for n, v in zip(["ang", "grav", "cmd", "phase", "q", "qd", "act"], per)))
print(f"[safe100_{variant}] its own env, 13 cm stairs, filter off, 12 s: success (progress >= 2.65 m) "
      f"{int((max_prog >= 2.65).sum())}/{N}, still standing {int(alive.sum())}/{N}")


# ---- same policy on mjlab's compiled MjModel, stepped by plain CPU MuJoCo (no mjlab / MuJoCo-Warp) -----------------
import mujoco  # noqa: E402

from arena.world import joint_index, state_from  # noqa: E402

m = env.sim.mj_model
print("mjlab model: timestep", m.opt.timestep, "cone", m.opt.cone, "iterations", m.opt.iterations,
      "impratio", m.opt.impratio, "integrator", m.opt.integrator, "nu", m.nu)
mujoco.mj_saveModel(m, os.path.join(REPO, "output/safe100_mjlab_model.mjb"))
d = mujoco.MjData(m)
d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
d.qvel[:] = 0
mujoco.mj_forward(m, d)
names, qadr, vadr = joint_index(m)
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
tor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
names = [n.split("/")[-1] for n in names]
act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).split("/")[-1] for i in range(m.nu)]
p2 = Safe100(variant)
st = state_from(m, d, pel, tor, qadr, vadr, names, 0.0)
p2.reset(st)
x0, fell = d.qpos[0], None
for k in range(int(12.0 / 0.02)):
    st = state_from(m, d, pel, tor, qadr, vadr, names, d.time)
    st.get = lambda arr, joints: np.array([arr[names.index(j)] for j in joints])
    tgt = p2.act(st, Command(0.6, 0, 0))
    for i, a in enumerate(act_names):
        d.ctrl[i] = tgt[list(p2.joints).index(a.replace("_joint", "_joint"))] if a in p2.joints else d.ctrl[i]
    for _ in range(4):
        mujoco.mj_step(m, d)
    if fell is None and (st.gravity_b[2] > -0.5):
        fell = d.time
print(f"[safe100_{variant}] mjlab's own compiled model on CPU MuJoCo, 0.6 m/s: fell at {fell}, progress "
      f"{d.qpos[0] - x0:.2f} m, pelvis z {d.qpos[2]:.2f}")
