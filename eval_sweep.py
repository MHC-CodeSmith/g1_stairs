"""Headless robustness sweep for run_stairs.py: success rate over forward speed x start-x offset x start-yaw.

  docker run --rm -v $PWD/eval_sweep.py:/opt/eval_sweep.py:ro --entrypoint python g1-stairs:latest /opt/eval_sweep.py
"""
import itertools
import math
import sys

sys.argv = sys.argv[:1]
sys.path.insert(0, "/opt")
import run_stairs as rs  # noqa: E402

up = rs.load_upstream()
r = up.G1DwaqMujocoRunner(cfg=up.G1DwaqSim2SimCfg(), checkpoint_path=rs.DEFAULT_CKPT, model_path=rs.DEFAULT_SCENE)
stairs = rs.StairProfile(r.model)
ctrl_dt = r.cfg.sim.dt * r.cfg.sim.decimation

for vx in (0.6, 0.8, 0.9, 1.0):
    top = full = clean = n = 0
    for x0, yaw0 in itertools.product((0.0, 0.1, 0.2, 0.3), (-0.1, 0.0, 0.1)):
        rs.reset(r)
        r.data.qpos[0] = x0
        r.data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        rs.mujoco.mj_forward(r.model, r.data)
        r.obs_history[:] = r.get_current_obs()
        pilot, tr = rs.Autopilot(stairs, vx, 1.0, stairs.x_end + 0.6), rs.Tracker(stairs)
        for _ in range(int(45 / ctrl_dt)):
            yaw = rs.yaw_of(r.data.qpos[3:7])
            r.command_vel[:] = pilot.command(r.data.time, r.data.qpos[0], r.data.qpos[1], yaw)
            rs.policy_step(r)
            for _ in range(r.cfg.sim.decimation):
                rs.physics_substep(r)
            r.gait_phase_time += ctrl_dt
            tr.update(r.data.time, r.data.qpos[0:3], r.get_gravity_orientation(r.data.qpos[3:7])[2], yaw)
            if tr.fell or r.data.qpos[0] > stairs.x_end + 0.5:
                break
        ok = tr.reached_platform and not tr.fell and r.data.qpos[0] > stairs.x_end
        n += 1
        top += tr.reached_platform
        full += ok
        clean += ok and tr.max_backtrack < 0.3
    print(f"vx={vx:.1f}  reached top {top:2d}/{n}  full up+down crossing {full:2d}/{n}  clean (no backtrack) {clean:2d}/{n}", flush=True)
