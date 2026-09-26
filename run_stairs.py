"""Unitree G1 (29-DoF) stair climbing in MuJoCo with the pretrained DWAQ blind-locomotion policy.

Reuses the upstream G1DWAQ_Lab sim2sim runner (policy, observation layout, PD gains, timing) unchanged
and adds:
  * an autopilot that walks the robot up the 10-step flight, across the platform and down the other side
    (forward velocity + heading hold toward the stair centerline, the same command interface used in training)
  * a headless mode that renders an MP4 with a tracking camera
  * pass/fail metrics measured against the stair geometry in the scene

Modes:
  record  - headless, offscreen render to MP4 (MUJOCO_GL=egl or osmesa)
  viewer  - interactive MuJoCo viewer in real time (needs an X display)
"""

import argparse
import importlib.util
import math
import os
import sys
import types

import mujoco
import numpy as np
import torch

torch.set_num_threads(1)  # a 512-256-128 MLP is fastest single-threaded; avoids oversubscription

ROOT = os.environ.get("TIENKUNG_ROOT", "/opt/TienKung-Lab")
UPSTREAM = os.path.join(ROOT, "legged_lab/scripts/sim2sim_g1_dwaq.py")
DEFAULT_SCENE = os.path.join(ROOT, "legged_lab/assets/unitree/g1/mjcf/g1_stairs_scene.xml")
DEFAULT_CKPT = os.path.join(ROOT, "logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt")


def load_upstream():
    # pynput needs an X server at import time; the autopilot does not use it, so stub it when unavailable.
    try:
        import pynput  # noqa: F401
    except Exception:
        stub = types.ModuleType("pynput")
        stub.keyboard = types.SimpleNamespace(Listener=None, Key=None, KeyCode=None)
        sys.modules["pynput"] = stub
    spec = importlib.util.spec_from_file_location("sim2sim_g1_dwaq", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.G1DwaqMujocoRunner.compute_gait_phase = _gait_phase_training_order
    return module


def _gait_phase_training_order(self):
    """Gait-phase obs in the order the policy was trained with.

    Training (g1_dwaq_env.compute_current_observations) appends sin(2*pi*leg_phase) then cos(2*pi*leg_phase),
    i.e. [sin_L, sin_R, cos_L, cos_R]; upstream sim2sim_g1_dwaq.py sends [sin_L, cos_L, sin_R, cos_R].
    """
    period, offset = self.cfg.gait_phase.period, self.cfg.gait_phase.offset
    phase = np.array([(self.gait_phase_time % period) / period,
                      ((self.gait_phase_time / period) + offset) % 1.0])
    return np.concatenate([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]).astype(np.float32)


class StairProfile:
    """Terrain height under a point, built from the stair/platform box geoms in the scene."""

    def __init__(self, model):
        self.boxes = []
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and ("stair" in name or "platform" in name):
                pos, size = model.geom_pos[g], model.geom_size[g]
                self.boxes.append((pos[0] - size[0], pos[0] + size[0], pos[1] - size[1], pos[1] + size[1],
                                   pos[2] + size[2], name))
        if not self.boxes:
            raise RuntimeError("scene has no geoms named stair*/platform*")
        self.top = max(b[4] for b in self.boxes)
        self.x_start = min(b[0] for b in self.boxes)
        self.x_end = max(b[1] for b in self.boxes)
        self.half_width = min(b[3] for b in self.boxes)
        platform = [b for b in self.boxes if b[4] == self.top]
        self.platform_x = (min(b[0] for b in platform), max(b[1] for b in platform))

    def height(self, x, y):
        h = 0.0
        for x0, x1, y0, y1, z, _ in self.boxes:
            if x0 <= x < x1 and y0 <= y < y1:
                h = max(h, z)
        return h


class Autopilot:
    """Forward velocity with a stiff heading hold (yaw -> 0) and lateral re-centering through the vy command.

    Steering back to the centerline with vy instead of yaw keeps the torso square to the steps; in a 12-start
    sweep (x/yaw offsets) this raised full up-and-down crossings from ~7/12 to 11-12/12 at vx 0.8-1.0 m/s.
    """

    def __init__(self, stairs, vx, settle, stop_x):
        self.stairs, self.vx, self.settle, self.stop_x = stairs, vx, settle, stop_x

    def command(self, t, x, y, yaw):
        if t < self.settle or x > self.stop_x:
            return np.zeros(3, dtype=np.float32)
        err = math.atan2(math.sin(-yaw), math.cos(-yaw))
        vy = float(np.clip(-1.0 * y, -0.3, 0.3))
        yaw_rate = float(np.clip(2.0 * err, -1.5, 1.5))
        return np.array([self.vx, vy, yaw_rate], dtype=np.float32)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def policy_step(r):
    """One 50 Hz control step: identical math to the upstream G1DwaqMujocoRunner.run() loop body."""
    obs = r.normalize_obs(r.get_current_obs())
    r.update_obs_history(obs)
    with torch.no_grad():
        act = r.policy.act_inference(torch.from_numpy(obs).unsqueeze(0),
                                     torch.from_numpy(r.get_flattened_obs_history().astype(np.float32)).unsqueeze(0))
    r.action[:] = np.clip(act.squeeze(0).numpy(), -r.cfg.sim.clip_actions, r.cfg.sim.clip_actions)


def physics_substep(r):
    r.data.ctrl[:r.num_actions] = r.pd_control(r.position_control())
    mujoco.mj_step(r.model, r.data)


def reset(r):
    mujoco.mj_resetData(r.model, r.data)
    r.set_initial_pose()
    r.action[:] = 0.0
    r.command_vel[:] = 0.0
    r.gait_phase_time = 0.0
    r.episode_length_buf = 0
    init = r.get_current_obs()
    for i in range(r.cfg.sim.dwaq_obs_history_length):
        r.obs_history[i] = init


class Tracker:
    def __init__(self, stairs):
        self.stairs = stairs
        self.max_x = 0.0
        self.max_clearance_z = 0.0  # pelvis height above world floor
        self.reached_platform = False
        self.fell = False
        self.fall_t = None
        self.min_rel_h = 9.0
        self.max_backtrack = 0.0  # largest drop of x below its running max [m]
        self.max_abs_yaw = 0.0

    def update(self, t, pos, grav_z, yaw=0.0):
        x, y, z = pos
        self.max_backtrack = max(self.max_backtrack, self.max_x - x)
        self.max_abs_yaw = max(self.max_abs_yaw, abs(yaw))
        ground = self.stairs.height(x, y)
        rel = z - ground
        self.max_x = max(self.max_x, x)
        self.max_clearance_z = max(self.max_clearance_z, z)
        if t > 1.0:
            self.min_rel_h = min(self.min_rel_h, rel)
        p0, p1 = self.stairs.platform_x
        if p0 <= x <= p1 and ground >= self.stairs.top - 1e-6 and rel > 0.55:
            self.reached_platform = True
        if not self.fell and (rel < 0.4 or grav_z > -0.6):
            self.fell, self.fall_t = True, t
        return ground, rel


def summarize(tr, r, stairs):
    x, y, z = r.data.qpos[0:3]
    upright = r.get_gravity_orientation(r.data.qpos[3:7])[2] < -0.9
    crossed = tr.reached_platform and (not tr.fell) and x > stairs.x_end and upright
    clean = crossed and tr.max_backtrack < 0.3
    print("\n" + "=" * 64)
    print(f" stairs: {len(stairs.boxes) - 1} steps + platform, top surface at {stairs.top:.3f} m, "
          f"x in [{stairs.x_start:.2f}, {stairs.x_end:.2f}]")
    print(f" peak pelvis height : {tr.max_clearance_z:.3f} m (standing on flat floor ~0.79 m)")
    print(f" reached platform   : {tr.reached_platform}")
    print(f" fell               : {tr.fell}" + (f" (t={tr.fall_t:.2f}s)" if tr.fell else ""))
    print(f" min pelvis-to-tread: {tr.min_rel_h:.3f} m")
    print(f" max |yaw| / backtr.: {math.degrees(tr.max_abs_yaw):.0f} deg / {tr.max_backtrack:.2f} m")
    print(f" final pose         : x={x:.2f} y={y:.2f} z={z:.2f} upright={upright}")
    print(f" RESULT             : {('SUCCESS - climbed up, crossed the platform and descended' + ('' if clean else ' (with a stumble/backtrack)')) if crossed else ('CLIMBED (reached top)' if tr.reached_platform and not tr.fell else 'FAILED')}")
    print("=" * 64)
    return crossed


def run_record(r, args, stairs):
    import imageio.v2 as imageio

    r.model.vis.global_.offwidth = max(r.model.vis.global_.offwidth, args.width)
    r.model.vis.global_.offheight = max(r.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(r.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = args.cam_distance, args.cam_azimuth, args.cam_elevation
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)

    pilot = Autopilot(stairs, args.vx, args.settle, stairs.x_end + 0.6)
    tr = Tracker(stairs)
    reset(r)
    next_frame, frame_dt = 0.0, 1.0 / args.fps
    steps = int(args.duration / (r.cfg.sim.dt * r.cfg.sim.decimation))
    for k in range(steps):
        t = r.data.time
        r.command_vel[:] = pilot.command(t, r.data.qpos[0], r.data.qpos[1], yaw_of(r.data.qpos[3:7]))
        policy_step(r)
        for _ in range(r.cfg.sim.decimation):
            physics_substep(r)
            if r.data.time >= next_frame:
                renderer.update_scene(r.data, camera=cam)
                writer.append_data(renderer.render())
                next_frame += frame_dt
        r.gait_phase_time += r.cfg.sim.dt * r.cfg.sim.decimation
        yaw = yaw_of(r.data.qpos[3:7])
        ground, rel = tr.update(r.data.time, r.data.qpos[0:3], r.get_gravity_orientation(r.data.qpos[3:7])[2], yaw)
        if k % 50 == 0:
            print(f"t={r.data.time:5.1f}s cmd=[{r.command_vel[0]:.2f},{r.command_vel[1]:+.2f},{r.command_vel[2]:+.2f}] "
                  f"x={r.data.qpos[0]:5.2f} y={r.data.qpos[1]:+.2f} yaw={math.degrees(yaw):+4.0f} z={r.data.qpos[2]:.2f} tread={ground:.2f} rel={rel:.2f}")
        if tr.fell and r.data.time > tr.fall_t + 1.5:
            break
    writer.close()
    print(f"[record] wrote {args.out}")
    return summarize(tr, r, stairs)


def run_viewer(r, args, stairs):
    import time

    import mujoco.viewer

    pilot = Autopilot(stairs, args.vx, args.settle, stairs.x_end + 0.6)
    with mujoco.viewer.launch_passive(r.model, r.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = args.cam_distance, args.cam_azimuth, args.cam_elevation
        while viewer.is_running():
            reset(r)
            tr = Tracker(stairs)
            while viewer.is_running() and r.data.time < args.duration:
                t0 = time.time()
                r.command_vel[:] = pilot.command(r.data.time, r.data.qpos[0], r.data.qpos[1], yaw_of(r.data.qpos[3:7]))
                policy_step(r)
                for _ in range(r.cfg.sim.decimation):
                    physics_substep(r)
                r.gait_phase_time += r.cfg.sim.dt * r.cfg.sim.decimation
                tr.update(r.data.time, r.data.qpos[0:3], r.get_gravity_orientation(r.data.qpos[3:7])[2], yaw_of(r.data.qpos[3:7]))
                viewer.sync()
                if tr.fell and r.data.time > tr.fall_t + 1.5:
                    break
                time.sleep(max(0.0, r.cfg.sim.dt * r.cfg.sim.decimation / args.speed - (time.time() - t0)))
            summarize(tr, r, stairs)
            if not args.loop:
                break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["record", "viewer"], default="record")
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--vx", type=float, default=0.9, help="forward speed command [m/s] (trained range -0.6..1.0)")
    p.add_argument("--settle", type=float, default=1.0, help="seconds standing still before walking")
    p.add_argument("--duration", type=float, default=18.0)
    p.add_argument("--out", default="/output/g1_stairs.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cam-distance", type=float, default=3.2)
    p.add_argument("--cam-azimuth", type=float, default=110.0)
    p.add_argument("--cam-elevation", type=float, default=-12.0)
    p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed vs real time")
    p.add_argument("--loop", action="store_true", help="viewer: restart the run after each episode")
    args = p.parse_args()

    up = load_upstream()
    cfg = up.G1DwaqSim2SimCfg()
    cfg.sim.sim_duration = args.duration
    r = up.G1DwaqMujocoRunner(cfg=cfg, checkpoint_path=args.checkpoint, model_path=args.scene)
    stairs = StairProfile(r.model)
    if args.mode == "record":
        sys.exit(0 if run_record(r, args, stairs) else 1)
    run_viewer(r, args, stairs)


if __name__ == "__main__":
    main()
