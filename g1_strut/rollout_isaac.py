"""Roll out a G1 DWAQ policy on a staircase in Isaac Sim 5.1 / Isaac Lab (headless physics) -> .npz, with metrics.

  python g1_strut/rollout_isaac.py --checkpoint logs/g1_dwaq_strut/<run>/model_XXXX.pt --out output/isaac_strut.npz

One pyramid-stairs tile (default 0.15 m rise / 0.31 m tread: 8 steps up, platform, 8 steps down). The robot spawns on
the flat border facing +x; an autopilot holds heading 0 and steers back to the centre line with vy. The .npz holds
every link's world pose per control step and the terrain mesh; g1_strut/render_isaac6.py replays it in Isaac Sim 6.0 for video (Isaac Sim 5.1's RTX renderer
crashes on this host's 595 driver; its physics is fine).
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="g1_dwaq_bully")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out", type=str, default="output/isaac_strut.npz")
parser.add_argument("--vx", type=float, default=0.9)
parser.add_argument("--upper", type=str, default="bully",
                    help="g1_body only: upper-body source from its motion library (bully, groove, strut, hold, random)")
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--step_height", type=float, default=0.15)
parser.add_argument("--step_width", type=float, default=0.31)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.terrains as terrain_gen  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402

import g1_strut.tasks  # noqa: E402,F401
import g1_strut.body  # noqa: E402,F401  (registers g1_body)
import rsl_rl.runners.dwaq_on_policy_runner as _runner_mod  # noqa: E402
from g1_strut.distill import DistillDWAQPPO  # noqa: E402

_runner_mod.DistillDWAQPPO = DistillDWAQPPO  # g1_body's agent config names it; the runner resolves it with eval()
from g1_strut import chown_to_host, dance, rewards  # noqa: E402

TILE = 8.0
BORDER = 0.5
PLATFORM = 1.5


def make_env():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.max_episode_length_s = args_cli.seconds + 5.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
    env_cfg.commands.rel_standing_envs = 0.0
    # Commands come from the autopilot in main() (as in run_stairs.py): heading held at 0, lateral drift
    # corrected with vy. The generator only resamples (never, here) and leaves vel_command_b alone otherwise.
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.heading = None
    env_cfg.commands.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.resampling_time_range = (1e6, 1e6)
    env_cfg.commands.debug_vis = False
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = TerrainGeneratorCfg(
        curriculum=False, size=(TILE, TILE), border_width=10.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
        sub_terrains={"stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(args_cli.step_height, args_cli.step_height),
            step_width=args_cli.step_width, platform_width=PLATFORM, border_width=BORDER)},
    )
    env = task_registry.get_task_class(args_cli.task)(env_cfg, True)
    return env, env_cfg, agent_cfg


def main():
    env, env_cfg, agent_cfg = make_env()
    terrain = env.scene.terrain
    top = float(terrain.terrain_origins[0, 0, 2])  # pyramid origin is the centre of the top platform
    # Move the spawn from the platform to the flat border, facing the stairs. Shift terrain_origins as well so
    # curriculum-driven origin updates on reset keep the new spawn.
    for t in (terrain.terrain_origins.view(-1, 3), terrain.env_origins):
        t[:, 0] -= TILE / 2 - BORDER / 2 - 0.1
        t[:, 2] = 0.0
    env.reset(torch.arange(env.num_envs, device=env.device))
    start_x = float(terrain.env_origins[0, 0])

    runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.alg.policy.load_state_dict(torch.load(args_cli.checkpoint, weights_only=False)["model_state_dict"])
    runner.eval_mode()
    policy = runner.alg.policy

    robot = env.scene["robot"]
    upper_ids, _ = robot.find_joints(dance.JOINTS, preserve_order=True)
    body_pos, body_quat = [], []
    obs, obs_hist = env.get_observations()
    max_z, max_x, fell_t, sq_err, n = 0.0, 0.0, None, 0.0, 0
    start_y = float(terrain.env_origins[0, 1])
    if hasattr(env, "motion"):  # g1_body: pin the upper body to one motion-library source, from its start
        m = env.motion
        m.source[:] = m.names.index(args_cli.upper)
        m.offset[:], m.speed[:] = 0.0, 1.0

        def ref_fn(e):
            return e.motion.targets(e.episode_length_buf.float() * e.step_dt)
        ref_fn.__name__ = f"{args_cli.upper}_motion"
    elif hasattr(env, "dance_reference"):  # clip played on the arms: error = how well the PD follows it
        def ref_fn(e):
            return e.dance_reference()
        ref_fn.__name__ = "clip_reference"
    else:
        ref_fn = rewards.groove_dance_reference if hasattr(env, "dance_phase") else rewards.strut_dance_reference
    for k in range(int(args_cli.seconds / env.step_dt)):
        w, qx, qy, qz = robot.data.root_quat_w[0].tolist()
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        y_rel = float(robot.data.root_pos_w[0, 1]) - start_y
        env.command_generator.vel_command_b[:] = torch.tensor(
            [args_cli.vx, max(-0.3, min(0.3, -1.0 * y_rel)), max(-1.5, min(1.5, -2.0 * yaw))], device=env.device)
        with torch.inference_mode():
            actions = policy.act_inference(obs, obs_hist.to(env.device))
            obs, _, dones, extras = env.step(actions)
            obs_hist = extras["observations"]["obs_hist"]
        pos = robot.data.root_pos_w[0].cpu().numpy()
        t = (k + 1) * env.step_dt
        if bool(dones[0]) and fell_t is None and t < args_cli.seconds - env.step_dt:
            fell_t = t
            cs = env.contact_sensor
            f = torch.norm(cs.data.net_forces_w_history[0], dim=-1).max(0).values
            touching = [n for n, v in zip(cs.body_names, f.tolist()) if v > 1.0 and "ankle" not in n]
            print(f"[reset] t={t:.2f}s pelvis z={pos[2]:.2f} bodies in contact: {touching}", flush=True)
        max_z, max_x = max(max_z, float(pos[2])), max(max_x, float(pos[0]) - start_x)
        if t > 1.0:
            err = robot.data.joint_pos[0, upper_ids] - ref_fn(env)[0]
            sq_err += float(torch.mean(err**2)); n += 1
        body_pos.append(robot.data.body_pos_w[0].cpu().numpy().copy())
        body_quat.append(robot.data.body_quat_w[0].cpu().numpy().copy())  # (w, x, y, z)
        if k % 50 == 0:
            print(f"t={t:5.2f}s x={pos[0] - start_x:5.2f} y={pos[1]:+.2f} z={pos[2]:.2f}", flush=True)
    # The importer does not keep the mesh; the tile is deterministic (fixed step height, no curriculum), so
    # regenerate it (the generator applies the same centering transform the imported mesh got).
    mesh = TerrainGenerator(cfg=env_cfg.scene.terrain_generator, device="cpu").terrain_mesh
    rms = float(np.sqrt(sq_err / max(n, 1)))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    np.savez_compressed(
        args_cli.out, dt=env.step_dt, body_names=np.array(robot.body_names), body_pos=np.array(body_pos),
        body_quat=np.array(body_quat), terrain_vertices=np.asarray(mesh.vertices, dtype=np.float32),
        terrain_faces=np.asarray(mesh.faces, dtype=np.int32), start_x=start_x, top=top, vx=args_cli.vx,
        reached_top=max_z > top + 0.6, fell=fell_t is not None, max_x=max_x, upper_body_rms=rms,
        checkpoint=args_cli.checkpoint)
    chown_to_host(args_cli.out)
    climb = TILE - 2 * BORDER  # distance from the first step to the last
    print("=" * 64)
    print(f" stairs: rise {args_cli.step_height:.2f} m, tread {args_cli.step_width:.2f} m, top at {top:.2f} m")
    print(f" peak pelvis height : {max_z:.2f} m (top + standing height ~ {top + 0.78:.2f} m)")
    print(f" reached top        : {max_z > top + 0.6}")
    print(f" distance covered   : {max_x:.2f} m (stairs end at ~{climb + 0.35:.2f} m)")
    print(f" fell / reset       : {fell_t is not None}" + (f" (t={fell_t:.2f}s)" if fell_t else ""))
    print(f" upper-body RMS err : {rms:.3f} rad vs {ref_fn.__name__.split('_')[0]} reference")
    print(f" rollout            : {args_cli.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang indefinitely in headless Isaac Sim 5.1; everything is already written.
    sys.stdout.flush()
    os._exit(0)
