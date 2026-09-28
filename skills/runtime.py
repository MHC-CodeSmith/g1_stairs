"""Run skills (skills/adapters.py) on the G1 in Isaac Lab: one control loop, per-joint PD gains per skill.

Import after Isaac Sim's AppLauncher. The env is a TienKung-Lab G1 task used for its scene (robot asset, terrain,
height scanner, sensors); its reward/obs/reset machinery is bypassed and the loop below writes joint targets and
steps physics directly.
"""
from __future__ import annotations

import torch
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils import task_registry

import g1_rl.body  # noqa: F401  (registers g1_body)
from skills.adapters import RobotState

TILE, BORDER, PLATFORM = 8.0, 0.5, 1.5


def stairs_terrain(step_height=0.15, step_width=0.31):
    return TerrainGeneratorCfg(
        curriculum=False, size=(TILE, TILE), border_width=10.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
        sub_terrains={"stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(step_height, step_height), step_width=step_width,
            platform_width=PLATFORM, border_width=BORDER)})


def make_env(task="g1_dwaq", terrain="flat", num_envs=1, episode_s=60.0):
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.max_episode_length_s = episode_s
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {}
    env_cfg.commands.debug_vis = False
    if terrain == "flat":
        env_cfg.scene.terrain_type, env_cfg.scene.terrain_generator = "plane", None
    elif terrain == "stairs":
        env_cfg.scene.terrain_type, env_cfg.scene.terrain_generator = "generator", stairs_terrain()
    env = task_registry.get_task_class(task)(env_cfg, True)
    if terrain == "stairs":  # spawn on the flat border facing +x instead of on top of the pyramid
        t = env.scene.terrain
        for o in (t.terrain_origins.view(-1, 3), t.env_origins):
            o[:, 0] -= TILE / 2 - BORDER / 2 - 0.1
            o[:, 2] = 0.0
    env.reset(torch.arange(env.num_envs, device=env.device))
    return env, env_cfg, agent_cfg


class SkillRunner:
    """Per control step: build RobotState, ask each active skill for its joints' targets, write targets and the
    gains of the skill that owns each joint, step physics `decimation` times."""

    def __init__(self, env):
        self.env, self.robot = env, env.scene["robot"]
        self.names = list(self.robot.joint_names)
        self.kp0 = self.robot.data.joint_stiffness.clone()
        self.kd0 = self.robot.data.joint_damping.clone()
        self.t = torch.zeros(env.num_envs, device=env.device)
        self.target = self.robot.data.default_joint_pos.clone()
        self._gains_key = None

    def ids(self, joints):
        lut = {n: i for i, n in enumerate(self.names)}
        return torch.tensor([lut[j] for j in joints], device=self.env.device)

    def ground_height(self):
        sc = self.env.height_scanner
        hits = torch.nan_to_num(sc.data.ray_hits_w, nan=-1e3)
        d = torch.norm(hits[..., :2] - sc.data.pos_w[:, None, :2], dim=-1)
        near = torch.topk(-d, k=4, dim=1).indices  # the 4 rays closest to below the torso
        return torch.gather(hits[..., 2], 1, near).max(dim=1).values

    def state(self) -> RobotState:
        d = self.robot.data
        return RobotState(self.names, d.joint_pos.clone(), d.joint_vel.clone(), d.root_ang_vel_b.clone(),
                          d.projected_gravity_b.clone(), d.root_pos_w[:, 2] - self.ground_height(), self.t.clone())

    def set_gains(self, assignments):
        """assignments: list of (joint ids, kp, kd); written only when the ownership/gains change."""
        key = tuple((tuple(i.tolist()), tuple(kp.tolist()), tuple(kd.tolist())) for i, kp, kd in assignments)
        if key == self._gains_key:
            return
        kp, kd = self.kp0.clone(), self.kd0.clone()
        for i, p, dmp in assignments:
            kp[:, i], kd[:, i] = p, dmp
        self.robot.write_joint_stiffness_to_sim(kp)
        self.robot.write_joint_damping_to_sim(kd)
        self._gains_key = key

    def step(self, targets_by_ids):
        """targets_by_ids: list of (joint ids, (N, k) absolute targets). Unlisted joints keep their last target."""
        for i, tgt in targets_by_ids:
            self.target[:, i] = tgt
        env = self.env
        for _ in range(env.cfg.sim.decimation):
            self.robot.set_joint_position_target(self.target)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        self.t += env.step_dt

    def fallen(self):
        return (self.robot.data.projected_gravity_b[:, 2] > -0.55) | (
            self.robot.data.root_pos_w[:, 2] - self.ground_height() < 0.35)
