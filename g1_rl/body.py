"""Task `g1_body`: one lower-body controller distilled from teachers of different origins, under any upper body.

Student: DWAQ actor-critic (blind, 5-frame history) driving legs + waist pitch. The upper body (arms, waist
yaw/roll) is driven by a motion library that stands in for an upper-body skill (default pose, fixed holds, random
reaches; later teleop or a manipulation policy). Command: [vx, vy, yaw rate] + pelvis height.

Teachers, chosen per env from what the (blind) student can observe, and only where each was validated
(docs/SCORECARD.md):
  - height command below nominal (standing on a flat tile, command forced to zero): NVIDIA GR00T-WholeBodyControl
    (Balance), legs + waist pitch, its targets converted to our PD gains (gain_equivalent_target). It crouches to
    0.50 m in the arena; AGILE, the previous crouch teacher, stops at 0.62 m. Its imitation weight is not annealed.
  - everything else (walking, stairs, rough, at nominal height): the G1DWAQ_Lab stair policy, legs + waist pitch,
    with the annealed imitation weight.
A crouch context is drawn for half of the envs on flat tiles (~15% of all envs), so the crouch teacher gets enough
samples. Training: PPO on the task reward + distillation toward the active teacher (g1_rl/distill.py).
"""
from __future__ import annotations

import copy
import math

import isaaclab.terrains as terrain_gen
import numpy as np
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg, G1DwaqRewardCfg
from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.utils.task_registry import task_registry

from skills import registry
from skills.adapters import RobotState, gain_equivalent_target

CROUCH_RANGE = (0.50, 0.70)         # GR00T WBC tracks 0.50 m with 1.2 cm error (arena crouch test)
CROUCH_PROB = 0.5                   # share of flat-tile envs given a crouch context
NOMINAL_HEIGHT = 0.72
FLAT_SHARE, STAIRS_SHARE = 0.30, 0.40

UPPER_JOINTS = [  # driven by the motion library, not the student
    "waist_yaw_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# upper-body sources: name -> probability
SOURCES = {"default": 0.4, "hold": 0.3, "random": 0.3}
HOLDS = [  # (name, {joint: angle}) arm poses for carrying / reaching / idle
    ("tienkung_default", {"left_shoulder_pitch_joint": .35, "right_shoulder_pitch_joint": .35,
                          "left_shoulder_roll_joint": .18, "right_shoulder_roll_joint": -.18,
                          "left_elbow_joint": .87, "right_elbow_joint": .87}),
    ("carry_front", {"left_shoulder_pitch_joint": -.9, "right_shoulder_pitch_joint": -.9,
                     "left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15,
                     "left_elbow_joint": .3, "right_elbow_joint": .3}),
    ("arms_up", {"left_shoulder_pitch_joint": -2.6, "right_shoulder_pitch_joint": -2.6,
                 "left_elbow_joint": 1.3, "right_elbow_joint": 1.3}),
    ("reach_right", {"right_shoulder_pitch_joint": -1.3, "right_shoulder_roll_joint": -.6, "right_elbow_joint": 1.3,
                     "waist_yaw_joint": -.3, "left_shoulder_pitch_joint": .3, "left_elbow_joint": .9}),
]


def height_tracking(env, std: float) -> torch.Tensor:
    """exp(-(pelvis height - command)^2 / std^2), only for envs with an active height command (crouch context)."""
    err = env.base_height() - env.height_cmd
    return torch.exp(-err.pow(2) / std**2) * env.crouch_active.float()


@configclass
class G1BodyRewardCfg(G1DwaqRewardCfg):
    height = RewTerm(func=height_tracking, weight=1.0, params={"std": 0.05})

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # The upper body is driven externally: keep the default-pose pull only on waist pitch, and penalize only leg
        # contacts (moving arms may touch the torso).
        self.joint_deviation_arms.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])}
        self.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_sensor", body_names=["pelvis", ".*_hip_.*", ".*_knee_.*"])


@configclass
class G1BodyEnvCfg(G1DwaqEnvCfg):
    reward = G1BodyRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # the upstream config assigns a module-level TerrainGeneratorCfg; copy before editing
        self.scene.terrain_generator = copy.deepcopy(self.scene.terrain_generator)
        subs = self.scene.terrain_generator.sub_terrains
        stairs = [k for k in subs if k.startswith("stairs")]
        rest = [k for k in subs if k not in stairs]
        other = sum(subs[k].proportion for k in rest)
        for k in stairs:
            subs[k].proportion = STAIRS_SHARE / len(stairs)
        for k in rest:
            subs[k].proportion *= (1 - STAIRS_SHARE - FLAT_SHARE) / other
        subs["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=FLAT_SHARE)


@configclass
class G1BodyAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "g1_body"
    wandb_project: str = "g1_body"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.save_interval = 100
        self.algorithm.class_name = "DistillDWAQPPO"


class MotionLibrary:
    """Per-env upper-body targets (N, 16) in UPPER_JOINTS order, eased in from the default pose after a reset."""

    def __init__(self, env, blend_in=1.0):
        self.env, self.device, self.n, self.blend_in = env, env.device, env.num_envs, blend_in
        self.names = list(SOURCES)
        self.probs = torch.tensor([SOURCES[k] for k in self.names], device=self.device)
        ids = env.robot.find_joints(UPPER_JOINTS, preserve_order=True)[0]
        self.default = env.robot.data.default_joint_pos[0, ids].clone()
        lim = env.robot.data.soft_joint_pos_limits[0, ids]
        self.lo, self.hi = 0.6 * lim[:, 0], 0.6 * lim[:, 1]
        self.lo[:2], self.hi[:2] = torch.tensor([-.4, -.2], device=self.device), torch.tensor([.4, .2], device=self.device)
        self.holds = torch.stack([self.default.clone() for _ in HOLDS])
        for i, (_, pose) in enumerate(HOLDS):
            for j, v in pose.items():
                self.holds[i, UPPER_JOINTS.index(j)] = v
        self.source = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.hold_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.seg_from = self.default.repeat(self.n, 1)
        self.seg_to = self.default.repeat(self.n, 1)
        self.seg_t0 = torch.zeros(self.n, device=self.device)
        self.seg_len = torch.ones(self.n, device=self.device)
        self.resample(torch.arange(self.n, device=self.device))

    def resample(self, ids):
        k = len(ids)
        if k == 0:
            return
        self.source[ids] = torch.multinomial(self.probs, k, replacement=True)
        self.hold_id[ids] = torch.randint(len(HOLDS), (k,), device=self.device)
        self.seg_from[ids] = self.default
        self.seg_to[ids] = self.default
        self.seg_t0[ids] = 0.0
        self.seg_len[ids] = 0.5

    def targets(self, t):
        out = self.default.repeat(self.n, 1)
        for si, name in enumerate(self.names):
            m = self.source == si
            if not m.any():
                continue
            if name == "default":
                continue
            if name == "hold":
                out[m] = self.holds[self.hold_id[m]]
            else:  # random smooth reaches: cosine segments to uniform targets, 0.5-2 s each
                done = m & (t - self.seg_t0 >= self.seg_len)
                if done.any():
                    d = done.nonzero().flatten()
                    self.seg_from[d] = self.seg_to[d]
                    self.seg_to[d] = self.lo + (self.hi - self.lo) * torch.rand(len(d), len(self.default), device=self.device)
                    self.seg_t0[d] = t[d]
                    self.seg_len[d] = 0.5 + 1.5 * torch.rand(len(d), device=self.device)
                s = ((t[m] - self.seg_t0[m]) / self.seg_len[m]).clamp(0, 1).unsqueeze(1)
                s = 0.5 - 0.5 * torch.cos(math.pi * s)
                out[m] = self.seg_from[m] + s * (self.seg_to[m] - self.seg_from[m])
        alpha = (t / self.blend_in).clamp(0, 1).unsqueeze(1)
        return self.default + alpha * (out - self.default)


class G1BodyEnv(G1DwaqEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        dev = self.device
        self.height_cmd = torch.full((self.num_envs,), NOMINAL_HEIGHT, device=dev)
        self.height_goal = self.height_cmd.clone()
        self.crouch_ctx = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)     # drawn: crouch now
        self.crouch_active = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)  # drawn, or still coming back up
        self.upper_ids = torch.tensor(self.robot.find_joints(UPPER_JOINTS, preserve_order=True)[0], device=dev)
        self.motion = MotionLibrary(self)
        names = list(self.robot.joint_names)
        self.joint_names = names
        # which terrain columns are flat (Isaac Lab's curriculum column -> sub-terrain assignment)
        gen = self.cfg.scene.terrain_generator
        props = np.array([c.proportion for c in gen.sub_terrains.values()])
        props /= props.sum()
        keys = list(gen.sub_terrains)
        col_kind = [keys[int(np.min(np.where(i / gen.num_cols + 0.001 < np.cumsum(props))[0]))] for i in range(gen.num_cols)]
        self.flat_cols = torch.tensor([k == "flat" for k in col_kind], device=dev)
        # teachers
        self.t_dwaq = registry.load("dwaq_upstream", device=dev)
        self.t_crouch = registry.load("gr00t_wbc", device=dev)
        self.crouch_ids = torch.tensor([names.index(j) for j in self.t_crouch.joints], device=dev)
        self.lower_ids = torch.tensor([names.index(j) for j in names if j not in UPPER_JOINTS], device=dev)  # legs + waist pitch
        self.kp = self.robot.data.joint_stiffness[0].clone()
        self.kd = self.robot.data.joint_damping[0].clone()
        self.teacher_actions = torch.zeros(self.num_envs, self.num_actions, device=dev)
        self.teacher_weights = torch.zeros(self.num_envs, 2 * self.num_actions, device=dev)   # [annealed | fixed]
        self.w_dwaq = torch.zeros(self.num_actions, device=dev)
        self.w_dwaq[self.lower_ids] = 1.0
        self.w_crouch = torch.zeros(self.num_actions, device=dev)
        self.w_crouch[[names.index(j) for j in self.t_crouch.joints if j not in UPPER_JOINTS]] = 1.0  # legs + waist pitch
        self.w_crouch[names.index("waist_pitch_joint")] = 0.5
        self._crouch_last = torch.zeros(self.num_envs, 15, device=dev)
        self._update_context(torch.arange(self.num_envs, device=dev))

    def obs_map_from(self, old_obs: int):
        """Warm start: the first 100 obs are the upstream DWAQ layout. From a g1_body checkpoint (101 obs) keep
        everything; from an upstream DWAQ checkpoint (100 obs) start the height command with zero weights."""
        assert old_obs >= 100
        if old_obs == 101:
            return list(range(101))
        return list(range(100)) + [None]

    # --- helpers -----------------------------------------------------------------------------------------------
    def base_height(self):
        sc = self.height_scanner
        hits = torch.nan_to_num(sc.data.ray_hits_w, nan=-1e3)
        d = torch.norm(hits[..., :2] - sc.data.pos_w[:, None, :2], dim=-1)
        near = torch.topk(-d, k=4, dim=1).indices
        ground = torch.gather(hits[..., 2], 1, near).max(dim=1).values
        return self.robot.data.root_pos_w[:, 2] - ground

    def robot_state(self):
        d = self.robot.data
        return RobotState(self.joint_names, d.joint_pos, d.joint_vel, d.root_ang_vel_b, d.projected_gravity_b,
                          self.base_height(), self.episode_length_buf.float() * self.step_dt)

    def _update_context(self, ids):
        """Crouch context: half of the envs on flat tiles, standing, height goal in the crouch teacher's range."""
        on_flat = self.flat_cols[self.scene.terrain.terrain_types[ids]]
        ctx = on_flat & (torch.rand(len(ids), device=self.device) < CROUCH_PROB)
        self.crouch_ctx[ids] = ctx
        goal = CROUCH_RANGE[0] + (CROUCH_RANGE[1] - CROUCH_RANGE[0]) * torch.rand(len(ids), device=self.device)
        self.height_goal[ids] = torch.where(ctx, goal, torch.full_like(goal, NOMINAL_HEIGHT))
        self._refresh_crouch()

    def _refresh_crouch(self):
        """Active while drawn or while the height command is still returning to nominal; those envs stand still
        (the velocity command generator zeroes standing envs), so the height command alone tells the student which
        teacher it is imitating."""
        self.crouch_active = self.crouch_ctx | (self.height_cmd < NOMINAL_HEIGHT - 0.005)
        cg = self.command_generator
        cg.is_standing_env[self.crouch_active] = True
        cg.vel_command_b[self.crouch_active] = 0.0

    # --- observations: upstream 100 + (height command - nominal) -----------------------------------------------
    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        h = getattr(self, "height_cmd", None)
        h = (h - NOMINAL_HEIGHT if h is not None else torch.zeros(actor.shape[0], device=actor.device)).unsqueeze(1)
        n = actor.shape[1]
        return torch.cat([actor, h], dim=-1), torch.cat([critic[:, :n], h, critic[:, n:]], dim=-1)

    def check_reset(self):
        """Falls only: arm-torso contact from the externally driven upper body is not a fall."""
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.55
        low = self.base_height() < 0.35
        time_out = self.episode_length_buf >= self.max_episode_length
        return tilted | low | time_out, time_out

    def reset(self, env_ids):
        super().reset(env_ids)
        if not hasattr(self, "motion"):
            return
        self.motion.resample(env_ids)
        self.t_dwaq.reset(env_ids)
        self.t_crouch.reset(env_ids)
        self._crouch_last[env_ids] = 0.0
        self.height_cmd[env_ids] = NOMINAL_HEIGHT
        self._update_context(env_ids)

    # --- teacher labels for the observation the student acts on next --------------------------------------------
    def _label(self):
        st = self.robot_state()
        cmd = self.command_generator.command
        last = self.action_buffer._circular_buffer.buffer[:, -1, :]
        tgt_d = self.t_dwaq.act(st, cmd, last_action=last)                              # absolute, our gains
        raw_d = (tgt_d - self.robot.data.default_joint_pos) / self.action_scale
        cmd4 = torch.cat([cmd, self.height_cmd.unsqueeze(1)], dim=1)
        tgt_c = self.t_crouch.act(st, cmd4, last_action=self._crouch_last)              # absolute, GR00T gains
        q, qd = st.q[:, self.crouch_ids], st.qd[:, self.crouch_ids]
        tgt_c = gain_equivalent_target(tgt_c, q, qd, self.t_crouch.kp, self.t_crouch.kd,
                                       self.kp[self.crouch_ids], self.kd[self.crouch_ids])
        raw_c = raw_d.clone()
        raw_c[:, self.crouch_ids] = (tgt_c - self.robot.data.default_joint_pos[:, self.crouch_ids]) / self.action_scale
        ctx = self.crouch_active.unsqueeze(1)
        self.teacher_actions = torch.where(ctx, raw_c, raw_d)
        zero = torch.zeros_like(self.w_dwaq)
        self.teacher_weights = torch.cat([torch.where(ctx, zero, self.w_dwaq),         # annealed: DWAQ
                                          torch.where(ctx, self.w_crouch, zero)], 1)    # fixed: crouch teacher

    def get_observations(self):
        out = super().get_observations()
        if hasattr(self, "motion"):
            self._label()
        return out

    def step(self, actions: torch.Tensor):
        actions = actions.clone()
        t = self.episode_length_buf.float() * self.step_dt
        up = self.motion.targets(t + 0.04)
        actions[:, self.upper_ids] = (up - self.robot.data.default_joint_pos[:, self.upper_ids]) / self.action_scale
        # height command: rate-limited toward the goal; new goals every ~5 s
        resample = (torch.rand(self.num_envs, device=self.device) < self.step_dt / 5.0).nonzero().flatten()
        self._update_context(resample)
        self.height_cmd += (self.height_goal - self.height_cmd).clamp(-0.1 * self.step_dt, 0.1 * self.step_dt)
        self._refresh_crouch()
        # crouch teacher's "last action": what was executed on its 15 joints, in its own action space and gains
        d = self.robot.data
        exe = d.default_joint_pos[:, self.crouch_ids] + self.action_scale * torch.clamp(actions[:, self.crouch_ids], -100, 100)
        exe_c = gain_equivalent_target(exe, d.joint_pos[:, self.crouch_ids], d.joint_vel[:, self.crouch_ids],
                                       self.kp[self.crouch_ids], self.kd[self.crouch_ids], self.t_crouch.kp, self.t_crouch.kd)
        self._crouch_last = self.t_crouch.encode(exe_c)
        out = super().step(actions)
        self._refresh_crouch()
        self._label()
        log = self.extras.setdefault("log", {})
        log["Body/crouch_frac"] = self.crouch_active.float().mean()
        if self.crouch_active.any():
            err = (self.base_height() - self.height_cmd)[self.crouch_active]
            log["Body/height_err_crouch"] = err.abs().mean()
            low = self.height_cmd[self.crouch_active] < 0.58
            if low.any():
                log["Body/height_err_below_0.58"] = err[low].abs().mean()
        return out


task_registry.register("g1_body", G1BodyEnv, G1BodyEnvCfg(), G1BodyAgentCfg())
