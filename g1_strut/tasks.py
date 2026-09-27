"""Tasks `g1_dwaq_strut` / `g1_dwaq_groove`: the G1DWAQ stair task + phase-locked upper-body dance tracking."""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg, G1DwaqRewardCfg
from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.utils.task_registry import task_registry

from g1_strut import clip_dance, dance, groove, rewards

UPPER_BODY = SceneEntityCfg("robot", joint_names=dance.JOINTS, preserve_order=True)
STAIR_SHARE = 0.7  # fraction of terrain tiles that are stairs (upstream: 0.4)


@configclass
class G1StrutRewardCfg(G1DwaqRewardCfg):
    strut_dance = RewTerm(func=rewards.strut_dance_tracking, weight=2.0, params={"std": 0.3, "asset_cfg": UPPER_BODY})

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        # Upstream pulls waist + arms toward the default pose; that would fight the dance. Keep it only on the
        # joint the dance leaves free (waist pitch, used for leaning into the stairs).
        self.joint_deviation_arms.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint"])}


@configclass
class G1StrutEnvCfg(G1DwaqEnvCfg):
    reward = G1StrutRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        subs = self.scene.terrain_generator.sub_terrains
        stairs = [k for k in subs if k.startswith("stairs")]
        other = sum(subs[k].proportion for k in subs if k not in stairs)
        for k in subs:
            subs[k].proportion = STAIR_SHARE / len(stairs) if k in stairs else subs[k].proportion * (1 - STAIR_SHARE) / other


@configclass
class G1StrutAgentCfg(G1DwaqAgentCfg):
    experiment_name: str = "g1_dwaq_strut"
    wandb_project: str = "g1_dwaq_strut"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100


task_registry.register("g1_dwaq_strut", G1DwaqEnv, G1StrutEnvCfg(), G1StrutAgentCfg())


class G1DanceEnv(G1DwaqEnv):
    """G1DwaqEnv + a dance clock: sin/cos of the phrase phase appended to the actor obs (and after the actor block
    in the critic obs), so the policy knows which beat of the 8-beat phrase comes next."""

    dance_period = groove.PERIOD

    @property
    def dance_phase(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt  # same clock as the gait phase
        return (t % self.dance_period) / self.dance_period

    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        ph = 2 * torch.pi * self.dance_phase
        clock = torch.stack([torch.sin(ph), torch.cos(ph)], dim=-1)
        n = actor.shape[1]
        return torch.cat([actor, clock], dim=-1), torch.cat([critic[:, :n], clock, critic[:, n:]], dim=-1)


@configclass
class G1GrooveRewardCfg(G1StrutRewardCfg):
    strut_dance = None
    # Two kernels: with one tight kernel (std 0.3) the 8 widely spread poses left a ~0.4 rad error where the reward is
    # flat, and the arms settled on the average pose. The coarse one keeps a gradient far from the target, the fine
    # one pays for hitting the pose.
    groove_coarse = RewTerm(func=rewards.groove_dance_tracking, weight=1.5, params={"std": 0.6, "asset_cfg": UPPER_BODY})
    groove_fine = RewTerm(func=rewards.groove_dance_tracking, weight=2.0, params={"std": 0.25, "asset_cfg": UPPER_BODY})


@configclass
class G1GrooveEnvCfg(G1StrutEnvCfg):
    reward = G1GrooveRewardCfg()


@configclass
class G1GrooveAgentCfg(G1StrutAgentCfg):
    experiment_name: str = "g1_dwaq_groove"
    wandb_project: str = "g1_dwaq_groove"


task_registry.register("g1_dwaq_groove", G1DanceEnv, G1GrooveEnvCfg(), G1GrooveAgentCfg())


class G1ClipDanceEnv(G1DanceEnv):
    """Upper body (arms + waist yaw/roll, dance.JOINTS) plays the retargeted clip open-loop; the policy's outputs for
    those joints are replaced by the reference before the PD targets are set. The policy still controls legs and
    waist pitch and learns to climb while the arms dance. The clip clock is in the observation (G1DanceEnv).

    The reference is eased in from the default pose over BLEND_IN seconds after each reset, and read LEAD seconds
    ahead so the PD-tracked arms land on the beat.
    """

    BLEND_IN = 1.0
    LEAD = 0.04

    def __init__(self, cfg, headless):
        q, dt, period, joints = clip_dance.load()
        assert joints == dance.JOINTS, joints
        self.dance_period = period
        self._clip_dt = dt
        super().__init__(cfg, headless)
        self._clip_q = torch.tensor(q, device=self.device)
        self._upper_ids, _ = self.robot.find_joints(dance.JOINTS, preserve_order=True)

    def dance_reference(self, lead: float = 0.0) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.step_dt
        idx = ((t + lead) / self._clip_dt).long() % self._clip_q.shape[0]
        ref = self._clip_q[idx]
        default = self.robot.data.default_joint_pos[:, self._upper_ids]
        alpha = torch.clamp(t / self.BLEND_IN, 0.0, 1.0).unsqueeze(1)
        return default + alpha * (ref - default)

    def check_reset(self):
        """Fall = tilted past ~57 deg or pelvis within 0.35 m of the ground under it. Upstream resets on any torso
        contact, which here also fires when a dancing arm brushes the chest (self-collision), with the robot upright."""
        tilted = self.robot.data.projected_gravity_b[:, 2] > -0.55
        ground = torch.nan_to_num(self.height_scanner.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0).mean(1)
        low = self.robot.data.root_pos_w[:, 2] - ground < 0.35
        time_out = self.episode_length_buf >= self.max_episode_length
        return tilted | low | time_out, time_out

    def step(self, actions: torch.Tensor):
        actions = actions.clone()
        target = self.dance_reference(self.LEAD)
        actions[:, self._upper_ids] = (target - self.robot.data.default_joint_pos[:, self._upper_ids]) / self.action_scale
        return super().step(actions)


@configclass
class G1ClipRewardCfg(G1GrooveRewardCfg):
    groove_coarse = None  # arms are driven by the clip, nothing to learn there
    groove_fine = None

    def __post_init__(self):
        super().__post_init__()
        # arm/torso self-contacts come from the choreography the policy doesn't control; penalize leg bumps only
        self.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_sensor", body_names=["pelvis", ".*_hip_.*", ".*_knee_.*"])


@configclass
class G1ClipEnvCfg(G1StrutEnvCfg):
    reward = G1ClipRewardCfg()


@configclass
class G1ClipAgentCfg(G1StrutAgentCfg):
    experiment_name: str = "g1_dwaq_bully"
    wandb_project: str = "g1_dwaq_bully"


task_registry.register("g1_dwaq_bully", G1ClipDanceEnv, G1ClipEnvCfg(), G1ClipAgentCfg())
