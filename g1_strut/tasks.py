"""Task `g1_dwaq_strut`: the G1DWAQ stair task + phase-locked strut dance tracking for the upper body."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.envs.g1.g1_dwaq_config import G1DwaqAgentCfg, G1DwaqEnvCfg, G1DwaqRewardCfg
from legged_lab.envs.g1.g1_dwaq_env import G1DwaqEnv
from legged_lab.utils.task_registry import task_registry

from g1_strut import dance, rewards

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
