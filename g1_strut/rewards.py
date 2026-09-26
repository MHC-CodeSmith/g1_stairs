import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from g1_strut import dance, groove

_TABLES: dict[str, torch.Tensor] = {}


def _table(device) -> torch.Tensor:
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(dance.table(), dtype=torch.float32, device=device)
    return _TABLES[key]


def strut_dance_reference(env) -> torch.Tensor:
    """(num_envs, len(dance.JOINTS)) reference at each env's left-leg gait phase."""
    tab = _table(env.device)
    idx = (env.phase * tab.shape[0]).long() % tab.shape[0]
    return tab[idx]


def strut_dance_tracking(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """exp(-mean squared upper-body joint error / std^2) against the phase-locked strut reference.

    asset_cfg.joint_names must be dance.JOINTS with preserve_order=True so joint_ids line up with the table columns.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - strut_dance_reference(env)
    return torch.exp(-torch.mean(torch.square(err), dim=1) / std**2)


def groove_dance_reference(env) -> torch.Tensor:
    """(num_envs, len(dance.JOINTS)) groove reference at each env's dance phase (see tasks.G1DanceEnv)."""
    key = f"groove:{env.device}"
    if key not in _TABLES:
        _TABLES[key] = torch.tensor(groove.table(), dtype=torch.float32, device=env.device)
    tab = _TABLES[key]
    return tab[(env.dance_phase * tab.shape[0]).long() % tab.shape[0]]


def groove_dance_tracking(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Same kernel as strut_dance_tracking, against the 8-beat groove reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - groove_dance_reference(env)
    return torch.exp(-torch.mean(torch.square(err), dim=1) / std**2)
