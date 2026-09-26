import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from g1_strut import dance

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
