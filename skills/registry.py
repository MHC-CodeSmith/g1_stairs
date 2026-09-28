"""Load skills by name from skills/registry.yaml."""
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(REPO, "skills", "registry.yaml")) as f:
    SKILLS = yaml.safe_load(f)


def path(name):
    p = SKILLS[name]["path"]
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load(name, device="cpu"):
    from skills import adapters

    spec = SKILLS[name]
    kind = spec["kind"]
    if kind == "dwaq":
        return adapters.DwaqPolicy(path(name), device=device, name=name, extras=spec.get("extras"))
    if kind == "agile_velocity_height":
        return adapters.AgileVelocityHeight(path(name), device=device)
    if kind == "gr00t_wbc":
        d = path(name)
        return adapters.Gr00tWbc(os.path.join(d, "walk.pt"), os.path.join(d, "balance.pt"), device=device)
    raise ValueError(f"unknown skill kind {kind}")
