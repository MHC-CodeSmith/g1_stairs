"""Charts for docs/ARENA_REPORT.md from the arena results.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/report_figures.py

Inputs: output/scorecard.json (arena.scorecard), output/stairs_bench.json (arena.stairs_bench),
output/g1_body_training.json (tools/export_training_curves.py), output/g1_body_checkpoints.json.
Outputs: docs/figures/*.png.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.stairs_bench import SWEEP, summary  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs/figures")
OURS = "g1_body"
C_OURS, C_OK, C_FELL, C_GRID = "#d1495b", "#2e86ab", "#bbbbbb", "#eeeeee"


def load(name):
    p = os.path.join(REPO, "output", name)
    return json.load(open(p)) if os.path.exists(p) else []


def style(ax, title, xlabel=None):
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=C_GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("wrote", name)


def barh(names, vals, colors, title, xlabel, name, labels=None, xmax=None):
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(names) + 1.1))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for i, lab in enumerate(labels or []):
        ax.text(vals[i], i, "  " + lab, va="center", fontsize=8)
    if xmax:
        ax.set_xlim(0, xmax)
    style(ax, title, xlabel)
    save(fig, name)


def stairs_figs():
    res = load("stairs_bench.json")
    if not res:
        return
    S = summary(res)
    order = sorted(S, key=lambda p: -S[p].get("hb_return", 0))
    barh(order, [S[p].get("hb_return", 0) for p in order],
         [C_OURS if p == OURS else (C_OK if S[p].get("hb_climbed", 0) > 0.3 else C_FELL) for p in order],
         "HumanoidBench stair course: return over 20 s (max 1000)", "return",
         "stairs_humanoidbench.png",
         [f"{S[p].get('hb_climbed', 0):.2f} m climbed" + (", fell" if S[p].get("hb_fell") else "") for p in order])
    # sweep heatmap
    order = sorted(S, key=lambda p: (-S[p]["max_rise"], -S[p].get("hb_return", 0)))
    M = np.array([[1.0 if S[p]["sweep"].get(r) else 0.0 for r in SWEEP] for p in order])
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(order) + 1.3))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#f3d9dc", "#6aaa64"]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(SWEEP)), [f"{int(r * 100)}" for r in SWEEP])
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("step rise [cm] (0.30 m treads, 8 up / 8 down)")
    ax.set_title("Step-height sweep: crossed (green) or not", fontsize=11, loc="left", fontweight="bold")
    save(fig, "stairs_sweep.png")
    order = sorted(S, key=lambda p: -S[p]["safe100_success"])
    barh(order, [100 * S[p]["safe100_success"] for p in order],
         [C_OURS if p == OURS else C_OK for p in order],
         "Safe100 staircase (6 x 13 cm): runs ending on the top platform", "% of 16 runs",
         "stairs_safe100.png", [f"{100 * S[p]['safe100_success']:.0f}%" for p in order], xmax=110)


def loco_figs():
    res = [r for r in load("scorecard.json") if r.get("kind") == "loco" and "error" not in r]
    if not res:
        return
    by = {(r["policy"], r["test"]): r for r in res}
    pols = sorted({r["policy"] for r in res})

    def err(p, t):
        r = by.get((p, t))
        return None if r is None else (np.nan if r["fell"] else r["vx_err"])
    order = sorted(pols, key=lambda p: (np.isnan(err(p, "flat")), err(p, "flat")))
    v = [err(p, "flat") for p in order]
    barh(order, [0.3 if np.isnan(x) else x for x in v],
         [C_FELL if np.isnan(x) else (C_OURS if p == OURS else C_OK) for p, x in zip(order, v)],
         "Flat walk: mean |vx error| (grey = fell)", "m/s", "loco_flat.png",
         ["fell" if np.isnan(x) else f"{x:.3f}" for x in v])
    tests = ["flat", "no_hands", "rough", "arms", "stairs", "crouch"]
    M = np.full((len(pols), len(tests)), np.nan)
    for i, p in enumerate(pols):
        for j, t in enumerate(tests):
            r = by.get((p, t))
            if r is None:
                continue
            ok = not r["fell"] if t != "stairs" else bool(r.get("crossed"))
            M[i, j] = 1.0 if ok else 0.0
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(pols) + 1.3))
    cm = matplotlib.colors.ListedColormap(["#f3d9dc", "#6aaa64"])
    cm.set_bad("#f4f4f4")
    ax.imshow(np.ma.masked_invalid(M), cmap=cm, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(tests)), ["flat", "no hands", "rough", "arms\nwaving", "stairs\n(crossed)", "crouch"])
    ax.set_yticks(range(len(pols)), pols)
    ax.set_title("Locomotion tests: passed (green), failed (pink), n/a (grey)", fontsize=11, loc="left",
                 fontweight="bold")
    save(fig, "loco_matrix.png")
    order = sorted(pols, key=lambda p: -by.get((p, "push"), {}).get("max_push_N", 0))
    vals = [by.get((p, "push"), {}).get("max_push_N", 0) for p in order]
    barh(order, vals, [C_OURS if p == OURS else C_OK for p in order],
         "Largest lateral push survived (0.1 s on the pelvis, while walking)", "N", "loco_push.png",
         [f"{v:.0f}" for v in vals], xmax=1150)


def track_figs():
    res = [r for r in load("scorecard.json") if r.get("kind") == "track" and "error" not in r]
    if not res:
        return
    clips = sorted({r["test"] for r in res})
    trackers = ["sonic_tracking", "gmt", "twist", "own"]
    colors = {"sonic_tracking": "#2e86ab", "gmt": "#f18f01", "twist": "#6a4c93", "own": "#6aaa64"}
    by = {(r["policy"], r["test"]): r for r in res}
    for c in clips:                          # unitree_rl_lab's dance policies track only their own clip
        if (c, c) in by:
            by[("own", c)] = by[(c, c)]
    for key, title, fname, unit in (("joint_err", "Motion tracking: mean joint error", "track_joint.png", "rad"),
                                    ("root_xy_err", "Motion tracking: mean root position error", "track_root.png", "m"),
                                    ("yaw_err", "Motion tracking: mean heading error", "track_yaw.png", "rad")):
        fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(clips) + 1.2))
        y = np.arange(len(clips))
        for k, t in enumerate(trackers):
            vals = [by.get((t, c), {}).get(key, np.nan) for c in clips]
            fell = [by.get((t, c), {}).get("fell", False) for c in clips]
            ax.barh(y + (k - 1.5) * 0.2, vals, height=0.2, color=colors[t],
                    label="clip's own policy (unitree_rl_lab)" if t == "own" else t,
                    hatch=None, edgecolor=["black" if f else colors[t] for f in fell])
        ax.set_yticks(y, clips)
        ax.invert_yaxis()
        if key == "root_xy_err":
            ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax, title + " (black edge = fell)", unit)
        save(fig, fname)


def training_figs():
    tr = load("g1_body_training.json")
    if tr:
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
        for ax, (k, title) in zip(axs, (("Train/mean_reward", "mean reward"),
                                        ("Body/height_err_crouch", "crouch height error [m]"),
                                        ("Loss/distill_coef", "imitation weight (walk/stairs teacher)"))):
            if k in tr:
                it, v = zip(*tr[k])
                ax.plot(it, v, color=C_OURS)
            ax.axvline(700, color="#555", ls="--", lw=1)
            style(ax, title, "iteration")
            ax.grid(axis="y", color=C_GRID)
        save(fig, "g1_body_training.png")
    ck = load("g1_body_checkpoints.json")
    if ck:
        it = [c["it"] for c in ck]
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        ax.plot(it, [c["flat_vx_err"] for c in ck], "o-", color=C_OK, label="flat |vx err| [m/s]")
        ax.plot(it, [c["crouch_err"] * 10 for c in ck], "s-", color="#6a4c93", label="crouch height err x10 [m]")
        for c in ck:
            ax.text(c["it"], 0.03, "stairs ✓" if c["stairs"] else "stairs ✗", ha="center", fontsize=8,
                    color="#2a7" if c["stairs"] else C_OURS, transform=ax.get_xaxis_transform())
        ax.axvline(700, color="#555", ls="--", lw=1)
        ax.legend(frameon=False, fontsize=8)
        style(ax, "g1_body checkpoints scored in the arena (released: 700)", "iteration")
        save(fig, "g1_body_checkpoints.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stairs_figs()
    loco_figs()
    track_figs()
    training_figs()
