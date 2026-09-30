"""Markdown tables for docs/ARENA_REPORT.md, generated from the result files (no hand-copied numbers).

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/report_tables.py > output/report_tables.md
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.stairs_bench import SWEEP, summary  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f(x, d=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


sc = json.load(open(os.path.join(REPO, "output/scorecard.json")))
sb = json.load(open(os.path.join(REPO, "output/stairs_bench.json")))
by = {(r["policy"], r["test"]): r for r in sc if "error" not in r}
S = summary(sb)

print("### Stairs ranking\n")
print("| # | policy | HumanoidBench return (max 1000) | climbed on HB course [m] | tallest step crossed | "
      "Safe100 staircase success |")
print("|---|---|---|---|---|---|")
order = sorted(S, key=lambda p: -S[p].get("hb_return", 0))
for i, p in enumerate(order, 1):
    s = S[p]
    rise = "none" if s["max_rise"] == 0 else f"{round(s['max_rise'] * 100)} cm"
    print(f"| {i} | {p} | {f(s.get('hb_return'), 0)}{' (fell)' if s.get('hb_fell') else ''} | "
          f"{f(s.get('hb_climbed'), 2)} | {rise} | "
          f"{f(100 * s['safe100_success'], 0)}% |")

print("\n### Step-height sweep (one run per height; ✓ = up, over and down still standing)\n")
print("| policy | " + " | ".join(f"{int(r * 100)} cm" for r in SWEEP) + " |")
print("|---|" + "---|" * len(SWEEP))
for p in sorted(S, key=lambda p: -S[p]["max_rise"]):
    print(f"| {p} | " + " | ".join("✓" if S[p]["sweep"].get(r) else "·" for r in SWEEP) + " |")

print("\n### Locomotion scorecard\n")
print("| policy | flat vx / wz err | no hands vx err | rough | arms waving vx err | push survived [N] | "
      "crouch err (lowest) |")
print("|---|---|---|---|---|---|---|")
pols = sorted({r["policy"] for r in sc if r.get("kind") == "loco"})


def cell(p, t, key="vx_err"):
    r = by.get((p, t))
    if r is None:
        e = [x for x in sc if x.get("policy") == p and x.get("test") == t]
        return ("needs hands" if e and "hand" in e[0].get("error", "") else "n/a")
    if r["fell"] and t != "push":
        return f"fell {r['fall_t']:.1f} s"
    return f(r.get(key))


for p in pols:
    fl = by.get((p, "flat"))
    flat = cell(p, "flat") if fl is None or fl["fell"] else f"{f(fl['vx_err'])} / {f(fl['wz_err'])}"
    c = by.get((p, "crouch"))
    crouch = "n/a" if c is None else (f"fell {c['fall_t']:.1f} s" if c["fell"] else
                                     f"{f(c.get('height_err'))} ({f(c.get('min_height'), 2)} m)")
    pu = by.get((p, "push"))
    print(f"| {p} | {flat} | {cell(p, 'no_hands')} | {cell(p, 'rough')} | {cell(p, 'arms')} | "
          f"{f(pu.get('max_push_N'), 0) if pu else '-'} | {crouch} |")

print("\n### Motion tracking\n")
print("| clip | tracker | joint err [rad] | root xy err [m] (final) | heading err [rad] (final) | fell |")
print("|---|---|---|---|---|---|")
tr = [r for r in sc if r.get("kind") == "track" and "error" not in r]
for c in sorted({r["test"] for r in tr}):
    for r in sorted([r for r in tr if r["test"] == c], key=lambda r: r["joint_err"]):
        who = r["policy"] + (" (its own clip)" if r["policy"] == c else "")
        print(f"| {c} ({r['seconds']} s) | {who} | {f(r['joint_err'])} | {f(r['root_xy_err'], 2)} "
              f"({f(r['root_xy_err_final'], 2)}) | {f(r['yaw_err'], 2)} ({f(r['yaw_err_final'], 2)}) | "
              f"{'fell %.1f s' % r['fall_t'] if r['fell'] else ''} |")
