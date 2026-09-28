"""Go/no-go checks for imported skills in our Isaac Lab G1 env (Isaac Sim 5.1, TienKung-Lab G1 asset).

  docker compose run --rm -T train skills/validate.py --check agile_native   (one check per process)

  dwaq_equivalence  DwaqPolicy adapter vs the env + rsl_rl policy it came from (must match to ~1e-5)
  agile_native      AGILE velocity-height on flat ground with its own PD gains: walk, turn, crouch, rise
  agile_remap       same, executed with our (TienKung) gains through gain_equivalent_target at 50 Hz
  agile_arms        agile_native with the bully clip on the arms (AGILE saw random arms only while standing)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--check", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--ankle_effort", type=float, default=None,
                    help="override the ankle effort limit (TienKung asset: 35 N m; AGILE training and the real G1: 50)")
parser.add_argument("--upper", choices=["zero", "tienkung", "clear"], default="clear",
                    help="upper-body hold pose for the agile checks: all zero (AGILE default, hands at the thighs), "
                         "TienKung default, or arms abducted with bent elbows (hands clear of the thighs)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

from skills import registry  # noqa: E402
from skills.adapters import AgileVelocityHeight, DwaqPolicy, gain_equivalent_target  # noqa: E402
from skills.runtime import SkillRunner, make_env  # noqa: E402

# (start s, end s, vx, vy, yaw rate, height start, height end)
SCHEDULE = [(0, 2, 0, 0, 0, .72, .72), (2, 7, .6, 0, 0, .72, .72), (7, 10, .3, 0, .5, .72, .72),
            (10, 12, 0, 0, 0, .72, .72), (12, 15, 0, 0, 0, .72, .50), (15, 17, 0, 0, 0, .50, .50),
            (17, 19, 0, 0, 0, .50, .72)]


def command_at(t, n, device):
    for a, b, vx, vy, wz, h0, h1 in SCHEDULE:
        if a <= t < b:
            h = h0 + (h1 - h0) * (t - a) / (b - a)
            return torch.tensor([[vx, vy, wz, h]], device=device).repeat(n, 1)
    return torch.tensor([[0, 0, 0, .72]], device=device).repeat(n, 1)


def check_dwaq_equivalence():
    from rsl_rl.runners import DWAQOnPolicyRunner

    env, _, agent_cfg = make_env("g1_dwaq", "flat", args.num_envs)
    ckpt = registry.path("dwaq_upstream")
    runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt, load_optimizer=False)
    policy = runner.get_inference_policy(env.device)
    skill = DwaqPolicy(ckpt, device=env.device)
    sr = SkillRunner(env)
    obs, hist = env.get_observations()
    worst, seg_worst = 0.0, {}
    segs = [("ang_vel", 3), ("gravity", 3), ("command", 3), ("joint_pos", 29), ("joint_vel", 29), ("last_action", 29),
            ("gait_sin", 2), ("gait_cos", 2)]
    for k in range(150):
        st = sr.state()
        st.time = env.episode_length_buf.float() * env.step_dt
        f, off = skill.frame(st, env.command_generator.command), 0
        for name, d in segs:  # which observation term disagrees, if any
            seg_worst[name] = max(seg_worst.get(name, 0.0), float((f[:, off:off + d] - obs[:, off:off + d]).abs().max()))
            off += d
        mine = skill.act(st, env.command_generator.command, last_action=obs[:, 67:96])
        with torch.inference_mode():
            a = policy(obs, hist.to(env.device))
        # Reference = the same policy modules with the VAE means. The fork's act_inference samples the latent
        # (reparameterise), so `a` is noisy; the executed action stays `a`, as in training/rollouts.
        pol = runner.alg.policy
        with torch.inference_mode():
            e = pol.encoder(hist.to(env.device))
            code = torch.cat([pol.encode_mean_vel(e), pol.encode_mean_latent(e)], dim=1)
            ref = skill.default29 + 0.25 * pol.actor(torch.cat([code, obs], dim=1))
        worst = max(worst, float((mine - ref[:, sr.ids(skill.joints)]).abs().max()))
        obs, _, _, extras = env.step(a)
        hist = extras["observations"]["obs_hist"]
        if (env.episode_length_buf == 0).any():  # an env was reset: its histories restart, stop comparing
            break
    print(f"[dwaq_equivalence] max |adapter - policy (VAE means)| target diff over {k + 1} steps: {worst:.2e}  "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}; per-term obs max diff: "
          + ", ".join(f"{n} {v:.1e}" for n, v in seg_worst.items()))


def run_agile(mode):
    env, _, _ = make_env("g1_dwaq", "flat", args.num_envs)
    sr = SkillRunner(env)
    agile = AgileVelocityHeight(registry.path("agile_velocity_height"), device=env.device)
    legs = sr.ids(agile.joints)
    upper_names = [n for n in sr.names if n not in agile.joints]
    upper = sr.ids(upper_names)
    clip = registry.load("bully_clip", device=env.device) if mode == "arms" else None
    tk_kp = torch.tensor([float(next(v for k, v in DwaqPolicy._kp.items() if k in n)) for n in agile.joints], device=env.device)
    tk_kd = torch.tensor([float(next(v for k, v in DwaqPolicy._kd.items() if k in n)) for n in agile.joints], device=env.device)
    if args.ankle_effort:
        ank = sr.ids([n for n in sr.names if "ankle" in n])
        lim = sr.robot.data.joint_effort_limits.clone()
        print(f"    ankle effort limit {float(lim[0, ank[0]]):.0f} -> {args.ankle_effort:.0f} N m")
        lim[:, ank] = args.ankle_effort
        sr.robot.write_joint_effort_limit_to_sim(lim)
    if mode == "remap":
        sr.set_gains([])  # keep TienKung gains everywhere
    else:
        sr.set_gains([(legs, agile.kp, agile.kd)])
    hold = {"zero": {}, "tienkung": {"left_shoulder_pitch_joint": .35, "right_shoulder_pitch_joint": .35,
                                    "left_shoulder_roll_joint": .18, "right_shoulder_roll_joint": -.18,
                                    "left_elbow_joint": .87, "right_elbow_joint": .87},
            "clear": {"left_shoulder_roll_joint": .35, "right_shoulder_roll_joint": -.35,
                      "left_elbow_joint": 1.1, "right_elbow_joint": 1.1}}[args.upper]
    upper_hold = torch.tensor([hold.get(n, 0.0) for n in upper_names], device=env.device).repeat(env.num_envs, 1)
    vx_err, h_err, n_walk, n_h = 0.0, 0.0, 0, 0
    fell_at = torch.full((env.num_envs,), float("nan"), device=env.device)
    T = SCHEDULE[-1][1]
    for k in range(int(T / env.step_dt)):
        t = k * env.step_dt
        st = sr.state()
        cmd = command_at(t, env.num_envs, env.device)
        tgt = agile.act(st, cmd)
        if mode == "remap":
            q, qd = st.q[:, legs], st.qd[:, legs]
            tgt = gain_equivalent_target(tgt, q, qd, agile.kp, agile.kd, tk_kp, tk_kd)
        up = upper_hold
        if clip is not None:
            cidx = [upper_names.index(j) for j in clip.joints]
            up = upper_hold.clone()
            up[:, cidx] = clip.act(st)
        sr.step([(legs, tgt), (upper, up)])
        f = sr.fallen() & torch.isnan(fell_at)
        fell_at[f] = t
        v = sr.robot.data.root_lin_vel_b
        if cmd[0, 0] > 0 and t > 3.0 and t < 10:
            vx_err += float((v[:, 0] - cmd[:, 0]).abs().mean()); n_walk += 1
        if abs(t - 1.9) < 1e-6 or abs(t - 16.9) < 1e-6:
            print(f"    t={t:.1f}s cmd h {float(cmd[0, 3]):.2f}: measured pelvis-above-ground {float(st.base_height.mean()):.3f}, "
                  f"root z {float(sr.robot.data.root_pos_w[:, 2].mean()):.3f}, ground {float(sr.ground_height().mean()):.3f}")
        if 13 <= t < 17:
            h_err += float((st.base_height - cmd[:, 3]).abs().mean()); n_h += 1
    falls = int((~torch.isnan(fell_at)).sum())
    print(f"[agile_{mode}, upper={args.upper}] envs {env.num_envs}: falls {falls} (at {sorted(round(x, 1) for x in fell_at[~torch.isnan(fell_at)].tolist())}), "
          f"mean |vx err| walking {vx_err / max(n_walk, 1):.3f} m/s, mean |height err| crouching {h_err / max(n_h, 1):.3f} m, "
          f"final pelvis height {float(sr.state().base_height.mean()):.2f} m")


checks = {"dwaq_equivalence": check_dwaq_equivalence, "agile_native": lambda: run_agile("native"),
          "agile_remap": lambda: run_agile("remap"), "agile_arms": lambda: run_agile("arms")}
checks[args.check]()  # one per process: Isaac Lab builds one scene per app
sys.stdout.flush()
os._exit(0)
