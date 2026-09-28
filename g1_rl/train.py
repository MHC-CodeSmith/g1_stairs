"""Train a G1 task in Isaac Sim / Isaac Lab (headless), starting from a pretrained policy.

  python g1_rl/train.py --headless --task g1_body --num_envs 4096 --max_iterations 1500 \
      --init_checkpoint /workspace/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt

Same flow as TienKung-Lab/legged_lab/scripts/train.py, plus --init_checkpoint: load policy weights (not the
optimizer or iteration counter) from the pretrained stair policy, so fine-tuning starts from a stair climber.
If the task's observation differs from the source's (g1_body adds a height command), the weights are mapped onto
the new layout (warmstart.py).
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import legged_lab.utils.cli_args as cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="g1_body")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--init_checkpoint", type=str, default=None, help="policy weights to start from")
parser.add_argument("--log_root", type=str, default="/workspace/g1_stairs/logs")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime  # noqa: E402

import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import DWAQOnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403  (registers upstream tasks)
from legged_lab.utils import task_registry  # noqa: E402
from legged_lab.utils.cli_args import update_rsl_rl_cfg  # noqa: E402

import g1_rl.body  # noqa: E402,F401  (registers g1_body)
import rsl_rl.runners.dwaq_on_policy_runner as _runner_mod  # noqa: E402
from g1_rl.distill import DistillDWAQPPO  # noqa: E402

_runner_mod.DistillDWAQPPO = DistillDWAQPPO  # the runner resolves algorithm.class_name with eval() in its module
from g1_rl import chown_to_host  # noqa: E402
from g1_rl.warmstart import load_expanded, load_mapped  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device  # learner on the sim device (AppLauncher's --device, default cuda:0)
    env_cfg.scene.seed = agent_cfg.seed
    env = task_registry.get_task_class(args_cli.task)(env_cfg, args_cli.headless)

    log_root = os.path.join(os.path.abspath(args_cli.log_root), agent_cfg.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))
    runner = DWAQOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if isinstance(runner.alg, DistillDWAQPPO):
        runner.alg.teacher_source = env
        runner.alg.total_iterations = agent_cfg.max_iterations
        base_log = runner.log

        def log(locs, *a, **kw):  # the runner only records its own three losses
            base_log(locs, *a, **kw)
            extra = getattr(runner.alg, "last_losses", {})
            for k, v in extra.items():
                if runner.writer is not None:
                    runner.writer.add_scalar(f"Loss/{k}", v, locs["it"])
            print("  " + "  ".join(f"Loss/{k}: {v:.4f}" for k, v in extra.items()), flush=True)

        runner.log = log
    if agent_cfg.resume:
        path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] resuming from {path}")
        runner.load(path)
    elif args_cli.init_checkpoint:
        print(f"[INFO] initializing policy from {args_cli.init_checkpoint}")
        src = torch.load(args_cli.init_checkpoint, weights_only=False, map_location="cpu")["model_state_dict"]
        old_obs = src["decoder.4.weight"].shape[0]
        new_obs = runner.alg.policy.state_dict()["decoder.4.weight"].shape[0]
        hist = env.cfg.robot.dwaq_obs_history_length
        if hasattr(env, "obs_map_from"):  # the task says how its observation relates to older ones
            lines = load_mapped(runner.alg.policy, src, env.obs_map_from(old_obs), hist)
        else:
            lines = load_expanded(runner.alg.policy, src, old_obs, hist)
        for line in lines:
            print(f"[INFO] warm start expanded {line}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        chown_to_host(log_dir)


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang indefinitely in headless Isaac Sim 5.1; everything is already written.
    sys.stdout.flush()
    os._exit(0)
