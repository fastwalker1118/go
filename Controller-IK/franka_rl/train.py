# Franka pick-and-place RL training with delta actions and online IK teacher
"""Train Franka Panda pick-and-place RL policy.

Usage:
    python franka_rl/train.py --headless --num_envs 4096
    python franka_rl/train.py --headless --num_envs 2048  # less GPU memory
"""

import argparse
import os
import sys

# Ensure franka_rl package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Enable omni.replicator for inline video recording
if "--enable" not in " ".join(sys.argv):
    sys.argv.extend(["--enable", "omni.replicator.core"])

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka pick-and-place RL policy.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=30000, help="Max training iterations")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument(
    "--load_checkpoint", type=str, default=None, help="Specific checkpoint path to load"
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml

# Now import our modules (after AppLauncher)
from franka_rl.config import FrankaPickPlaceEnvCfg, FrankaPickPlaceAgentCfg
from franka_rl.env import FrankaPickPlaceEnv
from franka_rl.bc_pretrain import bc_pretrain

from rsl_rl.runners import OnPolicyRunner

# Set wandb dir
_workspace_dir = os.path.dirname(os.path.abspath(__file__))
_wandb_dir = os.path.join(_workspace_dir, "logs")
os.makedirs(_wandb_dir, exist_ok=True)
os.environ["WANDB_DIR"] = _wandb_dir

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class _EarlyStop(Exception):
    pass


def train():
    env_cfg = FrankaPickPlaceEnvCfg()
    agent_cfg = FrankaPickPlaceAgentCfg()

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.resume = args_cli.resume

    env = FrankaPickPlaceEnv(env_cfg, args_cli.headless)

    log_root_path = os.path.join(_workspace_dir, "logs", agent_cfg.experiment_name)
    os.makedirs(log_root_path, exist_ok=True)

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args_cli.load_checkpoint:
        print(f"[INFO]: Loading model checkpoint from: {args_cli.load_checkpoint}")
        runner.load(args_cli.load_checkpoint)
        agent_cfg.resume = True  # skip BC
    elif agent_cfg.resume:
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── BC pretraining (skip if 0 epochs) ─────────────────────────────────
    if not agent_cfg.resume and env_cfg.use_online_teacher and env_cfg.bc_num_epochs > 0:
        from rsl_rl.utils.wandb_utils import WandbSummaryWriter

        runner.logger_type = "wandb"
        runner.writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=agent_cfg.to_dict())
        runner.writer.log_config(env_cfg, runner.cfg, runner.alg_cfg, runner.policy_cfg)

        bc_pretrain(
            runner,
            env,
            num_epochs=env_cfg.bc_num_epochs,
            steps_per_epoch=env_cfg.bc_steps_per_epoch,
            lr=env_cfg.bc_lr,
        )

    _ = runner.get_inference_policy(device=agent_cfg.device)

    # ── Save hook: early stopping ────────────────────────────────────────
    _original_save = runner.save
    _success_history = []

    def _save_with_early_stop(path):
        _original_save(path)
        sr = env.get_success_rate()
        _success_history.append(sr)
        window = env_cfg.early_stop_window
        threshold = env_cfg.early_stop_success_rate
        if len(_success_history) >= window:
            avg = sum(_success_history[-window:]) / window
            if avg >= threshold:
                print(
                    f"\n[EarlyStop] Avg success rate {avg:.3f} >= {threshold} "
                    f"over {window} saves. Stopping training."
                )
                raise _EarlyStop()

    runner.save = _save_with_early_stop

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except _EarlyStop:
        final_path = os.path.join(log_dir, "model_final.pt")
        _original_save(final_path)
        print(f"[EarlyStop] Final model saved to {final_path}")


if __name__ == "__main__":
    train()
    simulation_app.close()
