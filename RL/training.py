import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import os
import argparse
import numpy as np
import torch
import wandb

from dataset import SokobanDataset
from sokoban_env import SokobanEnv
from dqn_agent import DQNAgent
from trainer import train_dqn

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataset(data_split_dir):
    dataset_paths=[]

    for group_path in data_split_dir.iterdir():
        for path in group_path.iterdir():
            if path.with_suffix(".json"):
                dataset_paths.append(path)

    if len(dataset_paths) == 0:
        raise RuntimeError(f"Không tìm thấy file JSON trong: {data_split_dir}")

    print(f"Found {len(dataset_paths)} levels in {data_split_dir}")

    return SokobanDataset(dataset_paths)

def main():

    seed = 42

    base_dir = Path(__file__).parent.parent.resolve()
    data_dir= base_dir / "levels" / "training_dataset"

    train_dir = data_dir/"train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    num_episodes = 1_000_000
    max_pushes = 100
    batch_size = 512
    learning_rate = 1e-4
    gamma = 0.99
    replay_capacity = 1_000_000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 1_500_000 
    target_update_frequency = 5000
    val_every = 10_000
    log_every = 2_000
    n_step = 6
    parser = argparse.ArgumentParser(description="Train DQN for Sokoban")
    parser.add_argument("--model-path", "--checkpoint", "-m", type=str, default=None)
    parser.add_argument("--run-id", "-r", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = base_dir / "RL" / "models" / "dqn_sokoban_best.pt"
    last_checkpoint_path = base_dir / "RL" / "models" / "dqn_sokoban_last.pt"

    custom_model_path = args.model_path or os.getenv("MODEL_PATH", "").strip() or None
    if custom_model_path:
        resume_checkpoint_path = Path(custom_model_path)
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Can't find checkpoint in: {resume_checkpoint_path}")
        print(f"[Model] Using checkpoint: {resume_checkpoint_path}")
    else:
        resume_checkpoint_path = last_checkpoint_path if last_checkpoint_path.exists() else checkpoint_path

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(train_dir)
    val_dataset= build_dataset(val_dir)

    sample_map, sample_dead_squares = train_dataset[0]

    env = SokobanEnv(
        map_matrix=sample_map,
        dead_squares=sample_dead_squares,
        max_pushes=max_pushes,
        avoid_dead_squares=False
    )

    print(f"Environment size: {env.height} x {env.width}")

    print(f"Action space: {env.num_actions}")

    agent = DQNAgent(
        lr=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_frequency=target_update_frequency,
        device=device,
        per_beta_frames=num_episodes * 50,
        per_alpha=0.6,
        per_beta_start=0.4,
    )

    start_episode = 1
    resume_best_solve_rate = -1.0
    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).exists():
        chkpt = agent.load(resume_checkpoint_path)
        resume_best_solve_rate = chkpt.get("best_val_solve_rate", -1.0)
        start_episode = chkpt.get("episode", 0) + 1
        saved_batch_size = chkpt.get("batch_size", None)

        if saved_batch_size is not None and saved_batch_size != batch_size:
            agent.optimizer = torch.optim.AdamW(agent.policy_net.parameters(), lr=learning_rate)
            optimizer_status = f"RESET (batch {saved_batch_size}→{batch_size})"
        else:
            optimizer_status = f"RESTORED (batch_size={batch_size})"

        print(f"Resumed from checkpoint: {resume_checkpoint_path}")
        print(f"  start_episode    = {start_episode:,}")
        print(f"  total_steps      = {agent.total_steps:,}")
        print(f"  gradient_steps   = {agent.gradient_steps:,}")
        print(f"  epsilon          = {agent.get_epsilon():.4f}")
        print(f"  best_solve_rate  = {resume_best_solve_rate*100:.1f}%")
        print(f"  optimizer        = {optimizer_status}")
    else:
        print("Starting fresh training (no checkpoint loaded)")

    wandb_run_id = args.run_id or os.getenv("WANDB_RUN_ID", "").strip() or None
    use_wandb = os.getenv("USE_WANDB", "1") == "1" and wandb is not None
    if use_wandb:
        try:
            init_kwargs = {
                "project": os.getenv("WANDB_PROJECT", "sokoban-dqn"),
                "config": {
                    "seed": seed,
                    "num_episodes": num_episodes,
                    "max_pushes": max_pushes,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "replay_capacity": replay_capacity,
                    "epsilon_start": epsilon_start,
                    "epsilon_end": epsilon_end,
                    "epsilon_decay": epsilon_decay,
                    "target_update_frequency": target_update_frequency,
                    "val_every": val_every,
                    "log_every": log_every,
                    "n_step": n_step,
                },
            }
            if wandb_run_id:
                init_kwargs["id"] = wandb_run_id
                init_kwargs["resume"] = "allow"
                print(f"[W&B] Resuming run with ID: {wandb_run_id}")
            else:
                init_kwargs["name"] = os.getenv("WANDB_RUN_NAME", "train_sokoban")

            wandb.init(**init_kwargs)
            print(f"[W&B] Initialized run successfully (Run ID: {wandb.run.id}).")
        except Exception as e:
            print(f"[W&B] Could not initialize wandb: {e}. Continuing training without wandb...")
            use_wandb = False

    print("\nStart training...\n")

    best_val_solve_rate = train_dqn(
        agent=agent,
        env=env,
        dataset=train_dataset,
        num_episodes=num_episodes,
        val_dataset=val_dataset,
        val_every=val_every,
        checkpoint_path=checkpoint_path,
        log_every=log_every,
        n_step=n_step,
        resume_best_solve_rate=resume_best_solve_rate,
        start_episode=start_episode,
        use_wandb=use_wandb,
    )

    if use_wandb and wandb is not None and wandb.run is not None:
        wandb.finish()

    print(f"\nTraining finished.")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Best val solve rate: {best_val_solve_rate*100:.1f}%")

if __name__ == "__main__":
    main()