import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "cpp"))

import os
import random
import argparse
import numpy as np
import torch
from dqn_agent import DQNAgent
from dataset import SokobanDataset
from trainer import train_dqn

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sokoban_cpp
import wandb

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataset(data_split_dir: Path) -> SokobanDataset:
    dataset_paths = []

    if not data_split_dir.exists():
        raise RuntimeError(f"Could not find folder: {data_split_dir}")
    
    for item in data_split_dir.iterdir():
        dataset_paths.extend(p for p in item.iterdir() if p.suffix == ".json")

    print(f"Found {len(dataset_paths):,} levels in {data_split_dir}")
    return SokobanDataset(dataset_paths)

def main():

    parser = argparse.ArgumentParser(description="Train DQN for Sokoban")
    parser.add_argument("--model-path", "-m", type=str, default=None)
    parser.add_argument("--run-id", "-r", type=str, default=None)
    parser.add_argument("--dataset-path","-d", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)

    num_episodes = 1_000_000
    max_pushes = 100
    batch_size = 256
    learning_rate = 1e-4
    gamma = 0.99
    replay_capacity = 1_000_000
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 1_500_000 
    target_update_frequency = 2000
    val_every = 10_000
    log_every = 2_000
    n_step = 6

    base_dir = Path(__file__).parent.parent.resolve()

    checkpoint_path = base_dir / "RL" / "models" / "dqn_sokoban_best.pt"
    last_checkpoint_path = base_dir / "RL" / "models" / "dqn_sokoban_last.pt"

    dataset_path = args.dataset_path or os.getenv("DATASET_PATH", "").strip() or None
    if dataset_path:
        data_dir = Path(dataset_path)
    else:
        data_dir = base_dir / "levels" / "training_dataset"

    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    train_dataset = build_dataset(train_dir)
    val_dataset = build_dataset(val_dir)

    env = sokoban_cpp.SokobanEnv(100, True)

    agent = DQNAgent(
        lr=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_frequency=target_update_frequency,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=num_episodes * 50,
    )

    resume_path = args.model_path or os.getenv("MODEL_PATH", "").strip() or None
    if resume_path:
        resume_path = Path(resume_path)

    start_episode= 1
    resume_best_solve_rate = -1.0

    if resume_path and resume_path.exists():

        chkpt = agent.load(resume_path)
        resume_best_solve_rate = chkpt.get("best_val_solve_rate", -1.0)
        start_episode = chkpt.get("episode", 0) + 1

        print(f"Resumed from: {resume_path} | ep={start_episode:,} | "
              f"best={resume_best_solve_rate*100:.1f}%")
    else:
        print("Starting fresh training.")

    
    kwargs = {
        "project": os.getenv("WANDB_PROJECT", "sokoban-dqn"),
        "config": {
            "seed": 42,
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
            "dataset_path": str(data_dir),
        },
    }

    if args.run_id:
        kwargs["id"] = args.run_id
        kwargs["resume"] = "allow"

    else:
        kwargs["name"] = "train_sokoban"

    wandb.init(**kwargs)
    print(f"[W&B] Initialized run successfully (Run ID: {wandb.run.id}).")

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
        start_episode=start_episode
    )

    wandb.finish()

    print(f"\nTraining finished.")
    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Best val solve rate: {best_val_solve_rate*100:.1f}%")
    
if __name__ == "__main__":
    main()
