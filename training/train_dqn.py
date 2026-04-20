from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from env.project_env import ProjectEnv
from training.config import DQNTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN baseline for Project Manager Chaos.")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dqn_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNTrainingConfig(total_timesteps=args.timesteps, seed=args.seed)

    env = ProjectEnv(seed=config.seed)
    model = DQN(
        config.policy,
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        seed=config.seed,
    )
    model.learn(total_timesteps=config.total_timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "dqn_project_manager"
    model.save(model_path.as_posix())
    print(f"Saved DQN baseline to {model_path}")


if __name__ == "__main__":
    main()
