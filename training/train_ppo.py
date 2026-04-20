from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from env.project_env import ProjectEnv
from training.config import PPOTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO baseline for Project Manager Chaos.")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PPOTrainingConfig(total_timesteps=args.timesteps, seed=args.seed)

    env = ProjectEnv(seed=config.seed)
    model = PPO(
        config.policy,
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        seed=config.seed,
    )
    model.learn(total_timesteps=config.total_timesteps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "ppo_project_manager"
    model.save(model_path.as_posix())
    print(f"Saved PPO baseline to {model_path}")


if __name__ == "__main__":
    main()
