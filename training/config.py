from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PPOTrainingConfig:
    total_timesteps: int = 10000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 7
    policy: str = "MlpPolicy"


@dataclass(slots=True)
class DQNTrainingConfig:
    total_timesteps: int = 10000
    learning_rate: float = 1e-3
    buffer_size: int = 5000
    learning_starts: int = 100
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 250
    seed: int = 7
    policy: str = "MlpPolicy"
