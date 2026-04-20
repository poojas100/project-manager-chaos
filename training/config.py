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
