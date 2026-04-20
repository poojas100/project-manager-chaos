from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO

from env.project_env import ProjectEnv


class SB3PolicyAgent:
    def __init__(self, env: ProjectEnv, model: Any):
        self.env = env
        self.model = model

    @classmethod
    def from_path(cls, env: ProjectEnv, algorithm: str, model_path: str | Path) -> "SB3PolicyAgent":
        loader_map = {
            "ppo": PPO,
            "dqn": DQN,
        }
        algorithm_key = algorithm.lower()
        if algorithm_key not in loader_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        model = loader_map[algorithm_key].load(str(model_path))
        return cls(env=env, model=model)

    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        observation = self.env.encode_observation(state)
        action_index, _ = self.model.predict(observation, deterministic=True)
        return self.env.decode_action(int(action_index))
