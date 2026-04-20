from __future__ import annotations

from typing import Any

import numpy as np

from env.project_env import ProjectEnv


class RandomAgent:
    def __init__(self, env: ProjectEnv, seed: int | None = None):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        del state
        action_index = int(self.rng.integers(0, self.env.action_space.n))
        return self.env.decode_action(action_index)
