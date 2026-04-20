from __future__ import annotations

from copy import deepcopy
from typing import Any

from env.project_env import ProjectEnv


class GreedyAgent:
    def __init__(self, env: ProjectEnv):
        self.env = env

    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        del state
        best_action_index = 0
        best_score = float("-inf")

        for action_index in range(self.env.action_space.n):
            candidate_env = deepcopy(self.env)
            _, reward, terminated, truncated, info = candidate_env.step(action_index)
            metrics = info["metrics"]
            score = reward
            score += 2.0 * metrics["completed_tasks"]
            score -= 1.0 * metrics["missed_deadlines"]
            if terminated:
                score += 5.0
            if truncated and not terminated:
                score -= 0.5
            if info["invalid_action"]:
                score -= 2.0

            if score > best_score:
                best_score = score
                best_action_index = action_index

        return self.env.decode_action(best_action_index)
