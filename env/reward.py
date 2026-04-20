from __future__ import annotations

from typing import Any


def compute_reward(
    previous_state: dict[str, Any],
    current_state: dict[str, Any],
    step_info: dict[str, Any],
) -> float:
    reward = 0.0

    reward += 12.0 * step_info["new_completions"]
    reward -= 6.0 * step_info["new_missed_deadlines"]
    reward -= 2.5 * step_info["invalid_action"]
    reward -= 1.0 * step_info["budget_overspend"]
    reward -= 1.5 * step_info["burnout_incidents"]

    satisfaction_delta = current_state["client_satisfaction"] - previous_state["client_satisfaction"]
    reward += 8.0 * satisfaction_delta

    fatigue_delta = sum(
        current_state["team"][name]["fatigue"] - previous_state["team"][name]["fatigue"]
        for name in current_state["team"]
    )
    reward -= 2.0 * max(0.0, fatigue_delta)

    if step_info["event_name"] == "none":
        reward += 0.25

    if step_info["all_tasks_completed"]:
        reward += 20.0

    return round(reward, 4)
