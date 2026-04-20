from __future__ import annotations

import json
from typing import Any


def build_action_prompt(state: dict[str, Any]) -> str:
    compact_state = {
        "time_left": state["time_left"],
        "budget_left": state["budget_left"],
        "client_satisfaction": state["client_satisfaction"],
        "events": state["events"],
        "team": state["team"],
        "tasks": state["tasks"],
    }
    return (
        "You are a project manager operating in a simulation.\n"
        "Choose the next action and respond with JSON only.\n\n"
        "Allowed actions:\n"
        '- {"action": "work_on_task", "task_id": <int>, "effort_level": 1 or 2}\n'
        '- {"action": "reassign_task", "task_id": <int>, "new_owner": "engineer" or "designer"}\n'
        '- {"action": "prioritize_task", "task_id": <int>}\n'
        '- {"action": "allocate_budget", "task_id": <int>, "amount": 100}\n'
        '- {"action": "do_nothing"}\n\n'
        f"Current state:\n{json.dumps(compact_state, indent=2)}"
    )
