from __future__ import annotations

from typing import Any

from .state import get_task


EVENT_SEQUENCE = [
    "none",
    "engineer_sick",
    "urgent_bug",
    "scope_change",
]


def reset_event_flags(state: dict[str, Any]) -> None:
    state["events"]["engineer_sick"] = False
    state["events"]["urgent_bug"] = False
    state["events"]["scope_change"] = False
    state["events"]["last_event"] = "none"


def apply_event(state: dict[str, Any], event_name: str) -> str:
    reset_event_flags(state)
    state["events"]["last_event"] = event_name

    if event_name == "none":
        return event_name

    if event_name == "engineer_sick":
        engineer = state["team"]["engineer"]
        engineer["sick"] = True
        engineer["fatigue"] = min(1.0, engineer["fatigue"] + 0.2)
        state["events"]["engineer_sick"] = True
        state["client_satisfaction"] = max(0.0, state["client_satisfaction"] - 0.03)
        return event_name

    if event_name == "urgent_bug":
        task = get_task(state, 3)
        if task is not None and not task["completed"]:
            task["priority"] = 3
            task["deadline"] = max(1, min(task["deadline"], 2))
            task["blocked"] = False
        state["events"]["urgent_bug"] = True
        state["client_satisfaction"] = max(0.0, state["client_satisfaction"] - 0.02)
        return event_name

    if event_name == "scope_change":
        next_id = max(task["id"] for task in state["tasks"]) + 1
        state["tasks"].append(
            {
                "id": next_id,
                "name": "Client Revision",
                "owner": "designer",
                "deadline": 2,
                "progress": 0.0,
                "priority": 2,
                "blocked": False,
                "budget_spent": 0,
                "required_specialty": "design",
                "completed": False,
            }
        )
        state["events"]["scope_change"] = True
        state["client_satisfaction"] = max(0.0, state["client_satisfaction"] - 0.04)
        return event_name

    raise ValueError(f"Unknown event: {event_name}")


def sample_event(rng: Any) -> str:
    return str(rng.choice(EVENT_SEQUENCE))
