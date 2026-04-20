from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_SPECIALTIES = {
    "engineer": "backend",
    "designer": "design",
}

DEFAULT_TEAM = {
    "engineer": {
        "capacity": 1.0,
        "fatigue": 0.15,
        "specialty": DEFAULT_SPECIALTIES["engineer"],
        "sick": False,
    },
    "designer": {
        "capacity": 0.9,
        "fatigue": 0.1,
        "specialty": DEFAULT_SPECIALTIES["designer"],
        "sick": False,
    },
}

DEFAULT_TASKS = [
    {
        "id": 1,
        "name": "Backend API",
        "owner": "engineer",
        "deadline": 4,
        "progress": 0.1,
        "priority": 2,
        "blocked": False,
        "budget_spent": 0,
        "required_specialty": "backend",
        "completed": False,
    },
    {
        "id": 2,
        "name": "Landing Page",
        "owner": "designer",
        "deadline": 3,
        "progress": 0.0,
        "priority": 3,
        "blocked": False,
        "budget_spent": 0,
        "required_specialty": "design",
        "completed": False,
    },
    {
        "id": 3,
        "name": "QA Sweep",
        "owner": "engineer",
        "deadline": 5,
        "progress": 0.0,
        "priority": 1,
        "blocked": False,
        "budget_spent": 0,
        "required_specialty": "backend",
        "completed": False,
    },
]


def make_initial_state(max_steps: int) -> dict[str, Any]:
    return {
        "step": 0,
        "max_steps": max_steps,
        "time_left": max_steps,
        "budget_left": 1000,
        "client_satisfaction": 0.85,
        "tasks": deepcopy(DEFAULT_TASKS),
        "team": deepcopy(DEFAULT_TEAM),
        "events": {
            "engineer_sick": False,
            "urgent_bug": False,
            "scope_change": False,
            "last_event": "none",
        },
        "metrics": {
            "completed_tasks": 0,
            "missed_deadlines": 0,
            "budget_allocated": 0,
            "burnout_incidents": 0,
        },
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(state)


def get_task(state: dict[str, Any], task_id: int) -> dict[str, Any] | None:
    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    return None


def active_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for task in state["tasks"] if not task["completed"]]


def completed_tasks(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [task for task in state["tasks"] if task["completed"]]
