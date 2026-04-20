from __future__ import annotations

from typing import Any


class HeuristicAgent:
    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        team = state["team"]
        active_tasks = [task for task in state["tasks"] if not task["completed"]]
        if not active_tasks:
            return {"action": "do_nothing"}

        active_tasks.sort(key=lambda task: (task["deadline"], -task["priority"], task["progress"]))

        for task in active_tasks:
            if task["blocked"] and state["budget_left"] >= 100:
                return {"action": "allocate_budget", "task_id": task["id"], "amount": 100}

        urgent_task = active_tasks[0]
        preferred_owner = "designer" if urgent_task["required_specialty"] == "design" else "engineer"
        if urgent_task["owner"] != preferred_owner and not team[preferred_owner]["sick"]:
            return {
                "action": "reassign_task",
                "task_id": urgent_task["id"],
                "new_owner": preferred_owner,
            }

        if urgent_task["priority"] < 3 and urgent_task["deadline"] <= 2:
            return {"action": "prioritize_task", "task_id": urgent_task["id"]}

        if not team[urgent_task["owner"]]["sick"]:
            effort_level = 2 if urgent_task["deadline"] <= 1 else 1
            return {
                "action": "work_on_task",
                "task_id": urgent_task["id"],
                "effort_level": effort_level,
            }

        for task in active_tasks[1:]:
            if not team[task["owner"]]["sick"]:
                return {"action": "work_on_task", "task_id": task["id"], "effort_level": 1}

        return {"action": "do_nothing"}
