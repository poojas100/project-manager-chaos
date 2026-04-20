from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from .events import EVENT_SEQUENCE, apply_event, sample_event
from .reward import compute_reward
from .state import active_tasks, clone_state, get_task, make_initial_state


class ProjectEnv(Env):
    metadata = {"render_modes": ["ansi"]}

    ACTION_TEMPLATES = (
        {"action": "do_nothing"},
        {"action": "work_on_task", "task_id": 1, "effort_level": 1},
        {"action": "work_on_task", "task_id": 2, "effort_level": 1},
        {"action": "work_on_task", "task_id": 3, "effort_level": 1},
        {"action": "work_on_task", "task_id": 4, "effort_level": 1},
        {"action": "reassign_task", "task_id": 1, "new_owner": "designer"},
        {"action": "reassign_task", "task_id": 2, "new_owner": "engineer"},
        {"action": "reassign_task", "task_id": 3, "new_owner": "designer"},
        {"action": "reassign_task", "task_id": 4, "new_owner": "engineer"},
        {"action": "prioritize_task", "task_id": 1},
        {"action": "prioritize_task", "task_id": 2},
        {"action": "prioritize_task", "task_id": 3},
        {"action": "prioritize_task", "task_id": 4},
        {"action": "allocate_budget", "task_id": 1, "amount": 100},
        {"action": "allocate_budget", "task_id": 2, "amount": 100},
        {"action": "allocate_budget", "task_id": 3, "amount": 100},
        {"action": "allocate_budget", "task_id": 4, "amount": 100},
    )

    def __init__(self, max_steps: int = 6, event_probability: float = 0.6, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.event_probability = event_probability
        self.action_space = spaces.Discrete(len(self.ACTION_TEMPLATES))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.state = make_initial_state(max_steps=self.max_steps)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self._base_seed is not None:
            self.rng = np.random.default_rng(self._base_seed)
        self.state = make_initial_state(max_steps=self.max_steps)
        observation = self.encode_observation(self.state)
        return observation, {"state": clone_state(self.state)}

    def step(self, action: int | dict[str, Any]):
        previous_state = clone_state(self.state)
        structured_action = self.normalize_action(action)
        step_info = {
            "action": structured_action,
            "invalid_action": 0,
            "new_completions": 0,
            "new_missed_deadlines": 0,
            "burnout_incidents": 0,
            "budget_overspend": 0,
            "all_tasks_completed": False,
            "event_name": "none",
        }

        if not self.apply_action(structured_action):
            step_info["invalid_action"] = 1

        self.advance_project_clock(step_info)
        step_info["event_name"] = self.maybe_apply_event()
        reward = compute_reward(previous_state, self.state, step_info)

        terminated = step_info["all_tasks_completed"]
        truncated = self.state["step"] >= self.max_steps
        observation = self.encode_observation(self.state)
        info = {
            "state": clone_state(self.state),
            "event": step_info["event_name"],
            "action": structured_action,
            "metrics": clone_state(self.state["metrics"]),
            "invalid_action": bool(step_info["invalid_action"]),
        }
        return observation, reward, terminated, truncated, info

    def normalize_action(self, action: int | dict[str, Any]) -> dict[str, Any]:
        if isinstance(action, (int, np.integer)):
            index = int(action)
            if 0 <= index < len(self.ACTION_TEMPLATES):
                return dict(self.ACTION_TEMPLATES[index])
            return {"action": "do_nothing"}

        if isinstance(action, dict):
            return dict(action)

        return {"action": "do_nothing"}

    def decode_action(self, action_index: int) -> dict[str, Any]:
        return dict(self.ACTION_TEMPLATES[action_index])

    def encode_observation(self, state: dict[str, Any]) -> np.ndarray:
        values: list[float] = [
            state["time_left"] / max(1, self.max_steps),
            max(0.0, min(1.0, state["budget_left"] / 1000.0)),
            max(0.0, min(1.0, state["client_satisfaction"])),
        ]

        for member_name in ("engineer", "designer"):
            member = state["team"][member_name]
            values.extend(
                [
                    member["capacity"],
                    member["fatigue"],
                    1.0 if member["sick"] else 0.0,
                ]
            )

        tasks = sorted(state["tasks"], key=lambda task: task["id"])[:4]
        while len(tasks) < 4:
            tasks.append(
                {
                    "progress": 0.0,
                    "deadline": 0,
                    "priority": 0,
                    "blocked": True,
                    "completed": True,
                }
            )

        for task in tasks:
            values.extend(
                [
                    task["progress"],
                    max(0.0, min(1.0, task["deadline"] / 5.0)),
                    task["priority"] / 3.0,
                    1.0 if task["blocked"] else 0.0,
                    1.0 if task["completed"] else 0.0,
                ]
            )

        values.extend(
            [
                1.0 if state["events"]["engineer_sick"] else 0.0,
                1.0 if state["events"]["urgent_bug"] else 0.0,
                1.0 if state["events"]["scope_change"] else 0.0,
            ]
        )

        return np.array(values, dtype=np.float32)

    def apply_action(self, action: dict[str, Any]) -> bool:
        action_name = action.get("action")
        if action_name == "do_nothing":
            return True

        if action_name == "work_on_task":
            task = get_task(self.state, int(action.get("task_id", -1)))
            effort_level = int(action.get("effort_level", 1))
            if task is None or task["completed"] or task["blocked"]:
                return False
            owner = task["owner"]
            member = self.state["team"].get(owner)
            if member is None or member["sick"]:
                return False
            progress_gain = 0.18 + 0.08 * max(1, min(2, effort_level))
            if member["specialty"] == task["required_specialty"]:
                progress_gain += 0.08
            progress_gain *= max(0.35, 1.0 - member["fatigue"])
            task["progress"] = min(1.0, round(task["progress"] + progress_gain, 3))
            member["fatigue"] = min(1.0, round(member["fatigue"] + 0.08 * effort_level, 3))
            return True

        if action_name == "reassign_task":
            task = get_task(self.state, int(action.get("task_id", -1)))
            new_owner = action.get("new_owner")
            if task is None or new_owner not in self.state["team"] or task["completed"]:
                return False
            if task["owner"] == new_owner:
                return False
            task["owner"] = new_owner
            self.state["client_satisfaction"] = max(0.0, self.state["client_satisfaction"] - 0.01)
            return True

        if action_name == "prioritize_task":
            task = get_task(self.state, int(action.get("task_id", -1)))
            if task is None or task["completed"]:
                return False
            for other_task in self.state["tasks"]:
                if other_task["id"] != task["id"] and other_task["priority"] > 1:
                    other_task["priority"] -= 1
            task["priority"] = 3
            return True

        if action_name == "allocate_budget":
            task = get_task(self.state, int(action.get("task_id", -1)))
            amount = int(action.get("amount", 0))
            if task is None or task["completed"] or amount <= 0:
                return False
            actual_amount = min(amount, self.state["budget_left"])
            task["budget_spent"] += actual_amount
            self.state["budget_left"] -= actual_amount
            self.state["metrics"]["budget_allocated"] += actual_amount
            if actual_amount < amount:
                return False
            if task["blocked"]:
                task["blocked"] = False
            task["progress"] = min(1.0, round(task["progress"] + 0.05, 3))
            self.state["client_satisfaction"] = min(1.0, self.state["client_satisfaction"] + 0.01)
            return True

        return False

    def advance_project_clock(self, step_info: dict[str, Any]) -> None:
        self.state["step"] += 1
        self.state["time_left"] = max(0, self.max_steps - self.state["step"])

        for member in self.state["team"].values():
            if member["sick"]:
                member["sick"] = False
            member["fatigue"] = max(0.0, round(member["fatigue"] - 0.02, 3))
            if member["fatigue"] > 0.75:
                step_info["burnout_incidents"] += 1
                self.state["metrics"]["burnout_incidents"] += 1
                self.state["client_satisfaction"] = max(0.0, self.state["client_satisfaction"] - 0.03)

        for task in self.state["tasks"]:
            if task["completed"]:
                continue
            task["deadline"] -= 1
            if task["progress"] >= 1.0:
                task["completed"] = True
                step_info["new_completions"] += 1
                self.state["metrics"]["completed_tasks"] += 1
                self.state["client_satisfaction"] = min(1.0, self.state["client_satisfaction"] + 0.05)
                continue
            if task["deadline"] < 0:
                step_info["new_missed_deadlines"] += 1
                self.state["metrics"]["missed_deadlines"] += 1
                self.state["client_satisfaction"] = max(0.0, self.state["client_satisfaction"] - 0.08)

        if self.state["budget_left"] <= 0:
            step_info["budget_overspend"] = 1
            self.state["client_satisfaction"] = max(0.0, self.state["client_satisfaction"] - 0.05)

        step_info["all_tasks_completed"] = len(active_tasks(self.state)) == 0

    def maybe_apply_event(self) -> str:
        if float(self.rng.random()) > self.event_probability:
            return apply_event(self.state, "none")
        event_name = sample_event(self.rng)
        return apply_event(self.state, event_name)

    def render(self) -> str:
        lines = [
            f"Step {self.state['step']}/{self.max_steps}",
            f"Budget left: {self.state['budget_left']}",
            f"Client satisfaction: {self.state['client_satisfaction']:.2f}",
            f"Last event: {self.state['events']['last_event']}",
        ]
        for task in self.state["tasks"]:
            lines.append(
                f"Task {task['id']} {task['name']}: owner={task['owner']} "
                f"progress={task['progress']:.2f} deadline={task['deadline']} "
                f"priority={task['priority']} completed={task['completed']}"
            )
        return "\n".join(lines)
