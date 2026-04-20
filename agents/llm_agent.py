from __future__ import annotations

import json
from typing import Any, Protocol

from utils.prompt_builder import build_action_prompt


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class LLMJSONAgent:
    def __init__(self, model: TextGenerator, fallback_action: dict[str, Any] | None = None):
        self.model = model
        self.fallback_action = fallback_action or {"action": "do_nothing"}

    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = build_action_prompt(state)
        raw_response = self.model.generate(prompt)
        return self.parse_action(raw_response)

    def parse_action(self, raw_response: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return dict(self.fallback_action)

        if not isinstance(parsed, dict) or "action" not in parsed:
            return dict(self.fallback_action)

        action_name = parsed["action"]
        if action_name not in {
            "work_on_task",
            "reassign_task",
            "prioritize_task",
            "allocate_budget",
            "do_nothing",
        }:
            return dict(self.fallback_action)

        return parsed
