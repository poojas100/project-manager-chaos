from __future__ import annotations

from typing import Any, Protocol


class Agent(Protocol):
    def act(self, state: dict[str, Any]) -> dict[str, Any]:
        ...
