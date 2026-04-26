from __future__ import annotations

import random
from typing import Any, Dict

from agents.base_agent import BaseAgent
from models import ACTIONS


class RandomAgent(BaseAgent):
    algorithm_id = "random"
    display_name = "Random Agent"
    trainable = False

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def select_action(self, state: Dict[str, str], observation: Dict[str, Any]) -> str:
        return self.random.choice(ACTIONS)
