from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    algorithm_id = "base"
    display_name = "Base Agent"
    trainable = False

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.training_mode = False

    @property
    def needs_next_action(self) -> bool:
        return False

    def clone(self) -> "BaseAgent":
        return copy.deepcopy(self)

    def reset(self) -> None:
        return None

    def start_episode(self, training: bool = False) -> None:
        self.training_mode = training
        self.reset()

    def end_episode(self, episode_reward: float) -> None:
        return None

    def act(self, state: Dict[str, str], observation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        return self.select_action(state=state, observation=observation), {}

    def observe_transition(
        self,
        state: Dict[str, str],
        observation: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, str],
        next_observation: Dict[str, Any] | None,
        done: bool,
        action_data: Dict[str, Any] | None = None,
        next_action: str | None = None,
        next_action_data: Dict[str, Any] | None = None,
    ) -> None:
        return None

    @abstractmethod
    def select_action(self, state: Dict[str, str], observation: Dict[str, Any]) -> str:
        raise NotImplementedError
