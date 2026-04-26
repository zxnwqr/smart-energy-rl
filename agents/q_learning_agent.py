from __future__ import annotations

import random
from typing import Any, Dict

from agents.base_agent import BaseAgent
from agents.rl_utils import action_index, action_name, ensure_q_row, epsilon_greedy_index, greedy_index, state_to_key


class QLearningAgent(BaseAgent):
    algorithm_id = "q_learning"
    display_name = "Q-Learning Agent"
    trainable = True

    def __init__(
        self,
        seed: int | None = None,
        learning_rate: float = 0.18,
        gamma: float = 0.92,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.975,
        min_epsilon: float = 0.05,
    ) -> None:
        super().__init__(seed=seed)
        self.random = random.Random(seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table: dict[tuple[str, ...], list[float]] = {}

    def _q_values(self, state: Dict[str, str]) -> list[float]:
        return ensure_q_row(self.q_table, state_to_key(state))

    def select_action(self, state: Dict[str, str], observation: Dict[str, Any]) -> str:
        q_values = self._q_values(state)
        if self.training_mode:
            index = epsilon_greedy_index(q_values, self.epsilon, self.random)
        else:
            index = greedy_index(q_values)
        return action_name(index)

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
        if not self.training_mode:
            return

        current_values = self._q_values(state)
        current_index = action_index(action)
        future_reward = 0.0 if done else max(self._q_values(next_state))
        target = reward + self.gamma * future_reward
        current_values[current_index] += self.learning_rate * (target - current_values[current_index])

    def end_episode(self, episode_reward: float) -> None:
        if self.training_mode:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
