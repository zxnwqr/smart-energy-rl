from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict

from agents.base_agent import BaseAgent
from agents.rl_utils import (
    ACTION_INDEX,
    action_name,
    copy_matrix,
    dot_product,
    encode_state_features,
    epsilon_greedy_index,
    greedy_index,
)
from models import ACTIONS


class DQNAgent(BaseAgent):
    algorithm_id = "dqn"
    display_name = "DQN Agent"
    trainable = True

    def __init__(
        self,
        seed: int | None = None,
        learning_rate: float = 0.035,
        gamma: float = 0.92,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.98,
        min_epsilon: float = 0.04,
        batch_size: int = 18,
        target_sync_interval: int = 18,
        replay_buffer_size: int = 900,
    ) -> None:
        super().__init__(seed=seed)
        self.random = random.Random(seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.replay_buffer: deque[tuple[list[float], int, float, list[float] | None, bool]] = deque(
            maxlen=replay_buffer_size
        )
        self.online_weights: list[list[float]] = []
        self.target_weights: list[list[float]] = []
        self.training_steps = 0

    def _ensure_network(self, feature_count: int) -> None:
        if self.online_weights:
            return
        self.online_weights = [
            [self.random.uniform(-0.03, 0.03) for _ in range(feature_count)] for _ in ACTIONS
        ]
        self.target_weights = copy_matrix(self.online_weights)

    def _predict(self, features: list[float], use_target: bool = False) -> list[float]:
        self._ensure_network(len(features))
        weights = self.target_weights if use_target else self.online_weights
        return [dot_product(weight_row, features) for weight_row in weights]

    def act(self, state: Dict[str, str], observation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        features = encode_state_features(state, observation)
        q_values = self._predict(features)
        if self.training_mode:
            index = epsilon_greedy_index(q_values, self.epsilon, self.random)
        else:
            index = greedy_index(q_values)
        return action_name(index), {"features": features, "action_index": index}

    def select_action(self, state: Dict[str, str], observation: Dict[str, Any]) -> str:
        action, _ = self.act(state=state, observation=observation)
        return action

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

        features = list((action_data or {}).get("features", encode_state_features(state, observation)))
        next_features = None if done or next_observation is None else encode_state_features(next_state, next_observation)
        chosen_action = (action_data or {}).get("action_index", ACTION_INDEX[action])
        self.replay_buffer.append((features, chosen_action, reward, next_features, done))

        if len(self.replay_buffer) >= max(8, self.batch_size // 2):
            self._replay_update()

    def _replay_update(self) -> None:
        batch_size = min(self.batch_size, len(self.replay_buffer))
        transitions = self.random.sample(list(self.replay_buffer), batch_size)

        for features, chosen_action, reward, next_features, done in transitions:
            predicted_value = dot_product(self.online_weights[chosen_action], features)
            target_value = reward
            if not done and next_features is not None:
                target_value += self.gamma * max(self._predict(next_features, use_target=True))

            td_error = max(-6.0, min(6.0, target_value - predicted_value))
            weights = self.online_weights[chosen_action]
            for index, feature_value in enumerate(features):
                weights[index] += self.learning_rate * td_error * feature_value

        self.training_steps += 1
        if self.training_steps % self.target_sync_interval == 0:
            self.target_weights = copy_matrix(self.online_weights)

    def end_episode(self, episode_reward: float) -> None:
        if self.training_mode:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if self.online_weights:
                self.target_weights = copy_matrix(self.online_weights)
