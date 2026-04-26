from __future__ import annotations

import random
from typing import Any, Dict

from agents.base_agent import BaseAgent
from agents.rl_utils import (
    ACTION_INDEX,
    action_name,
    discounted_returns,
    greedy_index,
    sample_from_probabilities,
    softmax,
    state_to_key,
)
from models import ACTIONS


class PPOAgent(BaseAgent):
    algorithm_id = "ppo"
    display_name = "PPO Agent"
    trainable = True

    def __init__(
        self,
        seed: int | None = None,
        gamma: float = 0.92,
        actor_learning_rate: float = 0.09,
        critic_learning_rate: float = 0.14,
        clip_ratio: float = 0.2,
        update_epochs: int = 4,
    ) -> None:
        super().__init__(seed=seed)
        self.random = random.Random(seed)
        self.gamma = gamma
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.policy_table: dict[tuple[str, ...], list[float]] = {}
        self.value_table: dict[tuple[str, ...], float] = {}
        self.trajectory: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.trajectory = []

    def _ensure_policy(self, state_key: tuple[str, ...]) -> list[float]:
        if state_key not in self.policy_table:
            self.policy_table[state_key] = [0.0 for _ in ACTIONS]
        return self.policy_table[state_key]

    def act(self, state: Dict[str, str], observation: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        state_key = state_to_key(state)
        logits = self._ensure_policy(state_key)
        probabilities = softmax(logits)
        if self.training_mode:
            action_choice = sample_from_probabilities(probabilities, self.random)
        else:
            action_choice = greedy_index(probabilities)
        return action_name(action_choice), {
            "state_key": state_key,
            "action_index": action_choice,
            "action_probability": max(probabilities[action_choice], 1e-8),
        }

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

        metadata = action_data or {}
        self.trajectory.append(
            {
                "state_key": metadata.get("state_key", state_to_key(state)),
                "action_index": metadata.get("action_index", ACTION_INDEX[action]),
                "action_probability": metadata.get("action_probability", 1.0 / len(ACTIONS)),
                "reward": reward,
            }
        )

    def end_episode(self, episode_reward: float) -> None:
        if not self.training_mode or not self.trajectory:
            return

        rewards = [step["reward"] for step in self.trajectory]
        returns = discounted_returns(rewards, self.gamma)
        advantages: list[float] = []
        for trajectory_step, total_return in zip(self.trajectory, returns):
            state_key = trajectory_step["state_key"]
            current_value = self.value_table.get(state_key, 0.0)
            advantages.append(total_return - current_value)

        advantage_mean = sum(advantages) / len(advantages)
        advantage_variance = sum((value - advantage_mean) ** 2 for value in advantages) / len(advantages)
        advantage_std = max(advantage_variance**0.5, 1e-6)
        normalized_advantages = [(value - advantage_mean) / advantage_std for value in advantages]

        for _ in range(self.update_epochs):
            for trajectory_step, total_return, advantage in zip(self.trajectory, returns, normalized_advantages):
                state_key = trajectory_step["state_key"]
                action_choice = trajectory_step["action_index"]
                old_probability = max(trajectory_step["action_probability"], 1e-8)

                logits = self._ensure_policy(state_key)
                probabilities = softmax(logits)
                new_probability = max(probabilities[action_choice], 1e-8)
                ratio = new_probability / old_probability

                if advantage >= 0:
                    effective_ratio = min(ratio, 1.0 + self.clip_ratio)
                else:
                    effective_ratio = max(ratio, 1.0 - self.clip_ratio)

                policy_scale = effective_ratio * advantage
                for index in range(len(logits)):
                    gradient = (1.0 if index == action_choice else 0.0) - probabilities[index]
                    logits[index] += self.actor_learning_rate * policy_scale * gradient

                current_value = self.value_table.get(state_key, 0.0)
                self.value_table[state_key] = current_value + self.critic_learning_rate * (total_return - current_value)
