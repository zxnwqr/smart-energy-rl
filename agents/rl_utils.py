from __future__ import annotations

import math
import random
from typing import Any, Dict, Sequence

from models import ACTIONS, BATTERY_CATEGORIES, PRICE_CATEGORIES, TEMPERATURE_CATEGORIES, TIME_OF_DAY_CATEGORIES

ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def state_to_key(state: Dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        state["room_temperature_category"],
        state["electricity_price_category"],
        state["human_presence"],
        state["battery_level_category"],
        state["time_of_day"],
    )


def ensure_q_row(table: Dict[tuple[str, ...], list[float]], state_key: tuple[str, ...]) -> list[float]:
    if state_key not in table:
        table[state_key] = [0.0 for _ in ACTIONS]
    return table[state_key]


def action_index(action: str) -> int:
    return ACTION_INDEX[action]


def action_name(index: int) -> str:
    return ACTIONS[index]


def greedy_index(values: Sequence[float]) -> int:
    best_value = max(values)
    for index, value in enumerate(values):
        if value == best_value:
            return index
    return 0


def epsilon_greedy_index(values: Sequence[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(values))

    best_value = max(values)
    candidates = [index for index, value in enumerate(values) if value == best_value]
    return rng.choice(candidates)


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def copy_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running_total = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_total = rewards[index] + gamma * running_total
        returns[index] = running_total
    return returns


def softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exponents = [math.exp(max(-20.0, min(20.0, logit - max_logit))) for logit in logits]
    total = sum(exponents)
    if total == 0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exponents]


def sample_from_probabilities(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def one_hot(value: str, categories: Sequence[str]) -> list[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def encode_state_features(state: Dict[str, str], observation: Dict[str, Any]) -> list[float]:
    devices = observation["devices"]
    features = [1.0]
    features.extend(one_hot(state["room_temperature_category"], TEMPERATURE_CATEGORIES))
    features.extend(one_hot(state["electricity_price_category"], PRICE_CATEGORIES))
    features.extend([1.0 if state["human_presence"] == "home" else 0.0, 1.0 if state["human_presence"] == "away" else 0.0])
    features.extend(one_hot(state["battery_level_category"], BATTERY_CATEGORIES))
    features.extend(one_hot(state["time_of_day"], TIME_OF_DAY_CATEGORIES))
    features.extend(
        [
            observation["current_temperature"] / 35.0,
            (observation["outside_temperature"] + 5.0) / 40.0,
            observation["electricity_price"] / 0.4,
            observation["battery_level"] / 100.0,
            observation["hour"] / 24.0,
            1.0 if devices["light_on"] else 0.0,
            1.0 if devices["heater_on"] else 0.0,
            1.0 if devices["cooler_on"] else 0.0,
            1.0 if devices["battery_charging"] else 0.0,
            1.0 if devices["using_battery"] else 0.0,
        ]
    )
    return features
