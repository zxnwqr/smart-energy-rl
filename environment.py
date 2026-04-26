from __future__ import annotations

import math
import random
from typing import Any, Dict, Tuple

from models import ACTIONS, ActionLogRow, DeviceState, EpisodeMetrics, RoomConfig
from pydantic_models import EnvironmentSettings


class SmartRoomEnvironment:
    def __init__(
        self,
        config: RoomConfig | None = None,
        seed: int | None = None,
        settings: EnvironmentSettings | None = None,
    ) -> None:
        self.config = config or RoomConfig()
        self.seed = seed
        self.random = random.Random(seed)
        self.settings = settings or EnvironmentSettings()

        self.devices = DeviceState()
        self.metrics = EpisodeMetrics()

        self.time_step = 0
        self.current_temperature = self.config.initial_temperature
        self.outside_temperature = 18.0
        self.electricity_price = 0.2
        self.human_presence = "home"
        self.battery_level = self.config.initial_battery_level

        self.temperature_history: list[float] = []
        self.cost_history: list[float] = []
        self.reward_history: list[float] = []
        self.action_log: list[dict[str, Any]] = []

    def reset(self) -> Dict[str, str]:
        self.time_step = 0
        self.metrics = EpisodeMetrics()
        self.devices = DeviceState()
        self.current_temperature = round(self._initial_room_temperature(), 2)
        self.battery_level = round(self.random.uniform(40.0, 70.0), 2)
        self.outside_temperature = self._sample_outside_temperature(hour=0)
        self.electricity_price = self._sample_electricity_price(hour=0)
        self.human_presence = self._sample_human_presence(hour=0)

        self.temperature_history = []
        self.cost_history = []
        self.reward_history = []
        self.action_log = []

        return self.get_state()

    def get_time_of_day(self, hour: int | None = None) -> str:
        current_hour = self.time_step if hour is None else hour
        normalized_hour = current_hour % 24
        if normalized_hour < 6:
            return "night"
        if normalized_hour < 12:
            return "morning"
        if normalized_hour < 18:
            return "afternoon"
        return "evening"

    def get_state(self) -> Dict[str, str]:
        return {
            "room_temperature_category": self._temperature_category(self.current_temperature),
            "electricity_price_category": self._price_category(self.electricity_price),
            "human_presence": self.human_presence,
            "battery_level_category": self._battery_category(self.battery_level),
            "time_of_day": self.get_time_of_day(),
        }

    def get_state_key(self) -> Tuple[str, str, str, str, str]:
        state = self.get_state()
        return (
            state["room_temperature_category"],
            state["electricity_price_category"],
            state["human_presence"],
            state["battery_level_category"],
            state["time_of_day"],
        )

    def get_observation(self) -> Dict[str, Any]:
        return {
            "hour": self.time_step,
            "time_label": f"{self.time_step:02d}:00",
            "time_of_day": self.get_time_of_day(),
            "current_temperature": round(self.current_temperature, 2),
            "outside_temperature": round(self.outside_temperature, 2),
            "electricity_price": round(self.electricity_price, 3),
            "human_presence": self.human_presence,
            "battery_level": round(self.battery_level, 2),
            "devices": self.devices.as_dict(),
            "environment_settings": self.settings.model_dump(),
        }

    def get_episode_snapshot(self) -> Dict[str, Any]:
        return {
            "metrics": self.get_metrics(),
            "temperature_history": self.temperature_history,
            "cost_history": self.cost_history,
            "reward_history": self.reward_history,
            "action_log": self.action_log,
            "environment_settings": self.settings.model_dump(),
            "behavior_highlights": self._generate_behavior_highlights(),
        }

    def get_metrics(self) -> Dict[str, float]:
        return {
            "total_reward": round(self.metrics.total_reward, 3),
            "energy_cost": round(self.metrics.total_cost, 3),
            "comfort_score": round(self.metrics.comfort_average, 3),
            "battery_usage": round(self.metrics.battery_usage_total, 3),
            "energy_consumed": round(self.metrics.energy_consumed_total, 3),
        }

    def step(self, action: str) -> tuple[Dict[str, str], float, bool, Dict[str, Any]]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        state_before = self.get_state()
        observation_before = self.get_observation()
        price_category = self._price_category(self.electricity_price)

        self.devices.heater_on = action == "heater_on"
        self.devices.cooler_on = action == "cooler_on"
        self.devices.battery_charging = False
        self.devices.using_battery = False

        if action == "toggle_light":
            self.devices.light_on = not self.devices.light_on

        invalid_action = False

        charge_gain = 0.0
        discharge_amount = 0.0
        battery_offset = 0.0

        if action == "charge_battery":
            if self.battery_level >= 99.0:
                invalid_action = True
            else:
                self.devices.battery_charging = True
                charge_gain = min(10.0, 100.0 - self.battery_level)

        if action == "use_battery":
            if self.battery_level <= 5.0:
                invalid_action = True
            else:
                self.devices.using_battery = True
                discharge_amount = min(12.0, self.battery_level)
                battery_offset = round(discharge_amount * 0.07, 3)

        self.battery_level = round(
            max(0.0, min(100.0, self.battery_level + charge_gain - discharge_amount)),
            2,
        )

        self._update_temperature()

        base_energy = 0.2 if self.human_presence == "home" else 0.1
        gross_energy = base_energy
        if self.devices.heater_on:
            gross_energy += 1.6
        if self.devices.cooler_on:
            gross_energy += 1.5
        if self.devices.light_on:
            gross_energy += 0.25
        if self.devices.battery_charging:
            gross_energy += 0.7

        grid_energy = round(max(0.0, gross_energy - battery_offset), 3)
        cost = round(grid_energy * self.electricity_price * self._price_cost_multiplier(), 3)
        comfort_score = self._comfort_score(self.current_temperature, self.human_presence)
        reward, reward_breakdown = self._calculate_reward(
            action=action,
            cost=cost,
            comfort_score=comfort_score,
            price_category=price_category,
            invalid_action=invalid_action,
            battery_offset=battery_offset,
        )

        self.metrics.total_reward += reward
        self.metrics.total_cost += cost
        self.metrics.comfort_total += comfort_score
        self.metrics.battery_usage_total += battery_offset
        self.metrics.energy_consumed_total += grid_energy
        self.metrics.steps += 1

        self.temperature_history.append(round(self.current_temperature, 2))
        self.cost_history.append(cost)
        self.reward_history.append(round(reward, 3))

        log_row = ActionLogRow(
            time=observation_before["time_label"],
            state=state_before,
            action=action,
            reward=round(reward, 3),
            cost=cost,
            comfort_score=comfort_score,
            battery_usage=round(battery_offset, 3),
            current_temperature=round(self.current_temperature, 2),
            outside_temperature=round(self.outside_temperature, 2),
            electricity_price=round(self.electricity_price, 3),
            devices=self.devices.as_dict(),
        )
        self.action_log.append(log_row.as_dict())

        self.time_step += 1
        done = self.time_step >= self.config.episode_length

        if not done:
            self.outside_temperature = self._sample_outside_temperature(hour=self.time_step)
            self.electricity_price = self._sample_electricity_price(hour=self.time_step)
            self.human_presence = self._sample_human_presence(hour=self.time_step)

        info = {
            "state": state_before,
            "observation": observation_before,
            "reward_breakdown": reward_breakdown,
            "metrics": self.get_metrics(),
            "log_entry": log_row.as_dict(),
        }
        return self.get_state(), round(reward, 3), done, info

    def _update_temperature(self) -> None:
        climate_drift_multiplier = {"cold": 1.25, "normal": 1.0, "hot": 1.18}[self.settings.temperature_level]
        drift = 0.18 * climate_drift_multiplier * (self.outside_temperature - self.current_temperature)
        hvac_effect = 0.0
        if self.devices.heater_on:
            hvac_effect += 1.6 if self.settings.temperature_level == "cold" else 1.4
        if self.devices.cooler_on:
            hvac_effect -= 1.6 if self.settings.temperature_level == "hot" else 1.4

        occupancy_heat = 0.2 if self.human_presence == "home" else -0.05
        noise = self.random.uniform(-0.25, 0.25)

        self.current_temperature = round(
            max(12.0, min(34.0, self.current_temperature + drift + hvac_effect + occupancy_heat + noise)),
            2,
        )

    def _calculate_reward(
        self,
        action: str,
        cost: float,
        comfort_score: float,
        price_category: str,
        invalid_action: bool,
        battery_offset: float,
    ) -> tuple[float, Dict[str, float]]:
        comfort_weight = 1.0 if self.human_presence == "home" else 0.45
        if self.settings.temperature_level == "cold" and self.human_presence == "home":
            comfort_weight += 0.2

        comfort_reward = 0.0
        if self.human_presence == "home":
            if self.config.comfort_min_temp <= self.current_temperature <= self.config.comfort_max_temp:
                comfort_reward = 3.0
            elif 19.0 <= self.current_temperature <= 26.0:
                comfort_reward = 1.0
            else:
                comfort_reward = -2.0 - 0.2 * abs(self.current_temperature - 22.5)
        elif self.current_temperature < 16.0 or self.current_temperature > 30.0:
            comfort_reward = -1.0

        cost_penalty = cost * self._reward_cost_multiplier()

        waste_penalty = 0.0
        if self.human_presence == "away" and (self.devices.heater_on or self.devices.cooler_on):
            waste_penalty += 1.15
        if self.human_presence == "away" and self.devices.light_on:
            waste_penalty += 0.65
        if action == "charge_battery" and price_category == "high":
            waste_penalty += 0.7

        temperature_penalty = 0.0
        if self.current_temperature < 18.0 or self.current_temperature > 27.0:
            temperature_penalty += 1.2 + 0.1 * abs(self.current_temperature - 22.5)
        if self.settings.temperature_level == "cold" and self.current_temperature < self.config.comfort_min_temp:
            temperature_penalty += 0.45
        if self.settings.temperature_level == "hot" and self.current_temperature > self.config.comfort_max_temp:
            temperature_penalty += 0.45

        battery_bonus = 0.0
        if action == "use_battery" and battery_offset > 0 and price_category == "high":
            battery_bonus = 1.3

        invalid_action_penalty = 0.8 if invalid_action else 0.0

        reward = (
            comfort_weight * comfort_reward
            + battery_bonus
            - cost_penalty
            - waste_penalty
            - temperature_penalty
            - invalid_action_penalty
        )

        return round(reward, 3), {
            "comfort_reward": round(comfort_weight * comfort_reward, 3),
            "cost_penalty": round(cost_penalty, 3),
            "waste_penalty": round(waste_penalty, 3),
            "temperature_penalty": round(temperature_penalty, 3),
            "battery_bonus": round(battery_bonus, 3),
            "invalid_action_penalty": round(invalid_action_penalty, 3),
            "comfort_score": round(comfort_score, 3),
        }

    def _comfort_score(self, temperature: float, presence: str) -> float:
        ideal_temperature = 22.5 if presence == "home" else 21.0
        spread = 4.5 if presence == "home" else 7.0
        score = max(0.0, 1.0 - abs(temperature - ideal_temperature) / spread)
        return round(score, 3)

    def _sample_outside_temperature(self, hour: int) -> float:
        baseline_map = {"cold": 5.0, "normal": 18.0, "hot": 29.0}
        amplitude_map = {"cold": 4.0, "normal": 6.0, "hot": 5.0}
        baseline = baseline_map[self.settings.temperature_level] + amplitude_map[self.settings.temperature_level] * math.sin(
            (hour - 6) * math.pi / 12.0
        )
        return round(baseline + self.random.uniform(-1.4, 1.4), 2)

    def _sample_electricity_price(self, hour: int) -> float:
        if self.settings.price_level == "low":
            if 0 <= hour < 6:
                base = 0.08
            elif 6 <= hour < 12:
                base = 0.12
            elif 12 <= hour < 18:
                base = 0.11
            else:
                base = 0.16
        elif self.settings.price_level == "high":
            if 0 <= hour < 6:
                base = 0.18
            elif 6 <= hour < 12:
                base = 0.29
            elif 12 <= hour < 18:
                base = 0.26
            else:
                base = 0.38
        else:
            if 0 <= hour < 6:
                base = 0.11
            elif 6 <= hour < 12:
                base = 0.19
            elif 12 <= hour < 18:
                base = 0.16
            else:
                base = 0.32
        return round(base + self.random.uniform(-0.025, 0.03), 3)

    def _sample_human_presence(self, hour: int) -> str:
        if self.settings.presence == "home":
            return "home"
        if self.settings.presence == "away":
            return "away"
        if 0 <= hour < 7:
            probability_home = 0.95
        elif 7 <= hour < 9:
            probability_home = 0.7
        elif 9 <= hour < 17:
            probability_home = 0.15
        elif 17 <= hour < 22:
            probability_home = 0.9
        else:
            probability_home = 0.98
        return "home" if self.random.random() < probability_home else "away"

    def _temperature_category(self, temperature: float) -> str:
        if temperature < 19.0:
            return "cold"
        if temperature < 21.0:
            return "cool"
        if temperature <= 24.0:
            return "comfortable"
        if temperature <= 26.5:
            return "warm"
        return "hot"

    def _price_category(self, price: float) -> str:
        if price < 0.15:
            return "low"
        if price < 0.27:
            return "medium"
        return "high"

    def _battery_category(self, battery_level: float) -> str:
        if battery_level < 30.0:
            return "low"
        if battery_level < 70.0:
            return "medium"
        return "high"

    def _price_cost_multiplier(self) -> float:
        return {"low": 0.85, "medium": 1.0, "high": 1.2}[self.settings.price_level]

    def _reward_cost_multiplier(self) -> float:
        base = {"low": 4.2, "medium": 5.0, "high": 6.6}[self.settings.price_level]
        if self.human_presence == "away":
            base += 0.8
        return base

    def _initial_room_temperature(self) -> float:
        if self.settings.temperature_level == "cold":
            return self.random.uniform(17.8, 21.3)
        if self.settings.temperature_level == "hot":
            return self.random.uniform(23.5, 27.4)
        return self.random.uniform(20.5, 23.5)

    def _generate_behavior_highlights(self) -> list[str]:
        highlights: list[str] = []
        if self.settings.price_level == "high":
            highlights.append("High electricity price increases the penalty for heavy energy use.")
        elif self.settings.price_level == "low":
            highlights.append("Low electricity price makes energy usage cheaper, so battery charging is safer.")

        if self.settings.temperature_level == "cold":
            highlights.append("Cold outside temperature creates stronger heat loss, so heating decisions matter more.")
        elif self.settings.temperature_level == "hot":
            highlights.append("Hot outside temperature pushes the room upward, so cooling becomes more valuable.")

        if self.settings.presence == "away":
            highlights.append("When the user is away, the reward cares less about comfort and more about avoiding cost.")
        else:
            highlights.append("When the user is home, keeping the room comfortable gives more reward.")

        return highlights
