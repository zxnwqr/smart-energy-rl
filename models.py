from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

ACTIONS: List[str] = [
    "do_nothing",
    "heater_on",
    "cooler_on",
    "toggle_light",
    "charge_battery",
    "use_battery",
]

TEMPERATURE_CATEGORIES: List[str] = ["cold", "cool", "comfortable", "warm", "hot"]
PRICE_CATEGORIES: List[str] = ["low", "medium", "high"]
BATTERY_CATEGORIES: List[str] = ["low", "medium", "high"]
TIME_OF_DAY_CATEGORIES: List[str] = ["night", "morning", "afternoon", "evening"]


@dataclass
class RoomConfig:
    episode_length: int = 24
    comfort_min_temp: float = 21.0
    comfort_max_temp: float = 24.0
    initial_temperature: float = 22.0
    initial_battery_level: float = 55.0


@dataclass
class DeviceState:
    heater_on: bool = False
    cooler_on: bool = False
    light_on: bool = False
    battery_charging: bool = False
    using_battery: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "heater_on": self.heater_on,
            "cooler_on": self.cooler_on,
            "light_on": self.light_on,
            "battery_charging": self.battery_charging,
            "using_battery": self.using_battery,
        }


@dataclass
class EpisodeMetrics:
    total_reward: float = 0.0
    total_cost: float = 0.0
    comfort_total: float = 0.0
    battery_usage_total: float = 0.0
    energy_consumed_total: float = 0.0
    steps: int = 0

    @property
    def comfort_average(self) -> float:
        if self.steps == 0:
            return 0.0
        return round(self.comfort_total / self.steps, 3)


@dataclass
class ActionLogRow:
    time: str
    state: Dict[str, Any]
    action: str
    reward: float
    cost: float
    comfort_score: float
    battery_usage: float
    current_temperature: float
    outside_temperature: float
    electricity_price: float
    devices: Dict[str, bool]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "cost": self.cost,
            "comfort_score": self.comfort_score,
            "battery_usage": self.battery_usage,
            "current_temperature": self.current_temperature,
            "outside_temperature": self.outside_temperature,
            "electricity_price": self.electricity_price,
            "devices": self.devices,
        }
