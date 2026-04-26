from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    algorithm_id = "rule_based"
    display_name = "Rule-Based Agent"
    trainable = False

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select_action(self, state: Dict[str, str], observation: Dict[str, Any]) -> str:
        temperature = observation["current_temperature"]
        battery_level = observation["battery_level"]
        electricity_price = observation["electricity_price"]
        human_presence = observation["human_presence"]
        time_of_day = observation["time_of_day"]
        light_on = observation["devices"]["light_on"]

        if human_presence == "home" and temperature < 21.0:
            return "heater_on"

        if human_presence == "home" and temperature > 24.0:
            return "cooler_on"

        if human_presence == "away" and light_on:
            return "toggle_light"

        if electricity_price >= 0.28 and battery_level >= 25.0:
            return "use_battery"

        if electricity_price <= 0.15 and battery_level <= 80.0:
            return "charge_battery"

        if human_presence == "home" and time_of_day in {"evening", "night"} and not light_on:
            return "toggle_light"

        return "do_nothing"
