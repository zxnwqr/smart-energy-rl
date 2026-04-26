from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class EnvironmentSettings(BaseModel):
    price_level: Literal["low", "medium", "high"] = "medium"
    temperature_level: Literal["cold", "normal", "hot"] = "normal"
    presence: Literal["home", "away"] = "home"


class StepLog(BaseModel):
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


class SimulationResult(BaseModel):
    algorithm: str
    algorithm_display_name: str
    mode: Literal["simulation", "training"] = "simulation"
    episodes: int
    total_reward: float
    average_reward: float
    energy_cost: float
    average_cost: float
    comfort_score: float
    average_comfort_score: float
    battery_usage: float
    average_battery_usage: float
    energy_consumed: float
    average_energy_consumed: float
    reward_over_episodes: List[float] = Field(default_factory=list)
    temperature_history: List[float] = Field(default_factory=list)
    cost_history: List[float] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    action_log: List[StepLog] = Field(default_factory=list)
    environment_settings: EnvironmentSettings
    behavior_highlights: List[str] = Field(default_factory=list)
    simulation_insights: List[str] = Field(default_factory=list)
    summary: str


class TrainingResult(BaseModel):
    algorithm: str
    algorithm_display_name: str
    episodes: int
    trainable: bool
    message: str
    reward_over_episodes: List[float] = Field(default_factory=list)
    average_reward: float
    best_episode_reward: float


class AlgorithmPerformance(BaseModel):
    algorithm: str
    display_name: str
    average_reward: float
    average_cost: float
    average_comfort_score: float
    average_battery_usage: float
    notes: str = ""


class AlgorithmComparison(BaseModel):
    compared_algorithms: List[AlgorithmPerformance]
    best_algorithm: str
    best_algorithm_display_name: str
    ranking: List[str]
    comparison_summary: str
    pending_algorithms: List[str] = Field(default_factory=list)


class AIExplanation(BaseModel):
    best_algorithm: str
    why_it_performed_best: str
    environment_effect: str
    source: Literal["mock", "groq", "openai"] = "mock"
    prompt_used: str = ""


class TrainRequest(BaseModel):
    algorithm: str
    episodes: int = Field(default=30, ge=1, le=1000)
    seed: Optional[int] = None


class SimulateRequest(BaseModel):
    algorithm: str
    episodes: int = Field(default=1, ge=1, le=100)
    seed: Optional[int] = None


class CompareRequest(BaseModel):
    episodes_per_algorithm: int = Field(default=5, ge=1, le=100)
    training_episodes: Optional[int] = Field(default=None, ge=1, le=1500)
    seed: Optional[int] = None


class ExplainRequest(BaseModel):
    source: Literal["comparison", "simulation"] = "comparison"


class LatestResults(BaseModel):
    latest_training: Optional[TrainingResult] = None
    latest_simulation: Optional[SimulationResult] = None
    latest_comparison: Optional[AlgorithmComparison] = None
    latest_explanation: Optional[AIExplanation] = None
    current_environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    available_algorithms: List[str] = Field(default_factory=list)
    planned_algorithms: List[str] = Field(default_factory=list)
