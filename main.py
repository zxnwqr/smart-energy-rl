from __future__ import annotations

from pathlib import Path
from statistics import fmean
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.sarsa_agent import SARSAAgent
from ai_explainer import generate_ai_explanation
from environment import SmartRoomEnvironment
from pydantic_models import (
    AIExplanation,
    AlgorithmComparison,
    AlgorithmPerformance,
    CompareRequest,
    EnvironmentSettings,
    ExplainRequest,
    LatestResults,
    SimulateRequest,
    SimulationResult,
    StepLog,
    TrainRequest,
    TrainingResult,
)

app = FastAPI(
    title="RL Smart Energy Agent",
    description="A beginner-friendly smart room reinforcement learning project.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ALGORITHM_ORDER = ["random", "rule_based", "q_learning", "sarsa", "dqn", "ppo"]

IMPLEMENTED_AGENTS = {
    "random": RandomAgent,
    "rule_based": RuleBasedAgent,
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
}

DISPLAY_NAMES = {
    "random": "Random Agent",
    "rule_based": "Rule-Based Agent",
    "q_learning": "Q-Learning Agent",
    "sarsa": "SARSA Agent",
    "dqn": "DQN Agent",
    "ppo": "PPO Agent",
}

DEFAULT_TRAINING_EPISODES = {
    "random": 30,
    "rule_based": 30,
    "q_learning": 140,
    "sarsa": 150,
    "dqn": 170,
    "ppo": 160,
}

LATEST_RESULTS: Dict[str, Any] = {
    "latest_training": None,
    "latest_simulation": None,
    "latest_comparison": None,
    "latest_explanation": None,
}

TRAINED_AGENT_STORE: Dict[str, Dict[str, Any]] = {}
CURRENT_ENVIRONMENT_SETTINGS = EnvironmentSettings()


def normalize_algorithm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def build_agent(algorithm: str, seed: int | None = None) -> BaseAgent:
    normalized = normalize_algorithm_name(algorithm)
    agent_class = IMPLEMENTED_AGENTS.get(normalized)
    if agent_class is None:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm: {algorithm}")
    return agent_class(seed=seed)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(fmean(values), 3)


def generate_simulation_insights(
    simulation_data: dict[str, Any],
    environment_settings: EnvironmentSettings,
) -> list[str]:
    action_log = simulation_data.get("action_log", [])
    cost_history = simulation_data.get("cost_history", [])
    metrics = simulation_data.get("metrics", {})

    heater_actions = sum(1 for row in action_log if row["action"] == "heater_on")
    cooler_actions = sum(1 for row in action_log if row["action"] == "cooler_on")
    battery_actions = sum(1 for row in action_log if row["action"] == "use_battery")
    charge_actions = sum(1 for row in action_log if row["action"] == "charge_battery")
    high_price_battery_actions = sum(
        1
        for row in action_log
        if row["action"] == "use_battery" and row["state"]["electricity_price_category"] == "high"
    )

    insights: list[str] = []

    if environment_settings.price_level == "high":
        insights.append("Agent reduced energy usage because electricity price was high.")
        if battery_actions > 0:
            insights.append("Agent used battery support more often during expensive electricity hours.")
    elif environment_settings.price_level == "low":
        insights.append("Low electricity price allowed the agent to use devices with less cost pressure.")

    if environment_settings.temperature_level == "cold":
        if heater_actions > 0:
            insights.append("Heater usage increased because the outside temperature was cold.")
        else:
            insights.append("Cold weather made heating more important for room comfort.")
    elif environment_settings.temperature_level == "hot":
        if cooler_actions > 0:
            insights.append("Cooling actions increased because the outside temperature was hot.")
        else:
            insights.append("Hot weather created more pressure to cool the room.")

    if environment_settings.presence == "away":
        insights.append("Because the user was away, the agent focused more on saving cost than perfect comfort.")
    else:
        insights.append("Because the user was home, the agent kept comfort as a higher priority.")

    if high_price_battery_actions > 0:
        insights.append("Battery energy was used during high-price periods to reduce expensive grid usage.")
    elif charge_actions > 0 and environment_settings.price_level == "low":
        insights.append("The agent charged the battery when energy conditions were cheaper.")

    total_cost = metrics.get("energy_cost", 0.0)
    if len(cost_history) >= 2 and cost_history[-1] < cost_history[0]:
        insights.append("Energy cost became lower later in the run than at the beginning.")

    previous_simulation = LATEST_RESULTS.get("latest_simulation")
    if previous_simulation is not None and total_cost < previous_simulation.energy_cost:
        insights.append("Total cost decreased compared to the previous simulation run.")

    deduplicated: list[str] = []
    for insight in insights:
        if insight not in deduplicated:
            deduplicated.append(insight)
    return deduplicated[:5]


def is_trainable(algorithm: str) -> bool:
    normalized = normalize_algorithm_name(algorithm)
    return bool(getattr(IMPLEMENTED_AGENTS[normalized], "trainable", False))


def get_default_training_episodes(algorithm: str) -> int:
    return DEFAULT_TRAINING_EPISODES[normalize_algorithm_name(algorithm)]


def build_seed(
    base_seed: int | None,
    algorithm: str,
    episode_index: int = 0,
    phase: str = "evaluation",
) -> int | None:
    if base_seed is None:
        return None
    normalized = normalize_algorithm_name(algorithm)
    algorithm_offset = ALGORITHM_ORDER.index(normalized) * 1_000
    phase_offset = 10_000 if phase == "training" else 20_000 if phase == "evaluation" else 0
    return base_seed + phase_offset + algorithm_offset + episode_index


def run_episode(
    agent: BaseAgent,
    env_seed: int | None,
    training: bool,
    environment_settings: EnvironmentSettings | None = None,
) -> Dict[str, Any]:
    env = SmartRoomEnvironment(seed=env_seed, settings=environment_settings or CURRENT_ENVIRONMENT_SETTINGS)
    state = env.reset()
    observation = env.get_observation()
    agent.start_episode(training=training)

    queued_action: str | None = None
    queued_action_data: Dict[str, Any] | None = None
    done = False

    while not done:
        if queued_action is None:
            action, action_data = agent.act(state=state, observation=observation)
        else:
            action, action_data = queued_action, queued_action_data or {}
            queued_action = None
            queued_action_data = None

        next_state, reward, done, _ = env.step(action)
        next_observation = None if done else env.get_observation()

        next_action: str | None = None
        next_action_data: Dict[str, Any] | None = None
        if training and agent.needs_next_action and next_observation is not None:
            next_action, next_action_data = agent.act(state=next_state, observation=next_observation)
            queued_action = next_action
            queued_action_data = next_action_data

        agent.observe_transition(
            state=state,
            observation=observation,
            action=action,
            reward=reward,
            next_state=next_state,
            next_observation=next_observation,
            done=done,
            action_data=action_data,
            next_action=next_action,
            next_action_data=next_action_data,
        )

        state = next_state
        if next_observation is not None:
            observation = next_observation

    episode_reward = env.get_metrics()["total_reward"]
    agent.end_episode(episode_reward)
    return env.get_episode_snapshot()


def train_agent_model(
    algorithm: str,
    episodes: int,
    seed: int | None,
    update_latest: bool = False,
) -> TrainingResult:
    normalized = normalize_algorithm_name(algorithm)
    agent = build_agent(normalized, seed=build_seed(seed, normalized, phase="training"))
    reward_over_episodes: list[float] = []

    for episode_index in range(episodes):
        snapshot = run_episode(
            agent=agent,
            env_seed=build_seed(seed, normalized, episode_index=episode_index, phase="training"),
            training=agent.trainable,
            environment_settings=CURRENT_ENVIRONMENT_SETTINGS,
        )
        reward_over_episodes.append(snapshot["metrics"]["total_reward"])

    training_result = TrainingResult(
        algorithm=normalized,
        algorithm_display_name=DISPLAY_NAMES[normalized],
        episodes=episodes,
        trainable=agent.trainable,
        message=(
            f"{DISPLAY_NAMES[normalized]} was trained for {episodes} episode(s) using a lightweight local implementation."
            if agent.trainable
            else (
                f"{DISPLAY_NAMES[normalized]} is a baseline policy, so training runs only collect reference scores."
            )
        ),
        reward_over_episodes=[round(value, 3) for value in reward_over_episodes],
        average_reward=average(reward_over_episodes),
        best_episode_reward=round(max(reward_over_episodes), 3),
    )

    if agent.trainable:
        TRAINED_AGENT_STORE[normalized] = {
            "agent": agent.clone(),
            "episodes": episodes,
            "seed": seed,
            "training_result": training_result,
        }

    if update_latest:
        LATEST_RESULTS["latest_training"] = training_result

    return training_result


def ensure_trained_agent(
    algorithm: str,
    training_episodes: int | None = None,
    seed: int | None = None,
    refresh: bool = False,
) -> Dict[str, Any] | None:
    normalized = normalize_algorithm_name(algorithm)
    if not is_trainable(normalized):
        return None

    episodes_to_use = training_episodes or get_default_training_episodes(normalized)
    existing = TRAINED_AGENT_STORE.get(normalized)
    if not refresh and existing is not None and existing["episodes"] >= episodes_to_use:
        return existing

    train_agent_model(
        algorithm=normalized,
        episodes=episodes_to_use,
        seed=seed,
        update_latest=False,
    )
    return TRAINED_AGENT_STORE.get(normalized)


def evaluate_algorithm(
    algorithm: str,
    episodes: int,
    seed: int | None,
    mode: str = "simulation",
    training_episodes: int | None = None,
    refresh_training: bool = False,
    environment_settings: EnvironmentSettings | None = None,
) -> SimulationResult:
    normalized = normalize_algorithm_name(algorithm)
    trainable = is_trainable(normalized)
    selected_environment = environment_settings or CURRENT_ENVIRONMENT_SETTINGS
    training_entry = ensure_trained_agent(
        algorithm=normalized,
        training_episodes=training_episodes,
        seed=seed,
        refresh=refresh_training,
    )

    if trainable:
        if training_entry is None:
            raise HTTPException(status_code=500, detail=f"Failed to prepare {DISPLAY_NAMES[normalized]}.")
        agent = training_entry["agent"].clone()
        training_note = f"Evaluated after training for {training_entry['episodes']} episode(s)."
    else:
        agent = build_agent(normalized, seed=build_seed(seed, normalized, phase="evaluation"))
        training_note = "No training was needed because this is a baseline policy."

    episode_rewards: list[float] = []
    episode_costs: list[float] = []
    episode_comfort_scores: list[float] = []
    episode_battery_usage: list[float] = []
    episode_energy_consumed: list[float] = []
    final_snapshot: dict[str, Any] | None = None

    for episode_index in range(episodes):
        snapshot = run_episode(
            agent=agent,
            env_seed=build_seed(seed, normalized, episode_index=episode_index, phase="evaluation"),
            training=False,
            environment_settings=selected_environment,
        )
        metrics = snapshot["metrics"]
        final_snapshot = snapshot
        episode_rewards.append(metrics["total_reward"])
        episode_costs.append(metrics["energy_cost"])
        episode_comfort_scores.append(metrics["comfort_score"])
        episode_battery_usage.append(metrics["battery_usage"])
        episode_energy_consumed.append(metrics["energy_consumed"])

    if final_snapshot is None:
        raise HTTPException(status_code=500, detail="The simulation did not produce any data.")

    return SimulationResult(
        algorithm=normalized,
        algorithm_display_name=DISPLAY_NAMES[normalized],
        mode=mode,
        episodes=episodes,
        total_reward=round(episode_rewards[-1], 3),
        average_reward=average(episode_rewards),
        energy_cost=round(episode_costs[-1], 3),
        average_cost=average(episode_costs),
        comfort_score=round(episode_comfort_scores[-1], 3),
        average_comfort_score=average(episode_comfort_scores),
        battery_usage=round(episode_battery_usage[-1], 3),
        average_battery_usage=average(episode_battery_usage),
        energy_consumed=round(episode_energy_consumed[-1], 3),
        average_energy_consumed=average(episode_energy_consumed),
        reward_over_episodes=[round(value, 3) for value in episode_rewards],
        temperature_history=final_snapshot["temperature_history"],
        cost_history=final_snapshot["cost_history"],
        reward_history=final_snapshot["reward_history"],
        action_log=[StepLog(**row) for row in final_snapshot["action_log"]],
        environment_settings=EnvironmentSettings(**final_snapshot["environment_settings"]),
        behavior_highlights=final_snapshot["behavior_highlights"],
        simulation_insights=generate_simulation_insights(
            simulation_data=final_snapshot,
            environment_settings=selected_environment,
        ),
        summary=(
            f"{DISPLAY_NAMES[normalized]} finished {episodes} evaluation episode(s). "
            f"Average reward: {average(episode_rewards)}. "
            f"Average cost: {average(episode_costs)}. "
            f"{training_note} Environment: price={selected_environment.price_level}, "
            f"outside temperature={selected_environment.temperature_level}, presence={selected_environment.presence}."
        ),
    )


@app.get("/", response_class=FileResponse)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/train", response_model=TrainingResult)
def train_algorithm(request: TrainRequest) -> TrainingResult:
    normalized = normalize_algorithm_name(request.algorithm)
    return train_agent_model(
        algorithm=normalized,
        episodes=request.episodes,
        seed=request.seed,
        update_latest=True,
    )


@app.post("/set-environment", response_model=EnvironmentSettings)
def set_environment(settings: EnvironmentSettings) -> EnvironmentSettings:
    global CURRENT_ENVIRONMENT_SETTINGS
    CURRENT_ENVIRONMENT_SETTINGS = settings
    return CURRENT_ENVIRONMENT_SETTINGS


@app.post("/simulate", response_model=SimulationResult)
def simulate_algorithm(request: SimulateRequest) -> SimulationResult:
    result = evaluate_algorithm(
        algorithm=request.algorithm,
        episodes=request.episodes,
        seed=request.seed,
        mode="simulation",
        training_episodes=get_default_training_episodes(request.algorithm) if is_trainable(request.algorithm) else None,
        refresh_training=False,
    )
    LATEST_RESULTS["latest_simulation"] = result
    return result


@app.get("/results", response_model=LatestResults)
def get_results() -> LatestResults:
    return LatestResults(
        latest_training=LATEST_RESULTS["latest_training"],
        latest_simulation=LATEST_RESULTS["latest_simulation"],
        latest_comparison=LATEST_RESULTS["latest_comparison"],
        latest_explanation=LATEST_RESULTS["latest_explanation"],
        current_environment=CURRENT_ENVIRONMENT_SETTINGS,
        available_algorithms=ALGORITHM_ORDER,
        planned_algorithms=[],
    )


@app.post("/compare", response_model=AlgorithmComparison)
def compare_algorithms(request: CompareRequest) -> AlgorithmComparison:
    results: list[AlgorithmPerformance] = []

    for algorithm in ALGORITHM_ORDER:
        if is_trainable(algorithm):
            training_entry = ensure_trained_agent(
                algorithm=algorithm,
                training_episodes=request.training_episodes or get_default_training_episodes(algorithm),
                seed=request.seed,
                refresh=True,
            )
            notes = (
                f"Trained for {training_entry['episodes']} episode(s) before comparison."
                if training_entry is not None
                else "Training failed."
            )
        else:
            notes = "Baseline policy, so no training was needed."

        simulation_result = evaluate_algorithm(
            algorithm=algorithm,
            episodes=request.episodes_per_algorithm,
            seed=request.seed,
            mode="simulation",
            training_episodes=request.training_episodes,
            refresh_training=False,
        )
        results.append(
            AlgorithmPerformance(
                algorithm=algorithm,
                display_name=DISPLAY_NAMES[algorithm],
                average_reward=simulation_result.average_reward,
                average_cost=simulation_result.average_cost,
                average_comfort_score=simulation_result.average_comfort_score,
                average_battery_usage=simulation_result.average_battery_usage,
                notes=notes,
            )
        )

    ranked_results = sorted(results, key=lambda item: item.average_reward, reverse=True)
    best_result = ranked_results[0]
    comparison = AlgorithmComparison(
        compared_algorithms=ranked_results,
        best_algorithm=best_result.algorithm,
        best_algorithm_display_name=best_result.display_name,
        ranking=[item.display_name for item in ranked_results],
        comparison_summary=(
            f"{best_result.display_name} achieved the highest average reward in this run. "
            "The comparison used lightweight local training settings so it stays fast on a laptop."
        ),
        pending_algorithms=[],
    )
    LATEST_RESULTS["latest_comparison"] = comparison
    return comparison


@app.post("/explain", response_model=AIExplanation)
def explain_results(request: ExplainRequest) -> AIExplanation:
    comparison = LATEST_RESULTS["latest_comparison"] if request.source == "comparison" else None
    simulation = LATEST_RESULTS["latest_simulation"]

    if request.source == "comparison" and comparison is None:
        raise HTTPException(status_code=400, detail="Run /compare before requesting a comparison explanation.")

    if request.source == "simulation" and simulation is None:
        raise HTTPException(status_code=400, detail="Run /simulate before requesting a simulation explanation.")

    explanation = generate_ai_explanation(comparison=comparison, simulation=simulation)
    LATEST_RESULTS["latest_explanation"] = explanation
    return explanation
