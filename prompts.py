from __future__ import annotations

from pydantic_models import AlgorithmComparison, SimulationResult

EXPLANATION_SYSTEM_PROMPT = """
You are an AI teaching assistant.
Explain reinforcement learning results in simple, clear English for a university presentation.
Focus on which algorithm performed best, why it performed best, and how the environment affected its behavior.
Keep the explanation short and easy to understand.
""".strip()

EXPLANATION_USER_TEMPLATE = """
Project: RL Smart Energy Agent

Task:
Explain the latest experiment result in simple English for a university defense.

Instructions:
- Say which algorithm performed best.
- Explain why it performed best.
- Explain how the environment affected the agent's behavior.
- Use short and clear sentences.
- Keep the explanation at A2-B1 English level.

Data:
{context}
""".strip()


def build_explanation_prompt(
    comparison: AlgorithmComparison | None,
    simulation: SimulationResult | None,
) -> str:
    context = build_explanation_context(comparison=comparison, simulation=simulation)
    return EXPLANATION_USER_TEMPLATE.format(context=context)


def build_explanation_context(
    comparison: AlgorithmComparison | None,
    simulation: SimulationResult | None,
) -> str:
    if comparison is not None:
        ranking = ", ".join(comparison.ranking)
        top_rows = []
        for item in comparison.compared_algorithms[:3]:
            top_rows.append(
                f"{item.display_name}: reward={item.average_reward}, cost={item.average_cost}, "
                f"comfort={item.average_comfort_score}, battery={item.average_battery_usage}"
            )
        environment_context = ""
        if simulation is not None:
            environment_context = (
                f" Latest simulation environment: price={simulation.environment_settings.price_level}, "
                f"outside_temperature={simulation.environment_settings.temperature_level}, "
                f"presence={simulation.environment_settings.presence}. "
                f"Behavior highlights: {' | '.join(simulation.behavior_highlights)}."
            )
        return (
            f"Best algorithm: {comparison.best_algorithm_display_name}. "
            f"Ranking: {ranking}. "
            f"Summary: {comparison.comparison_summary}. "
            f"Top results: {' | '.join(top_rows)}"
            f"{environment_context}"
        )

    if simulation is not None:
        return (
            f"Algorithm: {simulation.algorithm_display_name}. "
            f"Reward: {simulation.total_reward}. "
            f"Cost: {simulation.energy_cost}. "
            f"Comfort: {simulation.comfort_score}. "
            f"Battery usage: {simulation.battery_usage}. "
            f"Episodes: {simulation.episodes}. "
            f"Environment: price={simulation.environment_settings.price_level}, "
            f"outside_temperature={simulation.environment_settings.temperature_level}, "
            f"presence={simulation.environment_settings.presence}. "
            f"Behavior highlights: {' | '.join(simulation.behavior_highlights)}. "
            f"Summary: {simulation.summary}"
        )

    return "No simulation or comparison data is available yet."
