from __future__ import annotations

import json
import os
from urllib import error, request

from pydantic_models import AIExplanation, AlgorithmComparison, SimulationResult
from prompts import build_explanation_prompt


def generate_ai_explanation(
    comparison: AlgorithmComparison | None = None,
    simulation: SimulationResult | None = None,
) -> AIExplanation:
    prompt_text = build_explanation_prompt(comparison=comparison, simulation=simulation)

    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key:
        groq_result = _try_groq_explanation(prompt_text=prompt_text, api_key=groq_api_key)
        if groq_result is not None:
            return groq_result

    if openai_api_key:
        openai_result = _try_openai_explanation(prompt_text=prompt_text, api_key=openai_api_key)
        if openai_result is not None:
            return openai_result

    return _build_mock_explanation(
        comparison=comparison,
        simulation=simulation,
        prompt_text=prompt_text,
    )


def _build_mock_explanation(
    comparison: AlgorithmComparison | None,
    simulation: SimulationResult | None,
    prompt_text: str,
) -> AIExplanation:
    if comparison is not None and comparison.compared_algorithms:
        best = comparison.compared_algorithms[0]
        second = comparison.compared_algorithms[1] if len(comparison.compared_algorithms) > 1 else None
        margin_text = ""
        if second is not None:
            margin = round(best.average_reward - second.average_reward, 3)
            margin_text = f" It finished about {margin} reward points above {second.display_name}."
        environment_note = ""
        environment_strategy = ""
        environment_tradeoff = ""
        if simulation is not None:
            environment = simulation.environment_settings
            environment_note = (
                f" The latest environment was {environment.price_level} price, "
                f"{environment.temperature_level} outside temperature, and {environment.presence} presence."
            )
            environment_strategy = " " + _environment_behavior_line(simulation)
            environment_tradeoff = " " + _environment_tradeoff_line(simulation)

        return AIExplanation(
            best_algorithm=best.display_name,
            why_it_performed_best=(
                f"Interestingly, {best.display_name} performed best."
                f"{margin_text} This happened because it handled the room conditions more consistently than the other agents."
            ),
            environment_effect=(
                f"{environment_note.strip()}" if environment_note else ""
            )
            + (
                f" {environment_strategy.strip()}" if environment_strategy else ""
            )
            + (
                f" {environment_tradeoff.strip()}" if environment_tradeoff else ""
            ),
            source="mock",
            prompt_used=prompt_text,
        )

    if simulation is not None:
        environment = simulation.environment_settings
        return AIExplanation(
            best_algorithm=simulation.algorithm_display_name,
            why_it_performed_best=(
                f"{simulation.algorithm_display_name} handled this run best in the latest simulation view. "
                f"It reached a reward of {simulation.total_reward} while adapting its actions to the selected conditions."
            ),
            environment_effect=(
                f"The environment was set to {environment.price_level} electricity price, "
                f"{environment.temperature_level} outside temperature, and the user was {environment.presence}. "
                f"{_environment_behavior_line(simulation)} {_environment_tradeoff_line(simulation)}"
            ),
            source="mock",
            prompt_used=prompt_text,
        )

    return AIExplanation(
        best_algorithm="No data",
        why_it_performed_best="No simulation results are available yet.",
        environment_effect="Run a simulation or comparison first.",
        source="mock",
        prompt_used=prompt_text,
    )


def _try_groq_explanation(prompt_text: str, api_key: str) -> AIExplanation | None:
    try:
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "Return only JSON with the required explanation fields."},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": 0.3,
        }
        response_data = _post_json(
            url="https://api.groq.com/openai/v1/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        content = response_data["choices"][0]["message"]["content"]
        parsed = _extract_explanation_json(content)
        if parsed is None:
            return None
        return AIExplanation(source="groq", prompt_used=prompt_text, **parsed)
    except Exception:
        return None


def _try_openai_explanation(prompt_text: str, api_key: str) -> AIExplanation | None:
    try:
        payload = {
            "model": "gpt-4.1-mini",
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Return only JSON with the required explanation fields."}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt_text}]},
            ],
        }
        response_data = _post_json(
            url="https://api.openai.com/v1/responses",
            payload=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        content = _extract_openai_response_text(response_data)
        parsed = _extract_explanation_json(content)
        if parsed is None:
            return None
        return AIExplanation(source="openai", prompt_used=prompt_text, **parsed)
    except Exception:
        return None


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request_headers = {
        "Content-Type": "application/json",
        **headers,
    }
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_openai_response_text(response_data: dict) -> str:
    output = response_data.get("output", [])
    collected_parts: list[str] = []
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                collected_parts.append(text)
    return "\n".join(collected_parts)


def _extract_explanation_json(raw_text: str) -> dict | None:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _environment_behavior_line(simulation: SimulationResult) -> str:
    settings = simulation.environment_settings
    parts: list[str] = []
    if settings.price_level == "high":
        parts.append("Because electricity price was high, the agent had more reason to reduce grid usage and use battery support.")
    elif settings.price_level == "low":
        parts.append("Because electricity price was low, charging the battery and using devices was less risky.")

    if settings.temperature_level == "cold":
        parts.append("Because the weather was cold, the agent needed to react faster when the room temperature dropped.")
    elif settings.temperature_level == "hot":
        parts.append("Because the weather was hot, cooling actions became more important.")

    if settings.presence == "away":
        parts.append("Because the user was away, the agent could sacrifice some comfort to save more cost.")
    else:
        parts.append("Because the user was home, comfort remained an important goal.")
    return " ".join(parts)


def _environment_tradeoff_line(simulation: SimulationResult) -> str:
    settings = simulation.environment_settings
    if settings.presence == "away":
        return "In this run, saving electricity was more important than perfect comfort because nobody was in the room."
    if settings.price_level == "high":
        return "In this run, the agent had to protect comfort without using too much expensive electricity."
    return "In this run, the agent balanced comfort and energy cost under the selected room conditions."
