"""
Adapter Agent

Compares predicted vs actual outcomes, learns from the gap
and updates scoring weights for the next cycle

Input:  data/evaluator_output.json, data/simulator_output.json
Output: data/adapter_output.json, data/adapter_memory.json
"""

import json
import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MEMORY_PATH = "data/adapter_memory.json"

DEFAULT_WEIGHTS = {
    "completion_rate_gain":  0.35,
    "energy_alignment_gain": 0.25,
    "deep_work_hours_gain":  0.25,
    "conflict_reduction":    0.15,
}


# Pydantic schema 

class ScoringWeights(BaseModel):
    completion_rate_gain: float = Field(description="Weight for completion rate gain")
    energy_alignment_gain: float = Field(description="Weight for energy alignment gain")
    deep_work_hours_gain: float = Field(description="Weight for deep work hours gain")
    conflict_reduction: float = Field(description="Weight for conflict reduction")

class AdapterOutput(BaseModel):
    lessons_learned: list[str] = Field(description="2-3 specific lessons for the next cycle")
    updated_weights: ScoringWeights = Field(description="Adjusted scoring weights, must still sum to 1.0")
    next_cycle_focus: str = Field(description="The inefficiency to prioritise next cycle")
    adaptation_summary: str = Field(description="2-sentence plain-English summary of what changed and why")


# Memory helpers 

def load_memory() -> dict:
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH) as f:
            return json.load(f)
    return {"cycles": [], "current_weights": DEFAULT_WEIGHTS.copy(), "cumulative_lessons": []}

def save_memory(memory: dict):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


#  Main run 

def run(actual_metrics: dict, strategy_title: str) -> dict:
    """
    actual_metrics: dict with keys completion_rate, avg_energy_alignment,
                    deep_work_hours_per_week, scheduling_conflicts
    strategy_title: which strategy was trialled
    """
    memory = load_memory()

    with open("data/simulator_output.json") as f:
        simulator = json.load(f)

    # Find predicted metrics the strategy 
    projected = next(
        (s["projected_metrics"] for s in simulator["simulations"]
         if s["strategy_title"] == strategy_title),
        simulator["simulations"][0]["projected_metrics"]
    )

    # Compute deltas (predicted - actual)
    deltas = {k: round(projected[k] - actual_metrics[k], 2) for k in actual_metrics}

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
        You are the Adapter Agent in a productivity optimization system.
        Compare what was predicted vs what actually happened, then update the system.

        STRATEGY TRIALLED: {strategy_title}
        PREDICTED: {json.dumps(projected, indent=2)}
        ACTUAL:    {json.dumps(actual_metrics, indent=2)}
        DELTAS (predicted - actual): {json.dumps(deltas, indent=2)}

        CURRENT WEIGHTS: {json.dumps(memory["current_weights"], indent=2)}
        PAST LESSONS:    {json.dumps(memory["cumulative_lessons"], indent=2) or "None yet"}

        Tasks:
        1. Write 2-3 specific lessons_learned from the prediction gap.
        2. Suggest updated_weights — only adjust by ±0.05 max, must sum to 1.0.
        3. Set next_cycle_focus to the inefficiency still needing attention.
        4. Write a 2-sentence adaptation_summary for the user.
        """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AdapterOutput,
        ),
    )

    output = AdapterOutput.model_validate_json(response.text)

    # Persist to memory
    memory["cycles"].append({
        "strategy_title": strategy_title,
        "projected": projected,
        "actual": actual_metrics,
        "deltas": deltas,
        "lessons": output.lessons_learned,
    })
    memory["current_weights"] = output.updated_weights.model_dump()
    memory["cumulative_lessons"].extend(output.lessons_learned)
    save_memory(memory)

    result = output.model_dump()

    print("[ Adapter ] Done.\n")
    print("=== LESSONS LEARNED ===")
    for lesson in output.lessons_learned:
        print(f"  - {lesson}")
    print(f"\n=== NEXT CYCLE FOCUS ===\n  {output.next_cycle_focus}")
    print(f"\n=== SUMMARY ===\n  {output.adaptation_summary}")

    return result


if __name__ == "__main__":
    # Example: 
    run(
        strategy_title="Shift deep work to morning slots",
        actual_metrics={
            "completion_rate": 79.5,
            "avg_energy_alignment": 4.0,
            "deep_work_hours_per_week": 3.6,
            "scheduling_conflicts": 1,
        },
    )

    with open("data/adapter_output.json", "w") as f:
        result = load_memory()
        json.dump(result["cycles"][-1], f, indent=2)
    print("\n[ Adapter ] Saved to data/adapter_output.json")
