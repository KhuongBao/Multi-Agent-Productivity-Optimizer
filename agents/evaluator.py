"""
Evaluator Agent

Reads simulator_output.json and scores each simulated strategy against
weighted criteria to produce a final ranked recommendation

Scoring criteria (weights sum to 1.0):
(default weights)
  - completion_rate_gain   : 0.35  (biggest driver of productivity)
  - energy_alignment_gain  : 0.25  (sustainability of the change)
  - deep_work_hours_gain   : 0.25  (raw focus time recovered)
  - conflict_reduction     : 0.15  (scheduling quality improvement)

Input:  data/simulator_output.json + data/planner_output.json
Output: data/evaluator_output.json
"""

import json
import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# 1. Pydantic schemas

class CriteriaScores(BaseModel):
    completion_rate_gain: float = Field(
        description="Score 0-10 for projected improvement in completion rate"
    )
    energy_alignment_gain: float = Field(
        description="Score 0-10 for projected improvement in energy alignment"
    )
    deep_work_hours_gain: float = Field(
        description="Score 0-10 for projected increase in deep work hours per week"
    )
    conflict_reduction: float = Field(
        description="Score 0-10 for projected reduction in scheduling conflicts"
    )

class StrategyEvaluation(BaseModel):
    strategy_title: str = Field(description="Title of the strategy being evaluated")
    criteria_scores: CriteriaScores = Field(description="Raw 0-10 scores per criterion")
    weighted_total: float = Field(
        description="Final weighted score out of 10, computed using the defined weights"
    )
    implementation_difficulty: str = Field(
        description="'easy', 'medium', or 'hard' — how disruptive is this to adopt?"
    )
    implementation_difficulty_reason: str = Field(
        description="One sentence explaining the difficulty rating"
    )
    risks: list[str] = Field(description="1-2 risks or caveats for this strategy")
    recommendation: str = Field(
        description="'strongly recommend', 'recommend', or 'consider with caution'"
    )

class EvaluatorOutput(BaseModel):
    evaluations: list[StrategyEvaluation] = Field(
        description="One evaluation per strategy, ranked highest weighted_total first"
    )
    top_recommendation: str = Field(
        description="Title of the single best strategy to implement first"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation of why this is the top pick over the others"
    )
    quick_win: str = Field(
        description="The one specific action the user can take today to start improving"
    )


# 2. Load inputs 

def load_inputs(simulator_path: str = "data/simulator_output.json", planner_path:   str = "data/planner_output.json") -> tuple[dict, dict]:
    with open(simulator_path) as f:
        simulator = json.load(f)
    with open(planner_path) as f:
        planner = json.load(f)
    return simulator, planner


# 3. Compute numeric deltas from simulation results

def compute_deltas(simulations: list[dict]) -> list[dict]:
    """
    Pre-computes the before→after delta for each metric so Gemini
    has clean numbers to score against rather than doing subtraction itself.
    """
    enriched = []
    for sim in simulations:
        b = sim["baseline_metrics"]
        p = sim["projected_metrics"]
        enriched.append({
            "strategy_title": sim["strategy_title"],
            "confidence": sim["confidence"],
            "net_benefit_summary": sim["net_benefit_summary"],
            "key_assumptions": sim["key_assumptions"],
            "deltas": {
                "completion_rate_gain":  round(p["completion_rate"] - b["completion_rate"], 2),
                "energy_alignment_gain": round(p["avg_energy_alignment"] - b["avg_energy_alignment"], 2),
                "deep_work_hours_gain":  round(p["deep_work_hours_per_week"] - b["deep_work_hours_per_week"], 2),
                "conflict_reduction":    round(b["scheduling_conflicts"] - p["scheduling_conflicts"], 2),
            },
            "baseline": b,
            "projected": p,
        })
    return enriched


# 4. Call Gemini to score and rank strategies 

def evaluate_strategies(enriched_simulations: list[dict], planner_output: dict) -> EvaluatorOutput:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Pass the scoring weights explicitly so Gemini applies them correctly
    weights = {
        "completion_rate_gain":  0.35,
        "energy_alignment_gain": 0.25,
        "deep_work_hours_gain":  0.25,
        "conflict_reduction":    0.15,
    }

    prompt = f"""
    You are the Evaluator Agent in a Personal Productivity Optimization system.
    Your job is to score and rank the simulated strategies to produce a final
    recommendation for the user.

    SCORING INSTRUCTIONS:
    Score each criterion from 0-10 based on the delta values below:
    - completion_rate_gain   (weight 0.35): +15% gain = 10, +10% = 7, +5% = 4, <=0% = 0
    - energy_alignment_gain  (weight 0.25): +1.0 gain = 10, +0.5 = 6, +0.2 = 3, <=0 = 0
    - deep_work_hours_gain   (weight 0.25): +2h/wk = 10, +1h = 6, +0.5h = 3, <=0 = 0
    - conflict_reduction     (weight 0.15): -3 conflicts = 10, -1 = 5, -0.5 = 3, <=0 = 0

    weighted_total = sum(score_i * weight_i) for all four criteria.

    Also consider:
    - Confidence level from the Simulator (high/medium/low) — lower confidence
        should nudge implementation_difficulty up and temper your recommendation.
    - Strategies with large gains but low confidence should be 'consider with caution'.
    - Identify 1-2 realistic risks per strategy (e.g. meeting constraints, habit change).

    Return evaluations ranked by weighted_total descending.
    The top_recommendation should be the highest scorer unless confidence is low,
    in which case prefer the second-highest with higher confidence.

    SCORING WEIGHTS:
    {json.dumps(weights, indent=2)}

    SIMULATION RESULTS WITH DELTAS:
    {json.dumps(enriched_simulations, indent=2)}

    ORIGINAL STRATEGIES (for context on actions and rationale):
    {json.dumps(planner_output["strategies"], indent=2)}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluatorOutput,
        ),
    )

    return EvaluatorOutput.model_validate_json(response.text)


# 5. Main run function 

def run(simulator_path: str = "data/simulator_output.json", planner_path:   str = "data/planner_output.json") -> dict:
    print("[ Evaluator ] Loading inputs...")
    simulator, planner = load_inputs(simulator_path, planner_path)

    print("[ Evaluator ] Computing deltas...")
    enriched = compute_deltas(simulator["simulations"])

    print("[ Evaluator ] Scoring and ranking strategies via Gemini...")
    output: EvaluatorOutput = evaluate_strategies(enriched, planner)

    result = {
        "evaluations": [e.model_dump() for e in output.evaluations],
        "top_recommendation": output.top_recommendation,
        "reasoning": output.reasoning,
        "quick_win": output.quick_win,
    }

    print("\n[ Evaluator ] Done.\n")

    print("=== RANKED STRATEGIES ===\n")
    for i, ev in enumerate(result["evaluations"], 1):
        sc = ev["criteria_scores"]
        print(f"  #{i} {ev['strategy_title']}")
        print(f"      Weighted score : {ev['weighted_total']:.2f} / 10")
        print(f"      Criteria scores:")
        print(f"        Completion rate gain  : {sc['completion_rate_gain']:.1f}  (x0.35)")
        print(f"        Energy alignment gain : {sc['energy_alignment_gain']:.1f}  (x0.25)")
        print(f"        Deep work hours gain  : {sc['deep_work_hours_gain']:.1f}  (x0.25)")
        print(f"        Conflict reduction    : {sc['conflict_reduction']:.1f}  (x0.15)")
        print(f"      Difficulty   : {ev['implementation_difficulty']} — {ev['implementation_difficulty_reason']}")
        print(f"      Recommend    : {ev['recommendation']}")
        print(f"      Risks        :")
        for r in ev["risks"]:
            print(f"        - {r}")
        print()

    print(f"=== TOP RECOMMENDATION ===")
    print(f"  {result['top_recommendation']}")
    print(f"\n  {result['reasoning']}")
    print(f"\n=== QUICK WIN ===")
    print(f"  {result['quick_win']}")

    return result


# 6. Run standalone 

if __name__ == "__main__":
    output = run()

    with open("data/evaluator_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[ Evaluator ] Output saved to data/evaluator_output.json")