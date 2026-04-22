"""
Simulator Agent

Takes each strategy from the Planner and runs a what-if projection
against the historical log data to estimate its real impact

For each strategy it:
  1. Extracts the relevant historical subset (e.g. all deep work tasks
     in the afternoon that would be moved to morning)
  2. Computes a baseline from actual data
  3. Uses Gemini to reason about the projected outcome under the new strategy
  4. Returns a SimulationResult per strategy with before/after numbers

Input:  data/planner_output.json + data/observer_output.json
Output: data/simulator_output.json
"""

import json
import os
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# 1. Pydantic schemas 

class ScenarioMetrics(BaseModel):
    completion_rate: float = Field(description="Projected completion rate as a percentage, e.g. 82.5")
    avg_energy_alignment: float = Field(description="Projected average energy level during tasks (1-5 scale)")
    deep_work_hours_per_week: float = Field(description="Projected focused deep work hours per week")
    scheduling_conflicts: int = Field(description="Projected number of scheduling overlaps per week")

class SimulationResult(BaseModel):
    strategy_title: str = Field(description="Title of the strategy being simulated")
    baseline_metrics: ScenarioMetrics = Field(description="Current metrics derived from historical data")
    projected_metrics: ScenarioMetrics = Field(description="Estimated metrics if strategy is applied")
    key_assumptions: list[str] = Field(description="2-3 assumptions made in this simulation")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")
    confidence_reason: str = Field(description="One sentence explaining the confidence level")
    net_benefit_summary: str = Field(description="One sentence summarising the overall projected gain or loss")

class SimulatorOutput(BaseModel):
    simulations: list[SimulationResult] = Field(
        description="One simulation result per strategy, in the same order as the Planner's strategies"
    )
    best_strategy_title: str = Field(
        description="Title of the strategy with the highest projected net benefit"
    )


# 2. Load inputs 

def load_inputs(observer_path: str = "data/observer_output.json", 
                planner_path:  str = "data/planner_output.json") -> tuple[dict, dict, pd.DataFrame]:
    with open(observer_path) as f:
        observer = json.load(f)
    with open(planner_path) as f:
        planner = json.load(f)

    df = pd.read_csv("data/productivity_logs.csv")
    df["start_dt"] = pd.to_datetime(df["date"] + " " + df["start_time"])
    df["end_dt"]   = pd.to_datetime(df["date"] + " " + df["end_time"])
    df["duration_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60
    df["hour"] = df["start_dt"].dt.hour
    df["completed"] = df["completed"].astype(str).str.lower() == "true"

    return observer, planner, df


# 3. Compute baseline metrics from raw data 

def compute_baseline(df: pd.DataFrame) -> dict:
    """
    Computes the real current-state numbers that the Simulator
    uses as the 'before' side of each what-if comparison.
    """
    total_days = df["date"].nunique()
    weeks = max(total_days / 5, 1)   # approximate working weeks

    # Deep work stats
    dw = df[df["category"] == "deep_work"]
    dw_completion   = round(dw["completed"].mean() * 100, 1)
    dw_hours_week   = round((dw["duration_min"].sum() / 60) / weeks, 1)

    # Energy during deep work
    dw_avg_energy   = round(dw["energy_level"].mean(), 2)

    # Overall completion
    overall_completion = round(df["completed"].mean() * 100, 1)

    # Scheduling overlaps per week
    df_sorted = df.sort_values(["date", "start_dt"]).reset_index(drop=True)
    overlap_count = 0
    for i in range(len(df_sorted) - 1):
        a = df_sorted.iloc[i]
        b = df_sorted.iloc[i + 1]
        if a["date"] == b["date"] and b["start_dt"] < a["end_dt"]:
            overlap_count += 1
    overlaps_per_week = round(overlap_count / weeks, 1)

    # Deep work by time bucket (for Gemini context)
    def time_bucket(h):
        if h < 10:   return "early (8-10am)"
        elif h < 12: return "mid-morning (10-12pm)"
        elif h < 14: return "early afternoon (12-2pm)"
        elif h < 16: return "mid-afternoon (2-4pm)"
        else:        return "late (4pm+)"

    dw = dw.copy()
    dw["time_bucket"] = dw["hour"].apply(time_bucket)
    dw_by_time = (
        dw.groupby("time_bucket")
        .agg(count=("completed", "count"),
             completed=("completed", "sum"),
             avg_energy=("energy_level", "mean"))
        .reset_index()
    )
    dw_by_time["completion_rate"] = (
        dw_by_time["completed"] / dw_by_time["count"] * 100
    ).round(1)
    dw_by_time["avg_energy"] = dw_by_time["avg_energy"].round(2)

    return {
        "overall_completion_rate":    overall_completion,
        "deep_work_completion_rate":  dw_completion,
        "deep_work_avg_energy":       dw_avg_energy,
        "deep_work_hours_per_week":   dw_hours_week,
        "scheduling_overlaps_per_week": overlaps_per_week,
        "deep_work_by_time_slot":     dw_by_time.to_dict(orient="records"),
        "total_weeks_observed":       round(weeks, 1),
    }


# 4. Call Gemini to simulate each strategy 

def simulate_strategies(
    planner_output: dict,
    observer_output: dict,
    baseline: dict,
) -> SimulatorOutput:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    strategies_text = json.dumps(planner_output["strategies"], indent=2)
    baseline_text   = json.dumps(baseline, indent=2)
    stats_text      = json.dumps(observer_output["raw_stats"]["by_time_bucket"], indent=2)

    prompt = f"""
    You are the Simulator Agent in a Personal Productivity Optimization system.
    Your job is to run what-if projections for each strategy proposed by the
    Planner Agent, using real historical baseline data.

    For EACH strategy:
    - Use the baseline metrics as the "before" state
    - Project realistic "after" metrics if the strategy were fully applied
    - Be data-driven: reference the time-slot and energy data when estimating
    - Be conservative — don't overestimate improvements
    - State your key assumptions clearly

    The four projected metrics to estimate for each strategy:
    1. completion_rate (%)
    2. avg_energy_alignment (1-5 scale)
    3. deep_work_hours_per_week
    4. scheduling_conflicts (count per week)

    For the baseline_metrics fields, use the real numbers provided below.
    For projected_metrics, reason about what would change under the strategy.

    ---
    BASELINE (current state from historical data):
    {baseline_text}

    COMPLETION RATE BY TIME BUCKET (all tasks):
    {stats_text}

    STRATEGIES TO SIMULATE:
    {strategies_text}

    OBSERVER INEFFICIENCIES CONTEXT:
    {json.dumps(observer_output["inefficiencies"], indent=2)}
    ---

    Return one SimulationResult per strategy, in the same order.
    Also identify which strategy has the highest net projected benefit.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SimulatorOutput,
        ),
    )

    return SimulatorOutput.model_validate_json(response.text)


# 5. Main run function 

def run(observer_path: str = "data/observer_output.json", planner_path:  str = "data/planner_output.json") -> dict:
    print("[ Simulator ] Loading inputs...")
    observer, planner, df = load_inputs(observer_path, planner_path)

    print("[ Simulator ] Computing baseline metrics from raw data...")
    baseline = compute_baseline(df)

    print("[ Simulator ] Running what-if simulations via Gemini...")
    output: SimulatorOutput = simulate_strategies(planner, observer, baseline)

    result = {
        "baseline": baseline,
        "simulations": [s.model_dump() for s in output.simulations],
        "best_strategy_title": output.best_strategy_title,
    }

    print("\n[ Simulator ] Done.\n")
    print(f"=== BEST STRATEGY === {result['best_strategy_title']}\n")

    for sim in result["simulations"]:
        b = sim["baseline_metrics"]
        p = sim["projected_metrics"]
        print(f"  Strategy : {sim['strategy_title']}")
        print(f"  Completion rate  : {b['completion_rate']}%  →  {p['completion_rate']}%")
        print(f"  Avg energy       : {b['avg_energy_alignment']}  →  {p['avg_energy_alignment']}")
        print(f"  Deep work hrs/wk : {b['deep_work_hours_per_week']}  →  {p['deep_work_hours_per_week']}")
        print(f"  Conflicts/wk     : {b['scheduling_conflicts']}  →  {p['scheduling_conflicts']}")
        print(f"  Confidence       : {sim['confidence']} — {sim['confidence_reason']}")
        print(f"  Summary          : {sim['net_benefit_summary']}")
        print(f"  Assumptions      :")
        for a in sim["key_assumptions"]:
            print(f"    - {a}")
        print()

    return result


# 6. Run standalone 

if __name__ == "__main__":
    output = run()

    with open("data/simulator_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[ Simulator ] Output saved to data/simulator_output.json")