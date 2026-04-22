"""
Planner Agent

Reads observer_output.json and generates 2-3 structured optimization
strategies using the Google Gemini API with Pydantic schema enforcement

Input:  data/observer_output.json  (produced by observer.py)
Output: data/planner_output.json
"""

import json
import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# 1. Pydantic schemas 

class Strategy(BaseModel):
    title: str = Field(
        description="Short name for the strategy, e.g. 'Shift deep work to morning'"
    )
    rationale: str = Field(
        description="Why this strategy addresses a detected inefficiency"
    )
    actions: list[str] = Field(
        description="2-4 concrete steps the user should take to implement this"
    )
    targets_inefficiency: str = Field(
        description="The specific inefficiency from the Observer this strategy fixes"
    )
    predicted_completion_boost: str = Field(
        description="Estimated improvement, e.g. '+12% deep work completion rate'"
    )
    predicted_energy_impact: str = Field(
        description="Expected effect on energy alignment, e.g. 'Reduces low-energy deep work by ~3 sessions/week'"
    )

class PlannerOutput(BaseModel):
    strategies: list[Strategy] = Field(
        description="2-3 optimization strategies ranked by expected impact"
    )
    prioritized_inefficiency: str = Field(
        description="The single most critical inefficiency to fix first"
    )


# 2. Load Observer output 

def load_observer_output(filepath: str = "data/observer_output.json") -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


# 3. Call Gemini to generate strategies 

def generate_strategies(observer_output: dict) -> PlannerOutput:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
            You are the Planner Agent in a Personal Productivity Optimization system.
            The Observer Agent has analyzed the user's productivity logs and identified
            the following insights and inefficiencies.

            Your job is to generate 2-3 concrete, actionable optimization strategies that
            directly address the inefficiencies. Each strategy must:
            - Target a specific inefficiency from the Observer's list
            - Include concrete actions the user can take (not vague advice)
            - Estimate the impact on completion rate and energy alignment
            - Be realistic given the data (e.g. don't suggest eliminating all meetings)

            Rank strategies by expected impact — most impactful first.

            OBSERVER INSIGHTS:
            {json.dumps(observer_output["insights"], indent=2)}

            OBSERVER INEFFICIENCIES:
            {json.dumps(observer_output["inefficiencies"], indent=2)}

            KEY STATS SUMMARY:
            {json.dumps(observer_output["raw_stats"]["summary"], indent=2)}

            DEEP WORK BY TIME SLOT:
            {json.dumps(observer_output["raw_stats"]["deep_work_by_time"], indent=2)}

            COMPLETION BY TIME BUCKET:
            {json.dumps(observer_output["raw_stats"]["by_time_bucket"], indent=2)}

            SCHEDULING OVERLAPS DETECTED: {observer_output["raw_stats"]["overlap_count"]}
                """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlannerOutput,
        ),
    )

    return PlannerOutput.model_validate_json(response.text)


# 4. Main run function 

def run(observer_filepath: str = "data/observer_output.json") -> dict:
    print("[ Planner ] Loading Observer output...")
    observer_output = load_observer_output(observer_filepath)

    print("[ Planner ] Generating strategies via Gemini...")
    output: PlannerOutput = generate_strategies(observer_output)

    result = {
        "prioritized_inefficiency": output.prioritized_inefficiency,
        "strategies": [s.model_dump() for s in output.strategies],
    }

    print("\n[ Planner ] Done.\n")
    print(f"=== PRIORITY TARGET ===")
    print(f"  {result['prioritized_inefficiency']}")

    print("\n=== STRATEGIES ===")
    for i, s in enumerate(result["strategies"], 1):
        print(f"\n  {i}. {s['title']}")
        print(f"     Rationale : {s['rationale']}")
        print(f"     Targets   : {s['targets_inefficiency']}")
        print(f"     Completion: {s['predicted_completion_boost']}")
        print(f"     Energy    : {s['predicted_energy_impact']}")
        print(f"     Actions:")
        for action in s["actions"]:
            print(f"       - {action}")

    return result


# 5. Run standalone 

if __name__ == "__main__":
    output = run("data/observer_output.json")

    with open("data/planner_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[ Planner ] Output saved to data/planner_output.json")