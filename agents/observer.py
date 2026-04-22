"""
Observer Agent

Reads productivity_logs.csv, computes statistics, and uses the
Google Gemini API to generate structured insights via Pydantic schema

Output: dict with keys:
  - raw_stats      : computed numbers (for downstream agents)
  - insights       : list of finding strings (from LLM)
  - inefficiencies : list of specific problems for the Planner
"""

import json
import os
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# 1. Pydantic schema for Gemini structured output

class ObserverOutput(BaseModel):
    insights: list[str] = Field(
        description="4-6 concise findings referencing actual numbers from the stats"
    )
    inefficiencies: list[str] = Field(
        description="3-5 specific problems the Planner agent should address"
    )


# 2. Load & clean data 

def load_logs(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["start_dt"] = pd.to_datetime(df["date"] + " " + df["start_time"])
    df["end_dt"]   = pd.to_datetime(df["date"] + " " + df["end_time"])
    df["duration_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60
    df["hour"] = df["start_dt"].dt.hour
    df["completed"] = df["completed"].astype(str).str.lower() == "true"

    return df


# 3. Compute statistics 

def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}

    # Completion rate + energy by category
    cat_stats = (
        df.groupby("category")
        .agg(
            total_tasks=("completed", "count"),
            completed_tasks=("completed", "sum"),
            avg_energy=("energy_level", "mean"),
            total_hours=("duration_min", lambda x: round(x.sum() / 60, 1)),
        )
        .reset_index()
    )
    cat_stats["completion_rate"] = (
        cat_stats["completed_tasks"] / cat_stats["total_tasks"] * 100
    ).round(1)
    cat_stats["avg_energy"] = cat_stats["avg_energy"].round(2)
    stats["by_category"] = cat_stats.to_dict(orient="records")

    # Completion rate by time of day bucket
    def time_bucket(hour):
        if hour < 10:
            return "early (8-10am)"
        elif hour < 12:
            return "mid-morning (10-12pm)"
        elif hour < 14:
            return "early afternoon (12-2pm)"
        elif hour < 16:
            return "mid-afternoon (2-4pm)"
        else:
            return "late (4pm+)"

    df["time_bucket"] = df["hour"].apply(time_bucket)
    time_stats = (
        df.groupby("time_bucket")
        .agg(
            total_tasks=("completed", "count"),
            completed_tasks=("completed", "sum"),
            avg_energy=("energy_level", "mean"),
        )
        .reset_index()
    )
    time_stats["completion_rate"] = (
        time_stats["completed_tasks"] / time_stats["total_tasks"] * 100
    ).round(1)
    time_stats["avg_energy"] = time_stats["avg_energy"].round(2)
    stats["by_time_bucket"] = time_stats.to_dict(orient="records")

    # Deep work by time slot
    dw = df[df["category"] == "deep_work"].copy()
    dw["time_bucket"] = dw["hour"].apply(time_bucket)
    dw_time = (
        dw.groupby("time_bucket")
        .agg(
            total=("completed", "count"),
            completed=("completed", "sum"),
            avg_energy=("energy_level", "mean"),
        )
        .reset_index()
    )
    dw_time["completion_rate"] = (dw_time["completed"] / dw_time["total"] * 100).round(1)
    dw_time["avg_energy"] = dw_time["avg_energy"].round(2)
    stats["deep_work_by_time"] = dw_time.to_dict(orient="records")

    # Energy level vs completion rate
    energy_stats = (
        df.groupby("energy_level")
        .agg(
            total=("completed", "count"),
            completed=("completed", "sum"),
        )
        .reset_index()
    )
    energy_stats["completion_rate"] = (
        energy_stats["completed"] / energy_stats["total"] * 100
    ).round(1)
    stats["by_energy_level"] = energy_stats.to_dict(orient="records")

    # Scheduling overlaps 
    df_sorted = df.sort_values(["date", "start_dt"]).reset_index(drop=True)
    overlaps = []
    for i in range(len(df_sorted) - 1):
        a = df_sorted.iloc[i]
        b = df_sorted.iloc[i + 1]
        if a["date"] == b["date"] and b["start_dt"] < a["end_dt"]:
            overlaps.append({
                "date": a["date"],
                "task_a": a["task_name"],
                "task_b": b["task_name"],
                "overlap_min": round(
                    (a["end_dt"] - b["start_dt"]).total_seconds() / 60
                ),
            })
    stats["scheduling_overlaps"] = overlaps
    stats["overlap_count"] = len(overlaps)

    # Deep work sessions with energy <= 2
    low_energy_dw = df[
        (df["category"] == "deep_work") & (df["energy_level"] <= 2)
    ][["date", "task_name", "energy_level", "hour", "completed"]].to_dict(orient="records")
    stats["low_energy_deep_work"] = low_energy_dw

    # Overall summary
    stats["summary"] = {
        "total_tasks": len(df),
        "overall_completion_rate": round(df["completed"].mean() * 100, 1),
        "avg_energy": round(df["energy_level"].mean(), 2),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "total_hours_logged": round(df["duration_min"].sum() / 60, 1),
    }

    return stats


# 4. Call Gemini to generate insights 

def generate_insights(stats: dict) -> ObserverOutput:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
            You are the Observer Agent in a Personal Productivity Optimization system.
            You have analyzed a user's productivity logs. Based on the computed statistics
            below, identify the most important inefficiencies and patterns.

            For insights: write 4-6 concise, specific findings that reference actual numbers.
            For inefficiencies: write 3-5 specific problems the Planner agent should address.
            Name the category, time slot, or pattern — be concrete.

            STATISTICS:
            {json.dumps(stats, indent=2)}
            """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ObserverOutput,
        ),
    )

    return ObserverOutput.model_validate_json(response.text)


# 5. Main run function 

def run(filepath: str = "data/productivity_logs.csv") -> dict:
    print("[ Observer ] Loading logs...")
    df = load_logs(filepath)

    print("[ Observer ] Computing statistics...")
    stats = compute_stats(df)

    print("[ Observer ] Generating insights via Gemini...")
    output: ObserverOutput = generate_insights(stats)

    result = {
        "raw_stats": stats,
        "insights": output.insights,
        "inefficiencies": output.inefficiencies,
    }

    print("\n[ Observer ] Done.\n")
    print("=== INSIGHTS ===")
    for i, insight in enumerate(result["insights"], 1):
        print(f"  {i}. {insight}")

    print("\n=== INEFFICIENCIES ===")
    for i, item in enumerate(result["inefficiencies"], 1):
        print(f"  {i}. {item}")

    return result


# 6. Run standalone 

if __name__ == "__main__":
    output = run("data/productivity_logs.csv")

    with open("data/observer_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[ Observer ] Output saved to data/observer_output.json")