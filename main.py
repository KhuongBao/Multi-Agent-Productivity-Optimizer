"""
main.py — Productivity Optimization Agent Pipeline

Runs all five agents in sequence:
  Observer → Planner → Simulator → Evaluator → (Adapter)

Usage:
  python main.py                  # full pipeline, no Adapter
  python main.py --with-adapter   # include Adapter with example metrics
"""

import argparse
import json
import os
import sys

from agents import observer, planner, simulator, evaluator, adapter


def check_env():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set.")
        print("Run: export GEMINI_API_KEY=your-key-here")
        sys.exit(1)

def banner(text: str):
    print(f"\n{'─' * 50}\n  {text}\n{'─' * 50}")

def save(data: dict, path: str):
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-adapter", action="store_true", help="Also run the Adapter with example observed metrics")
    parser.add_argument("--csv", default="data/productivity_logs.csv")
    args = parser.parse_args()

    # Observer 
    banner("AGENT 1 — Observer")
    obs = observer.run(filepath=args.csv)
    save(obs, "data/observer_output.json")

    # Planner
    banner("AGENT 2 — Planner")
    plan = planner.run()
    save(plan, "data/planner_output.json")

    # Simulator
    banner("AGENT 3 — Simulator")
    sim = simulator.run()
    save(sim, "data/simulator_output.json")

    # Evaluator
    banner("AGENT 4 — Evaluator")
    ev = evaluator.run()
    save(ev, "data/evaluator_output.json")

    # Adapter (optional)
    if args.with_adapter:
        banner("AGENT 5 — Adapter")
        # Replace these with real observed metrics after trialling the strategy (this is just one example)
        result = adapter.run(
            strategy_title=ev["top_recommendation"],
            actual_metrics={
                "completion_rate": 79.5,
                "avg_energy_alignment": 4.0,
                "deep_work_hours_per_week": 3.6,
                "scheduling_conflicts": 1,
            },
        )
        save(result, "data/adapter_output.json")

    # Final summary 
    print(f"\n{'═' * 50}")
    print("  DONE — TOP RECOMMENDATION")
    print(f"{'═' * 50}")
    print(f"  {ev['top_recommendation']}")
    print(f"\n  {ev['reasoning']}")
    print(f"\n  Quick win: {ev['quick_win']}")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
