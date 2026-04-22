"""
Streamlit UI 

User interface for the productivity agent application
Note: Design elements and layout code was helped by Cursor. 
"""

import streamlit as st
import pandas as pd
import json
import os

from agents import observer, planner, simulator, evaluator, adapter


def save(data: dict, path: str):
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


st.set_page_config(page_title="Productivity Loop", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; letter-spacing: 0.02em; }
    div[data-testid="metric-container"] { background: #f8f8f8; border-radius: 8px; padding: 0.75rem 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("Productivity Loop")
st.caption("An observe → plan → simulate → evaluate → adapt cycle that gets smarter every iteration.")

tab1, tab2 = st.tabs(["Pipeline", "Feedback"])

# TAB 1: Pipeline 
with tab1:
    st.markdown("Run the full four-agent pipeline against your historical logs. Takes ~30 seconds.")

    if st.button("Run pipeline", type="primary"):

        # 1. OBSERVER
        st.markdown("### Observer")
        with st.spinner("Scanning logs for patterns…"):
            obs_out = observer.run()
            save(obs_out, "data/observer_output.json")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**What's working**")
            for item in obs_out["insights"]:
                st.info(item, icon="✅")
        with col_b:
            st.markdown("**What isn't**")
            for item in obs_out["inefficiencies"]:
                st.warning(item, icon="⚠️")

        st.divider()

        # 2. PLANNER
        st.markdown("### Planner")
        with st.spinner("Drafting strategies…"):
            plan_out = planner.run()
            save(plan_out, "data/planner_output.json")

        st.markdown(f"**Targeting:** {plan_out['prioritized_inefficiency']}")
        for idx, strat in enumerate(plan_out["strategies"], 1):
            with st.expander(f"{idx}. {strat['title']}"):
                st.write(strat["rationale"])
                for action in strat["actions"]:
                    st.markdown(f"- {action}")

        st.divider()

        # 3. SIMULATOR
        st.markdown("### Simulator")
        with st.spinner("Running what-if projections…"):
            sim_out = simulator.run()
            save(sim_out, "data/simulator_output.json")

        for sim in sim_out["simulations"]:
            b, p = sim["baseline_metrics"], sim["projected_metrics"]
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"**{sim['strategy_title']}**")
                right.markdown(
                    f"`{sim['confidence'].upper()}` confidence — {sim['confidence_reason']}",
                    help=sim["confidence_reason"]
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Completion rate", f"{p['completion_rate']}%",
                          f"{round(p['completion_rate'] - b['completion_rate'], 1)}%")
                m2.metric("Avg energy", p["avg_energy_alignment"],
                          round(p["avg_energy_alignment"] - b["avg_energy_alignment"], 2))
                m3.metric("Deep work / wk", f"{p['deep_work_hours_per_week']}h",
                          round(p["deep_work_hours_per_week"] - b["deep_work_hours_per_week"], 1))
                m4.metric("Conflicts / wk", p["scheduling_conflicts"],
                          p["scheduling_conflicts"] - b["scheduling_conflicts"],
                          delta_color="inverse")

        st.divider()

        # 4. EVALUATOR
        st.markdown("### Evaluator")
        with st.spinner("Scoring and ranking…"):
            eval_out = evaluator.run()
            save(eval_out, "data/evaluator_output.json")

        st.success(f"**Top pick:** {eval_out['top_recommendation']}", icon="🏆")
        st.write(eval_out["reasoning"])
        st.markdown(f"**Quick win:** {eval_out['quick_win']}")

        st.markdown("**Full rankings**")
        for ev in eval_out["evaluations"]:
            with st.expander(f"{ev['weighted_total']}/10 — {ev['strategy_title']}"):
                st.markdown(
                    f"**Difficulty:** {ev['implementation_difficulty'].title()} "
                    f"— {ev['implementation_difficulty_reason']}"
                )
                for risk in ev["risks"]:
                    st.markdown(f"- {risk}")


# TAB 2: Feedback 
with tab2:
    st.markdown(
        "You tried the top strategy for a week. Enter what actually happened "
        "and the Adapter will reconcile the gap and update the system's priors."
    )

    try:
        with open("data/evaluator_output.json") as f:
            top_strat = json.load(f)["top_recommendation"]
    except FileNotFoundError:
        top_strat = ""

    trial_strategy = st.text_input(
        "Strategy you trialled",
        value=top_strat,
        placeholder="Run the pipeline first to auto-fill this"
    )

    st.markdown("**Actual results after one week**")
    c1, c2, c3, c4 = st.columns(4)
    act_comp      = c1.number_input("Completion rate (%)", value=80.0, step=0.5)
    act_energy    = c2.number_input("Avg energy (1-5)",    value=4.0,  step=0.1)
    act_dw        = c3.number_input("Deep work (hrs/wk)",  value=4.0,  step=0.5)
    act_conflicts = c4.number_input("Conflicts / wk",      value=1,    step=1)

    if st.button("Run Adapter", type="primary"):
        actual_metrics = {
            "completion_rate":        act_comp,
            "avg_energy_alignment":   act_energy,
            "deep_work_hours_per_week": act_dw,
            "scheduling_conflicts":   act_conflicts,
        }

        with st.spinner("Reconciling projections vs. reality…"):
            try:
                adapt_out = adapter.run(actual_metrics, trial_strategy)
                save(adapt_out, "data/adapter_output.json")

                st.info(adapt_out["adaptation_summary"])

                st.markdown("**Lessons learned**")
                for lesson in adapt_out["lessons_learned"]:
                    st.markdown(f"- {lesson}")

                st.markdown(f"**Next cycle focus:** {adapt_out['next_cycle_focus']}")

                with st.expander("Updated system weights"):
                    st.json(adapt_out["updated_weights"])

            except Exception as e:
                st.error(
                    f"Couldn't run the Adapter — make sure you've completed the pipeline first. "
                    f"({e})"
                )