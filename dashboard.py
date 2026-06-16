#!/usr/bin/env python3
"""
GAM3ARCH Interactive Dashboard

Launch with:
  pip install streamlit
  streamlit run dashboard.py

Three tabs:
  1. Bridge Extractor  — Upload CSV, visualize transition patterns
  2. Simulation        — Run scenarios with configurable parameters
  3. Results           — Compare burnout rates and resonance
"""

import io
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

# Import from local modules
from gam3arch_v3 import (
    GAM3ARCHSim, SimulationConfig, SCENARIOS,
    P_BASE, ZONE_NAMES,
)


# ── Page Config ──
st.set_page_config(
    page_title="GAM3ARCH Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("GAM3ARCH — Cognitive Burnout Simulator")
st.markdown("*Four zones. One simulation. Burnout doesn't stand a chance.*")


# ═══════════════════════════════════════════════════════════════
# Tab 1: Bridge Extractor
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🔗 Bridge Extractor", "⚡ Simulation", "📊 Results"])

with tab1:
    st.header("Bridge Extractor")
    st.markdown("""
    Upload a player telemetry CSV with columns: `player_id`, `timestamp`, `zone`.
    Zones must be one of: Forge, Nexus, Back, Horizon.
    """)

    uploaded = st.file_uploader("Upload telemetry CSV", type=["csv"], key="bridge_csv")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Raw Data")
            st.dataframe(df.head(20), use_container_width=True)

            # Validate
            required = {"player_id", "timestamp", "zone"}
            if not required.issubset(set(df.columns)):
                st.error(f"Missing columns: {required - set(df.columns)}")
            else:
                # Compute transitions
                from bridge_extractor import (
                    compute_transition_counts,
                    compute_bridge_matrix,
                    compute_bridge_health,
                )

                counts = compute_transition_counts(df)
                B = compute_bridge_matrix(counts)
                health = compute_bridge_health(B)

                st.subheader("Transition Matrix")
                B_df = pd.DataFrame(
                    B,
                    index=ZONE_NAMES,
                    columns=ZONE_NAMES,
                )
                st.dataframe(B_df.style.format("{:.4f}"), use_container_width=True)

                st.subheader("Bridge Health")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Bridge Strength", f"{health['mean_bridge_strength']:.4f}")
                with col2:
                    st.metric("Min Bridge Strength", f"{health['min_bridge_strength']:.4f}")

                if health["weak_bridges"]:
                    st.warning(f"Weak bridges detected: {', '.join(health['weak_bridges'])}")
                else:
                    st.success("No weak bridges detected (all > 0.15)")

                # Heatmap
                st.subheader("Transition Heatmap")
                fig_data = B_df.values
                st.bar_chart(pd.DataFrame(fig_data, index=ZONE_NAMES, columns=ZONE_NAMES))

                # Download JSON
                output = {
                    "transition_matrix": {
                        "zone_order": ZONE_NAMES,
                        "matrix": [[round(float(B[i][j]), 4) for j in range(4)] for i in range(4)],
                    },
                    "bridge_health": health,
                }
                st.download_button(
                    "Download B_matrix.json",
                    data=json.dumps(output, indent=2),
                    file_name="B_matrix.json",
                    mime="application/json",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Upload a CSV file to begin.")


# ═══════════════════════════════════════════════════════════════
# Tab 2: Simulation
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.header("Run Simulation")

    col1, col2 = st.columns(2)

    with col1:
        scenario = st.selectbox("Scenario", list(SCENARIOS.keys()))
        n_agents = st.slider("Number of agents", 50, 1000, 500, 50)
        n_steps = st.slider("Time steps", 100, 2000, 500, 100)
        n_runs = st.slider("Monte Carlo runs", 5, 50, 20, 5)

    with col2:
        config = SCENARIOS[scenario]
        st.subheader("Scenario Parameters")
        st.markdown(f"""
        - **Bridge modifier**: {config.bridge_modifier}
        - **Alpha** (fatigue rate): {config.alpha}
        - **Beta** (recovery rate): {config.beta}
        - **Burnout threshold**: {config.F_burn}
        - **Burnout window**: {config.burn_window} steps
        """)

        if scenario == "Intervention":
            st.markdown(f"""
            - **Intervention threshold**: {config.intervention_threshold}
            - **Intervention modifier**: {config.intervention_modifier}
            """)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            config = SimulationConfig(
                n_agents=n_agents,
                n_steps=n_steps,
                n_mc_runs=n_runs,
                bridge_modifier=SCENARIOS[scenario].bridge_modifier,
                scenario_name=scenario,
                intervention_threshold=SCENARIOS[scenario].intervention_threshold,
                intervention_modifier=SCENARIOS[scenario].intervention_modifier,
            )
            sim = GAM3ARCHSim(config)
            results = sim.run_experiment()

        st.session_state["last_results"] = results
        st.success("Simulation complete!")

        # Show immediate results
        r = results.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Burnout Rate", f"{r['burnout_rate']:.3f}")
        with col2:
            st.metric("Mean Resonance", f"{r['mean_resonance']:.3f}")
        with col3:
            st.metric("Mean Fatigue", f"{r['mean_fatigue']:.3f}")


# ═══════════════════════════════════════════════════════════════
# Tab 3: Results
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.header("Results Comparison")

    # Check for saved results
    import os
    if os.path.exists("results/summary.csv"):
        df = pd.read_csv("results/summary.csv")
        st.subheader("Saved Results")
        st.dataframe(df, use_container_width=True)

        if len(df) > 1:
            st.subheader("Burnout Rate by Scenario")
            st.bar_chart(df, x="scenario", y="burnout_rate")

            st.subheader("Resonance by Scenario")
            st.bar_chart(df, x="scenario", y="mean_resonance")
    else:
        st.info("No saved results found. Run a simulation first (Tab 2), or run `python run.py` from the command line.")

    # Show last in-session results
    if "last_results" in st.session_state:
        st.subheader("Last Session Result")
        st.dataframe(st.session_state["last_results"], use_container_width=True)
