#!/usr/bin/env python3
"""
GAM3ARCH — Main Simulator (Paper Reproduction)

Runs the full agent-based simulation across four scenarios:
  1. Baseline       — default bridge strengths
  2. StrongBridges  — reinforced inter-zone connectivity
  3. WeakBridges    — degraded connectivity (overload model)
  4. Intervention   — adaptive bridge repair when fatigue rises

Outputs:
  results/summary.csv  — per-scenario burnout and resonance statistics

Usage:
  python run.py
  python run.py --agents 500 --steps 500 --runs 20
"""

import argparse
import json
import os
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Constants & Transition Matrix
# ═══════════════════════════════════════════════════════════════

P_BASE = np.array([
    [0.50, 0.25, 0.15, 0.10],   # Forge   → Forge, Nexus, Back, Horizon
    [0.20, 0.40, 0.25, 0.15],   # Nexus   → ...
    [0.30, 0.20, 0.35, 0.15],   # Back    → ...
    [0.25, 0.20, 0.15, 0.40],   # Horizon → ...
])

ZONE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]
ZONE_INDEX = {name: i for i, name in enumerate(ZONE_NAMES)}


# ═══════════════════════════════════════════════════════════════
# Simulation Parameters
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Configuration for a single GAM3ARCH simulation run."""
    n_agents: int = 500
    n_steps: int = 500
    n_mc_runs: int = 20
    seed: int = 42

    # Fatigue dynamics
    alpha: float = 0.015       # fatigue accumulation per step (Forge)
    beta: float = 0.008        # fatigue recovery per step (Back)
    noise_std: float = 0.003   # stochastic noise in fatigue

    # Resonance function
    R_max: float = 1.0
    s_n: float = 5.0           # sensitivity to bridge strength
    F_burn: float = 0.80       # burnout fatigue threshold
    burn_window: int = 50      # rolling window for burnout detection

    # Bridge modifiers per scenario
    bridge_modifier: float = 1.0  # multiplier on off-diagonal transitions


# ═══════════════════════════════════════════════════════════════
# Agent Model
# ═══════════════════════════════════════════════════════════════

@dataclass
class Agent:
    """A single simulated participant."""
    agent_id: int
    zone: int = 0             # start in Forge
    fatigue: float = 0.0      # 0–1
    motivation: float = 0.7   # 0–1
    burned_out: bool = False
    fatigue_history: List[float] = field(default_factory=list)
    zone_history: List[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def build_transition_matrix(P_base: np.ndarray, bridge_modifier: float) -> np.ndarray:
    """
    Apply bridge modifier to off-diagonal elements of P_BASE.

    - bridge_modifier > 1.0  → stronger inter-zone transitions (healthy bridges)
    - bridge_modifier < 1.0  → weaker transitions (weak bridges)
    - Diagonal elements are reduced/increased to keep rows stochastic.
    """
    P = P_base.copy()
    off_diag_mask = ~np.eye(len(P), dtype=bool)
    P[off_diag_mask] *= bridge_modifier
    # Re-normalize rows to sum to 1
    P = P / P.sum(axis=1, keepdims=True)
    return P


def fatigue_delta(zone: int, alpha: float, beta: float, noise_std: float,
                  rng: np.random.Generator) -> float:
    """Compute fatigue change for one time step based on current zone."""
    if zone == ZONE_INDEX["Forge"]:
        return alpha + rng.normal(0, noise_std)
    elif zone == ZONE_INDEX["Back"]:
        return -beta + rng.normal(0, noise_std)
    elif zone == ZONE_INDEX["Nexus"]:
        return -0.3 * beta + rng.normal(0, noise_std)
    else:  # Horizon
        return -0.5 * beta + rng.normal(0, noise_std)


def motivation_delta(zone: int, noise_std: float,
                     rng: np.random.Generator) -> float:
    """Compute motivation change for one time step."""
    if zone == ZONE_INDEX["Forge"]:
        return 0.002 + rng.normal(0, noise_std)
    elif zone == ZONE_INDEX["Nexus"]:
        return 0.004 + rng.normal(0, noise_std)
    elif zone == ZONE_INDEX["Horizon"]:
        return 0.003 + rng.normal(0, noise_std)
    else:  # Back — motivation stays stable or dips slightly
        return -0.001 + rng.normal(0, noise_std)


def resonance(R_max: float, s_n: float, bridge_strength: float,
              fatigue: float, motivation: float, horizon_frac: float) -> float:
    """
    GAM3ARCH Resonance Function:
      R = R_max × (1 + s_n × S) × F(Fatigue) × M(Motivation) × H(Horizon)

    where:
      S  = mean off-diagonal transition probability (bridge strength)
      F  = 1 - fatigue^2   (fatigue penalty)
      M  = motivation       (motivation amplifier)
      H  = 0.5 + 0.5 × horizon_frac  (horizon meaning factor)
    """
    F = max(0.0, 1.0 - fatigue ** 2)
    M = max(0.0, min(1.0, motivation))
    H = 0.5 + 0.5 * horizon_frac
    R = R_max * (1 + s_n * bridge_strength) * F * M * H
    return min(R, R_max * (1 + s_n * 1.0))  # cap at theoretical max


def check_burnout(fatigue_history: List[float], window: int,
                  threshold: float) -> bool:
    """Burnout = windowed mean fatigue exceeds threshold."""
    if len(fatigue_history) < window:
        return False
    return np.mean(fatigue_history[-window:]) > threshold


# ═══════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════

def run_simulation(config: SimulationConfig) -> Dict:
    """
    Run a full Monte Carlo simulation with the given configuration.

    Returns a dictionary with:
      - burnout_rate: fraction of agents that burned out
      - mean_resonance: average resonance across agents
      - mean_fatigue: average final fatigue
      - burnout_step: mean step at which burnout was detected (or n_steps)
    """
    rng = np.random.default_rng(config.seed)
    P = build_transition_matrix(P_BASE, config.bridge_modifier)

    # Bridge strength = mean off-diagonal transition probability
    off_diag = P[~np.eye(len(P), dtype=bool)]
    bridge_strength = float(np.mean(off_diag))

    mc_burnout_rates = []
    mc_resonances = []
    mc_fatigues = []

    for run_idx in range(config.n_mc_runs):
        run_rng = np.random.default_rng(config.seed + run_idx + 1)
        agents = [Agent(agent_id=i) for i in range(config.n_agents)]

        for step in range(config.n_steps):
            horizon_count = sum(1 for a in agents if a.zone == ZONE_INDEX["Horizon"])
            horizon_frac = horizon_count / config.n_agents

            for agent in agents:
                if agent.burned_out:
                    continue

                # Transition to new zone
                agent.zone = int(run_rng.choice(4, p=P[agent.zone]))
                agent.zone_history.append(agent.zone)

                # Update fatigue
                delta_f = fatigue_delta(agent.zone, config.alpha, config.beta,
                                        config.noise_std, run_rng)
                agent.fatigue = max(0.0, min(1.0, agent.fatigue + delta_f))
                agent.fatigue_history.append(agent.fatigue)

                # Update motivation
                delta_m = motivation_delta(agent.zone, config.noise_std, run_rng)
                agent.motivation = max(0.0, min(1.0, agent.motivation + delta_m))

                # Check burnout
                if check_burnout(agent.fatigue_history, config.burn_window,
                                 config.F_burn):
                    agent.burned_out = True

        # Collect run results
        burned = sum(1 for a in agents if a.burned_out)
        mc_burnout_rates.append(burned / config.n_agents)

        final_resonances = [
            resonance(config.R_max, config.s_n, bridge_strength,
                      a.fatigue, a.motivation,
                      sum(1 for z in a.zone_history[-50:] if z == 3) / 50)
            for a in agents
        ]
        mc_resonances.append(np.mean(final_resonances))
        mc_fatigues.append(np.mean([a.fatigue for a in agents]))

    return {
        "burnout_rate": np.mean(mc_burnout_rates),
        "burnout_rate_std": np.std(mc_burnout_rates),
        "mean_resonance": np.mean(mc_resonances),
        "mean_resonance_std": np.std(mc_resonances),
        "mean_fatigue": np.mean(mc_fatigues),
        "bridge_strength": bridge_strength,
    }


def run_intervention(config: SimulationConfig) -> Dict:
    """
    Intervention scenario: adaptively strengthen bridges when
    aggregate fatigue exceeds 0.5.
    """
    rng = np.random.default_rng(config.seed)
    P_normal = build_transition_matrix(P_BASE, 1.0)
    P_strong = build_transition_matrix(P_BASE, 1.5)

    off_diag_n = P_normal[~np.eye(len(P_normal), dtype=bool)]
    off_diag_s = P_strong[~np.eye(len(P_strong), dtype=bool)]

    mc_burnout_rates = []
    mc_resonances = []
    mc_fatigues = []

    for run_idx in range(config.n_mc_runs):
        run_rng = np.random.default_rng(config.seed + run_idx + 1)
        agents = [Agent(agent_id=i) for i in range(config.n_agents)]

        for step in range(config.n_steps):
            # Check aggregate fatigue — switch to strong bridges if high
            agg_fatigue = np.mean([a.fatigue for a in agents if not a.burned_out] or [0.0])
            P = P_strong if agg_fatigue > 0.5 else P_normal
            bridge_strength = float(np.mean(P[~np.eye(len(P), dtype=bool)]))

            horizon_count = sum(1 for a in agents if a.zone == ZONE_INDEX["Horizon"])
            horizon_frac = horizon_count / config.n_agents

            for agent in agents:
                if agent.burned_out:
                    continue

                agent.zone = int(run_rng.choice(4, p=P[agent.zone]))
                agent.zone_history.append(agent.zone)

                delta_f = fatigue_delta(agent.zone, config.alpha, config.beta,
                                        config.noise_std, run_rng)
                agent.fatigue = max(0.0, min(1.0, agent.fatigue + delta_f))
                agent.fatigue_history.append(agent.fatigue)

                delta_m = motivation_delta(agent.zone, config.noise_std, run_rng)
                agent.motivation = max(0.0, min(1.0, agent.motivation + delta_m))

                if check_burnout(agent.fatigue_history, config.burn_window,
                                 config.F_burn):
                    agent.burned_out = True

        burned = sum(1 for a in agents if a.burned_out)
        mc_burnout_rates.append(burned / config.n_agents)

        final_P = P_strong  # resonance computed with last-used matrix
        bs = float(np.mean(final_P[~np.eye(len(final_P), dtype=bool)]))
        final_resonances = [
            resonance(config.R_max, config.s_n, bs,
                      a.fatigue, a.motivation,
                      sum(1 for z in a.zone_history[-50:] if z == 3) / 50)
            for a in agents
        ]
        mc_resonances.append(np.mean(final_resonances))
        mc_fatigues.append(np.mean([a.fatigue for a in agents]))

    return {
        "burnout_rate": np.mean(mc_burnout_rates),
        "burnout_rate_std": np.std(mc_burnout_rates),
        "mean_resonance": np.mean(mc_resonances),
        "mean_resonance_std": np.std(mc_resonances),
        "mean_fatigue": np.mean(mc_fatigues),
        "bridge_strength": "adaptive",
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

SCENARIOS = {
    "Baseline":       SimulationConfig(bridge_modifier=1.0),
    "StrongBridges":  SimulationConfig(bridge_modifier=1.5),
    "WeakBridges":    SimulationConfig(bridge_modifier=0.5),
}


def main():
    parser = argparse.ArgumentParser(
        description="GAM3ARCH — Agent-Based Burnout Simulation"
    )
    parser.add_argument("--agents", type=int, default=500,
                        help="Number of agents (default: 500)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Simulation time steps (default: 500)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Monte Carlo runs per scenario (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Override defaults
    for cfg in SCENARIOS.values():
        cfg.n_agents = args.agents
        cfg.n_steps = args.steps
        cfg.n_mc_runs = args.runs
        cfg.seed = args.seed

    print("=" * 60)
    print("GAM3ARCH — Cognitive Burnout Simulation")
    print(f"  Agents: {args.agents}  |  Steps: {args.steps}  |  MC runs: {args.runs}")
    print("=" * 60)

    results = []

    for name, config in SCENARIOS.items():
        print(f"\n>>> Running scenario: {name} (bridge_modifier={config.bridge_modifier})")
        result = run_simulation(config)
        result["scenario"] = name
        results.append(result)
        print(f"    Burnout rate: {result['burnout_rate']:.3f} ± {result['burnout_rate_std']:.3f}")
        print(f"    Mean resonance: {result['mean_resonance']:.3f} ± {result['mean_resonance_std']:.3f}")
        print(f"    Bridge strength: {result['bridge_strength']:.3f}")

    # Intervention scenario
    print(f"\n>>> Running scenario: Intervention (adaptive bridges)")
    int_config = SimulationConfig(
        n_agents=args.agents, n_steps=args.steps,
        n_mc_runs=args.runs, seed=args.seed,
    )
    int_result = run_intervention(int_config)
    int_result["scenario"] = "Intervention"
    results.append(int_result)
    print(f"    Burnout rate: {int_result['burnout_rate']:.3f} ± {int_result['burnout_rate_std']:.3f}")
    print(f"    Mean resonance: {int_result['mean_resonance']:.3f} ± {int_result['mean_resonance_std']:.3f}")
    print(f"    Bridge strength: {int_result['bridge_strength']}")

    # Save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    cols = ["scenario", "burnout_rate", "burnout_rate_std",
            "mean_resonance", "mean_resonance_std", "mean_fatigue",
            "bridge_strength"]
    df = df[cols]
    df.to_csv("results/summary.csv", index=False)
    print(f"\nResults saved to results/summary.csv")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<16} {'Burnout':>10} {'Resonance':>12} {'Bridge':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['scenario']:<16} {r['burnout_rate']:>10.3f} "
              f"{r['mean_resonance']:>12.3f} {str(r['bridge_strength']):>10}")


if __name__ == "__main__":
    main()
