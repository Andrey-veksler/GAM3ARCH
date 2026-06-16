#!/usr/bin/env python3
"""
GAM3ARCH V3 — Dataclass-Based Simulator (Paper Reproduction)

A cleaner, more extensible version of the GAM3ARCH simulation engine.
Supports JSON configuration, experiment logging, and reproducible seeds.

Usage:
  from gam3arch_v3 import GAM3ARCHSim, SimulationConfig

  config = SimulationConfig(n_agents=500, bridge_modifier=1.5)
  sim = GAM3ARCHSim(config)
  results = sim.run_experiment()
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

P_BASE = np.array([
    [0.50, 0.25, 0.15, 0.10],
    [0.20, 0.40, 0.25, 0.15],
    [0.30, 0.20, 0.35, 0.15],
    [0.25, 0.20, 0.15, 0.40],
])

ZONE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """All tuneable parameters for a GAM3ARCH simulation."""
    n_agents: int = 500
    n_steps: int = 500
    n_mc_runs: int = 20
    seed: int = 42

    # Fatigue dynamics
    alpha: float = 0.015       # fatigue accumulation (Forge)
    beta: float = 0.008        # fatigue recovery (Back)
    noise_std: float = 0.003

    # Resonance
    R_max: float = 1.0
    s_n: float = 5.0
    F_burn: float = 0.80
    burn_window: int = 50

    # Bridge modifier
    bridge_modifier: float = 1.0

    # Intervention threshold (only used in Intervention scenario)
    intervention_threshold: float = 0.5
    intervention_modifier: float = 1.5

    # Scenario name
    scenario_name: str = "Custom"

    @classmethod
    def from_json(cls, path: str) -> "SimulationConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Predefined Scenarios
# ═══════════════════════════════════════════════════════════════

SCENARIOS = {
    "Baseline": SimulationConfig(
        bridge_modifier=1.0, scenario_name="Baseline"
    ),
    "StrongBridges": SimulationConfig(
        bridge_modifier=1.5, scenario_name="StrongBridges"
    ),
    "WeakBridges": SimulationConfig(
        bridge_modifier=0.5, scenario_name="WeakBridges"
    ),
    "Intervention": SimulationConfig(
        bridge_modifier=1.0,
        intervention_threshold=0.5,
        intervention_modifier=1.5,
        scenario_name="Intervention",
    ),
}


# ═══════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════

class GAM3ARCHSim:
    """
    GAM3ARCH agent-based simulation engine.

    Usage:
        config = SimulationConfig(n_agents=100)
        sim = GAM3ARCHSim(config)
        result = sim.run_single()       # one MC run
        results = sim.run_experiment()  # all MC runs
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._P_base = self._build_matrix(config.bridge_modifier)
        if config.scenario_name == "Intervention":
            self._P_intervention = self._build_matrix(config.intervention_modifier)

    @staticmethod
    def _build_matrix(bridge_modifier: float) -> np.ndarray:
        """Build transition matrix with bridge modifier applied."""
        P = P_BASE.copy()
        mask = ~np.eye(len(P), dtype=bool)
        P[mask] *= bridge_modifier
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def _fatigue_delta(self, zone: int, rng: np.random.Generator) -> float:
        c = self.config
        zone_fatigue = {
            0: c.alpha,           # Forge
            1: -0.3 * c.beta,    # Nexus
            2: -c.beta,          # Back
            3: -0.5 * c.beta,    # Horizon
        }
        return zone_fatigue.get(zone, 0) + rng.normal(0, c.noise_std)

    def _motivation_delta(self, zone: int, rng: np.random.Generator) -> float:
        c = self.config
        zone_motiv = {
            0: 0.002,   # Forge
            1: 0.004,   # Nexus
            2: -0.001,  # Back
            3: 0.003,   # Horizon
        }
        return zone_motiv.get(zone, 0) + rng.normal(0, c.noise_std)

    @staticmethod
    def _resonance(R_max, s_n, bridge_strength, fatigue, motivation, horizon_frac):
        F = max(0.0, 1.0 - fatigue ** 2)
        M = max(0.0, min(1.0, motivation))
        H = 0.5 + 0.5 * horizon_frac
        R = R_max * (1 + s_n * bridge_strength) * F * M * H
        return min(R, R_max * (1 + s_n))

    def run_single(self, run_seed: int) -> Dict:
        """Execute a single Monte Carlo run."""
        rng = np.random.default_rng(run_seed)
        c = self.config

        zones = np.zeros(c.n_agents, dtype=int)
        fatigue = np.zeros(c.n_agents)
        motivation = np.full(c.n_agents, 0.7)
        burned = np.zeros(c.n_agents, dtype=bool)
        fatigue_buffer = np.zeros((c.n_agents, c.burn_window))
        buf_idx = 0

        for step in range(c.n_steps):
            # Select transition matrix
            if c.scenario_name == "Intervention":
                active_fatigue = fatigue[~burned] if (~burned).any() else np.array([0.0])
                P = self._P_intervention if np.mean(active_fatigue) > c.intervention_threshold else self._P_base
            else:
                P = self._P_base

            # Move agents between zones
            for i in range(c.n_agents):
                if burned[i]:
                    continue
                zones[i] = rng.choice(4, p=P[zones[i]])

                # Update fatigue
                delta_f = self._fatigue_delta(zones[i], rng)
                fatigue[i] = np.clip(fatigue[i] + delta_f, 0.0, 1.0)

                # Update motivation
                delta_m = self._motivation_delta(zones[i], rng)
                motivation[i] = np.clip(motivation[i] + delta_m, 0.0, 1.0)

                # Record fatigue in rolling buffer
                fatigue_buffer[i, buf_idx % c.burn_window] = fatigue[i]

                # Check burnout (only after buffer is full)
                if step >= c.burn_window:
                    window_mean = np.mean(fatigue_buffer[i])
                    if window_mean > c.F_burn:
                        burned[i] = True

            buf_idx += 1

        # Compute final metrics
        bridge_strength = float(np.mean(P[~np.eye(len(P), dtype=bool)]))
        horizon_frac = np.mean(zones == 3)

        resonances = np.array([
            self._resonance(c.R_max, c.s_n, bridge_strength,
                            fatigue[i], motivation[i], horizon_frac)
            for i in range(c.n_agents)
        ])

        return {
            "burnout_rate": float(np.mean(burned)),
            "mean_resonance": float(np.mean(resonances)),
            "mean_fatigue": float(np.mean(fatigue)),
            "bridge_strength": bridge_strength,
        }

    def run_experiment(self) -> pd.DataFrame:
        """Run all Monte Carlo runs and return a summary DataFrame."""
        c = self.config
        records = []

        for run_idx in range(c.n_mc_runs):
            result = self.run_single(c.seed + run_idx + 1)
            result["run"] = run_idx + 1
            result["scenario"] = c.scenario_name
            records.append(result)

        df = pd.DataFrame(records)

        summary = pd.DataFrame([{
            "scenario": c.scenario_name,
            "burnout_rate": df["burnout_rate"].mean(),
            "burnout_rate_std": df["burnout_rate"].std(),
            "mean_resonance": df["mean_resonance"].mean(),
            "mean_resonance_std": df["mean_resonance"].std(),
            "mean_fatigue": df["mean_fatigue"].mean(),
            "bridge_strength": df["bridge_strength"].iloc[0],
        }])

        return summary


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAM3ARCH V3 Simulator")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        help="Which scenario to run")
    parser.add_argument("--agents", type=int, default=500)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    if args.config:
        configs = {"Custom": SimulationConfig.from_json(args.config)}
    elif args.scenario == "all":
        configs = SCENARIOS
    else:
        configs = {args.scenario: SCENARIOS[args.scenario]}

    # Override common params
    for cfg in configs.values():
        cfg.n_agents = args.agents
        cfg.n_steps = args.steps
        cfg.n_mc_runs = args.runs

    all_results = []
    for name, config in configs.items():
        print(f"Running: {name} (bridge_modifier={config.bridge_modifier})")
        sim = GAM3ARCHSim(config)
        summary = sim.run_experiment()
        all_results.append(summary)
        print(f"  Burnout: {summary['burnout_rate'].values[0]:.3f}  "
              f"Resonance: {summary['mean_resonance'].values[0]:.3f}")

    final = pd.concat(all_results, ignore_index=True)
    print("\n" + final.to_string(index=False))
