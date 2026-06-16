#!/usr/bin/env python3
"""
GAM3ARCH V2 — Simplified Simulation

A lighter version for quick exploration and educational use.
Uses dictionary-based transition probabilities and includes
a FOMO (Fear of Missing Out) intervention scenario.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


ZONE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]

# Transition probabilities per zone (dictionary form)
TRANSITIONS = {
    "Forge":   {"Forge": 0.50, "Nexus": 0.25, "Back": 0.15, "Horizon": 0.10},
    "Nexus":   {"Forge": 0.20, "Nexus": 0.40, "Back": 0.25, "Horizon": 0.15},
    "Back":    {"Forge": 0.30, "Nexus": 0.20, "Back": 0.35, "Horizon": 0.15},
    "Horizon": {"Forge": 0.25, "Nexus": 0.20, "Back": 0.15, "Horizon": 0.40},
}

SCENARIOS = {
    "Baseline": {
        "description": "Default transition probabilities",
        "bridge_mult": 1.0,
        "fomo_prob": 0.0,
    },
    "StrongBridges": {
        "description": "Reinforced inter-zone connectivity",
        "bridge_mult": 1.5,
        "fomo_prob": 0.0,
    },
    "WeakBridges": {
        "description": "Degraded connectivity (overload model)",
        "bridge_mult": 0.5,
        "fomo_prob": 0.0,
    },
    "FOMO": {
        "description": "Baseline + FOMO events pulling agents back to Forge",
        "bridge_mult": 1.0,
        "fomo_prob": 0.05,  # 5% chance per step of FOMO pull
    },
}


@dataclass
class AgentV2:
    """Lightweight agent for the V2 simulation."""
    zone: str = "Forge"
    fatigue: float = 0.0
    motivation: float = 0.7
    burned_out: bool = False
    fatigue_history: List[float] = None

    def __post_init__(self):
        if self.fatigue_history is None:
            self.fatigue_history = []


def apply_bridge_multiplier(transitions: Dict, multiplier: float) -> Dict:
    """Scale off-diagonal probabilities by bridge multiplier."""
    result = {}
    for src, probs in transitions.items():
        new_probs = {}
        diag = probs[src]
        off_diag_sum = sum(v for k, v in probs.items() if k != src)
        for dst, prob in probs.items():
            if dst == src:
                new_probs[dst] = diag
            else:
                new_probs[dst] = prob * multiplier
        # Re-normalize
        total = sum(new_probs.values())
        result[src] = {k: v / total for k, v in new_probs.items()}
    return result


def run_v2(
    n_agents: int = 200,
    n_steps: int = 300,
    burn_window: int = 50,
    burn_threshold: float = 0.80,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Run all four V2 scenarios and return summary statistics.

    Returns:
        dict mapping scenario name to results dict with keys:
        burnout_rate, mean_fatigue, mean_resonance
    """
    rng = np.random.default_rng(seed)
    results = {}

    for name, scenario in SCENARIOS.items():
        trans = apply_bridge_multiplier(TRANSITIONS, scenario["bridge_mult"])

        agents = [AgentV2() for _ in range(n_agents)]

        for step in range(n_steps):
            for agent in agents:
                if agent.burned_out:
                    continue

                # FOMO pull: random chance of being yanked back to Forge
                if scenario["fomo_prob"] > 0 and agent.zone != "Forge":
                    if rng.random() < scenario["fomo_prob"]:
                        agent.zone = "Forge"

                # Normal transition
                probs = trans[agent.zone]
                zones = list(probs.keys())
                weights = list(probs.values())
                agent.zone = rng.choice(zones, p=weights)

                # Update fatigue
                if agent.zone == "Forge":
                    agent.fatigue += 0.015
                elif agent.zone == "Back":
                    agent.fatigue -= 0.008
                elif agent.zone == "Nexus":
                    agent.fatigue -= 0.003
                else:
                    agent.fatigue -= 0.004

                agent.fatigue = max(0.0, min(1.0, agent.fatigue))
                agent.fatigue_history.append(agent.fatigue)

                # Check burnout
                if len(agent.fatigue_history) >= burn_window:
                    window_mean = np.mean(agent.fatigue_history[-burn_window:])
                    if window_mean > burn_threshold:
                        agent.burned_out = True

        # Collect results
        burned = sum(1 for a in agents if a.burned_out)
        fatigues = [a.fatigue for a in agents]

        # Simple resonance: 1 - fatigue^2, scaled by motivation
        resonances = [
            max(0, (1 - a.fatigue ** 2) * a.motivation)
            for a in agents
        ]

        results[name] = {
            "burnout_rate": burned / n_agents,
            "mean_fatigue": np.mean(fatigues),
            "mean_resonance": np.mean(resonances),
        }

    return results


if __name__ == "__main__":
    print("GAM3ARCH V2 — Simplified Simulation\n")
    results = run_v2()

    print(f"{'Scenario':<16} {'Burnout':>10} {'Fatigue':>10} {'Resonance':>10}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<16} {r['burnout_rate']:>10.3f} "
              f"{r['mean_fatigue']:>10.3f} {r['mean_resonance']:>10.3f}")
