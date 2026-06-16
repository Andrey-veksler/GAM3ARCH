#!/usr/bin/env python3
"""
GAM3ARCH Unit Tests

Run with:
  pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import (
    build_transition_matrix, fatigue_delta, motivation_delta,
    resonance, check_burnout, P_BASE, ZONE_INDEX, SimulationConfig,
)
from gam3arch_v3 import GAM3ARCHSim, SimulationConfig as V3Config, SCENARIOS
from gam3arch_v2_clean import apply_bridge_multiplier, TRANSITIONS, run_v2
from bridge_extractor import (
    compute_transition_counts, compute_bridge_matrix, compute_bridge_health,
    ZONE_NAMES, ZONE_INDEX as BE_ZI,
)


# ═══════════════════════════════════════════════════════════════
# Transition Matrix
# ═══════════════════════════════════════════════════════════════

class TestTransitionMatrix:
    def test_base_matrix_rows_sum_to_one(self):
        """Each row of P_BASE must sum to 1.0."""
        for row in P_BASE:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_built_matrix_rows_sum_to_one(self):
        """Built matrix with any modifier must have rows summing to 1.0."""
        for modifier in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            P = build_transition_matrix(P_BASE, modifier)
            for row in P:
                assert abs(sum(row) - 1.0) < 1e-10, f"Failed with modifier={modifier}"

    def test_strong_bridges_increase_off_diagonal(self):
        """Modifier > 1 should increase off-diagonal values."""
        P_default = build_transition_matrix(P_BASE, 1.0)
        P_strong = build_transition_matrix(P_BASE, 1.5)
        off_default = P_default[~np.eye(4, dtype=bool)]
        off_strong = P_strong[~np.eye(4, dtype=bool)]
        assert np.mean(off_strong) > np.mean(off_default)

    def test_weak_bridges_decrease_off_diagonal(self):
        """Modifier < 1 should decrease off-diagonal values."""
        P_default = build_transition_matrix(P_BASE, 1.0)
        P_weak = build_transition_matrix(P_BASE, 0.5)
        off_default = P_default[~np.eye(4, dtype=bool)]
        off_weak = P_weak[~np.eye(4, dtype=bool)]
        assert np.mean(off_weak) < np.mean(off_default)

    def test_matrix_shape(self):
        """Transition matrix must be 4x4."""
        P = build_transition_matrix(P_BASE, 1.0)
        assert P.shape == (4, 4)


# ═══════════════════════════════════════════════════════════════
# Fatigue & Motivation
# ═══════════════════════════════════════════════════════════════

class TestFatigueDynamics:
    def test_forge_increases_fatigue(self):
        """Forge zone should accumulate fatigue on average."""
        rng = np.random.default_rng(42)
        deltas = [fatigue_delta(ZONE_INDEX["Forge"], 0.015, 0.008, 0.001, rng)
                  for _ in range(1000)]
        assert np.mean(deltas) > 0

    def test_back_decreases_fatigue(self):
        """Back zone should reduce fatigue on average."""
        rng = np.random.default_rng(42)
        deltas = [fatigue_delta(ZONE_INDEX["Back"], 0.015, 0.008, 0.001, rng)
                  for _ in range(1000)]
        assert np.mean(deltas) < 0


# ═══════════════════════════════════════════════════════════════
# Resonance Function
# ═══════════════════════════════════════════════════════════════

class TestResonance:
    def test_zero_fatigue_high_motivation(self):
        """With zero fatigue and high motivation, resonance should be positive."""
        R = resonance(1.0, 5.0, 0.2, fatigue=0.0, motivation=1.0, horizon_frac=0.25)
        assert R > 0

    def test_max_fatigue_kills_resonance(self):
        """Maximum fatigue should drive resonance toward zero."""
        R = resonance(1.0, 5.0, 0.2, fatigue=1.0, motivation=1.0, horizon_frac=0.25)
        assert R < 0.1  # very low due to (1 - 1^2) = 0

    def test_higher_bridge_strength_higher_resonance(self):
        """Stronger bridges should produce higher resonance."""
        R_weak = resonance(1.0, 5.0, 0.1, fatigue=0.3, motivation=0.7, horizon_frac=0.2)
        R_strong = resonance(1.0, 5.0, 0.3, fatigue=0.3, motivation=0.7, horizon_frac=0.2)
        assert R_strong > R_weak

    def test_resonance_is_bounded(self):
        """Resonance should not exceed theoretical maximum."""
        R = resonance(1.0, 5.0, 1.0, fatigue=0.0, motivation=1.0, horizon_frac=1.0)
        R_max_theoretical = 1.0 * (1 + 5.0 * 1.0)
        assert R <= R_max_theoretical


# ═══════════════════════════════════════════════════════════════
# Burnout Detection
# ═══════════════════════════════════════════════════════════════

class TestBurnout:
    def test_no_burnout_short_history(self):
        """Burnout should not trigger with insufficient history."""
        assert check_burnout([0.9, 0.9], window=50, threshold=0.8) is False

    def test_burnout_detected(self):
        """Sustained high fatigue should trigger burnout."""
        history = [0.85] * 50
        assert check_burnout(history, window=50, threshold=0.8) == True

    def test_no_burnout_moderate_fatigue(self):
        """Moderate fatigue should not trigger burnout."""
        history = [0.5] * 50
        assert check_burnout(history, window=50, threshold=0.8) == False


# ═══════════════════════════════════════════════════════════════
# V3 Simulator
# ═══════════════════════════════════════════════════════════════

class TestV3Simulator:
    def test_baseline_run(self):
        """Baseline scenario should complete without errors."""
        config = V3Config(n_agents=50, n_steps=50, n_mc_runs=3, seed=42)
        sim = GAM3ARCHSim(config)
        result = sim.run_experiment()
        assert len(result) == 1
        assert 0 <= result["burnout_rate"].values[0] <= 1

    def test_weak_bridges_higher_burnout(self):
        """WeakBridges should produce more burnout than StrongBridges."""
        config_weak = V3Config(
            n_agents=50, n_steps=100, n_mc_runs=5,
            seed=42, bridge_modifier=0.5, scenario_name="WeakBridges"
        )
        config_strong = V3Config(
            n_agents=50, n_steps=100, n_mc_runs=5,
            seed=42, bridge_modifier=1.5, scenario_name="StrongBridges"
        )
        sim_w = GAM3ARCHSim(config_weak)
        sim_s = GAM3ARCHSim(config_strong)
        r_w = sim_w.run_experiment()
        r_s = sim_s.run_experiment()
        assert r_w["burnout_rate"].values[0] >= r_s["burnout_rate"].values[0]


# ═══════════════════════════════════════════════════════════════
# V2 Simplified
# ═══════════════════════════════════════════════════════════════

class TestV2Simulator:
    def test_bridge_multiplier_preserves_sums(self):
        """Multiplier should keep probabilities summing to 1."""
        modified = apply_bridge_multiplier(TRANSITIONS, 1.5)
        for src, probs in modified.items():
            assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_run_v2_completes(self):
        """V2 simulation should run all scenarios without errors."""
        results = run_v2(n_agents=30, n_steps=50, seed=42)
        assert len(results) == 4
        for name, r in results.items():
            assert 0 <= r["burnout_rate"] <= 1


# ═══════════════════════════════════════════════════════════════
# Bridge Extractor
# ═══════════════════════════════════════════════════════════════

class TestBridgeExtractor:
    def test_counts_shape(self):
        """Transition counts should produce a 4x4 matrix."""
        counts = np.zeros((4, 4), dtype=int)
        # Simulate a few transitions
        counts[0][1] = 5  # Forge -> Nexus
        counts[1][2] = 3  # Nexus -> Back
        B = compute_bridge_matrix(counts)
        assert B.shape == (4, 4)

    def test_normalized_rows(self):
        """Bridge matrix rows should sum to 1 (where transitions exist)."""
        counts = np.random.randint(1, 20, size=(4, 4))
        B = compute_bridge_matrix(counts)
        for row in B:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_health_detection(self):
        """Bridge health should detect weak bridges."""
        B = np.array([
            [0.80, 0.05, 0.10, 0.05],  # Very weak off-diagonal
            [0.10, 0.70, 0.10, 0.10],
            [0.15, 0.15, 0.55, 0.15],
            [0.10, 0.10, 0.10, 0.70],
        ])
        health = compute_bridge_health(B)
        assert len(health["weak_bridges"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
