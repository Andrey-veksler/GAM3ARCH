#!/usr/bin/env python3
"""
GAM3ARCH Bridge Extractor

Infers the bridge strength matrix B from player telemetry data.
Input: CSV with columns (player_id, timestamp, zone)
Output: B_matrix.json with the normalized transition matrix

The bridge matrix quantifies how easily players transition between
cognitive zones. Low off-diagonal values indicate WeakBridges —
players get stuck and cannot reach recovery zones.

Usage:
  python bridge_extractor.py --input data/telemetry.csv --output B_matrix.json
  python bridge_extractor.py --input examples/sample_telemetry.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ZONE_NAMES = ["Forge", "Nexus", "Back", "Horizon"]
ZONE_INDEX = {name: i for i, name in enumerate(ZONE_NAMES)}

# Healthy interval: median time between transitions (seconds)
T0_HEALTHY = 300  # 5 minutes


def load_telemetry(path: str) -> pd.DataFrame:
    """
    Load and validate telemetry CSV.

    Expected columns:
      - player_id: unique player identifier
      - timestamp: ISO 8601 or Unix timestamp
      - zone: one of Forge, Nexus, Back, Horizon
    """
    df = pd.read_csv(path)

    required = {"player_id", "timestamp", "zone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate zone names
    invalid = set(df["zone"].unique()) - set(ZONE_NAMES)
    if invalid:
        raise ValueError(f"Invalid zone names: {invalid}. Expected: {ZONE_NAMES}")

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["player_id", "timestamp"]).reset_index(drop=True)

    return df


def compute_transition_counts(df: pd.DataFrame) -> np.ndarray:
    """
    Count zone-to-zone transitions across all players.

    Returns a 4x4 matrix where entry [i][j] is the number of
    transitions from zone i to zone j.
    """
    counts = np.zeros((4, 4), dtype=int)

    for player_id, group in df.groupby("player_id"):
        zones = group["zone"].values
        for k in range(len(zones) - 1):
            i = ZONE_INDEX[zones[k]]
            j = ZONE_INDEX[zones[k + 1]]
            counts[i][j] += 1

    return counts


def compute_bridge_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Normalize transition counts to a stochastic matrix.

    Each row sums to 1.0, representing the probability of
    transitioning from zone i to zone j.
    """
    row_sums = counts.sum(axis=1, keepdims=True)
    # Handle rows with zero transitions (avoid division by zero)
    row_sums[row_sums == 0] = 1
    B = counts / row_sums
    return B


def compute_median_dwell_times(df: pd.DataFrame) -> dict:
    """Compute median dwell time per zone in seconds."""
    dwell = {name: [] for name in ZONE_NAMES}

    for player_id, group in df.groupby("player_id"):
        times = group["timestamp"].values
        zones = group["zone"].values

        for k in range(len(zones) - 1):
            dt = (pd.Timestamp(times[k + 1]) - pd.Timestamp(times[k])).total_seconds()
            dwell[zones[k]].append(dt)

    return {name: float(np.median(vals)) if vals else 0.0 for name, vals in dwell.items()}


def compute_bridge_health(B: np.ndarray) -> dict:
    """
    Assess bridge health from the transition matrix.

    Returns per-bridge strengths and an overall health score.
    A healthy system has off-diagonal values > 0.15.
    """
    bridges = {}
    for i, src in enumerate(ZONE_NAMES):
        for j, dst in enumerate(ZONE_NAMES):
            if i != j:
                key = f"{src}->{dst}"
                bridges[key] = round(float(B[i][j]), 4)

    off_diag = B[~np.eye(len(B), dtype=bool)]
    mean_bridge = float(np.mean(off_diag))
    min_bridge = float(np.min(off_diag))

    weak_bridges = []
    for i, src in enumerate(ZONE_NAMES):
        for j, dst in enumerate(ZONE_NAMES):
            if i != j and B[i][j] < 0.15:
                weak_bridges.append(f"{src}->{dst} ({B[i][j]:.3f})")

    return {
        "mean_bridge_strength": round(mean_bridge, 4),
        "min_bridge_strength": round(min_bridge, 4),
        "weak_bridges": weak_bridges,
        "all_bridges": bridges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GAM3ARCH Bridge Extractor — Infer bridge matrix from telemetry"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to telemetry CSV file")
    parser.add_argument("--output", "-o", type=str, default="B_matrix.json",
                        help="Output JSON path (default: B_matrix.json)")
    args = parser.parse_args()

    print(f"GAM3ARCH Bridge Extractor")
    print(f"  Input: {args.input}")

    # Load data
    df = load_telemetry(args.input)
    n_players = df["player_id"].nunique()
    n_records = len(df)
    print(f"  Players: {n_players}  |  Records: {n_records}")

    # Compute transition matrix
    counts = compute_transition_counts(df)
    B = compute_bridge_matrix(counts)

    # Compute dwell times
    dwell = compute_median_dwell_times(df)

    # Assess bridge health
    health = compute_bridge_health(B)

    # Build output
    output = {
        "transition_matrix": {
            "zone_order": ZONE_NAMES,
            "matrix": [[round(float(B[i][j]), 4) for j in range(4)] for i in range(4)],
        },
        "bridge_health": health,
        "median_dwell_times_seconds": {k: round(v, 1) for k, v in dwell.items()},
        "metadata": {
            "n_players": n_players,
            "n_records": n_records,
            "source_file": str(args.input),
        },
    }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Bridge matrix saved to: {args.output}")

    # Print summary
    print(f"\n  Transition Matrix:")
    print(f"  {'':>10}", end="")
    for name in ZONE_NAMES:
        print(f"  {name:>10}", end="")
    print()
    for i, name in enumerate(ZONE_NAMES):
        print(f"  {name:>10}", end="")
        for j in range(4):
            print(f"  {B[i][j]:>10.4f}", end="")
        print()

    print(f"\n  Bridge Health:")
    print(f"    Mean strength:  {health['mean_bridge_strength']:.4f}")
    print(f"    Min strength:   {health['min_bridge_strength']:.4f}")
    if health["weak_bridges"]:
        print(f"    Weak bridges:   {', '.join(health['weak_bridges'])}")
    else:
        print(f"    Weak bridges:   None detected (all > 0.15)")

    print(f"\n  Dwell Times (median seconds):")
    for name, dt in dwell.items():
        print(f"    {name:>10}: {dt:>8.1f}s")


if __name__ == "__main__":
    main()
