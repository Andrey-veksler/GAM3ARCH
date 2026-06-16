[GAM3ARCH_README (1).md](https://github.com/user-attachments/files/28990988/GAM3ARCH_README.1.md)
<div align="center">

# GAM3ARCH

### A Cognitive Architecture for Ethical Game Design and Player Burnout Analysis

**Four zones. One simulation. Burnout doesn't stand a chance.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-Academia.edu-A62423.svg)](https://www.academia.edu/144868913/GAM3ARCH_A_Cognitive_Architecture_for_Ethical_Game_Design_and_Player_Burnout_Analysis)
[![DOI](https://img.shields.io/badge/DOI-Zenodo-0A71B5.svg)](https://zenodo.org/records/20708078)
[![Framework](https://img.shields.io/badge/SpinForge-Extension-8E44AD.svg)](https://github.com/veksler-ship/SpinForge)

</div>

---

## Overview

GAM3ARCH is a research framework that models player engagement through four cognitive zones — **Forge** (growth), **Nexus** (social connection), **Back** (recovery), and **Horizon** (meaning) — and simulates how transitions between these zones determine burnout risk and long-term retention.

The framework introduces **WeakBridges** as a structural mechanism: when inter-zone connectivity drops below critical thresholds, players become trapped in high-demand zones without adequate recovery, leading to cognitive burnout. By measuring and reinforcing bridge strength, designers can build games that sustain engagement without exploiting players.

**Key result:** Agent-based simulation (500 agents, 500 time steps, 20 Monte Carlo runs) shows that strengthening bridge connectivity above 0.3 reduces burnout incidence from ~42% to ~8% — a fivefold improvement.

---

## The Four Zones

```
         ┌──────────────┐
         │    FORGE      │  Growth, mastery, challenge
         │   94-100 BPM  │  "One more run"
         └──────┬───────┘
                │ bridge
    ┌───────────┼───────────┐
    │           │           │
┌───┴────┐  ┌──┴───┐  ┌────┴─────┐
│ NEXUS  │  │BACK  │  │ HORIZON  │
│ Social │  │Rest  │  │ Meaning  │
│79-85BPM│  │72-78 │  │ 75-80    │
└────────┘  └──────┘  └──────────┘
```

| Zone | Function | Player Experience | BPM Range |
|------|----------|-------------------|-----------|
| **Forge** | Growth, mastery, challenge | Raids, farming, ranked grind | 94–100 |
| **Nexus** | Social connection, community | Discord, co-op, streams | 79–85 |
| **Back** | Recovery, reflection | Downtime, casual play, logout ritual | 72–78 |
| **Horizon** | Meaning, meta-awareness | Long-term goals, life integration | 75–80 |

When a player is stuck in Forge without access to Back or Nexus → **Impotence** (high demand, low reward) → **Burnout**.

---

## How It Works

### 1. Define the Transition Matrix

The 4×4 Markov matrix `P_BASE` defines how players move between zones:

```python
P_BASE = [
    [0.50, 0.25, 0.15, 0.10],  # Forge → Forge, Nexus, Back, Horizon
    [0.20, 0.40, 0.25, 0.15],  # Nexus → ...
    [0.30, 0.20, 0.35, 0.15],  # Back → ...
    [0.25, 0.20, 0.15, 0.40],  # Horizon → ...
]
```

### 2. Run the Simulation

```bash
git clone https://github.com/veksler-ship/GAM3ARCH.git
cd GAM3ARCH
pip install -r requirements.txt
python run.py
```

### 3. Compare Scenarios

| Scenario | Bridge Strength | Burnout Incidence | Resonance Index |
|----------|----------------|-------------------|-----------------|
| Baseline | 0.15 (default) | ~0.42 | 0.48 |
| StrongBridges | 0.30+ | ~0.08 | 0.76 |
| WeakBridges | < 0.10 | ~0.67 | 0.21 |
| Intervention | Adaptive | ~0.15 | 0.69 |

### 4. Extract Bridges from Your Data

```bash
python bridge_extractor.py --input data/telemetry.csv --output B_matrix.json
```

Upload your player telemetry (player_id, timestamp, zone) and the extractor infers the bridge matrix automatically.

---

## Interactive Dashboard

Launch the Streamlit dashboard for visual exploration:

```bash
pip install streamlit
streamlit run dashboard.py
```

**Three tabs:**
- **Bridge Extractor** — Upload CSV, visualize transition patterns
- **Simulation** — Run scenarios with configurable parameters
- **Results** — Compare burnout rates and resonance across conditions

---

## Core Metrics

### Resonance Function
```
R = R_max × (1 + s_n × S) × F(Fatigue) × M(Motivation) × H(Horizon)
```
A composite metric measuring ethical engagement — not just retention, but sustainable, meaningful participation.

### Five Derivative Metrics

| Metric | Abbreviation | What It Measures |
|--------|-------------|-----------------|
| FOMO Load | F | Pressure from fear of missing out |
| Transparency Index | T | How visible game mechanics are to players |
| Healthy Return Coefficient | R_h | Probability of voluntary return after rest |
| Voluntary Participation Level | V | Share of non-coerced engagement |
| Resilience Index | RI | Resistance to burnout under stress |

---

## File Structure

```
GAM3ARCH/
├── run.py                  # Main entry point (paper reproduction)
├── gam3arch_v3.py          # V3.1 dataclass-based simulator
├── gam3arch_v2_clean.py    # V2 simplified simulation
├── bridge_extractor.py     # Infer bridge matrix from telemetry
├── dashboard.py            # Streamlit interactive dashboard
├── requirements.txt        # numpy, pandas, scipy
├── data/
│   └── survey_results.csv  # Player survey data (N=512)
├── model/
│   └── gam3arch_sim.py     # Original simple simulation
├── examples/
│   ├── sample_data.csv
│   └── sample_telemetry.csv
└── docs/
    └── GAM3ARCH_theory.txt # Theory summary
```

---

## Theoretical Grounding

| Theory | Authors | Connection to GAM3ARCH |
|--------|---------|----------------------|
| Self-Determination Theory | Deci & Ryan (2000) | Autonomy, competence, relatedness map to zone balance |
| Flow Theory | Csikszentmihalyi (1990) | Challenge-skill balance in Forge zone |
| Uses & Gratifications | Katz et al. (1973) | Motivational drivers across zones |
| Job Demands-Resources | Bakker & Demerouti (2017) | Demand-recovery cycle models burnout dynamics |
| Conservation of Resources | Hobfoll (1989) | Resource depletion under weak bridges |

---

## SpinForge Extension

GAM3ARCH provides the cognitive architecture. **[SpinForge](https://github.com/veksler-ship/SpinForge)** applies it to esports event design — translating the four-zone model into tournament scheduling with the Event Pulse monitoring system. If GAM3ARCH is the theory, SpinForge is the practice.

---

## Publications

- **GAM3ARCH: A Cognitive Architecture for Ethical Game Design and Player Burnout Analysis** — [Academia.edu](https://www.academia.edu/144868913/GAM3ARCH_A_Cognitive_Architecture_for_Ethical_Game_Design_and_Player_Burnout_Analysis) | [Zenodo](https://zenodo.org/records/20708078)
- **SpinForge: A Cognitive-Oriented Methodology for Designing Esports Events** — [Zenodo](https://zenodo.org/records/SpinForge)

---

## Citation

```bibtex
@article{skrobov2025gam3arch,
  title={GAM3ARCH: A Cognitive Architecture for Ethical Game Design and Player Burnout Analysis},
  author={Skrobov, Andrey A.},
  year={2025},
  publisher={NeuroFun Palace},
  url={https://zenodo.org/records/20708078}
}
```

---

## License

MIT License — use freely, cite fairly.

---

<div align="center">

*"Players aren't tired of games — they're tired of being treated like resources."*

**GAM3ARCH Research Initiative · NeuroFun Palace · 2025**

</div>
