# Mathematical Framework for Misalignment Sensitivity Thresholds in LLM Trading Agents

## Overview

This project develops a rigorous mathematical framework for quantifying misalignment sensitivity in LLM-based trading agents. We define and characterize the **misalignment sensitivity threshold** τ — the minimum intervention intensity required to induce statistically significant behavioral deviation from aligned baselines.

## Key Results

- **Theorem 1**: The misalignment threshold τ exists and is finite for any finite trading MDP with softmax policies
- **Theorem 2**: τ = √(2δ/‖G‖²_F) where ‖G‖²_F is the Fisher information norm of the policy gradient sensitivity
- **Corollary 1 (Main finding)**: τ ∝ T (linear in policy temperature) with R² = 0.999 — the threshold is determined by the policy's internal parameterization, not market conditions
- **Propositions 1-2**: Market volatility σ and decision horizon h have negligible effects (|exponents| < 0.02), **refuting** the original power-law hypothesis τ ∝ σ^(−α) · h^(−β)

## Practical Implication

Safety against misalignment is controlled by the agent's **policy temperature** (how concentrated its decisions are), not by external market conditions. Lower temperature = more vulnerable to misalignment.

## Repository Structure

```
├── REPORT.md              # Full research report with proofs and results
├── planning.md            # Research plan and methodology
├── definitions.md         # Formal definitions and notation
├── literature_review.md   # Synthesized literature review
├── resources.md           # Resource catalog
├── src/
│   ├── symbolic_proofs.py     # SymPy verification of key lemmas
│   ├── verify_threshold.py    # Initial numerical experiments (v1)
│   ├── verify_threshold_v2.py # Improved experiments with larger MDPs
│   └── verify_temperature.py  # Temperature dependence experiment
├── results/
│   ├── numerical_results.json     # V1 experiment data
│   ├── numerical_results_v2.json  # V2 experiment data
│   ├── temperature_results.json   # Temperature experiment data
│   └── symbolic_results.json      # Symbolic verification outputs
├── figures/
│   ├── tau_vs_sigma_v2.png         # Threshold vs volatility
│   ├── tau_vs_horizon_v2.png       # Threshold vs horizon
│   ├── tau_vs_temperature.png      # Threshold vs temperature (main result)
│   └── architecture_dependence_v2.png  # Architecture comparison
└── papers/                # Downloaded reference papers (19 papers)
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add sympy numpy scipy matplotlib

# Run experiments
python src/symbolic_proofs.py       # Symbolic verification (~10s)
python src/verify_threshold_v2.py   # Numerical experiments (~3min)
python src/verify_temperature.py    # Temperature dependence (~2min)
```

## Dependencies

- Python ≥ 3.10
- SymPy, NumPy, SciPy, Matplotlib
