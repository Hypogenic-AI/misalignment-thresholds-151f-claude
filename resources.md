# Resources Catalog

## Summary

This document catalogs all resources gathered for the mathematics research project: "Mathematical Framework for Misalignment Sensitivity Thresholds in LLM Trading Agents."

**Total papers downloaded**: 19
**Prior results cataloged**: 6 key theorems/propositions
**Computational tools set up**: 5 Python packages (SymPy, NumPy, SciPy, NetworkX, Matplotlib)

---

## Papers

| # | Title | Authors | Year | File | Key Results |
|---|-------|---------|------|------|-------------|
| 1 | Defining and Characterizing Reward Hacking | Skalse et al. | 2022 | papers/2209.13085_*.pdf | Formal hackability definition; impossibility theorem over open policy sets |
| 2 | Reward Hacking as Equilibrium under Finite Evaluation | Wang & Huang | 2026 | papers/2603.28063_*.pdf | Distortion inevitability; agentic amplification; Goodhart-Campbell conjecture |
| 3 | Scaling Compute Is Not All You Need | Debenedetti et al. | 2023 | papers/2312.13131_*.pdf | Power law: accuracy = C × FLOPs^α, α ≈ 0.01 |
| 4 | Scaling Patterns in Adversarial Alignment | Nathanson et al. | 2026 | papers/2511.13788_*.pdf | Harm ∝ log(size ratio), r = 0.510 |
| 5 | Adversarial Robustness Limits via Scaling-Law | Various | 2024 | papers/2404.09349_*.pdf | First scaling laws for adversarial training |
| 6 | Scaling Trends in LLM Robustness | Various | 2024 | papers/2407.18213_*.pdf | Scale improves adversarial training efficiency |
| 7 | TradingAgents Multi-Agent Framework | Xiao et al. | 2024 | papers/2412.20138_*.pdf | Multi-agent LLM trading architecture |
| 8 | FinCon LLM Financial Decision | Yu et al. | 2024 | papers/2407.06567_*.pdf | Conceptual verbal reinforcement for finance |
| 9 | ATLAS Adaptive Trading | Papadakis et al. | 2025 | papers/2510.15949_*.pdf | Dynamic prompt optimization for trading |
| 10 | FINRS Risk-Sensitive Trading | Liu & Dang | 2025 | papers/2511.12599_*.pdf | Risk-sensitive multi-step prediction |
| 11 | FinPos Position-Aware Trading | Liu & Dang | 2025 | papers/2510.27251_*.pdf | Position-aware LLM trading |
| 12 | Orchestration Framework Financial Agents | Li et al. | 2025 | papers/2512.02227_*.pdf | Agentic trading orchestration |
| 13 | Regime Switching Stochastic Volatility | Mitra | 2009 | papers/0904.1756_*.pdf | Perturbation theory for regime-switching volatility |
| 14 | Unified Alignment Agents Humans | Yang et al. | 2024 | papers/2402.07744_*.pdf | Multi-dimensional alignment framework |
| 15 | Specification Gaming Reasoning Models | Bondarenko et al. | 2025 | papers/2502.13295_*.pdf | Spec gaming in reasoning LLMs |
| 16 | SecAlign Prompt Injection Defense | Chen et al. | 2024 | papers/2410.05451_*.pdf | Defense against LLM manipulation |
| 17 | Assessing Adversarial Robustness LLMs | Various | 2024 | papers/2405.02764_*.pdf | Empirical LLM robustness assessment |
| 18 | LLM Behavioral Economics Games | Xie et al. | 2024 | papers/2412.12362_*.pdf | LLM decision-making in economic games |
| 19 | Latent Adversarial Training | Various | 2024 | papers/2407.15549_*.pdf | Robustness improvement techniques |

See papers/README.md for detailed descriptions.

---

## Prior Results Catalog

Key theorems and lemmas available for our proofs:

| Result | Source | Statement Summary | Used For |
|--------|--------|-------------------|----------|
| Impossibility of Unhackability (Thm 1) | Skalse et al. 2022 | Non-trivial unhackability impossible over open policy sets | Justifies τ > 0 generically; motivates finite policy restriction |
| Inevitability of Distortion (Prop 1) | Wang & Huang 2026 | Agent equilibrium always differs from first-best under finite eval | Establishes structural basis for misalignment threshold |
| Agentic Amplification (Prop 2) | Wang & Huang 2026 | Distortion grows without bound as tool count increases | Models how trading agent complexity affects τ |
| Power Law for Robustness | Debenedetti et al. 2023 | accuracy = C × FLOPs^0.01 | Empirical precedent for power-law form of τ |
| Log-Linear Scaling of Harm | Nathanson et al. 2026 | Harm ∝ log(size ratio) | Supports log/power-law functional form |
| Perturbation Expansion | Fouque/Mitra 2009 | P^ε = P₀ + √ε P₁ + ε P₂ + ... | Mathematical technique for σ-dependent threshold analysis |

---

## Computational Tools

| Tool | Purpose | Location | Notes |
|------|---------|----------|-------|
| SymPy | Symbolic computation | pip (in .venv) | For perturbation expansions and algebraic manipulation |
| NumPy | Numerical arrays | pip (in .venv) | For numerical verification |
| SciPy | Scientific computing | pip (in .venv) | For power-law curve fitting and statistical tests |
| NetworkX | Graph algorithms | pip (in .venv) | For multi-agent structure modeling |
| Matplotlib | Plotting | pip (in .venv) | For visualization of scaling relationships |

See code/README.md for usage details.

---

## Resource Gathering Notes

### Search Strategy
1. **Primary**: ArXiv API search across 8 query formulations covering alignment theory, adversarial robustness, LLM trading, and mathematical finance.
2. **Secondary**: Web search for recent papers (2024-2026) on scaling laws, reward hacking formalization, and LLM agent robustness.
3. **Tertiary**: Citation following from key papers (Skalse et al. references, Wang & Huang references).

### Selection Criteria
- Papers with formal mathematical definitions and theorems (not purely empirical)
- Papers establishing scaling laws or power-law relationships in robustness/alignment
- Papers providing mathematical frameworks for volatility modeling and perturbation theory
- Papers describing LLM trading agent architectures (for grounding the framework)

### Challenges Encountered
- The paper-finder service was unavailable; manual arXiv API and web search were used instead.
- The specific intersection of "misalignment thresholds + trading agents + power laws" has no direct prior work — this is genuinely novel.
- The connection between volatility regimes and alignment properties has not been previously explored, requiring synthesis across disparate fields.

---

## Recommendations for Proof Construction

Based on gathered resources, we recommend:

### 1. Proof Strategy
**Primary approach**: Combine the reward hacking formalism (Skalse et al.) with the principal-agent distortion framework (Wang & Huang) in a trading MDP with regime-switching volatility (Mitra). The threshold τ emerges as the minimum perturbation intensity that causes the distortion index to exceed a statistical detection threshold.

**Power law derivation**: Use perturbation expansion around the aligned equilibrium. The σ^(−α) dependence arises from the volatility regime's effect on the policy space geometry (higher σ → larger policy space → easier to find hackable pairs → lower τ). The h^(−β) dependence arises from error amplification over decision horizons (longer h → cumulative deviation grows → easier to detect → lower τ).

### 2. Key Prerequisites
- Theorem 1 from Skalse et al. (hackability geometry)
- Proposition 1 from Wang & Huang (distortion inevitability)
- Fouque perturbation theory (volatility expansion)
- Standard MDP theory (value functions, policy orderings)

### 3. Computational Tools
- SymPy for deriving threshold expressions symbolically
- SciPy for fitting power laws to simulated threshold data
- NumPy for spectral analysis of transition kernels

### 4. Potential Difficulties
- Bridging the abstract MDP formalism to concrete LLM trading architectures requires careful modeling
- The power-law form may require regularity conditions on the MDP that need to be stated precisely
- Architecture dependence of α, β needs clear operationalization (what "architecture" means mathematically)
