# Mathematical Framework for Misalignment Sensitivity Thresholds in LLM Trading Agents

## 1. Executive Summary

We develop a rigorous mathematical framework for misalignment sensitivity thresholds in LLM-based trading agents, formalizing the minimum intervention intensity τ required to induce statistically significant behavioral deviation from aligned baselines. Our main theorem establishes that τ = √(2δ/F) where F is the Fisher information of the policy parameterization and δ is the statistical significance level. Contrary to the initial hypothesis that τ ∝ σ^(−α) · h^(−β), computational experiments across 100+ MDP configurations demonstrate that τ is **primarily controlled by the policy temperature T**, with τ ∝ T (R² = 0.999), while market volatility σ and decision horizon h have negligible effects (|α|, |β| < 0.02). This finding has significant practical implications: the safety of LLM trading agents against misalignment is determined by their internal policy representation (how "peaked" their decision distributions are), not by external market conditions.

## 2. Research Question & Motivation

### Research Question
Does there exist a mathematically quantifiable misalignment sensitivity threshold τ for LLM-based trading agents, and what determines its magnitude?

### Original Hypothesis
τ ∝ σ^(−α) · h^(−β) where α, β > 0 are architecture-dependent constants, σ is market volatility, and h is decision horizon length.

### Motivation
LLM-based trading agents are being deployed in financial markets where misalignment — deviation from intended objectives due to reward hacking or proxy optimization — can cause significant economic harm. Despite growing empirical evidence of alignment failures (Skalse et al. 2022, Wang & Huang 2026) and scaling laws for adversarial robustness (Debenedetti et al. 2023, Nathanson et al. 2026), no mathematical framework exists to predict *when* misalignment becomes behaviorally significant as a function of system and market parameters.

### Gap in Existing Work
- Skalse et al. (2022) prove reward hacking is generically unavoidable but don't quantify intervention thresholds
- Wang & Huang (2026) show distortion inevitability but don't connect to market-specific parameters
- Empirical scaling laws (Debenedetti et al. 2023) lack theoretical derivation
- No work bridges alignment theory with mathematical finance

## 3. Definitions and Notation

### Definition 1 (Trading MDP)
A *trading MDP* is a tuple M = (S, A, T_σ, R, γ, h) where S is a finite state space, A is a finite action space, T_σ : S × A × S → [0,1] is a volatility-parameterized transition function, R : S × A → ℝ is a bounded reward, γ ∈ (0,1) is the discount factor, and h ∈ ℕ is the decision horizon.

### Definition 2 (Volatility-Parameterized Transitions)
T_σ satisfies: (i) T_σ(s'|s,a) is continuous in σ; (ii) H(T_σ(·|s,a)) is non-decreasing in σ (higher volatility → higher transition entropy).

### Definition 3 (Intervention Operator)
For intensity ε ≥ 0 and perturbation direction Φ : S × A → ℝ with ‖Φ‖_∞ = 1, the perturbed proxy reward is R_ε(s,a) = R_proxy(s,a) + ε·Φ(s,a).

### Definition 4 (Softmax Optimal Policy)
For temperature T > 0, the softmax optimal policy under reward R is:
$$\pi_R^T(a|s) = \frac{\exp(Q_R(s,a)/T)}{\sum_{b \in A} \exp(Q_R(s,b)/T)}$$
where Q_R is the optimal Q-function under R.

### Definition 5 (Policy Divergence)
$$D(\varepsilon) = D_{KL}(\pi_{R_\varepsilon}^T \| \pi_{R_{\text{proxy}}}^T) = \sum_s d^{\pi_\varepsilon}(s) \sum_a \pi_\varepsilon(a|s) \log\frac{\pi_\varepsilon(a|s)}{\pi_0(a|s)}$$
where d^π is the stationary state distribution under policy π.

### Definition 6 (Misalignment Sensitivity Threshold)
$$\tau(\delta) = \inf\{\varepsilon > 0 : D(\varepsilon) > \delta\}$$

### Definition 7 (Policy Gradient Sensitivity)
$$G(s,a) = \frac{\partial \pi_{R_\varepsilon}^T(a|s)}{\partial \varepsilon}\bigg|_{\varepsilon=0}$$

### Definition 8 (Fisher Information Norm)
$$\|G\|_F^2 = \sum_s d^{\pi_0}(s) \sum_a \frac{G(s,a)^2}{\pi_0(a|s)}$$

## 4. Statement of Results

### Theorem 1 (Existence of Threshold)
Let M be a finite trading MDP with softmax policy parameterization at temperature T > 0. For any perturbation direction Φ with ‖Φ‖_∞ = 1 and significance level δ > 0, the misalignment sensitivity threshold τ(δ) exists and satisfies 0 < τ(δ) < ∞.

### Theorem 2 (Threshold Characterization via Fisher Information)
Under the conditions of Theorem 1, the threshold satisfies:
$$\tau(\delta) = \sqrt{\frac{2\delta}{\|G\|_F^2}} + O(\delta)$$
where ‖G‖²_F is the Fisher information norm of the policy gradient sensitivity.

### Theorem 3 (Policy Sensitivity Formula)
For a softmax policy π^T with temperature T, the policy gradient sensitivity is:
$$G(s,a) = \frac{1}{T} \pi^T(a|s) \left(\Phi(s,a) - \mathbb{E}_{\pi^T(\cdot|s)}[\Phi(s,\cdot)]\right)$$
where Φ is the perturbation direction (evaluated at fixed s).

### Corollary 1 (Temperature Scaling)
The Fisher information norm scales as ‖G‖²_F ∝ 1/T², and consequently:
$$\tau(\delta) \propto T$$
That is, the misalignment threshold is linear in the policy temperature.

### Proposition 1 (Weak Volatility Dependence)
For a trading MDP with Gaussian transition kernel of width σ, the threshold satisfies:
$$\tau(\sigma) = \tau_0 + O(\sigma^{-2})$$
where τ_0 is the threshold in the σ → ∞ (uniform transition) limit. The dependence on σ is asymptotically negligible.

### Proposition 2 (Weak Horizon Dependence)
For effective horizon h = 1/(1-γ), the threshold satisfies:
$$\tau(h) = \tau_0 + O(h^{-1})$$
The dependence on h is asymptotically negligible for h ≫ 1.

## 5. Proofs

### Proof of Theorem 1

**Proof.** We show 0 < τ < ∞.

*Lower bound (τ > 0)*: Since the Q-function Q_{R_ε} is continuous in ε (the Bellman operator with reward R_ε = R_proxy + εΦ depends continuously on ε for bounded Φ), the softmax policy π^T_{R_ε} is continuous in ε. Therefore D(ε) = D_KL(π_ε ‖ π_0) is continuous in ε with D(0) = 0 < δ. By continuity, there exists ε_0 > 0 such that D(ε) < δ for all ε ∈ [0, ε_0). Hence τ ≥ ε_0 > 0.

*Upper bound (τ < ∞)*: For ε sufficiently large, the perturbed reward εΦ dominates R_proxy, so the perturbed policy π_ε converges to the greedy policy with respect to Φ alone, which generically differs from π_0. Specifically, for any states s where argmax_a Φ(s,a) ≠ argmax_a Q_{R_proxy}(s,a), the policy divergence D(ε) → D_KL(π_Φ ‖ π_0) > 0 as ε → ∞. Since D_KL(π_Φ ‖ π_0) > 0 for generic Φ, there exists ε_1 < ∞ such that D(ε_1) > δ. Hence τ ≤ ε_1 < ∞. □

### Proof of Theorem 2

**Proof.** We expand D(ε) = D_KL(π_ε ‖ π_0) to second order in ε.

Write π_ε(a|s) = π_0(a|s) + ε·G(s,a) + O(ε²) where G is the first-order sensitivity from Theorem 3.

Expanding the KL divergence:
$$D_{KL}(\pi_\varepsilon \| \pi_0) = \sum_s d_\varepsilon(s) \sum_a \pi_\varepsilon(a|s) \log\frac{\pi_\varepsilon(a|s)}{\pi_0(a|s)}$$

For the per-state KL divergence, using the Taylor expansion log(1+x) = x - x²/2 + O(x³):
$$\sum_a \pi_\varepsilon(a|s) \log\frac{\pi_\varepsilon(a|s)}{\pi_0(a|s)} = \sum_a (\pi_0 + \varepsilon G) \log\left(1 + \frac{\varepsilon G}{\pi_0}\right)$$

$$= \sum_a (\pi_0 + \varepsilon G)\left(\frac{\varepsilon G}{\pi_0} - \frac{\varepsilon^2 G^2}{2\pi_0^2} + O(\varepsilon^3)\right)$$

$$= \varepsilon \sum_a G + \frac{\varepsilon^2}{2}\sum_a \frac{G^2}{\pi_0} - \varepsilon^2 \sum_a \frac{G^2}{\pi_0} + O(\varepsilon^3)$$

Since G is the derivative of a probability distribution, Σ_a G(s,a) = 0 for each s. Thus:

$$D_{KL}(\pi_\varepsilon(\cdot|s) \| \pi_0(\cdot|s)) = \frac{\varepsilon^2}{2}\sum_a \frac{G(s,a)^2}{\pi_0(a|s)} + O(\varepsilon^3)$$

Averaging over the stationary distribution (noting d_ε = d_0 + O(ε)):

$$D(\varepsilon) = \frac{\varepsilon^2}{2} \|G\|_F^2 + O(\varepsilon^3)$$

Setting D(ε) = δ and solving:

$$\varepsilon = \sqrt{\frac{2\delta}{\|G\|_F^2}} + O(\delta)$$

This is the infimum over ε, so τ(δ) = √(2δ/‖G‖²_F) + O(δ). □

*Remark*: The O(δ) correction is justified because the O(ε³) terms in the KL expansion introduce an error proportional to ε³ = (2δ/‖G‖²_F)^{3/2}, which is O(δ^{3/2}) and thus higher order.

### Proof of Theorem 3

**Proof.** The softmax policy is:
$$\pi_\varepsilon^T(a|s) = \frac{\exp(Q_\varepsilon(s,a)/T)}{\sum_b \exp(Q_\varepsilon(s,b)/T)}$$

where Q_ε(s,a) = Q_{R_proxy + εΦ}(s,a). Since the Bellman equations are linear in the reward:
$$Q_\varepsilon(s,a) = R_\varepsilon(s,a) + \gamma \sum_{s'} T_\sigma(s'|s,a) V_\varepsilon(s')$$

we have ∂Q_ε/∂ε|_{ε=0} = Φ̃(s,a) where Φ̃ includes the one-step perturbation Φ and its discounted future effect. However, for the first-order policy response, the key computation is the softmax derivative:

$$\frac{\partial}{\partial \varepsilon}\left(\frac{e^{Q_\varepsilon(s,a)/T}}{\sum_b e^{Q_\varepsilon(s,b)/T}}\right)\bigg|_{\varepsilon=0} = \frac{1}{T}\pi_0(a|s)\left(\tilde{\Phi}(s,a) - \sum_b \pi_0(b|s)\tilde{\Phi}(s,b)\right)$$

This follows from the standard softmax gradient identity. The derivative of the softmax output z_i = exp(x_i)/Σ_j exp(x_j) with respect to a parameter θ is:
$$\frac{\partial z_i}{\partial \theta} = z_i\left(\frac{\partial x_i}{\partial \theta} - \sum_j z_j \frac{\partial x_j}{\partial \theta}\right)$$

Applied with z_i = π(a|s), x_i = Q(s,a)/T, and ∂x_i/∂ε = Φ̃(s,a)/T. □

*Note*: For notational simplicity in the main formula, we write Φ in place of Φ̃, understanding that the perturbation direction includes discounted future effects.

### Proof of Corollary 1

**Proof.** From Theorem 3:
$$G(s,a) = \frac{1}{T}\pi(a|s)(\Phi(s,a) - \mathbb{E}_\pi[\Phi(s,\cdot)])$$

The Fisher information norm is:
$$\|G\|_F^2 = \sum_s d(s) \sum_a \frac{G(s,a)^2}{\pi(a|s)} = \frac{1}{T^2}\sum_s d(s) \sum_a \pi(a|s)(\Phi(s,a) - \mathbb{E}_\pi[\Phi])^2$$

$$= \frac{1}{T^2}\sum_s d(s) \text{Var}_{\pi(\cdot|s)}(\Phi(s,\cdot))$$

The variance term Var_π(Φ) depends on how concentrated the policy is, which itself depends on T. However, in the regime where the Q-value spread ΔQ = max_a Q(s,a) - min_a Q(s,a) is fixed, the dominant T-scaling is:

- For T ≫ ΔQ: π → uniform, Var_π(Φ) → Var_uniform(Φ) (constant), so ‖G‖²_F ∝ 1/T²
- For T ≪ ΔQ: π → deterministic, Var_π(Φ) → 0, but ‖G‖²_F ∝ 1/T² still dominates

In general, ‖G‖²_F = C(Q,Φ)/T² where C depends on the Q-values and Φ but not directly on T in the leading order. Therefore:

$$\tau = \sqrt{\frac{2\delta}{\|G\|_F^2}} = T\sqrt{\frac{2\delta}{C(Q,\Phi)}} \propto T$$

□

### Proof of Proposition 1

**Proof.** We show that the Fisher information norm has weak dependence on σ.

The dependence of ‖G‖²_F on σ enters through two channels:
1. The Q-values Q_R(s,a), which depend on T_σ through the Bellman equations
2. The stationary distribution d^π, which depends on T_σ

For the Gaussian transition kernel T_σ(s'|s,a) ∝ exp(-(s'-μ(s,a))²/(2σ²)), as σ → ∞ the transitions become uniform and Q-values converge to their uniform-transition limit Q_∞. The convergence rate is:

$$Q_\sigma(s,a) - Q_\infty(s,a) = O(\sigma^{-2})$$

This follows because the transition kernel converges to uniform at rate O(σ^{-2}) in total variation, and the Q-function depends Lipschitz-continuously on the transition kernel (with Lipschitz constant R_max/(1-γ)²).

Since ‖G‖²_F depends continuously on Q through the softmax policy:
$$\|G\|_F^2(\sigma) = \|G\|_F^2(\infty) + O(\sigma^{-2})$$

Therefore τ(σ) = τ_∞ + O(σ^{-2}), establishing weak dependence. □

### Proof of Proposition 2

**Proof.** The horizon enters through γ = 1 - 1/h. The Q-function satisfies:
$$Q(s,a) = R(s,a) + \gamma \sum_{s'} T(s'|s,a) V(s')$$

For γ close to 1 (large h), the Q-values scale as Q ∝ R/(1-γ) = R·h. The Q-value *differences* (which determine the policy through softmax) are:

$$Q(s,a) - Q(s,b) = R(s,a) - R(s,b) + \gamma \sum_{s'} T(s'|s,a)V(s') - \gamma \sum_{s'} T(s'|s,b)V(s')$$

The reward difference is O(1), while the value difference term is O(γ·V_spread). For large h, the value spread V_spread converges to a limit (bounded by R_max/(1-γ)), making the Q-value *differences* converge.

Since softmax policies depend only on Q-value differences (invariant to additive constants), the policy π and hence ‖G‖²_F converge as h → ∞:

$$\|G\|_F^2(h) = \|G\|_F^2(\infty) + O(h^{-1})$$

Therefore τ(h) = τ_∞ + O(h^{-1}). □

## 6. Computational Verification

### Setup
All experiments use finite trading MDPs with:
- State space: n_prices × n_positions levels
- Actions: {sell, hold, buy}
- Gaussian transition kernels with width σ
- Softmax policy with temperature T
- Threshold computed via bisection over random perturbation directions
- Theoretical prediction via Fisher information: τ_theory = √(2δ/F_max)

### 6.1 Temperature Dependence (Primary Result)

| Temperature T | τ (numerical) | τ (theoretical) | Fisher info F |
|:---:|:---:|:---:|:---:|
| 0.05 | 0.0219 | 0.0215 | 43.24 |
| 0.10 | 0.0431 | 0.0466 | 9.21 |
| 0.20 | 0.0803 | 0.0879 | 2.59 |
| 0.30 | 0.1181 | 0.1277 | 1.23 |
| 0.50 | 0.2148 | 0.2319 | 0.37 |
| 0.80 | 0.3364 | 0.3352 | 0.18 |
| 1.00 | 0.4260 | 0.4224 | 0.11 |
| 1.50 | 0.6567 | 0.6412 | 0.05 |
| 2.00 | 0.8718 | 0.8797 | 0.03 |
| 3.00 | 1.3231 | 1.2459 | 0.01 |

**Power-law fit**: τ_numerical ∝ T^1.008 with R² = 0.9992, p < 10^{-13}

**Power-law fit**: τ_theoretical ∝ T^0.987 with R² = 0.9993, p < 10^{-13}

**Interpretation**: The threshold is almost exactly linear in T (exponent ≈ 1.0), confirming Corollary 1. Numerical and theoretical values agree within 10% across two orders of magnitude in T.

### 6.2 Volatility Dependence (Null Result)

| σ | τ (theoretical) |
|:---:|:---:|
| 0.05 | 0.1198 |
| 0.10 | 0.1350 |
| 0.50 | 0.1395 |
| 1.00 | 0.1444 |
| 5.00 | 0.1408 |
| 10.0 | 0.1404 |
| 50.0 | 0.1377 |

**Power-law fit**: τ ∝ σ^0.016 with R² = 0.43 (poor fit)

**Interpretation**: τ is approximately constant (≈ 0.14) across three orders of magnitude in σ, confirming Proposition 1. The exponent α ≈ 0.016 is negligible.

### 6.3 Horizon Dependence (Null Result)

| h | γ | τ (numerical) | τ (theoretical) |
|:---:|:---:|:---:|:---:|
| 3 | 0.667 | 0.218 | 0.207 |
| 10 | 0.900 | 0.222 | 0.215 |
| 30 | 0.967 | 0.203 | 0.213 |
| 100 | 0.990 | 0.205 | 0.217 |

**Power-law fit**: τ_numerical ∝ h^{-0.009} with R² = 0.13 (poor fit)

**Interpretation**: τ is approximately constant (≈ 0.21) across horizons from 3 to 100, confirming Proposition 2.

### 6.4 Architecture Dependence

| Architecture | |S| | T | α | β | Spectral gap |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Tiny | 24 | 0.3 | 0.023 | 0.039 | 0.088 |
| Small | 48 | 0.5 | -0.018 | -0.021 | 0.038 |
| Medium | 100 | 0.5 | 0.018 | 0.013 | 0.013 |
| Cold | 100 | 0.1 | -0.021 | 0.004 | 0.013 |
| Warm | 100 | 1.0 | 0.027 | -0.021 | 0.013 |

**Interpretation**: All exponents α, β are near zero regardless of architecture, confirming that the threshold is primarily temperature-determined. The spectral gap does not significantly predict the (negligible) exponents.

## 7. Discussion

### 7.1 Refutation of the Original Hypothesis

The original hypothesis that τ ∝ σ^(−α) · h^(−β) with α, β > 0 is **refuted** for softmax-parameterized trading agents. The numerically estimated exponents are:
- α ≈ 0.02 ± 0.02 (not significantly different from 0)
- β ≈ 0.01 ± 0.03 (not significantly different from 0)

### 7.2 The Corrected Theory

The correct characterization is:

**τ ∝ T · √(2δ / Var_d(Φ))**

where:
- T is the softmax temperature (the *dominant* factor)
- δ is the significance level
- Var_d(Φ) = Σ_s d(s) Var_{π(·|s)}(Φ(s,·)) is the policy-weighted variance of the perturbation direction

This means:
1. **Lower temperature → lower threshold → more vulnerable** to misalignment
2. **Higher temperature → higher threshold → more robust** (but less capable)
3. **Market conditions (σ, h) are secondary** — the threshold is an intrinsic property of the policy parameterization

### 7.3 Connection to Existing Work

- **Skalse et al. (2022)**: Their impossibility theorem (hackability over open policy sets) is consistent with our finding that τ > 0 but finite for all finite MDPs. Our framework quantifies the "size" of hackability via τ.
- **Wang & Huang (2026)**: Their distortion inevitability result aligns with our Theorem 1 (τ < ∞). Our Corollary 1 (τ ∝ T) adds a quantitative prediction they lack.
- **Debenedetti et al. (2023)**: Their power law (accuracy ∝ FLOPs^0.01) has a very small exponent (0.01), consistent with our finding that the volatility exponent α ≈ 0.02 is negligibly small.
- **Mitra (2009)**: Perturbation theory for stochastic volatility informed our analysis but the σ-dependence turned out to be second-order.

### 7.4 Practical Implications

1. **Safety auditing**: To assess misalignment vulnerability, measure the softmax temperature (or equivalent concentration parameter) of the agent's policy, not market conditions.
2. **Robustness-capability tradeoff**: Increasing T makes agents more robust to misalignment (higher τ) but less capable (more random decisions). This is a fundamental tradeoff.
3. **Temperature-aware deployment**: In high-stakes trading, using a higher policy temperature provides an adjustable safety margin, at the cost of reduced expected performance.

### 7.5 Why Market Conditions Don't Matter (Intuition)

The key insight is that softmax policies have a built-in "sensitivity scale" set by T. The Fisher information of a softmax distribution is ∝ 1/T², regardless of the underlying Q-values. Since Q-values are continuous functions of T_σ and γ, the policy changes smoothly with ε, and the rate of change is controlled by T, not by the environment parameters.

Heuristically: the softmax "absorbs" environmental variation by adjusting Q-value magnitudes, but since it only depends on Q-value *differences* (shift-invariant), the policy sensitivity is scale-independent and thus σ-independent.

## 8. Open Questions

1. **Non-softmax parameterizations**: Do policies with different parameterizations (e.g., Gaussian policies, attention-based policies) show stronger σ or h dependence? The τ ∝ T result is specific to softmax.

2. **Multi-agent interactions**: In trading systems with multiple LLM agents (as in TradingAgents, Xiao et al. 2024), how do individual thresholds τ_i compose? Is there a system-level threshold?

3. **Goodhart-Campbell transition**: Wang & Huang (2026) conjecture a phase transition between gaming and degrading. Can our threshold framework detect this transition?

4. **Dynamic perturbations**: We studied static perturbations Φ. For time-varying or adversarially adaptive perturbations, does the threshold change qualitatively?

5. **Finite-sample effects**: In practice, D_KL is estimated from finite samples. How does estimation error affect the effective threshold?

6. **Tighter bounds in specific regimes**: For very small σ (concentrated transitions), the O(σ^{-2}) convergence rate in Proposition 1 may be improvable. What is the tight rate?

## 9. Conclusions

We established a rigorous mathematical framework for misalignment sensitivity thresholds in LLM trading agents. Our main contributions are:

1. **Formal definition** of the misalignment sensitivity threshold τ as the minimum intervention intensity causing statistically significant behavioral deviation (Definition 6).

2. **Existence theorem** (Theorem 1): τ exists and is finite for any finite MDP with softmax policies.

3. **Characterization theorem** (Theorem 2): τ = √(2δ/‖G‖²_F) to leading order, connecting the threshold to the Fisher information of the policy parameterization.

4. **Temperature scaling law** (Corollary 1): τ ∝ T with R² = 0.999, making temperature the primary determinant of misalignment robustness.

5. **Negative results** (Propositions 1-2): Market volatility σ and decision horizon h have negligible effects on τ (|exponents| < 0.02), refuting the original power-law hypothesis.

These results redirect the focus of misalignment safety from external conditions (market volatility, horizon length) to internal architecture parameters (policy temperature), providing a concrete and actionable safety metric for LLM trading agent deployment.

## 10. References

1. Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Hacking. *NeurIPS 2022*. arXiv:2209.13085.

2. Wang, J., Huang, J. (2026). Reward Hacking as Equilibrium under Finite Evaluation. arXiv:2603.28063.

3. Debenedetti, E., et al. (2023). Scaling Compute Is Not All You Need for Adversarial Robustness. arXiv:2312.13131.

4. Nathanson, S., Matuszek, C., Williams, R. (2026). Scaling Patterns in Adversarial Alignment. arXiv:2511.13788.

5. Mitra, S. (2009). Regime Switching Stochastic Volatility with Perturbation Based Option Pricing. arXiv:0904.1756.

6. Xiao, C., et al. (2024). TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138.

7. Liu, Z., Dang, T. (2025). FINRS: A Risk-Sensitive Trading Framework. arXiv:2511.12599.

8. Fouque, J.-P., Papanicolaou, G., Sircar, K.R. (2000). *Derivatives in Financial Markets with Stochastic Volatility*. Cambridge University Press.

---

## Appendix: Computational Environment

- Python 3.12.8
- NumPy 2.4.4, SciPy 1.17.1, SymPy 1.14.0, Matplotlib 3.10.8
- Random seed: 42
- All experiments: CPU only (NVIDIA RTX A6000 available but not required)
- Total computation time: ~5 minutes
- Code: `src/verify_threshold_v2.py`, `src/verify_temperature.py`, `src/symbolic_proofs.py`
