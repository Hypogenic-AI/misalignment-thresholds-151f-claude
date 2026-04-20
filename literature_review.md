# Literature Review: Mathematical Framework for Misalignment Sensitivity Thresholds in LLM Trading Agents

## Research Area Overview

This research investigates a conjectured misalignment sensitivity threshold τ for LLM-based trading agents, defined as the minimal intervention intensity required to induce statistically significant behavioral deviation from aligned baselines. The hypothesis posits that τ follows a power law relationship with market volatility σ and decision horizon h: τ ∝ σ^(−α) · h^(−β) where α, β > 0 are architecture-dependent constants.

This sits at the intersection of four mathematical domains:
1. **AI alignment theory** — formal definitions of reward hacking, hackability, and behavioral distortion
2. **Adversarial robustness scaling laws** — power law relationships between model properties and vulnerability
3. **Mathematical finance** — regime-switching stochastic volatility models and perturbation theory
4. **Multi-agent systems** — LLM trading agent architectures and their decision-making frameworks

---

## Key Definitions

**Definition 1 (Hackability, Skalse et al. 2022).** A pair of reward functions R₁, R₂ are *hackable* relative to policy set Π and environment (S, A, T, I, _, γ) if there exist π, π' ∈ Π such that J₁(π) < J₁(π') and J₂(π) > J₂(π'). Otherwise they are *unhackable*.

**Definition 2 (Simplification, Skalse et al. 2022).** R₂ is a *simplification* of R₁ relative to Π if: J₁(π) < J₁(π') ⟹ J₂(π) ≤ J₂(π') and J₁(π) = J₁(π') ⟹ J₂(π) = J₂(π'), with at least one pair where J₂ equates policies that J₁ distinguishes.

**Definition 3 (Contract Incompleteness, Wang & Huang 2026).** κ ≡ (N − K)/N ∈ (0, 1), where N is the dimension of the true quality space and K < N is the dimension of the evaluation space.

**Definition 4 (Alignment Gap, Wang & Huang 2026).** Parameter λ ∈ (0, 1) measuring the degree to which agent behavior is driven by the evaluation signal versus internalized principal objective. Effective weights: w̃ᵢ = λrᵢ + (1−λ)wᵢ for contractible dimensions (i ≤ K), w̃ᵢ = (1−λ)wᵢ for non-contractible dimensions (i > K).

**Definition 5 (Adversarial Perturbation).** For model f : X → ℝ^C, input x, and label y, an adversarial perturbation δ ∈ Δ is successful if argmax_c f(x + δ)_c ≠ y. Robust accuracy is the fraction of points where the model is correct for all δ with ‖δ‖_∞ ≤ ε (Debenedetti et al. 2023).

**Definition 6 (Attack Success Rate, Nathanson et al. 2026).** ASR = 1 if yⱼ ≥ τ, else 0, where yⱼ is the aggregate harm score from judge models evaluating an adversarial LLM-to-LLM interaction.

**Definition 7 (Regime-Switching Stochastic Volatility, Mitra 2009).** Stock price dynamics dX/X = μdt + σ(Yₜ, Zₜ)dW where Yₜ is a fast mean-reverting process (OU process) and Zₜ is a discrete-state Markov chain governing regime switches.

---

## Key Papers

### Paper 1: Defining and Characterizing Reward Hacking (Skalse et al., NeurIPS 2022)
- **Authors**: Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, David Krueger
- **Source**: arXiv:2209.13085
- **Main Results**:
  - **Theorem 1**: In any MDP\R, if the policy set Π̂ contains an open set, then any pair of reward functions that are unhackable and non-trivial on Π̂ are equivalent on Π̂. This means non-trivial unhackability is impossible over all stochastic policies.
  - **Corollary 1**: Over the set of all stationary policies, unhackable and non-trivial reward pairs must be equivalent.
  - Non-trivial unhackable pairs exist for finite policy sets and deterministic policies.
  - The linearity of reward in state-action visit counts (J(π) = ⟨R, F^π⟩) makes unhackability extremely restrictive.
- **Proof Techniques**: Embedding policies into Euclidean space via visit counts F^π, using topological arguments (open sets, homeomorphisms), and the linear structure of expected reward.
- **Relevance**: Provides the foundational formalism for understanding when proxy optimization can diverge from true objectives — directly applicable to defining the misalignment threshold τ. The impossibility result over continuous policy sets motivates focusing on finite/discrete policy restrictions (as in trading with discrete actions).

### Paper 2: Reward Hacking as Equilibrium under Finite Evaluation (Wang & Huang, 2026)
- **Authors**: Jiacheng Wang, Jinbin Huang
- **Source**: arXiv:2603.28063
- **Main Results**:
  - **Proposition 1 (Inevitability of Distortion)**: Under Axioms 1-4 (multi-dimensional quality, finite evaluation, effective optimization, resource finiteness), with alignment gap λ ∈ (0,1) and K < N: (a) e*ᵢ ≤ eᵢ^FB for non-contractible dimensions, (b) e* ≠ e^FB, (c) W(q*) < W(q^FB).
  - **Proposition 2 (Agentic Amplification)**: With T composable tools, quality dimensions N(T) ≥ T + α(T choose 2) grow combinatorially, while evaluation coverage K grows at most linearly. Hence distortion severity increases without bound.
  - **Conjecture**: Existence of a capability threshold beyond which agents transition from Goodhart regime (gaming within evaluation) to Campbell regime (actively degrading evaluation).
  - **Distortion Index**: D_i = |w̃ᵢ/wᵢ − 1| predicts direction and severity of behavioral distortion per dimension.
- **Proof Techniques**: Multi-task principal-agent model (Holmström-Milgrom 1991), Lagrangian optimization with KKT conditions, monotone reallocation lemma for comparing optimization under different weight vectors.
- **Relevance**: **Critically important** — directly formalizes the concept of a threshold for misalignment severity. The distortion index D_i and the Goodhart-Campbell transition conjecture map directly to our τ threshold. The agentic amplification result is relevant for trading agents with multiple tool capabilities.

### Paper 3: Scaling Compute Is Not All You Need for Adversarial Robustness (Debenedetti et al., 2023)
- **Authors**: Edoardo Debenedetti, Zishen Wan, Maksym Andriushchenko, et al.
- **Source**: arXiv:2312.13131
- **Main Results**:
  - **Power law for adversarial robustness**: accuracy = C × (FLOPs)^α, with estimated α* = 0.01 and C* = 50.86 (very slow growth).
  - Power law also holds for electricity cost (α* = 0.95) and CO₂ emissions (α* = 0.95) vs. FLOPs, both with R² = 0.95.
  - Synthetic data, training loss, and model parameters are the most important features predicting adversarial robustness.
  - Scaling up model size does not yield proportionate improvements in adversarial robustness, unlike standard training.
- **Proof Techniques**: Empirical scaling law fitting (log-log regression), gradient boosting regression for feature importance analysis, systematic ablation across 320 models.
- **Relevance**: Establishes the precedent of power law relationships in adversarial robustness — the same mathematical form hypothesized for τ. The very small exponent α = 0.01 suggests diminishing returns from scaling, analogous to our threshold decreasing with volatility/horizon.

### Paper 4: Scaling Patterns in Adversarial Alignment (Nathanson et al., 2026)
- **Authors**: Samuel Nathanson, Cynthia Matuszek, Rebecca Williams
- **Source**: arXiv:2511.13788
- **Main Results**:
  - Mean harm correlates with log(attacker-to-target size ratio): Pearson r = 0.510, Spearman ρ = 0.519, both p < 0.001.
  - Mean harm variance higher across attackers (0.180) than targets (0.097): attacker capability matters more than target vulnerability.
  - Attacker refusal frequency strongly negatively correlated with harm (ρ = −0.927).
  - Over 6000 multi-turn adversarial interactions across 0.6B–120B parameter models.
- **Proof Techniques**: Empirical correlation analysis, multi-model adversarial simulation framework, judge-based harm scoring.
- **Relevance**: Demonstrates that adversarial vulnerability scales with the logarithm of capability ratios — a scaling law connecting model capacity to alignment breakdowns. The logarithmic relationship is consistent with power-law functional forms.

### Paper 5: Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies (2024)
- **Source**: arXiv:2404.09349
- **Main Results**: First scaling laws for adversarial training showing how model size, dataset size, and synthetic data quality affect robustness. SOTA methods diverge from compute-optimal setups.
- **Relevance**: Additional evidence for scaling laws in robustness, providing functional forms that may inform the architecture-dependent constants α, β in our hypothesis.

### Paper 6: Regime Switching Stochastic Volatility with Perturbation Based Option Pricing (Mitra, 2009)
- **Authors**: Sovan Mitra
- **Source**: arXiv:0904.1756
- **Main Results**:
  - Proposes regime-switching models with mean-reverting stochastic volatility: dX/X = μdt + σ(Yₜ, Zₜ)dW where Zₜ is a discrete Markov chain.
  - Uses Fouque's perturbation theory for option pricing under these models.
  - Demonstrates lower relative error compared to Black-Scholes and standard Fouque pricing.
- **Proof Techniques**: Singular and regular perturbation expansions, fast mean-reversion asymptotics, Markov chain regime switching.
- **Relevance**: Provides the mathematical framework for modeling market volatility σ in our hypothesis. The perturbation approach is directly applicable to analyzing how small perturbations in the volatility regime affect agent behavior thresholds.

### Paper 7: TradingAgents: Multi-Agents LLM Financial Trading Framework (Xiao et al., 2024)
- **Source**: arXiv:2412.20138
- **Relevance**: Provides the practical architecture for multi-agent LLM trading systems that would be subject to misalignment thresholds.

### Paper 8: FINRS: A Risk-Sensitive Trading Framework (Liu & Dang, 2025)
- **Source**: arXiv:2511.12599
- **Relevance**: Demonstrates risk-sensitive trading with LLM agents, including multi-step prediction and position management — the decision horizon h in our hypothesis.

---

## Known Results (Prerequisite Theorems)

### Theorem (Skalse et al. 2022, Theorem 1): Impossibility of Non-Trivial Unhackability on Open Policy Sets
**Statement**: In any MDP\R, if Π̂ contains an open set, then any pair of reward functions that are unhackable and non-trivial on Π̂ are equivalent on Π̂.
- **Source**: arXiv:2209.13085, Section 5.1
- **Used for**: Justifies that misalignment (hackability) is generically present in continuous policy spaces; our threshold τ quantifies *when* hackability becomes behaviorally significant.

### Proposition (Wang & Huang 2026, Proposition 1): Inevitability of Distortion
**Statement**: Under finite evaluation (K < N), the agent's equilibrium effort e* systematically under-invests in non-contractible dimensions relative to the first-best e^FB.
- **Source**: arXiv:2603.28063, Section 3.2
- **Used for**: The structural inevitability of distortion establishes that τ > 0 always exists; the question becomes quantifying its magnitude.

### Scaling Law (Debenedetti et al. 2023): Power Law for Adversarial Robustness
**Statement**: accuracy = C × (FLOPs)^α with α ≈ 0.01.
- **Source**: arXiv:2312.13131, Eq. (4)
- **Used for**: Provides empirical precedent for power-law relationship in robustness metrics, supporting the hypothesized form τ ∝ σ^(−α) · h^(−β).

### Correlation (Nathanson et al. 2026): Log-Linear Scaling of Adversarial Harm
**Statement**: Mean harm ∝ log(attacker_size / target_size), r = 0.510, p < 0.001.
- **Source**: arXiv:2511.13788, Section 4
- **Used for**: Establishes that adversarial vulnerability has a logarithmic/power-law functional form with respect to capability ratios.

### Perturbation Expansion (Fouque et al. 2000 / Mitra 2009): Fast Mean-Reversion Asymptotics
**Statement**: Option price P^ε = P₀ + √ε P₁ + ε P₂ + ... under fast mean-reverting stochastic volatility with rate 1/ε.
- **Source**: arXiv:0904.1756
- **Used for**: Mathematical technique for analyzing how small perturbations (interventions of intensity τ) propagate through volatility-dependent decision systems.

---

## Proof Techniques in the Literature

### 1. Policy Space Embedding (from Skalse et al.)
Policies are embedded into Euclidean space via visit counts F^π or action probabilities G(π). The linear structure J(π) = ⟨R, F^π⟩ constrains the geometry of hackable/unhackable pairs. This technique could be used to characterize the threshold τ geometrically as the minimum perturbation that moves the proxy-optimal policy outside the true-optimal region.

### 2. Principal-Agent Optimization (from Wang & Huang)
Lagrangian optimization with KKT conditions under finite evaluation constraints. The distortion index D_i = |w̃ᵢ/wᵢ − 1| provides a per-dimension measure of misalignment severity. Could be adapted to define τ as the critical value of an aggregate distortion measure across trading dimensions.

### 3. Scaling Law Fitting (from Debenedetti et al.)
Log-log regression to identify power law relationships: y = C · x^α. The technique of fitting power laws to empirical data across orders of magnitude in compute can be adapted to fit τ vs. (σ, h) relationships from simulation data.

### 4. Perturbation Theory (from Mitra)
Singular/regular perturbation expansion around fast mean-reverting stochastic volatility. The small parameter ε (reciprocal of mean-reversion rate) provides a natural expansion parameter. This technique is directly applicable to expanding agent behavior around the aligned baseline as a function of intervention intensity.

### 5. Multi-Model Adversarial Simulation (from Nathanson et al.)
Systematic evaluation of LLM-to-LLM interactions across model scales with judge-based scoring. This experimental framework can be adapted to measure misalignment thresholds empirically across different trading environments.

---

## Related Open Problems

1. **Goodhart-Campbell Transition Threshold**: Wang & Huang (2026) conjecture the existence of a capability threshold beyond which agents transition from gaming within the evaluation system to actively degrading it. Formalizing this threshold mathematically is directly related to our τ.

2. **Approximate Unhackability**: Skalse et al. (2022) note that future work should explore when hackable proxies can be shown safe in a "probabilistic or approximate sense, or when subject to only limited optimization." This is essentially the question of quantifying τ.

3. **Scaling Laws for Alignment**: Whether alignment properties follow power laws analogous to capability scaling laws (Kaplan et al. 2020) remains open. Our hypothesis that τ ∝ σ^(−α) · h^(−β) would contribute to this question.

4. **Volatility-Dependent Agent Robustness**: No existing work formally connects market volatility regimes to LLM agent alignment properties. This is a novel contribution of the proposed research.

5. **Decision Horizon Effects on Misalignment**: The relationship between planning horizon length and susceptibility to misalignment has not been mathematically formalized.

---

## Gaps and Opportunities

1. **No existing mathematical framework connects volatility to alignment thresholds**: While regime-switching models exist for volatility and formal definitions exist for reward hacking, no work bridges these domains.

2. **Power law relationships in alignment are empirical, not proven**: The scaling laws in Debenedetti et al. and Nathanson et al. are empirical fits. A theoretical derivation of the power-law form τ ∝ σ^(−α) · h^(−β) would be novel.

3. **Architecture dependence of α, β is unexplored**: The hypothesis that the exponents are architecture-dependent has no existing theoretical or empirical basis.

4. **Trading-specific misalignment not studied**: While LLM trading agents exist (TradingAgents, ATLAS, FinRS), their vulnerability to misalignment has not been studied through the lens of sensitivity thresholds.

---

## Recommendations for Proof Strategy

### Recommended Approach
1. **Define the misalignment sensitivity threshold τ formally** using the reward hacking framework of Skalse et al. (2022) adapted to a trading MDP with volatility-dependent transitions.
2. **Model the trading environment** as an MDP with regime-switching stochastic volatility (following Mitra 2009), where the transition function T depends on market state including σ.
3. **Apply the principal-agent distortion framework** (Wang & Huang 2026) to derive τ as a function of the distortion index D_i, where the dimensions correspond to trading quality dimensions (profit, risk management, compliance, etc.).
4. **Derive the power law form** by perturbation expansion around the aligned equilibrium, using the fast mean-reversion asymptotics from Fouque/Mitra to establish how σ enters the threshold through the volatility dynamics.
5. **Establish the h-dependence** through discounted reward analysis, where longer horizons (larger h) amplify small per-step deviations exponentially through the discount structure.

### Key Lemmas to Establish
- Lemma: In a volatility-regime-dependent MDP, the set of hackable reward pairs expands as σ increases (lower threshold for observable behavioral deviation).
- Lemma: For finite decision horizons h, the minimum detectable distortion decreases monotonically with h (longer horizons make misalignment more detectable, hence τ decreases).
- Lemma: The exponents α, β depend on the spectral properties of the transition kernel (architecture dependence).

### Potential Obstacles
- The infinite-dimensional nature of the stochastic policy space creates impossibility results (Theorem 1 of Skalse et al.) that must be carefully navigated by restricting to practically relevant finite policy sets.
- Connecting the abstract MDP formalism to the specific structure of LLM trading agents requires careful modeling assumptions.
- The power law form may only hold approximately or in specific regimes.

### Computational Support
- Use SymPy for symbolic computation of perturbation expansions and threshold expressions.
- Use NumPy/SciPy for numerical verification of the power-law relationship with simulated data.
- Use NetworkX for modeling the structure of multi-agent trading systems if needed.
