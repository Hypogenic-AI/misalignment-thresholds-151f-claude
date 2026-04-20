# Definitions and Notation

## Trading MDP

**Definition 1 (Trading MDP).** A *trading MDP* is a tuple M = (S, A, T_σ, R, γ, h) where:
- S is a finite state space encoding market state (price, volatility regime, position)
- A is a finite action space (e.g., {buy, hold, sell} × position sizes)
- T_σ : S × A × S → [0,1] is a volatility-parameterized transition function with σ > 0
- R : S × A → ℝ is a bounded reward function with |R(s,a)| ≤ R_max
- γ ∈ (0,1) is the discount factor
- h ∈ ℕ is the decision horizon (finite)

**Definition 2 (Volatility-Parameterized Transitions).** The transition kernel T_σ satisfies:
- T_σ(s'|s,a) is continuous in σ for all (s,a,s')
- Higher σ increases transition entropy: H(T_σ(·|s,a)) is non-decreasing in σ for all (s,a)
- There exists a base transition T_0 = lim_{σ→0} T_σ (deterministic limit)

## Reward Structure

**Definition 3 (True and Proxy Rewards).** A trading agent operates with:
- R_true : S × A → ℝ — the principal's true objective (risk-adjusted returns)
- R_proxy : S × A → ℝ — the proxy objective the agent is trained on

**Definition 4 (Intervention Operator).** For intervention intensity ε ≥ 0, define the perturbed proxy reward:
  R_ε(s,a) = R_proxy(s,a) + ε · Φ(s,a)
where Φ : S × A → ℝ is a bounded perturbation direction with ||Φ||_∞ = 1.

## Policies and Divergence

**Definition 5 (Optimal Policy Map).** For a finite MDP with reward R, let π_R denote an optimal policy:
  π_R = argmax_π V^π_R(s_0)
where V^π_R is the value function under R. When the argmax is not unique, we select the policy closest to π_{R_proxy} in KL divergence.

**Definition 6 (Policy Divergence).** The behavioral divergence between the aligned baseline and perturbed policy is:
  D(ε) = D_KL(π_{R_ε} || π_{R_proxy})
where D_KL is the KL divergence averaged over the stationary state distribution:
  D_KL(π || π') = E_{s~d^π}[ Σ_a π(a|s) log(π(a|s)/π'(a|s)) ]

## Misalignment Threshold

**Definition 7 (Misalignment Sensitivity Threshold).** The misalignment sensitivity threshold at significance level δ > 0 is:
  τ(σ, h, δ) = inf{ ε > 0 : D(ε) > δ }
If D(ε) ≤ δ for all ε > 0, set τ = ∞ (the system is robust). If D(ε) > δ for all ε > 0, set τ = 0 (maximally sensitive).

## Spectral Quantities

**Definition 8 (Policy Gradient Sensitivity).** The sensitivity matrix of the optimal policy to reward perturbation is:
  G(σ, h) = ∂π_{R_ε}/∂ε |_{ε=0}
This is a matrix in ℝ^{|S|×|A|} describing the first-order response of the policy to intervention.

**Definition 9 (Fisher Information of Transitions).** The Fisher information of the transition kernel with respect to σ is:
  I_F(σ; s, a) = E_{s'~T_σ(·|s,a)}[ (∂ log T_σ(s'|s,a)/∂σ)^2 ]

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| S, A | State and action spaces |
| T_σ | Volatility-parameterized transition kernel |
| σ | Market volatility parameter |
| h | Decision horizon |
| ε | Intervention intensity |
| Φ | Perturbation direction |
| R_true, R_proxy | True and proxy reward functions |
| π_R | Optimal policy under reward R |
| D(ε) | Policy divergence at intervention ε |
| τ | Misalignment sensitivity threshold |
| δ | Statistical significance level |
| γ | Discount factor |
| G | Policy gradient sensitivity matrix |
| I_F | Fisher information of transitions |
| R_max | Reward bound |
