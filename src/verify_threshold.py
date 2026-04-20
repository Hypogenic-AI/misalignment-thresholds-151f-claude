"""
Computational verification of misalignment sensitivity threshold theory.

This script verifies the theoretical predictions:
  τ ∝ σ^(-α) · h^(-β)
by constructing synthetic trading MDPs and computing thresholds numerically.
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime

# Reproducibility
SEED = 42
np.random.seed(SEED)

# ============================================================
# Core MDP Classes
# ============================================================

class TradingMDP:
    """
    A finite trading MDP with volatility-parameterized transitions.

    States encode (price_level, volatility_regime, position).
    Actions are {sell, hold, buy}.
    """

    def __init__(self, n_prices=5, n_vol_regimes=2, n_positions=3, sigma=1.0, gamma=0.95, horizon=10):
        self.n_prices = n_prices
        self.n_vol_regimes = n_vol_regimes
        self.n_positions = n_positions
        self.n_states = n_prices * n_vol_regimes * n_positions
        self.n_actions = 3  # sell, hold, buy
        self.sigma = sigma
        self.gamma = gamma
        self.horizon = horizon

        self._build_transitions()
        self._build_rewards()

    def _state_index(self, price, vol, pos):
        return price * (self.n_vol_regimes * self.n_positions) + vol * self.n_positions + pos

    def _decode_state(self, idx):
        pos = idx % self.n_positions
        vol = (idx // self.n_positions) % self.n_vol_regimes
        price = idx // (self.n_vol_regimes * self.n_positions)
        return price, vol, pos

    def _build_transitions(self):
        """Build transition kernel T_σ parameterized by volatility σ."""
        nS, nA = self.n_states, self.n_actions
        self.T = np.zeros((nS, nA, nS))

        for s in range(nS):
            price, vol, pos = self._decode_state(s)
            for a in range(nA):
                # New position after action
                new_pos = max(0, min(self.n_positions - 1, pos + (a - 1)))

                # Price transition: random walk with volatility-dependent spread
                for dp in range(-1, 2):  # price can go down, stay, or up
                    new_price = max(0, min(self.n_prices - 1, price + dp))

                    # Higher sigma -> more uniform (higher entropy) transitions
                    # Base probabilities: concentrated at dp=0
                    if dp == 0:
                        p_price = 1.0 / (1.0 + self.sigma)
                    else:
                        p_price = (self.sigma / 2.0) / (1.0 + self.sigma)

                    # Volatility regime transitions (Markov chain)
                    for new_vol in range(self.n_vol_regimes):
                        if new_vol == vol:
                            p_vol = 0.8  # stay in same regime
                        else:
                            p_vol = 0.2 / max(1, self.n_vol_regimes - 1)

                        new_s = self._state_index(new_price, new_vol, new_pos)
                        self.T[s, a, new_s] += p_price * p_vol

        # Normalize
        for s in range(nS):
            for a in range(nA):
                total = self.T[s, a].sum()
                if total > 0:
                    self.T[s, a] /= total

    def _build_rewards(self):
        """Build true and proxy reward functions."""
        nS, nA = self.n_states, self.n_actions

        # True reward: risk-adjusted return (profit - risk penalty)
        self.R_true = np.zeros((nS, nA))
        # Proxy reward: simple P&L (no risk adjustment)
        self.R_proxy = np.zeros((nS, nA))

        for s in range(nS):
            price, vol, pos = self._decode_state(s)
            for a in range(nA):
                new_pos = max(0, min(self.n_positions - 1, pos + (a - 1)))

                # Proxy: simple profit from price movement expectation
                price_centered = (price - self.n_prices // 2) / self.n_prices
                position_value = (new_pos - self.n_positions // 2) / self.n_positions
                self.R_proxy[s, a] = price_centered * position_value

                # True: profit minus volatility-scaled risk
                risk_penalty = 0.3 * (vol + 1) * abs(position_value)
                self.R_true[s, a] = self.R_proxy[s, a] - risk_penalty

    def solve_soft_policy(self, reward, temperature=0.1):
        """
        Compute softmax optimal policy via value iteration.
        Returns stochastic policy π(a|s) as |S|×|A| matrix.
        """
        nS, nA = self.n_states, self.n_actions
        V = np.zeros(nS)

        for _ in range(200):  # Value iteration
            Q = np.zeros((nS, nA))
            for s in range(nS):
                for a in range(nA):
                    Q[s, a] = reward[s, a] + self.gamma * self.T[s, a] @ V

            # Softmax over actions
            Q_shifted = Q - Q.max(axis=1, keepdims=True)
            exp_Q = np.exp(Q_shifted / temperature)
            V_new = temperature * np.log(exp_Q.sum(axis=1))

            if np.max(np.abs(V_new - V)) < 1e-10:
                break
            V = V_new

        # Extract policy
        Q_final = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                Q_final[s, a] = reward[s, a] + self.gamma * self.T[s, a] @ V

        Q_shifted = Q_final - Q_final.max(axis=1, keepdims=True)
        exp_Q = np.exp(Q_shifted / temperature)
        pi = exp_Q / exp_Q.sum(axis=1, keepdims=True)

        return pi

    def compute_stationary_distribution(self, pi):
        """Compute stationary state distribution under policy pi."""
        nS = self.n_states
        # Build state transition matrix under policy pi
        P = np.zeros((nS, nS))
        for s in range(nS):
            for a in range(self.n_actions):
                P[s] += pi[s, a] * self.T[s, a]

        # Find stationary distribution via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        d = np.real(eigenvectors[:, idx])
        d = np.abs(d)
        d /= d.sum()
        return d

    def kl_divergence(self, pi1, pi2):
        """Compute average KL divergence D_KL(pi1 || pi2) under pi1's stationary distribution."""
        d = self.compute_stationary_distribution(pi1)

        kl = 0.0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if pi1[s, a] > 1e-15 and pi2[s, a] > 1e-15:
                    kl += d[s] * pi1[s, a] * np.log(pi1[s, a] / pi2[s, a])

        return kl


def compute_threshold(mdp, delta=0.01, n_perturbation_dirs=20, eps_values=None):
    """
    Compute misalignment sensitivity threshold τ for a given MDP.

    τ = inf{ε > 0 : D_KL(π_ε || π_0) > δ}

    We search over multiple random perturbation directions Φ and take
    the minimum threshold (worst-case direction).
    """
    if eps_values is None:
        eps_values = np.logspace(-4, 1, 50)

    # Baseline policy
    pi_0 = mdp.solve_soft_policy(mdp.R_proxy)

    min_threshold = np.inf

    for _ in range(n_perturbation_dirs):
        # Random perturbation direction
        Phi = np.random.randn(mdp.n_states, mdp.n_actions)
        Phi /= np.max(np.abs(Phi))  # Normalize to ||Phi||_∞ = 1

        for eps in eps_values:
            R_perturbed = mdp.R_proxy + eps * Phi
            pi_eps = mdp.solve_soft_policy(R_perturbed)

            div = mdp.kl_divergence(pi_eps, pi_0)

            if div > delta:
                min_threshold = min(min_threshold, eps)
                break

    return min_threshold


# ============================================================
# Experiment 1: τ vs σ (volatility dependence)
# ============================================================

def experiment_sigma_dependence():
    """Test H2: τ ∝ σ^(-α)."""
    print("=" * 60)
    print("Experiment 1: τ vs σ (volatility dependence)")
    print("=" * 60)

    sigma_values = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    tau_values = []

    for sigma in sigma_values:
        print(f"  σ = {sigma:.2f} ... ", end="", flush=True)
        mdp = TradingMDP(n_prices=5, n_vol_regimes=2, n_positions=3,
                         sigma=sigma, gamma=0.95, horizon=10)
        tau = compute_threshold(mdp, delta=0.01, n_perturbation_dirs=30)
        tau_values.append(tau)
        print(f"τ = {tau:.6f}")

    tau_values = np.array(tau_values)

    # Filter out infinite thresholds
    valid = np.isfinite(tau_values) & (tau_values > 0)
    if valid.sum() < 3:
        print("WARNING: Not enough valid data points for power-law fit")
        return sigma_values, tau_values, None, None

    sigma_valid = sigma_values[valid]
    tau_valid = tau_values[valid]

    # Fit power law: log(τ) = -α log(σ) + c
    log_sigma = np.log(sigma_valid)
    log_tau = np.log(tau_valid)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sigma, log_tau)
    alpha = -slope
    C = np.exp(intercept)

    print(f"\n  Power law fit: τ = {C:.4f} · σ^({slope:.4f})")
    print(f"  α = {alpha:.4f}")
    print(f"  R² = {r_value**2:.6f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  std_err(slope) = {std_err:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Log-log plot
    axes[0].loglog(sigma_valid, tau_valid, 'bo', markersize=8, label='Computed τ')
    sigma_fit = np.logspace(np.log10(sigma_valid.min()), np.log10(sigma_valid.max()), 100)
    axes[0].loglog(sigma_fit, C * sigma_fit**slope, 'r-', linewidth=2,
                   label=f'Fit: τ = {C:.3f}·σ^({slope:.3f})')
    axes[0].set_xlabel('Market volatility σ', fontsize=12)
    axes[0].set_ylabel('Threshold τ', fontsize=12)
    axes[0].set_title(f'Misalignment Threshold vs Volatility\n(R²={r_value**2:.4f}, α={alpha:.3f})', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Residuals
    predicted = C * sigma_valid**slope
    residuals = (tau_valid - predicted) / predicted * 100
    axes[1].bar(range(len(sigma_valid)), residuals, color='steelblue')
    axes[1].set_xlabel('Data point index', fontsize=12)
    axes[1].set_ylabel('Relative residual (%)', fontsize=12)
    axes[1].set_title('Power-law fit residuals', fontsize=12)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/tau_vs_sigma.png', dpi=150, bbox_inches='tight')
    plt.close()

    return sigma_values, tau_values, alpha, r_value**2


# ============================================================
# Experiment 2: τ vs h (horizon dependence)
# ============================================================

def experiment_horizon_dependence():
    """Test H3: τ ∝ h^(-β)."""
    print("\n" + "=" * 60)
    print("Experiment 2: τ vs h (decision horizon dependence)")
    print("=" * 60)

    # For this experiment, horizon affects discount factor effective range
    # We model horizon through γ: effective horizon ~ 1/(1-γ)
    # So h = 1/(1-γ), equivalently γ = 1 - 1/h
    h_values = np.array([5, 10, 20, 50, 100, 200])
    tau_values = []

    for h in h_values:
        gamma = 1.0 - 1.0 / h
        print(f"  h = {h} (γ = {gamma:.4f}) ... ", end="", flush=True)
        mdp = TradingMDP(n_prices=5, n_vol_regimes=2, n_positions=3,
                         sigma=1.0, gamma=gamma, horizon=h)
        tau = compute_threshold(mdp, delta=0.01, n_perturbation_dirs=30)
        tau_values.append(tau)
        print(f"τ = {tau:.6f}")

    tau_values = np.array(tau_values)

    valid = np.isfinite(tau_values) & (tau_values > 0)
    if valid.sum() < 3:
        print("WARNING: Not enough valid data points for power-law fit")
        return h_values, tau_values, None, None

    h_valid = h_values[valid]
    tau_valid = tau_values[valid]

    # Fit power law: log(τ) = -β log(h) + c
    log_h = np.log(h_valid.astype(float))
    log_tau = np.log(tau_valid)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_tau)
    beta = -slope
    C = np.exp(intercept)

    print(f"\n  Power law fit: τ = {C:.4f} · h^({slope:.4f})")
    print(f"  β = {beta:.4f}")
    print(f"  R² = {r_value**2:.6f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  std_err(slope) = {std_err:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(h_valid, tau_valid, 'go', markersize=8, label='Computed τ')
    h_fit = np.logspace(np.log10(h_valid.min()), np.log10(h_valid.max()), 100)
    axes[0].loglog(h_fit, C * h_fit**slope, 'r-', linewidth=2,
                   label=f'Fit: τ = {C:.3f}·h^({slope:.3f})')
    axes[0].set_xlabel('Decision horizon h', fontsize=12)
    axes[0].set_ylabel('Threshold τ', fontsize=12)
    axes[0].set_title(f'Misalignment Threshold vs Horizon\n(R²={r_value**2:.4f}, β={beta:.3f})', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    predicted = C * h_valid**slope
    residuals = (tau_valid - predicted) / predicted * 100
    axes[1].bar(range(len(h_valid)), residuals, color='seagreen')
    axes[1].set_xlabel('Data point index', fontsize=12)
    axes[1].set_ylabel('Relative residual (%)', fontsize=12)
    axes[1].set_title('Power-law fit residuals', fontsize=12)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/tau_vs_horizon.png', dpi=150, bbox_inches='tight')
    plt.close()

    return h_values, tau_values, beta, r_value**2


# ============================================================
# Experiment 3: Joint (σ, h) surface
# ============================================================

def experiment_joint_surface():
    """Verify joint power-law: τ ∝ σ^(-α) · h^(-β)."""
    print("\n" + "=" * 60)
    print("Experiment 3: Joint (σ, h) surface")
    print("=" * 60)

    sigma_values = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    h_values = np.array([5, 10, 20, 50, 100])

    results = np.zeros((len(sigma_values), len(h_values)))

    for i, sigma in enumerate(sigma_values):
        for j, h in enumerate(h_values):
            gamma = 1.0 - 1.0 / h
            print(f"  σ={sigma:.1f}, h={h} ... ", end="", flush=True)
            mdp = TradingMDP(n_prices=5, n_vol_regimes=2, n_positions=3,
                             sigma=sigma, gamma=gamma, horizon=h)
            tau = compute_threshold(mdp, delta=0.01, n_perturbation_dirs=20)
            results[i, j] = tau
            print(f"τ = {tau:.6f}")

    # Fit joint model: log(τ) = -α log(σ) - β log(h) + log(C)
    log_sigma_list = []
    log_h_list = []
    log_tau_list = []

    for i, sigma in enumerate(sigma_values):
        for j, h in enumerate(h_values):
            if np.isfinite(results[i, j]) and results[i, j] > 0:
                log_sigma_list.append(np.log(sigma))
                log_h_list.append(np.log(float(h)))
                log_tau_list.append(np.log(results[i, j]))

    if len(log_tau_list) < 5:
        print("WARNING: Not enough valid data points for joint fit")
        return results, None

    # Multiple linear regression: log(τ) = a0 + a1·log(σ) + a2·log(h)
    X = np.column_stack([np.ones(len(log_sigma_list)), log_sigma_list, log_h_list])
    y = np.array(log_tau_list)

    coeffs, residuals_sum, _, _ = np.linalg.lstsq(X, y, rcond=None)

    C_joint = np.exp(coeffs[0])
    alpha_joint = -coeffs[1]
    beta_joint = -coeffs[2]

    # R² calculation
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_joint = 1 - ss_res / ss_tot

    print(f"\n  Joint fit: τ = {C_joint:.4f} · σ^({-alpha_joint:.4f}) · h^({-beta_joint:.4f})")
    print(f"  α = {alpha_joint:.4f}, β = {beta_joint:.4f}")
    print(f"  R² = {r2_joint:.6f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_results = np.where(np.isfinite(results) & (results > 0), results, np.nan)
    im = ax.imshow(np.log10(valid_results), aspect='auto', cmap='viridis',
                   extent=[np.log10(h_values[0]), np.log10(h_values[-1]),
                           np.log10(sigma_values[-1]), np.log10(sigma_values[0])])
    ax.set_xlabel('log₁₀(h)', fontsize=12)
    ax.set_ylabel('log₁₀(σ)', fontsize=12)
    ax.set_title(f'log₁₀(τ) — Joint dependence\nτ = {C_joint:.2f}·σ^(-{alpha_joint:.2f})·h^(-{beta_joint:.2f}), R²={r2_joint:.4f}', fontsize=12)
    plt.colorbar(im, label='log₁₀(τ)')
    plt.tight_layout()
    plt.savefig('figures/tau_joint_surface.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results, {'C': C_joint, 'alpha': alpha_joint, 'beta': beta_joint, 'R2': r2_joint}


# ============================================================
# Experiment 4: Architecture dependence (spectral analysis)
# ============================================================

def experiment_architecture_dependence():
    """Test H4: α, β depend on spectral properties of transition kernel."""
    print("\n" + "=" * 60)
    print("Experiment 4: Architecture dependence of exponents")
    print("=" * 60)

    # Vary "architecture" by changing MDP structure parameters
    configs = [
        {"name": "Small (3×2×2)", "n_prices": 3, "n_vol_regimes": 2, "n_positions": 2},
        {"name": "Medium (5×2×3)", "n_prices": 5, "n_vol_regimes": 2, "n_positions": 3},
        {"name": "Large (7×3×3)", "n_prices": 7, "n_vol_regimes": 3, "n_positions": 3},
        {"name": "Wide (5×3×5)", "n_prices": 5, "n_vol_regimes": 3, "n_positions": 5},
    ]

    architecture_results = []

    for config in configs:
        print(f"\n  Architecture: {config['name']}")

        # Compute α from σ sweep
        sigma_values = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
        tau_sigma = []
        for sigma in sigma_values:
            mdp = TradingMDP(n_prices=config['n_prices'],
                           n_vol_regimes=config['n_vol_regimes'],
                           n_positions=config['n_positions'],
                           sigma=sigma, gamma=0.95, horizon=10)
            tau = compute_threshold(mdp, delta=0.01, n_perturbation_dirs=15)
            tau_sigma.append(tau)
        tau_sigma = np.array(tau_sigma)

        valid = np.isfinite(tau_sigma) & (tau_sigma > 0)
        if valid.sum() >= 3:
            slope_s, _, r_s, _, _ = stats.linregress(np.log(sigma_values[valid]), np.log(tau_sigma[valid]))
            alpha_arch = -slope_s
        else:
            alpha_arch = np.nan
            r_s = 0

        # Compute β from h sweep
        h_values = np.array([5, 10, 20, 50, 100])
        tau_h = []
        for h in h_values:
            gamma = 1.0 - 1.0 / h
            mdp = TradingMDP(n_prices=config['n_prices'],
                           n_vol_regimes=config['n_vol_regimes'],
                           n_positions=config['n_positions'],
                           sigma=1.0, gamma=gamma, horizon=h)
            tau = compute_threshold(mdp, delta=0.01, n_perturbation_dirs=15)
            tau_h.append(tau)
        tau_h = np.array(tau_h)

        valid_h = np.isfinite(tau_h) & (tau_h > 0)
        if valid_h.sum() >= 3:
            slope_h, _, r_h, _, _ = stats.linregress(np.log(h_values[valid_h].astype(float)), np.log(tau_h[valid_h]))
            beta_arch = -slope_h
        else:
            beta_arch = np.nan
            r_h = 0

        # Compute spectral gap of transition kernel at σ=1
        mdp_ref = TradingMDP(n_prices=config['n_prices'],
                            n_vol_regimes=config['n_vol_regimes'],
                            n_positions=config['n_positions'],
                            sigma=1.0, gamma=0.95, horizon=10)
        pi_ref = mdp_ref.solve_soft_policy(mdp_ref.R_proxy)
        P_ref = np.zeros((mdp_ref.n_states, mdp_ref.n_states))
        for s in range(mdp_ref.n_states):
            for a in range(mdp_ref.n_actions):
                P_ref[s] += pi_ref[s, a] * mdp_ref.T[s, a]

        eigenvalues = np.sort(np.abs(np.linalg.eigvals(P_ref)))[::-1]
        spectral_gap = 1 - eigenvalues[1] if len(eigenvalues) > 1 else 0

        result = {
            'name': config['name'],
            'n_states': mdp_ref.n_states,
            'alpha': alpha_arch,
            'beta': beta_arch,
            'R2_alpha': r_s**2 if not np.isnan(alpha_arch) else 0,
            'R2_beta': r_h**2 if not np.isnan(beta_arch) else 0,
            'spectral_gap': spectral_gap
        }
        architecture_results.append(result)
        print(f"    α = {alpha_arch:.4f} (R²={result['R2_alpha']:.4f}), "
              f"β = {beta_arch:.4f} (R²={result['R2_beta']:.4f}), "
              f"spectral_gap = {spectral_gap:.4f}")

    # Plot architecture comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r['name'] for r in architecture_results]
    alphas = [r['alpha'] for r in architecture_results]
    betas = [r['beta'] for r in architecture_results]
    gaps = [r['spectral_gap'] for r in architecture_results]

    x = range(len(names))
    axes[0].bar(x, alphas, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('α (volatility exponent)', fontsize=12)
    axes[0].set_title('Volatility exponent α by architecture', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, betas, color='seagreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('β (horizon exponent)', fontsize=12)
    axes[1].set_title('Horizon exponent β by architecture', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Scatter: spectral gap vs α, β
    axes[2].scatter(gaps, alphas, c='steelblue', s=80, label='α')
    axes[2].scatter(gaps, betas, c='seagreen', s=80, marker='s', label='β')
    axes[2].set_xlabel('Spectral gap', fontsize=12)
    axes[2].set_ylabel('Exponent value', fontsize=12)
    axes[2].set_title('Exponents vs spectral gap', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/architecture_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()

    return architecture_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"SciPy: {stats.scipy.__version__ if hasattr(stats, 'scipy') else 'N/A'}")
    print(f"Seed: {SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {}

    # Experiment 1: σ dependence
    sigma_vals, tau_sigma, alpha, r2_sigma = experiment_sigma_dependence()
    all_results['sigma_dependence'] = {
        'sigma_values': sigma_vals.tolist(),
        'tau_values': [float(t) if np.isfinite(t) else None for t in tau_sigma],
        'alpha': float(alpha) if alpha is not None else None,
        'R2': float(r2_sigma) if r2_sigma is not None else None,
    }

    # Experiment 2: h dependence
    h_vals, tau_h, beta, r2_h = experiment_horizon_dependence()
    all_results['horizon_dependence'] = {
        'h_values': h_vals.tolist(),
        'tau_values': [float(t) if np.isfinite(t) else None for t in tau_h],
        'beta': float(beta) if beta is not None else None,
        'R2': float(r2_h) if r2_h is not None else None,
    }

    # Experiment 3: Joint surface
    joint_results, joint_params = experiment_joint_surface()
    all_results['joint_surface'] = {
        'params': joint_params,
    }

    # Experiment 4: Architecture dependence
    arch_results = experiment_architecture_dependence()
    all_results['architecture_dependence'] = arch_results

    # Save all results
    with open('results/numerical_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to results/numerical_results.json")
    print(f"Figures saved to figures/")
