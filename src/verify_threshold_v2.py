"""
Improved numerical verification of misalignment sensitivity threshold theory.

Uses larger state spaces and finer epsilon grids to observe continuous
power-law behavior of τ vs σ and h.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
from datetime import datetime

SEED = 42
np.random.seed(SEED)


class ContinuousTradingMDP:
    """
    Trading MDP with larger state space for smoother threshold behavior.

    Uses n_prices price levels to approximate continuous price dynamics.
    The key improvement: σ directly controls transition kernel entropy,
    making the effect on policy sensitivity measurable.
    """

    def __init__(self, n_prices=20, n_positions=5, sigma=1.0, gamma=0.95, temperature=0.5):
        self.n_prices = n_prices
        self.n_positions = n_positions
        self.n_states = n_prices * n_positions
        self.n_actions = 3  # sell, hold, buy
        self.sigma = sigma
        self.gamma = gamma
        self.temperature = temperature
        self._build_transitions()
        self._build_rewards()

    def _state_index(self, price, pos):
        return price * self.n_positions + pos

    def _decode_state(self, idx):
        pos = idx % self.n_positions
        price = idx // self.n_positions
        return price, pos

    def _build_transitions(self):
        """Build transitions with σ-controlled spread."""
        nS, nA = self.n_states, self.n_actions
        self.T = np.zeros((nS, nA, nS))

        for s in range(nS):
            price, pos = self._decode_state(s)
            for a in range(nA):
                new_pos = max(0, min(self.n_positions - 1, pos + (a - 1)))

                # Price transition: discretized Gaussian with std = σ
                # Higher σ → wider spread → higher entropy
                for new_price in range(self.n_prices):
                    dp = new_price - price
                    # Gaussian kernel with width proportional to σ
                    p = np.exp(-dp**2 / (2 * self.sigma**2 + 0.01))
                    new_s = self._state_index(new_price, new_pos)
                    self.T[s, a, new_s] = p

        # Normalize
        for s in range(nS):
            for a in range(nA):
                total = self.T[s, a].sum()
                if total > 0:
                    self.T[s, a] /= total

    def _build_rewards(self):
        """Reward: position * price_return - risk_penalty."""
        nS, nA = self.n_states, self.n_actions
        self.R_proxy = np.zeros((nS, nA))
        self.R_true = np.zeros((nS, nA))

        mid_price = self.n_prices / 2
        mid_pos = self.n_positions / 2

        for s in range(nS):
            price, pos = self._decode_state(s)
            for a in range(nA):
                new_pos = max(0, min(self.n_positions - 1, pos + (a - 1)))
                price_signal = (price - mid_price) / self.n_prices
                position_signal = (new_pos - mid_pos) / self.n_positions

                self.R_proxy[s, a] = price_signal * position_signal
                risk = 0.2 * abs(position_signal) * (1 + self.sigma * 0.1)
                self.R_true[s, a] = self.R_proxy[s, a] - risk

    def solve_soft_policy(self, reward):
        """Softmax value iteration."""
        nS, nA = self.n_states, self.n_actions
        V = np.zeros(nS)
        T = self.temperature

        for iteration in range(500):
            Q = reward + self.gamma * np.einsum('sab,b->sa', self.T, V)
            Q_shifted = Q - Q.max(axis=1, keepdims=True)
            exp_Q = np.exp(Q_shifted / T)
            V_new = T * np.log(exp_Q.sum(axis=1))

            if np.max(np.abs(V_new - V)) < 1e-12:
                break
            V = V_new

        Q_final = reward + self.gamma * np.einsum('sab,b->sa', self.T, V)
        Q_shifted = Q_final - Q_final.max(axis=1, keepdims=True)
        exp_Q = np.exp(Q_shifted / T)
        pi = exp_Q / exp_Q.sum(axis=1, keepdims=True)
        return pi

    def stationary_distribution(self, pi):
        """Compute stationary distribution under policy."""
        nS = self.n_states
        P = np.zeros((nS, nS))
        for s in range(nS):
            for a in range(self.n_actions):
                P[s] += pi[s, a] * self.T[s, a]

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        d = np.real(eigenvectors[:, idx])
        d = np.abs(d)
        d /= d.sum()
        return d

    def kl_divergence(self, pi1, pi2):
        """KL divergence D_KL(pi1 || pi2) weighted by stationary distribution."""
        d = self.stationary_distribution(pi1)
        eps_floor = 1e-15
        pi1_safe = np.clip(pi1, eps_floor, 1.0)
        pi2_safe = np.clip(pi2, eps_floor, 1.0)
        kl = np.sum(d[:, None] * pi1_safe * np.log(pi1_safe / pi2_safe))
        return max(kl, 0.0)

    def fisher_information_norm(self, pi, Phi):
        """
        Compute ||G||²_F = Σ_s d(s) Σ_a (G(s,a))²/π(a|s)
        where G(s,a) = (1/T)·π(a|s)·(Φ(a) - E_π[Φ](s))

        This is the Fisher information metric that controls τ.
        """
        d = self.stationary_distribution(pi)
        T = self.temperature

        # E_π[Φ](s) = Σ_a π(a|s)·Φ(s,a)
        E_Phi = np.sum(pi * Phi, axis=1)  # |S|-dim

        # G(s,a) = (1/T)·π(a|s)·(Φ(s,a) - E_Phi(s))
        G = (1/T) * pi * (Phi - E_Phi[:, None])

        # Fisher norm: Σ_s d(s) Σ_a G(s,a)²/π(a|s)
        fisher = np.sum(d[:, None] * G**2 / np.clip(pi, 1e-15, 1.0))
        return fisher


def compute_threshold_bisection(mdp, delta=0.01, n_dirs=50, eps_range=(1e-6, 10.0)):
    """
    Compute τ using bisection for each perturbation direction.
    Returns the minimum threshold over all directions (worst case).
    """
    pi_0 = mdp.solve_soft_policy(mdp.R_proxy)
    min_tau = np.inf

    for trial in range(n_dirs):
        Phi = np.random.randn(mdp.n_states, mdp.n_actions)
        Phi /= np.max(np.abs(Phi))

        # Bisection: find smallest ε where D_KL > δ
        lo, hi = eps_range

        # First check if hi is large enough
        R_hi = mdp.R_proxy + hi * Phi
        pi_hi = mdp.solve_soft_policy(R_hi)
        div_hi = mdp.kl_divergence(pi_hi, pi_0)

        if div_hi <= delta:
            continue  # This direction doesn't cross threshold

        # Check if lo already crosses
        R_lo = mdp.R_proxy + lo * Phi
        pi_lo = mdp.solve_soft_policy(R_lo)
        div_lo = mdp.kl_divergence(pi_lo, pi_0)

        if div_lo > delta:
            min_tau = min(min_tau, lo)
            continue

        # Bisection
        for _ in range(40):
            mid = np.sqrt(lo * hi)  # Geometric mean for log-scale
            R_mid = mdp.R_proxy + mid * Phi
            pi_mid = mdp.solve_soft_policy(R_mid)
            div_mid = mdp.kl_divergence(pi_mid, pi_0)

            if div_mid > delta:
                hi = mid
            else:
                lo = mid

            if hi / lo < 1.01:
                break

        min_tau = min(min_tau, hi)

    return min_tau


def compute_theoretical_tau(mdp, delta=0.01, n_dirs=50):
    """
    Compute theoretical τ from Fisher information: τ = √(2δ/F_max)
    where F_max is the maximum Fisher information over perturbation directions.
    """
    pi_0 = mdp.solve_soft_policy(mdp.R_proxy)
    max_fisher = 0.0

    for _ in range(n_dirs):
        Phi = np.random.randn(mdp.n_states, mdp.n_actions)
        Phi /= np.max(np.abs(Phi))

        F = mdp.fisher_information_norm(pi_0, Phi)
        max_fisher = max(max_fisher, F)

    if max_fisher > 0:
        tau_theoretical = np.sqrt(2 * delta / max_fisher)
    else:
        tau_theoretical = np.inf

    return tau_theoretical, max_fisher


# ============================================================
# Experiment 1: τ vs σ
# ============================================================

def experiment_sigma(n_prices=20, n_positions=5, temperature=0.5):
    """Measure τ(σ) both numerically and theoretically."""
    print("=" * 60)
    print("Experiment 1: τ vs σ (volatility dependence)")
    print(f"  n_prices={n_prices}, n_positions={n_positions}, T={temperature}")
    print("=" * 60)

    sigma_values = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    tau_numerical = []
    tau_theoretical = []
    fisher_values = []

    for sigma in sigma_values:
        print(f"  σ = {sigma:.2f} ... ", end="", flush=True)
        mdp = ContinuousTradingMDP(n_prices=n_prices, n_positions=n_positions,
                                    sigma=sigma, gamma=0.95, temperature=temperature)

        tau_n = compute_threshold_bisection(mdp, delta=0.01, n_dirs=40)
        tau_t, F = compute_theoretical_tau(mdp, delta=0.01, n_dirs=40)

        tau_numerical.append(tau_n)
        tau_theoretical.append(tau_t)
        fisher_values.append(F)
        print(f"τ_num = {tau_n:.6f}, τ_theory = {tau_t:.6f}, F = {F:.4f}")

    tau_numerical = np.array(tau_numerical)
    tau_theoretical = np.array(tau_theoretical)
    fisher_values = np.array(fisher_values)

    results = {}
    for name, tau_vals in [('numerical', tau_numerical), ('theoretical', tau_theoretical)]:
        valid = np.isfinite(tau_vals) & (tau_vals > 0) & (tau_vals < 100)
        if valid.sum() >= 4:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(sigma_values[valid]), np.log(tau_vals[valid]))
            alpha = -slope
            C = np.exp(intercept)
            print(f"\n  {name}: τ = {C:.4f}·σ^({slope:.4f}), α={alpha:.4f}, R²={r_value**2:.4f}, p={p_value:.2e}")
            results[name] = {'alpha': alpha, 'C': C, 'R2': r_value**2, 'p': p_value}
        else:
            print(f"\n  {name}: insufficient valid data points ({valid.sum()})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid_n = np.isfinite(tau_numerical) & (tau_numerical > 0) & (tau_numerical < 100)
    valid_t = np.isfinite(tau_theoretical) & (tau_theoretical > 0) & (tau_theoretical < 100)

    if valid_n.any():
        axes[0].loglog(sigma_values[valid_n], tau_numerical[valid_n], 'bo-', markersize=7, label='Numerical τ')
    if valid_t.any():
        axes[0].loglog(sigma_values[valid_t], tau_theoretical[valid_t], 'rs--', markersize=7, label='Theoretical τ')
    if 'numerical' in results:
        r = results['numerical']
        s_fit = np.logspace(np.log10(sigma_values.min()), np.log10(sigma_values.max()), 100)
        axes[0].loglog(s_fit, r['C'] * s_fit**(-r['alpha']), 'b:', alpha=0.5,
                       label=f'Fit: σ^(-{r["alpha"]:.3f}), R²={r["R2"]:.3f}')
    axes[0].set_xlabel('σ (volatility)')
    axes[0].set_ylabel('τ (threshold)')
    axes[0].set_title('Misalignment threshold vs volatility')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Fisher information vs σ
    axes[1].loglog(sigma_values, fisher_values, 'g^-', markersize=7)
    axes[1].set_xlabel('σ (volatility)')
    axes[1].set_ylabel('F (Fisher information)')
    axes[1].set_title('Fisher information vs volatility')
    axes[1].grid(True, alpha=0.3)

    # Theoretical vs numerical comparison
    if valid_n.any() and valid_t.any():
        both_valid = valid_n & valid_t
        if both_valid.any():
            axes[2].scatter(tau_theoretical[both_valid], tau_numerical[both_valid], c='purple', s=60)
            lims = [min(tau_theoretical[both_valid].min(), tau_numerical[both_valid].min()) * 0.8,
                    max(tau_theoretical[both_valid].max(), tau_numerical[both_valid].max()) * 1.2]
            axes[2].plot(lims, lims, 'k--', alpha=0.5, label='y=x')
            axes[2].set_xlabel('Theoretical τ')
            axes[2].set_ylabel('Numerical τ')
            axes[2].set_title('Theory vs computation')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figures/tau_vs_sigma_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    return sigma_values, tau_numerical, tau_theoretical, fisher_values, results


# ============================================================
# Experiment 2: τ vs h (via γ = 1 - 1/h)
# ============================================================

def experiment_horizon(n_prices=20, n_positions=5, temperature=0.5):
    """Measure τ(h) both numerically and theoretically."""
    print("\n" + "=" * 60)
    print("Experiment 2: τ vs h (horizon dependence)")
    print("=" * 60)

    h_values = np.array([3, 5, 8, 10, 15, 20, 30, 50, 80, 100])
    tau_numerical = []
    tau_theoretical = []

    for h in h_values:
        gamma = 1.0 - 1.0 / h
        print(f"  h = {h} (γ = {gamma:.4f}) ... ", end="", flush=True)
        mdp = ContinuousTradingMDP(n_prices=n_prices, n_positions=n_positions,
                                    sigma=1.0, gamma=gamma, temperature=temperature)

        tau_n = compute_threshold_bisection(mdp, delta=0.01, n_dirs=40)
        tau_t, F = compute_theoretical_tau(mdp, delta=0.01, n_dirs=40)

        tau_numerical.append(tau_n)
        tau_theoretical.append(tau_t)
        print(f"τ_num = {tau_n:.6f}, τ_theory = {tau_t:.6f}")

    tau_numerical = np.array(tau_numerical)
    tau_theoretical = np.array(tau_theoretical)

    results = {}
    for name, tau_vals in [('numerical', tau_numerical), ('theoretical', tau_theoretical)]:
        valid = np.isfinite(tau_vals) & (tau_vals > 0) & (tau_vals < 100)
        if valid.sum() >= 4:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(h_values[valid].astype(float)), np.log(tau_vals[valid]))
            beta = -slope
            C = np.exp(intercept)
            print(f"\n  {name}: τ = {C:.4f}·h^({slope:.4f}), β={beta:.4f}, R²={r_value**2:.4f}, p={p_value:.2e}")
            results[name] = {'beta': beta, 'C': C, 'R2': r_value**2, 'p': p_value}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_n = np.isfinite(tau_numerical) & (tau_numerical > 0) & (tau_numerical < 100)
    valid_t = np.isfinite(tau_theoretical) & (tau_theoretical > 0) & (tau_theoretical < 100)

    if valid_n.any():
        ax.loglog(h_values[valid_n], tau_numerical[valid_n], 'bo-', markersize=7, label='Numerical τ')
    if valid_t.any():
        ax.loglog(h_values[valid_t], tau_theoretical[valid_t], 'rs--', markersize=7, label='Theoretical τ')
    if 'numerical' in results:
        r = results['numerical']
        h_fit = np.logspace(np.log10(h_values.min()), np.log10(h_values.max()), 100)
        ax.loglog(h_fit, r['C'] * h_fit**(-r['beta']), 'b:', alpha=0.5,
                  label=f'Fit: h^(-{r["beta"]:.3f}), R²={r["R2"]:.3f}')
    ax.set_xlabel('Decision horizon h')
    ax.set_ylabel('Threshold τ')
    ax.set_title('Misalignment threshold vs decision horizon')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/tau_vs_horizon_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    return h_values, tau_numerical, tau_theoretical, results


# ============================================================
# Experiment 3: Architecture dependence via spectral gap
# ============================================================

def experiment_architecture():
    """Measure how exponents depend on MDP structure."""
    print("\n" + "=" * 60)
    print("Experiment 3: Architecture dependence")
    print("=" * 60)

    configs = [
        {"name": "Tiny", "n_prices": 8, "n_positions": 3, "T": 0.3},
        {"name": "Small", "n_prices": 12, "n_positions": 4, "T": 0.5},
        {"name": "Medium", "n_prices": 20, "n_positions": 5, "T": 0.5},
        {"name": "Large", "n_prices": 30, "n_positions": 5, "T": 0.5},
        {"name": "Cold", "n_prices": 20, "n_positions": 5, "T": 0.1},
        {"name": "Warm", "n_prices": 20, "n_positions": 5, "T": 1.0},
    ]

    arch_results = []

    for cfg in configs:
        print(f"\n  Config: {cfg['name']} (n_p={cfg['n_prices']}, n_pos={cfg['n_positions']}, T={cfg['T']})")

        # Compute α from σ sweep
        sigma_vals = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        taus_sigma = []
        for sigma in sigma_vals:
            mdp = ContinuousTradingMDP(n_prices=cfg['n_prices'], n_positions=cfg['n_positions'],
                                        sigma=sigma, gamma=0.95, temperature=cfg['T'])
            tau_t, _ = compute_theoretical_tau(mdp, delta=0.01, n_dirs=30)
            taus_sigma.append(tau_t)
        taus_sigma = np.array(taus_sigma)

        valid = np.isfinite(taus_sigma) & (taus_sigma > 0) & (taus_sigma < 100)
        if valid.sum() >= 3:
            slope, _, r_val, p_val, _ = stats.linregress(np.log(sigma_vals[valid]), np.log(taus_sigma[valid]))
            alpha = -slope
            r2_alpha = r_val**2
        else:
            alpha, r2_alpha = np.nan, np.nan

        # Compute β from h sweep
        h_vals = np.array([5, 10, 20, 50, 100])
        taus_h = []
        for h in h_vals:
            gamma = 1.0 - 1.0 / h
            mdp = ContinuousTradingMDP(n_prices=cfg['n_prices'], n_positions=cfg['n_positions'],
                                        sigma=1.0, gamma=gamma, temperature=cfg['T'])
            tau_t, _ = compute_theoretical_tau(mdp, delta=0.01, n_dirs=30)
            taus_h.append(tau_t)
        taus_h = np.array(taus_h)

        valid_h = np.isfinite(taus_h) & (taus_h > 0) & (taus_h < 100)
        if valid_h.sum() >= 3:
            slope_h, _, r_val_h, p_val_h, _ = stats.linregress(np.log(h_vals[valid_h].astype(float)), np.log(taus_h[valid_h]))
            beta = -slope_h
            r2_beta = r_val_h**2
        else:
            beta, r2_beta = np.nan, np.nan

        # Spectral gap
        mdp_ref = ContinuousTradingMDP(n_prices=cfg['n_prices'], n_positions=cfg['n_positions'],
                                        sigma=1.0, gamma=0.95, temperature=cfg['T'])
        pi_ref = mdp_ref.solve_soft_policy(mdp_ref.R_proxy)
        P_ref = np.zeros((mdp_ref.n_states, mdp_ref.n_states))
        for s in range(mdp_ref.n_states):
            for a in range(mdp_ref.n_actions):
                P_ref[s] += pi_ref[s, a] * mdp_ref.T[s, a]
        eigs = np.sort(np.abs(np.linalg.eigvals(P_ref)))[::-1]
        spec_gap = 1 - eigs[1] if len(eigs) > 1 else 0

        entry = {
            'name': cfg['name'],
            'n_states': mdp_ref.n_states,
            'temperature': cfg['T'],
            'alpha': float(alpha),
            'r2_alpha': float(r2_alpha),
            'beta': float(beta),
            'r2_beta': float(r2_beta),
            'spectral_gap': float(spec_gap),
        }
        arch_results.append(entry)
        print(f"    α={alpha:.4f} (R²={r2_alpha:.4f}), β={beta:.4f} (R²={r2_beta:.4f}), Δ={spec_gap:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [r['name'] for r in arch_results]
    alphas = [r['alpha'] for r in arch_results]
    betas = [r['beta'] for r in arch_results]
    gaps = [r['spectral_gap'] for r in arch_results]

    x = range(len(names))
    axes[0].bar(x, alphas, color='steelblue', alpha=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('α (volatility exponent)')
    axes[0].set_title('Exponent α across architectures')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, betas, color='seagreen', alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('β (horizon exponent)')
    axes[1].set_title('Exponent β across architectures')
    axes[1].grid(True, alpha=0.3, axis='y')

    # α and β vs spectral gap
    valid_a = [i for i in range(len(alphas)) if not np.isnan(alphas[i])]
    valid_b = [i for i in range(len(betas)) if not np.isnan(betas[i])]

    if valid_a:
        axes[2].scatter([gaps[i] for i in valid_a], [alphas[i] for i in valid_a],
                       c='steelblue', s=80, label='α', zorder=3)
    if valid_b:
        axes[2].scatter([gaps[i] for i in valid_b], [betas[i] for i in valid_b],
                       c='seagreen', s=80, marker='s', label='β', zorder=3)
    axes[2].set_xlabel('Spectral gap Δ')
    axes[2].set_ylabel('Exponent value')
    axes[2].set_title('Exponents vs spectral gap')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/architecture_dependence_v2.png', dpi=150, bbox_inches='tight')
    plt.close()

    return arch_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Seed: {SEED}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {}

    # Experiment 1: σ dependence
    s_vals, tau_n_s, tau_t_s, fisher_s, res_s = experiment_sigma()
    all_results['sigma'] = {
        'sigma_values': s_vals.tolist(),
        'tau_numerical': [float(t) if np.isfinite(t) else None for t in tau_n_s],
        'tau_theoretical': [float(t) if np.isfinite(t) else None for t in tau_t_s],
        'fisher': fisher_s.tolist(),
        'fits': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res_s.items()},
    }

    # Experiment 2: h dependence
    h_vals, tau_n_h, tau_t_h, res_h = experiment_horizon()
    all_results['horizon'] = {
        'h_values': h_vals.tolist(),
        'tau_numerical': [float(t) if np.isfinite(t) else None for t in tau_n_h],
        'tau_theoretical': [float(t) if np.isfinite(t) else None for t in tau_t_h],
        'fits': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res_h.items()},
    }

    # Experiment 3: Architecture dependence
    arch_res = experiment_architecture()
    all_results['architecture'] = arch_res

    # Save
    with open('results/numerical_results_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE (V2)")
    print("=" * 60)
    print("Results: results/numerical_results_v2.json")
    print("Figures: figures/tau_vs_sigma_v2.png, figures/tau_vs_horizon_v2.png, figures/architecture_dependence_v2.png")
