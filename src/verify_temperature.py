"""
Experiment: Threshold dependence on softmax temperature.

Tests the revised hypothesis that τ ∝ T (proportional to policy temperature),
with only weak dependence on σ and h.
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys

np.random.seed(42)

# Import from v2
sys.path.insert(0, 'src')
from verify_threshold_v2 import ContinuousTradingMDP, compute_theoretical_tau, compute_threshold_bisection


def experiment_temperature():
    """Test τ vs T (softmax temperature)."""
    print("=" * 60)
    print("Experiment: τ vs T (policy temperature)")
    print("=" * 60)

    T_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    tau_numerical = []
    tau_theoretical = []

    for T in T_values:
        print(f"  T = {T:.2f} ... ", end="", flush=True)
        mdp = ContinuousTradingMDP(n_prices=20, n_positions=5,
                                    sigma=1.0, gamma=0.95, temperature=T)
        tau_n = compute_threshold_bisection(mdp, delta=0.01, n_dirs=40)
        tau_t, F = compute_theoretical_tau(mdp, delta=0.01, n_dirs=40)
        tau_numerical.append(tau_n)
        tau_theoretical.append(tau_t)
        print(f"τ_num = {tau_n:.6f}, τ_theory = {tau_t:.6f}, F = {F:.4f}")

    tau_numerical = np.array(tau_numerical)
    tau_theoretical = np.array(tau_theoretical)

    # Fit power law: τ ∝ T^κ
    for name, tvals in [('numerical', tau_numerical), ('theoretical', tau_theoretical)]:
        valid = np.isfinite(tvals) & (tvals > 0) & (tvals < 100)
        if valid.sum() >= 4:
            slope, intercept, r_val, p_val, se = stats.linregress(
                np.log(T_values[valid]), np.log(tvals[valid]))
            print(f"\n  {name}: τ ∝ T^{slope:.4f}, R²={r_val**2:.4f}, p={p_val:.2e}")

    # Also test asymptotic σ behavior at extreme values
    print("\n" + "=" * 60)
    print("Asymptotic σ test (very small and very large σ)")
    print("=" * 60)

    sigma_extreme = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    tau_extreme = []
    for sigma in sigma_extreme:
        mdp = ContinuousTradingMDP(n_prices=30, n_positions=5,
                                    sigma=sigma, gamma=0.95, temperature=0.3)
        tau_t, F = compute_theoretical_tau(mdp, delta=0.01, n_dirs=30)
        tau_extreme.append(tau_t)
        print(f"  σ={sigma:6.2f}: τ_theory={tau_t:.6f}, F={F:.6f}")

    tau_extreme = np.array(tau_extreme)

    valid_e = np.isfinite(tau_extreme) & (tau_extreme > 0) & (tau_extreme < 100)
    if valid_e.sum() >= 4:
        slope_e, _, r_e, p_e, _ = stats.linregress(
            np.log(sigma_extreme[valid_e]), np.log(tau_extreme[valid_e]))
        print(f"\n  Extended σ range: τ ∝ σ^{slope_e:.4f}, α={-slope_e:.4f}, R²={r_e**2:.4f}, p={p_e:.2e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # τ vs T
    valid_n = np.isfinite(tau_numerical) & (tau_numerical > 0) & (tau_numerical < 100)
    valid_t = np.isfinite(tau_theoretical) & (tau_theoretical > 0) & (tau_theoretical < 100)

    if valid_n.any():
        axes[0].loglog(T_values[valid_n], tau_numerical[valid_n], 'bo-', markersize=7, label='Numerical')
    if valid_t.any():
        axes[0].loglog(T_values[valid_t], tau_theoretical[valid_t], 'rs--', markersize=7, label='Theoretical')

    # Fit line
    if valid_t.any():
        slope_t, intercept_t, r_t, _, _ = stats.linregress(
            np.log(T_values[valid_t]), np.log(tau_theoretical[valid_t]))
        T_fit = np.logspace(np.log10(T_values.min()), np.log10(T_values.max()), 100)
        axes[0].loglog(T_fit, np.exp(intercept_t) * T_fit**slope_t, 'r:', alpha=0.5,
                       label=f'τ ∝ T^{slope_t:.2f} (R²={r_t**2:.3f})')

    axes[0].set_xlabel('Softmax Temperature T', fontsize=12)
    axes[0].set_ylabel('Threshold τ', fontsize=12)
    axes[0].set_title('τ vs Policy Temperature\n(Primary architecture-dependent factor)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Extended σ range
    if valid_e.any():
        axes[1].loglog(sigma_extreme[valid_e], tau_extreme[valid_e], 'g^-', markersize=7, label='Theoretical τ')
        s_fit = np.logspace(np.log10(sigma_extreme[valid_e].min()),
                            np.log10(sigma_extreme[valid_e].max()), 100)
        axes[1].loglog(s_fit, np.exp(np.polyval([slope_e, np.log(tau_extreme[valid_e]).mean() - slope_e * np.log(sigma_extreme[valid_e]).mean()], np.log(s_fit))),
                       'r:', alpha=0.5, label=f'τ ∝ σ^{slope_e:.3f}')
    axes[1].set_xlabel('σ (extended range)', fontsize=12)
    axes[1].set_ylabel('Threshold τ', fontsize=12)
    axes[1].set_title('Extended σ range test', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/tau_vs_temperature.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results = {
        'T_values': T_values.tolist(),
        'tau_numerical': [float(t) if np.isfinite(t) else None for t in tau_numerical],
        'tau_theoretical': [float(t) if np.isfinite(t) else None for t in tau_theoretical],
        'sigma_extreme': sigma_extreme.tolist(),
        'tau_extreme': [float(t) if np.isfinite(t) else None for t in tau_extreme],
    }
    with open('results/temperature_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/temperature_results.json")
    print("Figure saved to figures/tau_vs_temperature.png")


if __name__ == "__main__":
    experiment_temperature()
