# Computational Tools

## Installed Packages

| Tool | Purpose | Location | Notes |
|------|---------|----------|-------|
| SymPy 1.14.0 | Symbolic computation | pip package | For perturbation expansions, threshold derivations, symbolic manipulation of τ expressions |
| NumPy 2.4.4 | Numerical computation | pip package | For numerical verification of power-law fits, matrix operations |
| SciPy 1.17.1 | Scientific computing | pip package | For optimization, statistical testing, curve fitting (power law regression) |
| NetworkX 3.6.1 | Graph theory | pip package | For modeling multi-agent trading system structure |
| Matplotlib 3.10.8 | Visualization | pip package | For plotting scaling relationships, regime diagrams |

## Usage Notes

All packages are installed in the local `.venv` virtual environment. Activate with:
```bash
source .venv/bin/activate
```

### Key computational tasks these tools support:

1. **Perturbation expansion** (SymPy): Expand agent value functions around aligned baseline as series in intervention intensity parameter.
2. **Power law fitting** (SciPy): `scipy.optimize.curve_fit` for fitting τ = C · σ^(-α) · h^(-β) to simulation data.
3. **Eigenvalue analysis** (NumPy/SciPy): Spectral analysis of transition kernels for architecture-dependent constants.
4. **Statistical testing** (SciPy): Hypothesis testing for significance of power-law relationships.
5. **Agent network modeling** (NetworkX): If multi-agent interaction structure affects thresholds.
