# Calibration

Model parameter fitting via gradient-based optimization.

## `calibrate_sabr`

```python
from valax.calibration import calibrate_sabr

fitted_model, solution = calibrate_sabr(
    strikes: Float[Array, " n"],          # option strikes
    market_vols: Float[Array, " n"],      # observed implied vols
    forward: Float[Array, ""],            # forward price
    expiry: Float[Array, ""],             # time to expiry
    initial_guess: SABRModel | None,      # starting point (default: heuristic)
    fixed_beta: Float[Array, ""] | None,  # fix beta (recommended)
    weights: Float[Array, " n"] | None,   # per-strike weights (default: uniform)
    solver: str = "levenberg_marquardt",   # "levenberg_marquardt", "bfgs", "optax_adam"
    max_steps: int = 256,
) -> tuple[SABRModel, optimistix.Solution]
```

Returns the fitted `SABRModel` and an `optimistix.Solution` with convergence diagnostics.

## `calibrate_heston`

```python
from valax.calibration import calibrate_heston

fitted_model, solution = calibrate_heston(
    strikes: Float[Array, " n"],          # option strikes
    market_prices: Float[Array, " n"],    # observed option prices
    spot: Float[Array, ""],               # spot price
    rate: Float[Array, ""],               # risk-free rate
    dividend: Float[Array, ""],           # dividend yield
    expiry: Float[Array, ""],             # time to expiry
    pricing_fn: Callable,                 # Heston pricer (injected)
    initial_guess: HestonModel | None,    # starting point
    weights: Float[Array, " n"] | None,   # per-strike weights
    solver: str = "levenberg_marquardt",
    max_steps: int = 512,
) -> tuple[HestonModel, optimistix.Solution]
```

The `pricing_fn` must have signature `(model, strike, spot, rate, dividend, expiry) -> price`.

## `calibrate_slv_leverage`

Pass-2 calibration of the SLV leverage function $L(k, t)$ via Markovian projection of the SLV SDE onto Dupire local volatility. The conditional expectation $\mathbb{E}[V_t \mid k_t = k]$ is estimated from a simulated particle swarm; the calibrated $L$ satisfies $L^2(k, t)\,\mathbb{E}[V_t \mid k_t = k] = \sigma_{\mathrm{Dupire}}^2(k, t)$ by construction. See [theory §4.5](../theory.md#45-stochastic-local-volatility) and the [SLV guide](../guide/slv.md) for the mathematics.

```python
from valax.calibration import calibrate_slv_leverage

leverage = calibrate_slv_leverage(
    heston:              HestonModel,                # pass-1 calibrated Heston
    surface,                                          # any total_variance(k,T) surface
    spot:                Float[Array, ""],
    log_moneyness_grid:  Float[Array, "n_k"],        # sorted, ascending
    time_grid:           Float[Array, "n_t"],        # sorted, ascending (> 0)
    n_paths:             int,
    key:                 jax.Array,
    *,
    method:              Literal["particle", "kernel"] = "particle",
    n_iterations:        int = 1,                    # outer fixed-point iterations
    bandwidth:           float | Callable | None = None,   # default: Silverman per slice
    ridge:               float = 1e-3,               # used only for method="kernel"
    L_max:               float = 5.0,
    L_min:               float = 0.05,
) -> LeverageGrid
```

`method="particle"` uses pure Nadaraya-Watson; `method="kernel"` adds a Tikhonov ridge that biases the estimator toward the empirical particle mean in low-density regions (smoother tails, mildly biased centre). `n_iterations=1` recovers the classical one-shot Guyon-Henry-Labordère (2012) particle method; `n_iterations ≥ 2` re-simulates the swarm under the previous-iteration leverage and rebuilds the grid for tighter self-consistency.

## `calibrate_slv`

End-to-end two-pass SLV calibration. Wraps `calibrate_heston` (Pass 1) and `calibrate_slv_leverage` (Pass 2) into one call returning a fully-built `SLVModel`.

```python
from valax.calibration import calibrate_slv

slv_model = calibrate_slv(
    # Pass 1 args (forwarded to calibrate_heston):
    strikes:               Float[Array, " n"],
    market_prices:         Float[Array, " n"],
    spot, rate, dividend, expiry, pricing_fn,
    # Pass 2 args (forwarded to calibrate_slv_leverage):
    surface,
    log_moneyness_grid, time_grid, n_paths, key,
    *,
    method:                Literal["particle", "kernel"] = "particle",
    n_iterations:          int = 1,
    bandwidth:             float | Callable | None = None,
    ridge:                 float = 1e-3,
    heston_initial_guess:  HestonModel | None = None,
    heston_solver:         str = "levenberg_marquardt",
    heston_max_steps:      int = 512,
) -> SLVModel
```

See the [SLV guide](../guide/slv.md) for a worked end-to-end example and discussion of estimator choice / convergence behaviour.

## Transforms

Reparametrization utilities for mapping constrained parameters to unconstrained space.

```python
from valax.calibration import positive, bounded, correlation, unit_interval

positive()          # x > 0 (softplus)
bounded(lo, hi)     # lo < x < hi (scaled sigmoid)
unit_interval()     # 0 < x < 1 (sigmoid)
correlation()       # -1 < x < 1 (tanh)
```

### Model-level helpers

```python
from valax.calibration import model_to_unconstrained, unconstrained_to_model, SABR_TRANSFORMS

raw = model_to_unconstrained(sabr_model, SABR_TRANSFORMS)
# raw = {"alpha": ..., "beta": ..., "rho": ..., "nu": ...}

recovered = unconstrained_to_model(raw, SABR_TRANSFORMS, template=sabr_model)
```

## Loss Functions

```python
from valax.calibration import vol_residuals, price_residuals, weighted_sse
```

These follow the `optimistix` signature `fn(y, args) -> residuals` and are used internally by the calibration routines. They can also be used directly for custom calibration workflows.
