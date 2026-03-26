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
