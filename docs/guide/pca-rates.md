# PCA Curve Shocks for Rates Risk

The classic Litterman–Scheinkman result is that the first three principal components of historical yield-curve returns explain ≥95% of pillar variance and admit a clean interpretation: **level**, **slope**, and **curvature**. VALAX exposes that workflow as `valax.risk.factors.RatesFactorModel`, which sits on top of the generic [`pca_jacobian`](../api/risk.md) primitive and plugs into the standard [`apply_scenario`](../api/risk.md) / [`pnl_attribution`](../api/risk.md) machinery.

For the underlying math see [Models & Theory § 7.8](../theory.md). For the runnable end-to-end example see [`examples/09_pca_rates_pnl.py`](https://github.com/ACea15/VALAX/blob/main/examples/09_pca_rates_pnl.py).

---

## 1. The pipeline at a glance

```
DiscountCurve snapshots
        │
        │  zero_rate_returns_from_snapshots(curves, query_dates)
        ▼
returns: Float[Array, "n_obs n_pillars"]
        │
        │  fit_rates_pca(returns, pillar_times, n_components=3)
        ▼
RatesFactorModel(pillar_times, jacobian, eigenvalues, fraction_explained)
        │
        │  model.shock_curve(curve, pc_scores)      ← single-curve path
        │  model.scenario(pc_scores, n_assets=...)  ← full MarketScenario path
        ▼
shocked curve  →  pricing_fn / pnl_attribution
```

Three layers, one Jacobian, no duplicated math.

---

## 2. Extract returns from a snapshot stack

```python
from valax.risk.factors import zero_rate_returns_from_snapshots

# query_dates is a fixed absolute pillar grid shared across snapshots.
returns = zero_rate_returns_from_snapshots(curves, query_dates)
#   returns.shape == (len(curves) - 1, len(query_dates))
```

Why absolute dates rather than tenors? Real yield-curve PCA is run on a fixed pillar grid that does not roll forward with the snapshots — the rolling time-to-maturity drift is part of what the PCs absorb. If you need a true constant-maturity series, resample each snapshot onto a shifted grid before calling `zero_rate_returns_from_snapshots`.

The result is in the same units as `MarketScenario.rate_shocks` — additive continuously-compounded zero-rate changes per pillar.

---

## 3. Fit the factor model

```python
from valax.risk.factors import fit_rates_pca

model = fit_rates_pca(returns, pillar_times, n_components=3)
```

Returned as a typed `eqx.Module` pytree, which means `jax.grad`, `jax.vmap`, and `eqx.filter_jit` all work on functions that close over `model`. The four data fields carry everything needed downstream:

| Field | Shape | What it holds |
|---|---|---|
| `pillar_times` | `(n_pillars,)` | Year fractions used for diagnostics and labelling. |
| `jacobian` | `(n_pillars, n_components)` | Orthonormal loading matrix; column `k` is PC<sub>k</sub>. |
| `eigenvalues` | `(n_components,)` | Variances of the PC scores; `sqrt(eigenvalues)` is the one-σ score. |
| `fraction_explained` | scalar | Fraction of total return variance captured. |

### Sign convention

PCA loadings are sign-ambiguous (`v` and `-v` are both valid eigenvectors). `fit_rates_pca` defaults to `sign_convention="positive_level"`, which flips columns so that every PC has a non-negative mean loading. That makes PC1 unambiguously "parallel up", PC2 a positive long-end slope, and so on — the standard rates-PCA orientation. Pass `sign_convention="raw"` to keep the SVD's raw signs.

### Diagnostics

Always inspect the per-pillar reconstruction quality before trusting the model:

```python
r2 = model.r_squared_per_pillar(returns)   # shape (n_pillars,)
```

On a well-behaved yield curve every entry should be ≥ 0.95; ≥ 0.99 at every pillar is the typical bar for production rates risk.

---

## 4. Shock a curve, price the trade

The single-curve path is `model.shock_curve(curve, pc_scores)`:

```python
import jax.numpy as jnp

sigmas = jnp.sqrt(model.eigenvalues)
scores = jnp.array([+1.0, +1.0, 0.0]) * sigmas   # +1σ level, +1σ slope
shocked = model.shock_curve(base_curve, scores)
pnl = bond_ladder_pv(shocked) - bond_ladder_pv(base_curve)
```

For the full risk pipeline (multi-asset `MarketData`, vol shocks, scenario sets, P&L attribution), use the `scenario` path:

```python
from valax.risk.shocks import apply_scenario
from valax.risk.var import pnl_attribution

scen = model.scenario(scores, n_assets=len(base_market.spots))
shocked_market = apply_scenario(base_market, scen)
attribution = pnl_attribution(pricing_fn, instruments, base_market, scen)
```

Both paths produce identical rate moves on the discount curve — the test suite asserts that bit-for-bit.

### One-shot autodiff

The whole shock-and-price pipeline is differentiable, so PC-score sensitivities come from a single `jax.grad`:

```python
import jax

def pnl(scores):
    return bond_ladder_pv(model.shock_curve(base_curve, scores)) \
         - bond_ladder_pv(base_curve)

pc_sensitivities = jax.grad(pnl)(jnp.zeros(3))   # (n_components,)
```

This is the natural risk-management object for a 3-factor view of the desk: one number per principal component instead of one number per pillar.

---

## 5. Building scenario sets

For VaR / stress testing, draw a batch of PC scores from a multivariate normal with diagonal covariance equal to `model.eigenvalues`, then stack the resulting scenarios:

```python
import jax
from valax.market.scenario import stack_scenarios

key = jax.random.PRNGKey(0)
scores = jax.random.normal(key, (1000, model.n_components)) \
       * jnp.sqrt(model.eigenvalues)
scenarios = stack_scenarios([
    model.scenario(s, n_assets=n_assets) for s in scores
])
```

The diagonal covariance is exact in PC coordinates — that's the whole point of using PCA factors as the working coordinate system for rates risk.

---

## 6. Relationship to other VALAX primitives

| Primitive | Lives in | What it does | When to use it directly |
|---|---|---|---|
| `pca_jacobian` | `valax.risk.bucketing` | Generic SVD-based loading matrix. | Bespoke or non-rates PCA (vol-surface factors, equity systematic factors). |
| `level_slope_curvature_jacobian` | `valax.risk.bucketing` | Fixed (non-data-driven) 3-factor basis. | When you want PC-like reparameterisation without enough history to fit. |
| `pca_curve_shock` | `valax.risk.shocks` | One-shot `(curve, J, scores) → DiscountCurve`. | You already have a Jacobian and just want the shock primitive. |
| `RatesFactorModel` | `valax.risk.factors` | Fitted typed model + scenario / shock methods. | **Default** for rates risk management. |

The LMM stochastic-rates model (`valax.models.lmm.compute_loading_matrix`) keeps its own correlation-matrix-based PCA — that workflow inputs a correlation matrix rather than a returns matrix and exists for forward-rate factor reduction, not historical risk. The two are intentionally not unified.
