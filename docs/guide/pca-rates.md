# PCA Curve Shocks for Rates Risk

The classic Litterman–Scheinkman result is that the first three principal components of historical yield-curve returns explain ≥95% of pillar variance and admit a clean interpretation: **level**, **slope**, and **curvature**. VALAX exposes that workflow as `valax.risk.factors.RatesFactorModel`, which sits on top of the generic [`pca_jacobian`](../api/risk.md) primitive. Its loadings feed a [`KeyedScenario`](../api/market.md#keyed-scenarios-for-marketstate) applied to a `MarketState` by [`shock`](../api/risk.md#keyed-shocks-on-a-marketstate) and repriced through `price()` — the current risk path — and its `scenario` method still produces a positional `MarketScenario` for the legacy [`apply_scenario`](../api/risk.md#full-scenario-application) / [`pnl_attribution`](../api/risk.md#pl-attribution) machinery on `MarketData`, which is scheduled for removal.

For the underlying math see [Models & Theory § 7.8](../theory/index.md). For the runnable end-to-end example — PCA fitted to a curve history, then PC and basis moves applied as `KeyedScenario`s to a dual-curve book on `MarketState` — see [`examples/19_unified_book_workflow.py`](../examples.md#a-rates-book-end-to-end).

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
        │  model.shock_curve(curve, pc_scores)      ← one-curve kernel path
        │  KeyedScenario(curve_shocks={id: J @ s})  ← MarketState path (shock → price)
        │  model.scenario(pc_scores, n_assets=...)  ← legacy MarketScenario path
        ▼
shocked curve / MarketState  →  price() / group_scenario_pnl
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

The result is in the same units as `KeyedScenario.curve_shocks` (and the legacy `MarketScenario.rate_shocks`) — additive continuously-compounded zero-rate changes per pillar.

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

The one-curve kernel path is `model.shock_curve(curve, pc_scores)`, on a bare `DiscountCurve`:

```python
import jax.numpy as jnp

sigmas = jnp.sqrt(model.eigenvalues)
scores = jnp.array([+1.0, +1.0, 0.0]) * sigmas   # +1σ level, +1σ slope
shocked = model.shock_curve(base_curve, scores)
pnl = bond_ladder_pv(shocked) - bond_ladder_pv(base_curve)
```

On a `MarketState` — the market `price()` takes — put the loadings into a `KeyedScenario` addressed to the curve they were fitted on. The Jacobian's rows must line up with that curve's pillars (example 19 prepends a zero row for a reference pillar that never moves). On a dual-curve market, name both curves, or add a basis move on the projection curve only:

```python
from valax.core import price
from valax.market import KeyedScenario
from valax.risk import shock

dz = model.jacobian @ scores                        # one bump per curve pillar
moved = shock(market, KeyedScenario(curve_shocks={"USD.SOFR.OIS": dz,
                                                  "USD.SOFR.3M": dz}))
pnl = price(swap, None, moved).price - price(swap, None, market).price
```

**Legacy path.** `model.scenario(...)` returns a positional `MarketScenario` for the flat `MarketData` pipeline (`apply_scenario`, `pnl_attribution`). It can only move the default curve and goes with `MarketData` after one deprecation cycle:

```python
from valax.risk.shocks import apply_scenario
from valax.risk.var import pnl_attribution

scen = model.scenario(scores, n_assets=len(base_market.spots))
shocked_market = apply_scenario(base_market, scen)
attribution = pnl_attribution(pricing_fn, instruments, base_market, scen)
```

`shock_curve` and the `scenario` path produce identical rate moves on the discount curve — the test suite asserts that bit-for-bit.

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

For VaR / stress testing, draw a batch of PC scores from a multivariate normal with diagonal covariance equal to `model.eigenvalues`. On a `MarketState`, one `KeyedScenario` whose leaves carry a leading scenario axis is the whole set, and `group_scenario_pnl` reprices a group under it through `price()`:

```python
import jax
from valax.market import KeyedScenario
from valax.risk import group_scenario_pnl, value_at_risk

key = jax.random.PRNGKey(0)
scores = jax.random.normal(key, (1000, model.n_components)) \
       * jnp.sqrt(model.eigenvalues)
scenarios = KeyedScenario(curve_shocks={"USD.SOFR.OIS": scores @ model.jacobian.T})
pnl = group_scenario_pnl(swaps, None, market, scenarios)   # (1000,)
var_99 = value_at_risk(pnl, 0.99)
```

`examples/19_unified_book_workflow.py` runs exactly this on a dual-curve book. The legacy equivalent stacks `model.scenario(s, n_assets=n_assets)` with `valax.market.stack_scenarios` into a `ScenarioSet` for the flat `MarketData` pipeline.

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
