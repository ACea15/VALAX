# Risk Bucketing

*Re-expressing fine-grained autodiff sensitivities in coarser, standardised coordinates — FRTB/SIMM vertices, trader buckets, PCA factors. Prerequisite: [Bucketed Sensitivities](risk-sensitivities.md). Factor registry: [Risk Factors § 7](risk-factors.md#7-bucketing). Maths: [Models & Theory § 7.8](theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations).*

## Risk Bucketing

Raw autodiff sensitivities live in the *finest* possible factor space — one DV01 per pillar, one vega per asset, one credit delta per hazard pillar. For reporting, regulatory capital, and stable VaR, every desk re-expresses the same risk in coarser coordinates:

- **Regulatory bucketing** (FRTB SBA, ISDA SIMM) — standard tenor vertices, equity sectors, credit rating buckets.
- **Trader-friendly bucketing** — "short / belly / wings" on a curve, "tech / energy / financials" on equities.
- **Factor reduction** — PCA scores (level / slope / curvature) for well-conditioned VaR on a 30-pillar curve.
- **Parametric vol bucketing** — push grid-vol sensitivities into SABR or SVI parameter space via the calibration Jacobian.

VALAX provides two transformation families in `valax/risk/bucketing.py`. They share the same matrix algebra, but they correspond to two different mental models and are named accordingly. See [Models & Theory § 7.8](theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations) for the derivation.

### Linear aggregation: `BucketMap`

A `BucketMap` is a thin wrapper around an aggregation matrix `A` of shape `(n_buckets, n_factors)`:

```python
from valax.risk import BucketMap, aggregate, pushforward_scenario, aggregate_covariance
import jax.numpy as jnp

# Map 4 pillars → 2 buckets (short / long)
A = jnp.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
])
rate_bm = BucketMap(matrix=A, bucket_labels=("short", "long"))

# Bucket the key-rate DV01s
bucket_dv01 = aggregate(rate_bm, sens.delta_rate)           # shape (2,)

# Push a bucket-level shock back to raw factors (the dual operation)
bucket_shock = jnp.array([0.0010, 0.0025])                  # +10 bp / +25 bp
raw_shock = pushforward_scenario(rate_bm, bucket_shock)     # shape (4,)

# Aggregate a 4×4 covariance to a 2×2 bucket covariance
bucket_cov = aggregate_covariance(rate_bm, raw_cov)
```

`pushforward_scenario` is the *dual* of `aggregate`: it is the unique factor shock that makes the bucket P&L equal to the raw P&L. Use it when you have a stress defined in bucket terms (e.g. "+10 bp on the short tenor") and need to apply it via `apply_scenario` / `portfolio_pnl`.

### Standard bucket builders

```python
from valax.risk import tenor_bucket_map, equal_weight_bucket_map

# FRTB-style tenor vertices with indicator weights
pillar_times = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
frtb_vertices = jnp.array([1.0, 5.0, 30.0])  # short, belly, long
rate_bm = tenor_bucket_map(pillar_times, frtb_vertices, weight="indicator")

# Linear (piecewise-linear) splitting — smooth re-binning
rate_bm_smooth = tenor_bucket_map(pillar_times, frtb_vertices, weight="linear")

# Sector / currency / rating bucketing
sector_bm = equal_weight_bucket_map(
    group_membership=(0, 0, 1, 1, 0),    # 5 stocks → 2 sectors
    n_buckets=2,
    bucket_labels=("tech", "energy"),
)
```

### Jacobian reparameterization

When the new factors are a smooth (possibly nonlinear) function of the raw factors — PCA scores, level/slope/curvature, SVI/SABR parameters — `BucketMap` is not the right abstraction. Use a Jacobian instead:

```python
from valax.risk import (
    pushforward_sensitivities,
    pullback_shocks,
    reparameterize_covariance,
    jacobian_from_fn,
    level_slope_curvature_jacobian,
    pca_jacobian,
)

# Hand-built level / slope / curvature factors (fixed Jacobian)
J = level_slope_curvature_jacobian(pillar_times)  # (n_pillars, 3)
lsc_delta = pushforward_sensitivities(J, sens.delta_rate)  # 3 numbers

# PCA on 250 days of curve returns → top 3 components
J_pca, eigvals, frac_explained = pca_jacobian(
    curve_returns,        # shape (250, n_pillars)
    n_components=3,
)
pca_delta = pushforward_sensitivities(J_pca, sens.delta_rate)
pca_cov = reparameterize_covariance(J_pca, raw_cov)

# Stress scenario in PCA space: +1 std on level only
pca_shock = jnp.array([1.0, 0.0, 0.0]) * jnp.sqrt(eigvals)
raw_shock_from_pca = pullback_shocks(J_pca, pca_shock)
```

For nonlinear reparameterizations (e.g. an SVI vol-grid function), build the Jacobian on-the-fly via autodiff:

```python
def svi_to_grid(svi_params):
    # svi_params -> vol[grid_strike, expiry] reconstruction
    return svi_vol_function(svi_params, strikes, expiries).flatten()

J = jacobian_from_fn(svi_to_grid, svi_params_base)
# Now J is the autodiff Jacobian; push grid-vega through it.
svi_param_vega = pushforward_sensitivities(J, raw_grid_vega)
```

For the rates-focused PCA workflow (level/slope/curvature scores from historical curve moves), see the [PCA Curve Shocks (Rates)](guide/pca-rates.md) guide.

### Bucketing a full sensitivity set

`bucket_sensitivities` applies independent bucket maps to every component of a `FlatSensitivities`, including bilateral aggregation of the cross-gamma blocks:

```python
from valax.risk import compute_flat_sensitivities, bucket_sensitivities

sens = compute_flat_sensitivities(pricing_fn, instruments, base_market)

bucketed = bucket_sensitivities(
    sens,
    rate_bucket=rate_bm,        # bucket pillar dimension
    spot_bucket=sector_bm,      # bucket asset dimension
    # vol_bucket / div_bucket: omitted ⇒ left at full granularity
)

bucketed.delta_rate          # (n_rate_buckets,)
bucketed.delta_spot          # (n_spot_buckets,)
bucketed.cross_spot_rate     # (n_spot_buckets, n_rate_buckets) — bilaterally bucketed
bucketed.rate_bucket_labels  # human-readable bucket names
```

The `BucketedSensitivities` result is itself a pytree — it can be fed back into `waterfall_pnl`-style arithmetic when paired with bucket-level scenarios.

### When to use which

| Goal | Use |
|---|---|
| Sum DV01s into FRTB tenor vertices | `tenor_bucket_map` + `aggregate` |
| Aggregate equity Greeks by sector | `equal_weight_bucket_map` + `aggregate` |
| Apply a bucket-level stress to raw factors | `pushforward_scenario` |
| Project a 30 × 30 covariance to a 10 × 10 regulatory covariance | `aggregate_covariance` |
| Yield-curve PCA factors (level/slope/curvature from data) | `pca_jacobian` + `pushforward_sensitivities` |
| Hand-picked level/slope/curvature | `level_slope_curvature_jacobian` + `pushforward_sensitivities` |
| SABR/SVI parameter Greeks from grid Greeks | `jacobian_from_fn` + `pushforward_sensitivities` |
| Stress in bucket / PC coords, evaluate raw P&L | `pushforward_scenario` (linear) or `pullback_shocks` (Jacobian) |
| Variance-reduced parametric VaR on a sparse factor set | `pca_jacobian` + `reparameterize_covariance` |
