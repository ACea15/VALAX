# Risk

Scenario generation, shock application, and risk measures.

## Market Data

### `MarketData`

```python
class MarketData(eqx.Module):
    spots: Float[Array, "n_assets"]
    vols: Float[Array, "n_assets"]
    dividends: Float[Array, "n_assets"]
    discount_curve: DiscountCurve
```

Complete market state. All fields are differentiable.

### `MarketScenario`

```python
class MarketScenario(eqx.Module):
    spot_shocks: Float[Array, "n_assets"]
    vol_shocks: Float[Array, "n_assets"]
    rate_shocks: Float[Array, "n_pillars"]
    dividend_shocks: Float[Array, "n_assets"]
    multiplicative: bool = False  # static
```

Additive risk factor changes. When `multiplicative=True`, spot shocks are returns.

### `ScenarioSet`

Same fields as `MarketScenario` but with a leading `(n_scenarios, ...)` axis on every leaf. Designed for `jax.vmap`.

### `zero_scenario`

```python
zero_scenario(n_assets, n_pillars) -> MarketScenario
```

No-op scenario (all shocks zero).

### `stack_scenarios`

```python
stack_scenarios(scenarios: list[MarketScenario]) -> ScenarioSet
```

Stack a list of single scenarios into a batched `ScenarioSet`.

---

## Shock Application

### `bump_curve_zero_rates`

```python
bump_curve_zero_rates(curve, rate_bumps) -> DiscountCurve
```

Apply additive zero-rate bumps at each pillar: $DF_{\text{new}}(t_i) = DF_{\text{old}}(t_i) \cdot e^{-\Delta r_i \cdot t_i}$.

### `parallel_shift`

```python
parallel_shift(curve, bump) -> DiscountCurve
```

Uniform bump to all pillar zero rates.

### `key_rate_bump`

```python
key_rate_bump(curve, pillar_index, bump) -> DiscountCurve
```

Bump a single pillar's zero rate.

### Multi-curve shocks

```python
bump_discount_curve(mcs: MultiCurveSet, rate_bumps) -> MultiCurveSet
bump_forward_curve(mcs: MultiCurveSet, tenor: str, rate_bumps) -> MultiCurveSet
parallel_basis_shift(mcs: MultiCurveSet, tenor: str, bump) -> MultiCurveSet
```

Shock primitives for the `MultiCurveSet` (OIS discount + tenor-keyed forwards). `bump_discount_curve` shifts only the OIS leg; `bump_forward_curve` shifts only the named forward curve, leaving every other curve untouched. `parallel_basis_shift` is the convenience constant-bump version of `bump_forward_curve` — a pure `IR.BASIS.<ccy>.<tenor>.OIS` parallel move.

### Credit shocks (SurvivalCurve)

```python
bump_hazard_rates(curve: SurvivalCurve, hazard_bumps) -> SurvivalCurve
parallel_credit_spread_shift(curve, spread_bump, recovery_rate=0.4) -> SurvivalCurve
key_rate_hazard_bump(curve, pillar_index, bump) -> SurvivalCurve
```

Credit-curve perturbations. `bump_hazard_rates` applies an additive bump to each piecewise-constant hazard interval and propagates the effect through survival probabilities. `parallel_credit_spread_shift` converts a CDS-spread move into a hazard bump via the credit triangle Δh = Δs / (1 − R). `key_rate_hazard_bump` only affects pillars at or beyond the targeted interval — the credit analogue of an IR key-rate bump.

### `apply_scenario`

```python
apply_scenario(base: MarketData, scenario: MarketScenario) -> MarketData
```

Apply all shocks (spot, vol, rate, dividend) to produce a new market state. Fully differentiable.

---

## Scenario Generation

### `parametric_scenarios`

```python
parametric_scenarios(key, cov, n_scenarios, n_assets, n_pillars,
                     distribution="normal", df=5.0) -> ScenarioSet
```

Correlated samples via Cholesky decomposition. Column ordering in `cov`: `[spots, vols, rates, dividends]`. Supports `"normal"` and `"t"` distributions.

### `historical_scenarios`

```python
historical_scenarios(returns, n_assets, n_pillars) -> ScenarioSet
```

Slice observed daily changes (shape `(n_obs, n_factors)`) into a `ScenarioSet`.

### `stress_scenario`

```python
stress_scenario(n_assets, n_pillars, spot_shock=0.0, vol_shock=0.0,
                parallel_rate_shift=0.0, rate_shocks=None,
                dividend_shock=0.0) -> MarketScenario
```

Deterministic stress with optional custom per-pillar rate shocks.

### `steepener`

```python
steepener(n_assets, n_pillars, short_bump, long_bump) -> MarketScenario
```

Linear rate profile from `short_bump` to `long_bump`.

### `flattener`

```python
flattener(n_assets, n_pillars, short_bump, long_bump) -> MarketScenario
```

Linear rate profile (typically short up, long down).

### `butterfly`

```python
butterfly(n_assets, n_pillars, wing_bump, belly_bump) -> MarketScenario
```

Quadratic rate profile: wings at `wing_bump`, belly at `belly_bump`.

---

## Pricing Function Adapter

### `wrap_equity_pricing_fn`

```python
wrap_equity_pricing_fn(fn) -> Callable
```

Adapts an equity-style pricing function with signature `(instrument, spot, vol, rate, dividend) -> price` to the market-data-aware signature `(instrument, MarketData) -> price` used by the risk engine. Extracts a scalar rate from the discount curve's shortest maturity.

---

## Risk Measures

### `reprice_under_scenario`

```python
reprice_under_scenario(pricing_fn, instruments, base, scenario) -> Float[Array, ""]
```

Reprice a portfolio under a single scenario. The `pricing_fn` must have signature `(instrument, MarketData) -> price`. Each instrument receives per-asset `MarketData` (scalar spot, vol, dividend) with the shared discount curve via `jax.vmap`.

### `portfolio_pnl`

```python
portfolio_pnl(pricing_fn, instruments, base, scenarios) -> Float[Array, "n_scenarios"]
```

Compute P&L under each scenario via `jax.vmap`. Returns `portfolio_value(scenario_i) - portfolio_value(base)`.

### `value_at_risk`

```python
value_at_risk(pnl, confidence=0.99) -> Float[Array, ""]
```

VaR: negative of the $(1 - \alpha)$ quantile. Positive VaR indicates a loss threshold.

### `expected_shortfall`

```python
expected_shortfall(pnl, confidence=0.99) -> Float[Array, ""]
```

CVaR: mean of losses beyond VaR. Always $\geq$ VaR.

### `parametric_var`

```python
parametric_var(pricing_fn, instruments, base, cov, confidence=0.99) -> Float[Array, ""]
```

Delta-normal VaR using autodiff sensitivities. Computes the portfolio gradient w.r.t. all risk factors via `jax.grad`, then:

$$\text{VaR} = z_\alpha \cdot \sqrt{\delta^T \Sigma \delta}$$

where $\delta$ is the sensitivity vector and $\Sigma$ is the covariance matrix. Risk factor ordering in `cov`: `[spots, vols, rates, dividends]`. DF sensitivities are converted to zero-rate sensitivities internally.

---

## P&L Attribution

### `pnl_attribution`

```python
pnl_attribution(pricing_fn, instruments, base, scenario) -> dict[str, Float[Array, ""]]
```

Decompose a scenario's P&L into risk factor contributions using a second-order Taylor expansion with autodiff sensitivities. Returns:

| Key | Description |
|-----|-------------|
| `delta_spot` | First-order spot contribution |
| `delta_vol` | First-order vol contribution (vega) |
| `delta_rate` | First-order rate contribution (rho/DV01) |
| `delta_div` | First-order dividend contribution |
| `gamma_spot` | Second-order spot convexity |
| `total_first_order` | Sum of all delta terms |
| `total_second_order` | First order + gamma |
| `actual` | True P&L from full repricing |
| `unexplained` | Actual − second-order approximation |

---

## P&L Vectors

### `risk_theoretical_pnl_vector`

```python
risk_theoretical_pnl_vector(ladder, scenarios, base) -> Float[Array, "n_scenarios"]
```

Predicted P&L for each scenario from a precomputed `SensitivityLadder`. Implements the 10-rung waterfall (delta, vega, rho, dividend, gamma_spot, gamma_rate, vanna, volga, cross spot×rate, cross vol×rate) as batched array contractions — one cheap pass per scenario, no repricing. Respects `scenarios.multiplicative`.

### `hypothetical_pnl_vector`

```python
hypothetical_pnl_vector(pricing_fn, instruments, base, scenarios) -> Float[Array, "n_scenarios"]
```

Full-revaluation P&L for each scenario. Alias for `portfolio_pnl`, exposed under this name for clarity in P&L-explain and FRTB PLA workflows.

### `explained_unexplained_vector`

```python
explained_unexplained_vector(pricing_fn, instruments, base, scenarios, ladder=None) -> dict
```

Returns a dict with three `(n_scenarios,)` vectors: `"rtpl"` (ladder prediction), `"hpl"` (full revaluation), and `"unexplained" = hpl - rtpl`. If `ladder` is `None`, it is computed internally.

---

## Backtesting

### `var_breaches`

```python
var_breaches(actual_pnl, var_forecast) -> Bool[Array, "n_days"]
```

Boolean breach sequence: `-actual_pnl > var_forecast` per day. `actual_pnl` is signed (negative = loss), `var_forecast` is a non-negative loss threshold.

### `kupiec_pof`

```python
kupiec_pof(breaches, confidence=0.99) -> dict
```

Kupiec proportion-of-failures LR test for unconditional coverage. Returns `{"n", "x", "lr_uc", "p_value"}`. `lr_uc` is asymptotically χ²₁; reject the model at 5% if `lr_uc > 3.84`.

### `christoffersen_independence`

```python
christoffersen_independence(breaches) -> dict
```

Christoffersen LR test for independence of breaches (no clustering). Returns `{"lr_ind", "p_value", "n00", "n01", "n10", "n11"}`, asymptotically χ²₁.

### `christoffersen_conditional_coverage`

```python
christoffersen_conditional_coverage(breaches, confidence=0.99) -> dict
```

Joint test of correct breach rate AND independence: `lr_cc = lr_uc + lr_ind ~ χ²₂`. Returns `{"lr_cc", "p_value", "lr_uc", "lr_ind"}`.

### `basel_traffic_light`

```python
basel_traffic_light(n_breaches, n_obs=250, confidence=0.99) -> str
```

Returns `"green" | "yellow" | "red"`. Uses fixed thresholds 4 / 9 for the regulatory `(250, 0.99)` window; for non-standard parameters the cumulative binomial 95% / 99.99% quantiles are recomputed.

### `pla_spearman`

```python
pla_spearman(rtpl, hpl) -> Float[Array, ""]
```

Spearman rank correlation of two P&L vectors.

### `ks_statistic`

```python
ks_statistic(x, y) -> Float[Array, ""]
```

Two-sample Kolmogorov–Smirnov distance between empirical CDFs.

### `pla_ks`

```python
pla_ks(rtpl, hpl) -> Float[Array, ""]
```

Alias for `ks_statistic` under the FRTB name.

### `pla_traffic_light`

```python
pla_traffic_light(spearman, ks_stat, n_obs=250, ...) -> str
```

BCBS d558 PLA zone: green if Spearman ≥ 0.80 *and* KS p-value ≥ 0.264; red if Spearman < 0.70 or KS p-value < 0.055; amber otherwise. The overall zone is the worse of the two test zones.

---

## Bucketing

Linear aggregation and Jacobian reparameterization for risk-factor coordinate changes. See [Models & Theory § 7.8](../theory.md#78-risk-bucketing-linear-and-jacobian-transformations) for the mathematics.

### `BucketMap`

```python
class BucketMap(eqx.Module):
    matrix: Float[Array, "n_buckets n_factors"]
    bucket_labels: tuple = ()   # static
    factor_labels: tuple = ()   # static
```

Linear aggregation matrix `A` plus optional labels.

### Linear aggregation operations

```python
aggregate(bm: BucketMap, sensitivities) -> bucketed
pushforward_scenario(bm: BucketMap, bucket_shocks) -> factor_shocks
aggregate_covariance(bm: BucketMap, cov) -> bucketed_cov
aggregate_matrix(bm_rows: BucketMap, M, bm_cols: BucketMap) -> bilaterally_bucketed
```

`aggregate` is `A @ δ`; `pushforward_scenario` is `Aᵀ @ Δb` (the unique factor shock that preserves bucket-space P&L); `aggregate_covariance` is `A @ Σ @ Aᵀ`; `aggregate_matrix` is the bilateral form `A_r @ M @ A_cᵀ` used to bucket cross-gamma blocks.

### Jacobian reparameterization operations

```python
pushforward_sensitivities(J, sensitivities) -> bucketed       # Jᵀ @ δ
pullback_shocks(J, bucket_shocks) -> factor_shocks            # J @ Δb
reparameterize_covariance(J, cov) -> bucketed_cov             # Jᵀ @ Σ @ J
jacobian_from_fn(b_to_x, b_base) -> jacobian                  # jax.jacobian wrapper
```

Same algebra, different semantics: `J = ∂x/∂b` is supplied (or autodiff-computed) instead of an aggregation matrix.

### Bucket builders

```python
tenor_bucket_map(pillar_times, bucket_edges, weight="indicator"|"linear") -> BucketMap
equal_weight_bucket_map(group_membership, n_buckets, bucket_labels=()) -> BucketMap
level_slope_curvature_jacobian(pillar_times) -> Float[Array, "n_pillars 3"]
pca_jacobian(returns, n_components, center=True) -> (J, eigvals, frac_explained)
```

`tenor_bucket_map` with `weight="indicator"` produces the standard FRTB / SIMM nearest-vertex bucketing; with `weight="linear"` it gives a smooth piecewise-linear re-binning. `equal_weight_bucket_map` is the canonical "one-of-N" sector / currency / rating assignment. `level_slope_curvature_jacobian` produces a fixed 3-column Jacobian; `pca_jacobian` produces a data-driven one with orthonormal columns, sorted by decreasing eigenvalue.

### Rates-PCA workflow

```python
fit_rates_pca(returns, pillar_times, n_components=3,
              center=True, sign_convention="positive_level") -> RatesFactorModel
zero_rate_returns_from_snapshots(curves, query_dates) -> Float[Array, "n_obs n_pillars"]
pca_curve_shock(curve, jacobian, pc_scores) -> DiscountCurve
```

`RatesFactorModel` is the typed end-to-end wrapper over `pca_jacobian` for yield-curve risk: it carries `pillar_times`, `jacobian`, `eigenvalues`, and `fraction_explained`, and exposes `shock_curve(curve, scores)` and `scenario(scores, n_assets)` methods that plug straight into `apply_scenario` / `pnl_attribution`. `zero_rate_returns_from_snapshots` builds the input matrix from a stack of `DiscountCurve` snapshots. `pca_curve_shock` is the underlying single-call shock primitive — `model.shock_curve` is a thin wrapper around it. See the [PCA Curve Shocks guide](../guide/pca-rates.md) for the end-to-end pattern.

### `bucket_sensitivity_ladder`

```python
bucket_sensitivity_ladder(
    ladder,
    *,
    rate_bucket: BucketMap | None = None,
    spot_bucket: BucketMap | None = None,
    vol_bucket: BucketMap | None = None,
    div_bucket: BucketMap | None = None,
) -> BucketedLadder
```

Apply independent bucket maps to every component of a `SensitivityLadder`. Missing arguments default to identity bucketing (no change). Cross blocks (`cross_spot_rate`, `cross_vol_rate`) are bilaterally bucketed. The returned `BucketedLadder` pytree preserves the bucket labels for reporting.
