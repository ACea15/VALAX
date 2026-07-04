# Risk

Scenario generation, shock application, and risk measures. Everything
is fully differentiable — `jax.grad` flows through `apply_scenario`
into the underlying `MarketData` fields, which is what powers
`parametric_var`'s delta-normal computation.

For `MarketData`, `MarketScenario`, `ScenarioSet`, `zero_scenario`,
and `stack_scenarios`, see [Market Data](market.md). This page covers
the perturbation and measurement layer built on top.

## Shock application

### Yield-curve shocks

::: valax.risk.shocks.bump_curve_zero_rates

::: valax.risk.shocks.parallel_shift

::: valax.risk.shocks.key_rate_bump

### Multi-curve shocks

Perturbations for the `MultiCurveSet` (OIS discount + tenor-keyed
forwards). `bump_discount_curve` shifts only the OIS leg;
`bump_forward_curve` shifts only the named forward curve, leaving
every other curve untouched. `parallel_basis_shift` is the
convenience constant-bump version of `bump_forward_curve`.

::: valax.risk.shocks.bump_discount_curve

::: valax.risk.shocks.bump_forward_curve

::: valax.risk.shocks.parallel_basis_shift

### Credit shocks

Perturbations for `SurvivalCurve`. `bump_hazard_rates` applies an
additive bump to each piecewise-constant hazard interval and
propagates through survival probabilities.
`parallel_credit_spread_shift` uses the credit triangle
\(\Delta h = \Delta s / (1 - R)\). `key_rate_hazard_bump` only
affects pillars at or beyond the targeted interval — the credit
analogue of an IR key-rate bump.

::: valax.risk.shocks.bump_hazard_rates

::: valax.risk.shocks.parallel_credit_spread_shift

::: valax.risk.shocks.key_rate_hazard_bump

### Full scenario application

::: valax.risk.shocks.apply_scenario

## Scenario generation

::: valax.risk.scenarios.parametric_scenarios

::: valax.risk.scenarios.historical_scenarios

::: valax.risk.scenarios.stress_scenario

### Curve-shape stress presets

::: valax.risk.scenarios.steepener

::: valax.risk.scenarios.flattener

::: valax.risk.scenarios.butterfly

## Pricing function adapter

Adapts equity-style `(instrument, spot, vol, rate, dividend) -> price`
pricers to the market-data-aware signature
`(instrument, MarketData) -> price` used by the risk engine.

::: valax.risk.var.wrap_equity_pricing_fn

## Risk measures

::: valax.risk.var.reprice_under_scenario

::: valax.risk.var.portfolio_pnl

::: valax.risk.var.value_at_risk

::: valax.risk.var.expected_shortfall

### Parametric (delta-normal) VaR

Delta-normal VaR using autodiff sensitivities. Portfolio gradient
w.r.t. all risk factors comes from `jax.grad`, then

\[
    \text{VaR} = z_\alpha \cdot \sqrt{\delta^\top \Sigma\, \delta}.
\]

Risk-factor ordering in `cov`: `[spots, vols, rates, dividends]`.
Discount-factor sensitivities are converted to zero-rate sensitivities
internally.

::: valax.risk.var.parametric_var

## P&L attribution

Decompose scenario P&L into risk-factor contributions using a
second-order Taylor expansion with autodiff sensitivities.

::: valax.risk.var.pnl_attribution

## P&L vectors

### Risk-theoretical P&L (RTPL)

Predicted P&L for each scenario from a precomputed `SensitivityLadder`.
Implements the 10-rung waterfall (delta, vega, rho, dividend,
gamma-spot, gamma-rate, vanna, volga, cross spot×rate, cross vol×rate)
as batched array contractions — one cheap pass per scenario, no
repricing. Respects `scenarios.multiplicative`.

::: valax.risk.pnl_vectors.risk_theoretical_pnl_vector

### Hypothetical P&L (HPL)

Full-revaluation P&L for each scenario. Alias for `portfolio_pnl`,
exposed under this name for clarity in P&L-explain and FRTB PLA
workflows.

::: valax.risk.pnl_vectors.hypothetical_pnl_vector

### Explain–unexplain

::: valax.risk.pnl_vectors.explained_unexplained_vector

## Sensitivity ladders

::: valax.risk.ladders.SensitivityLadder

::: valax.risk.ladders.WaterfallPnL

::: valax.risk.ladders.compute_ladder

::: valax.risk.ladders.waterfall_pnl

::: valax.risk.ladders.waterfall_pnl_report

## Backtesting

### VaR backtests

::: valax.risk.backtesting.var_breaches

::: valax.risk.backtesting.kupiec_pof

::: valax.risk.backtesting.christoffersen_independence

::: valax.risk.backtesting.christoffersen_conditional_coverage

::: valax.risk.backtesting.basel_traffic_light

### FRTB P&L attribution tests (BCBS d558)

Overall PLA zone: green if Spearman ≥ 0.80 **and** KS p-value ≥ 0.264;
red if Spearman < 0.70 or KS p-value < 0.055; amber otherwise. The
overall zone is the worse of the two test zones.

::: valax.risk.backtesting.pla_spearman

::: valax.risk.backtesting.ks_statistic

::: valax.risk.backtesting.pla_ks

::: valax.risk.backtesting.pla_traffic_light

## Bucketing

Linear aggregation and Jacobian reparameterization for risk-factor
coordinate changes. See
[Models & Theory §7.8](../theory.md#78-risk-bucketing-linear-and-jacobian-transformations)
for the mathematics.

### `BucketMap`

::: valax.risk.bucketing.BucketMap

### Linear aggregation operations

`aggregate` is \(A \delta\); `pushforward_scenario` is
\(A^\top \Delta b\) (the unique factor shock that preserves
bucket-space P&L); `aggregate_covariance` is \(A \Sigma A^\top\);
`aggregate_matrix` is the bilateral form \(A_r M A_c^\top\) used to
bucket cross-gamma blocks.

::: valax.risk.bucketing.aggregate

::: valax.risk.bucketing.pushforward_scenario

::: valax.risk.bucketing.aggregate_covariance

::: valax.risk.bucketing.aggregate_matrix

### Jacobian reparameterization

Same algebra, different semantics: \(J = \partial x / \partial b\) is
supplied (or autodiff-computed) instead of an aggregation matrix.

::: valax.risk.bucketing.pushforward_sensitivities

::: valax.risk.bucketing.pullback_shocks

::: valax.risk.bucketing.reparameterize_covariance

::: valax.risk.bucketing.jacobian_from_fn

### Bucket builders

`tenor_bucket_map` with `weight="indicator"` produces the standard
FRTB / SIMM nearest-vertex bucketing; with `weight="linear"` it gives
a smooth piecewise-linear re-binning.

::: valax.risk.bucketing.tenor_bucket_map

::: valax.risk.bucketing.equal_weight_bucket_map

::: valax.risk.bucketing.level_slope_curvature_jacobian

::: valax.risk.bucketing.pca_jacobian

### Rates-PCA workflow

`RatesFactorModel` is the typed end-to-end wrapper over `pca_jacobian`
for yield-curve risk: it carries `pillar_times`, `jacobian`,
`eigenvalues`, and `fraction_explained`, and exposes `shock_curve`
and `scenario` methods that plug straight into `apply_scenario` /
`pnl_attribution`. See the
[PCA Curve Shocks guide](../guide/pca-rates.md) for the end-to-end
pattern.

::: valax.risk.factors.RatesFactorModel

::: valax.risk.factors.fit_rates_pca

::: valax.risk.factors.zero_rate_returns_from_snapshots

::: valax.risk.shocks.pca_curve_shock

### Bucketed sensitivity ladders

Apply independent bucket maps to every component of a
`SensitivityLadder`. Cross blocks (`cross_spot_rate`,
`cross_vol_rate`) are bilaterally bucketed. The returned
`BucketedLadder` pytree preserves the bucket labels for reporting.

::: valax.risk.bucketing.BucketedLadder

::: valax.risk.bucketing.bucket_sensitivity_ladder
