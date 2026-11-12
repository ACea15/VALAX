# Risk

Scenario generation, shock application, and risk measures. Everything
is fully differentiable — `jax.grad` flows through `shock` into the
underlying `MarketState` (and through `apply_scenario` into the legacy
`MarketData` fields), which is what powers both `group_market_gradient`
and `parametric_var`'s delta-normal computation.

For `MarketData`, `MarketScenario`, `ScenarioSet`, `KeyedScenario`,
`zero_scenario`, and `stack_scenarios`, see [Market Data](market.md).
This page covers the perturbation and measurement layer built on top —
both the keyed shocks on a `MarketState` (the current API) and the legacy
positional shocks on a `MarketData`, which are scheduled for removal after
one deprecation cycle. Single-instrument market risk is
[`valax.greeks.market_risk`](greeks.md#valax.greeks.market_risk).

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

### Curve-graph shocks

Perturbations for the [`CurveGraph`](curves.md#valax.curves.CurveGraph)
container: bump any named curve by identifier while every other curve
on the graph is held fixed. Whether the move is a discounting shock or
a basis move is purely a matter of which `curve_id` is passed.

::: valax.risk.shocks.bump_graph_curve

::: valax.risk.shocks.parallel_graph_shift

::: valax.risk.shocks.key_rate_graph_bump

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

`apply_scenario` is the **legacy** (positional) container-level shock: it
maps the four flat `MarketScenario` vectors onto a `MarketData`. Because
`rate_shocks` carries no curve identity it can only move the default
curve — `discount_curve` plus the `"_default_"` entry of an attached
`curve_graph`, which are the same curve seen through two views. Named
projection / basis curves on the graph are carried through unshocked; use
the keyed API below for those. It is removed together with `MarketData`
after one deprecation cycle ([roadmap](../roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)).

::: valax.risk.shocks.apply_scenario

## Keyed shocks on a `MarketState`

The keyed counterpart: `shock` applies a
[`KeyedScenario`](market.md#keyed-scenarios-for-marketstate) to a
[`MarketState`](core.md#marketstate), addressing curves, surfaces, spots
and dividends **by identifier**. Curve moves are routed through
`bump_graph_curve` above, so an OIS shock and a projection-curve (basis)
shock are two separately expressible scenarios on one graph. Vol moves go
through `shift_surface`, a parallel shift that is uniform across every
surface type. The result is a new pytree (`eqx.tree_at`, never mutation)
and is differentiable in both the base market and the shock magnitudes.

The guide walkthrough, including the surface design fork and what it
implies for the sensitivity ladders, is in
[Scenarios & Shocks § Keyed scenarios](../risk-scenarios.md#keyed-scenarios-on-a-marketstate).

::: valax.risk.state_shocks.shock

::: valax.risk.state_shocks.shift_surface

## Scenario generation

!!! note "Legacy flat generators"
    These draw `ScenarioSet`s over the positional
    `[spots, vols, rates, dividends]` layout of a `MarketData` and stay for
    one deprecation cycle. On a `MarketState` use the
    [keyed generators](#keyed-generators) below, which address every named
    curve separately. `curve_shape_bumps` is shared by both layouts.

::: valax.risk.scenarios.parametric_scenarios

::: valax.risk.scenarios.historical_scenarios

::: valax.risk.scenarios.stress_scenario

### Curve-shape stress presets

::: valax.risk.scenarios.steepener

::: valax.risk.scenarios.flattener

::: valax.risk.scenarios.butterfly

::: valax.risk.scenarios.curve_shape_bumps

## Keyed, per-curve factor layout

The replacement for the flat positional `[spots, vols, rates, dividends]`
layout: an ordered tuple of blocks, **one per named market object**, so
every curve on the graph owns its own coordinates in a scenario, in a
covariance matrix and in a ladder alike. Rationale and recipes:
[Scenarios & Shocks § Keyed generators](../risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout).

::: valax.risk.scenarios.FactorBlock

::: valax.risk.scenarios.KeyedFactorLayout

### The basis as a coordinate

`as_basis_layout` turns a projection curve's block into a **spread** over a
discount curve, which is what gives
`IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>`
([Risk Factors § 3.2](../risk-factors.md#32-interest-rates-multi-curve)) a variance, a
P&L rung and a VaR contribution of its own rather than only a scenario.

::: valax.risk.scenarios.as_basis_layout

::: valax.risk.scenarios.basis_factor_prefix

### Per-curve covariance assembly

::: valax.risk.scenarios.block_covariance

::: valax.risk.scenarios.historical_block_covariance

::: valax.risk.scenarios.psd_factor

### Keyed generators

::: valax.risk.scenarios.keyed_parametric_scenarios

::: valax.risk.scenarios.keyed_historical_scenarios

::: valax.risk.scenarios.keyed_stress_scenario

::: valax.risk.scenarios.keyed_curve_stress

::: valax.risk.scenarios.keyed_steepener

::: valax.risk.scenarios.keyed_flattener

::: valax.risk.scenarios.keyed_butterfly

::: valax.risk.scenarios.keyed_basis_shift

## Group repricing on a `MarketState`

Repricing, P&L and gradients for one **homogeneous group**, all routed
through [`price`](core.md#the-unified-signature) — no pricing-function adapter. The
group primitives return the three quantities that aggregate additively
across groups (scalar PV, gradient pytree, P&L vector) and deliberately
never a risk measure. Walkthrough and a runnable multi-curve example:
[VaR & P&L § Repricing through `price()`](../risk-var.md#repricing-through-price).

A shock the group's model *shadows* — a curve shock against a model
carrying its own scalar `rate`, for instance — cannot reach the price and
contributes exactly zero P&L. `group_reprice` and `group_scenario_pnl`
emit a `UserWarning` rather than return that zero silently; see
[the warning's rationale](../risk-var.md#repricing-through-price).

::: valax.risk.state_var.group_pv

::: valax.risk.state_var.group_prices

::: valax.risk.state_var.group_reprice

::: valax.risk.state_var.group_scenario_pnl

::: valax.risk.state_var.group_market_gradient

::: valax.risk.state_var.group_model_gradient

::: valax.risk.state_var.group_size

## Pricing function adapter (deprecated)

Adapts equity-style `(instrument, spot, vol, rate, dividend) -> price`
pricers to the market-data-aware signature
`(instrument, MarketData) -> price` used by the legacy risk engine.

!!! warning "Deprecated — removed with `MarketData`"
    Emits a `DeprecationWarning`. The wrapper collapses the discount
    curve to a single pillar's zero rate (`_extract_short_rate`), so a
    wrapped payoff carries no sensitivity to any other pillar. Use the
    group primitives above, which need no adapter and see the whole
    curve.

::: valax.risk.var.wrap_equity_pricing_fn

## Risk measures

`value_at_risk` / `expected_shortfall` take any P&L vector, including one
from `group_scenario_pnl`. The repricing entry points below are the
legacy `MarketData` / `MarketScenario` path, frozen for one deprecation
cycle.

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

This is the **legacy** flat-`MarketData` form, still to be retired; the
`MarketState` counterpart, with a per-curve covariance and an additive
Euler decomposition, is
[`keyed_parametric_var` / `keyed_component_var`](#per-curve-delta-normal-var).

::: valax.risk.var.parametric_var

## Spread risk: OAS & Z-spread

Option-adjusted spread, Z-spread, and their autodiff spread-Greeks, in
`valax/risk/oas.py`. The OAS is the constant continuously-compounded
parallel shift on the model's discount curve that reprices a bond to its
market price; for an option-free bond it collapses to the Z-spread. See
[Models & Theory §7.9](../theory/risk-measures.md#79-option-adjusted-spread-and-z-spread)
for the mathematics (including why, under Hull-White, the parallel-shift
and re-fit definitions of OAS coincide, and why a callable's effective
convexity is negative near the call boundary — recovered as an autodiff
Greek once the PDE's call projection is cell-averaged).

### Option-adjusted spread

`callable_bond_oas` root-finds the spread against the Hull-White PDE
pricer via `optimistix.Newton`; it is implicitly differentiable, so
`jax.grad` through it gives spread sensitivities. It dispatches on the
instrument type, so it also serves option-free bonds (returning the
Z-spread). `price_under_spread` is the underlying repricing map.

::: valax.risk.oas.callable_bond_oas

::: valax.risk.oas.price_under_spread

### Z-spread (analytic, option-free)

::: valax.risk.oas.bond_z_spread

### Model-based effective risk

Spread-Greeks of a bond under the short-rate model — the right tool for
instruments with embedded optionality, where the option must be repriced
at each spread. Derivatives are exact autodiff through the PDE exercise
projection.

::: valax.risk.oas.effective_duration

::: valax.risk.oas.effective_convexity

### Z-spread risk (analytic, option-free)

Spread duration, spread DV01 (cash P&L per basis point of Z-spread), and
spread convexity for option-free bonds, taken by autodiff of the analytic
curve pricer. For a single-curve bullet a Z-spread shift is the same curve
move as a parallel zero-rate shift, so these equal the bond's parallel-curve
(IR) risk; the *spread* framing is what separates credit/liquidity from
pure rate risk in a report.

::: valax.risk.oas.z_spread_duration

::: valax.risk.oas.z_spread_dv01

::: valax.risk.oas.z_spread_convexity

### Spread-convention conversions

Convert a spread quoted as a rate between continuous and periodic
compounding bases. Useful when reconciling with tools (e.g. QuantLib)
that quote OAS on a compounded basis — though note the exact
continuous-equivalent of a *compounded parallel curve shift* is
maturity-dependent, so on a non-flat curve these are a first-order
adjustment, not an exact OAS bridge.

::: valax.risk.oas.continuous_to_compounded_spread

::: valax.risk.oas.compounded_to_continuous_spread

## P&L attribution

Decompose scenario P&L into risk-factor contributions using a
second-order Taylor expansion with autodiff sensitivities.

Still on the flat `MarketData` / `MarketScenario` path; its port is
pending. For a per-curve attribution on a `MarketState` use
[`keyed_waterfall_pnl`](#per-block-waterfall).

::: valax.risk.var.pnl_attribution

## P&L vectors

The RTPL / HPL pair still consumes the flat `MarketData` view (a
`SensitivityLadder` and a `ScenarioSet`); re-targeting it onto the keyed
ladder is open work on the
[roadmap](../roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation).

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

The flat, positional ladder over a `MarketData` (legacy, kept for one
deprecation cycle and still the input of the RTPL vector). On a
`MarketState` use the [keyed ladders](#keyed-ladders-on-a-marketstate)
below.

::: valax.risk.ladders.SensitivityLadder

::: valax.risk.ladders.WaterfallPnL

::: valax.risk.ladders.compute_ladder

::: valax.risk.ladders.waterfall_pnl

::: valax.risk.ladders.waterfall_pnl_report

## Keyed ladders on a `MarketState`

The counterpart of the above for the container the unified pricer takes,
indexed by a
[`KeyedFactorLayout`](#keyed-per-curve-factor-layout) rather than
positionally — so a dual-curve book reports an OIS ladder and a projection
(or basis) ladder instead of one summed `delta_rate` vector.

Sensitivities are taken with respect to the **shock magnitudes** of a
keyed scenario rather than the market leaves, which removes the
DF → zero-rate chain rule entirely and gives one well-defined vega per
surface whatever its type. Rationale and recipes:
[Sensitivity Ladders § Keyed ladders](../risk-ladders.md#keyed-ladders-on-a-marketstate).

::: valax.risk.state_ladders.KeyedLadder

::: valax.risk.state_ladders.keyed_ladder

### Per-block waterfall

::: valax.risk.state_ladders.KeyedWaterfall

::: valax.risk.state_ladders.keyed_waterfall_pnl

::: valax.risk.state_ladders.keyed_waterfall_report

### Per-curve delta-normal VaR

`keyed_component_var` is the number the flat layout cannot produce: an
additive decomposition of the book's VaR in which a projection curve — or,
on an `as_basis_layout` layout, the `IR.BASIS...` factor — carries its own
contribution.

::: valax.risk.state_ladders.keyed_parametric_var

::: valax.risk.state_ladders.keyed_component_var

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
[Models & Theory §7.8](../theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations)
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
and `scenario` methods. `scenario` returns a legacy `MarketScenario` for
`apply_scenario` / `pnl_attribution`; on a `MarketState`, put
`jacobian @ scores` into a `KeyedScenario(curve_shocks={curve_id: ...})`
and apply it with [`shock`](#keyed-shocks-on-a-marketstate). See the
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
