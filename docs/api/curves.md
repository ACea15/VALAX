# Curves

Discount, inflation, and survival curve construction, interpolation,
and bootstrapping. All curve objects are `equinox.Module` pytrees —
fully differentiable, so `jax.grad` through any pricing function that
takes a curve gives per-pillar key-rate durations for free.

## Discount curve

Interpolation is log-linear on discount factors (equivalent to
piecewise-constant continuously-compounded forward rates) with flat
extrapolation beyond the pillar range. Callable directly:
`df = curve(date)` (scalar or vectorized).

::: valax.curves.DiscountCurve

### Zero and forward rates

::: valax.curves.forward_rate

::: valax.curves.zero_rate

## Inflation curve

Term structure of forward CPI levels. Interpolates in **log-CPI**
space for smooth implied forward inflation rates.

::: valax.curves.InflationCurve

::: valax.curves.forward_cpi

::: valax.curves.zc_inflation_rate

::: valax.curves.yoy_forward_rate

::: valax.curves.from_zc_rates

## Survival (credit) curve

Term structure of survival probabilities for a single credit entity.
Mirrors `DiscountCurve` in design: log-linear interpolation between
pillars — equivalent to **piecewise-constant hazard rate** — which is
the standard market convention. `jax.grad` gives per-pillar
credit-delta sensitivities (the credit analogue of KRD / DV01).

::: valax.curves.SurvivalCurve

::: valax.curves.survival_probability

::: valax.curves.hazard_rate

::: valax.curves.piecewise_hazards

### Survival constructors

`from_hazard_rates` treats `hazards[i]` as the constant hazard on
\((\text{pillar}_{i-1}, \text{pillar}_i]\). `from_cds_spreads` uses
the credit triangle \(h \approx s / (1 - R)\) to bootstrap survival
from a flat-spread CDS curve.

::: valax.curves.from_hazard_rates

::: valax.curves.from_cds_spreads

See [Risk Factors](../risk-factors.md) — Credit section — for the role
of the survival curve in the wider risk-factor catalogue, and
[`valax.risk.shocks`](risk.md#credit-shocks) for the credit-side bump
primitives.

## Bootstrap instruments

Bootstrap instruments are the *quotes* used to calibrate curves. Each
one implements the [`BootstrapInstrument`](#the-bootstrapinstrument-protocol)
protocol — it carries a static `curves_touched` tuple of curve
identifiers and exposes a `residual(graph, fixings, ref_date)` method
that returns zero when the curve graph correctly reprices the quote.

These are calibration *inputs*, distinct from the tradeable
instruments in [`valax.instruments`](instruments.md) (which carry
notional and direction). They live in `valax.curves` so the
dependency graph stays clean.

For the no-arbitrage motivation behind each instrument's residual, see
[theory §3.7](../theory.md#37-no-arbitrage-relations-across-curves)
and [§3.8](../theory.md#38-joint-multi-curve-calibration). For the
production roadmap that uses these instruments in a joint solver, see
[production.md §11](../architecture/production.md#11-multi-curve-framework).

### The `BootstrapInstrument` protocol

::: valax.curves.bootstrap_proto.BootstrapInstrument

### Curve graph

`CurveGraph` is a flat, identifier-keyed registry of discount curves.
Lookup via `graph[curve_id]`, membership via `curve_id in graph`.
Single-curve callers use the sentinel id `"_default_"`; multi-curve
callers use semantic identifiers like `"USD.SOFR.OIS"`,
`"EUR.EURIBOR.6M"`.

::: valax.curves.CurveGraph

::: valax.curves.CurveSpec

### Joint multi-curve solver

Joint multi-curve Newton solver — the central deliverable of
MC-Curves-2. Concatenates the log-DFs of every curve in the graph into
one state vector, zeros every instrument's residual simultaneously via
`optimistix.Newton`, and supports `jax.grad` through the calibrated
graph via `optimistix.ImplicitAdjoint`.

::: valax.curves.bootstrap_curve_graph

::: valax.curves.CurveBuildDiagnostics

`quote_jacobian` is a convenience wrapper around `jax.jacrev` over
`bootstrap_curve_graph`. Rows are laid out per curve in the order of
`curve_specs`; within each curve, one row per pillar. Columns follow
the order of `instruments`. Uses implicit-adjoint so the cost is one
linear solve per column regardless of Newton iteration count.

::: valax.curves.quote_jacobian

### Fixings

Realised fixings for partially-seasoned floating legs, keyed by index
identifier (e.g. `USD.SOFR`, `USD.SOFR.3M`, `EUR.EURIBOR.6M`) and
fixing date. `jax.grad` can flow through a lookup, and `eqx.tree_at`
can replace a single series without rebuilding the registry. See
theory §3.9 for the no-arbitrage motivation.

::: valax.curves.FixingSeries

::: valax.curves.FixingHistory

::: valax.curves.empty_fixing_history

### Convexity-adjustment plug-ins

Used by `MoneyMarketFuture`. Two variants ship with MC-Curves-1;
`hull_white_convexity_adj(model)` is reserved for a follow-up PR gated
on short-rate-model integration with the curve build (see
[theory §3.9](../theory.md#39-futures-convexity-adjustment-and-fixings)).

::: valax.curves.no_convexity_adj

::: valax.curves.constant_convexity_adj

### Single-curve quote types

Each type's `curves_touched` is a 1-tuple defaulting to
`("_default_",)`.

::: valax.curves.DepositRate

::: valax.curves.FRA

::: valax.curves.SwapRate

::: valax.curves.OISSwapRate

::: valax.curves.MoneyMarketFuture

::: valax.curves.TurnInstrument

### Multi-curve same-currency quote types

`IborSwapRate` is dual-curve — `curves_touched` convention
`(discount_curve_id, forward_curve_id)`. Past coupons
(`fixing_date <= ref_date` and recorded in `FixingHistory[index_id]`)
use the realised rate instead of the projection.

::: valax.curves.IborSwapRate

`TenorBasisSwap` is three-curve — `curves_touched` convention
`(discount_id, leg_a_curve_id, leg_b_curve_id)`. The spread is added
per period to the leg indicated by `spread_on_leg`. Both legs use
forward projection from their respective forward curves and
discounting from the shared OIS curve.

::: valax.curves.TenorBasisSwap

### Cross-currency quote types

`FXForward` uses `curves_touched` convention
`(domestic_curve_id, foreign_curve_id)`; `fx_spot` and
`quoted_forward` are in *domestic per foreign* units. Residual is the
covered-interest-parity relation
\(F^\text{quote} - S(0) \cdot DF_\text{foreign}(T) / DF_\text{domestic}(T)\).

::: valax.curves.FXForward

`FXSwap` reduces to `FXForward`'s residual when
`near_date == ref_date`.

::: valax.curves.FXSwap

`CrossCurrencyBasisSwap` (CCBS) uses `curves_touched` convention
`(dom_discount_id, dom_forward_id, for_discount_id, for_forward_id)`.

- **`variant = "constant_notional"`**: residual is
  \([\text{dom\_pv} + (1 - DF_\text{dom}(T_n))] - [\text{for\_pv} + (1 - DF_\text{for}(T_n))]\);
  `fx_spot` cancels in the normalised residual.
- **`variant = "mtm"`** (mark-to-market — notional resets at each
  fixing): the foreign discount curve does not enter the residual
  (Henrard 2014 §10.5).

::: valax.curves.CrossCurrencyBasisSwap

## Multi-curve set

Legacy single-currency `MultiCurveSet` (OIS discount + tenor-keyed
forwards) plus its bootstrap. Superseded by the joint
[`bootstrap_curve_graph`](#joint-multi-curve-solver) for new callers;
still supported for existing code paths.

::: valax.curves.MultiCurveSet

::: valax.curves.bootstrap_multi_curve

## Single-curve bootstrap (legacy)

Sequential and simultaneous bootstrap for the single-curve
`("_default_",)` graph. New callers should prefer
[`bootstrap_curve_graph`](#joint-multi-curve-solver).

::: valax.curves.bootstrap_sequential

::: valax.curves.bootstrap_simultaneous
