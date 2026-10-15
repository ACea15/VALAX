# Calibration, Pricing & Risk Architecture

How VALAX connects market quotes to prices and back to risk, and how the
framework grows: new instruments, models, numerical methods, calibration levels
and risk factors each plug in through one small contract. The mathematics is on
[Calibration Chains](../theory/calibration-chains.md); a runnable walk-through
is [From Quotes to Risk](../guide/calibration-to-risk.md).

## 1. Three maps

Every risk number in VALAX is built from three maps:

```text
 quotes q, fixed inputs p
        │  calibrate_market(chain, q, p)          ── shared, once per snapshot
        ▼
 MarketState  ──────────────┐
        │  calibrate_model(node, q, market, p)    ── per basket / underlying
        ▼                   │
 model ─────────────────────┤
                            ▼
              price(instrument, model, market)     ── per trade
                            │
                            ▼  Σ quantity × price   ── per book (linear)
                         book value
```

1. **Calibration** — quotes and fixed inputs → calibrated objects, through a
   chain of levels ``R(θ; q, upstream, p) = 0``.
2. **Pricing** — ``price(instrument, model, market)``; reverse-mode AD gives the
   market and model cotangents.
3. **Aggregation** — linear, so it commutes with differentiation: a book's
   cotangent is the sum of its trades'.

The consequences are the design:

* **Separation.** Instruments, models and methods never see calibration;
  calibration never sees the trades being priced. Each axis grows on its own.
* **Risk in quote space is one sweep.** The chain adjoint is a linear map from
  the market (and model) cotangent to ``(q̄, p̄)``; it does not depend on the
  instrument, so a book is swept once with its summed cotangent.
* **Degenerate markets are data, not code paths.** Constant-parameter
  Black-Scholes is a chain with one explicit level; single-curve is a chain with
  one curve block; multi-curve is the same node with more specs.

## 2. Contracts

| Concern | Type / function | Module |
| --- | --- | --- |
| Quotes ``q`` and fixed inputs ``p`` | `QuoteSet` (keyed, differentiable, vmappable) | `valax.market.quotes` |
| Calibrated market | `MarketState` (`flat`, `single_curve`, keyed multi-curve) | `valax.core.market_state` |
| One level | `CalibrationLevel`, `LevelInputs(quotes, upstream, fixed, data)` (`data`: trade data, not a risk input) | `valax.calibration.level` |
| Solver choice and tolerances | `SolverSettings`, `run_solver`; defaults `EXACT_SETTINGS`, `FIT_SETTINGS`, `BASKET_SETTINGS`, `CHEYETTE_SETTINGS` | `valax.calibration.settings`, `.basket`, `.cheyette` |
| Result of a calibration | `CalibrationResult` (value, theta, inputs, `residual_fn`, `assemble_fn`, traceable diagnostics, `labels`, `problem`, `provenance`, `children`; `check()`, `failed_labels()`, `condition_number()`, `converged`) | `valax.calibration.result` |
| Level builders | `solve_level` (one solve), `solve_batched_level` (one solve per node, `vmap`-ed), `explicit_level` (explicit maps, fixed points); `stack_slices`, `period_label(s)` | `valax.calibration.solve` |
| Model-to-basket fit | `basket_fit_level` (behind every short-rate calibrator) | `valax.calibration.basket` |
| Engine provenance | `CalibrationProvenance`, `check_engine` | `valax.calibration.result` |
| Calibration step | `CalibrationNode` (market scope: `install_fn`; model scope: none) | `valax.calibration.chain` |
| Market calibration | `CalibrationChain`, `calibrate_market` → `CalibratedMarket` | `valax.calibration.chain` |
| Model calibration | `calibrate_model(node, q, market, p)` | `valax.calibration.chain` |
| Node factories | `market_inputs_node`, `curve_graph_node`, `sabr_smile_node`, `swaption_cube_node`, `caplet_strip_node`, `optionlet_sabr_node`, `price_fit_node`, `swaption_basket_node` (+ `coterminal_swaptions`) | `valax.calibration.nodes` |
| Curve binding | `curve_requirements`, `bind_model`, `bind_curves`; `basis_overlay` | `valax.core.binding`, `valax.pricing.multicurve` |
| Pricing | `price(instrument, model, market, method=, config=)` | `valax.core.pricing` |
| Quote risk (adjoint) | `chain_adjoint`, `quote_risk_adjoint` | `valax.greeks.quote_risk` |
| Quote risk (autodiff) | `quote_risk_autodiff`, `reconcile_quote_risk` | `valax.greeks.quote_risk` |

## 3. Decisions

**D1 — Two differentiation paths, both first-class.** The explicit adjoint
sweep (`chain_adjoint`) reuses each level's residual at the solution, reports
per-level stationarity, and supports book-level aggregation. The autodiff path
(`quote_risk_autodiff`) differentiates through the solves with the implicit
adjoint `optimistix` attaches to each. They share no differentiation code, fail
in different ways (an ill-conditioned level vs a non-traceable calibrator), and
agree to solver tolerance — `reconcile_quote_risk` reports both and the
mismatch, never silently preferring one. Neither path survives a non-converged
calibration, which is what the convergence flag is for.

**D2 — The model is a separate input from the market.** The market is shared
by every trade; a model is calibrated on a basket chosen by the trade (rates
exotics) or by the underlying (equity books). ``price`` takes them as separate
arguments, `eqx.filter_grad` over ``(model, market)`` returns both cotangents,
and `chain_adjoint` folds model levels in through `model_levels`. Market-scope
and model-scope calibrations are the *same* `CalibrationNode` contract; only
the destination of the value differs, so one model class (e.g. Heston) can
calibrate a market surface or price an exotic.

**D3 — One solver factory, all-static settings, traceable convergence.**
Every calibration routes through `run_solver` with a `SolverSettings`
(algorithm, `rtol`, `atol`, `max_steps`, `throw`). Convergence is reported as a
JAX boolean on `CalibrationResult.diagnostics.converged`, so it survives
`jit` and `vmap`; `result.check()` / `calibrate_market(..., strict=True)` raise
eagerly. Exact levels default to Newton at ``1e-10``; least-squares levels to
Levenberg-Marquardt at ``1e-8`` (price-based fits have a noise floor a tighter
tolerance cannot get under). The short-rate basket fits default to
line-searched BFGS (`BASKET_SETTINGS`, ``1e-12`` for the closed-form
Hull-White / G2++ engines; `CHEYETTE_SETTINGS`, ``1e-10`` on the PDE): the
Jamshidian and Brigo-Mercurio swaption formulas nest a root-find whose own
implicit derivative is non-finite far from the solution, where an undamped LM
step can land, and BFGS's line search keeps the iterates where the pricer is
defined. No calibrator selects a solver by string or builds one itself, and
none discards `optimistix`'s failure flag.

**D4 — The level is emitted by the calibrator.** A level's residual is the
function the solver drove to zero, packaged with its solution by
`solve_level` / `solve_batched_level`; nothing rebuilds a residual for the
adjoint. This holds for the standalone calibrators too (`calibrate_sabr`,
`calibrate_hull_white`, `strip_caplet_vols`, `bootstrap_curve_graph`, …):
each is a thin wrapper over a level builder and returns the same
`CalibrationResult` a chain node does — `bootstrap_curve_graph` *is* the
root level of `curve_graph_node`, `calibrate_swaption_cube_from_curves`
shares `swaption_cube_level` with `swaption_cube_node`.

**D5 — No `custom_vjp`.** Calibrations are differentiated by the implicit
function theorem (explicitly, or through `optimistix`), pricers by plain AD.
A custom rule would be reverse-only and could hide non-convergence.

**D6 — Validation happens when a node is built, not when it runs.** Node
factories check shapes, squareness and key counts eagerly; the calibration
itself is traceable, so a whole chain can be `jit`-ed, differentiated and
`vmap`-ed over quote scenarios.

**D7 — Loops of independent fits are one batched, labelled level.** A smile
per expiry, SABR per cube cell, SVI per slice and a strip per strike column
are solved by `solve_batched_level`: one independent solve per node inside a
single `vmap` (one trace, one compile whatever the node count), with
per-node diagnostics aligned to `labels`. A failure is therefore *named*
(`"5Yx10Y"`, `"2Y"`, `"cap 3Y"`, `"K=3.00% cap 3Y"`) by `failed_labels()` and in
the `check()` error, instead of collapsing to one flag for the grid. For the
adjoint the nodes are still one level: its residual stacks the per-node
residuals (first-order conditions for fits), so `dR/dθ` is block-diagonal and
`condition_number()` is taken over all blocks. Per-node inputs are a function
of the level inputs, so forwards read from upstream curves stay
differentiable. A sequentially bootstrapped node (the cap strip, one bucket
after another, each with its own flag) supplies its own `node_solve_fn` but is
still packaged as one lower-triangular root level (`caplet_strip_level`).
Ragged grids are padded by `stack_slices` with zero weight, so padding
contributes and receives nothing.

**D8 — Composite calibrations are one result with children.** A calibration
built from sub-calibrations returns the outer level with the inner results
in `children`: `calibrate_slv` returns the leverage level with the Heston fit
as `children[0]`; `strip_optionlet_surface` returns the SABR level with the
strip as its child. `converged` and `check()` cover every child, so a caller
checks one object, and each child keeps its own level for the adjoint.

**D9 — A Monte Carlo fixed point is an explicit level differentiated by
plain AD, never by the IFT.** The SLV leverage `L_{k+1} = G(L_k)` is computed
by `n_iterations` unrolled passes, each a self-consistent Guyon–Henry-Labordère
sweep of the particle swarm with fresh randomness. There is no residual the
solver drives to zero — the
increment never falls below the Monte Carlo noise floor — so there is no
`R(θ) = 0` at which the implicit function theorem could be applied, and
pretending otherwise would differentiate a fiction. `calibrate_slv_leverage`
therefore returns an explicit level (`theta=None`, `problem="fixed_point"`):
its value is a differentiable map of the inputs (Heston model, target surface,
spot) and its risk is reverse-mode AD through the passes. Convergence is still
reported, by `fixed_point_diagnostics`: the residual is the last increment
`L_K − L_{K−1}`, `converged` is `max|ΔL| ≤ settings.atol` with `atol`
(`SLV_SETTINGS`, `5e-2`) set deliberately above the noise floor
(≈ `1/√n_paths` relative), `n_iterations=1` is always reported not converged
(it measures only the distance from the `L ≡ 1` warm start, hence the default
of 2), and the share of grid cells clipped to `[L_min, L_max]` is reported in
`diagnostics.extras["clip_fraction"]` without deciding convergence. Recipe:
[SLV § fixed point](../guide/slv.md#how-the-fixed-point-is-reported); maths:
[Calibration Chains § I.8](../theory/calibration-chains.md#i8-fixed-point-levels).

**D10 — Engine provenance is recorded on the result and enforced only where
it carries a modelling bias.** Model parameters belong to the model when the
calibration engine is exact (Jamshidian) or approximate only by a
discretisation that converges away; when the engine has a *modelling* bias
(frozen-`y` Cheyette replaces the stochastic accumulator by its deterministic
limit) the parameters are valid on that engine only. Each result records
`CalibrationProvenance(method, engine_kind, config)` with `engine_kind` ∈
{`exact`, `discretised`, `modelling_bias`} — as result metadata, not a model
field, so models fitted on different engines have one type, stack under
`vmap`, and the engine is not a grouping axis of the book. `check_engine(result,
method, config)` refuses a mismatched engine only for `modelling_bias`
(`cheyette_provenance("frozen", cfg)`); for `exact` (`CLOSED_FORM_PROVENANCE`,
Hull-White / G2++ closed forms) and `discretised` (PDE / MC baskets,
`cheyette_provenance("exact", cfg)`) it is audit metadata. Calibrators record
their engine directly; nodes that price through `price()` (`price_fit_node`,
`swaption_basket_node`) derive it with `engine_provenance(model, method,
config)` — a `singledispatch` on the model type (default: `"analytic"` →
exact, anything else → discretised; Cheyette's `"pde"` → modelling bias),
with `method="auto"` resolved per instrument first. `price()` does not consult
provenance: the check is an eager, explicit opt-in before pricing.

**D11 — Curves are bound to models by the market, through one declared
requirement.** Exact-fit short-rate models (Hull-White, G2++, Cheyette 1F/2F)
carry `initial_curve=None` and a static `curve_id`; their calibrators return
curve-free models. `curve_requirements(model, instrument) -> {field: curve_id}`
(a `singledispatch`) declares which fields are curves and which market curve
fills each — `model.curve_id` or else the instrument's discount curve — and
`bind_model` inside `price()` fills them from the `MarketState`. `price()`
**raises** if a declared field arrives pre-bound: the market is the single
source of truth for curve risk, and an embedded curve would both shadow the
market and leak curve sensitivity into the model gradient. Kernels raise if
unbound (`require_initial_curve`); `bind_curves(model, curve_or_market)` binds
explicitly for kernel use. Whether a model may price a projection curve distinct
from its discount curve is the `basis_overlay(model)` dispatch (Hull-White /
G2++ yes, Cheyette no), so a trade that resolves a projection curve its model
cannot accept raises instead of being priced single-curve. The requirements
protocol is the extension point for multi-curve and stochastic-basis models
(one initial curve per modelled index); a keyed set of forward curves in
`ResolvedMarket` is still open (roadmap §5.6).

**D12 — A per-trade model is a model-scope level read off the calibrated
market.** `swaption_basket_node` fits a model to one trade's basket (e.g.
`coterminal_swaptions(bermudan)`): targets are the basket swaptions priced off
the calibrated cube and curves, model prices go through `price()` (Jamshidian
for Hull-White), and the level's only inputs are the market and its fixed
inputs. So `chain_adjoint` with `model_levels` pushes the trade's `λ_m` into
the cube level and the curve level and on to the quotes. Zero-valued **target
shifts** `s_k` (one fixed input per basket instrument) make
`p̄_k = ∂V/∂T_k` — the vega profile across the basket — fall out of the same
sweep, checkable by bumping the shift. Baskets may be traced, so same-shaped
per-trade baskets calibrate in one `eqx.filter_vmap`; the curve-free models
(D11) are what makes the result stackable.

**D13 — Time-dependent parameters live on the model; their breakpoints are
data.** A short-rate volatility is always an array of buckets
(`Float[Array, " n_buckets"]`; a scalar is promoted to one bucket), with
keyword-only `volatility_times` holding the `n_buckets - 1` interior
breakpoints. The breakpoints are a **traced leaf, not a static field**:
per-trade models whose buckets sit on different exercise dates have the same
shapes, so they batch under `eqx.filter_vmap` and stack like any other model,
and only the bucket *count* is static. Every kernel has a trace-time
`is_constant` branch that keeps the constant-volatility formula and its
evaluation order verbatim, so a one-bucket model is bit-identical to the
pre-σ(t) code (checked against a golden file). Reordering the arithmetic is
not harmless: once it changed a reverse-mode gradient by one ulp. A model
whose two factor volatilities must share buckets is parametrised as a first
volatility and a **ratio** (G2++ `volatility_x` + `volatility_ratio`,
Cheyette 2F `volatility_1` + `volatility_ratio`; the second volatility is a
read-only property), so a bootstrap fits one bucketed field with the ratio
held as a reported fixed input. `volatility_bucket_fields(model)` (a
`singledispatch`) names the bucketed fields, and is how a node validates which
field may be bootstrapped. The bucket convention and maths are in
[Time-Dependent Volatility](../theory/time-dependent-volatility.md).

**D14 — Triangular levels are solved bucket by bucket.** A co-terminal
bootstrap (`swaption_basket_node(mode="coterminal_bootstrap")`) is square and
lower-triangular: bucket *k*'s equation reads buckets 1..*k*. It is built by
`triangular_root_level`, which solves one scalar Newton root per bucket with
the earlier buckets passed in the solver's `args` (so autodiff differentiates
through every stage), and reports the full triangular residual (one
triangular solve for the IFT adjoint) and **per-bucket** diagnostics. A
failed bucket installs NaN and is named as the root cause; every bucket after
it is solved against finite stand-ins and reported failed, and every bucket
before it keeps its exact root. Each bucket's reported residual is its own,
evaluated with the later buckets at finite stand-ins, because a later
bucket's NaN enters an earlier equation with weight exactly zero and
`0 · NaN` is NaN. A **joint** solver (one Newton solve of the whole system,
Jacobian tagged lower triangular so `lineax` substitutes) was built and
measured against it, and removed: every step builds the full Jacobian
(Hull-White, *n* = 4 / 6 / 10 co-terminals: 120 / 252 / 770 pricings vs
40 / 58 / 108; Cheyette PDE, *n* = 3 / 5: 60 / 150 PDE solves vs 28 / 48), it
compiled slower and never ran faster, and on an infeasible bucket it marked
every bucket failed under the wrong root cause (before the nested root-finds
were guarded, it raised at runtime instead). Roots agreed to ≤ 1.4e-13. The
evidence is in `scripts/bench_coterminal_bootstrap.py`.

## 4. Extending the framework

| To add… | You provide | It then works with |
| --- | --- | --- |
| **Instrument** | An `eqx.Module` (data only, static selectors such as `discount_id`) and `@register_pricer` adapters | `price`, both risk paths, `price_fit_node` baskets |
| **Model** | An `eqx.Module` with no embedded market objects (scalar fields left `None` are bound from the market; curve fields declared by registering `curve_requirements` and left `None`) plus pricers; a `basis_overlay` registration if it prices a distinct projection curve by the deterministic overlay | `price`, `price_fit_node` / `swaption_basket_node` calibration at either scope, stacking under `vmap` |
| **Numerical method** | A config class and pricers registered under the method name; a `CalibrationProvenance` for fits on it (`engine_kind="modelling_bias"` if its parameters are engine-bound) | `price(..., method=...)`, calibration through `price_fit_node(method=...)`, `check_engine` |
| **Calibration level** | A node factory: residual ``(θ, inputs) → vector``, `assemble` ``(θ, inputs) → object``, `install` ``(market, object) → market`` (or none), routed through `solve_level` (one solve), `solve_batched_level` (one solve per labelled node) or `explicit_level` (explicit map or fixed point), with a `SolverSettings` default | `calibrate_market`, both risk paths, diagnostics, named failures, conditioning |
| **Standalone calibrator** | A thin wrapper building the same level as its node (e.g. `calibrate_sabr` → `solve_level`, `calibrate_hull_white` → `basket_fit_level`) and returning its `CalibrationResult` | `.check()`, `.converged` under `jit` / `vmap`, `level_adjoint` |
| **Risk factor** | A quote key and the node that consumes it (for a new market slot: an installer and a `MarketState` field) | quote risk keyed by that identifier |

A new level needs **no adjoint code**: `solve_level` packages the residual,
`chain_adjoint` pulls cotangents back through `assemble`, and the
`LevelInputs` split delivers ``q̄``, the upstream cotangent and ``p̄`` from one
VJP. The tests for a new level are the same three every level has: the two
paths reconcile, both match full-recalibration finite differences, and a
deliberately unconvergeable fit is reported (see
`tests/test_greeks/test_quote_risk.py` and
`tests/test_calibration/test_settings.py`).

## 5. Scaling

* **Many trades, one market.** Sum the trades' market cotangents and sweep the
  chain once (`chain_adjoint`); per-trade pricing batches with
  `eqx.filter_vmap` over homogeneous groups (`valax.risk.state_var`).
* **Many scenarios.** `QuoteSet` values are leaves, so a stacked quote set is a
  scenario batch: `jax.vmap` over `calibrate_market` recalibrates every
  scenario in one compiled program.
* **Many baskets.** Same-shaped model calibrations (one per trade or per
  underlying) batch with `eqx.filter_vmap(calibrate_model)`; a failed member
  flags itself through its `converged` entry.
* **Heterogeneous books** group by structure and array shape, with manual
  schedule padding where it pays; see
  [Heterogeneous Portfolios](../scaling/heterogeneous.md).
* **Devices.** Multi-device sharding is out of scope for now; see
  [Parallelism & Devices](../scaling/parallelism.md).

## 6. Open work

Tracked on the [roadmap](../roadmap.md#unified-calibration-pricing-and-risk):
richer synthetic quote noise; constrained (KKT) SVI and FX smile levels;
credit and FX quote levels; reaching every pricer through `price` (the
remaining rates pricers including the
Hull-White tree, caps / swaptions off the cube surfaces, default configs); a
keyed set of forward curves in `ResolvedMarket` for multi-index instruments
(D11); derivative convergence of the numerical engines; the position /
portfolio layer; a global piecewise σ(t) calibration on a user grid for
portfolio / XVA simulation (D13). Hull-White, G2++ (multi-curve, 2-D PDE
basket) and Cheyette 1F/2F (single-curve, frozen-y PDE basket) are all
validated per trade through the same `swaption_basket_node`, both as an exact
co-terminal bootstrap (D14) and as a least-squares fit of constant
parameters, where G2++ needs a basket beyond the co-terminals to identify both
factors (see [examples](../examples.md#per-trade-g2-and-cheyette)).
