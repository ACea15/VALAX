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
2. **Pricing** — ``price(instrument, model, market)``; AAD (reverse-mode autodiff) gives the
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
function theorem (explicitly, or through `optimistix`), pricers by plain AAD.
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
plain AAD, never by the IFT.** The SLV leverage `L_{k+1} = G(L_k)` is computed
by `n_iterations` unrolled passes, each a self-consistent Guyon–Henry-Labordère
sweep of the particle swarm with fresh randomness. There is no residual the
solver drives to zero — the
increment never falls below the Monte Carlo noise floor — so there is no
`R(θ) = 0` at which the implicit function theorem could be applied, and
pretending otherwise would differentiate a fiction. `calibrate_slv_leverage`
therefore returns an explicit level (`theta=None`, `problem="fixed_point"`):
its value is a differentiable map of the inputs (Heston model, target surface,
spot) and its risk is AAD through the passes. Convergence is still
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
(one initial curve per modelled index — a mapping-valued requirement binds
a keyed dict of curves into one field, D16); the trade side, a keyed set of
curves in `ResolvedMarket`, is D16.

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

**D15 — `price()` is the one entry point, and every pair carries an explicit
policy.** `method="auto"` used to fall back to a global order (analytic →
lattice → pde → mc) for 56 of the 66 registered pairs, so a new pricer could
silently change `auto` for a pair it touched; and `config=None` crashed the
PDE and MC routes. Now every registered `(instrument, model)` pair has an
explicit auto method **and** a default config for each of its numerical
methods, both in one table, `valax.core.policy` (`AUTO_METHODS`, the
`*_DEFAULTS` maps, `set_default_config`). The fallback is gone: a pair without
a policy (only possible for a third-party registration) makes `auto` raise,
naming the pair and its methods. The frozen map reproduces, pair for pair,
what the fallback chose, so no `auto` price moved (pinned against the Phase 4
golden file). Config resolution is **explicit `config=` > the pair's default
for that method > error**, exposed as `effective_config(instrument, model,
method, config)`; `PricingResult.config` records the config actually used
(`None` only for configless analytic engines), and calibration provenance
(`engine_provenance`, `swaption_basket_node`) records it too, so a fit on the
default grid is bound to that grid. Defaults: `PDEConfig()` / `PDEConfig2D()`,
the Cheyette calibrator's own frozen-y grids (so a fit and a price with
`config=None` share one grid), `MCConfig(100_000, 100)` everywhere, and
`TreeConfig()` for the Hull-White tree. The short-rate MC recipes now honour
`MCConfig.n_steps` (they used a recipe kwarg, 100, whatever the config said).
Two families of routes were added: `ImpliedVolModel(surface_id)` — a
parameter-free "model" pricing caplets, caps and swaptions at the implied vol
read off the named surface (cube, optionlet surface, or a `ConstantVol`
quoted normal or lognormal), which `swaption_basket_node` now uses for its
targets, bit-identical to the direct call — and the Hull-White tree under
`method="lattice"` for callable / puttable bonds and European / Bermudan
swaptions (auto unchanged). The tree routes run on the general tree with a
half-width sized from the config alone, `J = ⌈2·n_steps⌉`, so they work on a
traced model under `jit` / `grad`: a step moves at most one level, but where
σ(t) drops the finer levels need more nodes for the same rates (measured
width / `n_steps`: up to 1.17 for a 0.012 → 0.009 drop, 1.25–1.30 for ×1.5 /
×2 zig-zags and humps, 2.00 for a ×3 drop); a narrower tree is flagged
invalid and prices NaN. The coverage matrix's
[`price()` routes column](../applications/instrument-pricing.md#coverage-matrix)
is checked against the registry and the policy in both directions. Still not
routed after D15: the LMM recipes (routed with D18) and CMS /
range-accrual products on `ImpliedVolModel`; the multi-index trades followed
with D16.

**D16 — A trade resolves every market entry it reads, keyed by id, from its
static selectors.** `ResolvedMarket` used to hold one discount and one
projection curve, and guessed the projection curve as `curves_touched[1]` —
so a tenor basis swap as legs `(OIS, 3M, 6M)` silently lost its 6M curve. Now
`resolve` binds **keyed views** of exactly the entries the trade names:
`curves` (a `CurveGraph` sub-graph, `curve_ids_of`), `spots`
(`underlying_ids_of`, in the instrument's order), `fx_spots` (`fx_ids_of`) and
`inflation_curves` (`inflation_ids_of`). The discount / projection pair stays
as **derived views** (`discount_curve` / `forward_curve` are properties over
`curves`, the same objects as before, so every single-index price is
bit-identical); the projection curve is read only from an explicit
`forward_id` or a two-entry `curves_touched`, never guessed for three or more
curves. The views hold only what the trade names — a pricer reading an
undeclared entry fails with a `KeyError`, so a trade's declared dependencies
cannot drift from what it reads. **Which entries is decided by static
fields only** (`discount_id`, `forward_id`, each leg's ids,
`foreign_discount_id`, `inflation_id`, `underlying_ids`): `resolve` runs at
trace time, the values stay traced leaves of the market, and every route
works under `jit` / `grad` with the trade as a traced argument — a 3M or a
6M trade *names* its projection curve rather than having it inferred from
traced dates, one compiled program per id combination, `vmap` over trades
sharing their ids. Fixings stay keyed by index id and never select a curve
(`indices_touched` names fixing histories only; the market holds no index →
curve map). MC / PDE recipes receive the keyed views only by naming
`curves` / `spots` in their signature (as for the curve pair, never through
`**kwargs`), and `price()` refuses to run a trade touching curves outside its
primary pair on a recipe that cannot take them. `curve_requirements` values
may be a mapping `{key: curve_id}`, binding a **dict of curves** into one
model field — the slot for multi-index / stochastic-basis models. With this,
`MultiLegSwap` (tenor basis, amortising, seasoned), `CrossCurrencySwap`
(domestic + `foreign_discount_id`, FX spot keyed by the pair), the inflation
swaps and cap / floor (`inflation_id`, `discount_id`) and the multi-asset
options (`underlying_ids`) price through `price()`.

**D17 — Every route honours one onboarding contract, enforced by one test.**
"Priceable through `price()`" is a checkable property, not a convention. Each
registered (instrument, model) pair has a sample in
`tests/test_core/route_samples.py`, built on one shared market that also
holds unrelated decoy entries, and `tests/test_core/test_route_contract.py`
checks every registered (pair, method): it prices finitely, with its default
config and with the trade as a traced `eqx.filter_jit` argument; it batches
under `eqx.filter_vmap`; and its market gradient is finite, non-zero on the
entries the sample names and **exactly zero outside**
`declared_entries(instrument, model, market)` — the instrument's selectors
plus the model's `curve_requirements` and surface reference. A route whose
pair has no sample fails the completeness test, so a new route cannot land
outside the contract; deviations would be allow-listed with a reason, and the
list is empty. Risk is reported over the same declared set:
`market_risk(instrument, model, market)` returns, from one backward pass, the
pillar deltas of every declared curve and the gradients of every declared
spot, dividend, FX spot, inflation curve and surface, keyed by id. Pricing and
risk of a new instrument therefore need no new risk code — the checklist is
[tutorial § 10](../guide/tutorial-unified-pricing.md#onboarding-checklist-a-new-instrument-or-route-in-the-library).
Writing the samples found one real bug (`MarketState.dividend` raised for a
non-paying asset once any other asset paid one); `FXForward` and
`FXVanillaOption` were the first trades onboarded through the checklist.

**D18 — The LMM is curve-free, and its routes map trades onto its grid from
the dates.** `LMMModel` used to embed its curve: `build_lmm_model` froze the
initial forwards, `P(0, T_0)` and the factor loadings into model leaves, so
curve risk was a model gradient and `price()` had no curve to bind. Now the
model holds the tenor grid, the volatility and the correlation, with
`initial_curve=None` and a static `curve_id` as the short-rate models do;
`curve_requirements` binds the curve (the pinned id, else the trade's
discount curve), every curve-derived quantity is a property of the bound
model, and `price()` refuses a pre-bound one. `build_lmm_model(curve, …)`
returns the bound model for kernel-level calls, bit-identical to the
embedded one. The four recipes no longer take caller-supplied indices: the
trade's periods are looked up on `tenor_dates` (`lmm_period_indices`, a
traced `searchsorted`, so the trade may be a traced `jit` argument), dates
off the grid raise when concrete and give a NaN price when traced, the
trade's day count must be the curve's (one accrual per period serves the
forward and the payoff), and the Euler grid spreads `MCConfig.n_steps` over
the `N + 1` grid segments. The **grid stays a model choice**: deriving it
from each trade would make the volatility matrix's shape a function of the
trade and break stacking models across trades. Routing them found a kernel
bug — the European swaption read each forward at its own later fixing
instead of at expiry (+20% on a 2Y×4Y ATM payer, hidden behind an `xfail`);
the payoff now uses the forward curve at expiry and matches Black at
Rebonato's swap vol. With D18 every registered route prices through
`price()` and passes the D17 contract.

## 4. Extending the framework

| To add… | You provide | It then works with |
| --- | --- | --- |
| **Instrument** | An `eqx.Module` (data only, static selectors such as `discount_id`), `@register_pricer` adapters reading only `resolved`, a policy entry and a route sample that passes the route contract (D17; checklist in [tutorial § 10](../guide/tutorial-unified-pricing.md#onboarding-checklist-a-new-instrument-or-route-in-the-library)) | `price`, `market_risk`, both risk paths, `price_fit_node` baskets |
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
credit and FX quote levels; reaching the remaining pricers through `price`
(CMS and range-accrual products on `ImpliedVolModel`; D15 / D16); an LMM
calibrator (D18); derivative convergence of the numerical engines; the position /
portfolio layer; a global piecewise σ(t) calibration on a user grid for
portfolio / XVA simulation (D13). Hull-White, G2++ (multi-curve, 2-D PDE
basket) and Cheyette 1F/2F (single-curve, frozen-y PDE basket) are all
validated per trade through the same `swaption_basket_node`, both as an exact
co-terminal bootstrap (D14) and as a least-squares fit of constant
parameters, where G2++ needs a basket beyond the co-terminals to identify both
factors (see [examples](../examples.md#per-trade-g2-and-cheyette)).
