# Pricing Ontology & Unified Signature — Implementation Tracker

> **This file is a working note, not documentation.** It is deliberately
> outside `docs/` and is *not* wired into MkDocs. Its job is to track
> **what is built and what is still to build** behind the unified
> `price(instrument, model, market)` contract, and to record *why* the
> design is what it is.
>
> **If you want to learn or use the contract, read the docs instead:**
>
> | You want | Read |
> |---|---|
> | Tutorial: the contract, worked examples, recipes | `docs/guide/tutorial-unified-pricing.md` |
> | API reference: `price`, `MarketState`, `resolve` | `docs/api/core.md` |
> | Unified sensitivities API | `docs/api/greeks.md` |
> | What is priced by what, per instrument | `docs/applications/instrument-pricing.md` |
> | Library-wide planned vs shipped | `docs/roadmap.md` |
>
> Do **not** copy the contract or the examples back into this file, and do
> not copy this file's backlog into the docs — the roadmap is the only
> planned-vs-shipped list the docs carry.
>
> **Start here: [§0 Outstanding work](#0-outstanding-work-the-backlog).**

---

> **Status:** **P1–P5 and P5.1 shipped ✅.** The ontology, signatures,
> invariants, and sensitivity stack below are the agreed target architecture;
> the open questions of §11 are resolved. P2 (the core types `MarketState`,
> `resolve`, `PricingResult`, and the unified `price(...)` registry, landed as
> adapters over the existing pricers), P3 (static market-selector fields on
> instruments), P4 (unified market/model/contract greeks in
> `valax/greeks/market_greeks.py`), and P5 (the model↔market id-binding on the
> short-rate models — `curve_id`, `from_market`, `with_market_curve`) are
> implemented and tested — no pricer rewrites, nothing breaking. **P5.1** then
> carried the resolved curves into the MC and PDE dispatch and gave the
> Hull-White / G2++ recipes a genuine discount/projection split via the
> deterministic tenor-basis overlay `valax/pricing/multicurve.py` (§12).
> `initial_curve` remains embedded on the models as the *default* discount
> curve, now overridable per trade. See §8 for the phase table.
>
> **Author intent:** unify the pricing/risk call conventions behind one
> forward-carried contract that scales as instruments, models, and
> calibration routes grow — *without* rewriting existing pricers and
> *without* imposing multi-curve on single-curve trades.

---

## 0. Outstanding work (the backlog)

Everything still to build, consolidated from §12.8, §13 and §14.2 so this
one list is the thing to read before opening a session. Ordered roughly by
payoff. Tick an item here **and** flip the corresponding `docs/roadmap.md`
line in the same change (`AGENTS.md` definition of done).

### 0.1 Highest payoff

- [x] **Keyed scenarios + a `MarketState` shock primitive** (part 1 of the
  risk-layer migration). `KeyedScenario`
  (`valax/market/keyed_scenario.py`) addresses curves, surfaces, spots and
  dividends **by identifier**; `valax.risk.shock`
  (`valax/risk/state_shocks.py`) applies one to a `MarketState`, routing
  curve moves through `bump_graph_curve` and vol moves through
  `shift_surface` / `ShiftedVolSurface` (`valax/surfaces/shifted.py` —
  design fork resolved in favour of a uniform parallel-shift decorator over
  grid-specific node shocks, which keeps `SensitivityLadder.delta_vol` /
  `vanna` / `volga` one scalar per asset). `MarketScenario` /
  `apply_scenario` stay as the legacy view for one deprecation cycle, with
  `keyed_from_market_scenario` bridging them. Same change fixed the silent
  `curve_graph` drop in `reprice_under_scenario` / `parametric_var` /
  `pnl_attribution`.
- [x] **Port the repricing loop to `price(...)` and deprecate
  `wrap_equity_pricing_fn`** (part 2). `valax/risk/state_var.py` reprices
  a **homogeneous group** on a `MarketState` under a `KeyedScenario`
  through the unified front door, modelled on
  `valax/greeks/market_greeks.py::_price_scalar` for the `.price` unwrap:
  `group_pv` / `group_prices` / `group_reprice` / `group_scenario_pnl` /
  `group_market_gradient` / `group_model_gradient`, plus `group_size` as
  the pre-trace group validator. `method` / `config` / `key` are keyword
  arguments mirroring `price`'s own signature rather than fields of a
  group object — reifying the group is the aggregator's decision (below),
  not the risk layer's. Gates in `tests/test_risk/test_state_var.py`:
  **bit-exact** against `reprice_under_scenario` on a single-curve rates
  and equity book, the common-random-numbers gate for MC groups, and a
  dual-curve basis widening worth +9,817 that the unkeyed legacy vector
  reports as −150.

  Two findings worth carrying forward. (a) The legacy path **cannot** be
  re-expressed in terms of the new one — a `pricing_fn` is an opaque
  callable carrying no model, and `price` dispatches on
  `(instrument, model, method)` — so `valax/risk/var.py` is frozen at its
  current behaviour rather than reimplemented, and the whole existing
  test suite stays untouched. (b) `BlackScholesModel` carries its own
  `rate` / `vol` / `dividend`, so a curve or surface shock does not move
  an equity price through `price`; that risk lands on
  `group_model_gradient`. Pinned by `TestEquityCurveShockIsInert`, closed
  by full curve de-embedding (§0.3).

- [ ] **Portfolio aggregation layer** (part 3 — the session this one was
  explicitly scoped to enable). The group primitives above return the
  three additive quantities; what is missing is the layer that sums them
  over a heterogeneous book.

  **Decisions already taken, do not re-litigate:**

  1. **Risk layer does homogeneous groups; the portfolio layer
     aggregates.** Because every group differentiates w.r.t. the *same*
     `MarketState` pytree, per-group gradients share a structure and
     aggregate by `jax.tree_util.tree_map(add, …)` — aggregation happens
     in the market's own coordinate system, and `risk/bucketing.py` stays
     the coordinate change applied *after* aggregation, not a
     prerequisite. Verified by
     `TestAggregationContract::test_gradients_tree_map_add_across_groups`.
  2. **The grouping key is stricter than "instrument class":**
     `(instrument type, static field values, model type + its statics,
     method, config)`. `EuropeanOption.is_call` is `static=True`, so calls
     and puts cannot share a stacked pytree — cf. `sample_option_portfolio`,
     which already returns them separately. Differing statics change the
     treedef, so such instruments cannot be stacked at all; `group_size`
     catches the remaining ragged/unstacked cases pre-trace.
  3. **Return surface of the group-level primitives** — shipped: the
     scalar PV, the gradient *pytree*, and the `(n_scenarios,)` P&L
     vector, with an explicit `key` argument. **Not** a VaR number.
  4. **Three traps the aggregation must own** (the fourth is discharged).
     (a) *Risk measures are not additive*: a quantile of a sum is not a
     sum of quantiles, so aggregate the P&L **vector** and apply
     `value_at_risk` / `expected_shortfall` / the PLA tests only at the
     top; an API that permits adding per-desk VaRs will produce a
     plausible wrong number with no error — the same failure class as the
     `curve_graph` drop. (b) ~~*MC common random numbers*~~ — discharged:
     `group_scenario_pnl` closes over one key for the base and every
     shocked revaluation, gated by
     `TestCommonRandomNumbers::test_fresh_key_per_revaluation_is_noise`
     (measured −0.207 of spurious P&L on a zero scenario against a +1.16
     real move). Pass a different key per *group*, never per scenario.
     (c) *Absent ≠ zero*: `theta` raises `NotImplementedError` for
     date-based instruments and a market-surface vega does not exist for a
     group whose market has no surface, so the aggregate needs a keyed
     union with explicit absence, or a mixed equity/rates book reports an
     equity-only theta as the book's theta. Note the equity finding above
     makes this sharper: rate/vol risk for an equity group is on the
     *model* gradient, so a book-level "rho" is a keyed union across two
     different gradient pytrees, not a sum. (d) *Quantity and sign* belong
     to the portfolio layer — `price` returns a unit price and instruments
     are data-only; `group_prices` is the per-instrument vector to weight.
  5. **`valax/portfolio/` is not reusable as-is and there is no PV
     aggregation to copy.** It is two functions (`batch_price`,
     `batch_greeks`) on the pre-curve loose-scalar convention
     `(spots, vols, rates, dividends)`, homogeneous-only, with
     `batch_greeks` calling the old `greeks/autodiff.py`; there is no
     `Portfolio` / `Position` type and no `tests/test_portfolio/` package.
     Build the layer once for the three additive quantities rather than
     once per quantity, and treat `batch_price` / `batch_greeks` like
     `wrap_equity_pricing_fn` — public API in `docs/getting-started.md`
     and `docs/scaling/batching.md`, so they need a deprecation decision,
     not a quiet patch.

- [x] **Keyed generators and a per-curve covariance layout.** Shipped as
  `KeyedFactorLayout` / `FactorBlock` in `valax/risk/scenarios.py`: an
  ordered tuple of blocks, **one per named market object**, derived from
  the market (`from_market_state`), naming each coordinate with the
  hierarchical factor id of `docs/risk-factors.md` § 1.2, with
  `flatten` / `unflatten` as exact inverses. On top of it
  `block_covariance` (per-curve blocks + explicit cross-correlations,
  factored by `psd_factor` so a singular covariance samples rather than
  NaN-ing) and the generators `keyed_parametric_scenarios` /
  `keyed_historical_scenarios` / `keyed_stress_scenario` /
  `keyed_curve_stress` / `keyed_basis_shift`. A basis move now shows up in
  a **VaR** number, and `as_basis_layout` makes the basis a named
  coordinate so `keyed_component_var` attributes that VaR to it. The
  measurement layer followed in `valax/risk/state_ladders.py`
  (`KeyedLadder`, `keyed_waterfall_pnl`, `keyed_parametric_var`,
  `keyed_component_var`), which differentiates the *shock* rather than the
  market leaves and so needs no DF→zero-rate chain rule.
  `_warn_frozen_graph_curves` **stays**: it guards the legacy
  `MarketData` / `MarketScenario` path, which is still flat and still
  shipped for one deprecation cycle. What is left above this is portfolio
  aggregation (part 5 above) and re-targeting the FRTB PLA vectors in
  `valax/risk/pnl_vectors.py` onto the keyed ladder.
- [ ] **Close the quote-space (hedge-space) loop.** Compose
  `∇_MarketState price` (`valax/greeks/market_greeks.py`) with
  `valax/curves/bootstrap_graph.py::quote_jacobian` and
  `risk/bucketing.pushforward_sensitivities` so reported risk lands in
  quote space, not pillar space. The `J_bucket` leg shipped; the
  `J_build` leg is still left to the caller. (§6.3)
- [x] **Per-leg index descriptors for genuinely multi-index products.**
  Shipped as `valax.instruments.legs` (`FixedLeg`, `FloatingLeg`,
  `MultiLegSwap`) with pricing in `valax.pricing.analytic.legs`, built
  together with the cashflow/leg engine of
  [roadmap Tier 1.4](docs/roadmap.md#14-trade-representation-and-cashflow-engine)
  — one thing, not twice, as this item required. Each leg names its own
  `discount_id` / `forward_id` / `index_id`; a three-curve tenor basis
  swap is now expressible and tested. Details in §14.2. The legacy
  single-trade pytrees keep the singular pair, so the FX / credit /
  inflation selector item above is untouched by this.

### 0.2 Correctness / coverage gaps

- [ ] **Basis volatility.** The P5.1 tenor basis is frozen, so a basis
  swaption or tenor-spread option is worth exactly zero under the
  overlay. Needs a stochastic-basis multi-curve HJM extension (roadmap
  Tier 2.1, `docs/research-ideas.md`). (§12.8)
- [ ] **Multi-curve for the CMS-spread recipes** (G2++ PDE and MC). They
  rebuild par swap rates from the model's own bond strip, so they accept
  `discount_curve` but *raise* on a distinct `forward_curve`. (§12.8)
- [ ] **Multi-curve for the Cheyette and LMM recipes** — untouched, still
  single-curve (they do not name the curve parameters, so the dispatch
  bridge does not forward them). (§12.8) This also blocks dual-curve
  Cheyette *calibration*, which refuses rather than ignores a projection
  curve.
- [x] **Dual-curve calibration for the exact-fit short-rate models** —
  closed the consistency hole where P5.1 priced dual-curve while every
  calibrator fitted single-curve (the basis leaked into `σ`: measured at a
  62 % bias in the fitted vol for a 50 bp basis). Required extending the
  *closed forms* first — `hw_swaption_price` (Jamshidian, `par_value` on
  `hw_critical_rate`) and `g2pp_swaption_price` (eq. 4.31 boundary + strike
  leg), the latter previously raising — then threading `forward_curve` +
  `curve_id` through `calibrate_hull_white`, `calibrate_g2pp` and
  `swaption_prices_from_vols`. Docs: `docs/guide/calibration.md` § 6,
  `docs/theory/multi-curve-short-rate.md` § 5.1.
- [ ] **QuantLib cross-check for the rates calibrators** — none of the
  short-rate calibrators has a QL comparison in either mode; the current
  gates are internal (PDE agreement, payer/receiver parity), so they prove
  self-consistency, not convention. Results should be *logged* per run.
  Tracked in `docs/roadmap.md` (QuantLib Validation Pyramid, stage 2).
- [x] **Port the risk *measurement* layer onto `MarketState`.** Shipped as
  `valax/risk/state_ladders.py`: `keyed_ladder` → `KeyedLadder` indexed by
  a `KeyedFactorLayout`, `keyed_waterfall_pnl` /
  `keyed_waterfall_report` for a per-block explain, and
  `keyed_parametric_var` / `keyed_component_var` for a delta-normal VaR
  whose Euler decomposition is additive across curves. The successor is
  *better* than the `dict[curve_id, ladder]` originally sketched: a single
  factor vector indexed by one layout, which the scenarios and the
  covariance matrix share, so the three line up by name. It also
  differentiates the **shock magnitudes** rather than the market leaves,
  which deletes the DF→zero-rate chain rule (and its gamma correction) and
  makes vega well-defined for every surface type.
  `hypothetical_pnl_vector` needed no port: it is a one-line alias of
  `portfolio_pnl`, and `group_scenario_pnl` is already its keyed
  equivalent.
- [ ] **Re-target the FRTB PLA vectors.** `risk_theoretical_pnl_vector`
  (`valax/risk/pnl_vectors.py`) still consumes the *positional*
  `compute_ladder` output, and `pnl_attribution` / `parametric_var`
  differentiate the same flat layout. `KeyedLadder` is what they point at
  now; `compute_ladder` and `ladders.py` stay for one deprecation cycle.
  The node-level vega ladders and the single-instrument graph-level greeks
  API are the other two consumers of the keyed layout
  ([roadmap §1](docs/roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)).
- [ ] **Rates theta as a valuation-date roll.** `greeks/market_greeks.py::theta`
  raises `NotImplementedError` for date-based instruments; implement the
  `reference_date` roll (carry-and-roll repricing, never a derivative of
  an integer date). (§6.2)
- [ ] **`resolve` for FX and credit instruments.** `discount_id` /
  `forward_id` landed on rates + bonds and `underlying_id` on
  single-underlying equity options; `valax/instruments/{fx,credit}.py` and
  multi-asset baskets are deferred — they need currency and
  survival-curve handling `resolve` does not do yet
  ([roadmap](docs/roadmap.md)). (§9)
- [ ] **Pricer-side fixings.** `MarketState.fixings` is carried through
  `resolve` but **no pricer reads it**: seasoned FRNs still use an
  unrelated `fixing_rates` NaN array on the instrument pytree. Unify with
  the leg engine above. (§14.2)
- [ ] **Day-count mismatch between recipe time axis and curve** (pre-existing,
  surfaced by the P5.1 tests): the short-rate recipes build their time axis
  from the instrument's `day_count` while `DiscountCurve` uses its own —
  ~1.4 % for an `act_360` trade on an `act_365` curve. Orthogonal to
  multi-curve, worth its own pass. (§12.8)

### 0.3 Hygiene / tidy-up

- [ ] **Static/dynamic pytree hygiene audit.** Confirm no string id leaks
  into traced code across `MarketState`, `CurveGraph` and the instrument
  selectors. The convention has been followed by hand, but the audit was
  never run and nothing enforces it. (§14.2)
- [ ] **Full curve de-embedding.** The endpoint of §3.1: the model carries
  only `curve_id` and the exact-fit shift is computed at price time.
  P5 + P5.1 already deliver the *behaviour* (see §14.3), so this is a
  tidy-up **for the rates models**, not a blocker. (§14.2)

  Part 2 gave it a concrete cost and split it cleanly in two. Because
  `BlackScholesModel` carries its own `rate` / `vol` / `dividend`, a curve
  or surface shock does **not** move an equity price through `price`, so
  an equity group's rate and vol risk sit on `group_model_gradient` while
  its spot risk sits on `group_market_gradient`. A book-level rate ladder
  therefore has to union two different gradient pytrees. Pinned by
  `tests/test_risk/test_state_var.py::TestEquityCurveShockIsInert`, and no
  longer *silent*: `valax/risk/state_var.py::_warn_shadowed_shocks` warns
  at the user boundary when a scenario's shock family cannot reach the
  price, detected structurally from the model pytree's own fields
  (`TestShadowedShockGuard`).

  The affected set is exactly the equity family — `BlackScholesModel`,
  `HestonModel`, `LocalVolModel`, `SLVModel`, `MultiAssetGBMModel` all
  carry an intrinsic scalar `rate`. **The rates models are already
  correct, and measurably so:** a curve shock reaches a Hull-White price
  *bit-identically with or without* `with_market_curve`, because the
  recipe takes `discount_curve` from the market rather than from the
  model's `initial_curve` (measured −1.902 on an HW bond vs −1.901
  curve-discounted for the same +100 bp). That confirms §14.3 — P5/P5.1
  delivered the behaviour and `initial_curve` is vestigial in the price
  path. So de-embedding `HullWhiteModel` / `G2PPModel` is cosmetic.

  **Shipped for three of the five equity models** (`BlackScholesModel`,
  `HestonModel`, `LocalVolModel`): `rate` / `dividend` — and BS `vol` —
  may be left `None` and are sourced from the `MarketState` by
  `valax/core/binding.py::bind_model`, a **binding layer** between
  `resolve` and pricer dispatch. Design notes worth keeping:

  * The seam is *bind-then-price*, not optional reads at each pricer.
    `model.rate` / `.vol` / `.dividend` are read at ~34 sites across the
    MC path generators, the MC engine, the PDE coefficient builders and
    the PDE boundary conditions — and `pde/recipes.py` uses `model.vol` to
    **size the spot grid**. Binding once, above all of them, left every
    one of those sites untouched. Generalises
    `HullWhiteModel.with_market_curve`.
  * Because bind runs *inside* the differentiated path, a sourced
    parameter traces back to the curve/surface: an equity curve shock now
    moves the price (+0.92 on a 1Y two-option book for +100 bp) and
    `group_market_gradient` puts the delta on **the trade's own pillar**,
    not a fixed one. The `_warn_shadowed_shocks` guard then stops firing
    by itself, because `model.rate is None`.
  * A fully-specified model is returned by `bind_model` **unchanged**
    (identity, not a copy), which is what makes constant mode
    bit-identical and left all ~90 existing model constructions alone.
  * Required new primitive: `valax.dates.ordinal_from_year_fraction`,
    the inverse of `year_fraction`. No year-fraction → ordinal bridge
    existed anywhere, and the two *forward* conventions in the tree
    already disagreed (day-count-aware in the rates pricers vs hard-coded
    `/365.0` in `risk/var.py`). It returns a **fractional** ordinal so the
    round trip is exact and the path stays differentiable; `30_360` raises,
    having no closed-form inverse.
  * Guarded rather than trusting: `resolve_surface` does no kind checking
    and falls back to the market's default surface whatever it is, so
    `sample_vol` refuses `LeverageGrid` (same arity, log-moneyness units)
    and `SwaptionCube` (needs a tenor), and refuses `is_normal` surfaces
    outright rather than feeding Bachelier vols to a lognormal form.
  * Still a *constant-parameter* model: projecting at each trade's own
    maturity is right per trade but is not a term structure.

  Left for a follow-up (shape problems, not design problems): `SLVModel`
  (its `rate` / `dividend` precede the non-defaulted `surface` /
  `leverage`, and its docstring declares field order a frozen contract) and
  `MultiAssetGBMModel` (`vols` / `dividends` are per-asset vectors and
  `n_assets` derives from `vols.shape[0]`).
- [ ] **Retire `MarketData`.** Q4 granted it one minor release as a two-way
  view. Part 2 moved the repricing loop off it and deprecated
  `wrap_equity_pricing_fn` (which goes at the same time); what still holds
  it alive is the rest of `valax/risk/var.py`, the ladders, and the
  scenario generators (§0.2). Once those target `price(...)` it can go,
  along with `docs/api/market.md`'s deprecation note. (§4, §11-Q4)
- [ ] **End-to-end unified example + local example guard** — see §13. New
  `examples/10_unified_pricing.py` exercising `MarketState` → `price`
  (auto/mc/pde) → `market_greeks` → multi-curve selectors → Hull-White
  `from_market`/`with_market_curve`, plus
  `tests/test_examples/test_examples_run.py` (`runpy` smoke, MC/PDE ones
  marked `slow`) and widening the `examples.yml` glob to
  `examples/pitfalls/`. The docs tutorial
  (`docs/guide/tutorial-unified-pricing.md`) now covers the same ground
  prose-side, so this is about the *executable* guard.
- [ ] **Guard the tutorial's snippets too.** Every code block in
  `docs/guide/tutorial-unified-pricing.md` was executed in document order when
  the page was written (and the quoted numbers are that run's output), but
  nothing re-runs them, so the page can rot silently. Cheapest fix: extract
  the fenced `python` blocks and `exec` them in sequence from the same
  `tests/test_examples/` module as the examples guard, with the one
  intentionally-raising block whitelisted.

### 0.4 Done (for orientation)

P1 spec · P2 core types + unified dispatch · P3 instrument selectors ·
P4 unified greeks · P5 model↔market id-binding · P5.1 multi-curve MC/PDE.
Detail in §8; acceptance results in §10 and §12.6.

---

## 1. Why — the problem this fixes

Three pricing-call conventions coexist today, and market data lives in
inconsistent places. A future contributor cannot answer "what is the
signature of a pricer?" with one sentence.

### 1.1 The three conventions (current state, grounded)

| # | Convention | Example (current signature) | Where |
|---|---|---|---|
| 1 | **Analytic** — loose positional scalars | `black_scholes_price(option, spot, vol, rate, dividend)` | `valax/pricing/analytic/*.py` |
| 2 | **MC/PDE dispatch** — model pytree + `**market_args` | `pde_price_dispatch(instrument, model, config, **market_args)` · `mc_price_dispatch(instrument, model, config, key, **market_args)` | `valax/pricing/pde/dispatch.py`, `valax/pricing/mc/dispatch.py` |
| 3 | **Risk** — single market pytree | `pricing_fn(instrument, market: MarketData) -> price` | `valax/risk/var.py` |

`valax/risk/var.py::wrap_equity_pricing_fn` is the *only* existing adapter
bridging (1) → (3).

### 1.2 Market data is scattered

| Model | Curve? | Rate/div? | Surface? | File |
|---|---|---|---|---|
| `HullWhiteModel` | `initial_curve: DiscountCurve` (embedded) | via curve | — | `valax/models/hull_white.py` |
| `G2PPModel` | `initial_curve: DiscountCurve` (embedded) | via curve | — | `valax/models/g2pp.py` |
| `HestonModel` | — | scalar `rate`, `dividend` | — | `valax/models/heston.py` |
| `SABRModel` | — | — (forward at price time) | — | `valax/models/sabr.py` |
| `LocalVolModel` | — | scalar `rate`, `dividend` | `surface` | `valax/models/local_vol.py` |
| `SLVModel` | — | scalar `rate`, `dividend` | `surface` + `leverage` | `valax/models/slv.py` |

The risk layer meanwhile centralises everything in `MarketData`
(`spots`, `vols` *scalar-per-asset*, `dividends`, `discount_curve`,
optional `curve_graph`) — `valax/market/data.py`.

---

## 2. The ontology — 4 nouns, 6 verbs

### 2.1 Nouns (data pytrees)

1. **`Quote` / `QuoteSet`** — raw observables (deposit/swap rates, futures,
   vol quotes, option prices), each carrying a value + a *target id* naming
   the curve/surface it feeds. (The bootstrap instruments in
   `valax/curves/instruments.py` already carry `curves_touched` — this is
   the seed of the pattern.)
2. **`MarketState`** — the *calibrated market container* (see §4). Holds a
   `CurveGraph`, a keyed `dict[str, VolSurface]`, spots/fixings, credit and
   other curves. **Replaces `MarketData`** (rename accepted; it is a genuine
   generalisation, not a cosmetic rename — see §4).
3. **`Model`** — dynamics selection + calibrated params + **static id
   references** to the `MarketState` pieces it reads. `BlackScholesModel`
   counts (a closed form is a model). Intrinsic params (mean-reversion,
   vol-of-vol, `LeverageGrid`) live on the model; market observables are
   referenced by id, not duplicated (see §3.1).
4. **`Instrument`** — contract data (differentiable numeric leaves +
   `static` conventions) **plus static *market selectors*** (discount curve
   id, projection index id, fixing index). Selectors default to the
   single-curve sentinel (§5).

Supporting result/report pytrees: **`PricingResult`** (generalises the
existing `PDEResult`/`MCResult`: `price` + optional `stderr`/diagnostics),
**`Sensitivities`**, **`RiskLadder`**, **`BucketMap`** (already exists).

### 2.2 Verbs (pure functions / workflows)

| Verb | Signature (target) | Reuses today |
|---|---|---|
| `build_market` | `build_market(quotes) -> MarketState` | `bootstrap_curve_graph`, surface fitters |
| `calibrate` | `calibrate(model_family, market, targets) -> (model, solution)` | `valax/calibration/*` (all return `(model, optx.Solution)`) |
| `price` | `price(instrument, model, market, *, method="auto", config=None, key=None) -> PricingResult` | unified over `pde/mc` dispatch + analytic |
| `greeks` | `greeks(instrument, model, market, ...) -> Sensitivities` | `valax/greeks/autodiff.py`, `jax.grad` |
| `quote_sensitivities` | chain-rule of build/calibration Jacobians with raw greeks | `quote_jacobian`, `risk/bucketing.py` |
| `bucket` | `bucket(sensitivities, bucket_map) -> RiskLadder` | `risk/bucketing.py` |

### 2.3 The unified pricing signature (the headline decision)

*The signature as **shipped**, with worked examples, is documented in
`docs/guide/tutorial-unified-pricing.md` and `docs/api/core.md`. Repeated
here only so the decision and the alternative it beat stay legible.*

```python
price(
    instrument,          # contract data + static market selectors
    model,               # dynamics + calibrated params + static market refs
    market,              # MarketState (curves + surfaces + spots + credit …)
    *,
    method: str = "auto",   # "analytic" | "mc" | "pde" | "lattice" | "auto"
    config=None,            # method-specific numerical settings (PDEConfig/MCConfig)
    key=None,               # PRNG key for MC
) -> PricingResult
```

**Rejected alternative:** `V(instrument, ModelData, MarketData, *args, **kwargs)`
with open `*args/**kwargs`. Open kwargs are fragile, hard to JIT, and
undocumentable. Named `method`/`config`/`key` accommodate growth because new
methods and (instrument, model) pairs register into the **dispatch**, not the
signature. Open `**market_args` survive only at the internal recipe layer.

---

## 3. Design decisions (resolved forks)

### 3.1 Model ↔ market coupling — **hybrid reference-by-id** (chosen)

A model is a *function of* market data (it is calibrated to it), so the model
and market are not freely-combinable independent inputs.

- **Chosen:** model keeps *intrinsic* calibrated params; **references market
  observables (curves, surfaces) by static string id**, bound late at price
  time via a `resolve(instrument, model, market)` helper. Single source of
  truth; scenario-shocking `MarketState` flows through automatically; no
  duplicated curves.
- **Rejected (embed):** today's `HullWhiteModel.initial_curve` copies the
  curve into the model. Self-contained but duplicated, and a shocked market
  requires rebuilding the model.
- **Migration nuance:** short-rate models currently embed `initial_curve`.
  Phase 5 migrates them to id-references *behind backward-compatible
  constructors* (keep `HullWhiteModel(initial_curve=...)` working during the
  deprecation cycle).
- **Realised as a binding layer.** The chosen fork is implemented by
  `valax/core/binding.py::bind_model`, called by `price` between `resolve`
  and dispatch: a parameter given explicitly stays intrinsic, a parameter
  left `None` is read from the `MarketState`. Both modes on one model
  class, so the textbook `BlackScholesModel(vol=…, rate=…, dividend=…)`
  keeps working unchanged while `BlackScholesModel()` is fully
  market-driven. The four layers (kernel → model → binding → front door)
  are documented in `docs/api/core.md` § Model binding. Note this is
  *parameter* binding; the curve-**object** de-embedding of §0.3 (dropping
  `initial_curve` in favour of `curve_id` alone) is the remaining cosmetic
  step for the rates models.

### 3.2 Consistency invariant (state it in code + docs)

> **A model is only valid paired with the `MarketState` it was calibrated
> against (or a scenario-shocked version reusing the same ids).**

You never hand-assemble an independent `(model, market)` pair and hope they
agree: `calibrate(...)` *produces* the model from the market and records the
ids it depends on; `price` asserts those ids resolve in `market`. For
end-to-end differentiability w.r.t. raw vol quotes, calibration sits in the
differentiated path (implicit adjoint — same trick as the curve bootstrap).

### 3.3 Method selection — **one unified registry** (chosen)

Collapse the two identical registries (`pde/dispatch.py`, `mc/dispatch.py`,
both keyed on `(type(instrument), type(model))`) into one keyed on
`(instrument_type, model_type, method)`. Register analytic and lattice too.
`method="auto"` picks a default per pair (e.g. European+BlackScholes →
analytic; European+Heston → COS/ADI; Bermudan+HullWhite → pde). Existing
`pde_price_dispatch` / `mc_price_dispatch` become back-ends behind `price`.

### 3.4 Ownership of market selectors

- **Instrument owns *contractual* selectors** — which curve *this trade*
  discounts on, which index it projects, its fixing index. Resolves the
  instrument↔market coupling. Generalises `curves_touched`.
- **Model owns *modelling* references** — which surface/cube it reads, which
  curve it was calibrated against.
- Both are `eqx.field(static=True)` **strings** (pytree-static, JIT-safe, out
  of the gradient path).

---

## 4. `MarketState` — generalisation of `MarketData` (not a rename)

`MarketState` **subsumes** `MarketData` as a special case.

| Aspect | today's `MarketData` | target `MarketState` |
|---|---|---|
| curves | one `discount_curve` + optional `curve_graph` | **`CurveGraph` primary** (keyed; single-curve is a 1-entry graph) |
| vols | `vols`: **scalar per asset** | **`dict[str, VolSurface]`** (full surfaces/cubes) |
| spots/fixings | `spots`, `dividends` vectors | keyed spots + fixing histories |
| credit / inflation / FX | — | credit curves, inflation curves, FX (extensible) |

Surfaces use the **existing duck-typed protocol** (`total_variance(log_moneyness,
expiry)` / `__call__(strike, expiry)`) implemented by `GridVolSurface`, SVI,
`SABRVolSurface`, `LeverageGrid`, `swaption_cube`, `constant` — the one
genuinely polymorphic seam already in the codebase (`valax/surfaces/`).

**Compatibility:** keep `MarketData` for one deprecation cycle as a thin
view/adapter over `MarketState` (the `risk/var.py` framework consumes the
`MarketData` convention; a shim avoids breaking it during migration).

---

## 5. Single-curve must keep working (hard constraint)

**Multi-curve is opt-in, never imposed.** The codebase already supports
single-curve degeneracy — the migration must preserve it end to end:

- `CurveGraph` already degenerates to single-curve via the grandfathered
  `_default_` sentinel; `bootstrap_simultaneous` is a thin wrapper over
  `bootstrap_curve_graph` with `CurveSpec(curve_id="_default_")`
  (see [Multi-Curve Framework](docs/roadmap.md#multi-curve-framework-what-ships-whats-left)).
- `graph_pricing.py` wrappers already reproduce single-curve behaviour
  **bit-for-bit** when `forward_id=None`.

**Contract for the unified layer:**

1. `MarketState` may hold a full `CurveGraph` *or* a single `_default_` curve;
   both are valid.
2. Instrument market selectors **default to the sentinel** (`discount_id =
   "_default_"`, `forward_id = None`). A single-curve trade needs zero extra
   configuration.
3. `resolve(...)` returns the discount curve for the sentinel and treats a
   `None` forward id as "project off the discount curve" — the existing
   single-curve semantics.
4. Acceptance: a single-curve price through the *new* `price(...)` path must
   equal the legacy loose-scalar / single-`DiscountCurve` price to
   `atol≈1e-10` (see §8).

---

## 6. Sensitivity model — three families + the Jacobian stack

### 6.1 Three sensitivity families (all first-class)

| Family | Differentiate w.r.t. | Examples | Mechanism |
|---|---|---|---|
| **Market greeks** | `MarketState` leaves | delta, vega, curve DV01 | autodiff |
| **Model greeks** | model params | ∂/∂v0, ∂/∂κ, ∂/∂leverage | autodiff |
| **Contract greeks** | instrument leaves | **theta** (∂/∂expiry), **dual-delta** (∂/∂strike) | autodiff *or* bump (§6.2) |

> Correction to an earlier design note: instruments **do** carry
> differentiable contract leaves; contract greeks (theta, dual-delta) are
> legitimate. Instruments are not "non-differentiable data" — they are
> *contract parameters*, distinct from *hedgeable market factors*.

### 6.2 Theta / contract-derivative policy (autodiff vs bump)

The mechanism depends on whether the contract field is a *smooth float* or a
*discretisation-controlling / integer* field:

- **Analytic equity theta** — `EuropeanOption.expiry` is a smooth
  year-fraction float ⇒ `theta = -jax.grad(price, expiry)`. This should
  **replace** the current finite-difference `_theta` (which bumps `expiry`
  via `eqx.tree_at`, `dt=1/365`) *for the analytic path only*.
- **MC/PDE theta** — `expiry` also fixes the time-grid / step count (static
  discretisation); you cannot differentiate through step-count construction.
  Keep the **FD calendar bump** here.
- **Rates theta** — instrument dates are **integer ordinals** (`Int[Array]`),
  not smooth floats. "Theta" is a **valuation-date roll** (shift
  `reference_date` / carry-and-roll), a repricing bump — never a derivative
  of an integer.

The ontology classifies each sensitivity by *(which input) × (autodiff vs
bump, given field type/method)*.

### 6.3 The Jacobian stack (bucketing sits between market and reported risk)

```
                              ∇_MarketState price   (raw pillar / surface-node sensitivities)
 quotes ──J_build──▶ pillars ─────────────────────────────┤
      (bootstrap)     (surface nodes)                      ├─ J_bucket (n_buckets × n_pillars) ─▶ tenor-bucket risk (FRTB vertices)
                                                           ├─ J_quote  (∂pillar/∂quote)ᵀ        ─▶ quote / hedge-space risk
                                                           └─ J_pca                              ─▶ PCA factor risk
```

- **`J_build`** = `∂MarketState/∂quotes` — from `bootstrap_curve_graph` /
  `quote_jacobian` (`valax/curves/bootstrap_graph.py`) and surface calibration
  (implicit adjoint; computed **once** per snapshot).
- **`J_bucket`** = `BucketMap.matrix` (shape `n_buckets × n_pillars`) in
  `valax/risk/bucketing.py`, applied via `pushforward_sensitivities`
  (indicator or linear tenor splitting). **Bucketing is a linear coordinate
  change applied to the raw market-level sensitivity vector** — it sits
  between the pricer's market derivatives and the reported per-tenor risk.
- **`J_quote`**, **`J_pca`** are *sibling* coordinate changes over the same
  raw gradient (`pca_jacobian`, `level_slope_curvature_jacobian`). Nothing is
  recomputed per coordinate system — one raw gradient, three linear maps.
- **Hedge / model-risk translation:** hedgeable (quote-space) sensitivities
  are `∂price/∂quote = J_buildᵀ · ∇_MarketState price`; model-risk hedges
  compose additionally with the calibration Jacobian
  `∂model_params/∂quotes`.

---

## 7. Current-state grounding (so a future session need not re-discover)

| Concern | Symbol / file (current) |
|---|---|
| Analytic pricers (plain fns) | `valax/pricing/analytic/*.py` (e.g. `black_scholes_price`) |
| Curve-graph analytic wrappers | `valax/pricing/analytic/graph_pricing.py` (`*_from_graph`, `forward_id=None` ⇒ single-curve) |
| PDE dispatch + registry | `valax/pricing/pde/dispatch.py` (`register`, `pde_price_dispatch`, `PDEResult`); IR recipes in `pde/hull_white.py`, `pde/g2pp.py`, `pde/cheyette.py` |
| MC dispatch + registry | `valax/pricing/mc/dispatch.py` (`register`, `mc_price_dispatch`, `MCResult`, `discounted_mean_and_stderr`); recipes in `mc/recipes.py` |
| Models | `valax/models/{hull_white,g2pp,heston,sabr,local_vol,slv,black_scholes,cheyette,lmm,multi_asset}.py` |
| Calibration (→ `(model, solution)`) | `valax/calibration/{sabr,heston,hull_white,slv,g2pp,cheyette}.py`, `loss.py`, `transforms.py` |
| Vol surfaces (duck-typed) | `valax/surfaces/{grid,svi,sabr_surface,leverage,optionlet_surface,swaption_cube,constant,_interp}.py` |
| Greeks (scalar-arg autodiff; FD theta) | `valax/greeks/autodiff.py` (`greek`, `greeks`, `_theta`) |
| Market container | `valax/market/data.py` (`MarketData`) |
| Quote→pillar Jacobian | `valax/curves/bootstrap_graph.py::quote_jacobian` (`by ∈ {log_df, df, zero_rate}`, ImplicitAdjoint) |
| Bucketing / coordinate changes | `valax/risk/bucketing.py` (`jacobian_from_fn`, `tenor_bucket_map`, `BucketMap`, `pushforward_sensitivities`, `pca_jacobian`, `reparameterize_covariance`) |
| Shocks (return new pytrees) | `valax/risk/shocks.py` (`bump_curve_zero_rates`, `parallel_shift`, `key_rate_bump`, `bump_graph_curve`, `key_rate_graph_bump`, `bump_hazard_rates`) |
| Scenario repricing / VaR | `valax/risk/var.py` (`wrap_equity_pricing_fn`, `reprice_under_scenario`, `portfolio_pnl`, `value_at_risk`, `expected_shortfall`) |
| OAS (repricing-inversion) | `valax/risk/oas.py` (`price_under_spread` → `pde_price_dispatch`) |
| Instruments | `valax/instruments/{options,rates,bonds,fx,credit}.py`; selectors (`curves_touched`) only on `valax/curves/instruments.py` today |

---

## 8. Phased migration plan (each phase independently shippable, non-breaking)

| Phase | Deliverable | Notes / guardrails |
|---|---|---|
| **P1** ✅ | **This spec.** Ontology, signatures, invariants, sensitivity stack. | Design only; no code. (This document.) |
| **P2** ✅ | Core types + unified dispatch (thin, adapters only). `valax/core/market_state.py` (`MarketState`), `valax/core/pricing.py` (`PricingResult`, unified registry, `price(...)`), `valax/core/resolve.py`, `valax/core/adapters.py`. | **Shipped, no pricer rewrites.** Analytic/lattice pricers registered as adapters; the MC and PDE registries are *bridged* (a recipe registered there is reachable through `price` immediately — they cannot drift). `model=None` covers curve-discounted instruments. Tests: `tests/test_core/`. |
| **P3** ✅ | Static market-selector fields on instruments, defaulted to the sentinel. | **Shipped.** `discount_id`/`forward_id` on all rates + bond classes, `underlying_id` on single-underlying equity options (multi-asset baskets + fx/credit deferred). Read by `resolve` via `getattr` defaults, so single-curve trades are unaffected (§5); generalises `curves_touched`. Tests: `tests/test_core/test_pricing.py`. |
| **P4** ✅ | Unified greeks + quote/model sensitivities. `valax/greeks/market_greeks.py`. | **Shipped.** `market_gradient`/`model_gradient` (∇ over `price`), `spot_delta`, `curve_pillar_deltas` + `bucketed_curve_deltas` (the `J_bucket` leg via `risk.bucketing.aggregate`), `dual_delta`, and `theta` with the autodiff (analytic/lattice) vs FD-bump (PDE/MC) policy of §6.2; rates valuation-date-roll theta raises pending P4.1. Scalar `autodiff.py` untouched. Tests: `tests/test_greeks/test_market_greeks.py`. |
| **P5** ✅ | Model↔market id-binding on the short-rate models. `models/hull_white.py`, `g2pp.py`. | **Shipped (additive).** Static `curve_id` + `from_market(market, …, curve_id)` (binds & records the linkage) + `with_market_curve(market)` re-binder; `initial_curve` stays embedded so every existing pricer/test is untouched. Tests: `tests/test_models/test_market_binding.py`. |
| **P5.1** ✅ | Multi-curve in the MC/PDE solvers. `valax/pricing/multicurve.py` (new), `valax/core/pricing.py` (bridge), `pde/rates_payoffs.py`, `pde/{hull_white,g2pp}.py`, `mc/recipes.py`. | **Shipped (additive).** The dispatch bridge forwards the resolved `discount_curve`/`forward_curve` to any recipe that *names* them; Hull-White and G2++ honour the discount curve by re-fitting and the projection curve through a frozen deterministic tenor basis. `None` is a trace-time branch, so single-curve prices are bit-for-bit unchanged. Cheyette/LMM and the CMS-spread recipes stay single-curve (the latter refuse a distinct projection curve). Full §12. Tests: `tests/test_pricing/test_multicurve.py`, `tests/test_{pde,mc}/test_multicurve.py`. |

**Dependency order:** P1 → P2 → {P3, P4} → P5 → P5.1. P3 and P4 can proceed in
parallel once P2 lands.

---

## 9. File-by-file change list (for the migration session)

**Create:**

- ✅ `PRICING_ONTOLOGY.md` — this note (P1). Originally written into
  `docs/architecture/pricing-ontology.md`; moved out of the docs tree once
  its user-facing half became the tutorial
  `docs/guide/tutorial-unified-pricing.md`, because a phase-by-phase
  migration ledger is a contributor artefact, not documentation.
- ✅ `valax/core/market_state.py` — `MarketState` (+ the two-way `MarketData`
  view and the surface-id alphabet).
- ✅ `valax/core/pricing.py` — `PricingResult`, unified registry keyed
  `(instrument, model, method)`, `price(...)`, the `method="auto"` policy
  table, and the MC/PDE registry bridge.
- ✅ `valax/core/resolve.py` — `resolve(instrument, model, market)` id→curve/surface
  binding; sentinel + `forward_id=None` single-curve semantics.
- ✅ `valax/core/adapters.py` — thin adapters over the existing analytic and
  lattice pricers, plus the model-free (`model=None`) curve-discounted route.
- ✅ `valax/greeks/market_greeks.py` — market/model/contract greeks + theta
  policy (P4).
- ✅ `valax/pricing/multicurve.py` — the deterministic tenor-basis overlay
  (`with_discount_curve`, `tenor_basis`, `swap_leg_basis`, `apply_swap_basis`,
  `require_single_curve`) behind the MC/PDE discount/projection split (P5.1).
- ✅ `docs/theory/multi-curve-short-rate.md` — the modelling note for P5.1.
- Mirrored tests: ✅ `tests/test_core/test_pricing.py`,
  ✅ `tests/test_core/test_market_state.py`, ✅ `tests/test_core/test_resolve.py`,
  ✅ `tests/test_greeks/test_market_greeks.py`.

**Modify:**

- `valax/market/data.py` — `MarketData` → `MarketState` generalisation +
  compat view; update `valax/market/__init__.py` exports.
- ✅ ~~`valax/greeks/autodiff.py` — autodiff theta on the analytic path~~ —
  **not needed.** P4 added theta (autodiff analytic/lattice, FD PDE/MC) in the
  new `valax/greeks/market_greeks.py` instead; the scalar `autodiff.py` is left
  untouched as the legacy convenience API.
- ~~`valax/pricing/pde/dispatch.py`, `valax/pricing/mc/dispatch.py` — register
  under the unified registry~~ — **not needed, and better not done.** P2
  *bridges* those registries instead: `price` queries them directly, so a
  recipe registered the old way is reachable through the unified signature
  with no second registration to keep in sync.  Both files are untouched and
  remain the back-ends.
- ✅ `valax/instruments/{rates,bonds}.py` (`discount_id`/`forward_id`) and
  `valax/instruments/options.py` (`underlying_id`) — static selector fields,
  default sentinel (P3). `fx`/`credit` deferred (need currency / credit-curve
  handling `resolve` does not yet do).
- ✅ `valax/models/hull_white.py`, `valax/models/g2pp.py` — static `curve_id` +
  `from_market` / `with_market_curve` (P5, additive; `initial_curve` retained).
  Full de-embedding = P5.1.

**Reuse unchanged:** `valax/risk/bucketing.py` (`J_bucket`),
`valax/curves/bootstrap_graph.py` (`J_build` / `quote_jacobian`),
`valax/surfaces/*` (duck-typed protocol), `valax/calibration/*`.

**Docs / DoD (per `AGENTS.md`, in the same change as each code phase):**

- `docs/roadmap.md` — tick the relevant Tier 5.5 "unified market-quote
  container + quote-space risk" line **only when actually shipped**.
- `docs/applications/instrument-pricing.md` — no matrix change (signature
  refactor, not a new instrument/pricer) unless coverage changes.
- Owning guide pages + `docs/api/*` autodoc for new public symbols
  (`MarketState`, `price`, `market_greeks`). The contract's owning guide is
  `docs/guide/tutorial-unified-pricing.md` — a new method, selector, or
  named greek belongs there, with a runnable example.
- `mkdocs build --strict` must stay green. **This file is not part of the
  docs build** (it lives at the repo root, outside `docs/`), so nothing here
  is link-checked: keep its cross-references repo-root-relative and verify
  them by hand.

---

## 10. Acceptance criteria

1. **Single-curve parity (hard gate):** `price(instrument, model,
   single_curve_market)` equals the legacy loose-scalar / single-`DiscountCurve`
   price to `atol≈1e-10` for a representative vanilla per asset class, with
   **zero** selector configuration on the instrument.
2. **Multi-curve equivalence:** `price(...)` on a `MarketState` carrying a
   `CurveGraph` equals the corresponding `*_from_graph` result bit-for-bit.
3. **Dispatch coverage:** every `(instrument, model)` pair currently in the
   PDE/MC registries resolves through the unified `price(...)` with identical
   numbers.
4. **Greeks:** autodiff analytic theta matches the old FD theta within
   `rtol≈1e-4`; market/quote-space sensitivities match FD bumps within
   `rtol≈1e-4`; `J_bucket` aggregation preserves total DV01.
5. **No regressions:** full `pytest` green; `mkdocs build --strict` green.

**Status after P2:** 1 ✅ (equity vanillas, digitals, American lattice, and
curve-discounted bonds/swaps all match their legacy call to ≤ 1e-10 with zero
selector configuration), 2 ✅ (a selector-carrying instrument on a
`CurveGraph` reproduces the explicit dual-curve call; single-curve degenerates
exactly), 3 ✅ (`registered_pricers()` covers every pair in both legacy
registries, and MC/PDE prices are bit-identical to the dispatchers),
5 ✅. Criterion 4 belongs to P4 and is still open.

---

## 11. Open questions — resolved at the start of the P2 migration

All five were settled before any P2 code landed, on the same principle
that settled MC-Curves-2's Q8: *freeze the narrow thing now, leave the
wide thing pluggable.*

- **Q1 — `PricingResult` shape. → Resolved: one container with optional
  fields.** `PricingResult(price, stderr=None, method, diagnostics)`.
  A common protocol over three result types would force every caller to
  branch on the method it asked `price()` to choose for it, which defeats
  `method="auto"`. `PDEResult`/`MCResult` stay as the back-ends' own
  return types (no rewrites), and the adapters lift them; `stderr is
  None` is the honest encoding of "this method has no sampling error".
- **Q2 — `method="auto"` policy table. → Resolved: a separate
  `_AUTO_POLICY` dict plus a fallback priority order.** Keeping the
  policy off the registry keeps *availability* (what can be priced) and
  *preference* (what should be) independently reviewable, and lets the
  default be stated for a pair before every method for that pair is
  registered. Explicit policy entry wins; otherwise the first available
  method in `("analytic", "lattice", "pde", "mc")` — cheapest-and-exact
  first — is used.
- **Q3 — Surface keying convention. → Resolved: `<UNDERLYING>.<KIND>`
  `[.<QUALIFIER>]`**, validated at `MarketState` construction, with the
  `_default_` sentinel grandfathered exactly as for curve ids.
  `UNDERLYING` is a ticker / ISO pair / `<CCY>.<INDEX>` rates root and
  `KIND ∈ {VOL, LV, LEVERAGE, CUBE, OPTIONLET}` — e.g. `SPX.VOL`,
  `EURUSD.VOL`, `USD.SOFR.CUBE`. Rates cubes reuse the curve alphabet's
  first two fields so one regex covers both families.
- **Q4 — `MarketData` deprecation length. → Resolved: one minor release
  as a two-way view.** `MarketState.from_market_data` /
  `MarketState.to_market_data` are lossless for everything `MarketData`
  can express, so `risk/var.py` keeps its convention untouched while new
  code moves over. Longer than one release would mean maintaining two
  live market containers, which is the duplication this refactor exists
  to remove.
- **Q5 — Contract-greek surface. → Resolved: `theta` and `dual_delta`
  are promoted; everything else stays ad-hoc `jax.grad`.** Those two are
  quoted and risk-managed on a desk; the rest of the contract-leaf
  derivatives (∂/∂barrier, ∂/∂cap-strike) are instrument-specific and
  better served by the generic autodiff entry point than by a name in a
  dict. Mechanism per field type follows §6.2 (autodiff on the analytic
  path, bump/roll for MC/PDE and integer-dated rates trades).

---

## 12. Multi-curve in the MC/PDE solvers (P5.1) — **shipped ✅**

**Status:** **shipped** for Hull-White and G2++ (PDE *and* MC), additively and
non-breaking. The overlay lives in `valax/pricing/multicurve.py`; the theory is
`docs/theory/multi-curve-short-rate.md`; tests are
`tests/test_pricing/test_multicurve.py`, `tests/test_pde/test_multicurve.py`
and `tests/test_mc/test_multicurve.py`. Cheyette, LMM, and the CMS-spread
recipes remain single-curve — the CMS recipes *reject* a distinct projection
curve rather than ignoring it (§12.8).

### 12.1 Where multi-curve is / isn't wired

- ✅ **Analytic** rates pricers consume the dual curves. The model-free
  adapters call `swap_price / ois_swap_price / floating_rate_bond_price`
  with **both** `resolved.discount_curve` and `resolved.forward_curve`
  (`valax/core/adapters.py`), and the standalone `graph_pricing.py`
  `*_from_graph` wrappers do the same. So OIS-discounting + a distinct
  tenor projection curve works through `price(...)` for the analytic path.
- ✅ **MC and PDE**, *since P5.1*. Previously the unified bridge's
  `_recipe_market_args` forwarded **only `spot`**, so the resolved curves
  never reached a recipe and HW/G2PP products discounted **and** projected
  off the model's single embedded `initial_curve`. Now a recipe opts in by
  naming `discount_curve` / `forward_curve` in its signature, and the bridge
  forwards them from the `ResolvedMarket`.
- ⛔️ **Cheyette / LMM** recipes are untouched and remain single-curve: they
  do not declare the parameters, so the bridge does not forward them.

### 12.2 Why this is more than plumbing

A one-factor Hull-White (and G2++) is an **exact-fit-to-one-curve**
short-rate model: it produces a single term structure, so genuine
multi-curve (OIS discounting with a separate forward curve carrying a
basis) cannot be represented by handing the recipe a second curve alone.
It needs a **modelling decision** — the simplest defensible one is a
*deterministic tenor-basis overlay*: discount cashflows on the resolved
discount (OIS) curve, project floating rates off a forward curve, and
treat the projection-minus-discount basis as a deterministic spread the
stochastic factors do not touch. (A fully stochastic-basis multi-curve
short-rate / HJM setup is out of scope for P5.1.)

### 12.3 What shipped

- **P5.1a — dispatch takes curves.** `_recipe_market_args`
  (`valax/core/pricing.py`) now forwards `discount_curve` /
  `forward_curve` from the `ResolvedMarket`, but **only to a recipe that
  names them explicitly**. A `**market_args` catch-all deliberately does
  *not* qualify: nearly every rates recipe carries one, so a var-kwargs
  match would hand a projection curve to a recipe that silently drops it —
  a wrong price with no signal. Opting in is a property of the signature.
- **P5.1b/c — the overlay.** `valax/pricing/multicurve.py`:
  - `with_discount_curve(model, curve)` — re-fits an exact-fit short-rate
    model to the trade's discount curve (the price-time analogue of
    `with_market_curve`); returns the model *unchanged* for `None` or for
    the curve it already carries, which is what makes single-curve
    bit-for-bit rather than merely close.
  - `tenor_basis(...)` / `swap_leg_basis(...)` — the frozen multiplicative
    basis `b = (F(T_s)/F(T_e)) / (D(T_s)/D(T_e))`, returning `None` (not a
    vector of ones) in the single-curve case.
  - `apply_swap_basis(...)` — the one implementation of the dual-curve
    floating-leg algebra, shared by the PDE payoff layer and the MC
    swaption recipes.
  - `require_single_curve(...)` — the explicit refusal for payoffs that
    cannot project yet.
  `pde/rates_payoffs.py` gained an optional `basis=` on `swap_tail_values`,
  `caplet_fixing_values` and `caplet_values_by_level`; every HW/G2PP recipe
  in `pde/{hull_white,g2pp}.py` and `mc/recipes.py` was wired.

### 12.4 Modules touched

`valax/pricing/multicurve.py` (new), `valax/core/pricing.py` (bridge),
`valax/pricing/pde/rates_payoffs.py`,
`valax/pricing/pde/{hull_white,g2pp}.py`, `valax/pricing/mc/recipes.py`,
plus docstring contracts on `valax/pricing/{mc,pde}/dispatch.py`. Tests:
`tests/test_pricing/test_multicurve.py`, `tests/test_pde/test_multicurve.py`,
`tests/test_mc/test_multicurve.py`.

### 12.5 Definition of done (per `AGENTS.md`)

✅ theory page `docs/theory/multi-curve-short-rate.md` (nav-wired);
✅ solver guides `guide/pde.md` §5.7 and `guide/monte-carlo.md` §2 gained
the multi-curve recipe; ✅ roadmap Tier 2.1 + the Tier 5.4 status note;
✅ `docs/api/pricing.md` autodoc for the new public symbols plus a
multi-curve note on the short-rate engine coverage matrix; coverage matrix
(`applications/instrument-pricing.md`) unchanged — no instrument or method
column moved; ✅ `pytest` and `mkdocs build --strict` green.

### 12.6 Acceptance criteria — results

1. **Single-curve non-regression (hard gate) ✅.** `basis=None` and
   `discount_curve=None` are *trace-time* branches, so the single-curve
   code path is literally the pre-existing one. Asserted with exact
   equality (not a tolerance) for the HW/G2PP PDE swaption, the HW cap, and
   the HW/G2PP MC FRN; the ~600 existing rates tests are unchanged.
2. **Discounting / projection correctness ✅.** Payer-minus-receiver
   swaption parity *is* the forward swap value and is model-independent, so
   it isolates the discount/projection split exactly; it matches the
   textbook dual-curve NPV computed straight off the two curves to
   `rel ≈ 1e-4` under both the HW and G2++ PDEs. Independently, the HW and
   G2++ MC floating-rate note — a linear product — reproduces the analytic
   `floating_rate_bond_price(bond, ois, proj)` within sampling error, and
   its *basis increment* (dual minus single, common random numbers) matches
   the analytic increment to `rel ≈ 2e-3`. An option-free bond repriced on
   a distinct discount curve matches `fixed_rate_bond_price` on that curve,
   which pins down the model re-fit.
3. **Optionality ✅, with a correction to the criterion as originally
   written.** The payer swaption rises monotonically in the basis and
   collapses onto the single-curve price at zero basis — but it is **not**
   bracketed by the two single-curve prices. With the projection curve above
   OIS, the dual-curve trade keeps the *higher* OIS discount factors and
   gains the *higher* projected forwards, so both effects push the same way
   and the dual-curve price exceeds both single-curve legs. The test asserts
   that instead.

### 12.7 Risks (retired)

The change touched the MC/PDE recipe internals and the dispatch contract;
the single-curve non-regression gate was the guard and held. Every new
curve argument is optional and defaults to `None`, so existing callers are
untouched.

### 12.8 Known limits

- **No basis volatility.** The basis is frozen, so a basis swaption or a
  tenor-spread option is worth zero under this overlay. A stochastic-basis
  multi-curve HJM extension is tracked on the roadmap (Tier 2.1) and in
  `docs/research-ideas.md`.
- **CMS-spread recipes (G2++ PDE and MC) rebuild their par swap rates from
  the model's own bond strip**, so they accept `discount_curve` but raise on
  a distinct `forward_curve`.
- **Cheyette and LMM recipes are untouched** and remain single-curve.
- **Day-count caveat (pre-existing, surfaced by the tests).** The short-rate
  recipes build their *time axis* from the instrument's `day_count`, while a
  `DiscountCurve` uses its own. For an `act_360` trade on an `act_365` curve
  the two differ by ~1.4 %, which is why the curve-level oracles in
  `tests/test_pde/test_multicurve.py` use `act_365` instruments. Orthogonal
  to multi-curve, but worth a separate look.

---

## 13. Follow-up plan — end-to-end examples

**Status:** 📋 **partially superseded.** The *prose* half landed as the docs
tutorial `docs/guide/tutorial-unified-pricing.md`, which walks the contract
end to end with runnable snippets. What remains is the **executable** half
below: `examples/10_unified_pricing.py` and the local `runpy` smoke guard.
Tracked in §0.3.

### 13.1 Motivation

The migration was purely additive, so all 23 files in `examples/` still
run — but every one uses the **pre-ontology low-level APIs**
(`black_scholes_price(call, spot, vol, rate, dividend)`,
`swap_price(swap, curve)`, `greeks(black_scholes_price, …)`;
`09_pca_rates_pnl.py` uses the legacy `MarketData`). Nothing showcases the
unified `price(instrument, model, market)` front door.

*Correction to an earlier draft of this section:* the examples are **not**
unguarded. `.github/workflows/examples.yml` exit-code-runs every
`examples/*.py` and `examples/comparisons/*.py` on each push and PR. The
real gaps are narrower: there is no **local** `pytest` guard (so `pytest`
alone will not catch a broken example before CI does), and the CI glob
misses `examples/pitfalls/`.

### 13.2 Deliverables

- **New `examples/10_unified_pricing.py`** — the ontology front door,
  end to end:
  - build a `MarketState` (single-curve + spot + vol) and
    `price(option, BlackScholesModel, market)` with `method="auto"` →
    `"mc"` → `"pde"` (one call, three engines);
  - unified sensitivities — `market_greeks(...)`, `spot_delta`, and
    `curve_pillar_deltas` → `bucketed_curve_deltas` onto FRTB tenors;
  - a multi-curve swap priced via `discount_id` / `forward_id` selectors
    against a `CurveGraph` `MarketState`, and — since P5.1 — the same
    selectors carrying through to a Hull-White MC / PDE trade (§12);
  - `HullWhiteModel.from_market(...)` pricing a bond, then
    `with_market_curve(shocked)` to reprice under a shocked curve.
- **New smoke-test guard `tests/test_examples/test_examples_run.py`** —
  `runpy`-execute each example; mark the MC/PDE-heavy ones `slow`. Brings
  the guard local (CI already covers it) and lets a migration session
  iterate without a push. Widen the `examples.yml` glob to
  `examples/pitfalls/` at the same time.

### 13.3 What *not* to do

Do **not** rewrite the existing low-level examples — they teach the
low-level pricers, still work, and rewriting 23 files with no prior test
guard is churn. Migrating `09_pca_rates_pnl.py` to `MarketState` is
optional dogfooding, not required (`MarketData` remains a supported view).

### 13.4 Definition of done

New example runs clean (`python examples/10_unified_pricing.py`); the
smoke test passes; if any docs page lists the examples, add the new one;
`mkdocs build --strict` green.

---

## 14. Predecessor note — what it got right, what is left

This document's ancestor was a root-level working note,
`PRICING_MARKET_BOUNDARY.md` ("Pricing ↔ Market Boundary — Design Plan"),
written before any of P1–P5.1 landed. Its own §9.6 said to *"migrate this
doc's durable parts into `docs/architecture/`"*; this section is that
migration, and the note has been deleted.

It is worth recording rather than silently dropping, because the note
called the architecture correctly a long way out: the four-way
`(instrument, model, market, config)` separation, the two-tier
facade-over-kernels rule, curve selection by **id** rather than by a
singular `forward_curve` argument, and the insistence that the
explicit-argument kernels survive untouched — all of that is what shipped.

### 14.1 Supersession map

| Note § | Proposal | Outcome |
|---|---|---|
| §2 | `price(instrument, model, market, config)` | ✅ shipped as §2.3 / `valax/core/pricing.py`, with `method` / `config` / `key` as named keyword arguments rather than one positional `config` |
| §3 | Two-tier facade over unchanged Tier-1 kernels | ✅ shipped — `valax/core/adapters.py` plus the MC/PDE registry *bridge*; no pricer was rewritten, honouring the note's §10 non-goal exactly |
| §4 | A `Market` pytree (graph + surfaces + fixings + FX) | ✅ shipped as `valax.core.MarketState`, a superset (adds credit and inflation curves) |
| §5 | Instrument index descriptors | 🟡 **partial** — `discount_id` / `forward_id` on rates + bonds (P3). The per-leg `SwapLeg` / `legs` tuple was **not** built; see §14.2 |
| §6 | Model names its curve, does not own it | 🟡 **partial** — P5 added static `curve_id` + `from_market` / `with_market_curve`; `initial_curve` is still embedded |
| §7.1 | Curve authority | ⚠️ **shipped differently** — see §14.3 |
| §7.2 | Static/dynamic pytree hygiene audit | ❌ never run as an explicit pass; see §14.2 |
| §7.3 | Differentiability granularity via `grad` through `market` | ✅ P4 — `market_gradient` / `model_gradient` |
| §7.4 | Typed accessors, no god object | ✅ `MarketState.curve()` / `.surface()` / `.spot()` / `.dividend()` and `core/resolve.py` |
| §7.5 | One `Market` type, not one per asset class | ✅ resolved — one `MarketState`; `MarketData` survives one release as a two-way view (Q4) |
| §7.6 | Where fixings live | 🟡 `MarketState.fixings` exists and `resolve` carries it, but **no pricer reads it**; see §14.2 |
| §8, §9.4 | Risk engine targets the facade; `wrap_*` adapters deleted | ❌ **not done** — see §14.2 |
| §9.1–3, §9.5 | Rollout steps | ✅ delivered as P2 → P3 → P5.1 |
| §11 | Four "next steps" | superseded: §7.1 and §7.5 are settled, the `Market` prototype shipped, the facade is live across every product family, and G2++ landed |

### 14.2 Still open

- [ ] **Retire the legacy flat risk path.** The *new* path is done: a
  keyed scenario on a `MarketState`, generated from a per-curve covariance
  layout, repriced through `price(...)` (`valax/risk/state_var.py`) and
  measured by a keyed ladder (`valax/risk/state_ladders.py`) — so "every
  Greek is one `jax.grad` through one function" now holds along the whole
  chain for a group, with a projection curve and a basis each carrying
  their own P&L and VaR. What remains is **removal**:
  `wrap_equity_pricing_fn` (already `DeprecationWarning`),
  `reprice_under_scenario` / `portfolio_pnl` / `parametric_var` /
  `pnl_attribution`, `compute_ladder`, and `MarketData` itself, plus
  portfolio aggregation above the group primitives. See
  [Roadmap → Multi-Curve Framework §1](docs/roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)
  and §0.1.
- [x] **Per-leg index descriptors for genuinely multi-index products.**
  Shipped as `valax.instruments.legs`, built as **one** change with the
  P1.2 cashflow/leg engine
  ([Roadmap Tier 1.4](docs/roadmap.md#14-trade-representation-and-cashflow-engine))
  exactly as this item required. `FixedLeg` / `FloatingLeg` each carry
  `discount_id` / `forward_id` / `index_id`, and the trade is
  `MultiLegSwap(legs=tuple[...])` exposing `curves_touched` /
  `indices_touched` as the de-duplicated union across legs. A
  single-currency tenor basis swap now reads its three curves, gated by
  `tests/test_pricing/test_legs.py::TestPerLegIndexDescriptors`
  (bumping one projection curve must move exactly one leg).
  **Scope:** this lifts the ceiling for trades expressed as legs. The
  existing single-trade pytrees in `valax/instruments/{rates,bonds}.py`
  keep the singular pair and are untouched, so §0.1's selector items for
  FX / credit / inflation are unaffected.
- [ ] **Static/dynamic pytree hygiene audit.** The note called this
  "non-negotiable for JIT": confirm no string id leaks into traced code
  across `MarketState`, `CurveGraph`, and the instrument selectors. The
  convention has been followed by hand throughout, but the audit itself
  was never run, and nothing enforces it. The *convention* is now at
  least written down — "static is for shapes, dynamic is for values",
  with the four date representations measured against each other and the
  three sanctioned escape hatches, in
  [JAX Patterns § 2.5](docs/architecture/jax-patterns.md) — which is the
  specification such an audit would check against. Writing it down is
  not running it; this item stays open.
- [ ] **Full curve de-embedding.** §6's endpoint — the model carries only
  `curve_id` and the exact-fit shift is computed at price time — remains
  unbuilt; see §14.3 for why it is now less urgent.
- [ ] **Pricer-side fixings.** Half closed by the leg engine
  (`valax.pricing.analytic.legs`), which consumes a `FixingHistory`
  pricer-side under each leg's `index_id` and derives the straddling
  period's realised factor itself rather than taking a caller-supplied
  `accrued_compounding`. What remains is the *other* representation:
  `FloatingRateBond.fixing_rates` is still an unrelated NaN-sentinel
  array on the instrument pytree, and `floating_rate_bond_price` still
  reads it. Retiring that in favour of the history is a deprecation
  decision, not a missing capability.

### 14.3 Where the shipped design diverged

The note's §7.1 recommended making the **market authoritative**: strip
`initial_curve` off the short-rate models entirely and recompute the
exact-fit `α` / `φ` at price time from `market.graph[discount_id]`. Its
stated upside was that a scenario shock to OIS would auto-refit the model.

What shipped is the **hybrid** of §3.1: the model keeps `initial_curve` as
its *default* discount curve and records a static `curve_id` (P5), and
P5.1 lets a trade override it per-price via
`valax.pricing.multicurve.with_discount_curve`, which re-fits the model to
the resolved discount curve. That delivers the auto-refit property the
note wanted, at the point where it matters, **without** a breaking change
to every model constructor, pricer, and test that reads
`model.initial_curve` — and without paying the refit on every price when
nothing has moved.

So §7.1's *goal* is met and its *mechanism* was not adopted. Full
de-embedding stays on the list (§14.2) as a tidy-up, not as a blocker.
