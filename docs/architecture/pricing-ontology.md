# Pricing Ontology & Unified Signature — Migration Report (Design)

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
  (see [MC-Curves-2 Handoff](mc-curves-2.md)).
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

- ✅ `docs/architecture/pricing-ontology.md` — this report (P1).
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
  (`MarketState`, `price`, `market_greeks`).
- `mkdocs build --strict` must stay green (this page is wired into nav under
  *Architecture & Validation*).

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

**Status:** 📋 planned, not started. *(Owner note: revisit in its own
session.)*

### 13.1 Motivation

The migration was purely additive, so all 23 files in `examples/` still
run — but every one uses the **pre-ontology low-level APIs**
(`black_scholes_price(call, spot, vol, rate, dividend)`,
`swap_price(swap, curve)`, `greeks(black_scholes_price, …)`;
`09_pca_rates_pnl.py` uses the legacy `MarketData`). Nothing showcases the
unified `price(instrument, model, market)` front door, and **no test
executes the examples**, so they can rot silently.

### 13.2 Deliverables

- **New `examples/10_unified_pricing.py`** — the ontology front door,
  end to end:
  - build a `MarketState` (single-curve + spot + vol) and
    `price(option, BlackScholesModel, market)` with `method="auto"` →
    `"mc"` → `"pde"` (one call, three engines);
  - unified sensitivities — `market_greeks(...)`, `spot_delta`, and
    `curve_pillar_deltas` → `bucketed_curve_deltas` onto FRTB tenors;
  - a multi-curve swap priced via `discount_id` / `forward_id` selectors
    against a `CurveGraph` `MarketState` (analytic — note the MC/PDE
    multi-curve caveat from §12);
  - `HullWhiteModel.from_market(...)` pricing a bond, then
    `with_market_curve(shocked)` to reprice under a shocked curve.
- **New smoke-test guard `tests/test_examples/test_examples_run.py`** —
  `runpy`-execute each example; mark the MC/PDE-heavy ones `slow`. Closes
  the "examples can rot" gap and makes any future example migration safe.

### 13.3 What *not* to do

Do **not** rewrite the existing low-level examples — they teach the
low-level pricers, still work, and rewriting 23 files with no prior test
guard is churn. Migrating `09_pca_rates_pnl.py` to `MarketState` is
optional dogfooding, not required (`MarketData` remains a supported view).

### 13.4 Definition of done

New example runs clean (`python examples/10_unified_pricing.py`); the
smoke test passes; if any docs page lists the examples, add the new one;
`mkdocs build --strict` green.
