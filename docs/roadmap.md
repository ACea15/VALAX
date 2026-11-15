# Roadmap: Path to Production

VALAX today covers equity vanillas and exotics, a broad rates suite (vanilla and exotic), FX and inflation vanillas, four pricing methods (analytic, Monte Carlo, PDE, lattice), and around ten stochastic models — all with autodiff Greeks and JIT compilation. It is still a fraction of the **feature breadth** of production quant libraries at major banks (JPM Athena, Goldman SecDB, etc.), which have been built over 20-30 years by hundreds of quants.

The good news: the architecture (pure-functional JAX, autodiff-first, pytree data model) is arguably **more modern** than most bank stacks. The gap is breadth, not depth.

This roadmap organizes every missing piece into prioritized tiers. Each tier unlocks the next.

---

## Current State

| Area | What We Have |
|------|-------------|
| **Instruments** | **Priced:** `EuropeanOption`, `AmericanOption`, `EquityBarrierOption`, `AsianOption`, `LookbackOption`, `VarianceSwap`, `DigitalOption`, `SpreadOption`, `WorstOfBasketOption`; `ZeroCouponBond`, `FixedRateBond`, `FloatingRateBond`, `CallableBond`, `PuttableBond`; `Caplet`, `Cap`, `InterestRateSwap`, `Swaption`, `BermudanSwaption`, `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `CMSSpreadSwap`, `CMSCapFloor`, `RangeAccrual`; `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`, `InflationCapFloor`; `FXForward`, `FXVanillaOption`; `FixedLeg`, `FloatingLeg`, `MultiLegSwap` — the leg representation, carrying compounding conventions, an amortising notional and a curve descriptor **per leg** ([Tier 1.4](#14-trade-representation-and-cashflow-engine)). **Pytree defined, pricer on the roadmap:** `ConvertibleBond`, `CompoundOption`, `ChooserOption`, `Autocallable`, `Cliquet`, `FXBarrierOption`, `QuantoOption`, `TARF`, `FXSwap`, `CDS`, `CDOTranche` |
| **Models** | Black-Scholes, **Heston (MC + Fang-Oosterlee COS semi-analytic)**, SABR (analytic + MC), LMM (MC with PCA factors), **Hull-White (analytic ZCB, trinomial tree, MC, PDE)**, **G2++ (two-factor Gaussian; MC + PDE)**, **Cheyette (quasi-Gaussian 1F/2F skew; MC + PDE)** — all three short-rate models with constant or piecewise-constant σ(t), **Local Volatility (Dupire from SVI/SABR/Grid surfaces; MC + PDE)**, **SLV (Heston × leverage; MC)**, **Multi-asset GBM (correlated)** |
| **Pricing** | Black-Scholes, Black-76, Bachelier, SABR analytic; **Heston Fang-Oosterlee COS**; **Dupire local vol (Gatheral IV-space)**; Monte Carlo (GBM, Heston, SABR, Local Vol, **SLV**, **multi-asset GBM**, LMM, **Hull-White**, **G2++**, **Cheyette**); Crank-Nicolson 1-D PDE; **2-D ADI PDE (Heston)**; **short-rate PDE (Hull-White, G2++, Cheyette)**; CRR binomial tree; Hull-White trinomial tree |
| **Greeks** | 1st order (delta, vega, rho) and 2nd order (gamma, vanna, volga) via autodiff; key-rate durations via curve pytree differentiation |
| **Curves** | `DiscountCurve` with log-linear interpolation, flat extrapolation (the only interpolation mode shipped). Keyed multi-curve container `CurveGraph` / `CurveSpec`; joint multi-curve Newton bootstrap `bootstrap_curve_graph` over 11 quote types, returning a `CalibrationResult` (traceable convergence, repricing residuals, `condition_number()`; fitted quotes via `fitted_curve_quotes`); market-quote sensitivity `quote_jacobian` via implicit differentiation; `SurvivalCurve`, `InflationCurve`, `FixingHistory`, pluggable futures-convexity hooks |
| **Dates** | Act/360, Act/365, **Act/Act (ISDA, exact)**, 30/360, **Bus/252**; ordinal-based date arithmetic; schedule generation with day-of-month preservation, end-of-month rolls and short/long front/back **stubs**, returning `(accrual_start, accrual_end, payment_dates)`; **accrued interest and clean/dirty prices**; **business-day calendars and adjustment** for TARGET, SOFR/SIFMA, NYSE, London, Tokyo and Brazil, with union/intersection for cross-currency ([Tier 1.3](#13-business-calendars-and-holiday-schedules)). Adjustment is opt-in — the default convention remains `"unadjusted"` |
| **Calibration** | SABR, Heston, **Hull-White, G2++, Cheyette (1F/2F)**, and **SLV leverage** calibration with LM / BFGS / Optax solvers; parameter constraint transforms. Curve bootstrap (sequential, simultaneous, joint `CurveGraph`) with implicit-diff quote Jacobian; survival curves from CDS spreads via the closed-form credit-triangle approximation (*not* a repricing bootstrap — see [3.4](#34-credit-derivatives)); SABR / SVI vol surfaces, **swaption cube, optionlet surface** |
| **Portfolio** | `batch_price()` and `batch_greeks()` via vmap |
| **Market Data** | `MarketState` — the market object behind `price()`: a `CurveGraph` (OIS discount + projection curves, basis, XCCY), surfaces, spots and dividends, with `MarketState.flat(...)` for a constant single-curve market; `KeyedScenario` + `valax.risk.shock` for identifier-keyed risk-factor shocks. Legacy (scheduled for removal, see [Remaining work](#remaining-work)): the flat `MarketData` container and the positional `MarketScenario` / `ScenarioSet` + `apply_scenario`, reachable via `MarketState.to_market_data()`. Full synthetic-data subpackage (`valax.market.synthetic`) covering Stages 1–6 of the workflow — `SyntheticMarketConfig`, `SeedRegistry`, NSS / flat curve samplers, random PSD correlation matrices, ground-truth model parameters, noisy observation layer, option / swap portfolio samplers, correlated-GBM market tapes, scenario sets, and deliberate arbitrage injectors |
| **Testing infrastructure** | Versioned golden-dataset harness (`tests/golden/`, `assert_matches_golden`, `REGEN_GOLDEN=1` regeneration); shared `SeedRegistry` fixture; pytest markers `arbitrage` / `golden` / `detects`; non-tautological closed-loop tests (pricer × implied-vol roundtrip, analytic vs MC within 3·stderr, autodiff vs finite-difference Greeks, NSS curve self-consistency, SABR calibration at the observation-noise floor, bootstrap roundtrip on off-pillar dates); reserved arbitrage exception types in `valax.core.diagnostics`; `@pytest.mark.xfail` backlog of missing safety checks |
| **Risk** | Curve shocks (parallel, steepener, butterfly, key-rate), parametric/historical/stress scenarios, VaR, ES, bucketed sensitivities with waterfall P&L — keyed on a `MarketState` (`keyed_*` generators, `group_*` repricing through `price()`, `keyed_sensitivities`, `keyed_waterfall_pnl`, `keyed_parametric_var` / `keyed_component_var`; single-instrument risk via `valax.greeks.market_risk`), with the flat `MarketData` counterparts kept as legacy ([§ 1](#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)) |
| **MC infrastructure** | Unified `mc_price_dispatch(instrument, model, ...)` with `(instrument, model)` → recipe registry. Recipes span equity (GBM / Heston / local vol / SLV), multi-asset equity (correlated GBM), and rates (LMM plus the Hull-White / G2++ / Cheyette short-rate models). `register()` decorator for contributor extensions. Coverage per instrument: the *MC* column of the [coverage matrix](applications/instrument-pricing.md#coverage-matrix); engine mechanics: [Monte Carlo guide](guide/monte-carlo.md#2-recipes-by-model-family). |
| **Multi-asset MC** | `MultiAssetGBMModel` with per-asset vols/dividends + correlation matrix. `generate_correlated_gbm_paths` (exact log-Euler via Cholesky). Spread-option MC (validates Margrabe + Kirk analytical) and worst-of basket MC with correlation Greeks via `jax.grad`. |

### Instrument Coverage Matrix

<!-- coverage-map -->

The instrument × solver coverage — what is priced, by which solver family
(analytic, Monte Carlo, PDE, lattice), under which models, and what is
planned — is stated in one place: the
[coverage matrix](applications/instrument-pricing.md#coverage-matrix),
checked against the code by `tests/test_docs/test_price_routes.py`. Every
planned (🟡) cell there links to the open `- [ ]` item in the tier checklists
below that tracks it, together with its blocker; `python
scripts/coverage_report.py` lists them by solver and asset class. The other
coverage maps (calibration, risk factors, capabilities) are indexed on
[Coverage Maps](coverage.md).

---

## QuantLib Validation Pyramid

A three-stage validation campaign that converts the existing
fixed-scenario `tests/test_quantlib_comparison/` files into a
parametric sweep, then adds calibration-agreement and chain-validation
stages. Drives every assertion through the synthetic-market generator
so each test runs across ~10–20 seeds rather than one hardcoded case.

**Detailed plan and session log:**
[Architecture → QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md).

Stages:

1. **Pricer parametric sweep** — wrap the 7 existing comparison files
   in `@pytest.mark.parametrize("seed", range(20))`. Catches convention
   drift in pricers across the parameter space.
2. **Calibration agreement** — curve bootstrap, SABR smile fit, and
   QL-calibrates-VALAX-reprices Heston. Comparison is in observable
   space (DFs, smiles, prices), never parameter space.
   - [ ] **Cover *every* calibrator, and log the results.** Today only
     SABR has a QL comparison (`test_sabr_calibration_ql.py`). The
     short-rate calibrators (`calibrate_hull_white`, `calibrate_g2pp`,
     `calibrate_cheyette`) have **no** QL cross-check in either the
     single- or the dual-curve mode, so their agreement with market
     convention rests on internal oracles only (PDE agreement and
     payer/receiver parity — strong on *self*-consistency, silent on
     convention). Each run should be **recorded** (fitted parameters,
     observable-space residuals, seed, QL version) so drift is visible
     over time rather than re-discovered.
   - [ ] **Two known items to settle with QL specifically.** (a) G2++
     `∂price/∂σ_x` is *negative* for a payer swaption at
     `ρ = −0.7` with well-separated mean reversions — autodiff and
     finite differences agree, so it is the formula's behaviour, not an
     AD artefact, and it is plausibly the genuine effect of the negative
     cross-variance term; QL is the independent arbiter. (b) VALAX's
     dual-curve overlay freezes the tenor basis, which need not match
     QL's convention — a mismatch there would be convention, not error,
     and should be characterised before it is treated as a bug.
3. **Chain validation** — calibrated surface flows through VALAX and
   QL pricers; tests the quote→surface→exotic pipeline that desks
   actually run.

**Estimated effort:** ~3 days end-to-end. **Expected output:**
~1000 sampled markets agreeing with QuantLib at every stage of the
pricing chain.

## Multi-Curve Framework — What Ships, What's Left

The canonical status page for the multi-curve workstream (formerly the
MC-Curves-1/2/2b/3 session notes). The design lives in
[`production.md` §11](architecture/production.md#11-multi-curve-framework);
the narrative rationale in [Why the Multi-Curve Framework](guide/why-multicurve.md);
the short-rate overlay in
[Multi-Curve Short Rates](theory/multi-curve-short-rate.md).

**The boundary is sharp.** The `CurveGraph` flows cleanly through
**build → `MarketState` → `resolve` → analytic / MC / PDE rates pricers**, and
now through **short-rate calibration**. It stops at **surfaces and
portfolio**; **risk** is most of the way across — scenarios, scenario
generators, the repricing loop (`price(...)` on a `MarketState`), sensitivities,
waterfall and delta-normal VaR are all keyed, but the FRTB PLA vector pair,
`pnl_attribution` and the flat `parametric_var` still consume the legacy
flat `MarketData` view, and portfolio aggregation is not built. **Surfaces** are
partway across too: the rates vol objects (swaption cube, optionlet surface)
are curve-aware and carry quote-space vega, while the equity surfaces remain
curve-free:

```
quotes ─▶ bootstrap_curve_graph ─▶ MarketState ─▶ resolve ─▶ analytic          ✅
                   │                                      ├─ MC/PDE (HW, G2++) ✅
                   │                                      ├─ HW lattice        ❌
            quote_jacobian ✅                             └─ Cheyette / LMM    ❌
            curve_adjoint  ✅
                   │
                   │ λ_c composed: curve_quote_deltas ✅ (IFT, FD-gated)
                   │ λ_v composed: vol_quote_vegas    ✅ (IFT on the fit's FOC)
                   │ λ_m           per underlying ✅; per trade (HW basket) ✅ via chain_adjoint
                   │ whole chain:  calibrate_market → chain_adjoint ✅ (§5.6)
                   │ p̄  reported:  fixed_input_adjoints ✅ (no IFT step needed)
                   ▼
            greeks 🟡 ── risk 🟡 ── calibration 🟡 ── surfaces 🟡 ── portfolio ❌
                        (keyed scenarios ✅,   (HW/G2++ ✅,   (rates cubes +
                         repricing loop ✅,     Cheyette       optionlets ✅,
                         generators, Greeks,    refuses)       equity surfaces
                         VaR ✅; PLA flat)                     still curve-free)
```

So a book can be **priced** multi-curve today and its short-rate models
**calibrated** multi-curve; it can be **shocked** per curve
(`valax.risk.shock` on a `MarketState`) and **repriced** per curve
(`valax.risk.state_var` through `price(...)`), so a basis widening now
reaches a P&L number. Keyed generators (`keyed_parametric_scenarios`, …)
draw those moves from a per-curve `KeyedFactorLayout`, and `keyed_sensitivities` /
`keyed_waterfall_pnl` / `keyed_parametric_var` measure them per curve. What
is still driven by unkeyed `MarketScenario` vectors on the legacy
`MarketData` path is the rest of the measurement layer — `compute_flat_sensitivities`,
the RTPL/HPL vectors, `pnl_attribution` and the flat `parametric_var` (see
[Remaining work § 1](#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)).
On the **surfaces** side the rates objects are across: a swaption cube or
optionlet surface is calibrated off the curves and reports vega in *quote*
space, so a curve move reaches the volatility layer. The equity surfaces and
**portfolio** are untouched.

### Shipped

- [x] **Joint curve build** — `bootstrap_curve_graph` + `CurveSpec`
  (`valax/curves/bootstrap_graph.py`, `graph.py`), returning the same
  `CalibrationResult` root level as `curve_graph_node` (see
  [§5.6](#unified-calibration-pricing-and-risk));
  11 quote types in `valax/curves/instruments.py`; tenor-basis and
  EUR/USD four-curve builds verified end-to-end
- [x] **Quote-sensitivity Jacobian** — `quote_jacobian` via
  `optimistix.ImplicitAdjoint`, covering every registered quote scalar
  (`.rate`, `.spread`, `.futures_rate`, `.quoted_forward`, `.far_rate`,
  `.jump_size`)
- [x] **Quote-space (hedge-space) adjoint risk** — `valax.greeks.adjoint`:
  `curve_adjoint` → `curve_quote_deltas` gives `∂price/∂quote` through the
  implicit function theorem, never forming the quote Jacobian and never
  taping the Newton solve. `curve_residual_fn` exposes the *same*
  `_graph_residual` the solver drives to zero, so the differentiated system
  and the solved system cannot drift apart. `CurveAdjoint.factorisation`
  makes per-trade risk a back-substitution. Validated against full-rebuild
  central finite differences for every quote at two bump sizes, plus the
  parallel-shift identity and the quotes-fixed vs nodes-fixed distinction
  (`tests/test_greeks/test_quote_adjoint.py`, `pytest -m audit`)
- [x] **Conditioning reported at every calibration level** —
  `valax.greeks.adjoint.level_condition_number` reports `κ(dR/dθ)` for any
  `CalibrationLevel` (curve, smile or model; square or least-squares, flat or
  pytree `θ`), and `condition_alert` tests it against the single shared
  `ILL_CONDITIONED_THRESHOLD` (`1e10`), returning a message rather than
  warning so the caller picks the channel. Closes the gap where only Layer 1a
  had a conditioning report and no threshold existed anywhere in code.
  Measured: curve build ≈ 1.1, SABR slice ≈ 2.2e4 (≈ 5.6e6 with β free),
  Hull-White (a, σ) ≈ 5.6e3 vs ≈ 1 for σ alone — so the diagnostic quantifies
  the standing advice to fix β and to fix mean reversion. Gated by
  `tests/test_greeks/test_level_conditioning.py` (T-13, `pytest -m audit`)
- [x] **Fixed-input (`p`) adjoints and snapshot provenance** —
  `valax.greeks.fixed_inputs` (`FixedInput`, `FixedInputs`,
  `fixed_input_adjoints`, `FixedInputAdjoint`) reports sensitivities to the
  quasi-market inputs the calibration holds still — mean reversion, SABR β
  and shift — separately from `q̄`, as spec A-8/M-5 require. A `FixedInput`
  stores *where* the value lives rather than a copy of it, so the reported
  number belongs by construction to the leaf that was priced. Both readings
  are supported: direct (`∂V/∂p`, objects frozen) and total
  (recalibrate-then-price); they differ by **6.6×** on mean reversion, so
  reporting the wrong one overstates the exposure several times over.
  `valax.market.SnapshotProvenance` / `QuoteProvenance` carry per-quote type,
  units, source and timestamp under one snapshot id (Q-1), derived from the
  instruments in solver order so `label(q̄)` cannot mislabel; kept alongside
  `MarketState` rather than inside it, since even all-static metadata would
  enter JIT cache keys. Gated by `tests/test_greeks/test_fixed_inputs.py`
  (T-14) and `tests/test_market/test_provenance.py` (Q-1/Q-5/Q-6)
- [x] **Single-curve limit, smoothness and basket-repricing gated** —
  `tests/test_calibration/test_audit_conformance.py` (T-4, T-8, T-9).
  T-4 passes `forward_curve=curve` *explicitly* so the dual-curve path runs
  and must land on the single-curve answer — exact to 0.0 at the pricers,
  verified at all three layers (pricer, bootstrap, model fit), with a 25 bp
  basis gated as the non-degenerate complement. T-8 phrases "no steps" as a
  refinement rate (`max|Δ²price|` must fall 4×, `max|Δdelta|` 2×; observed
  3.99 and 2.00), which a discontinuity cannot satisfy at any tolerance.
  T-9 reprices the calibrated Hull-White basket with the production PDE and
  shows by grid refinement that the 1.5e-4 residual is discretisation error,
  not misfit
- [x] **Smile-dynamics delta gated** —
  `tests/test_surfaces/test_smile_dynamics.py::TestT12SmileDynamicsDelta`
  (T-12). At the fitted forward all three conventions return the same vol and
  therefore the same price, so the convention is observable *only* in delta:
  the gate checks `dσ/dF` against partials of `sabr_implied_vol` taken
  directly (0 exactly for `sticky_strike`, `−∂σ/∂K` for `sticky_moneyness`,
  `∂σ/∂F` for `sabr_backbone`), and pins that a flat smile would collapse the
  entire distinction. Measured spread: 14% of delta across the two extremes
  on an identically-priced OTM payer
- [x] **Rates option parities and the Bachelier ATM identity gated** —
  `tests/test_surfaces/test_audit_parities.py` (T-6, T-7). The naive form of
  T-6 is vacuous, because VALAX derives the floorlet and the receiver *by*
  parity, so each leg is instead checked against an independently written
  closed form and the forward, annuity, accrual and discount factor are
  rebuilt by hand — single- and dual-curve, since a `P_d`/`P_x` swap is
  invisible when the curves are the same object
- [x] **Day-count and schedule reference data gated** —
  `tests/test_dates/test_audit_conventions.py` (T-10) pins τ and schedule
  dates against references built from the **standard library only**, so a
  self-consistent-but-wrong ordinal algebra cannot pass. `act_360`,
  `act_365` and `30_360` match bit-for-bit; `act_act` to 1e-14, the
  difference being cumulative-subtraction vs ISDA's per-year summation
- [x] **Curve-id alphabet frozen** — `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]`,
  enforced by `valax.curves.graph._CURVE_ID_RE`; `_default_` grandfathered
- [x] **Legacy paths migrated** — `bootstrap_simultaneous` is a thin wrapper
  over the joint solver; the deprecated `bootstrap_multi_curve` has been
  removed; `MultiCurveSet` remains as a view over `CurveGraph`
  (`from_graph` / `to_graph`)
- [x] **Market container** — `valax.core.MarketState` holds the graph;
  `valax.core.resolve` binds `discount_id` / `forward_id` per trade
- [x] **Instrument selectors (rates + bonds)** — `discount_id` /
  `forward_id` on 11 classes in `instruments/rates.py` and 6 in
  `instruments/bonds.py`
- [x] **Analytic rates pricers dual-curve** — `analytic/{floating,swaptions,caplets,rates_exotics,rates_smile,graph_pricing}.py`
- [x] **MC and PDE dual-curve for Hull-White and G2++** (ontology P5.1) —
  `valax/pricing/multicurve.py`; see
  [Tier 2.1](#21-short-rate-models)
- [x] **Graph shock primitives** — `bump_graph_curve`,
  `parallel_graph_shift`, `key_rate_graph_bump` in `valax/risk/shocks.py`
- [x] **Keyed scenarios + `MarketState` shock** — `KeyedScenario`
  (`valax/market/keyed_scenario.py`) addresses curves, surfaces, spots and
  dividends **by identifier**, and `valax.risk.shock`
  (`valax/risk/state_shocks.py`) applies one to a `MarketState`, routing
  curve moves through `bump_graph_curve` and vol moves through
  `shift_surface` / `ShiftedVolSurface` (`valax/surfaces/shifted.py`). A
  pure basis move is now one scenario; `keyed_from_market_scenario` lifts
  the legacy flat vectors, which stay supported for one deprecation cycle.
  See [Scenarios & Shocks § Keyed scenarios](risk-scenarios.md#keyed-scenarios-on-a-marketstate)
- [x] **The VaR path stops dropping the graph** —
  `reprice_under_scenario`, `portfolio_pnl`, `parametric_var` and
  `pnl_attribution` (`valax/risk/var.py`) thread `curve_graph` into the
  per-instrument `MarketData` instead of letting it default to `None`, and
  warn when a legacy scenario cannot address a named graph curve;
  `apply_scenario` keeps the `"_default_"` graph entry in step with
  `discount_curve`
- [x] **Dual-curve analytic short-rate swaptions and calibration** —
  `hw_swaption_price` / `g2pp_swaption_price` accept a projection curve, and
  `calibrate_hull_white` / `calibrate_g2pp` / `swaption_prices_from_vols`
  accept `forward_curve=`; see § 2 below

### Remaining work

#### 1. Risk engine: keyed end to end, bar portfolio aggregation

The whole point of multi-curve is that OIS and projection curves are
**distinct risk factors** and the basis is itself hedgeable
([Why Multi-Curve § What this means for risk](guide/why-multicurve.md#what-this-means-for-risk)).
That now holds along the whole chain for a *group*: a keyed scenario shocks
any curve on the graph by identifier, keyed generators draw a whole
distribution of such moves from a **per-curve covariance layout**, the
repricing loop goes through `price(...)`, and keyed sensitivities report each
curve's — and the basis's — own DV01, P&L term and VaR contribution. What
is still flat is the **legacy** `valax/risk/var.py` / `valax/risk/sensitivities.py`
pair (kept for one deprecation cycle) and the **portfolio aggregation**
layer above the group primitives.

- [x] **Per-curve scenario axis** — `KeyedScenario.curve_shocks`
  (`valax/market/keyed_scenario.py`) is `dict[curve_id, per-pillar bumps]`,
  applied by `valax.risk.shock`; OIS and projection curves are separately
  shockable on one `CurveGraph`
- [x] **Keyed *generators* and a per-curve covariance layout** —
  `KeyedFactorLayout` (`valax/risk/scenarios.py`) replaces the flat
  positional factor order with an ordered tuple of `FactorBlock`s, **one
  block per named market object**, derived from the market itself
  (`from_market_state`) and naming every coordinate with the hierarchical
  identifier of [Risk Factors § 1.2](risk-factors.md#12-factor-identifier-convention).
  `flatten` / `unflatten` are exact inverses (batched or not), so the same
  layout indexes a scenario, a covariance matrix and a sensitivity vector. On top of it:
  `block_covariance` (per-curve blocks + explicit cross-correlations,
  factored by `psd_factor` so a singular covariance samples instead of
  returning NaNs), `historical_block_covariance`, and the generators
  `keyed_parametric_scenarios` / `keyed_historical_scenarios` /
  `keyed_stress_scenario` / `keyed_curve_stress` (+ `keyed_steepener` /
  `keyed_flattener` / `keyed_butterfly`, sharing `curve_shape_bumps` with
  the flat presets) / `keyed_basis_shift`. The legacy flat generators stay
  for one deprecation cycle. Gate:
  `tests/test_risk/test_keyed_scenarios.py`. Recipe:
  [Scenarios & Shocks § Keyed generators](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout)
- [x] **Route graph shocks through the container-level shock** —
  `valax.risk.shock` moves any named graph curve via `bump_graph_curve`, and
  `apply_scenario` now keeps the graph's `"_default_"` entry in step with
  `discount_curve` instead of passing the whole graph through unshocked
- [x] **Stop dropping the graph in the VaR path** —
  `reprice_under_scenario`, `portfolio_pnl`, `parametric_var` and
  `pnl_attribution` (`valax/risk/var.py`) thread `curve_graph` through to
  the pricer; a graph curve a `MarketScenario` cannot address is reported
  as a `UserWarning`, never silently. Gate:
  `tests/test_risk/test_var.py::TestCurveGraphSurvivesVaRPath`
- [ ] **Node-level (surface-keyed) vega sensitivities** — the keyed vol shock is
  a per-surface **parallel shift** (`ShiftedVolSurface`), which is why
  `FlatSensitivities.delta_vol` / `vanna` / `volga` stay one scalar per
  asset. Node-level (bucketed) vega needs those fields to become
  identifier-keyed per-node arrays; see
  [Scenarios & Shocks § Vol shocks](risk-scenarios.md#vol-shocks-parallel-shift-not-node-shock)
- [ ] **Keyed shocks for credit / inflation / FX on `MarketState`** —
  `shock` covers curves, surfaces, spots and dividends;
  `credit_curves`, `inflation_curves` and `fx_spots` are not addressable
  yet (the `SurvivalCurve` primitives in `valax/risk/shocks.py` exist and
  are unwired, exactly as the graph bumps were)
- [x] **Route the repricing loop through `price(...)` and retire
  `wrap_equity_pricing_fn`** — `valax/risk/state_var.py` reprices a
  homogeneous group on a `MarketState` under a `KeyedScenario` through
  the unified front door: `group_pv` / `group_prices` / `group_reprice` /
  `group_scenario_pnl` / `group_market_gradient` / `group_model_gradient`.
  Each named curve of the graph carries its own cotangent, so a basis
  widening produces a P&L (measured +9,817 on a two-swap book for a 25 bp
  projection-curve move, where the unkeyed legacy vector gives −150 for
  the same input). `wrap_equity_pricing_fn` now emits a
  `DeprecationWarning` and goes with `MarketData`. Gates:
  `tests/test_risk/test_state_var.py` — bit-exact against the legacy path
  on a single-curve book, plus the common-random-numbers gate for MC
  groups. Recipe: [VaR & P&L § Repricing through `price()`](risk-var.md#repricing-through-price)
- [x] **Port the *measurement* layer: keyed sensitivities, waterfall and VaR** —
  `valax/risk/state_sensitivities.py` is the `MarketState` counterpart of
  `sensitivities.py`: `keyed_sensitivities` returns a `KeyedSensitivities` indexed by a
  `KeyedFactorLayout` (one `jax.grad` for every first-order sensitivity, one
  `jax.hessian` for the second-order block matrix, all through
  `price(...)`), with `dv01(curve_id)` / `key_rate_dv01` / `vega` /
  `basis_dv01` / `block_totals` / `to_dict` accessors. The sensitivities are
  taken w.r.t. the **shock magnitudes** rather than the market leaves, which
  deletes the hand-written DF→zero-rate chain rule (and its gamma
  correction) and gives one well-defined vega for *every* surface type.
  `keyed_waterfall_pnl` / `keyed_waterfall_report` attribute P&L with one
  term per named market object — a pure basis widening puts exactly zero on
  the discount term — and `keyed_parametric_var` / `keyed_component_var`
  give a delta-normal VaR whose Euler decomposition is additive across
  curves, so a projection curve and a basis each carry their own VaR
  (measured 5,922 OIS + 494 basis on a two-swap book). Gate:
  `tests/test_risk/test_state_sensitivities.py` (sensitivities vs finite differences of
  the same shock map; second-order explain to <1e-5 of actual; component VaR
  summing to total; delta-normal within 5% of a 20k-scenario simulated VaR).
  Recipe: [Bucketed Sensitivities § Keyed sensitivities](risk-sensitivities.md#keyed-sensitivities-on-a-marketstate).
  *(Renamed from `KeyedLadder` / `keyed_ladder` / `state_ladders.py`; see the
  alias-removal item below.)*
- [ ] **Port the rest of the measurement layer** — `compute_flat_sensitivities` still
  builds DF-space Hessians over flat `MarketData` vectors and
  `risk_theoretical_pnl_vector` consumes *those* sensitivities, so the FRTB PLA
  path (`valax/risk/pnl_vectors.py`) and `pnl_attribution` are still
  positional. The keyed sensitivities above are what they re-target onto; the
  remaining work is the RTPL/HPL vector pair and retiring the flat
  `parametric_var`
- [x] **Industry terminology: *sensitivities*, not *ladders*** — base-point
  bucketed Greeks are now `FlatSensitivities` / `compute_flat_sensitivities`
  (`valax/risk/sensitivities.py`), `KeyedSensitivities` /
  `keyed_sensitivities` with `key_rate_dv01` (`valax/risk/state_sensitivities.py`)
  and `BucketedSensitivities` / `bucket_sensitivities`, matching FRTB SA /
  ISDA SIMM usage; *ladder* is reserved for revaluation at shifted levels.
  The old names stay importable with a `DeprecationWarning`
  (`valax/risk/_deprecation.py`). Gate:
  `tests/test_risk/test_ladder_deprecations.py`. Definition:
  [Theory § 7.4](theory/risk-measures.md#74-bucketed-sensitivities)
- [ ] **Remove the deprecated `*Ladder` aliases** — `SensitivityLadder`,
  `compute_ladder`, `KeyedLadder`, `keyed_ladder`,
  `KeyedLadder.curve_ladder`, `BucketedLadder`, `bucket_sensitivity_ladder`,
  the `ladder=` keyword and the shim modules `valax.risk.ladders` /
  `valax.risk.state_ladders`. Blocker: one deprecation cycle (the first
  tagged release after the rename); delete `valax/risk/_deprecation.py`
  and `tests/test_risk/test_ladder_deprecations.py` with them
- [ ] **Revaluation grids (ladders) at shifted levels** — `RevaluationGrid`
  / `revaluation_grid` / `revaluation_grid_pnl` in
  `valax/risk/revaluation_grids.py`: PV and directional delta (`jax.jvp`)
  at a grid of shift levels along per-pillar or parallel directions of a
  `KeyedFactorLayout`, `vmap`ped over levels, with P&L read off by cubic
  Hermite interpolation — all orders along each direction, unlike the
  second-order waterfall, and the engine FRTB SA curvature needs. Cross
  terms between directions come from `KeyedSensitivities.gamma`. Blocker:
  none (builds on `shock` / `group_pv`); 2-D product grids (spot × vol)
  follow. Definition: [Theory § 7.4](theory/risk-measures.md#74-bucketed-sensitivities)
- [x] **Market-sourced equity model parameters** — `BlackScholesModel`,
  `HestonModel` and `LocalVolModel` now accept `rate` / `dividend` (and
  BS `vol`) as `None`, sourced from the `MarketState` at price time by
  `valax/core/binding.py::bind_model`. A sourced rate is the
  continuously-compounded zero at **the trade's own maturity**, day-count
  aware via the new `valax.dates.ordinal_from_year_fraction`; a sourced
  vol is the surface at `(strike, expiry)`. Because binding sits inside the
  differentiated path, an equity group's curve shock now reaches the price
  (measured +0.92 on a 1Y two-option book for +100 bp) and
  `group_market_gradient` reports the pillar delta on the trade's own
  pillar instead of a structural zero — so the `TestShadowedShockGuard`
  warning stops firing by itself. Constant-mode pricing is unchanged,
  bit-for-bit. Gate: `tests/test_core/test_binding.py`. Recipe:
  [API → Core § Model binding](api/core.md#model-binding-constants-or-market-sourced)
- [ ] **Market-sourced parameters for the remaining two equity models** —
  `SLVModel` (its `rate` / `dividend` are followed by the non-defaulted
  `surface` / `leverage`, so optionality needs a field-order or
  `__check_init__` decision, and its docstring declares field order a
  frozen contract) and `MultiAssetGBMModel` (`vols` / `dividends` are
  per-asset **vectors** and `n_assets` is derived from `vols.shape[0]`,
  so sourcing them needs a per-asset market lookup). Everything else is
  in place; these two are shape problems, not design problems
- [x] **Full curve de-embedding for the short-rate models** — Hull-White,
  G2++ and Cheyette (1F/2F) carry `initial_curve: DiscountCurve | None = None`
  plus a static `curve_id`, and `from_market` / `with_market_curve` are gone.
  `price()` binds the curve through `valax.core.binding.curve_requirements`
  and refuses a pre-bound one, so curve risk lives only on the market; see
  [§5.6](#unified-calibration-pricing-and-risk) for the full contract
- [x] **`IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>` risk factor** — the
  basis is now a *coordinate*, not only a scenario: `as_basis_layout`
  reparameterises a projection curve's block into the spread
  `r_proj = r_disc + b` and names it via `basis_factor_prefix`
  (`USD.SOFR.3M` over `USD.OIS` → `IR.BASIS.USD.SOFR3M.OIS`), and
  `keyed_basis_shift` takes a **basis** as input rather than a raw
  zero-rate bump — the two things
  [Risk Factors § 3.2](risk-factors.md#32-interest-rates-multi-curve)
  asked for. Consequences: the basis carries its own variance in
  `block_covariance`, its own term in `keyed_waterfall_pnl`, its own
  `KeyedSensitivities.basis_dv01`, and its own `keyed_component_var` share — and
  a *pure discount* move in basis coordinates correctly drags the
  projection curve with it, which the absolute layout can only fake with a
  near-unit correlation
- [ ] **Graph-level greeks API** — *(delivered for groups; `market_greeks`
  itself unchanged)* `KeyedSensitivities` is the graph-aware, all-curves-at-once
  sensitivity API at the **group** level: one `jax.grad` fills every curve's
  block and `key_rate_dv01` / `dv01` / `block_totals` / `to_dict` read them
  back by identifier. What is still true of
  `valax/greeks/market_greeks.py` is unchanged: `curve_pillar_deltas`
  resolves exactly one `curve_id` per call and `bucketed_curve_deltas`
  takes a flat pillar vector with no curve identity, so the
  **single-instrument** greeks API still has no `dict[curve_id, key-rate DV01s]`
  form and no graph-aware bucketing. Retargeting it onto the keyed layout
  (and composing `BucketMap` with `KeyedFactorLayout` for graph-aware
  bucketing) is the remaining work
- [x] **Compose the curve build with `∇_MarketState price`** — the `J_build`
  leg of the Jacobian stack (`PRICING_ONTOLOGY.md` § 6.3) ships as
  `valax.greeks.adjoint`: `curve_adjoint` builds a per-snapshot handle
  carrying the factorised residual Jacobian, and `curve_quote_deltas`
  returns `∂price/∂quote` for every bootstrap instrument. It goes through
  the **implicit function theorem** rather than multiplying a dense
  `quote_jacobian`: one reverse sweep through the pricer, one transposed
  solve, one reverse sweep through the residual — so the cost is flat in
  the number of quotes and each extra trade is a back-substitution
  (`CurveAdjoint.factorisation`). `level_adjoint` / `CalibrationLevel`
  generalise the same step to any calibration written as `R(θ; inputs) = 0`.
  Gated by `tests/test_greeks/test_quote_adjoint.py` (AAD vs full-rebuild
  central finite differences at 0.1 bp and 1 bp, the parallel-shift
  identity, and quotes-fixed vs nodes-fixed). Walkthrough in
  [Tutorial — Unified Pricing § 9](guide/tutorial-unified-pricing.md#from-quotes-to-hedge-ratios);
  conformance scored in
  [Audit Conformance Map](architecture/aad-audit-conformance.md)

#### 2. Calibration is single-curve *(closed for the short-rate models)*

- [x] **Dual-curve rates calibration (Hull-White, G2++)** —
  `calibrate_hull_white` and `calibrate_g2pp` take `forward_curve=` (and
  `curve_id=` to record the market linkage), projecting through the frozen
  tenor basis while discounting on `curve`; `swaption_prices_from_vols` takes
  it too, so *both* legs of the residual see both curves. Required extending
  the closed forms first: `hw_swaption_price` (Jamshidian, via a `par_value`
  on `hw_critical_rate`) and `g2pp_swaption_price` — the latter previously
  *raised* on a projection curve. Validated against the dual-curve PDE recipes
  and by model-independent payer/receiver parity
  (`tests/test_pricing/test_dual_curve_swaptions.py`,
  `tests/test_calibration/test_dual_curve_calibration.py`). See the
  [calibration guide § 6](guide/calibration.md#6-dual-curve-rates-calibration-ois-discounting)
- [ ] **Dual-curve Cheyette calibration** — `calibrate_cheyette` /
  `calibrate_cheyette_2f` currently **raise** on a distinct `forward_curve`
  rather than silently ignoring it, because the Cheyette PDE engine has no
  discount/projection split at all. Needs the engine work in
  [§ 7 Modelling](#7-modelling) first, then the calibrator is a one-line
  pass-through
- [ ] **`MarketState`-native calibration entry points** — the standalone
  calibrators take explicit curves (the kernel convention); a facade taking a
  `MarketState` + `discount_id` / `forward_id` would remove the last manual
  step between the market container and a calibrated model. The chain nodes
  already do this (`curve_graph_node` → `swaption_cube_node` /
  `caplet_strip_node` → `swaption_basket_node` read their upstream curves from
  the `MarketState`; see [§5.6](#unified-calibration-pricing-and-risk))
- [ ] **Curve-aware equity calibration** — `calibrate_heston`,
  `calibrate_sabr`, `calibrate_slv` take loose scalars
  (`rate`, `dividend`, `forward`), never a curve object

#### 3. Vol-surface construction is curve-free *(closed for caps/floors and swaptions)*

`valax/surfaces/caplet_stripping.py` was the first curve-aware module in
`valax/surfaces/`, taking `DiscountCurve` arguments directly;
`calibrate_swaption_cube_from_curves` now closes the swaption side the same
way, and the legacy frozen-forward `calibrate_swaption_cube` has been removed.
`build_sabr_caplet_surface` retains its caller-supplied pre-computed
`forwards` array as an explicit opt-out, for when the forwards genuinely are
exogenous. Both rates vol objects are also chain levels on the upstream curves
(`swaption_cube_node`, `caplet_strip_node` → `optionlet_sabr_node`; see
[§5.6](#unified-calibration-pricing-and-risk)).

- [x] **Cap → caplet (optionlet) stripping** — `strip_caplet_vols`
  bootstraps a quoted cap strip into per-optionlet vols (one
  `optimistix.Newton` root-find per quoted maturity over nested
  schedules, so the whole strip is differentiable by implicit
  differentiation), and `strip_optionlet_surface` carries a whole
  strike × maturity quote matrix through to a per-expiry SABR
  `OptionletVolSurface`. *The* canonical dual-curve operation: forwards
  are projected off the tenor curve (`optionlet_forwards`) and payoffs
  discounted off OIS, so the same quotes strip to different vols under a
  different basis. `cap_price_from_strip` closes the round trip. Gates:
  `tests/test_surfaces/test_caplet_stripping.py` (repricing to 1e-10 rel.,
  single- and dual-curve; the first multi-curve tests under
  `tests/test_surfaces/`) and
  `tests/test_quantlib_comparison/test_cap_strip_on_caplet_vols_ql.py::TestVALAXStripRepricesQuotesInQL`,
  where QuantLib's own `BlackCapFloorEngine` reprices the quotes off the
  VALAX strip to 1e-10 — the counterpart of `ql.OptionletStripper1`.
  Recipe:
  [Volatility Surfaces § Cap → caplet stripping](guide/vol-surfaces.md#cap-caplet-optionlet-stripping-rates)
- [x] **Dual-curve swaption-cube calibration** —
  `calibrate_swaption_cube_from_curves` / `calibrate_swaption_cube_from_graph`
  take per-node `Swaption` pytrees plus a discount/projection pair and derive
  the grid axes (`swaption_node_coordinates`) and each node's $(S_x, A_d)$
  (`swaption_node_forwards_and_annuities`) themselves, through the pricer's own
  `swaption_forward_and_annuity`. So the fitted SABR parameters are
  differentiable functions of the curve pillars, where a fit on a frozen
  (`stop_gradient`-ed) forward gives identically `0.0`.

    This was the load-bearing gap in the adjoint chain, not just a
    convenience one: with the curve absent from the cube's graph,
    `dθ_v/dθ_c = 0` identically, which blocked three of the six paths by
    which curves enter model calibration (target annuities and forwards,
    the cube vol at a moneyness-dependent strike, and basket strikes) and
    therefore blocked the `λ_v` adjoint level. A missing `dR_v/dθ_c` is the
    single most common cause of wrong deltas.

    Delivered in the sequence decided in
    [Volatility Surfaces § Smile dynamics](guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves):

    1. The vol-source protocol is widened to
       `src(strike, expiry[, tenor], forward=F)`, with the live forward
       computed once by the pricer and reused in the payoff
       (`swaption_forward_and_annuity` +
       `swaption_black76_from_forward` / `swaption_bachelier_from_forward`,
       `caplet_forward` / `cap_forwards`). Breaking change; `ConstantVol` /
       `_FlatVol` ignore the new argument, so drop-in scalars and the acyclic
       `surfaces → pricing` dependency both survive.
    2. `dynamics` and `quote_form` static fields on `SwaptionCube` /
       `OptionletVolSurface`, implemented in `valax.surfaces.dynamics`,
       defaulting to `"sticky_strike"` so current numbers are preserved
       bit-for-bit, and validated against each other.
    3. The curve-aware calibration entry points above, accepting ATM-offset
       strikes so the quoted form is representable.
    4. `sabr_backbone` is selectable but is **never a default**, even when the
       quote form admits it: it reproduces the calibrated vol at a node only if
       $F = F_0$ there, which holds on route 3 and is asserted by the cube's
       T-1 repricing gate.

    Gates: `tests/test_surfaces/test_swaption_cube.py::TestCurveAwareCalibration`
    (T-1 repricing to 1e-8 in vol units, both quote forms; non-zero
    `d(fitted vol)/d(DF)`; dual-curve basis moves the anchors) and
    `tests/test_surfaces/test_smile_dynamics.py`.
- [x] **Swaption premium → vol conversion** —
  `valax.surfaces.swaption_implied_normal_vol` /
  `swaption_premium_residual` / `swaption_forward_and_numeraire` solve
  `N·𝒩(θ_c)·Bachelier(S_x(θ_c), K, σ_N, T) − q = 0` by scalar
  `optimistix.Newton` with implicit differentiation, honouring physical
  (numeraire $A_d$) versus cash settlement (numeraire
  $P_d(0,T^p)\,A_{\text{cash}}(S_x)$, via the promoted
  `cash_settled_annuity`). The implied vol therefore carries curve risk, which
  it cannot when the numeraire is frozen. Gate:
  `tests/test_surfaces/test_swaption_premium.py` (round trip to 1e-10, FD-matched
  curve gradients, the two settlements imply different vols).
- [x] **Vega in quote space (λ_v)** — `valax.greeks.vol_adjoint`
  (`sabr_slice_level`, `vol_quote_vegas`, `cube_node_level`,
  `optionlet_slice_level`, `cube_node_cotangent`) expresses a smile fit as a
  `CalibrationLevel` whose residual is the **first-order condition** of the
  weighted least squares — square by construction, so the implicit function
  theorem applies to an over-determined fit. Returns vega to the quoted vols
  *and* `forward_bar`, the `dR_v/dθ_c` term. Gate:
  `tests/test_greeks/test_vol_adjoint.py` — both blocks against full-rebuild
  central finite differences at two bump sizes, under both quote forms.

    Building this gate is what uncovered a long-standing derivative bug in
    `sabr_implied_vol`: Hagan's `z/x(z)` was pinned to the constant `1.0` at the
    money, which has the right value and zero slope, so
    `∂σ/∂F` was wrong by 18% (lognormal) and identically zero (normal) at ATM.
    Prices were never affected. Written up as
    [Numerical Pitfalls entry 5](architecture/numerical-pitfalls.md#5-a-value-correct-guard-with-no-derivative-jnpwhere-on-a-removable-singularity),
    reproduced in `examples/pitfalls/02_where_guard_kills_the_derivative.py`,
    regression-guarded by `tests/test_models/test_sabr.py::TestATMDerivatives`.

#### 4. FX / credit / inflation carry no selectors

- [x] **Selectors on the inflation instruments and the cross-currency swap**
  (Phase 4, D16) — `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`,
  `InflationCapFloor` carry `discount_id` / `inflation_id`;
  `CrossCurrencySwap` carries `foreign_discount_id` and keys its FX spot by
  `currency_pair`
- [ ] **`discount_id` / `forward_id` on `instruments/{fx,credit}.py`**
  — deferred at ontology P3; these classes silently fall back to
  `_default_` with no way to name a curve
- [x] **Extend `ResolvedMarket`** (`valax/core/resolve.py`, Phase 4
  [D16](architecture/calibration-pricing-risk.md#3-decisions)) — keyed views
  of every entry a trade names: `curves` (`curve_ids_of`, including the
  second-currency `foreign_discount_id`), `inflation_curves`
  (`inflation_ids_of`), `fx_spots` (`fx_ids_of`) and `spots`
  (`underlying_ids_of`), with the discount / projection pair as derived views
- [ ] **Survival curves in `ResolvedMarket`** (`credit_id`) — the same keyed
  pattern, with the credit pricers
- [ ] **True multi-curve cross-currency** — `cross_currency_swap_price`
  takes two curves but its own docstring states *"single-curve-per-currency
  projection: each currency's floating leg is forecasted off its own
  discount curve"*. XCCY is the multi-curve instrument par excellence
- [x] **Per-leg index descriptors — the architectural ceiling on the
  current selectors.** A singular `forward_id` expresses the *single-index*
  case only. A single-currency tenor basis swap needs **three** curves (OIS
  discount + two projections), CMS-spread three, cross-currency four or
  more. Built with [Tier 1.4](#14-trade-representation-and-cashflow-engine)
  as one change, as this item required: `valax.instruments.legs` puts
  `discount_id` / `forward_id` / `index_id` on **each leg** and the trade
  is `MultiLegSwap(legs=tuple[...])`, with `curves_touched` /
  `indices_touched` the de-duplicated union across legs (first-appearance
  order, so it is deterministic rather than hash-dependent). Gated by
  `tests/test_pricing/test_legs.py::TestPerLegIndexDescriptors`, which
  prices a three-curve tenor basis swap and checks that bumping one
  projection curve moves **exactly one** leg.
  **Note the scope:** this lifts the ceiling for trades expressed as legs.
  The *existing* single-trade pytrees in `valax.instruments.{rates,bonds}`
  still carry the singular pair and are unchanged, so the other items in
  this section — the FX / credit / inflation selectors above, and the
  single-curve chokepoints below — are not retired by this. Background in
  `PRICING_ONTOLOGY.md` § 0.1 (repo root)

#### 5. Single-curve chokepoints

- [ ] **`valax/pricing/lattice/hull_white_tree.py`** — reads the model's
  bound `initial_curve` (a curve-free model is bound first with
  `bind_curves`); the callable / puttable bond pricers have no
  `forward_curve` parameter (the swaption tree pricers take one). Reachable
  from `price(..., method="lattice")` since D15, whose bond routes refuse a
  distinct projection curve rather than price it single-curve. After P5.1 the
  tree↔PDE↔MC triangulation (load-bearing validation here) only holds
  single-curve, so a dual-curve callable bond has no independent oracle
- [ ] **`valax/risk/oas.py`** — `bond_z_spread`, `z_spread_duration`,
  `z_spread_dv01`, `z_spread_convexity` all take one `curve: DiscountCurve`
- [ ] **`valax/portfolio/batch.py`** — `batch_price` / `batch_greeks` take
  scalar `rates`, no curve object at all; pre-dates the curve layer
- [ ] **`valax/risk/factors.py`** — `curve: DiscountCurve` at two sites
- [ ] **`MarketData.curve_graph` is only carried, not consumed** — the
  scenario / VaR path now threads it end-to-end (§1 above) and
  `apply_scenario` keeps its `"_default_"` entry in step with
  `discount_curve`, so a graph-aware `pricing_fn` can read it; but no
  *library* pricer takes a `MarketData`, so nothing in `valax/pricing/`
  reads the field. Retire it with `MarketData` once the risk engine
  targets `price(...)` on a `MarketState`

#### 6. Curve build (absorbed from the retired MC-Curves-3 notes)

- [ ] **Alternative interpolation modes on `CurveSpec.interp`** —
  `log_linear_df` is the only accepted value (`_SUPPORTED_INTERP` in
  `valax/curves/graph.py` raises otherwise). Wanted: `linear_zero`,
  `monotone_convex` (Hagan-West), `tension_spline`. The dispatch site
  already exists. Log-linear is stable and monotone but produces
  discontinuous forward rates at pillars
- [ ] **Genuine turn-of-year step interpolation** — `TurnInstrument` ships
  as a quote type, but a sharp jump needs an interpolation mode admitting
  discontinuities; today it is approximated by two tightly-spaced pillars
- [ ] **Hull-White futures convexity** — `hull_white_convexity_adj(model)`;
  the plug-in framework is in `valax/curves/convexity.py`
  (`no_convexity_adj`, `constant_convexity_adj`), blocked on threading a
  calibrated `HullWhiteModel` into the curve build
- [ ] **CSA / collateralised discounting** — currency-of-collateral
  selection and FX-implied foreign-collateralised discount curves. No
  machinery and no design note exist; `CurveSpec.currency` is declared but
  consumed by nothing
- [ ] **Business-day calendars and roll conventions in the curve build** —
  the calendars themselves shipped with
  [Tier 1.3](#13-business-calendars-and-holiday-schedules)
  (`valax.dates.calendars`), and `ScheduleConventions` now applies them, but
  nothing in `valax/curves/` consults one: `CurveSpec` carries no calendar,
  so every bootstrap pillar is still whatever date the caller supplies, and
  `payment_lag` remains a calendar-day shift. What is left here is wiring,
  not machinery
- [ ] **Over-determined bootstrap** — an `optimistix.least_squares` variant
  of `bootstrap_curve_graph` with per-quote weights
- [ ] **Fixings seasoning** — `OISSwapRate` and the futures residual do
  `del fixings`, so partially-seasoned swaps ignore realised fixings and
  assume the curve already encodes them through `DF(start_date)`.
  `index_id` is reserved for this. (Unrelated to the payment-lag work
  above: the OIS leg is no longer telescope-only, but it is still
  fixing-free)
- [ ] **Constrained curves** — forward-rate positivity, calendar-spread
  no-arbitrage for inflation curves

#### 7. Modelling

- [ ] **Stochastic tenor basis** — the P5.1 overlay freezes the basis, so
  basis vega is structurally zero (a basis swaption or tenor-spread option
  prices at zero). Needs a multi-curve HJM / Mercurio extension with its
  own driver per tenor curve. See
  [Multi-Curve Short Rates § 3.3](theory/multi-curve-short-rate.md)
- [ ] **Cheyette and LMM dual-curve recipes** — untouched by P5.1; they do
  not declare `discount_curve` / `forward_curve`, so the dispatch bridge
  correctly does not forward them. This blocks dual-curve Cheyette
  *calibration* too (§ 2 above), which refuses rather than ignores a
  projection curve until the engine supports one
- [ ] **CMS-spread projection** — the G2++ PDE and MC recipes rebuild par
  swap rates from the model's own bond strip and therefore *reject* a
  distinct projection curve rather than ignoring it

#### 8. Test coverage holes

Curve construction, `resolve`, and the analytic / MC / PDE pricing paths
are well covered (`tests/test_curves/`, `tests/test_pricing/test_graph_pricing.py`,
`tests/test_{pde,mc,pricing}/test_multicurve.py`). Absent entirely:

- [x] **Multi-curve risk tests** — `tests/test_risk/test_state_shocks.py`
  covers keyed per-curve / basis shocks on a `MarketState`, and
  `tests/test_risk/test_var.py::TestCurveGraphSurvivesVaRPath` gates the
  graph through the whole VaR path. `test_curve_graph_carried_through`
  still asserts a *named* graph curve comes back unchanged from
  `apply_scenario` — which is now the documented contract of the unkeyed
  legacy vectors, not an accident. The VaR / sensitivity hole is now closed too:
  `tests/test_risk/test_keyed_scenarios.py::TestBasisRiskReachesPnL` draws a
  scenario set on the **basis block alone** and gets a loss distribution out
  of `group_scenario_pnl`, and
  `tests/test_risk/test_state_sensitivities.py::TestKeyedVaR` gates per-curve
  component VaR (additive, and zero for a curve with no variance) plus
  agreement between delta-normal and 20k-scenario simulated VaR
- [x] **Multi-curve calibration tests** — `tests/test_calibration/test_dual_curve_calibration.py`
  gates the dual-curve Hull-White / G2++ route end to end:
  `TestSingleCurveNonRegression`, `TestDualCurveRoundTrip`, `TestBasisMatters`
  (including `test_ignoring_the_basis_biases_sigma`), `TestVolsToPrices`,
  `TestCurveIdBinding` (re-binding a fitted model to a shocked `CurveGraph`),
  `TestCheyetteRefuses` (both 1F and 2F raise on a distinct projection curve)
  and `TestJax` (gradient flow through the dual-curve residual)
- [ ] **Multi-curve greeks tests** — `tests/test_greeks/test_market_greeks.py`
  runs entirely against single-curve markets; nothing asserts separate
  OIS vs projection key-rate DV01s. Note this is now a gap in the
  **single-instrument** greeks API only: separate OIS / projection /  basis
  sensitivities *are* gated at group level in
  `tests/test_risk/test_state_sensitivities.py::TestPerCurveSensitivities` (including
  the control that an unused projection curve carries exactly zero risk)
- [x] **Multi-curve surface tests** — `tests/test_surfaces/test_caplet_stripping.py`
  strips a cap strip against an OIS discount curve and a distinct 3M
  projection curve, asserts the basis moves the stripped vols, and
  differentiates a stripped vol w.r.t. the projection curve's pillars
- [ ] **Portfolio tests** — there is no `tests/test_portfolio/` package at all

### Recommended sequence

1. **Multi-curve risk** (§1) — highest leverage by a distance: it is the
   stated purpose of the framework. Four parts have landed: keyed
   scenarios + `shock` on a `MarketState`; the repricing loop and the
   gradient pytree through `price(...)` (`valax/risk/state_var.py`); keyed
   generators + the per-curve covariance layout (`KeyedFactorLayout`,
   `block_covariance`); and the keyed sensitivities, per-block waterfall and
   per-curve VaR decomposition (`valax/risk/state_sensitivities.py`), which also
   closed the `IR.BASIS` factor. So a basis move now reaches a P&L *and* a
   VaR number, attributed to the basis by name. What is left above it is
   the **portfolio aggregation** of the three additive quantities across
   groups (a quantile of a sum is not a sum of quantiles, so the risk
   measure belongs at the top only), re-targeting the FRTB PLA vectors
   (`valax/risk/pnl_vectors.py`) onto the keyed sensitivities, and the
   single-instrument graph-level greeks API — which wants the same
   per-curve plumbing and should reuse `KeyedFactorLayout` rather than
   grow its own.
2. **Dual-curve calibration** (§2) — smaller, and it closes the
   consistency gap that P5.1 opened between how models are fitted and how
   they are now priced.
3. ~~**Caplet stripping** (§3)~~ — **done**: `strip_caplet_vols` /
   `strip_optionlet_surface` unlock the quotes→surface→exotic chain and
   give the dual-curve vol story its first honest leg. What remains in §3
   is the swaption cube, which wants the same treatment.

Items in §6 are independently shippable and mostly self-contained; §7 is
research-scale.

---

## Production Priority Roadmap (Snapshot — reconciled against the codebase)

The tier-based feature inventory below is comprehensive but feature-area oriented. This section reorganizes the most critical work into **execution priorities** — what a bank evaluating VALAX would need to see, in roughly the order it should be built.

This is a **point-in-time snapshot**. Priorities will shift as items are completed and real user feedback arrives.

### Priority 1 — Foundational Infrastructure *(Blocks Everything)*

These items are hard prerequisites. Every downstream feature, product, and integration depends on them.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P1.1** | ~~**Business Calendars & Date Adjustment**~~ — **done**, [Tier 1.3](#13-business-calendars-and-holiday-schedules) | Every trade in a bank settles on business days. Wrong dates = wrong cashflows = wrong P&L. Shipped: `valax.dates.calendars` with holiday ordinal arrays for TARGET, NYSE, SOFR/SIFMA, London, Tokyo and Brazil, following/modified-following/preceding/modified-preceding and the end-of-month rule, union/intersection, and Bus/252. Remaining in this area is *wiring*, not machinery: the curve build and payment lags do not yet consult a calendar. | 1.3 |
| **P1.2** | ~~**Cashflow Engine (Legs, Compounding)**~~ — **done**, [Tier 1.4](#14-trade-representation-and-cashflow-engine) | Real swaps have compounding-in-arrears, amortizing notionals and fixing histories. Shipped: stub schedules and accrued interest (`generate_schedule`, `valax.dates.accrual`), then `FixedLeg` / `FloatingLeg` / `MultiLegSwap` in `valax.instruments.legs` with simple/compounded/averaged compounding, amortisation factors, pricing-time fixing histories and a curve descriptor **per leg** — priced by `valax.pricing.analytic.legs`. Remaining in this area is *adoption*, not machinery: the curve-build quote types and the legacy single-trade pytrees still carry their own leg arithmetic, and payment lag is caller-supplied rather than rolled. | 1.4 |
| **P1.3** | **CI/CD Pipeline** | No bank adopts a library without automated testing. GitHub Actions with: lint, type-check, full test suite, QuantLib comparison, coverage reporting. A gate before anything else ships. | — |
| **P1.4** | ~~**Short-Rate Models (Hull-White, G2++, Cheyette)**~~ ✅ **Done** | One-factor Hull-White, two-factor G2++, and quasi-Gaussian Cheyette (1F/2F) — analytic/tree/MC/PDE pricing, exact curve fit, and swaption-surface calibration (`calibrate_hull_white` / `calibrate_g2pp` / `calibrate_cheyette`). | 2.1 |

### Priority 2 — Production Pricing Capabilities

These unlock entire asset classes and bring pricing to the speed and accuracy required by trading desks.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P2.1** | ~~**Heston Semi-Analytic (COS / Fourier)**~~ ✅ **Fang-Oosterlee COS implemented** | Lord-Kahl "Little Trap" characteristic function + Fang-Oosterlee (2008) COS expansion with closed-form Heston cumulants. Agrees with QuantLib's `AnalyticHestonEngine` to < 5e-7 absolute across the seed × moneyness × call/put grid. End-to-end Heston calibration roundtrip recovers all 5 parameters to floating-point precision on noise-free synthetic data. | 2.2 |
| **P2.2** | ~~**Local Volatility + SLV**~~ ✅ **Both shipped** | LV (Tier 2.3) via `valax/pricing/analytic/dupire.py` + `valax/pricing/mc/local_vol_paths.py`. SLV (Tier 2.4) via `valax/calibration/slv.py` (Guyon-Henry-Labordère particle method + optional kernel-ridge stabilisation + outer fixed-point loop) and `valax/pricing/mc/slv_paths.py` (Andersen-QE variance + log-Euler/Milstein log-spot, registered with the unified MC dispatcher for European/Asian/Barrier/Lookback/VarianceSwap). | 2.3, 2.4 |
| **P2.3** | **FX Derivatives** — ✅ vanillas + delta conventions; barriers / quanto / TARF pending | Garman-Kohlhagen vanillas, forwards, and all three delta conventions ship; FX barrier, quanto and TARF pricers remain. FX has unique conventions (delta-space quoting). | 3.3 |
| **P2.4** | **Credit Derivatives** — ✅ survival curves; CDS / CDO pricers pending | Survival curves from CDS spreads ship (`valax.curves.survival`), using the closed-form credit-triangle approximation rather than a repricing bootstrap. The standalone CDS and CDO-tranche pricers remain — prerequisites for credit trading and the XVA workstream. | 3.4 |

### Priority 3 — Risk & Regulation *(Required for Sign-Off)*

Banks cannot go live without regulatory compliance. These items satisfy Basel requirements and complete the risk framework.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P3.1** | **XVA Suite (CVA, DVA, FVA at minimum)** | XVA is now a first-class P&L line at every bank. CVA alone changes pricing by 10–50 bps on uncollateralized trades. This is where JAX's GPU acceleration creates the biggest competitive advantage (nested MC). | 4.3 |
| **P3.2** | **FRTB Standardized Approach** | Basel III.1/IV mandates FRTB for market risk capital. The Standardized Approach (SA) is the minimum — sensitivity-based with prescribed risk weights. Banks need this for regulatory capital reporting. | — |
| **P3.3** | **Marginal / Component / Incremental VaR** | Current VaR is portfolio-level only. Desks need to attribute risk to individual trades and see the marginal impact of new trades. | 4.1 |

### Priority 4 — Service Layer *(Required for Integration)*

VALAX is currently a Python library. Banks integrate pricing engines as services. This priority turns the library into a deployable system.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P4.1** | **API Server (gRPC + REST)** | Banks integrate pricing via services, not Python imports. A gRPC server (low-latency inter-service) + REST (UIs/dashboards) with OpenAPI docs is the delivery mechanism. | — |
| **P4.2** | **Market Data Adapters & Persistence** | Production systems consume Bloomberg/Refinitiv feeds and store EOD snapshots. Need adapters for market data ingest and a storage layer for curves, surfaces, and fixings. | 6.1 |
| **P4.3** | **Audit Logging & Observability** | Regulatory requirement: every pricing must be reproducible. Structured logging (who priced what, when, with which market data), OpenTelemetry tracing, Prometheus metrics. | — |
| **P4.4** | **Docker + Helm Deployment** | Containerized deployment with GPU support, health checks, graceful shutdown, resource limits. Helm charts for Kubernetes — the standard bank deployment platform. | — |

### Priority 5 — Competitive Differentiators *(Leverage JAX Advantages)*

These are the features that make VALAX not just *another* pricing library but a fundamentally better one. They exploit the JAX-native architecture in ways traditional C++ stacks cannot replicate.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P5.1** | **Neural Surrogate Pricers (Differential ML)** | XVA nested MC is the most compute-intensive task in banking. Neural surrogates trained with autodiff Greeks reduce compute 100–1000x. Same framework for training and deployment — VALAX's killer feature. | 5.1 |
| **P5.2** | **GPU/TPU Benchmark Suite** | Prove the value proposition with numbers. Benchmark VALAX on GPU vs QuantLib on CPU for: batch European pricing, MC VaR, XVA exposure simulation. Publish results. | — |
| **P5.3** | **Deep Hedging** | RL-based hedging with transaction costs is a frontier area. JAX makes the entire training loop differentiable. Positions VALAX at the cutting edge. | 5.3 |

### Recommended Execution Order

```
Q1 (Foundation):
  ├─ [P1.1] Business Calendars (done) ──→ [P1.2] Cashflow Engine
  ├─ [P1.3] CI/CD Pipeline
  └─ [P1.4] Short-Rate Models (Hull-White)

Q2 (Pricing Depth):
  ├─ [P2.1] Heston COS / Fourier
  ├─ [P2.2] Local Vol → SLV
  ├─ [P2.3] FX Derivatives
  └─ [P2.4] Credit (CDS, Survival Curves)

Q3 (Risk & Regulation):
  ├─ [P3.1] XVA (CVA / DVA / FVA)
  ├─ [P3.2] FRTB Standardized Approach
  └─ [P3.3] Advanced VaR Decomposition

Q4 (Production Delivery):
  ├─ [P4.1] API Server (gRPC + REST)
  ├─ [P4.2] Market Data Layer
  ├─ [P4.3] Audit & Observability
  └─ [P4.4] Docker / K8s Deployment

Ongoing (Differentiators):
  ├─ [P5.1] Neural Surrogates
  ├─ [P5.2] GPU Benchmarks
  └─ [P5.3] Deep Hedging
```

### Assessment

VALAX has excellent bones. The pure-functional JAX architecture is genuinely superior to what most banks run internally (mutable C++ object graphs built 20+ years ago). Autodiff Greeks, `vmap` batching, and GPU portability are structural advantages that compound with every new feature added.

The most urgent gap is **foundational infrastructure** (P1: calendars, cashflows, CI/CD) — not because it's glamorous, but because every downstream feature depends on it. A bank quant evaluating VALAX will immediately ask: *"Can it handle modified-following date adjustment on a 10Y EUR swap with short front stub and compounding-in-arrears on the floating leg?"* — and today the answer is *mostly*: the **short front stub** is supported, accrual is measured on a real day-count basis, and **modified-following adjustment against a TARGET calendar** now works and is conformance-tested against QuantLib (Tier 1.3). **Compounding-in-arrears as a leg abstraction** now works too (Tier 1.4): `valax.instruments.FloatingLeg` carries the convention, an amortising notional and an index whose realised fixings season the leg, and `MultiLegSwap` gives every leg its own projection curve. The honest remaining caveat is *adoption* — the curve-build quote types and the legacy swap pytrees still carry their own leg arithmetic, so the answer to that quant's question is now *yes, if you express the trade as legs*.

The **highest-ROI parallel investment** is **short-rate models + Heston COS** alongside calendars/cashflows, because they unlock the most actively traded products (callable bonds, fast equity vol calibration) while the plumbing is being built.

---

## Tier 1: Critical Infrastructure

These are foundational pieces that block almost everything else. They should be built first.

### 1.1 Multi-Curve Bootstrapping

- [x] **Sequential bootstrap** from deposits, FRAs, and par swap rates — analytic solve per pillar
- [x] **Simultaneous bootstrap** via `optimistix.root_find` (Newton) in log-DF space — handles overlapping instruments
- [x] **Multi-curve framework** — `MultiCurveSet` with OIS discount curve + tenor-specific forward curves, dual-curve swap bootstrap (**deprecated in MC-Curves-2**; superseded by `bootstrap_curve_graph`). Since MC-Curves-2b it is a *view* over `CurveGraph` (`MultiCurveSet.from_graph` / `to_graph`); its two-tier `bootstrap_multi_curve` entry point has since been removed
- [x] **Joint multi-curve Newton solver** (MC-Curves-2) — `bootstrap_curve_graph` over an arbitrary `CurveGraph`, plus `CurveSpec` and the implicit-adjoint `quote_jacobian` helper. The build returns a `CalibrationResult` (the same root level as `curve_graph_node`); the old `CurveBuildDiagnostics` report is retired in favour of `result.diagnostics`, `result.condition_number()` and `fitted_curve_quotes`, and `quote_jacobian` raises `CalibrationError` on an unconverged build
- [x] **Basis curve** support (e.g., 1M vs 3M SOFR basis) — via `TenorBasisSwap` inside the joint solver
- [x] **Cross-currency graph** — `CrossCurrencyBasisSwap` + `FXForward` inside the joint solver (EUR-USD four-curve build verified)
- [x] **Jacobian of discount factors w.r.t. input rates** via autodiff (free with JAX, verified in tests)
- [x] **Payment lag** — accrual end $T^e$ and payment date $T^p$ are separate roles on every swap quote type (`payment_dates` / `payment_lag` on `SwapRate`, `OISSwapRate`, and per leg on `IborSwapRate`, `TenorBasisSwap`, `CrossCurrencyBasisSwap`), discounting at $T^p$ via `_floating_leg_pv(payment_dates=…)`. The OIS floating leg is now the per-period sum `_compounded_float_leg_pv`, with the telescoping identity retained as a zero-lag fast path (spec N-3; gated by `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping`). Calendar-day lags only. The calendars needed to roll a lag properly now exist ([Tier 1.3](#13-business-calendars-and-holiday-schedules): `HolidayCalendar.add_business_days`), but no quote type consults one, so a business-day lag must still be pre-rolled by the caller and supplied as `payment_dates`. Wiring it in belongs with the leg descriptors in [Tier 1.4](#14-trade-representation-and-cashflow-engine)
- [x] **Averaged vs compounded futures** — `MoneyMarketFuture.averaging` ∈ `{"compounded", "arithmetic"}` (1M RFR contracts settle on the arithmetic average of the daily index, not the compounded one), via `_arithmetic_average_forward`. The gap is the interest-on-interest averaging discards, of order $r^2\tau/2$, and survives on a flat curve (spec § 4.2; gated by `tests/test_curves/test_instrument_residuals.py::TestAveragedFutures`)
- [x] **Partly-accrued RFR coupons** — `FixingSeries.compounded_factor` reduces a daily history to the realised compounding factor $A$, and `accrued_compounding` on `IborSwapRate` / `TenorBasisSwap` / `CrossCurrencyBasisSwap` blends it with the curve forward for the remainder via $1 + \tau_j F_j = A \cdot P_x(0,t_0)/P_x(0,T_j^e)$ (spec C-3; T-11 gate in `tests/test_curves/test_instrument_residuals.py::TestPartlyAccruedFixings`)
- [ ] **Turn-of-year effects** and stub handling (`TurnInstrument` quote type shipped; genuine step-function interpolation deferred to MC-Curves-3)
- [ ] **Alternative interpolation modes** on `CurveSpec.interp` — linear-zero, monotone-convex (Hagan-West), tension spline (MC-Curves-3). `DiscountCurve` hard-codes log-linear-in-DF and has no `interp` field, so `CurveSpec.interp` is declared and checked but not forwarded. That is guarded, not latent: `graph._SUPPORTED_INTERP` is *derived from* `discount.DISCOUNT_CURVE_INTERP`, and `_build_graph_from_log_dfs` raises `NotImplementedError` at the point the mode would be forwarded — so the first new mode must ship with its dispatch (`tests/test_curves/test_graph.py::TestCurveSpecInterp`)
- [ ] **Hull-White futures convexity** (`hull_white_convexity_adj`) — plug-in framework in place, closed-form factory pending
- [ ] **CSA / collateralised discounting** and FX-implied foreign-currency-collateralised discount curves

> The full multi-curve picture — what ships end-to-end, and the gaps in
> risk, calibration, surfaces and portfolio — lives in
> [Multi-Curve Framework — What Ships, What's Left](#multi-curve-framework-what-ships-whats-left).
> This tier covers the **build** side only.

**Why:** Every rates product needs proper forward curves. Manual pillar specification doesn't scale.

**Approach:** Root-finding via `optimistix.root_find` on the system of bootstrapping equations. Each instrument (deposit, swap) provides one equation. The curve is a pytree — autodiff gives the Jacobian for free.

**Status:** MC-Curves-2 shipped — the joint multi-curve Newton solver `valax.curves.bootstrap_curve_graph` and its declarative `CurveSpec` interface live in `valax/curves/bootstrap_graph.py`. For the narrative rationale (post-2008 forces, why sequential bootstrap breaks down on tenor-basis and cross-currency quotes, and how the joint solve unlocks the implicit-adjoint quote-Jacobian), see [`guide/why-multicurve.md`](guide/why-multicurve.md). Eleven quote-type pytrees (`DepositRate`, `FRA`, `SwapRate`, `OISSwapRate`, `IborSwapRate`, `MoneyMarketFuture`, `TenorBasisSwap`, `FXForward`, `FXSwap`, `CrossCurrencyBasisSwap`, `TurnInstrument`) live in `valax/curves/instruments.py`. Tenor-basis and cross-currency joint solves are verified end-to-end in `tests/test_curves/test_bootstrap_graph.py`. MC-Curves-2b completed the migration: the sequential two-tier internals were deleted, `MultiCurveSet` became a view over `CurveGraph` (see `tests/test_curves/test_multi_curve.py`), and after its deprecation cycle `bootstrap_multi_curve` was removed outright. `quote_jacobian` exposes the implicit-adjoint quote-sensitivity Jacobian.

### 1.2 Volatility Surface Construction

- [x] **Vol surface object** — `GridVolSurface` with bilinear interpolation on (strike, expiry) grid
- [x] **SABR smile per expiry** — `SABRVolSurface` with per-expiry SABR calibration and parameter interpolation
- [x] **SVI parametric fitting** — `SVIVolSurface` with Gatheral's SVI parameterization and LM calibration
- [x] **Swaption cube** — `SwaptionCube` (strike × expiry × tenor), calibrated by the curve-aware `calibrate_swaption_cube_from_curves` / `_from_graph`, which derive the grid axes and each node's $(S_x, A_d)$ off the curves themselves (one batched level, `swaption_cube_level`; a failed cell is named, e.g. `"5Yx10Y"`), or as a chain level by `swaption_cube_node`. The frozen-forward `calibrate_swaption_cube` has been removed
- [x] **Declared smile dynamics** — `valax.surfaces.dynamics` plus the `dynamics` / `quote_form` static fields on `SwaptionCube` and `OptionletVolSurface`: sticky strike (default), sticky moneyness, or SABR backbone, with the pair validated for consistency
- [x] **Swaption premium → vol** — `swaption_implied_normal_vol` / `swaption_premium_residual` in `valax/surfaces/swaption_premium.py`, honouring physical vs cash settlement
- [x] **Optionlet (caplet) surface** — `OptionletVolSurface` via `build_sabr_caplet_surface`
- [x] **Cap → caplet (optionlet) stripping** — `strip_caplet_vols` /
      `strip_optionlet_surface` in `valax/surfaces/caplet_stripping.py`:
      flat quoted cap vols → per-optionlet term structure → per-expiry
      SABR surface, projecting off the tenor curve and discounting off
      OIS. See [Multi-Curve Framework §3](#3-vol-surface-construction-is-curve-free-closed-for-capsfloors-and-swaptions)
- [ ] **SSVI** — global arbitrage-free parameterization (Gatheral-Jacquier)
- [ ] **Sticky-strike vs sticky-delta** conventions
- [x] **Local vol extraction** from implied vol surface (Dupire formula) — shipped

**Why:** Every option product needs a vol surface. Scalar vol is a toy assumption.

**Approach:** Store the surface as an `eqx.Module` pytree with pillar data. Interpolation via cubic splines (or SABR per expiry). SVI fitting via `optimistix.least_squares`. The surface must be differentiable for Greeks.

**Status:** Three surface types implemented in `valax/surfaces/`: `GridVolSurface` (interpolated grid), `SABRVolSurface` (SABR per expiry with calibration), `SVIVolSurface` (SVI per expiry with calibration). All are callable, JIT-able, differentiable pytrees. See the [Vol Surfaces guide](guide/vol-surfaces.md).

### 1.3 Business Calendars and Holiday Schedules

- [x] **Holiday calendar objects** for major markets: TARGET, NYSE, SOFR (SIFMA), London, Tokyo, Brazil — `valax.dates.calendars.HolidayCalendar` plus cached per-market factories (`target_calendar`, `sofr_calendar`, `nyse_calendar`, `london_calendar`, `tokyo_calendar`, `brazil_calendar`); tables are a generated, version-stamped artifact in `valax.dates._holiday_data`
- [x] **Date adjustment conventions**: following, modified following, preceding, modified preceding and the end-of-month rule — `HolidayCalendar.adjust` / `last_business_day_of_month`, applied by `valax.dates.schedule.generate_schedule` and validated as a calendar/convention *pair* in `valax.dates.conventions.ScheduleConventions`
- [x] **Business day counting** (Bus/252 for Brazil) — `valax.dates.daycounts.bus_252`, registered in `CALENDAR_DAY_COUNT_FNS` and reachable through `year_fraction(..., calendar=...)`
- [x] **Calendar combination** (union/intersection for cross-currency) — `HolidayCalendar.union` / `HolidayCalendar.intersection`, narrowing the covered range to the overlap

**Why:** Every real trade settles on business days. Wrong dates = wrong cashflows = wrong prices.

**Approach:** Precomputed boolean arrays of holidays (JIT-compatible, no runtime lookups). Calendar as a pytree with a `jnp.array` of holiday ordinals. Adjustment functions operate on integer ordinals.

**Status:** Implemented in `valax/dates/calendars.py`. A calendar carries two integer tables: the sparse holiday ordinals, which are the canonical definition and what `union`/`intersection` and the generated artifact use, and a dense prefix sum of business days, which answers every query as a constant-time gather. Both membership and counting are therefore `O(1)` rather than `O(log n)` — a measured 27.8x reduction in marginal cost per cashflow (15.56 ns → 0.56 ns), which is what lets Bus/252 be counted per-coupon *inside* traced pricer code. Adjustment is **opt-in**: `business_day_convention` still defaults to `"unadjusted"`, and that branch is the pre-existing code reached by a static Python test, so every default schedule in the suite is bit-for-bit unchanged (gated by `tests/test_dates/test_schedule.py::TestUnadjustedPathIsUnchanged`). Conformance is `tests/test_quantlib_comparison/test_dates_ql.py`, which now runs every schedule case under both `NullCalendar`/`Unadjusted` and `ql.TARGET()`/`ql.ModifiedFollowing` and compares dates by exact integer equality. Each calendar declares the year range its data covers and **raises** outside it, rather than reporting an unknown year as holiday-free. Still open: payment lag is a business-day roll at the instrument level only (Tier 1.4), and there is no versioned index registry (see the [audit conformance map](architecture/aad-audit-conformance.md), rows N-5 and N-6). See the [Dates and Conventions guide](guide/dates-and-conventions.md#business-day-calendars-and-adjustment).

### 1.4 Trade Representation and Cashflow Engine

- [x] **Leg abstraction** — fixed leg, floating leg with compounding conventions — `valax.instruments.legs` (`FixedLeg`, `FloatingLeg`, `MultiLegSwap`, `COMPOUNDING_CONVENTIONS`) priced by `valax.pricing.analytic.legs` (`fixed_leg_pv`, `floating_leg_pv`, `leg_pv`, `swap_pv`, `swap_fair_rate`). `compounding` covers `"simple"` (one fixing per period), `"compounded"` (the RFR daily product, evaluated as a discount-factor ratio) and `"averaged"` (arithmetic, reusing `_arithmetic_average_forward`). In-advance vs in-arrears is carried by the `fixing_dates` array rather than a flag, deliberately: a switch that changed no behaviour would be another never-read `OISSwap.compounding`. **LIBOR-in-arrears is absent, not approximated** — it needs a convexity adjustment, so projecting it off the curve would be silently wrong
- [x] **Cashflow schedule generation** with **stub periods (short/long front/back stubs)**, day-of-month preservation and end-of-month rolls — `valax.dates.schedule.generate_schedule` returns a `Schedule` of `(accrual_start, accrual_end, payment_dates)`; conventions in `valax.dates.conventions.ScheduleConventions`. **Business-day adjustment landed with [Tier 1.3](#13-business-calendars-and-holiday-schedules)**: `calendar` and `business_day_convention` are now applied, validated as a pair, and conformance-tested against `ql.TARGET()` / `ql.ModifiedFollowing`. What remains open here is the payment **lag** — `generate_schedule` takes none, so `payment_dates` still equals `accrual_end`
- [x] **Fixing histories** — realized fixings for partially-settled floating legs — `floating_leg_pv(leg, graph, fixings, ref_date)` consults a `valax.curves.fixings.FixingHistory` under the leg's `index_id`. Settled, straddling and unborn periods collapse into one expression by clipping the valuation date into each period (`c_i = min(max(t_0, T_i^s), T_i^e)`), so the realised compounding factor and the remaining projection share one cut-over and cannot double-count the valuation day. The history is a *pricing-time argument*, not an instrument field, so one leg revalues against successive histories
- [x] **Amortization schedules** — notional step-down for amortizing swaps — `amortisation` on both leg types, an array of **dimensionless outstanding-notional factors** against a scalar reference `notional`. Factors rather than absolute notionals keep the notional outside the coupon sum, which is what lets a non-amortising leg reproduce the flat pricers bit for bit (`sum_i (N x_i)` and `N sum_i x_i` differ in the last bit for most inputs; multiplying by exactly `1.0` is exact)
- [x] **Accrued interest** calculation and the **clean/dirty distinction** — `valax.dates.accrual` (`accrued_interest`, `bond_accrued_interest`, `clean_from_dirty`, `dirty_from_clean`); coupon sizing via `fixed_coupon_amounts` (day-count year fraction per accrual period, replacing `coupon_rate / frequency` across the analytic, lattice, PDE and MC pricers); `price_type="clean"|"dirty"` on `valax.risk.oas.callable_bond_oas` / `bond_z_spread`; callable/puttable exercise now compares a clean call price against a clean continuation value in both `valax.pricing.lattice.hull_white_tree` and `valax.pricing.pde.rates_payoffs`

**Why:** Real swaps and bonds have complex cashflow structures. A partially-settled, amortising or multi-index trade could not previously be represented at all.

**Approach:** `FixedLeg` and `FloatingLeg` as `eqx.Module` pytrees. Schedule generation takes a calendar + conventions and produces arrays of (accrual_start, accrual_end, payment_date, notional) — all four now exist.

**Status:** Implemented in `valax/instruments/legs.py` (data) and `valax/pricing/analytic/legs.py` (pricing), keeping instruments data-only. The leg abstraction and the **per-leg index descriptors** of the [multi-curve framework's §4](#4-fx-credit-inflation-carry-no-selectors) were built as *one* change, as that item required: each leg names its own `discount_id` / `forward_id` / `index_id`, and `MultiLegSwap.curves_touched` is the union across legs. A single-currency tenor basis swap therefore reads its three curves — the case a singular `forward_id` on a trade pytree structurally cannot express.

The engine does not introduce new arithmetic so much as generalise three expressions that already shipped in narrower form, and each is reproduced **bit for bit** on its own path, asserted with bare `==`: `swaptions.annuity` (flat fixed leg), `_compounded_float_leg_pv` (compounded leg, zero lag) and `_floating_leg_pv` (dual-curve simple leg with realised fixings). Those functions are untouched — the change is additive, so no existing price moves. Independent authority comes from `tests/test_quantlib_comparison/test_legs_ql.py`: the amortising leg against `ql.FixedRateLeg` with a notional *vector* (rel. 1.6e-15), and the compounded leg against `ql.OvernightLeg`, which forms the daily product explicitly where VALAX evaluates a single discount-factor ratio — agreement there is **exact**.

Still open: `MultiLegSwap` sums its legs directly, so every leg must discount in the same currency (no FX conversion, and no `currency` field rather than a field nothing reads); the `"averaged"` convention needs a static `max_days` bound to be priced as a *traced argument*, since the per-rate-day subdivision is a data-dependent shape (it prices eagerly, or closed over, without one — see [JAX Patterns § 2.5](architecture/jax-patterns.md#25-when-you-genuinely-need-a-value-at-trace-time-data-dependent-shapes)); payment **lag** is still caller-supplied `payment_dates` rather than a business-day roll; a leg names no calendar, so `day_count="bus_252"` is refused at construction and Brazilian legs stay out of reach; and the legacy `valax.instruments.{rates,bonds}` pytrees are unchanged, so nothing is deprecated yet. See the [leg engine guide](guide/fixed-income.md#swap-legs-and-multi-leg-trades).

---

## Tier 2: Models

New stochastic models that unlock entire product categories.

### 2.1 Short-Rate Models

- [x] **Hull-White (one-factor)** — mean-reverting short rate; analytic bond & swaption prices (Jamshidian), trinomial tree, MC, PDE
- [x] **G2++ (two-factor)** — two correlated Gaussian factors; MC + PDE, decorrelation payoffs (CMS-spread)
- [x] **Cheyette (quasi-Gaussian, 1F/2F)** — state-dependent local vol adds a rate skew; MC + PDE; reduces to Hull-White / G2++ at β=0
- [ ] **CIR** — square-root diffusion, ensures positive rates
- [ ] **Vasicek** — Ornstein-Uhlenbeck for rates (allows negative rates)
- [x] **Calibration to swaption surface** — `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_cheyette` / `calibrate_cheyette_2f`, all thin wrappers over one basket level (`valax.calibration.basket.basket_fit_level`) returning a `CalibrationResult` whose value is a **curve-free** model; per trade off a calibrated cube via `swaption_basket_node` ([§5.6](#unified-calibration-pricing-and-risk))
- [x] **Short-rate closed forms through `price()`** — `Swaption × HullWhiteModel × "analytic"` (Jamshidian) and `Swaption × G2PPModel × "analytic"` (Brigo-Mercurio quadrature) are registered routes (`valax/core/adapters.py`) and the auto defaults for their pairs, so per-trade baskets price by the closed form unless told otherwise. Their exercise-boundary root-finds (`hw_critical_rate`, the G2++ `_critical_y`) return NaN on a failed Newton solve instead of the last iterate, so a failed decomposition surfaces as a named non-finite residual inside a calibration. They are **guarded** (Phase 4): a detached probe solve decides success and the differentiable solve runs on a well-posed stand-in otherwise, so an absurd or non-finite parameter gives a NaN price and gradient instead of a runtime raise from the implicit derivative inside `jit` / `vmap`, and a healthy trade in the same batch is unaffected (healthy values and gradients bit-identical; `tests/test_pricing/test_root_guard.py`; overhead in the [register](applications/calibration-register.md#hull-white))
- [x] **Multi-curve pricing in the MC and PDE short-rate solvers** (ontology P5.1) — `valax.pricing.multicurve` (`with_discount_curve`, `tenor_basis`, `swap_leg_basis`, `apply_swap_basis`): OIS discounting by re-fitting the exact-fit model to the trade's discount curve, plus a frozen deterministic tenor basis for projection. Wired into every Hull-White and G2++ recipe in `valax/pricing/pde/{hull_white,g2pp}.py` and `valax/pricing/mc/recipes.py`, and reached automatically from an instrument's `discount_id` / `forward_id` through `valax.core.price`. Single-curve prices are unchanged bit-for-bit. See [Multi-Curve Short Rates](theory/multi-curve-short-rate.md)
- [x] **Time-dependent (piecewise-constant) volatility** for Hull-White, G2++ and Cheyette 1F/2F (Phase 4) — every model's volatility is an array of buckets with traced breakpoints `volatility_times` (a scalar is one bucket, bit-identical to before against a golden file, `tests/test_models/test_phase4_golden.py`); shared primitives in `valax.models.short_rate` (`decayed_integral`, `piecewise_integral`, `bucket_index`, `volatility_bucket_fields`); Hull-White `hw_state_variance` / `hw_mean_variance_rate`; G2++ and Cheyette 2F parametrised as a first-factor vol and a ratio (`volatility_x` + `volatility_ratio`, `volatility_1` + `volatility_ratio`; `volatility_y` / `volatility_2` are read-only properties). Every engine reads σ(t): Jamshidian / Brigo-Mercurio closed forms, the PDE (exact step averages; `g2pp_operator_rows`, `cheyette_exact_operator_rows`, `cheyette_operator_rows_2d`), Monte Carlo (exact step variance / covariance) and a general Hull-White trinomial tree (`build_general_hull_white_tree`, `hw_tree_width`) with European and Bermudan swaption tree pricers (`swaption_tree_price`, `bermudan_swaption_tree_price`). Validated against quadrature (1e-12), QuantLib `Gsr` (Hull-White, within 5e-6 of `Gaussian1dSwaptionEngine`), PDE ↔ closed form ↔ tree ↔ MC triangulations, and the β = 0 reductions of Cheyette to Hull-White / G2++. Maths: [Time-Dependent Volatility](theory/time-dependent-volatility.md); the co-terminal bootstrap it enables: [§5.6](#unified-calibration-pricing-and-risk)
- [ ] **Global piecewise σ(t) calibration on a user grid** — the standalone calibrators (`calibrate_hull_white`, `calibrate_g2pp`, `calibrate_cheyette*`) still fit one constant bucket; a σ(t) fitted on a user-chosen grid to a whole basket (a square strip where the basket is triangular, regularised least squares otherwise — a smoothness penalty against the sawtooth an exact bootstrap makes of quote noise) is the model a portfolio / XVA simulation needs, where one model must serve many trades
- [ ] **Hull-White PDE sensitivity on the t = 0 pillar**: through `market_risk` the PDE route reports ≈ −price on the reference-date pillar (whose discount factor is 1 by construction) and a small offsetting term on the next pillar, while the analytic and tree routes report 0 there. Parallel DV01 and t > 0 pillar deltas agree across engines (`examples/21`), but raw DF deltas bucketed per pillar differ by engine on that pillar until the PDE reads the curve without the t = 0 normalisation
- [ ] **Stochastic tenor basis** — a genuine multi-curve HJM / Mercurio extension with its own driver per tenor curve; needed for basis *vega* (basis swaptions, tenor-spread options), which the deterministic overlay prices at zero

**Why:** Short-rate models are the backbone of rates desks for Bermudan swaptions, callable bonds, and path-dependent IR exotics. LMM alone is not enough.

**Approach:** All as `eqx.Module` with drift/diffusion for diffrax. Hull-White has closed-form bond prices (Jamshidian) — implement both analytic and MC pricing. Calibrate to ATM swaption vols via `optimistix`.

### 2.2 Heston Semi-Analytic Pricing

- [x] **Characteristic function** for Heston model — Lord-Kahl "Little Trap" form in `valax/pricing/analytic/heston.py`, branch-cut-safe for all expiries and admissible `rho`
- [x] **COS method** (Fang-Oosterlee) for option pricing from characteristic function — direct call/put payoff coefficients, cumulant-based truncation
- [ ] **Fourier inversion** (Carr-Madan / Lewis) as alternative
- [x] **Calibration to vol surface** using semi-analytic pricing — `calibrate_heston` now drives a real COS pricer in microseconds per evaluation; LM recovers all 5 parameters to floating-point precision on noise-free synthetic data

**Why:** MC-only Heston is too slow for calibration and real-time pricing. Semi-analytic methods give prices in microseconds.

**Status:** Shipped (this session). `heston_cos_price(option, spot, rate, dividend, model, *, N=160, L=12.0)` in `valax/pricing/analytic/heston.py`, re-exported from `valax.pricing.analytic`. Agrees with QuantLib's `AnalyticHestonEngine` to < 5e-7 absolute across the synthetic-market seed × moneyness × call/put grid (50/50 passing) and the Andersen-QE MC sits inside 3 standard errors of COS at every strike. See test suite at `tests/test_pricing/test_heston_cos.py` and `tests/test_quantlib_comparison/test_heston_ql.py::TestHestonCOSvsQL`. Carr-Madan/Lewis remain as a follow-up for parity-style benchmarking.

**Approach:** Implement the Heston characteristic function as a pure JAX function. COS method is matrix operations — naturally JAX-friendly. `vmap` over strikes for the full smile in one call.

### 2.3 Local Volatility

- [x] **Dupire local vol** extraction from implied vol surface — shipped in `valax/pricing/analytic/dupire.py` (Gatheral IV-space form, autodiff through `SVIVolSurface`).
- [x] **Local vol MC simulation** — shipped in `valax/pricing/mc/local_vol_paths.py` (`jax.lax.scan` + log-Euler with midpoint-in-time σ). Hooked into the unified MC dispatcher for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap`.
- [x] **Local vol PDE** pricing — European under `LocalVolModel` via 1-D Crank-Nicolson (`pde_price_dispatch`)

**Why:** Local vol is the standard model for exotic equity derivatives. It matches the entire vol surface by construction.

**Approach:** Dupire formula involves derivatives of the implied vol surface — autodiff through the surface pytree. Simulation via `lax.scan` with state-dependent diffusion.

### 2.4 Stochastic-Local Volatility (SLV)

- [x] **Leverage function** calibration — shipped in `valax/calibration/slv.py`: each pass is a true Guyon-Henry-Labordère sweep (row *i* built from the particles at *t_i*, which are then stepped with it, with the pricer's midpoint-in-time query; `n_substeps` particle steps per row), optional kernel ridge in pseudo-particle units (default 30), outer passes via `n_iterations` (default 2) measuring the estimator's Monte Carlo noise. Reported as a **fixed-point level** (`problem="fixed_point"`, last increment as residual, `clip_fraction` extra); see [SLV § fixed point](guide/slv.md#how-the-fixed-point-is-reported). The Heston backbone is controllable: `calibrate_heston(fixed_params=..., expiry=<per quote>)`, `calibrate_slv(heston=...)`
- [x] **Guarded local vol** — `valax.pricing.analytic.dupire.guarded_local_vol` / `DEFAULT_LOCAL_VOL_GUARD`, used by local-vol Monte Carlo and the SLV target: exact Dupire inside the domain, clamped moneyness, implied-variance fallback and a variance band outside it, finite gradients throughout
- [x] **SLV MC simulation** — shipped in `valax/pricing/mc/slv_paths.py` (Andersen-QE variance leg + log-Euler/Milstein log-spot leg, midpoint-in-time leverage query, approximate-correlation QE coupling). Registered with the unified MC dispatcher for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap`.
- [x] **Particle / kernel regression** — both estimators available via the `method=` kwarg on `calibrate_slv_leverage`.

**Why:** SLV is the industry standard for exotic equity pricing. It combines the smile-matching of local vol with realistic dynamics of stochastic vol.

**Approach:** Two-pass calibration: (1) calibrate Heston to vanillas, (2) compute leverage function via MC particle method. Leverage function stored as a 2D grid pytree (`valax/surfaces/leverage.py::LeverageGrid`).

**Open follow-up.** Sub-100 bp Dupire-consistency accuracy requires Fokker-Planck PDE calibration (cf. QuantLib's `HestonSLVProcess`); the particle-method ceiling at moderate budgets is ~100-250 bp. Tracked as an open follow-up.

### 2.6 LIBOR Market Model (LMM / BGM)

- [x] **Model pytree** — `LMMModel` with `PiecewiseConstantVol` / `RebonatoVol`
      abcd parametric volatility and `ExponentialCorrelation` /
      `TwoParameterCorrelation` correlation structures (`valax/models/lmm.py`)
- [x] **PCA factor reduction** — `compute_loading_matrix` (full-rank Cholesky,
      or the top-k PCA loading matrix for a reduced-factor simulation)
- [x] **Path generation** — `generate_lmm_paths` (`valax/pricing/mc/lmm_paths.py`)
- [x] **MC recipes** — caplet, cap, and European swaption under `LMMModel`;
      curve-free model, priced through `price()` with the periods mapped
      from the dates ([Phase 4, D18](#unified-calibration-pricing-and-risk))
- [x] **Bermudan swaption via Longstaff-Schwartz** — `bermudan_swaption_lsm`
      (see [Tier 3.1](#31-bermudan-swaptions))
- [ ] **Calibration — the headline gap.** There is no `calibrate_lmm`. LMM is
      the only model in the library with no calibrator, so its parameters must
      be supplied by hand and it cannot be fitted to market. **Its input now
      exists**: the caplet-stripping prerequisite shipped
      ([Multi-Curve Framework §3](#3-vol-surface-construction-is-curve-free-closed-for-capsfloors-and-swaptions)),
      so `strip_optionlet_surface` produces the per-expiry optionlet surface
      the calibrator has to match. What is left is the fit itself — the
      `RebonatoVol` abcd surface and the correlation parameters against that
      surface (and, for the swap-rate leg, a swaption cube, whose own
      dual-curve calibration is still open). The fit should be a level built
      with `solve_level` / `basket_fit_level` and `SolverSettings`, as every
      other calibrator is — see the
      [calibration guide](guide/calibration.md#solvers)
- [ ] **Dual-curve recipes** — the LMM recipes do not declare
      `discount_curve` / `forward_curve`, so they remain single-curve; see
      [Multi-Curve Framework §7](#7-modelling)
- [ ] **Lognormal only** — no shifted or stochastic-volatility (SABR-LMM)
      variant, so LMM cannot carry rate skew; skew lives only in the static
      SABR smile. The dynamic-skew alternatives that *did* ship are the
      Cheyette models ([Tier 2.1](#21-short-rate-models))

**Why:** LMM models observable forward rates directly and is the market
standard for callable rate exotics where the full forward-curve dynamics
matter. Without a calibrator it is currently a simulation engine rather than a
pricing model.

### 2.5 Jump-Diffusion Models

- [ ] **Merton jump-diffusion** — log-normal jumps added to GBM
- [ ] **Kou double-exponential** — asymmetric jump sizes
- [ ] **Bates** — Heston + Merton jumps (stochastic vol with jumps)
- [ ] **Semi-analytic pricing** via characteristic functions for all three

**Why:** Jumps explain short-dated smile steepness that diffusion models can't capture. Critical for short-dated FX and equity options.

**Approach:** Characteristic functions are straightforward extensions. MC simulation via diffrax with compound Poisson jump component.

---

## Tier 3: Products and Asset Classes

New instrument types and entire asset classes.

### 3.1 Bermudan Swaptions

- [x] **Hull-White PDE** with exercise as a pointwise projection on the value
      function — no regression, so no policy error
- [x] **Longstaff-Schwartz (LSM)** regression for early exercise on LMM paths — `bermudan_swaption_lsm`
- [x] **Per-trade Hull-White calibration** — `coterminal_swaptions(bermudan)` + `swaption_basket_node` fit the model to the trade's co-terminal basket off the calibrated cube (exactly, with one σ bucket per exercise date, in `mode="coterminal_bootstrap"`); the Bermudan then prices through `price(..., method="pde")` with quote risk via `chain_adjoint` (`examples/14_unified_rates_per_trade_model.py`; [§5.6](#unified-calibration-pricing-and-risk))
- [ ] **American Monte Carlo** with basis function selection (polynomials, neural network regressors)
- [ ] **Lower/upper bound** estimation (Andersen-Broadie)

**Why:** Bermudan swaptions are among the most traded IR derivatives. Every rates desk prices them daily.

**Approach:** Generate LMM paths (already have this). At each exercise date, regress continuation value on state variables. Use `jnp.polyval` or a small neural network for regression. The exercise boundary is differentiable for Greeks.

**Status (PR-3, delivered).** Under Hull-White the exercise decision needs no
regression at all: the tail-swap value is analytic at every node, so the PDE
projects exactly and the *only* numerical error is in the continuation value.
Validated against `ql.FdHullWhiteSwaptionEngine` (~2e-4 relative) and
`ql.TreeSwaptionEngine` (~1.2e-3), plus the structural check that a
single-exercise Bermudan reproduces the Jamshidian European.

### 3.2 Callable and Puttable Bonds

- [x] **Callable bond** pricing via backward induction on Hull-White tree/PDE
- [x] **OAS (option-adjusted spread)** calculation — `callable_bond_oas` (collapses to the Z-spread for option-free bonds)
- [x] **Effective duration and convexity** via autodiff on OAS — `effective_duration`, `effective_convexity`

**Why:** Callable bonds are a massive market (most corporate bonds are callable). OAS is the standard risk metric.

**Approach:** Requires Hull-White (Tier 2.1). Build a trinomial tree or use PDE with early exercise boundary. OAS is a shift to the discount curve — autodiff gives sensitivities.

**Status (PR-3, delivered).** Both the trinomial tree and the PDE price
callable/puttable bonds. The PDE is the more accurate of the two: it scales
each coupon by the analytic bond price from its snapped time level to its true
payment date, removing the O(dt) cashflow-timing error that limits the tree.
OAS, effective duration and effective convexity ship in `valax/risk/oas.py`.

### 3.3 FX Derivatives

- [x] **FX forward** pricing with domestic/foreign discount curves
- [x] **FX vanilla options** — Garman-Kohlhagen (modified Black-Scholes)
- [x] **FX smile conventions** — delta-based quoting (spot delta, forward delta, premium-adjusted delta), `strike_to_delta` and `delta_to_strike` conversion
- [ ] **FX barrier options** — continuous and discrete monitoring; Reiner-Rubinstein analytic and/or MC barrier pricer (`FXBarrierOption` pytree and FOR/DOM conventions defined)
- [ ] **Quanto options** — correlation between FX and underlying; blocked by the quanto drift adjustment (analytic, and as an MC recipe — correlated multi-asset paths already ship); `QuantoOption` pytree defined
- [ ] **TARFs (Target Accrual Range Forwards)** — path-dependent FX exotics via MC with early termination; `TARF` pytree defined
- [ ] **FX swap** pricing — near + far leg pair on the domestic / foreign curves and keyed FX spot (`FXSwap` pytree defined; today it is only a curve-bootstrap quote type)
- [ ] **FX Monte Carlo recipes** — `FXVanillaOption` / `FXBarrierOption` on `generate_gbm_paths` with drift \(r_d - r_f\) (no FX MC recipe ships)

**Why:** FX is one of the largest derivatives markets. FX options have unique smile conventions (delta-space, not strike-space).

**Approach:** Garman-Kohlhagen is a minor extension of Black-Scholes. FX smile needs delta-strike conversion utilities. TARFs via MC with diffrax.

**Status:** Core FX pricing implemented in `valax/pricing/analytic/fx.py` and instruments in `valax/instruments/fx.py`. Garman-Kohlhagen pricing, FX forward valuation, implied vol inversion, and all three delta conventions (spot, forward, premium-adjusted) with strike↔delta conversion. See the [Analytical Pricing guide](guide/analytical.md#fx-options-garman-kohlhagen). `FXForward` and `FXVanillaOption` price through `price()` on keyed domestic / foreign curves, FX spot and an FX vol surface ([Phase 4 close-out](#unified-calibration-pricing-and-risk); [recipe](guide/analytical.md#through-price)).

### 3.4 Credit Derivatives

- [x] **Survival curve** construction from CDS spreads — `valax.curves.survival` (`from_cds_spreads`, `SurvivalCurve`)
- [ ] **CDS pricing** — protection and premium leg valuation (instrument pytree defined; pricer pending). Prerequisite for XVA (CVA) and for a true repricing hazard bootstrap
- [x] **Piecewise-constant hazard curves** — `from_hazard_rates` builds a `SurvivalCurve` from a hazard term structure
- [ ] **Hazard rate bootstrapping** from CDS term structure — *not* implemented. `from_cds_spreads` applies the closed-form credit-triangle approximation `h = s / (1 - R)` (`valax/curves/survival.py`); it performs no root-find and does not reprice the CDS premium/protection legs, so it is not a bootstrap. A true bootstrap would route CDS quotes through `BootstrapInstrument` / `bootstrap_curve_graph` like the rates quotes
- [ ] **CDO tranche** pricing (Gaussian copula, base correlation) — blocked by copula simulation (the survival curve already ships); instrument pytree defined, pricer pending

**Why:** Credit derivatives are essential for bank trading books and CVA computation.

**Approach:** Survival curves as `eqx.Module` pytrees (analogous to discount curves). CDS pricing is cashflow discounting with survival probabilities. CDO requires copula simulation — JAX's PRNG system handles this well.

### 3.5 Inflation Derivatives

- [x] **Zero-coupon inflation swap** pricing — `zcis_price`, breakeven solver
- [x] **Year-on-year inflation swap** pricing — `yyis_price` (no convexity adjustment yet)
- [x] **Inflation cap/floor** — Black-76 on forward YoY inflation rate
- [ ] **Inflation cap/floor vol time to the fixing at period end**: `inflation_cap_floor_price_black76` (and its `price()` adapter) measures the Black-76 time `T` to the period *start*, while the YoY rate is fixed by CPI(Tᵢ) at the period *end*. A spot-starting cap's first caplet therefore gets no time value (T floored to 1e-10). Found by `examples/20`, which uses a forward-start cap to avoid it
- [ ] **CPI before the first pillar**: `forward_cpi` extrapolates flat *backwards* and `from_zc_rates` adds no reference-date pillar, so CPI(today) reads CPI(first pillar) rather than the base CPI, and a spot-start YoY swap or cap gets a zero first coupon. Anchor the curve at `(reference_date, base_cpi)`. `examples/20` adds the pillar by hand and asserts the bad behaviour
- [ ] **`index_lag` honoured**: the field is declared on all three inflation instruments but no kernel reads it
- [ ] **Seasonality adjustment** for monthly CPI
- [ ] **YoY convexity adjustment** (Jarrow-Yildirim)
- [ ] **Jarrow-Yildirim three-factor Monte Carlo** — the first inflation MC recipes (`YearOnYearInflationSwap`, `InflationCapFloor`); no inflation MC ships

**Why:** Inflation markets have grown significantly. Real-money investors use inflation swaps for liability hedging.

**Approach:** Inflation curve as a pytree mapping dates to forward CPI levels. Pricing is standard cashflow discounting with CPI ratios.

### 3.6 Equity Exotics

- [ ] **Autocallable / phoenix notes** — path-dependent early redemption; blocked by a multi-observation structured-payoff MC recipe (SLV and multi-asset paths already ship)
- [ ] **Cliquet / ratchet** — forward-starting structured-payoff MC recipe (LV / SLV already ship); `Cliquet` pytree defined
- [ ] **Chooser options** — BSM decomposition (choose call / put at a future date); `ChooserOption` pytree defined
- [ ] **American options by Monte Carlo** — lift the Longstaff-Schwartz engine of the LMM Bermudan (`bermudan_swaption_lsm`) to `AmericanOption` under BS / Heston / LV / SLV
- [ ] **Digital options by Monte Carlo** — wire `DigitalOption` recipes (smoothed-Heaviside payoff) for BS / Heston / LV / SLV
- [x] **Worst-of basket options** — multi-asset MC via `MultiAssetGBMModel`
- [x] **Lookback options** — MC floating and fixed strike
- [ ] **Touch / no-touch / one-touch** — digital barriers (cash-or-nothing `DigitalOption` ships; one-touch barriers pending)
- [x] **Range accruals** — Black-76 digital replication (rates `RangeAccrual`)
- [ ] **Compound options** — option on an option (Geske; pytree defined, pricer pending)
- [x] **Variance / volatility swaps** — analytic static replication + MC realized variance
- [ ] **Variance-swap MC annualisation from the time grid**: the MC recipes annualise realised variance with a fixed `annual_factor = 252` that `price()` cannot pass, so the fair strike is right only when `n_steps = 252·T` (at 126 steps over 1Y it is 2σ²). Annualise with `n_obs / T` instead. Demonstrated by `examples/22`
- [ ] **Per-asset vols for `BlackScholesModel` through `price()`**: the model has no `surface_id`, so `vol=None` reads only the `_default_` surface, and the error message wrongly says the market has no surface for the underlying even when `<ID>.VOL` exists. A multi-asset market must pass the BS vol explicitly, so the surface is not a declared entry and its vega is only a model greek (`examples/22`)
- [x] **Spread & digital options** — Margrabe / Kirk + MC (spread); analytic + PDE (digital)

**Why:** Structured products desks sell these daily. Autocallables alone are a multi-hundred-billion dollar market.

**Approach:** All priced via MC with appropriate payoff functions. Multi-asset requires correlated path generation (Cholesky on correlation matrix). Autocallables need path-wise barrier monitoring. SLV model (Tier 2.4) is the standard for pricing.

### 3.7 CMS and Spread Options

- [x] **CMS rate convexity adjustment** — `cms_convexity_adjustment`, Hagan (2003): closed-form quadratic approximation and static replication over the swaption smile (`valax/pricing/analytic/cms_convexity.py`)
- [x] **CMS cap/floor** pricing — `cms_cap_floor_price_black76`
- [x] **CMS spread options** — `CMSSpreadSwap` priced under two-factor G2++ / Cheyette (MC + PDE)
- [ ] **Range accrual on CMS** — structured coupon products
- [ ] **CMS and range-accrual Monte Carlo on LMM** — CMS swap / CMS cap-floor replication payoffs and full daily-observation range accrual on LMM paths (today these price only by their analytic kernels, outside `price()`)

**Why:** CMS products are actively traded in rates markets. The convexity adjustment is non-trivial and a classic quant problem.

**Approach:** CMS convexity adjustment via static replication (Hagan) or SABR integration. Spread options via 2D copula or Margrabe-like extensions.

### 3.8 Mortgage-Backed Securities and Convertible Bonds

- [ ] **MBS** — prepayment models (PSA, CPR), OAS, effective duration (very high complexity, US-market-specific; no pytree yet)
- [ ] **Convertible bonds** — equity-credit hybrid, 1-D PDE with credit (Tsiveriotis-Fernandes); `ConvertibleBond` pytree defined, pricer pending

**Why:** Large markets but specialized. Lower priority unless targeting specific desks.

**Approach:** MBS requires a prepayment model (empirical) and MC simulation on interest rate paths. Convertibles need a 1D PDE with credit and equity components.

---

## Tier 4: Risk and XVA

Portfolio-level risk management and valuation adjustments.

### 4.1 VaR and Expected Shortfall

- [x] **Historical VaR** — P&L distribution from historical scenarios
- [x] **Parametric VaR** — delta-normal using autodiff sensitivities and covariance matrix
- [x] **Monte Carlo VaR** — full revaluation under simulated scenarios (parametric + t-distribution)
- [x] **Expected Shortfall (CVaR)** — tail risk measure
- [ ] **Marginal / component / incremental VaR**

**Why:** Regulatory requirement (Basel III/IV). Every bank risk system computes these daily.

**Approach:** Scenario generation + batch repricing via `vmap`. Autodiff gives portfolio Greeks for parametric VaR. Full reval VaR benefits from JIT compilation and GPU acceleration — this is where JAX shines.

**Status:** Core framework implemented in `valax/risk/`. Parametric (normal/t) and historical scenario generation, curve shocks (parallel, steepener, flattener, butterfly, key-rate), full-revaluation VaR/ES via vmapped repricing. Parametric VaR via delta-normal with autodiff gradients. See the [Risk section overview](risk-overview.md).

### 4.2 Stress Testing and Scenario Analysis

- [x] **Scenario definition framework** — shifts to curves, vols, spots, correlations
- [x] **Identifier-keyed scenarios** — `KeyedScenario` + `valax.risk.shock` shock a `MarketState` by curve / surface / asset id (per-curve and basis moves; parallel vol shifts via `ShiftedVolSurface`); `keyed_from_market_scenario` lifts the legacy flat vectors
- [ ] **Historical stress scenarios** — replay historical crises (2008, COVID, SVB)
- [ ] **Reverse stress testing** — find scenarios that cause a target loss (optimization problem)
- [x] **P&L attribution** — second-order Taylor decomposition into delta (spot, vol, rate, div), gamma, and unexplained via autodiff

**Why:** Regulatory requirement and core risk management practice.

**Approach:** Scenarios as identifier-keyed pytree diffs (`KeyedScenario`) applied to a `MarketState` by `valax.risk.shock`; batch repricing via `vmap` through `price()`. P&L attribution via Taylor expansion using autodiff Greeks. Reverse stress testing via `optimistix.minimize`.

**Status:** Scenario framework implemented on the identifier-keyed `KeyedScenario` with `shock` over a `MarketState` (`valax/risk/state_shocks.py`) for per-curve, per-surface moves, plus keyed generators over a per-curve `KeyedFactorLayout` (`keyed_parametric_scenarios`, `keyed_historical_scenarios`, `keyed_stress_scenario`, `keyed_curve_stress`, `keyed_steepener` / `keyed_flattener` / `keyed_butterfly`, `keyed_basis_shift` in `valax/risk/scenarios.py`) and keyed waterfall attribution (`keyed_waterfall_pnl`). The positional `MarketScenario` / `ScenarioSet` + `apply_scenario` over a `MarketData` remain as the **legacy** path for one deprecation cycle; P&L attribution via `pnl_attribution()` (second-order Taylor) is still on that flat path. Historical crisis replay and reverse stress testing are next.

### 4.3 XVA Suite

- [ ] **CVA (Credit Valuation Adjustment)** — expected loss from counterparty default
- [ ] **DVA (Debit Valuation Adjustment)** — symmetric credit charge
- [ ] **FVA (Funding Valuation Adjustment)** — cost of funding uncollateralized exposure
- [ ] **MVA (Margin Valuation Adjustment)** — cost of posting initial margin
- [ ] **KVA (Capital Valuation Adjustment)** — cost of regulatory capital
- [ ] **Exposure simulation** — expected exposure (EE), potential future exposure (PFE), exposure at default (EAD)
- [ ] **Netting and collateral** — netting set aggregation, CSA modeling, margin period of risk

**Why:** XVA is now a first-class P&L item at every bank. XVA desks are among the most compute-intensive in banking.

**Approach:** Exposure simulation via MC (reuse existing path generators). At each simulation date, reprice the netting set (this is where GPU acceleration via JAX gives massive speedup). CVA = integral of discounted expected exposure * default probability. Autodiff gives XVA sensitivities.

---

## Tier 5: ML and Performance

Machine learning integration and computational scaling. For the *why* these matter — the JAX-native moat argument — see [Vision § 4](vision.md#4-frontier-uses-the-jax-native-moat); for *uncommitted* numerical-methods explorations behind §5.4, see [Research Ideas](research-ideas.md).

### 5.1 Neural Surrogate Pricers

- [ ] **Train neural networks** to approximate slow pricing functions (e.g., Bermudan swaption price as function of market state)
- [ ] **Differential ML** — use autodiff Greeks as training targets alongside prices
- [ ] **Online learning** — update surrogate as market moves

**Why:** XVA requires millions of nested pricings. Neural surrogates reduce compute by 100-1000x while preserving Greeks via autodiff through the network.

**Approach:** Standard neural network in equinox (already a JAX neural network library). Train on (market_state, price, greeks) triples. The surrogate is a pytree — `jax.grad` gives Greeks automatically.

### 5.2 Learned Volatility Surfaces

- [ ] **Neural SVI** — neural network parameterization of the vol surface that is arbitrage-free by construction
- [ ] **Variational autoencoder** for vol surface dynamics (generate realistic vol surface moves for risk simulation)

**Why:** Vol surface fitting is a daily challenge. Neural approaches can enforce arbitrage constraints more naturally than traditional parametric methods.

### 5.3 Reinforcement Learning for Hedging

- [ ] **Deep hedging** — RL agent learns hedging strategy that minimizes a risk measure (CVaR of P&L)
- [ ] **Transaction cost-aware** hedging
- [ ] **Model-free** — learns from historical data, no assumed dynamics

**Why:** Traditional delta hedging assumes continuous rebalancing and no transaction costs. Deep hedging handles realistic frictions.

**Approach:** Environment = simulated market paths. Agent = neural network policy. Train with JAX + optax. The entire training loop is differentiable.

### 5.4 Multi-Dimensional PDE Solvers

- [x] **ADI (Alternating Direction Implicit)** for 2D PDEs — **European Heston `(ln S, v)` delivered** (Douglas / Craig–Sneyd / Hundsdorfer–Verwer)
- [ ] **American / exotic under 2-D ADI** (early-exercise projection, barriers on the 2-D grid) — first target `AmericanOption` under `HestonModel`, cross-checked against MC / QuantLib FD
- [ ] **Local-stochastic vol (SLV) and two-asset** 2-D recipes — European `SLVModel` on `(S, v)`; `SpreadOption` on `(S₁, S₂)` cross-checked against Margrabe / Kirk
- [ ] **Sparse grid** methods for higher dimensions
- [ ] **Neural PDE solvers** (PINNs) for high-dimensional problems

**Why:** Current PDE solver is 1D only. Many important models (Heston, local-stochastic vol, multi-asset) require 2D+ PDE solvers.

**Approach:** ADI scheme decomposes 2D operator into 1D sweeps — reuse existing tridiagonal solver via lineax. Sparse grids reduce curse of dimensionality. The high-order / spectral-element / FEM alternatives and the accuracy-vs-DOF tradeoff are analysed in [Research Ideas → multi-dimensional PDE](research-ideas.md).

**Status (PR-2, delivered).** European options under `HestonModel` price via a 2-D
ADI finite-difference solver (`valax/pricing/pde/{operators2d,schemes2d}.py`),
registered on `pde_price_dispatch` through `PDEConfig2D`. Each implicit stage is a
`vmap`-batched tridiagonal solve per axis (reusing `linalg.py`); the mixed term is
explicit; the degenerate `v = 0` boundary is a drift-only upwind row (Feller-robust)
and `v = v_max` a linearity row. Validated against the COS oracle (`rel < 1e-3`
across strikes), Andersen-QE MC, and QuantLib `AnalyticHestonEngine` /
`FdHestonVanillaEngine`. A subtle mixed-term boundary bug found during this work is
written up in [Numerical Pitfalls](architecture/numerical-pitfalls.md).

**Status (PR-3, delivered).** The 1-D substrate now also carries the
**Hull-White short-rate PDE** (`valax/pricing/pde/hull_white.py`): fixed-rate,
callable and puttable bonds, European swaptions, and **Bermudan swaptions**.
Solved in the centred state `x` of `r = x + alpha(t)`, with a zero-curvature
edge treatment (no closed-form far field exists for a callable), an exactly
*integrated* `alpha` (midpoint sampling stalls convergence on any curve with
pillar kinks), and a discrete-event seam on the backward sweep for coupons and
exercise. Validated against the analytic curve price, Jamshidian, the HW tree,
and QuantLib's FD/tree engines — see
[Hull-White Finite Differences](theory/hull-white-pde.md). Three discretisation
bugs surfaced and were fixed in the process — a time-reversed boundary sampling
in *both* the 1-D and 2-D steppers, a stalled convergence from midpoint-sampling
a kinked curve, and an O(dt) cashflow-snapping error — all written up as
entries 2-4 of [Numerical Pitfalls](architecture/numerical-pitfalls.md). The same
short-rate substrate now also carries the **two-factor G2++** (2-D) and
**Cheyette** (1F/2F) PDEs — covering CMS-spread decorrelation and rate skew,
alongside fixed-rate bonds, caplets/caps, and European/Bermudan swaptions.

**Status (ontology P5.1, delivered).** The Hull-White and G2++ PDE *and* MC
recipes are now **multi-curve**: they accept the trade's resolved
`discount_curve` (honoured by re-fitting the exact-fit model to it) and
`forward_curve` (a frozen deterministic tenor basis), both supplied
automatically by `valax.core.price` from the instrument's `discount_id` /
`forward_id`. The overlay lives in `valax/pricing/multicurve.py`; see
[Multi-Curve Short Rates](theory/multi-curve-short-rate.md) for the derivation
and its limits (no basis volatility).

### 5.5 Portfolio and Multi-Device Scaling

Scaling the engine from a single instrument to a whole book and across devices —
see the [Scaling Up](scaling/index.md) section for the full design.

- [x] **Homogeneous batch pricing / Greeks** — `jax.vmap` over a batched instrument pytree (`valax/portfolio/batch.py`; `risk/sensitivities.py`, `risk/var.py`)
- [x] **Book-level aggregation & bucketing** — `FlatSensitivities`, `BucketMap`, PCA factor model (`valax/risk/{sensitivities,bucketing,factors}.py`)
- [ ] **`Portfolio` container** — heterogeneous trade collection with per-trade pricing keys (design in [Productionisation](architecture/production.md))
- [ ] **Heterogeneous dispatch-and-aggregate** — group a mixed book by `(instrument, model)`, price each group under one JIT kernel via the MC/PDE registry, aggregate to book value / Greeks
- [ ] **Shape-stable / padded batching** — fixed-shape padding so the JIT cache is reused across differently-sized books
- [ ] **Multi-device parallelism** — shard the instrument / scenario axis across GPUs/TPUs via `jax.sharding` `Mesh` / `shard_map` (today: single-device XLA + `vmap` only)
- [ ] **GPU/TPU benchmark suite** — published GPU-vs-CPU throughput (roadmap P5.2)
- [ ] **Unified market-quote container + quote-space (hedge-space) risk** — one container spanning all rate / FX / vol / credit quotes, with PV differentiated end-to-end through calibration so book Greeks/vega come out as hedge-ratio sensitivities to tradeable quotes. **Container shipped** (ontology P2): `valax.core.MarketState` — keyed `CurveGraph` + vol surfaces + spots + fixings + credit/inflation/FX — together with the unified `valax.core.price(instrument, model, market, method=...)` signature and `valax.core.resolve`. **Curve leg shipped** (P4): `valax.greeks.adjoint.curve_quote_deltas` differentiates PV end-to-end through the curve bootstrap by the implicit function theorem, with `CurveAdjoint` reusing one factorisation across a portfolio. **Volatility leg shipped**: `calibrate_swaption_cube_from_curves` derives each node's forward and annuity from the curves, so `dθ_v/dθ_c` is no longer identically zero, and `valax.greeks.vol_adjoint` (`sabr_slice_level` → `vol_quote_vegas`) gives `λ_v` — vega against vol quotes, plus the `dR_v/dθ_c` curve contribution, both FD-validated. **Fixed inputs shipped**: `valax.greeks.fixed_inputs` reports `p̄` separately, and `valax.market.SnapshotProvenance` labels the resulting vectors. **Quote container and model leg shipped**: `valax.market.QuoteSet` keys every quote and fixed input, and `valax.greeks.quote_risk.chain_adjoint` sweeps curves → vols → model generically (see §5.6) — per underlying (`price_fit_node`) and per trade (`swaption_basket_node`, Hull-White on a co-terminal basket off the cube), with the rates vol objects as chain nodes (`swaption_cube_node`, `caplet_strip_node` → `optionlet_sabr_node`). **Still missing:** FX and credit quote levels, and the position / portfolio layer that would carry a heterogeneous book through one sweep. See the [Audit Conformance Map](architecture/aad-audit-conformance.md) for the requirement-by-requirement status.

**Why:** the JAX-native design makes book- and business-scale a composition
problem (`vmap` / `jit` / `grad` stack over pure pytree pricers) rather than a
rewrite. Single-device batching ships today; the heterogeneous book engine and
multi-device sharding are the remaining pieces.

### 5.6 Unified Calibration, Pricing and Risk {#unified-calibration-pricing-and-risk}

One contract for every calibration and one sweep from price to quotes — see
[Calibration, Pricing & Risk Architecture](architecture/calibration-pricing-risk.md)
and the maths in [Calibration Chains](theory/calibration-chains.md). What each
calibration fits, holds fixed and assumes, and its known limitations, are in
the [Calibration Assumptions Register](applications/calibration-register.md),
whose "Improve" rows point at the open items below.

- [x] **Keyed quote container** — `valax.market.QuoteSet`: identifier-keyed quotes `q` and fixed inputs `p`, differentiable and vmappable
- [x] **Solver settings + traceable convergence** — `valax.calibration.SolverSettings` / `run_solver` (one solver factory), `CalibrationResult` with a JAX-boolean `converged` flag and eager `check()`
- [x] **Generic calibration level** — `valax.calibration.level` (`CalibrationLevel`, `LevelInputs`, `level_adjoint`) and the level builders in `valax.calibration.solve`: `solve_level` (one solve), `solve_batched_level` (one independent solve per node in one `vmap`, per-node diagnostics, failures named by label such as `"5Yx10Y"`, `"cap 3Y"`, `"K=3.00%"`; block-diagonal level residual) and `explicit_level` (explicit maps and fixed points), with `stack_slices` / `period_label(s)` helpers
- [x] **Calibration chains** — `CalibrationNode`, `CalibrationChain`, `calibrate_market` (traceable: `jit` / `grad` / `vmap` over quote scenarios), `calibrate_model` (model as a separate input)
- [x] **Chain nodes: explicit inputs, curves, SABR smiles, any model through `price`** — `market_inputs_node`, `curve_graph_node`, `sabr_smile_node`, `price_fit_node` (`valax.calibration.nodes`)
- [x] **Constant-parameter market as a chain** — `MarketState.flat` / `market_inputs_node`; quote risk reproduces textbook Black-Scholes Greeks
- [x] **Two differentiation paths + reconciliation** — `valax.greeks.quote_risk`: `chain_adjoint` / `quote_risk_adjoint` (explicit IFT sweep, book-linear) and `quote_risk_autodiff` (implicit adjoint through the solves), `reconcile_quote_risk`
- [x] **Model leg (per underlying)** — `price_fit_node` + `calibrate_model` gives `λ_m`, folded into `chain_adjoint` via `model_levels` (Heston validated against the autodiff path)
- [x] **Synthetic quote universes** — `valax.market.synthetic.universe`: true multi-curve / equity markets quoted with the library's own residuals and pricers (`valax.curves.par_quote`), Gaussian noise, matching chain nodes, quote scenarios and product samplers; noise-free snapshots recalibrate to the truth
- [x] **Rates-vol synthetic universe** — `RatesVolSpec` / `RatesVolWorld` / `RatesVolQuoteSet`, `sample_rates_vol_world`, `synthesize_rates_vol_quote_set`, `rates_vol_snapshot`: a true SABR swaption cube (normal vols, shifted normal SABR with β = 0, one ATM level ±5% per cell) and a true caplet strip (piecewise constant per bucket), quoted on the synthetic curves with the matching chain (curves → cube → strip → optionlet SABR). Noise-free quotes recover the curves to ≤ 1e-12, cube α to ≤ 1e-10, ρ / ν to ≤ 1e-8 and the strip to ≤ 1e-12 (`tests/test_calibration/test_rates_vol_nodes.py::TestTruthRecovery`); the optionlet SABR is a smoothing of the strip, not a recoverable truth
- [x] **End-to-end examples** — `examples/10`–`13` (market tiers, multi-curve book risk, equity smile + Heston + Monte Carlo exotic risk, vmapped quote scenarios with P&L explain), self-checking and run by `tests/test_examples/`
- [x] **Example 14 — rates per-trade model** — `examples/14_unified_rates_per_trade_model.py`: noisy curves + cube → per-trade Hull-White σ(t) by the co-terminal bootstrap (batched with `eqx.filter_vmap`; basket repriced to ≤ 1e-13, against 2.8% for a constant-σ least-squares fit shown alongside) → Bermudans through `price` (PDE) → quote risk by `λ_m` vs autodiff, with the vega profile across each basket, one entry per exercise date; self-checking (total cube vega equals the summed basket profile)
- [x] **Examples 15–18 — rates vol chain, diagnostics, SLV, per-trade G2++ / Cheyette** — `examples/15_unified_rates_caps_optionlet_chain.py` (curves → `caplet_strip_node` → `optionlet_sabr_node` → caps / floors through `price` → quote risk vs autodiff and full-recalibration FD), `16_unified_calibration_diagnostics.py` (bad quotes named by `check()` / `failed_labels()`, healthy nodes isolated, `condition_alert`, `check_engine` refusing a frozen-y Cheyette fit), `17_unified_slv_fixed_point.py` (leverage fixed point swept over `n_iterations`, LV / Heston / SLV barrier, risk by AD through the unrolled passes vs FD), `18_unified_rates_per_trade_g2pp_cheyette.py` (per-trade G2++ and Cheyette σ(t) by the co-terminal bootstrap with a, b, ρ, η/σ and κ, β held as fixed inputs; exact vs engine-consistent G2++ bootstrap; model risk vs Hull-White with every model repricing the co-terminals exactly); all self-checking and run by `tests/test_examples/`
- [x] **Example 19 — a rates book end to end** — `examples/19_unified_book_workflow.py`: synthetic OIS + 3M quotes → joint bootstrap → swaps, fixed-rate bonds and an FRN through `price()` (policy shown per trade) → `market_risk` per trade (Σ = whole-book gradient; bucketed zero-rate DV01s checked against a ±1 bp `shock`) → the bond kernels (`fixed_rate_bond_price` equals the routed price; YTM / duration / convexity explain a 25 bp move) → `fit_rates_pca` on a synthetic curve history → PC and basis `KeyedScenario`s revalued under `vmap` with P&L explain, PC-score sensitivities by `jax.grad`, VaR / ES; self-checking and run by `tests/test_examples/`
- [x] **Examples 20–22: FX / inflation / cross-currency, the rates engine ladder, multi-asset and path-dependent exotics**:
  - `examples/20_unified_fx_inflation_xccy.py`: FX forward and option, cross-currency, 3M-vs-6M multi-leg, ZCIS, YoY and cap / floor on one multi-currency market. Checked by covered interest parity, put–call parity, the leg sum and cap − floor = YoY; `market_risk` exactly zero outside declared entries and decoys.
  - `21_unified_rates_engine_ladder.py`: one European and one Bermudan swaption under Hull-White / G2++ / Cheyette 1F / 2F / LMM by every registered engine. Within-model agreement to measured tolerances against the closed forms, Bermudan ≥ European, the `auto` policy and its errors, curve deltas agreeing across engines.
  - `22_unified_multi_asset_exotics.py`: spread and worst-of under `MultiAssetGBMModel` vs Margrabe / Kirk; a barrier / Asian / lookback / variance-swap ladder under BS / Heston / LV / SLV with exact anchors; spot risk; a dual digital composed on the path kernel.

  All self-checking and run by `tests/test_examples/` (21 and 22 are `slow`). The route limitations they surfaced are open items in [§2.1](#21-short-rate-models), [§3.5](#35-inflation-derivatives) and [§3.6](#36-equity-exotics).
- [x] **Retire the `MarketData`-era examples** — `examples/07_synthetic_market.py`, `08_end_to_end_workflow.py`, `09_pca_rates_pnl.py` and `comparisons/08_risk_greeks_var.py` are removed: they were built on `MarketData` / `MarketScenario` / the legacy `valax/risk/var.py`, which are scheduled for removal. Successors: examples 10–13 and 19 (mapping in `docs/examples.md` § Retired examples); the calibration-residual loop of 08 stays tested in `tests/test_market/test_calibration_residual.py`. The kernel-layer examples 01–06, `comparisons/` and `pitfalls/` are kept on purpose: the kernels are public API alongside `price()`, not legacy
- [ ] **Richer synthetic noise** — bid/ask pairs as fit weights, correlated noise across a strip, stale / missing quotes
- [x] **Migrate the per-model calibrators** — every calibrator returns `CalibrationResult` and takes `settings: SolverSettings` (`EXACT_SETTINGS`, `FIT_SETTINGS`; `BASKET_SETTINGS` = BFGS to 1e-12 for Hull-White / G2++, whose closed forms nest a root-find with a non-finite implicit derivative far from the solution; `CHEYETTE_SETTINGS` = BFGS to 1e-10). `calibrate_sabr` / `calibrate_heston` / `calibrate_svi_slice` are `solve_level` wrappers; `calibrate_hull_white` / `calibrate_g2pp` / `calibrate_cheyette*` wrap `basket_fit_level` (curve-free models); the SABR / SVI surface, optionlet surface and swaption cube loops are batched levels (`sabr_batch_level`, `solve_batched_level`, `swaption_cube_level`); `strip_caplet_vols` is solved bucket by bucket but packaged as one lower-triangular root level (`caplet_strip_level`, strike columns batched by `strip_columns_level`); `strip_optionlet_surface` returns the SABR level with the strip as a child; `calibrate_slv*` are fixed-point levels. No string solver selection remains, there is one return type, and the six unchecked `throw=False` solves (Hull-White ×2, G2++ ×2, Cheyette `_run_solver` ×2) plus those in `strip_caplet_vols` / `swaption_implied_normal_vol` and the diagnostics discarded by the surface loops and curve wrappers now report traceable convergence. Migration evidence (old vs new): SABR LM bit-identical, SABR BFGS ≤ 1e-9 in parameters, Heston (COS) bit-identical, Hull-White ≤ 1e-14 relative in price, G2++ ≤ 7e-15, Cheyette frozen-`y` ≤ 1.2e-12 relative in price, caplet strip ≤ 3e-16 in vols, batched surfaces equal to per-slice fits to 1e-12
- [x] **Curve bootstrap as a calibration level** — `bootstrap_curve_graph` returns the same root level as `curve_graph_node` (labels name each instrument); `CurveBuildDiagnostics` is retired (fitted quotes via `fitted_curve_quotes`, conditioning via `result.condition_number()`), and `quote_jacobian` raises `CalibrationError` on an unconverged build
- [x] **Engine provenance** — `CalibrationProvenance(method, engine_kind ∈ {exact, discretised, modelling_bias}, config)` is recorded on the result (metadata, not a model field, so models fitted on different engines stack under `vmap`); `check_engine(result, method, config)` refuses a mismatched engine only for `modelling_bias` (frozen-`y` Cheyette, `cheyette_provenance("frozen", cfg)`); exact-`y` Cheyette is `discretised`, Hull-White / G2++ closed-form fits are `CLOSED_FORM_PROVENANCE` (`exact`; `JAMSHIDIAN_PROVENANCE` is an alias). Nodes pricing through `price()` (`price_fit_node`, `swaption_basket_node`) record it via the `engine_provenance(model, method, config)` dispatch, so a per-trade fit on a PDE basket is `discretised` with its grid and a per-trade Cheyette fit is engine-bound. `price()` does not consult provenance — the check is an eager opt-in
- [x] **De-embedded short-rate curves** — `HullWhiteModel`, `G2PPModel`, `CheyetteModel(2F)` carry `initial_curve=None` and a static `curve_id`; `valax.core.binding.curve_requirements` (`singledispatch`, `{field: curve_id}`) is consumed by `bind_model` inside `price()`, which raises if a declared curve field arrives pre-bound (the market is the single source of truth for curve risk); kernels raise when unbound (`require_initial_curve`), `bind_curves` binds explicitly. Projection-curve applicability is the `basis_overlay(model)` dispatch (Hull-White / G2++ yes, Cheyette no — `price()` now raises where it previously priced single-curve silently). Gated by `tests/test_core/test_curve_binding.py`: the model gradient has no curve leaf, the market gradient equals the old embedded-curve delta to 1e-12, curve-free models stack under `filter_vmap`
- [x] **Rates vol levels as chain nodes** — `swaption_cube_node` (forwards recomputed inside the level from the upstream curves with the pricer's own `swaption_forward_and_annuity`, so `dθ_v/dθ_c ≠ 0` and curve → vol → price is one `chain_adjoint` sweep; shares `swaption_cube_level` with `calibrate_swaption_cube_from_curves`), `caplet_strip_node` (strike-batched strip on the upstream curves) and `optionlet_sabr_node` (SABR per optionlet expiry on the strip, no quotes of its own, so its adjoint flows to the cap quotes and curves). Validated in `tests/test_calibration/test_rates_vol_nodes.py`: adjoint = autodiff on every quote and fixed input (rtol 1e-6), both = full-recalibration FD on curve quotes, vol quotes and the fixed SABR β (rtol 2e-4), finite condition numbers below `ILL_CONDITIONED_THRESHOLD`, a NaN cell named. The frozen-forward cube fit (`dθ_v/dθ_c = 0`) is removed; `valax.greeks.vol_adjoint` (rebuilding levels from fitted objects) remains and is still validated, but the nodes supersede it for chain use
- [x] **Per-trade model basket** — `coterminal_swaptions(bermudan)` + `swaption_basket_node` (model scope): the model is fitted per trade to the basket's prices implied by the calibrated cube and curves, model prices through `price()` (the closed form by default: Jamshidian for Hull-White, Brigo-Mercurio for G2++); `λ_m` flows through `chain_adjoint` into the cube and curve levels; zero-valued target-shift fixed inputs report `∂V/∂T_k`, the vega profile across the basket; traced baskets batch under `eqx.filter_vmap`. Validated for **Hull-White** in `tests/test_calibration/test_per_trade_basket.py` (adjoint vs autodiff rtol 1e-6; FD on a curve quote, a cube vol quote, the held-fixed mean reversion and a target shift rtol 5e-4; vmap over two baskets = individual fits to 1e-10; Jamshidian via `price()` matches the PDE). Also validated for **G2++** (multi-curve; σ, η fitted with a, b, ρ fixed, on the co-terminals plus the 1Y-tenor swaption at the first exercise — the co-terminals alone leave the two factors unidentified; basket by the closed form, Bermudan by the 2-D PDE; `filter_vmap` = individual fits to rtol 5e-8; the engine-consistent alternative — the basket fitted on the Bermudan's own PDE grid, recorded `discretised` — is validated too, in the slow `TestG2PPOnEngine`) and **Cheyette** (single-curve only; σ, β fitted with κ fixed, on the co-terminals plus one off-strike option; frozen-y PDE, recorded engine-bound so `check_engine` refuses another grid): adjoint vs autodiff rtol 1e-6, full-recalibration FD on a curve quote, a vol quote, the fixed inputs and a target shift rtol 5e-4 (`TestG2PPPerTrade`, `TestCheyettePerTrade`). This is the `mode="least_squares"` default (constant parameters, a compromise on a co-terminal strip); the exact co-terminal bootstrap with piecewise σ(t) is the `mode="coterminal_bootstrap"` item below, which examples 14 and 18 now use
- [x] **Calibration Assumptions Register** — [`applications/calibration-register.md`](applications/calibration-register.md): one entry per calibration (fits / holds fixed, targets and objective, engine and provenance, settings, market inputs, assumptions, limitations with evidence, validation, improvement path); every public calibration entry point must have an entry (`tests/test_docs/test_calibration_register.py`)
- [x] **Failure diagnosis that points at the cause** — batched levels with a sub-node axis (`solve_batched_level(sub_labels=...)`; a strip failure is `"K=3.00% cap 2Y"`), cap buckets labelled by maturity to match their quote keys, failed strip buckets install NaN like failed cube cells, sequential solves name their root cause (`root_cause_labels()`), joint root levels localise to non-finite quotes → the worst residual cluster (`FAILURE_CLUSTER_FRACTION`) → starting-point residuals, and any non-finite residual is not converged (`examples/16`; `tests/test_curves/test_bootstrap_graph.py::TestFailureLocalisation`, `tests/test_surfaces/test_caplet_stripping.py`)
- [x] **Tracer-free level inputs** — `LevelInputs.data` carries trade data (a traced basket) as pytree leaves instead of a closure, so results returned from `jit` / `vmap` stay usable eagerly (`condition_number()`); used by `swaption_basket_node`, `price_fit_node` and `basket_fit_level` (Hull-White, G2++, Cheyette calibrators)
- [x] **Callable negative convexity as an autodiff Greek** — a consequence of the smooth payoffs below: `effective_convexity` of a callable is negative near the call boundary (≈ −229 at the money, ≈ −4.2 deep in the money on the 10Y test callable, finite-difference-confirmed), resolving the research item that had attributed a "positive floor" to Hull-White when it was the pointwise grid (`tests/test_risk/test_oas.py`, [Theory § 7.9](theory/risk-measures.md#79-option-adjusted-spread-and-z-spread))
- [x] **Smooth short-rate PDE payoffs** — `valax.pricing.pde.payoff_smoothing`, `PDEConfig(2D).smooth_payoff=True` by default: terminal payoffs and exercise / call / put projections sampled as the cell average of the positive part (exact for a locally linear argument in 1-D, Pooley–Vetzal–Forsyth 2003; a bias-neutral C¹ kernel in 2-D), so Hull-White / G2++ / Cheyette prices are C¹ in their inputs, Hull-White converges cleanly at second order, and full-recalibration FD of Bermudan risk works at ordinary steps; see [PDE guide](guide/pde.md)
- [x] **Time-dependent short-rate volatility and the co-terminal bootstrap** (Phase 4, first modelling item) — piecewise-constant σ(t) for Hull-White, G2++ and Cheyette 1F/2F ([Tier 2.1](#21-short-rate-models)) and `swaption_basket_node(mode="coterminal_bootstrap")` (modes in `BASKET_MODES`): the single transformed volatility field gets **one bucket per co-terminal** with breakpoints at the basket expiries (traced, so per-trade baskets with different dates batch under `filter_vmap`), every other bucketed field held and broadcast (G2++ / Cheyette 2F volatility ratio, Cheyette β as fixed inputs, their sensitivities reported). The level is **square and lower-triangular**, a `root` level built by `triangular_root_level`: one scalar Newton solve per bucket (1e-12, relative price residual), each seeded at its co-terminal's cube normal vol, with per-bucket diagnostics — a failed bucket installs NaN and is named as the root cause while the buckets before it stay exact. Every co-terminal is repriced to ≤ 1e-13 (closed form) or to the solver tolerance on its own grid (PDE), λ_m has one entry per exercise date, and adjoint = autodiff (rtol 1e-6) = full-recalibration FD (rtol 5e-4) on curve quotes, cube vols, held parameters and target shifts, for Hull-White (closed form), G2++ (closed form), Cheyette 1F (frozen-y PDE, engine-bound) and Cheyette 2F (frozen-y 2-D PDE) (`tests/test_calibration/test_coterminal_bootstrap.py`). A joint solver (one Newton solve with a lower-triangular `lineax` tag) was built, measured and removed: 2–7× more pricings, slower compiles, and on an infeasible bucket it marked every bucket failed under the wrong root cause ([D14](architecture/calibration-pricing-risk.md#3-decisions); `scripts/bench_coterminal_bootstrap.py`). Examples 14 and 18 now bootstrap (basket misses 1–3% → ≤ 1e-13). Assumptions and limitations: [register](applications/calibration-register.md#per-trade-co-terminal-bootstrap); maths: [Calibration Chains § I.10](theory/calibration-chains.md#i10-per-trade-model-levels-and-the-vega-profile)
- [x] **Short-rate Monte Carlo time grid** (Phase 4, D15) — the Hull-White, G2++ and Cheyette 1F/2F MC recipes in `valax/pricing/mc/recipes.py` read the step count from `MCConfig.n_steps` (it was a recipe keyword, default 100, that `price()` could not pass, so `MCConfig(n_steps=...)` was silently ignored). The keyword survives only as an explicit override for direct calls; the default `MCConfig(100_000, 100)` reproduces the former 100-step prices bit for bit, and 4 vs 400 steps now differ (`tests/test_mc/test_short_rate_mc_steps.py`). A bias study found no discretisation bias at 6–400 steps (|z| < 1 over 8 × 200k paths); the Hull-White MC tests pool eight keys before their 2σ check
- [ ] **Event dates on the short-rate MC grid** — the grid is uniform on `[0, T]` and coupon / exercise dates snap to the nearest step; placing them on the grid would remove the snapping error at small `n_steps`. (The LMM grid already contains every tenor date and takes its steps from `MCConfig.n_steps`, D18.)
- [x] **Multi-index curve binding** (Phase 4, [D16](architecture/calibration-pricing-risk.md#3-decisions)) — `valax.core.resolve`: `ResolvedMarket.curves` holds every curve a trade names (`curve_ids_of`: the primary pair plus `curves_touched`), keyed by id, with `discount_curve` / `forward_curve` as derived views (bit-identical for single-index trades); the projection curve is never guessed for three or more curves; selection reads static fields only, so every route works with the trade traced under `jit` / `grad` (`tests/test_core/test_keyed_resolution.py`). `curve_requirements` values may be a mapping `{key: curve_id}`, binding a dict of curves into one model field. MC / PDE recipes opt into `curves` / `spots` by name, and `price()` refuses a trade touching curves its recipe cannot take. `indices_touched` names fixing histories only
- [x] **LMM curve de-embedding** (Phase 4, [D18](architecture/calibration-pricing-risk.md#3-decisions)) — `LMMModel(tenor_dates, vol_structure, corr_structure, initial_curve=None, curve_id=None)`: the initial forwards, accruals, tenor times, `P(0, T_0)` and factor loadings are properties of the bound curve (the loadings now follow `corr_structure`, so correlation gradients reach the paths); `curve_requirements(LMMModel)` binds the pinned id, else the trade's discount curve; `price()` refuses a pre-bound model; `build_lmm_model(curve, …)` returns the bound model for kernel calls, bit-identical paths and prices (`tests/test_models/test_lmm.py`)
- [ ] **Constrained (KKT) and FX smile levels** — arbitrage-free SVI, the FX delta system
- [x] **Explicit pricing policy and default configs** (Phase 4, workstream 2a, [D15](architecture/calibration-pricing-risk.md#3-decisions)) — `valax.core.policy` states the `auto` method of every registered pair (`AUTO_METHODS`; the global analytic → lattice → pde → mc fallback is removed, so `auto` raises for a pair without a policy) and a default config for each numerical method (`set_default_config`, `default_config`, `effective_config`; PDE grids, the Cheyette calibrator's frozen-y grids, `MCConfig(100_000, 100)`, `TreeConfig()`), so `price(..., config=None)` works for every method. `PricingResult.config` and calibration provenance record the config actually used. The frozen map reproduces every former `auto` choice (`tests/test_core/test_pricing_policy.py`, pinned to the Phase 4 golden file)
- [x] **Rates routes added to `price`** (Phase 4, workstream 2a) — `ImpliedVolModel(surface_id)` prices `Caplet` / `Cap` / `Swaption` off a market vol source (cube, optionlet surface, flat `ConstantVol` normal or lognormal) by the `rates_smile` kernels, bit-identical (`tests/test_models/test_implied_vol.py`); `swaption_basket_node` prices its targets through it and example 15 uses it. The Hull-White tree (`callable_bond_price`, `puttable_bond_price`, `swaption_tree_price`, `bermudan_swaption_tree_price`) is routed under `method="lattice"` with `TreeConfig` (general tree, half-width `⌈width_factor · n_steps⌉`, factor 2, so it runs on a traced model; `auto` unchanged) (`tests/test_lattice/test_hull_white_tree_routes.py`). The coverage matrix's `price()` routes column — and, since the coverage unification, its four solver columns (Analytic / MC / PDE / Lattice: routed, kernel-only, planned) and each row's status — are checked against the registry both ways (`tests/test_docs/test_price_routes.py`)
- [x] **Multi-asset routes priceable through `price`** (Phase 4, D16) — `SpreadOption` / `WorstOfBasketOption` name their assets in `underlying_ids`; `price()` resolves the spots in that order and hands them to the MC recipes as `spots`, bit-identical to a direct `mc_price_dispatch` (`tests/test_mc/test_multi_asset_route.py`)
- [x] **LMM routes priceable through `price`** (Phase 4, D18) — `Caplet`, `Cap`, `Swaption`, `BermudanSwaption` × `LMMModel` (`mc`): the recipes in `valax/pricing/mc/recipes.py` map the trade's periods onto `tenor_dates` from the dates (`valax.pricing.mc.lmm_period_indices`; off-grid dates raise when concrete, NaN when traced; the trade's day count must be the curve's) and spread `MCConfig.n_steps` over the grid segments (`n_steps_per_period` is an override); each route equals the recipe with explicit indices bit for bit, curve risk equals the bound-model delta to 1e-12, caplets match Black-76 within 2 pooled stderr, a one-exercise Bermudan equals the European (`tests/test_mc/test_lmm_routes.py`); all four pass the route contract. Routing them fixed the European swaption payoff (forwards at expiry, not at their own fixings: +20% before; now within 0.6 stderr of Black at Rebonato's swap vol, `tests/test_mc/test_lmm.py`)
- [ ] **LMM grid from the trade** — dates must sit on the model's tenor grid, a caplet fixing lag is not modelled, and the trade's day count must equal the curve's; separate fixing / accrual conventions per period would lift all three
- [ ] **CMS swap, CMS cap / floor and range accrual on `ImpliedVolModel`** — their only model input is the same surface; the convexity method becomes an analytic-config field (next phase)
- [x] **Multi-index trades through `price`** (Phase 4, D16) — `MultiLegSwap × None` (tenor basis, amortising, seasoned; `swap_pv` on `resolved.curves`, fixings from the market), `CrossCurrencySwap × None` (domestic and `foreign_discount_id` curves, FX spot keyed by the pair), `ZeroCouponInflationSwap` / `YearOnYearInflationSwap × None` and `InflationCapFloor × ImpliedVolModel` (keyed inflation curves), each equal to its kernel bit for bit with the trade traced under `jit`, gradients on exactly the named entries and `vmap` over same-id trades (`tests/test_core/test_multi_leg_route.py`, `test_cross_currency_route.py`, `test_inflation_routes.py`); graph pricers stay the `CurveGraph` twins of the routed kernels
- [x] **One onboarding contract for every route** (Phase 4 close-out, [D17](architecture/calibration-pricing-risk.md#3-decisions)) — `tests/test_core/route_samples.py` holds a sample for every registered (instrument, model) pair on one shared market with decoy entries; `tests/test_core/test_route_contract.py` checks every registered (pair, method): finite price, default config (`config=None`), trade traced under `eqx.filter_jit`, `eqx.filter_vmap` over a stacked pair, market gradient finite, non-zero on the sample's entries and exactly zero outside `valax.core.declared_entries` (instrument selectors + model `curve_requirements` + surface reference); a pair without a sample fails (the 4 LMM routes were the named exceptions until D18 routed them; none remain); no allow-listed deviations. Found and fixed `MarketState.dividend` raising for a non-paying asset among payers. Checklist: [tutorial § 10](guide/tutorial-unified-pricing.md#onboarding-checklist-a-new-instrument-or-route-in-the-library)
- [x] **Risk over every declared entry** (Phase 4 close-out) — `valax.greeks.market_risk` → `MarketRisk`: from one backward pass through `price()`, pillar deltas of every declared curve (a tenor basis swap reports all three), and gradients of declared spots, dividends, FX spots, inflation curves and surfaces, keyed by id (`tests/test_greeks/test_market_risk.py`)
- [x] **FX forward and FX vanilla through `price`** (Phase 4 close-out, first trades onboarded by the checklist) — `FXForward × None` and `FXVanillaOption × ImpliedVolModel` (`analytic`); new static selectors `discount_id` (domestic) / `foreign_discount_id`, FX spot keyed by the pair, rates `project_rate` of each curve to expiry, vol `sample_vol` at (strike, expiry); bit-identical to `fx_forward_price` / `garman_kohlhagen_price` (`tests/test_core/test_fx_routes.py`)
- [ ] **Derivative convergence of numerical engines** — PDE / lattice / MC Greeks against exact oracles with documented per-engine tolerances
- [ ] **Position / portfolio layer** — quantity and trade id, grouping by structure and shape, manual schedule padding, book-level quote risk through one sweep

### 5.7 Monte Carlo Variance Reduction

Neither `mc_price_dispatch` nor the low-level path pipeline applies variance
reduction today (engine notes: [Monte Carlo § 8](guide/monte-carlo.md#8-variance-reduction-planned)).

- [ ] **Antithetic variates** — plug-in at the path-generator level; ~2× for smooth payoffs
- [ ] **Control variates** — e.g. Black-Scholes as the control for Heston / SABR
- [ ] **Importance sampling** — problem-specific (deep out-of-the-money, barriers)
- [ ] **Stratified / quasi-MC (Sobol, Halton)** — blocked: `diffrax` does not yet expose quasi-random Brownian increments

---

## Tier 6: Operations and Integration

Production deployment concerns.

### 6.1 Market Data Integration

- [ ] **Market data adapters** — Bloomberg, Refinitiv, ICE, exchange feeds
- [ ] **Real-time curve and surface building** from streaming quotes
- [ ] **Market data snapping** — end-of-day snapshots for risk

### 6.2 Trade Lifecycle

- [ ] **Trade booking** — create, amend, cancel trades
- [ ] **Event processing** — fixings, exercises, expirations, settlements
- [ ] **Position management** — aggregation, netting

### 6.3 Real-Time Risk

- [ ] **Incremental risk** — update portfolio risk when a single trade changes (avoid full recomputation)
- [ ] **Streaming Greeks** — real-time sensitivity updates as market data ticks
- [ ] **Risk limits and alerts**

**Why:** These are necessary for production deployment but are more about systems engineering than quantitative finance. They become relevant once the pricing and risk engine is mature.

---
	
## VALAX Advantages to Leverage

Throughout this roadmap, VALAX's JAX-native architecture provides structural advantages:

| Capability | Traditional Libraries | VALAX |
|---|---|---|
| **Greeks** | Finite differences (bump-and-reprice) | Exact via `jax.grad` — faster, no numerical noise |
| **Calibration** | Numerical Jacobians | Analytic Jacobians via autodiff — faster convergence |
| **Batch pricing** | Serial loops or manual parallelism | `vmap` — one line, automatic vectorization |
| **GPU/TPU** | Requires rewrite (CUDA kernels) | Free via JAX backend swap |
| **XVA nested MC** | Cluster computing, days of runtime | GPU-accelerated, potentially real-time |
| **Neural surrogates** | Separate ML stack (PyTorch/TF) | Same framework — train and deploy in JAX |
| **Risk sensitivities** | Separate Greek engine | Any pricing function is automatically differentiable |

These advantages compound as the library grows. Each new pricing function automatically supports Greeks, GPU acceleration, and batch evaluation with zero additional code.
