# Roadmap: Path to Production

VALAX today covers equity vanillas and exotics, a broad rates suite (vanilla and exotic), FX and inflation vanillas, four pricing methods (analytic, Monte Carlo, PDE, lattice), and around ten stochastic models — all with autodiff Greeks and JIT compilation. It is still a fraction of the **feature breadth** of production quant libraries at major banks (JPM Athena, Goldman SecDB, etc.), which have been built over 20-30 years by hundreds of quants.

The good news: the architecture (pure-functional JAX, autodiff-first, pytree data model) is arguably **more modern** than most bank stacks. The gap is breadth, not depth.

This roadmap organizes every missing piece into prioritized tiers. Each tier unlocks the next.

---

## Current State

| Area | What We Have |
|------|-------------|
| **Instruments** | **Priced:** `EuropeanOption`, `AmericanOption`, `EquityBarrierOption`, `AsianOption`, `LookbackOption`, `VarianceSwap`, `DigitalOption`, `SpreadOption`, `WorstOfBasketOption`; `ZeroCouponBond`, `FixedRateBond`, `FloatingRateBond`, `CallableBond`, `PuttableBond`; `Caplet`, `Cap`, `InterestRateSwap`, `Swaption`, `BermudanSwaption`, `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `CMSSpreadSwap`, `CMSCapFloor`, `RangeAccrual`; `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`, `InflationCapFloor`; `FXForward`, `FXVanillaOption`. **Pytree defined, pricer on the roadmap:** `ConvertibleBond`, `CompoundOption`, `ChooserOption`, `Autocallable`, `Cliquet`, `FXBarrierOption`, `QuantoOption`, `TARF`, `FXSwap`, `CDS`, `CDOTranche` |
| **Models** | Black-Scholes, **Heston (MC + Fang-Oosterlee COS semi-analytic)**, SABR (analytic + MC), LMM (MC with PCA factors), **Hull-White (analytic ZCB, trinomial tree, MC, PDE)**, **G2++ (two-factor Gaussian; MC + PDE)**, **Cheyette (quasi-Gaussian 1F/2F skew; MC + PDE)**, **Local Volatility (Dupire from SVI/SABR/Grid surfaces; MC + PDE)**, **SLV (Heston × leverage; MC)**, **Multi-asset GBM (correlated)** |
| **Pricing** | Black-Scholes, Black-76, Bachelier, SABR analytic; **Heston Fang-Oosterlee COS**; **Dupire local vol (Gatheral IV-space)**; Monte Carlo (GBM, Heston, SABR, Local Vol, **SLV**, **multi-asset GBM**, LMM, **Hull-White**, **G2++**, **Cheyette**); Crank-Nicolson 1-D PDE; **2-D ADI PDE (Heston)**; **short-rate PDE (Hull-White, G2++, Cheyette)**; CRR binomial tree; Hull-White trinomial tree |
| **Greeks** | 1st order (delta, vega, rho) and 2nd order (gamma, vanna, volga) via autodiff; key-rate durations via curve pytree differentiation |
| **Curves** | `DiscountCurve` with log-linear interpolation, flat extrapolation |
| **Dates** | Act/360, Act/365, Act/Act, 30/360; ordinal-based date arithmetic; coupon schedule generation |
| **Calibration** | SABR, Heston, **Hull-White, G2++, Cheyette (1F/2F)**, and **SLV leverage** calibration with LM / BFGS / Optax solvers; parameter constraint transforms. Curve bootstrap (sequential, simultaneous, joint `CurveGraph`); **survival-curve bootstrap from CDS spreads**; SABR / SVI vol surfaces, **swaption cube, optionlet surface** |
| **Portfolio** | `batch_price()` and `batch_greeks()` via vmap |
| **Market Data** | `MarketData` container, `MarketScenario` / `ScenarioSet` for risk factor shocks; full synthetic-data subpackage (`valax.market.synthetic`) covering Stages 1–6 of the workflow — `SyntheticMarketConfig`, `SeedRegistry`, NSS / flat curve samplers, random PSD correlation matrices, ground-truth model parameters, noisy observation layer, option / swap portfolio samplers, correlated-GBM market tapes, scenario sets, and deliberate arbitrage injectors |
| **Testing infrastructure** | Versioned golden-dataset harness (`tests/golden/`, `assert_matches_golden`, `REGEN_GOLDEN=1` regeneration); shared `SeedRegistry` fixture; pytest markers `arbitrage` / `golden` / `detects`; non-tautological closed-loop tests (pricer × implied-vol roundtrip, analytic vs MC within 3·stderr, autodiff vs finite-difference Greeks, NSS curve self-consistency, SABR calibration at the observation-noise floor, bootstrap roundtrip on off-pillar dates); reserved arbitrage exception types in `valax.core.diagnostics`; `@pytest.mark.xfail` backlog of missing safety checks |
| **Risk** | Curve shocks (parallel, steepener, butterfly, key-rate), parametric/historical/stress scenarios, VaR, ES, sensitivity ladders with waterfall P&L |
| **MC infrastructure** | Unified `mc_price_dispatch(instrument, model, ...)` with `(instrument, model)` → recipe registry. Recipes span equity (GBM / Heston / local vol / SLV), multi-asset equity (correlated GBM), and rates (LMM plus the Hull-White / G2++ / Cheyette short-rate models). `register()` decorator for contributor extensions. See [Monte Carlo guide](guide/monte-carlo.md#2-coverage-map). |
| **Multi-asset MC** | `MultiAssetGBMModel` with per-asset vols/dividends + correlation matrix. `generate_correlated_gbm_paths` (exact log-Euler via Cholesky). Spread-option MC (validates Margrabe + Kirk analytical) and worst-of basket MC with correlation Greeks via `jax.grad`. |

### Instrument Coverage Matrix

A snapshot of every instrument class relevant to production bank systems, with implementation status and blocking dependencies.

#### ✅ Implemented

| Instrument | Asset class | Module | Pricing available |
|------------|-------------|--------|-------------------|
| `EuropeanOption` | Equity | `instruments/options.py` | BSM, Black-76, Bachelier, SABR, MC (GBM/Heston), PDE, Lattice |
| `ZeroCouponBond` | Fixed income | `instruments/bonds.py` | Curve discounting |
| `FixedRateBond` | Fixed income | `instruments/bonds.py` | Curve discounting, YTM, duration, convexity, KRDs |
| `FloatingRateBond` / FRN | Fixed income | `instruments/bonds.py` | Forward projection + single-curve discounting, seasoned fixings |
| `Caplet` | Rates | `instruments/rates.py` | Black-76, Bachelier |
| `Cap` (and Floor) | Rates | `instruments/rates.py` | Black-76, Bachelier (strip pricing) |
| `InterestRateSwap` | Rates | `instruments/rates.py` | Analytic (replication), curve discounting |
| `Swaption` | Rates | `instruments/rates.py` | Black-76, Bachelier |
| `OISSwap` / `SOFRSwap` | Rates | `instruments/rates.py` | Telescoping float-leg identity + fixed-leg annuity (single-curve) |
| `BermudanSwaption` | Rates | `instruments/rates.py` | Hull-White PDE (exact exercise projection); Longstaff-Schwartz MC on LMM paths |
| `FXForward` | FX | `instruments/fx.py` | Covered interest rate parity |
| `FXVanillaOption` | FX | `instruments/fx.py` | Garman-Kohlhagen, implied vol, 3 delta conventions |
| `AmericanOption` | Equity | `instruments/options.py` | Binomial tree (CRR), European/American exercise |
| `EquityBarrierOption` | Equity exotics | `instruments/options.py` | MC with sigmoid smoothing, knock-in/knock-out parity |
| `AsianOption` | Equity exotics | `instruments/options.py` | MC arithmetic and geometric averaging |
| `LookbackOption` | Equity exotics | `instruments/options.py` | MC floating-strike and fixed-strike variants |
| `VarianceSwap` | Volatility | `instruments/options.py` | Analytic (BSM), MC realized variance, seasoned pricing |
| `CrossCurrencySwap` | Rates / FX | `instruments/rates.py` | Two-curve telescoping + spot conversion, par basis solver |
| `TotalReturnSwap` | Equity / prime brokerage | `instruments/rates.py` | Self-financing reduction to spread annuity, optional unrealized return |
| `CMSSwap` | Rates | `instruments/rates.py` | Forward par swap rate (no convexity adjustment) |
| `CMSCapFloor` | Rates | `instruments/rates.py` | Black-76 on forward CMS rate (no convexity adjustment) |
| `RangeAccrual` | Rates / structured | `instruments/rates.py` | Black-76 digital-replication (snapshot probability) |
| `CallableBond` | Fixed income | `instruments/bonds.py` | Hull-White trinomial tree with backward induction, issuer-optimal call exercise |
| `PuttableBond` | Fixed income | `instruments/bonds.py` | Hull-White trinomial tree with backward induction, holder-optimal put exercise |
| `ZeroCouponInflationSwap` | Inflation | `instruments/inflation.py` | Forward CPI projection + nominal discounting, breakeven solver |
| `YearOnYearInflationSwap` | Inflation | `instruments/inflation.py` | YoY forward CPI ratio + nominal discounting (no convexity adjustment) |
| `InflationCapFloor` | Inflation | `instruments/inflation.py` | Black-76 on YoY forward inflation rate |
| `SpreadOption` | Multi-asset | `instruments/options.py` | Margrabe (K=0), Kirk (K≠0), Monte Carlo via `MultiAssetGBMModel` |
| `WorstOfBasketOption` | Multi-asset | `instruments/options.py` | Monte Carlo via `MultiAssetGBMModel` (correlation Greeks via `jax.grad`); analytical form does not exist |
| `DigitalOption` | Equity exotics | `instruments/options.py` | Analytic (cash-or-nothing $\Phi(d_2)$), PDE (Crank-Nicolson + Rannacher damping) |
| `CMSSpreadSwap` | Rates | `instruments/rates.py` | Two-factor MC + PDE (G2++, two-factor Cheyette) — decorrelation steepener/flattener |

#### 🟠 Missing — Medium Priority (specific desks)

All instrument pytrees below are already defined; only their pricers are pending.

| Instrument | Asset class | Complexity | Blocked by | Notes |
|------------|-------------|------------|------------|-------|
| `CDS` | Credit | Medium | CDS pricing engine (survival-curve bootstrap already ships) | Prerequisite for XVA (CVA). Huge market |
| `QuantoOption` | FX / equity cross | Medium | Quanto drift adjustment (multi-asset MC already ships) | FX-equity correlation adjustment |
| `FXBarrierOption` | FX | Medium | Reiner-Rubinstein analytic (or MC) barrier pricer | FOR/DOM conventions already defined |

#### 🟡 Missing — Lower Priority (structured / exotic)

| Instrument | Asset class | Complexity | Blocked by | Notes |
|------------|-------------|------------|------------|-------|
| `Autocallable` / Phoenix | Structured products | High | Multi-observation structured-payoff recipe (SLV + multi-asset paths already ship) | Multi-hundred-billion dollar market. Path-dependent |
| `CDOTranche` | Credit correlation | High | Copula simulation (survival curve already ships) | Gaussian copula, base correlation |
| `ConvertibleBond` | Equity-credit hybrid | High | 1-D PDE with credit (Tsiveriotis-Fernandes) | Equity + credit + optionality |
| `TARF` | FX structured | High | MC with early termination | Target accrual range forward |
| `Cliquet` / Ratchet | Equity structured | Medium | Forward-starting structured-payoff recipe (LV / SLV already ship) | Popular in structured notes |
| `MBS` | Securitized | Very high | Prepayment models, OAS (no pytree yet) | Very US-market-specific |
| `CompoundOption` | Exotic | Low | Geske decomposition (BSM extension) | Option on an option. Rare in practice |
| `ChooserOption` | Exotic | Low | BSM decomposition | Choose call/put at a future date |

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
now through **short-rate calibration**. It stops at **risk, surfaces, and
portfolio**:

```
quotes ─▶ bootstrap_curve_graph ─▶ MarketState ─▶ resolve ─▶ analytic          ✅
                   │                                      ├─ MC/PDE (HW, G2++) ✅
                   │                                      ├─ HW lattice        ❌
            quote_jacobian ✅                             └─ Cheyette / LMM    ❌
                   ┊
                   ┊ (never composed)
                   ▼
            greeks 🟡 ── risk ❌ ── calibration 🟡 ── surfaces ❌ ── portfolio ❌
                                     (HW/G2++ ✅,
                                      Cheyette refuses)
```

So a book can be **priced** multi-curve today and its short-rate models
**calibrated** multi-curve, but it cannot yet be **risk-managed** or
**surface-built** multi-curve.

### Shipped

- [x] **Joint curve build** — `bootstrap_curve_graph` + `CurveSpec` +
  `CurveBuildDiagnostics` (`valax/curves/bootstrap_graph.py`, `graph.py`);
  11 quote types in `valax/curves/instruments.py`; tenor-basis and
  EUR/USD four-curve builds verified end-to-end
- [x] **Quote-sensitivity Jacobian** — `quote_jacobian` via
  `optimistix.ImplicitAdjoint`, covering every registered quote scalar
  (`.rate`, `.spread`, `.futures_rate`, `.quoted_forward`, `.far_rate`,
  `.jump_size`)
- [x] **Curve-id alphabet frozen** — `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]`,
  enforced by `valax.curves.graph._CURVE_ID_RE`; `_default_` grandfathered
- [x] **Legacy paths migrated** — `bootstrap_simultaneous` and
  `bootstrap_multi_curve` are thin wrappers over the joint solver;
  `MultiCurveSet` is a view over `CurveGraph` (`from_graph` / `to_graph`)
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
- [x] **Dual-curve analytic short-rate swaptions and calibration** —
  `hw_swaption_price` / `g2pp_swaption_price` accept a projection curve, and
  `calibrate_hull_white` / `calibrate_g2pp` / `swaption_prices_from_vols`
  accept `forward_curve=`; see § 2 below

### Remaining work

#### 1. Risk pipeline is single-curve *(largest gap)*

The whole point of multi-curve is that OIS and projection curves are
**distinct risk factors** and the basis is itself hedgeable
([Why Multi-Curve § What this means for risk](guide/why-multicurve.md#what-this-means-for-risk)).
None of that is reachable today.

- [ ] **Per-curve scenario axis** — `MarketScenario.rate_shocks` /
  `ScenarioSet` (`valax/market/scenario.py`) carry one
  `Float[Array, " n_pillars"]` vector with no curve identity; every
  generator in `valax/risk/scenarios.py` assumes the fixed factor order
  `[spots, vols, rates, dividends]` with a single rate block
- [ ] **Route graph shocks through `apply_scenario`** — it shocks only
  `base.discount_curve` and passes `curve_graph` through **unshocked**
  (source comment admits it); `bump_graph_curve` and friends are exported
  but have **no caller** outside their own module and
  `tests/test_risk/test_shocks.py`
- [ ] **Stop dropping the graph in the VaR path** —
  `reprice_under_scenario`, `parametric_var` and `pnl_attribution`
  (`valax/risk/var.py`) rebuild `MarketData(...)` without `curve_graph`,
  so it silently defaults to `None`
- [ ] **Route the risk engine through `price(...)` and retire
  `wrap_equity_pricing_fn`** — the risk layer still bridges into the
  loose-scalar convention with its own adapter instead of calling the
  unified front door. Doing so is what makes "every Greek is one
  `jax.grad` through one function" true, and it subsumes the three items
  above. Tracked as the top backlog item in `PRICING_ONTOLOGY.md` § 0.1
  (repo root)
- [ ] **`IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>` risk factor** — already
  tracked as 🚧 in [Risk Factors § 3](risk-factors.md)
- [ ] **Graph-level greeks API** — `curve_pillar_deltas`
  (`valax/greeks/market_greeks.py`) resolves exactly one `curve_id` per
  call and `bucketed_curve_deltas` takes a flat pillar vector with no
  curve identity. Per-curve ladders are obtainable one at a time; there is
  no `dict[curve_id, ladder]` API and no graph-aware bucketing
- [ ] **Compose `quote_jacobian` with `∇_MarketState price`** — the
  `J_build` leg of the Jacobian stack (`PRICING_ONTOLOGY.md` § 6.3) is not
  packaged; `market_greeks` explicitly defers it ("left for the caller, since
  the build lives outside `price`"). The manual composition is shown in
  [Tutorial — Unified Pricing § 9](guide/tutorial-unified-pricing.md#from-quotes-to-hedge-ratios)

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
- [ ] **`MarketState`-native calibration entry points** — the calibrators take
  explicit curves (the kernel convention); a facade taking a `MarketState` +
  `discount_id` / `forward_id` would remove the last manual step between the
  market container and a calibrated model
- [ ] **Curve-aware equity calibration** — `calibrate_heston`,
  `calibrate_sabr`, `calibrate_slv` take loose scalars
  (`rate`, `dividend`, `forward`), never a curve object

#### 3. Vol-surface construction is curve-free

`valax/surfaces/` has **zero** references to `DiscountCurve`.
`build_sabr_caplet_surface` and `calibrate_swaption_cube` take a
caller-supplied pre-computed `forwards` array.

- [ ] **Cap → caplet (optionlet) stripping** — does not exist in any form.
  This is *the* canonical dual-curve operation: project off the tenor
  curve, discount off OIS. Its absence puts the quotes→surface leg of the
  chain outside the framework entirely
- [ ] **Dual-curve swaption-cube calibration** — same gap for
  `calibrate_swaption_cube`

#### 4. FX / credit / inflation carry no selectors

- [ ] **`discount_id` / `forward_id` on `instruments/{fx,credit,inflation}.py`**
  — deferred at ontology P3; these classes silently fall back to
  `_default_` with no way to name a curve
- [ ] **Extend `ResolvedMarket`** (`valax/core/resolve.py`) to bind survival
  curves, inflation curves, and FX spot / second-currency discount curves;
  it binds only `discount_curve`, `forward_curve`, `spot`, `surface`,
  `fixings`, `reference_date` today
- [ ] **True multi-curve cross-currency** — `cross_currency_swap_price`
  takes two curves but its own docstring states *"single-curve-per-currency
  projection: each currency's floating leg is forecasted off its own
  discount curve"*. XCCY is the multi-curve instrument par excellence
- [ ] **Per-leg index descriptors — the architectural ceiling on the
  current selectors.** A singular `forward_id` expresses the *single-index*
  case only. A single-currency tenor basis swap needs **three** curves (OIS
  discount + two projections), CMS-spread three, cross-currency four or
  more. The fix is per-leg descriptors (`SwapLeg(index_id=...)` +
  `legs: tuple[SwapLeg, ...]`), which is the same structure the cashflow
  engine needs — build it once, with
  [Tier 1.4](#14-trade-representation-and-cashflow-engine), not twice.
  Background in `PRICING_ONTOLOGY.md` § 0.1 (repo root)

#### 5. Single-curve chokepoints

- [ ] **`valax/pricing/lattice/hull_white_tree.py`** — reads
  `model.initial_curve`, no `forward_curve` parameter. After P5.1 the
  tree↔PDE↔MC triangulation (load-bearing validation here) only holds
  single-curve, so a dual-curve callable bond has no independent oracle
- [ ] **`valax/risk/oas.py`** — `bond_z_spread`, `z_spread_duration`,
  `z_spread_dv01`, `z_spread_convexity` all take one `curve: DiscountCurve`
- [ ] **`valax/portfolio/batch.py`** — `batch_price` / `batch_greeks` take
  scalar `rates`, no curve object at all; pre-dates the curve layer
- [ ] **`valax/risk/factors.py`** — `curve: DiscountCurve` at two sites
- [ ] **`MarketData.curve_graph` is inert** — consumed only by the
  `MarketState` round-trip and the `apply_scenario` pass-through; no
  pricer or risk function reads it

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
- [ ] **Business-day calendars and roll conventions** — see
  [Tier 1.3](#13-business-calendars-and-holiday-schedules); `valax/dates/`
  is holiday-agnostic, so every bootstrap pillar is an unadjusted date
- [ ] **Over-determined bootstrap** — an `optimistix.least_squares` variant
  of `bootstrap_curve_graph` with per-quote weights
- [ ] **Fixings seasoning** — `OISSwapRate` and the futures residual do
  `del fixings` (MVP: telescope only); partially-seasoned swaps ignore
  realised fixings. `index_id` is reserved for this
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

- [ ] **Multi-curve risk tests** — `tests/test_risk/test_shocks.py` is the
  only risk file touching a graph, and `test_curve_graph_carried_through`
  actively asserts the graph comes back *unchanged*
- [ ] **Multi-curve calibration tests** — none under `tests/test_calibration/`
- [ ] **Multi-curve greeks tests** — `tests/test_greeks/test_market_greeks.py`
  runs entirely against single-curve markets; nothing asserts separate
  OIS vs projection ladders
- [ ] **Multi-curve surface tests** — none under `tests/test_surfaces/`
- [ ] **Portfolio tests** — there is no `tests/test_portfolio/` package at all

### Recommended sequence

1. **Multi-curve risk** (§1) — highest leverage by a distance: it is the
   stated purpose of the framework, the shock primitives already exist
   unused, and it is the only item with an explicit "still needed" note in
   the docs. Fold the graph-level greeks API in, since both want the same
   per-curve plumbing.
2. **Dual-curve calibration** (§2) — smaller, and it closes the
   consistency gap that P5.1 opened between how models are fitted and how
   they are now priced.
3. **Caplet stripping** (§3) — unlocks the quotes→surface→exotic chain and
   is the honest prerequisite for a dual-curve vol story.

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
| **P1.1** | **Business Calendars & Date Adjustment** | Every trade in a bank settles on business days. Wrong dates = wrong cashflows = wrong P&L. Precomputed holiday arrays (TARGET, NYSE, SOFR/Fed, London, Tokyo) with modified-following/preceding and end-of-month conventions. | 1.3 |
| **P1.2** | **Cashflow Engine (Legs, Stubs, Compounding)** | Real swaps have short/long stubs, compounding-in-arrears, amortizing notionals, fixing histories. The current schedule generator can't represent production trades. `FixedLeg` / `FloatingLeg` pytrees with full convention support. | 1.4 |
| **P1.3** | **CI/CD Pipeline** | No bank adopts a library without automated testing. GitHub Actions with: lint, type-check, full test suite, QuantLib comparison, coverage reporting. A gate before anything else ships. | — |
| **P1.4** | ~~**Short-Rate Models (Hull-White, G2++, Cheyette)**~~ ✅ **Done** | One-factor Hull-White, two-factor G2++, and quasi-Gaussian Cheyette (1F/2F) — analytic/tree/MC/PDE pricing, exact curve fit, and swaption-surface calibration (`calibrate_hull_white` / `calibrate_g2pp` / `calibrate_cheyette`). | 2.1 |

### Priority 2 — Production Pricing Capabilities

These unlock entire asset classes and bring pricing to the speed and accuracy required by trading desks.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P2.1** | ~~**Heston Semi-Analytic (COS / Fourier)**~~ ✅ **Fang-Oosterlee COS implemented** | Lord-Kahl "Little Trap" characteristic function + Fang-Oosterlee (2008) COS expansion with closed-form Heston cumulants. Agrees with QuantLib's `AnalyticHestonEngine` to < 5e-7 absolute across the seed × moneyness × call/put grid. End-to-end Heston calibration roundtrip recovers all 5 parameters to floating-point precision on noise-free synthetic data. | 2.2 |
| **P2.2** | ~~**Local Volatility + SLV**~~ ✅ **Both shipped** | LV (Tier 2.3) via `valax/pricing/analytic/dupire.py` + `valax/pricing/mc/local_vol_paths.py`. SLV (Tier 2.4) via `valax/calibration/slv.py` (Guyon-Henry-Labordère particle method + optional kernel-ridge stabilisation + outer fixed-point loop) and `valax/pricing/mc/slv_paths.py` (Andersen-QE variance + log-Euler/Milstein log-spot, registered with the unified MC dispatcher for European/Asian/Barrier/Lookback/VarianceSwap). | 2.3, 2.4 |
| **P2.3** | **FX Derivatives** — ✅ vanillas + delta conventions; barriers / quanto / TARF pending | Garman-Kohlhagen vanillas, forwards, and all three delta conventions ship; FX barrier, quanto and TARF pricers remain. FX has unique conventions (delta-space quoting). | 3.3 |
| **P2.4** | **Credit Derivatives** — ✅ survival curves; CDS / CDO pricers pending | Survival / hazard-rate bootstrap from CDS spreads ships (`valax.curves.survival`). The standalone CDS and CDO-tranche pricers remain — prerequisites for credit trading and the XVA workstream. | 3.4 |

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
  ├─ [P1.1] Business Calendars ──→ [P1.2] Cashflow Engine
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

The most urgent gap is **foundational infrastructure** (P1: calendars, cashflows, CI/CD) — not because it's glamorous, but because every downstream feature depends on it. A bank quant evaluating VALAX will immediately ask: *"Can it handle modified-following date adjustment on a 10Y EUR swap with short front stub and compounding-in-arrears on the floating leg?"* — and today the answer is no.

The **highest-ROI parallel investment** is **short-rate models + Heston COS** alongside calendars/cashflows, because they unlock the most actively traded products (callable bonds, fast equity vol calibration) while the plumbing is being built.

---

## Tier 1: Critical Infrastructure

These are foundational pieces that block almost everything else. They should be built first.

### 1.1 Multi-Curve Bootstrapping

- [x] **Sequential bootstrap** from deposits, FRAs, and par swap rates — analytic solve per pillar
- [x] **Simultaneous bootstrap** via `optimistix.root_find` (Newton) in log-DF space — handles overlapping instruments
- [x] **Multi-curve framework** — `MultiCurveSet` with OIS discount curve + tenor-specific forward curves, dual-curve swap bootstrap (**deprecated in MC-Curves-2**; superseded by `bootstrap_curve_graph`). Since MC-Curves-2b it is a *view* over `CurveGraph` (`MultiCurveSet.from_graph` / `to_graph`) and `bootstrap_multi_curve` solves through the joint solver
- [x] **Joint multi-curve Newton solver** (MC-Curves-2) — `bootstrap_curve_graph` over an arbitrary `CurveGraph`, plus `CurveSpec`, `CurveBuildDiagnostics`, and the implicit-adjoint `quote_jacobian` helper
- [x] **Basis curve** support (e.g., 1M vs 3M SOFR basis) — via `TenorBasisSwap` inside the joint solver
- [x] **Cross-currency graph** — `CrossCurrencyBasisSwap` + `FXForward` inside the joint solver (EUR-USD four-curve build verified)
- [x] **Jacobian of discount factors w.r.t. input rates** via autodiff (free with JAX, verified in tests)
- [ ] **Turn-of-year effects** and stub handling (`TurnInstrument` quote type shipped; genuine step-function interpolation deferred to MC-Curves-3)
- [ ] **Alternative interpolation modes** on `CurveSpec.interp` — linear-zero, monotone-convex (Hagan-West), tension spline (MC-Curves-3)
- [ ] **Hull-White futures convexity** (`hull_white_convexity_adj`) — plug-in framework in place, closed-form factory pending
- [ ] **CSA / collateralised discounting** and FX-implied foreign-currency-collateralised discount curves

> The full multi-curve picture — what ships end-to-end, and the gaps in
> risk, calibration, surfaces and portfolio — lives in
> [Multi-Curve Framework — What Ships, What's Left](#multi-curve-framework-what-ships-whats-left).
> This tier covers the **build** side only.

**Why:** Every rates product needs proper forward curves. Manual pillar specification doesn't scale.

**Approach:** Root-finding via `optimistix.root_find` on the system of bootstrapping equations. Each instrument (deposit, swap) provides one equation. The curve is a pytree — autodiff gives the Jacobian for free.

**Status:** MC-Curves-2 shipped — the joint multi-curve Newton solver `valax.curves.bootstrap_curve_graph` and its declarative `CurveSpec` interface live in `valax/curves/bootstrap_graph.py`. For the narrative rationale (post-2008 forces, why sequential bootstrap breaks down on tenor-basis and cross-currency quotes, and how the joint solve unlocks the implicit-adjoint quote-Jacobian), see [`guide/why-multicurve.md`](guide/why-multicurve.md). Eleven quote-type pytrees (`DepositRate`, `FRA`, `SwapRate`, `OISSwapRate`, `IborSwapRate`, `MoneyMarketFuture`, `TenorBasisSwap`, `FXForward`, `FXSwap`, `CrossCurrencyBasisSwap`, `TurnInstrument`) live in `valax/curves/instruments.py`. Tenor-basis and cross-currency joint solves are verified end-to-end in `tests/test_curves/test_bootstrap_graph.py`. MC-Curves-2b completed the migration: `bootstrap_multi_curve` is retained for one deprecation cycle but now translates its two-tier input into `CurveSpec`s and calls the joint solver (the sequential internals are deleted), and `MultiCurveSet` is a view over `CurveGraph` — see `tests/test_curves/test_multi_curve.py`. `quote_jacobian` exposes the implicit-adjoint quote-sensitivity Jacobian.

### 1.2 Volatility Surface Construction

- [x] **Vol surface object** — `GridVolSurface` with bilinear interpolation on (strike, expiry) grid
- [x] **SABR smile per expiry** — `SABRVolSurface` with per-expiry SABR calibration and parameter interpolation
- [x] **SVI parametric fitting** — `SVIVolSurface` with Gatheral's SVI parameterization and LM calibration
- [x] **Swaption cube** — `SwaptionCube` with `calibrate_swaption_cube` (strike × expiry × tenor)
- [x] **Optionlet (caplet) surface** — `OptionletVolSurface` via `build_sabr_caplet_surface`
- [ ] **SSVI** — global arbitrage-free parameterization (Gatheral-Jacquier)
- [ ] **Sticky-strike vs sticky-delta** conventions
- [x] **Local vol extraction** from implied vol surface (Dupire formula) — shipped

**Why:** Every option product needs a vol surface. Scalar vol is a toy assumption.

**Approach:** Store the surface as an `eqx.Module` pytree with pillar data. Interpolation via cubic splines (or SABR per expiry). SVI fitting via `optimistix.least_squares`. The surface must be differentiable for Greeks.

**Status:** Three surface types implemented in `valax/surfaces/`: `GridVolSurface` (interpolated grid), `SABRVolSurface` (SABR per expiry with calibration), `SVIVolSurface` (SVI per expiry with calibration). All are callable, JIT-able, differentiable pytrees. See the [Vol Surfaces guide](guide/vol-surfaces.md).

### 1.3 Business Calendars and Holiday Schedules

- [ ] **Holiday calendar objects** for major markets: TARGET, NYSE, SOFR (Fed), London, Tokyo
- [ ] **Date adjustment conventions**: modified following, modified preceding, end-of-month
- [ ] **Business day counting** (e.g., Bus/252 for Brazil)
- [ ] **Calendar combination** (union/intersection for cross-currency)

**Why:** Every real trade settles on business days. Wrong dates = wrong cashflows = wrong prices.

**Approach:** Precomputed boolean arrays of holidays (JIT-compatible, no runtime lookups). Calendar as a pytree with a `jnp.array` of holiday ordinals. Adjustment functions operate on integer ordinals.

### 1.4 Trade Representation and Cashflow Engine

- [ ] **Leg abstraction** — fixed leg, floating leg with compounding conventions (in arrears, in advance, compounded, averaged)
- [ ] **Cashflow schedule generation** with proper date adjustment, stub periods (short/long front/back stubs)
- [ ] **Fixing histories** — realized fixings for partially-settled floating legs
- [ ] **Amortization schedules** — notional step-down for amortizing swaps
- [ ] **Accrued interest** calculation

**Why:** Real swaps and bonds have complex cashflow structures. The current simplified schedule generation can't handle production trades.

**Approach:** `FixedLeg` and `FloatingLeg` as `eqx.Module` pytrees. Schedule generation takes a calendar + conventions and produces arrays of (accrual_start, accrual_end, payment_date, notional). Cashflow projection is a pure function over these arrays.

---

## Tier 2: Models

New stochastic models that unlock entire product categories.

### 2.1 Short-Rate Models

- [x] **Hull-White (one-factor)** — mean-reverting short rate; analytic bond & swaption prices (Jamshidian), trinomial tree, MC, PDE
- [x] **G2++ (two-factor)** — two correlated Gaussian factors; MC + PDE, decorrelation payoffs (CMS-spread)
- [x] **Cheyette (quasi-Gaussian, 1F/2F)** — state-dependent local vol adds a rate skew; MC + PDE; reduces to Hull-White / G2++ at β=0
- [ ] **CIR** — square-root diffusion, ensures positive rates
- [ ] **Vasicek** — Ornstein-Uhlenbeck for rates (allows negative rates)
- [x] **Calibration to swaption surface** — `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_cheyette`
- [x] **Multi-curve pricing in the MC and PDE short-rate solvers** (ontology P5.1) — `valax.pricing.multicurve` (`with_discount_curve`, `tenor_basis`, `swap_leg_basis`, `apply_swap_basis`): OIS discounting by re-fitting the exact-fit model to the trade's discount curve, plus a frozen deterministic tenor basis for projection. Wired into every Hull-White and G2++ recipe in `valax/pricing/pde/{hull_white,g2pp}.py` and `valax/pricing/mc/recipes.py`, and reached automatically from an instrument's `discount_id` / `forward_id` through `valax.core.price`. Single-curve prices are unchanged bit-for-bit. See [Multi-Curve Short Rates](theory/multi-curve-short-rate.md)
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

- [x] **Leverage function** calibration — shipped in `valax/calibration/slv.py` (Guyon-Henry-Labordère particle method with optional kernel-ridge stabilisation, outer fixed-point loop via `n_iterations`).
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
- [x] **MC recipes** — caplet, cap, and European swaption under `LMMModel`
- [x] **Bermudan swaption via Longstaff-Schwartz** — `bermudan_swaption_lsm`
      (see [Tier 3.1](#31-bermudan-swaptions))
- [ ] **Calibration — the headline gap.** There is no `calibrate_lmm`. LMM is
      the only model in the library with no calibrator, so its parameters must
      be supplied by hand and it cannot be fitted to market. Needs **caplet /
      swaption volatility stripping**: fit the `RebonatoVol` abcd surface and
      the correlation parameters to a cap/floor optionlet surface and a
      swaption cube. Converges with
      [Multi-Curve Framework §3](#3-vol-surface-construction-is-curve-free) —
      the stripping routine that section calls for is the same one this needs.
      Expect the Levenberg-Marquardt arity trap documented in the
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
- [ ] **FX barrier options** — continuous and discrete monitoring (instrument defined, analytical pricing TBD)
- [ ] **Quanto options** — correlation between FX and underlying
- [ ] **TARFs (Target Accrual Range Forwards)** — path-dependent FX exotics via MC

**Why:** FX is one of the largest derivatives markets. FX options have unique smile conventions (delta-space, not strike-space).

**Approach:** Garman-Kohlhagen is a minor extension of Black-Scholes. FX smile needs delta-strike conversion utilities. TARFs via MC with diffrax.

**Status:** Core FX pricing implemented in `valax/pricing/analytic/fx.py` and instruments in `valax/instruments/fx.py`. Garman-Kohlhagen pricing, FX forward valuation, implied vol inversion, and all three delta conventions (spot, forward, premium-adjusted) with strike↔delta conversion. See the [Analytical Pricing guide](guide/analytical.md#fx-options-garman-kohlhagen).

### 3.4 Credit Derivatives

- [x] **Survival curve** construction from CDS spreads — `valax.curves.survival` (`from_cds_spreads`, `SurvivalCurve`)
- [ ] **CDS pricing** — protection and premium leg valuation (instrument pytree defined; pricer pending)
- [x] **Hazard rate bootstrapping** from CDS term structure — `from_cds_spreads` / `from_hazard_rates`
- [ ] **CDO tranche** pricing (Gaussian copula, base correlation) (instrument pytree defined; pricer pending)

**Why:** Credit derivatives are essential for bank trading books and CVA computation.

**Approach:** Survival curves as `eqx.Module` pytrees (analogous to discount curves). CDS pricing is cashflow discounting with survival probabilities. CDO requires copula simulation — JAX's PRNG system handles this well.

### 3.5 Inflation Derivatives

- [x] **Zero-coupon inflation swap** pricing — `zcis_price`, breakeven solver
- [x] **Year-on-year inflation swap** pricing — `yyis_price` (no convexity adjustment yet)
- [x] **Inflation cap/floor** — Black-76 on forward YoY inflation rate
- [ ] **Seasonality adjustment** for monthly CPI
- [ ] **YoY convexity adjustment** (Jarrow-Yildirim)

**Why:** Inflation markets have grown significantly. Real-money investors use inflation swaps for liability hedging.

**Approach:** Inflation curve as a pytree mapping dates to forward CPI levels. Pricing is standard cashflow discounting with CPI ratios.

### 3.6 Equity Exotics

- [ ] **Autocallable / phoenix notes** — path-dependent early redemption (SLV ships; structured-payoff recipe pending)
- [x] **Worst-of basket options** — multi-asset MC via `MultiAssetGBMModel`
- [x] **Lookback options** — MC floating and fixed strike
- [ ] **Touch / no-touch / one-touch** — digital barriers (cash-or-nothing `DigitalOption` ships; one-touch barriers pending)
- [x] **Range accruals** — Black-76 digital replication (rates `RangeAccrual`)
- [ ] **Compound options** — option on an option (Geske; pytree defined, pricer pending)
- [x] **Variance / volatility swaps** — analytic static replication + MC realized variance
- [x] **Spread & digital options** — Margrabe / Kirk + MC (spread); analytic + PDE (digital)

**Why:** Structured products desks sell these daily. Autocallables alone are a multi-hundred-billion dollar market.

**Approach:** All priced via MC with appropriate payoff functions. Multi-asset requires correlated path generation (Cholesky on correlation matrix). Autocallables need path-wise barrier monitoring. SLV model (Tier 2.4) is the standard for pricing.

### 3.7 CMS and Spread Options

- [x] **CMS rate convexity adjustment** — `cms_convexity_adjustment` (analytic); Hagan static replication is a follow-up
- [x] **CMS cap/floor** pricing — `cms_cap_floor_price_black76`
- [x] **CMS spread options** — `CMSSpreadSwap` priced under two-factor G2++ / Cheyette (MC + PDE)
- [ ] **Range accrual on CMS** — structured coupon products

**Why:** CMS products are actively traded in rates markets. The convexity adjustment is non-trivial and a classic quant problem.

**Approach:** CMS convexity adjustment via static replication (Hagan) or SABR integration. Spread options via 2D copula or Margrabe-like extensions.

### 3.8 Mortgage-Backed Securities and Convertible Bonds

- [ ] **MBS** — prepayment models (PSA, CPR), OAS, effective duration
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
- [ ] **Historical stress scenarios** — replay historical crises (2008, COVID, SVB)
- [ ] **Reverse stress testing** — find scenarios that cause a target loss (optimization problem)
- [x] **P&L attribution** — second-order Taylor decomposition into delta (spot, vol, rate, div), gamma, and unexplained via autodiff

**Why:** Regulatory requirement and core risk management practice.

**Approach:** Scenarios as pytree diffs applied to `MarketData`. Batch repricing via `vmap`. P&L attribution via Taylor expansion using autodiff Greeks. Reverse stress testing via `optimistix.minimize`.

**Status:** Scenario framework implemented. `MarketScenario` and `ScenarioSet` pytrees, `apply_scenario` for full market state shocks, named stress builders (steepener, flattener, butterfly). P&L attribution via `pnl_attribution()` with second-order Taylor expansion. Historical crisis replay and reverse stress testing are next.

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
- [ ] **American / exotic under 2-D ADI** (early-exercise projection, barriers on the 2-D grid)
- [ ] **Local-stochastic vol (SLV) and two-asset** 2-D recipes
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

- [x] **Homogeneous batch pricing / Greeks** — `jax.vmap` over a batched instrument pytree (`valax/portfolio/batch.py`; `risk/ladders.py`, `risk/var.py`)
- [x] **Book-level aggregation & bucketing** — `SensitivityLadder`, `BucketMap`, PCA factor model (`valax/risk/{ladders,bucketing,factors}.py`)
- [ ] **`Portfolio` container** — heterogeneous trade collection with per-trade pricing keys (design in [Productionisation](architecture/production.md))
- [ ] **Heterogeneous dispatch-and-aggregate** — group a mixed book by `(instrument, model)`, price each group under one JIT kernel via the MC/PDE registry, aggregate to book value / Greeks
- [ ] **Shape-stable / padded batching** — fixed-shape padding so the JIT cache is reused across differently-sized books
- [ ] **Multi-device parallelism** — shard the instrument / scenario axis across GPUs/TPUs via `jax.sharding` `Mesh` / `shard_map` (today: single-device XLA + `vmap` only)
- [ ] **GPU/TPU benchmark suite** — published GPU-vs-CPU throughput (roadmap P5.2)
- [ ] **Unified market-quote container + quote-space (hedge-space) risk** — one container spanning all rate / FX / vol / credit quotes, with PV differentiated end-to-end through calibration (AAD via `optimistix` implicit diff + `quote_jacobian`) so book Greeks/vega come out as hedge-ratio sensitivities to tradeable quotes. **Container shipped** (ontology P2): `valax.core.MarketState` — keyed `CurveGraph` + vol surfaces + spots + fixings + credit/inflation/FX — together with the unified `valax.core.price(instrument, model, market, method=...)` signature and `valax.core.resolve`. **Still missing:** the end-to-end wiring (P4) that composes `∇_MarketState price` with `quote_jacobian` and `bucketing.pushforward_sensitivities` into reported hedge-space risk.

**Why:** the JAX-native design makes book- and business-scale a composition
problem (`vmap` / `jit` / `grad` stack over pure pytree pricers) rather than a
rewrite. Single-device batching ships today; the heterogeneous book engine and
multi-device sharding are the remaining pieces.

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
