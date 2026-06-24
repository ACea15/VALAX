# Roadmap: Path to Production

VALAX today covers standard equity options, vanilla rates derivatives, four pricing methods, and three stochastic models — all with autodiff Greeks and JIT compilation. This is roughly **5-10% of feature parity** with production quant libraries at major banks (JPM Athena, Goldman SecDB, etc.), which have been built over 20-30 years by hundreds of quants.

The good news: the architecture (pure-functional JAX, autodiff-first, pytree data model) is arguably **more modern** than most bank stacks. The gap is breadth, not depth.

This roadmap organizes every missing piece into prioritized tiers. Each tier unlocks the next.

---

## Current State

| Area | What We Have |
|------|-------------|
| **Instruments** | `EuropeanOption`, `ZeroCouponBond`, `FixedRateBond`, `FloatingRateBond`, `Caplet`, `Cap`, `InterestRateSwap`, `Swaption`, `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `CMSCapFloor`, `RangeAccrual`, `CallableBond`, `PuttableBond`, `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`, `InflationCapFloor`, `FXForward`, `FXVanillaOption`, `FXBarrierOption` |
| **Models** | Black-Scholes, **Heston (MC + Fang-Oosterlee COS semi-analytic)**, SABR (analytic + MC), LMM (MC with PCA factors), Hull-White (analytic ZCB, trinomial tree), **Local Volatility (Dupire from SVI/SABR/Grid surfaces + MC)** |
| **Pricing** | Black-Scholes, Black-76, Bachelier analytic; **Heston Fang-Oosterlee COS**; **Dupire local vol (Gatheral IV-space)**; Monte Carlo (GBM, Heston, SABR, **Local Vol**, LMM); Crank-Nicolson PDE; CRR binomial tree; Hull-White trinomial tree (callable/puttable bonds) |
| **Greeks** | 1st order (delta, vega, rho) and 2nd order (gamma, vanna, volga) via autodiff; key-rate durations via curve pytree differentiation |
| **Curves** | `DiscountCurve` with log-linear interpolation, flat extrapolation |
| **Dates** | Act/360, Act/365, Act/Act, 30/360; ordinal-based date arithmetic; coupon schedule generation |
| **Calibration** | SABR and Heston calibration with LM, BFGS, and Optax solvers; parameter constraint transforms |
| **Portfolio** | `batch_price()` and `batch_greeks()` via vmap |
| **Market Data** | `MarketData` container, `MarketScenario` / `ScenarioSet` for risk factor shocks; full synthetic-data subpackage (`valax.market.synthetic`) covering Stages 1–6 of the workflow — `SyntheticMarketConfig`, `SeedRegistry`, NSS / flat curve samplers, random PSD correlation matrices, ground-truth model parameters, noisy observation layer, option / swap portfolio samplers, correlated-GBM market tapes, scenario sets, and deliberate arbitrage injectors |
| **Testing infrastructure** | Versioned golden-dataset harness (`tests/golden/`, `assert_matches_golden`, `REGEN_GOLDEN=1` regeneration); shared `SeedRegistry` fixture; pytest markers `arbitrage` / `golden` / `detects`; non-tautological closed-loop tests (pricer × implied-vol roundtrip, analytic vs MC within 3·stderr, autodiff vs finite-difference Greeks, NSS curve self-consistency, SABR calibration at the observation-noise floor, bootstrap roundtrip on off-pillar dates); reserved arbitrage exception types in `valax.core.diagnostics`; `@pytest.mark.xfail` backlog of missing safety checks (see [Arbitrage Detection — Session Backlog](#arbitrage-detection--session-backlog)) |
| **Risk** | Curve shocks (parallel, steepener, butterfly, key-rate), parametric/historical/stress scenarios, VaR, ES, sensitivity ladders with waterfall P&L |
| **MC infrastructure** | Unified `mc_price_dispatch(instrument, model, ...)` with `(instrument, model)` → recipe registry. 16 built-in recipes covering equity (GBM/Heston), multi-asset equity (correlated GBM), and rates (LMM). `register()` decorator for contributor extensions. See [Monte Carlo guide](guide/monte-carlo.md#2-coverage-map). |
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
| `BermudanSwaption` | Rates | `instruments/rates.py` | Longstaff-Schwartz MC on LMM paths |
| `FXForward` | FX | `instruments/fx.py` | Covered interest rate parity |
| `FXVanillaOption` | FX | `instruments/fx.py` | Garman-Kohlhagen, implied vol, 3 delta conventions |
| `FXBarrierOption` | FX | `instruments/fx.py` | Instrument defined; analytical pricing TBD |
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

#### 🟠 Missing — Medium Priority (specific desks)

| Instrument | Asset class | Complexity | Blocked by | Notes |
|------------|-------------|------------|------------|-------|
| `CDS` | Credit | Medium | Survival curve / hazard rates | Prerequisite for XVA (CVA). Huge market |
| `QuantoOption` | FX / equity cross | Medium | Correlated 2-asset MC | FX-equity correlation adjustment |

#### 🟡 Missing — Lower Priority (structured / exotic)

| Instrument | Asset class | Complexity | Blocked by | Notes |
|------------|-------------|------------|------------|-------|
| `Autocallable` / Phoenix | Structured products | High | Multi-observation scan engine (MC-B4 below); SLV for pricing accuracy | Multi-hundred-billion dollar market. Path-dependent, multi-asset. Multi-asset paths already exist |
| `CDOTranche` | Credit correlation | High | CDS + copula simulation | Gaussian copula, base correlation |
| `ConvertibleBond` | Equity-credit hybrid | High | PDE or tree with credit | Equity + credit + optionality |
| `TARF` | FX structured | High | MC with early termination | Target accrual range forward |
| `Cliquet` / Ratchet | Equity structured | Medium | Forward-starting option pricer | Popular in structured notes |
| `MBS` | Securitized | Very high | Prepayment models, OAS | Very US-market-specific |
| `CompoundOption` | Exotic | Low | BSM extension | Option on an option. Rare in practice |
| `ChooserOption` | Exotic | Low | BSM extension | Choose call/put at a future date |

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
3. **Chain validation** — calibrated surface flows through VALAX and
   QL pricers; tests the quote→surface→exotic pipeline that desks
   actually run.

**Estimated effort:** ~3 days end-to-end. **Expected output:**
~1000 sampled markets agreeing with QuantLib at every stage of the
pricing chain.

## Heston MC — Session Backlog

### HE-1 — Andersen QE Heston scheme &nbsp;✅ &nbsp;*Done (Sprint 5)*

**What was done.** Replaced the diffrax Euler-with-reflection scheme
in `valax/pricing/mc/paths.py::generate_heston_paths` with Andersen's
(2008) Quadratic-Exponential algorithm, implemented directly with
`jax.lax.scan`. The variance update is exact in distribution at each
step (quadratic-vs-exponential switch at `ψ_c = 1.5`); the log-spot
uses Andersen's central discretisation with trapezoidal weights
`γ₁ = γ₂ = 0.5`. The function signature is unchanged, so all callers
pick up the improvement transparently.

**Surfaced by.** The Stage-2 asymmetric calibration test
`tests/test_quantlib_comparison/test_heston_ql.py::TestHestonQLCalibratesVALAXReprices`
(see [QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md#sprint-3--stage-2-complete)).
QL's single-expiry Heston calibration routinely produced
Feller-violating parameter sets (`kappa → 0`); under those, the
old VALAX MC overpriced deep-ITM calls by a few bp absolute (4–7 MC
stderrs at `n_steps=500`).

**Result.** The xfail Heston-Asian test flipped to passing at 3 SE
with `n_steps=100` — a 5× reduction from the 500 that the Euler-with-
reflection scheme needed just to bury the bias. The previously skipped
Stage-3 chain test `test_exotic_on_heston_surface_ql.py` (Heston Asian
on calibrated surface) is now enabled and green across all
seed × moneyness combinations on the first run.

**Follow-up unlocked.** A semi-analytic Heston pricer (Carr-Madan /
Lewis FFT) — the MC reference is now trustworthy enough to bench it
against.

### HE-2 — Heston Fang-Oosterlee COS pricer &nbsp;✅ &nbsp;*Done*

**What was done.** Shipped `heston_cos_price` in
`valax/pricing/analytic/heston.py` using the Lord-Kahl (2010) "Little
Trap" characteristic function and the Fang-Oosterlee (2008) COS
expansion of the log-moneyness density. Call and put payoff cosine
coefficients are computed directly (not via put-call parity), the
truncation interval is set from closed-form Heston cumulants
(FO eq. 33), and defaults `N = 160, L = 12` carry < 1e-7 absolute
error in the typical equity moneyness range of 0.85–1.15.

**Validation.** Three independent gates:

1. `tests/test_pricing/test_heston_cos.py` — 24 self-contained checks
   (sanity, parity, BSM-near-limit, truncation convergence, autodiff
   vs central FD on all 5 model parameters, JIT/vmap, golden).
2. `tests/test_quantlib_comparison/test_heston_ql.py::TestHestonCOSvsQL`
   — direct analytic-vs-analytic comparison to QuantLib's
   `AnalyticHestonEngine` across 5 seeds × 5 moneynesses × call/put.
   All 50 pass at < 5e-7 absolute tolerance.
3. `tests/test_quantlib_comparison/test_heston_ql.py::TestHestonCOSvsMC`
   — closes the open HE-1 follow-up: the Andersen-QE MC sits inside
   3 standard errors of COS at every strike (25/25).

**Calibration unlock.** `calibrate_heston` (which had been waiting on
a real pricer since shipping) is now wired to COS end-to-end in
`tests/test_calibration/test_heston_calibration.py::test_roundtrip_real_cos_pricer`
— LM recovers all 5 Heston parameters to floating-point precision on
noise-free synthetic data (max reprice residual < 1e-10).

**Follow-up still open.** Carr-Madan / Lewis FFT pricer as an
independent analytic cross-check; multi-expiry surface calibration
objective (paired with SSVI under Tier 1.2).

## Local Volatility — Session Backlog

### LV-1 — Dupire local vol + Local-Vol MC &nbsp;✅ &nbsp;*Done*

**What was done.** Closed Tier 2.3 items 1 & 2 (Dupire extraction and
Local Vol MC simulation) as a single coherent slab. Three new modules:

* `valax/pricing/analytic/dupire.py` — Gatheral IV-space Dupire formula
  in total variance `w = σ²·T`. Uses `jax.grad` for `∂_k w`, `∂_{kk} w`,
  `∂_T w` directly through the surface's `total_variance` method —
  no finite differences anywhere. Numerator clamped at zero (calendar-
  arb floating-point noise); denominator left unclamped so butterfly
  arbitrage in the input surface surfaces as NaN (intentional
  diagnostic). Module-level `RuntimeError` if `jax_enable_x64` is off
  (Dupire's 2nd derivatives are precision-sensitive).
* `valax/models/local_vol.py` — `LocalVolModel(eqx.Module)` carrying a
  duck-typed surface + scalar rate/dividend. Factory
  `from_flat_rate(...)`. The model is intentionally minimal — no
  precomputed leverage grid; `σ_loc` is recomputed from the surface
  on demand so autodiff flows through surface params (SVI a/b/ρ/m/σ)
  into MC prices cleanly.
* `valax/pricing/mc/local_vol_paths.py` — `generate_local_vol_paths`
  using `jax.lax.scan` over time with log-Euler + Itô correction. Per-
  step Dupire evaluation `vmap`-ed across paths. **Midpoint-in-time σ**
  evaluation (`t_n + 0.5·dt`) — chosen over left-endpoint because (a)
  it avoids the `T → 0` singularity of Dupire's `1/w` terms, and (b)
  it gives a half-order weak-error improvement at no extra cost.

**Substrate changes** (made together so the LV PR does not leak
abstractions):

* `valax/surfaces/_interp.py` — factored out a reusable `bilinear_2d`
  utility from the hand-rolled bilinear inside `GridVolSurface`. Same
  numerics, single test surface; preps the ground for SLV's leverage
  grid.
* `total_variance(log_moneyness, expiry)` protocol method added to
  `SVIVolSurface`, `SABRVolSurface`, and `GridVolSurface`. Duck-typed
  Dupire input; identity `total_variance(log(K/F_T), T) == σ_IV(K,T)²·T`
  preserved exactly.
* `SVIVolSurface.__call__` and `.total_variance` now extrapolate
  linearly in `T` through the origin below the first slice (constant
  IV in the `T → 0⁺` limit) rather than flat-extrapolating `w` (which
  diverged IV and zeroed `∂_T w`). Fixes the Dupire-at-short-T case.
* `heston_cos_price` rate/dividend args now default to `None` and fall
  back to `model.rate`/`model.dividend`. Backward compatible — closes
  the cross-API footgun flagged at the end of HE-2.

**Validation.** Three independent gates:

1. `tests/test_pricing/test_dupire.py` — 35 unit checks (parametrised):
   flat-SVI ≡ constant σ to 1e-10 at every probe, autodiff vs central
   FD on SVI `a`/`b`/`ρ` to 1e-4 relative, butterfly-violation NaN
   diagnostic, x64 guard, JIT/vmap, surface-protocol coverage for all
   three surface types, golden 5×5 canonical Dupire grid pinned at
   `tests/golden/v1/dupire_canonical_grid.npz`.
2. `tests/test_pricing/test_local_vol_paths.py` — 14 unit checks
   covering output shape, initial condition, BSM reprice in the flat
   limit (3·stderr) across moneyness × call/put, the **headline
   Dupire-consistency gate** (4-seed × 100k × 500-step LV MC reprices
   the input SVI surface to < 20 bp absolute IV on all 5 strikes),
   1/√N MC convergence, ATM delta finiteness via `jax.grad`, JIT and
   vmap. Golden ATM price pinned at `tests/golden/v1/local_vol_mc_canonical.npz`.
3. `tests/test_quantlib_comparison/test_dupire_ql.py` — 6 cross-checks
   against `ql.LocalVolSurface` and `ql.AnalyticEuropeanEngine` in
   the flat-IV limit (5 (σ, K/F, T) probes at 1e-6 tolerance + 1 LV MC
   reprice at 3·stderr). Non-flat smile QL cross-check is deliberately
   skipped — see test docstring; QL interpolates total variance
   bilinearly while VALAX's grid interpolates IV bilinearly, so a
   non-flat smile gate would test interpolation conventions, not
   Dupire kernel quality. The SVI-self-consistency gate above is the
   right smile validation.

**Dispatcher integration.** `engine.py::mc_price`, `engine.py::mc_price_with_stderr`,
and `recipes.py::_equity_paths` all carry a new `isinstance(model,
LocalVolModel)` branch. Five MC recipes registered for `LocalVolModel`
(European, Asian, EquityBarrier, Lookback, VarianceSwap) — barrier is
the canonical exotic where local vol differs materially from BSM.

**Calibration unlock.** SLV's two-pass calibration (Heston → vanillas;
particle method → leverage `L(S,t)`) sits directly on top of this
substrate. The COS pricer + Heston calibrator from HE-2 supply pass 1;
the LV MC paths from LV-1 supply the particle method's path generator
for pass 2. No further pricing-layer work is needed before SLV.

**Follow-up still open.**

* ~~**Milstein scheme for LV MC**~~ ✅ *Shipped under LV-2 below as an
  opt-in scheme; empirically does **not** tighten the vanilla-reprice
  gate at typical MC budgets (weak-order equality with Euler hides the
  improvement behind the MC noise floor). Useful for path-statistics-
  sensitive payoffs; default scheme remains midpoint-Euler.*
* **Local vol PDE** pricing (Tier 2.3 item 3) — Fokker-Planck forward
  on a `(S, t)` grid. Deferred; MC suffices for SLV.
* **Forward-curve term structure** — `LocalVolModel.forward_curve` is
  currently fixed to `S₀·exp((r-q)·t)`. A `Callable[[t], F(t)]` field
  would unlock surfaces calibrated against term-structured rates /
  dividends. Trivial extension once needed.
* **Variance reduction for LV MC** — the empirical bias floor on
  vanilla repricing is set by Monte-Carlo noise, not discretisation.
  A BSM control variate at a reference vol (subtracting the MC BSM
  price and adding back the analytic BSM expectation) would deliver
  the 5-bp Dupire-consistency gate that the Milstein attempt under
  LV-2 could not. Tracked as a future LV-3 item if needed before SLV
  calibration tightens the requirement.
* **SLV** (Tier 2.4) — natural next session.

### LV-2 — Milstein scheme as opt-in path generator &nbsp;✅ &nbsp;*Done (honest scope)*

**What was done.** Added a `scheme: Literal["midpoint_euler",
"milstein"] = "midpoint_euler"` keyword-only argument to
`generate_local_vol_paths`, plumbed through the unified
`mc_price_dispatch` as the namespaced `lv_scheme` market argument.
Both schemes share the midpoint-in-time σ; Milstein adds the
strong-order-1 correction $+\tfrac{1}{2}\sigma_n(\partial\sigma_{\text{loc}}/\partial k)\Delta t(Z_n^2 - 1)$
via `jax.value_and_grad(dupire_local_vol, argnums=0)` vmapped across
paths.

**The honest empirical finding.** The LV-1 backlog premise — "Milstein
would tighten the vanilla-reprice gate from 20 bp to 5 bp at the same
MC budget" — turned out to be wrong. Both schemes have weak-order 1
(Milstein only improves the *strong* order), so for vanilla
expectations they converge at the same asymptotic rate, with only a
constant-factor difference. At typical MC budgets (4 seeds × 100k
paths × {100–1000} steps on a moderate equity skew), the MC noise
floor is ~10–20 bp and washes out the constant-factor improvement:

| `n_steps` | midpoint-Euler max bp | Milstein max bp |
|:---:|:---:|:---:|
| 100  |  6.6 | 11.9 |
| 200  |  5.9 |  9.6 |
| 500  | 18.8 | 20.1 |
| 1000 | 13.5 | 14.2 |

The non-monotonic `n_steps` profile (errors don't shrink with more
steps) is the signature of noise-dominated rather than bias-dominated
error. Both schemes produce statistically-indistinguishable vanilla
prices at this budget; Milstein pays 2× per-step compute (one extra
`jax.grad`) for no measurable benefit on European, Asian, and other
expectation-style payoffs.

**Where Milstein *does* help.** Paired-seed comparison on a down-and-
out call shows Milstein's bias is reliably smaller than Euler's at
coarse `n_steps` — the strong-order improvement surfaces on path-
distribution-sensitive payoffs. Empirically on 8 paired seeds at
`n_steps=200` the average paired difference (Euler − Milstein) is
`+0.0081 ± 0.0044` (paired t-stat 5.25 — highly significant). Pinned
in `tests/test_pricing/test_local_vol_paths.py::TestSchemeComparison`.

**Validation.** 5 new tests in
`tests/test_pricing/test_local_vol_paths.py`:

* `TestSchemeOptIn::test_milstein_runs_and_returns_finite` — both
  schemes produce finite paths of the right shape.
* `TestSchemeOptIn::test_milstein_differs_from_euler_at_same_key` —
  same PRNG key, different scheme ⇒ paths differ by the Milstein
  correction (sanity).
* `TestSchemeOptIn::test_invalid_scheme_raises_value_error` —
  validation error on a bad scheme name.
* `TestSchemeOptIn::test_dispatcher_routes_lv_scheme_kwarg` —
  `mc_price_dispatch` with `lv_scheme="milstein"` correctly threads
  through the recipe layer.
* `TestSchemeComparison::test_milstein_paired_bias_no_worse_than_euler_on_barrier`
  — the path-statistics-sensitive comparison described above.

**Default stays midpoint-Euler.** Documented in the module docstring,
the `generate_local_vol_paths` signature docstring, the API reference
(`docs/api/pricing.md`), the user guide
(`docs/guide/monte-carlo.md`), and the theory section
(`docs/theory.md §4.4`). Cross-links flow consistently from each
entry point to the empirical sweep table.

**Why ship anyway.** Three reasons:

1. The opt-in mechanism is needed downstream — barrier-sensitive
   exotics under LV, and (eventually) SLV's particle-method
   calibration may want strong-order accuracy in regimes where it
   matters.
2. The scheme infrastructure (`scheme=` kwarg, dispatcher plumbing)
   establishes the pattern for future MC schemes (e.g. predictor-
   corrector, jump-adapted) without churning the public API.
3. Honest documentation of what *doesn't* work is itself valuable —
   future contributors will not waste effort re-attempting the
   Milstein-tightens-vanilla-gate path that LV-1 originally
   anticipated.

## Stochastic-Local Volatility — Session Backlog

### SLV-1 — SLV leverage calibration + SLV MC &nbsp;✅ &nbsp;*Done*

**What was done.** Closed Tier 2.4 items 1, 2, and 3 (leverage-function
calibration, SLV MC simulation, particle / kernel regression) as a
single coherent slab. Four new modules:

* `valax/surfaces/leverage.py` — `LeverageGrid` equinox pytree:
  `log_moneyness_grid × time_grid × values` with the same
  `(n_t, n_k)` (y outer, x inner) storage convention as
  `GridVolSurface.vols`. Bilinear interpolation via the project-wide
  `bilinear_2d` helper (the hook was already advertised at
  `valax/surfaces/_interp.py:9`). Only `values` is a differentiable
  leaf; the grid axes are static scaffolding. `LeverageGrid.flat()`
  builds a constant-leverage grid — the pure-Heston-limit warm start
  for calibration *and* the canonical reduction-test fixture.
* `valax/models/slv.py` — `SLVModel` carries the Heston block + rate /
  div + surface (duck-typed via `total_variance`) + leverage. The
  surface is kept on the model so re-calibration does not need
  external bookkeeping; the path generator never queries it. x64 is
  enforced at construction (`from_heston_and_leverage`) matching the
  Dupire-layer policy.
* `valax/pricing/mc/slv_paths.py` — `generate_slv_paths` returning
  joint `(spot_paths, var_paths)` of shape `(n_paths, n_steps+1)`.
  Variance leg is Andersen-QE (factored into `_qe_variance_step_factory`,
  exact in distribution per step regardless of Feller's condition).
  Log-spot leg is selectable via `scheme=`: `"midpoint_euler"` (default,
  weak-order 1) or `"milstein"` (strong-order 1, ~2× per-step cost,
  one extra `jax.value_and_grad(L)` per step). Midpoint-in-time
  leverage query mirrors the LV-1 convention — avoids the Dupire
  `T = 0` singularity at the calibration-grid boundary.
* `valax/calibration/slv.py` — `calibrate_slv_leverage` implements the
  Guyon-Henry-Labordère (2012) particle method with both estimator
  variants (`method ∈ {"particle", "kernel"}`) and an outer
  fixed-point loop (`n_iterations`). The kernel-method ridge is
  Tikhonov-style — biases the Nadaraya-Watson estimator toward the
  empirical particle mean in low-density regions, smoothing the tails
  at the cost of a small bias in the centre. End-to-end
  `calibrate_slv(...)` wraps Pass 1 (`calibrate_heston`) and Pass 2 in
  one call.

**Substrate changes** (made together so the SLV PR does not leak
abstractions):

* `valax/surfaces/__init__.py`, `valax/models/__init__.py`,
  `valax/pricing/mc/__init__.py`, `valax/calibration/__init__.py` —
  re-exports for the new public types and functions.
* `valax/pricing/mc/engine.py` — `SLVModel` branch added to
  `mc_price` and `mc_price_with_stderr` (between the `HestonModel`
  and `LocalVolModel` branches), discarding the variance leg with
  `paths, _ = ...` to match the Heston pattern.
* `valax/pricing/mc/recipes.py` — `_equity_paths` extended with an
  `slv_scheme` kwarg + `SLVModel` branch; `_equity_recipe` plumbs the
  kwarg through. Five new `@register((Instrument, SLVModel))` entries
  for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`,
  `LookbackOption`, `VarianceSwap` — same set as LV-1.
* `valax/models/slv.py` — the `leverage` field is annotated as
  `eqx.Module` (matching `LocalVolModel.surface`'s `eqx.Module`
  annotation) and `LeverageGrid` is imported lazily inside
  `from_heston_and_leverage`, to break a latent import cycle in the
  pre-existing `surfaces → sabr_surface → calibration.sabr` chain
  that the SLV additions activate.

**Validation.** Five test modules totalling 68 tests, all green in
~30 s CPU wall-clock under x64:

1. `tests/test_surfaces/test_leverage.py` — 12 unit checks: flat
   factory, bilinear at node / midpoint, flat extrapolation,
   `eqx.filter_jit`, autodiff through `values` (partition-of-unity
   check on bilinear basis), `jax.vmap` over query points, autodiff
   w.r.t. query point matches the analytic derivative of the test
   surface.
2. `tests/test_models/test_slv.py` — 6 checks: `from_heston_and_leverage`
   round-trip, pytree flatten/unflatten, `eqx.tree_at` on `kappa`,
   `jax.grad` through `leverage.values` and through Heston `xi`, x64
   guard raises `RuntimeError`.
3. `tests/test_pricing/test_slv_paths.py` — 18 checks across nine
   classes. The headline reduction test
   (`TestFlatLeverageReducesToHeston`) verifies that with `L ≡ 1` the
   SLV MC agrees with `generate_heston_paths` within 3·stderr on a
   6-cell `(K, is_call)` grid — confirming the Andersen-QE variance
   leg + approximate-correlation coupling collapses cleanly to the
   pure-Heston K-formulation. The headline Dupire-consistency gate
   (`TestDupireConsistency::test_smile_repriced_within_gate`)
   exercises the full pipeline at 250 bp.
4. `tests/test_calibration/test_slv_calibration.py` — 7 checks:
   particle/kernel-method shape sanity, Markovian-projection identity
   ($L^2 \cdot \mathbb{E}[V|k] \approx \sigma_{\text{Dupire}}^2$ within
   10 % at ATM after calibration), kernel-method dense-region match,
   kernel-method tail smoothness (across-seed variance), fixed-point
   contraction in sup-norm, invalid `method` and `n_iterations` raise.
5. `tests/test_quantlib_comparison/test_slv_ql.py` — 25 cross-checks
   (5 moneyness × 5 seeds) verifying SLV at the flat-leverage limit
   matches QuantLib's `AnalyticHestonEngine` within 3·stderr. Mirrors
   the precedent of `test_dupire_ql.py`: flat-limit only, no
   calibrated-leverage QL cross-check (methodologically incompatible
   — QL's `HestonSLVProcess` uses Fokker-Planck PDE, we use the
   particle method).

Plus one golden artifact (`slv_mc_canonical` — terminal
$(\mathbb{E}[S_T], \mathbb{E}[V_T])$ at the flat-leverage fixture,
pinned for drift detection).

**Dispatcher integration.** `mc_price_dispatch(instrument, slv_model,
config=..., key=..., spot=..., slv_scheme="midpoint_euler" |
"milstein")` works out of the box for the five registered
instruments. The dispatcher recipe count climbed from 16 to 21.

**The honest empirical finding.** The Dupire-consistency gate for SLV
sits at **~125 bp**, not the sub-20 bp LV-1 achieves. The reason is
the **particle method's accuracy ceiling**, not a discretisation
artifact:

* The calibrated Markovian-projection identity $L^2 \cdot
  \mathbb{E}[V|k] = \sigma_{\text{Dupire}}^2$ holds within ~1 %
  pointwise at ATM (verified in
  `TestParticleMethod::test_markovian_projection_identity_at_atm`).
* The bias does *not* shrink with finer `n_steps` (verified up to
  `n_steps = 2000` in dev — in fact it slightly worsens, ruling out
  log-Euler bias).
* The bias does *not* shrink dramatically with more calibration
  particles or `n_iterations` — the iterations *do* contract the
  per-iteration update (verified in
  `TestFixedPoint::test_iterations_converge`), but the converged
  fixed point still sits ~125 bp from the SVI target.
* The Heston-only (`L ≡ 1`) MC reprices the same surface within
  ~91 bp at the same MC budget. Calibration meaningfully moves SLV
  toward SVI (the sign of the wing gap flips), but overshoots by a
  ~125 bp uniform offset across strikes — well-documented behaviour
  of the kernel-regression-based particle method (see Henry-Labordère
  2009, Ch. 12; Guyon-Henry-Labordère 2012, §5).

The 250 bp test gate is the **regression-detection ceiling**, not the
achievable precision. Sub-100 bp precision under particle calibration
would require pairing the particle MC with variance reduction (control
variates) and a `(n_paths_cal, n_iterations, ridge)` sweep — explicit
roadmap item **SLV-2**.

**Calibration unlock.** Autocallable / cliquet / forward-starting
pricing is now blocked only on the instrument-side payoff engine — the
SLV model and dispatcher recipes are ready to consume them. The
SLV substrate also unlocks downstream Greeks via `jax.grad` through
`leverage.values` and through the Heston block (verified end-to-end in
`tests/test_models/test_slv.py::TestSLVGreeks`).

**Follow-up still open.**

* **SLV-2 — Sub-100 bp Dupire-consistency.** Either (a) Fokker-Planck
  PDE leverage calibration (the QuantLib-style production path, ~500
  LOC), or (b) variance reduction on the existing particle MC (~100
  LOC). The former is the right long-term answer; the latter is the
  cheap interim option.
* **SLV-3 — Autocallable / Cliquet recipes.** The instrument
  scaffolds exist; the recipes need the SLV MC + a structured-payoff
  engine that supports multi-observation barriers and forward-start
  rolling.
* **SLV-4 — SABR-LV backbone.** Same Markovian-projection machinery
  with `generate_sabr_paths` substituted for `generate_heston_paths`.
  Useful for rates SLV (CMS spread, swaption smile dynamics).
* **`heston_cos_price` signature alignment** — surfaced during SLV
  development: `calibrate_heston` calls `pricing_fn(model, K, spot,
  rate, dividend, expiry)` (6 args), but `heston_cos_price` accepts
  `(option, spot, rate, dividend, model)` (5 args). Pre-existing bug;
  affects anyone using `calibrate_heston` with the COS pricer. Not
  blocked on SLV.

## Arbitrage Detection — Session Backlog

Five small, well-scoped detectors that turn `@pytest.mark.xfail(strict=True)`
items in `tests/test_market/test_arbitrage_handling.py` into passing
assertions. Each item has a fixed test waiting for it to flip green —
removing one xfail is a real engineering deliverable.

The infrastructure (`valax.core.diagnostics` reserved exception types,
`valax.market.synthetic.arbitrage` injectors with parameterised severity,
`ArbDiagnosis` objects) is already in place; only the consumer-side
checks are missing.

### ARB-1 — Non-PSD correlation in `MultiAssetGBMModel.__init__`

**What.** Call `validate_correlation` inside the model's construction
path and raise `NonPSDCorrelationError` when `min_eig < -tol`.

**Unlocks.** Two existing xfail tests:
`TestNonPSDCorrelation::test_model_constructor_should_raise`,
`TestNonPSDCorrelation::test_paths_should_raise_on_non_psd`.

**Effort.** ~15 LOC + a kwarg `validate: bool = True` so JIT call sites
that need to skip the check can.

**Blockers.** None.

### ARB-2 — Basket-variance overshoot in correlation matrix construction

**What.** Reject any constructor input where an off-diagonal entry
falls outside `[-1, 1]`. Strictly tighter than ARB-1 (the eigenvalue
check catches this indirectly, but a structural check raises with a
clearer error).

**Unlocks.**
`TestBasketVarianceViolation::test_constructor_should_raise_automatically`.

**Effort.** ~10 LOC, share with ARB-1.

**Blockers.** None.

### ARB-3 — Calendar-spread arbitrage in `SVIVolSurface`

**What.** On surface construction (or via an explicit
`is_arbitrage_free()` checker), verify that total implied variance
`w(k, T)` is non-decreasing in `T` at every log-moneyness, and raise
`CalendarArbError` otherwise.

**Unlocks.**
`TestCalendarArbitrage::test_surface_constructor_should_reject`.

**Effort.** ~30 LOC plus a slice-loop helper.

**Blockers.** Decision on whether the check is enforced at
construction or only via an opt-in method (probably opt-in: existing
fits may produce slightly arbitrageable surfaces under noise).

### ARB-4 — Put-call parity checker for quoted strips

**What.** A standalone validator `assert_put_call_parity(calls, puts,
spot, strikes, expiry, df_factor)` that raises `PutCallParityError`
when `|C − P − (S·e^{−qT} − K·e^{−rT})|` exceeds a tolerance.

**Unlocks.**
`TestPutCallParity::test_quote_validator_should_reject`.

**Effort.** ~25 LOC + a smoke test on injected and clean data.

**Blockers.** None.

### ARB-5 — Butterfly arbitrage on a strike grid

**What.** Detect `d²C/dK² < 0` from a discrete strike grid of call
prices and raise `ButterflyArbError`. Two flavours needed: from a vol
smile (Hagan / SVI / SABR converted to call prices) and from a raw
price grid.

**Unlocks.** No xfail today — this is the *first* tier where the
arbitrage injector exists (`inject_butterfly_arb`,
`inject_non_convex_smile`, `inject_negative_density`) but the test
file has no consumer because no detector exists to be called.
Landing this detector lets us write the test cases at the same time.

**Effort.** ~40 LOC + tests.

**Blockers.** Requires a stable spec for "what does the detector
operate on": vol grid, price grid, or both. Resolving that is the
first thing to do before coding.

### ARB-6 (stretch) — Inconsistent-quote rejection in bootstrap

**What.** After `bootstrap_simultaneous` returns, verify that every
input instrument reprices to within a quote-noise tolerance, and
raise `InconsistentQuotesError` on the worst offender if any.

**Unlocks.** No xfail today (this is also greenfield).

**Effort.** ~20 LOC inside `valax/curves/bootstrap.py`.

**Blockers.** Picking the right per-instrument tolerance; defer until
ARB-1..5 are in to avoid scope creep.

---

## Monte Carlo Dispatcher — Session Backlog

Four near-term MC expansions that slot directly into the existing
`mc_price_dispatch` registry. None of them require changes to the
dispatcher itself — they are new path generators and / or `@register`
entries plus tests.

Each item lists the new code, the unlocked recipes, effort estimate,
and blockers. Ordered by expected ROI.

### MC-B1 — Hull-White short-rate MC paths

**What.** A new path generator `generate_hull_white_paths(model, T, n_steps, n_paths, key)`
producing short-rate paths plus path-wise discount factors
$DF(0, t_k) = \exp(-\int_0^{t_k} r(s)\,ds)$. Hull-White has an affine
conditional distribution, so the simulation can use exact
discretization (no Euler bias).

**Unlocks.**

- `(FixedRateBond, HullWhiteModel)` — validates curve-discounting
  against stochastic-rates MC.
- `(FloatingRateBond, HullWhiteModel)` — same.
- `(CallableBond, HullWhiteModel)` — MC cross-check for the existing HW
  trinomial tree pricer (useful for validation and for cases where
  tree step density is a concern).
- `(PuttableBond, HullWhiteModel)` — same.
- Alternative MC route for Bermudan swaptions (currently LMM-only).

**Effort.** ~60 LOC generator + ~120 LOC recipes + tests.

**Blockers.** None. Hull-White model and callable/puttable tree
already exist.

### MC-B2 — American / Bermudan equity via LSM

**What.** Lift the Longstaff-Schwartz engine from
`valax/pricing/mc/bermudan.py` (currently LMM-specific) into a generic
`lsm_backward_induction(paths, exercise_indices, payoff_fn, continuation_basis)`
that operates on any single-asset path array. Register recipes for
early-exercise equity options.

**Unlocks.**

- `(AmericanOption, BlackScholesModel)` — MC alternative to the CRR
  binomial tree; validation target.
- `(AmericanOption, HestonModel)` — first American option pricer under
  stochastic vol in VALAX.
- Foundation for a new `BermudanEquityOption` instrument if one is
  needed (the existing `AmericanOption` already covers the exercise
  schedule case via discrete observation dates).

**Effort.** ~80 LOC LSM lift + polynomial basis for equity state
(log-spot is the natural choice) + ~100 LOC recipes + tests.

**Blockers.** Minor refactor: the LMM LSM in `bermudan.py` is tightly
coupled to the `LMMPathResult` pytree; needs a cleaner separation of
the regression core from the tenor-aware payoff evaluation.

### MC-B3 — FX Monte Carlo

**What.** Add a small `FXGBMModel` (domestic rate + foreign rate + FX
vol) and a `generate_fx_gbm_paths` wrapper that reuses
`generate_gbm_paths` with drift $r_d - r_f$. Register recipes for FX
instruments.

**Unlocks.**

- `(FXVanillaOption, FXGBMModel)` — MC validation of Garman-Kohlhagen.
- `(FXBarrierOption, FXGBMModel)` — the *only* tractable pricing route
  for discretely-monitored FX barriers right now; the instrument is
  defined but has no pricer.
- Foundation for `(QuantoOption, ...)` — needs correlated FX + asset MC
  which leans on the multi-asset infrastructure already shipped.
- Foundation for `(TARF, FXGBMModel)` — path-dependent early
  termination has no closed form; MC is the only route.

**Effort.** ~40 LOC `FXGBMModel` + wrapper + ~80 LOC recipes + tests.

**Blockers.** Decision needed: separate `FXGBMModel` vs. reusing
`BlackScholesModel` with a documented convention (`rate = r_d`,
`dividend = r_f`). A wrapper model is cleaner for downstream FX
smile / volatility-surface integration.

### MC-B4 — Autocallable engine

**What.** A generic "observation-date scanner" utility for
path-dependent payoffs that need to check multiple intermediate
observations — autocall barrier hit, coupon barrier hit, knock-in
barrier touch. Plus a dedicated `autocallable_payoff` function
handling the memory (phoenix) feature and partial redemption at
autocall events.

**Unlocks.**

- `(Autocallable, BlackScholesModel)` — single-underlying phoenix
  notes.
- `(Autocallable, HestonModel)` — with stochastic vol (more realistic
  for short-dated barriers).
- `(Autocallable, MultiAssetGBMModel)` — basket-referenced
  autocallables (Asian-market standard product; multi-hundred-billion
  dollar market).
- Building block for `Cliquet` / ratchet pricing (the observation
  scanner generalizes).

**Effort.** ~120 LOC scanner + payoff + ~100 LOC recipes + tests.
Largest of the four, but scanner is reusable.

**Blockers.** None for single-asset. The multi-asset variant is fully
unblocked by the multi-asset MC infrastructure shipped in this
session.

### Summary table

| ID | New code | Recipes added | Effort | Blocking |
|----|----------|--------------|--------|----------|
| MC-B1 | `generate_hull_white_paths` | 4 (bonds + callable/puttable + Bermudan alt.) | ~60 + tests | None |
| MC-B2 | Generic LSM lift | 2 (AmericanOption × GBM/Heston) | ~80 + tests | Minor refactor of `bermudan.py` |
| MC-B3 | `FXGBMModel` + wrapper | 2+ (FX vanilla/barrier, foundation for quanto/TARF) | ~40 + tests | Model-design decision |
| MC-B4 | Observation scanner + autocallable payoff | 3 (Autocallable × 3 models) | ~120 + tests | None (multi-asset unblocked) |

Total: ~300 LOC of new functionality + tests takes the dispatcher from
16 recipes to **~28 recipes** covering every currently-defined
instrument except the credit derivatives (which need a survival curve,
tracked separately under Tier 3.4).

---

## Production Priority Roadmap (Snapshot: June 2025)

The tier-based feature inventory below is comprehensive but feature-area oriented. This section reorganizes the most critical work into **execution priorities** — what a bank evaluating VALAX would need to see, in roughly the order it should be built.

This is a **point-in-time snapshot**. Priorities will shift as items are completed and real user feedback arrives.

### Priority 1 — Foundational Infrastructure *(Blocks Everything)*

These items are hard prerequisites. Every downstream feature, product, and integration depends on them.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P1.1** | **Business Calendars & Date Adjustment** | Every trade in a bank settles on business days. Wrong dates = wrong cashflows = wrong P&L. Precomputed holiday arrays (TARGET, NYSE, SOFR/Fed, London, Tokyo) with modified-following/preceding and end-of-month conventions. | 1.3 |
| **P1.2** | **Cashflow Engine (Legs, Stubs, Compounding)** | Real swaps have short/long stubs, compounding-in-arrears, amortizing notionals, fixing histories. The current schedule generator can't represent production trades. `FixedLeg` / `FloatingLeg` pytrees with full convention support. | 1.4 |
| **P1.3** | **CI/CD Pipeline** | No bank adopts a library without automated testing. GitHub Actions with: lint, type-check, full test suite, QuantLib comparison, coverage reporting. A gate before anything else ships. | — |
| **P1.4** | ~~**Short-Rate Models (Hull-White, G2++)**~~ ✅ **Hull-White implemented** | One-factor Hull-White with analytic ZCB pricing (exact-fit to initial curve), trinomial tree for callable/puttable bonds. G2++ and swaption calibration (Jamshidian) are follow-ups. | 2.1 |

### Priority 2 — Production Pricing Capabilities

These unlock entire asset classes and bring pricing to the speed and accuracy required by trading desks.

| # | Task | Why It's Critical | Tier Ref |
|---|------|-------------------|----------|
| **P2.1** | ~~**Heston Semi-Analytic (COS / Fourier)**~~ ✅ **Fang-Oosterlee COS implemented** | Lord-Kahl "Little Trap" characteristic function + Fang-Oosterlee (2008) COS expansion with closed-form Heston cumulants. Agrees with QuantLib's `AnalyticHestonEngine` to < 5e-7 absolute across the seed × moneyness × call/put grid. End-to-end Heston calibration roundtrip recovers all 5 parameters to floating-point precision on noise-free synthetic data. | 2.2 |
| **P2.2** | ~~**Local Volatility + SLV**~~ ✅ **Both shipped** | LV (Tier 2.3) via `valax/pricing/analytic/dupire.py` + `valax/pricing/mc/local_vol_paths.py` (backlog **LV-1**). SLV (Tier 2.4) via `valax/calibration/slv.py` (Guyon-Henry-Labordère particle method + optional kernel-ridge stabilisation + outer fixed-point loop) and `valax/pricing/mc/slv_paths.py` (Andersen-QE variance + log-Euler/Milstein log-spot, registered with the unified MC dispatcher for European/Asian/Barrier/Lookback/VarianceSwap). See backlog **SLV-1**. | 2.3, 2.4 |
| **P2.3** | **FX Derivatives (Garman-Kohlhagen, Barriers, Delta Conventions)** | FX is one of the largest derivatives markets with unique conventions (delta-space quoting). Unlocks an entire asset class. | 3.3 |
| **P2.4** | **Credit Derivatives (CDS, Survival Curves)** | Survival curves are a prerequisite for XVA (CVA). CDS pricing and hazard rate bootstrapping are foundational. Unlocks credit trading and the entire XVA workstream. | 3.4 |

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
- [x] **Multi-curve framework** — `MultiCurveSet` with OIS discount curve + tenor-specific forward curves, dual-curve swap bootstrap
- [x] **Jacobian of discount factors w.r.t. input rates** via autodiff (free with JAX, verified in tests)
- [ ] **Basis curve** support (e.g., 1M vs 3M SOFR basis)
- [ ] **Turn-of-year effects** and stub handling

**Why:** Every rates product needs proper forward curves. Manual pillar specification doesn't scale.

**Approach:** Root-finding via `optimistix.root_find` on the system of bootstrapping equations. Each instrument (deposit, swap) provides one equation. The curve is a pytree — autodiff gives the Jacobian for free.

**Status:** Core bootstrapping implemented in `valax/curves/bootstrap.py` and `valax/curves/multi_curve.py`. Bootstrap instrument pytrees (`DepositRate`, `FRA`, `SwapRate`) in `valax/curves/instruments.py`. Both sequential and simultaneous methods produce `DiscountCurve` objects compatible with all existing pricing functions. Dual-curve bootstrap supports OIS discounting with separate forward projection curves.

### 1.2 Volatility Surface Construction

- [x] **Vol surface object** — `GridVolSurface` with bilinear interpolation on (strike, expiry) grid
- [x] **SABR smile per expiry** — `SABRVolSurface` with per-expiry SABR calibration and parameter interpolation
- [x] **SVI parametric fitting** — `SVIVolSurface` with Gatheral's SVI parameterization and LM calibration
- [ ] **SSVI** — global arbitrage-free parameterization (Gatheral-Jacquier)
- [ ] **Sticky-strike vs sticky-delta** conventions
- [x] **Local vol extraction** from implied vol surface (Dupire formula) — shipped under LV-1

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

- [ ] **Hull-White (one-factor)** — mean-reverting short rate; analytic bond prices and swaption prices
- [ ] **G2++ (two-factor)** — two correlated Hull-White factors; richer term structure dynamics
- [ ] **CIR** — square-root diffusion, ensures positive rates
- [ ] **Vasicek** — Ornstein-Uhlenbeck for rates (allows negative rates)
- [ ] **Calibration to swaption surface** for Hull-White / G2++

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

- [x] **Dupire local vol** extraction from implied vol surface — shipped in `valax/pricing/analytic/dupire.py` (Gatheral IV-space form, autodiff through `SVIVolSurface`). See backlog item **LV-1** below.
- [x] **Local vol MC simulation** — shipped in `valax/pricing/mc/local_vol_paths.py` (`jax.lax.scan` + log-Euler with midpoint-in-time σ). Hooked into the unified MC dispatcher for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap`.
- [ ] **Local vol PDE** pricing (Fokker-Planck or backward Kolmogorov)

**Why:** Local vol is the standard model for exotic equity derivatives. It matches the entire vol surface by construction.

**Approach:** Dupire formula involves derivatives of the implied vol surface — autodiff through the surface pytree. Simulation via `lax.scan` with state-dependent diffusion.

### 2.4 Stochastic-Local Volatility (SLV)

- [x] **Leverage function** calibration — shipped in `valax/calibration/slv.py` (Guyon-Henry-Labordère particle method with optional kernel-ridge stabilisation, outer fixed-point loop via `n_iterations`). See backlog **SLV-1** below.
- [x] **SLV MC simulation** — shipped in `valax/pricing/mc/slv_paths.py` (Andersen-QE variance leg + log-Euler/Milstein log-spot leg, midpoint-in-time leverage query, approximate-correlation QE coupling). Registered with the unified MC dispatcher for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap`.
- [x] **Particle / kernel regression** — both estimators available via the `method=` kwarg on `calibrate_slv_leverage`.

**Why:** SLV is the industry standard for exotic equity pricing. It combines the smile-matching of local vol with realistic dynamics of stochastic vol.

**Approach:** Two-pass calibration: (1) calibrate Heston to vanillas, (2) compute leverage function via MC particle method. Leverage function stored as a 2D grid pytree (`valax/surfaces/leverage.py::LeverageGrid`).

**Open follow-up.** Sub-100 bp Dupire-consistency accuracy requires Fokker-Planck PDE calibration (cf. QuantLib's `HestonSLVProcess`); the particle-method ceiling at moderate budgets is ~100-250 bp. Tracked as **SLV-2** in the session backlog.

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

- [ ] **Longstaff-Schwartz (LSM)** regression for early exercise on LMM paths
- [ ] **American Monte Carlo** with basis function selection (polynomials, neural network regressors)
- [ ] **Lower/upper bound** estimation (Andersen-Broadie)

**Why:** Bermudan swaptions are among the most traded IR derivatives. Every rates desk prices them daily.

**Approach:** Generate LMM paths (already have this). At each exercise date, regress continuation value on state variables. Use `jnp.polyval` or a small neural network for regression. The exercise boundary is differentiable for Greeks.

### 3.2 Callable and Puttable Bonds

- [ ] **Callable bond** pricing via backward induction on Hull-White tree/PDE
- [ ] **OAS (option-adjusted spread)** calculation
- [ ] **Effective duration and convexity** via autodiff on OAS

**Why:** Callable bonds are a massive market (most corporate bonds are callable). OAS is the standard risk metric.

**Approach:** Requires Hull-White (Tier 2.1). Build a trinomial tree or use PDE with early exercise boundary. OAS is a shift to the discount curve — autodiff gives sensitivities.

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

- [ ] **Survival curve** construction from CDS spreads
- [ ] **CDS pricing** — protection and premium leg valuation
- [ ] **Hazard rate bootstrapping** from CDS term structure
- [ ] **CDO tranche** pricing (Gaussian copula, base correlation)

**Why:** Credit derivatives are essential for bank trading books and CVA computation.

**Approach:** Survival curves as `eqx.Module` pytrees (analogous to discount curves). CDS pricing is cashflow discounting with survival probabilities. CDO requires copula simulation — JAX's PRNG system handles this well.

### 3.5 Inflation Derivatives

- [ ] **Zero-coupon inflation swap** pricing
- [ ] **Year-on-year inflation swap** pricing
- [ ] **Inflation cap/floor** — Black-76 on forward CPI ratio
- [ ] **Seasonality adjustment** for monthly CPI

**Why:** Inflation markets have grown significantly. Real-money investors use inflation swaps for liability hedging.

**Approach:** Inflation curve as a pytree mapping dates to forward CPI levels. Pricing is standard cashflow discounting with CPI ratios.

### 3.6 Equity Exotics

- [ ] **Autocallable / phoenix notes** — path-dependent with early redemption barriers, coupon barriers, knock-in puts
- [ ] **Worst-of basket options** — multi-asset with correlation
- [ ] **Lookback options** — floating and fixed strike
- [ ] **Touch / no-touch / one-touch** — digital barriers
- [ ] **Range accruals** — accrue coupon while spot is in range
- [ ] **Compound options** — option on an option
- [ ] **Variance / volatility swaps** — realized vol payoffs

**Why:** Structured products desks sell these daily. Autocallables alone are a multi-hundred-billion dollar market.

**Approach:** All priced via MC with appropriate payoff functions. Multi-asset requires correlated path generation (Cholesky on correlation matrix). Autocallables need path-wise barrier monitoring. SLV model (Tier 2.4) is the standard for pricing.

### 3.7 CMS and Spread Options

- [ ] **CMS rate convexity adjustment** (replication or SABR-based)
- [ ] **CMS cap/floor** pricing
- [ ] **CMS spread options** — option on spread between two CMS rates
- [ ] **Range accrual on CMS** — structured coupon products

**Why:** CMS products are actively traded in rates markets. The convexity adjustment is non-trivial and a classic quant problem.

**Approach:** CMS convexity adjustment via static replication (Hagan) or SABR integration. Spread options via 2D copula or Margrabe-like extensions.

### 3.8 Mortgage-Backed Securities and Convertible Bonds

- [ ] **MBS** — prepayment models (PSA, CPR), OAS, effective duration
- [ ] **Convertible bonds** — equity-credit hybrid with call/put features, PDE or tree pricing

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

**Status:** Core framework implemented in `valax/risk/`. Parametric (normal/t) and historical scenario generation, curve shocks (parallel, steepener, flattener, butterfly, key-rate), full-revaluation VaR/ES via vmapped repricing. Parametric VaR via delta-normal with autodiff gradients. See the [Risk guide](guide/risk.md).

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

Machine learning integration and computational scaling.

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

- [ ] **ADI (Alternating Direction Implicit)** for 2D PDEs (e.g., Heston PDE in (S, v) space)
- [ ] **Sparse grid** methods for higher dimensions
- [ ] **Neural PDE solvers** (PINNs) for high-dimensional problems

**Why:** Current PDE solver is 1D only. Many important models (Heston, local-stochastic vol, multi-asset) require 2D+ PDE solvers.

**Approach:** ADI scheme decomposes 2D operator into 1D sweeps — reuse existing tridiagonal solver via lineax. Sparse grids reduce curse of dimensionality.

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
