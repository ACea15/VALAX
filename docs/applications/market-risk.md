# Market Risk & Model Validation

*How VALAX serves the two middle-office functions that share the same tooling — **Market Risk Management (MRM)** on the trading book, and **Model Validation** as the SR 11-7 / TRIM independent challenger to the front office.*

These are the [audiences](index.md) VALAX is best positioned to serve *today*, and this page is the pitch to both. Where [Regulatory: FRTB & CCAR](regulatory.md) covers the compliance outputs and [Treasury & ALM](treasury.md) covers the banking book, this page covers the **daily production loop of the middle office** — sensitivity ladders, limits, VaR/ES, backtests, P&L explain, and the challenger benchmarking that Model Validation is regulatorily required to run.

The theme: MRM and Model Validation both need an *analytics platform they own*, distinct from the front-office pricers they oversee. VALAX is architected exactly for that role: pure functions, deterministic replay, autodiff Greeks, `vmap` over books, and a QuantLib comparison test suite already in the tree.

---

## 1. What MRM and Model Validation actually do

Two distinct functions, one shared toolkit.

### 1.1 Market Risk Management (MRM)

Middle-office function reporting to the CRO. Owns:

| Responsibility | Cadence |
|---|---|
| End-of-day sensitivity computation across the trading book (DV01, delta, vega, gamma, ladders) | EOD |
| Full-revaluation VaR / ES and parametric VaR | EOD |
| Limit monitoring — per-desk DV01, vega, gamma, VaR limits | Intraday + EOD |
| Named stress scenarios (parallel, steepener, crash, credit widening) and named-stress limits | EOD + weekly |
| Backtesting: breach counting, Basel traffic light, Kupiec, Christoffersen | Monthly (rolling 250d) |
| P&L attribution / explain — decomposing yesterday's P&L to risk factors | EOD |
| Model performance review — where the risk engine under- or over-predicted | Monthly |
| Ad-hoc "what-if" for the CRO and desk heads | Intraday, on demand |

MRM is *not* the front office and *not* Finance. Its job is to be an independent, defensible, trader-friendly-but-not-trader-controlled view of the same book.

### 1.2 Model Validation (SR 11-7 / TRIM)

Second-line function, often reporting to a Chief Model Risk Officer. Owns:

| Responsibility | Cadence |
|---|---|
| Independent implementation and benchmarking of every pricing model used for capital or limits | Annual + on any material change |
| Boundary-case, edge-case, and arbitrage testing of pricers | Annual |
| Sensitivity analysis: does the Greek change smoothly across parameter space? | Annual |
| Ongoing performance monitoring: does the model still track observed prices? | Quarterly |
| Documented evidence pack for regulator inspections | Continuous |

The single defining requirement — the reason this function exists — is that the challenger implementation **must not derive from the same codebase as the champion**. That is a hard architectural constraint, not a preference.

### 1.3 Why they share a toolkit

MRM's daily production loop and Model Validation's challenger pricing consume the *same* mathematical primitives: an independent pricer, exact Greeks, deterministic replay, scenario support. They differ in cadence (EOD vs annual) and in audience (CRO vs regulator), but the code path is the same. A team can deploy one VALAX instance and serve both functions from it.

---

## 2. The daily MRM loop, mapped to VALAX

```
┌──────────────────────────────────────────────────────────────────┐
│  Daily MRM production loop                                       │
│                                                                  │
│  1. Refresh MarketData ────────────────► valax/market/data.py   │
│     (curves, spots, vols, dividends)     valax/curves/           │
│                                                                  │
│  2. Reprice book at today's close ────► pricing_fn +             │
│     (HPL versus prior close)              portfolio.batch         │
│                                                                  │
│  3. Compute full sensitivity ladder ──► compute_ladder           │
│     (Δ, γ, vanna, volga, DV01,           greeks/autodiff         │
│      rate γ, cross-gamma)                                        │
│                                                                  │
│  4. Waterfall P&L explain ────────────► waterfall_pnl_report    │
│     (10 rungs vs. actual repricing)                              │
│                                                                  │
│  5. Bucketed limits view ─────────────► bucket_sensitivity_     │
│     (desk / sector / tenor)              ladder                  │
│                                                                  │
│  6. Full-reval VaR / ES ──────────────► hypothetical_pnl_       │
│     (10 000 historical or MC scenarios)   vector, value_at_risk, │
│                                          expected_shortfall     │
│                                                                  │
│  7. Named stress P&L ─────────────────► stress_scenario,        │
│     (crash, steepener, credit widen)     steepener, butterfly    │
│                                                                  │
│  8. Rolling 250-day backtest ─────────► var_breaches,           │
│                                          kupiec_pof,             │
│                                          christoffersen_*,       │
│                                          basel_traffic_light     │
│                                                                  │
│  9. FRTB PLA (for desks on IMA) ──────► pla_spearman, pla_ks,   │
│                                          pla_traffic_light       │
│                                                                  │
│  10. Limits monitor + breach queue ───► arithmetic on ladder    │
│      (send breaches to trader / CRO)     + bucket outputs       │
└──────────────────────────────────────────────────────────────────┘
```

Every stage has a one-line VALAX primitive today. This is not aspirational — the guide walkthroughs in [Risk & Scenarios](../guide/risk.md) already exercise this loop end-to-end on synthetic books.

---

## 3. The core primitive: one ladder, every audience

The single most important object in the MRM stack is the `SensitivityLadder`. Once computed (one `jax.grad` + one `jax.hessian` pass on the pricer), it drives:

- **The trader's morning Greeks screen** — sliced by asset.
- **The desk head's DV01 by tenor** — sliced by pillar.
- **The CRO's cross-gamma exposure** — the `cross_spot_rate` block.
- **The FRTB SBA capital number** — bucketed to regulatory vertices via `bucket_sensitivity_ladder`.
- **The PLA test's RTPL vector** — contracted against 250 days of scenarios.
- **The P&L explain waterfall** — 10-rung decomposition of yesterday's move.
- **The limits monitor** — thresholded per-bucket sum.

```python
from valax.risk import compute_ladder, bucket_sensitivity_ladder, tenor_bucket_map

# One expensive call — Hessian on the pricer
ladder = compute_ladder(pricing_fn, instruments, base_market)

# Cheap arithmetic from here on
trader_view    = ladder.delta_spot                                    # per-asset delta
desk_view      = ladder.delta_rate                                    # per-pillar DV01
cro_cross      = ladder.cross_spot_rate                               # cross-gamma block
frtb_view      = bucket_sensitivity_ladder(ladder, rate_bucket=tenor_bm).delta_rate
```

The MRM engineer's mental model: **the ladder is the noun; every dashboard is a verb.** New views are five-line additions, not new subsystems.

---

## 4. Full-revaluation VaR and ES

MRM's regulatory VaR is full-revaluation on 250 days of historical scenarios (Basel) or on parametric Monte Carlo (internal). Both are one call:

```python
from valax.risk import (
    historical_scenarios, hypothetical_pnl_vector,
    value_at_risk, expected_shortfall,
)

# 250-day historical scenarios
scenarios = historical_scenarios(returns_matrix, n_assets, n_pillars)

# Full re-revaluation of the entire book under each scenario
pnl_vec = hypothetical_pnl_vector(pricing_fn, instruments, base_market, scenarios)
# pnl_vec.shape == (n_scenarios,)

var_99 = value_at_risk(pnl_vec, confidence=0.99)
es_975 = expected_shortfall(pnl_vec, confidence=0.975)
```

Under the hood, `jax.vmap` handles both axes — 10 000 scenarios × 5 000 instruments = 50 000 000 repricings compiled into one JIT call. On a single GPU this is minutes; on CPU it's competitive with the vendor risk engines it displaces.

For desks that want the faster parametric approximation as a cross-check, `parametric_var(...)` gives the delta-normal answer from the ladder and covariance in one line — and the *gap* between full-reval and parametric is itself a useful diagnostic on portfolio convexity.

---

## 5. Backtesting and traffic-light governance

Every 99% VaR forecast must be backtested on a rolling 250-day window under Basel. VALAX ships the full toolkit:

```python
from valax.risk import (
    var_breaches, kupiec_pof, christoffersen_conditional_coverage,
    basel_traffic_light,
)

breaches = var_breaches(actual_pnl_series, var_forecast_series)
kupiec_pof(breaches, confidence=0.99)                    # LR_uc + p-value
christoffersen_conditional_coverage(breaches, 0.99)      # joint rate + independence
zone = basel_traffic_light(int(breaches.sum()), 250, 0.99)
```

The zone (green / yellow / red) drives the capital multiplier directly. For desks on IMA, this is joined by the FRTB PLA test (already covered in [Regulatory: FRTB & CCAR § 2.3](regulatory.md#23-pl-attribution-pla-the-gate-to-ima)).

**Why this matters for MRM specifically:** running these tests inside the same code path that produces the VaR forecast means the backtest results reconcile with the forecast by construction. In vendor stacks the forecast comes from one system and the backtest from another, and the reconciliation is a permanent operational tax.

---

## 6. Limits monitoring

Every desk trades under a set of limits — DV01 by tenor, vega by expiry, gamma by asset, total VaR. MRM's job is to check every position addition against those limits *before* trading loosens them.

VALAX makes this trivial because the ladder is *just an array*:

```python
from valax.risk import compute_ladder, bucket_sensitivity_ladder

ladder = compute_ladder(pricing_fn, desk_positions, base_market)
bucketed = bucket_sensitivity_ladder(ladder, rate_bucket=desk_tenor_bm)

# Per-tenor DV01 vs. per-tenor limit
dv01_by_tenor = bucketed.delta_rate                  # shape (n_tenors,)
breaches = jnp.abs(dv01_by_tenor) > desk_dv01_limits # shape (n_tenors,) boolean
```

For pre-trade checks the incremental cost is negligible because `jax.grad` on the augmented portfolio uses the same JIT-compiled graph — no re-priced whole book. The result: **limit checks that are provably consistent with the EOD risk report**, because they came from the same code path.

---

## 7. P&L explain — the trader-vs-MRM dialogue

The most politically charged conversation on any trading floor happens at 09:00 the morning after a big move: the trader claims the loss was "market", MRM claims it was "the position", the CRO wants a number. VALAX's `waterfall_pnl_report` answers that question with 10 rungs and a residual:

```python
from valax.risk import compute_ladder, waterfall_pnl_report, apply_scenario
from valax.market import MarketScenario

# Yesterday's actual factor moves as a scenario
overnight = MarketScenario(
    spot_shocks=today.spots - yesterday.spots,
    vol_shocks=today.vols - yesterday.vols,
    rate_shocks=today.rates - yesterday.rates,
    dividend_shocks=today.dividends - yesterday.dividends,
)

ladder = compute_ladder(pricing_fn, positions, yesterday_market)
wf = waterfall_pnl_report(pricing_fn, positions, yesterday_market, overnight, ladder=ladder)

# The one-line answer the CRO wants:
print(f"Δ:{wf.delta_spot:+.0f} ν:{wf.delta_vol:+.0f} ρ:{wf.delta_rate:+.0f} "
      f"γ:{wf.gamma_spot:+.0f} vanna:{wf.vanna_pnl:+.0f} volga:{wf.volga_pnl:+.0f} "
      f"actual:{wf.actual:+.0f} unexplained:{wf.unexplained:+.0f}")
```

Every rung has a name a trader recognises; the `unexplained` residual is the honest measure of "how well the risk engine understands this book". A systematically large or biased unexplained is the single most actionable model-improvement signal MRM produces.

---

## 8. Model Validation: the independent challenger

Model Validation exists because regulators (SR 11-7 in the US, TRIM in Europe) require an independent implementation of every model that drives capital or limits. The independent implementation must not share code with the champion.

This creates a **permanent, budgeted need** for a second pricing stack that most banks under-serve because building one from scratch is expensive. VALAX is architected precisely to fill that role:

### 8.1 A different technology stack

Front-office pricers are C++ (Murex, Numerix, in-house). VALAX is Python + JAX + XLA. That is *genuine* independence — not a Python wrapper over the same C++, but a mathematically-equivalent implementation compiled by a different toolchain against different linear-algebra kernels. This is exactly the boundary a regulator looks for.

### 8.2 A shipped QuantLib comparison suite

The `tests/test_quantlib_comparison/` package already benchmarks VALAX against [QuantLib](https://www.quantlib.org/) — the industry's most widely-used open-source financial library — across:

- European options (Black-Scholes, dividends, deep ITM/OTM)
- Fixed income (bonds, YTM, duration, convexity)
- Heston and SABR pricing + calibration
- Caps, floors, swaptions, cap-strip on caplet vols
- Monte Carlo (GBM and Heston paths)
- PDE (Crank-Nicolson) and lattice (CRR binomial)
- Stochastic-local vol (Dupire, SLV surfaces)
- Risk Greeks

For Model Validation this comparison suite is the starting point of a submission pack. It is the difference between *"we have a challenger"* and *"we have a challenger with a maintained, versioned, git-tracked regression suite against the industry reference implementation"*. See [architecture/quantlib-validation-pyramid.md](../architecture/quantlib-validation-pyramid.md) for the layered validation strategy behind the suite.

### 8.3 Autodiff Greeks eliminate the bump-noise argument

The single most common Model Validation dispute is bump-width: the challenger and champion agree on price but disagree on delta because they used different bump sizes. VALAX's Greeks come from `jax.grad`, so they are exact by construction — the challenger delta is *the* delta, and any residual gap with the champion isolates a genuine model discrepancy rather than a numerical-differentiation artefact.

### 8.4 Deterministic replay

Every pricing function has the signature `V(instrument, MarketData) → price` and is a pure function with no mutable state. Model Validation can replay yesterday's, last quarter's, or last year's book bit-for-bit — which is exactly the audit trail supervisory inspection requires.

---

## 9. Why the middle office is the sweet spot commercially

Compared to the trading desk and the back office (see [Applications overview](index.md)), MRM and Model Validation are where every VALAX strength lines up with a real buyer need:

| Property | Front office (trading) | **MRM / Model Validation** | Back office |
|---|---|---|---|
| Incumbent lock-in | 🔴 Deep — decade-old C++ pricers wired into OMS | 🟢 Weak — MRM often builds its own | N/A |
| Regulator-mandated need | 🟡 Indirect | 🟢 Direct (SR 11-7 / FRTB / Basel) | ❌ |
| Batch cadence tolerates JIT warm-up | 🔴 Sub-ms streaming | 🟢 EOD + monthly | ❌ |
| Integration surface (OMS / market data) | 🔴 Huge | 🟢 Small — CSV in, CSV out | ❌ |
| Willingness to pay for independence | 🔴 Prefers vendor pricer | 🟢 Independence *is* the value | ❌ |
| Existing budget for build/buy | 🟡 Limited outside vendor extension | 🟢 Standing analytics budget | ❌ |

Neither the front office nor the back office would fund a VALAX deployment today. MRM and Model Validation would — and the CRO signs the invoice.

---

## 10. Coverage today vs. roadmap

| MRM / Validation need | Status | Component(s) |
|---|---|---|
| First- and second-order sensitivity ladders (autodiff) | ✅ | `compute_ladder`, `SensitivityLadder` |
| Waterfall P&L explain (10 rungs + unexplained) | ✅ | `waterfall_pnl_report`, `waterfall_pnl` |
| Full-revaluation VaR / ES (vmapped) | ✅ | `hypothetical_pnl_vector`, `value_at_risk`, `expected_shortfall` |
| Parametric delta-normal VaR | ✅ | `parametric_var` |
| Historical, parametric, and named-stress scenarios | ✅ | `historical_scenarios`, `parametric_scenarios`, `stress_scenario`, `steepener`, `butterfly`, `flattener` |
| Basel VaR backtest (Kupiec, Christoffersen, traffic light) | ✅ | `var_breaches`, `kupiec_pof`, `christoffersen_conditional_coverage`, `basel_traffic_light` |
| FRTB PLA test (Spearman + KS + d558 zone) | ✅ | `pla_spearman`, `pla_ks`, `pla_traffic_light` |
| Regulatory bucketing (FRTB / SIMM / sector / PCA) | ✅ | `tenor_bucket_map`, `equal_weight_bucket_map`, `pca_jacobian`, `bucket_sensitivity_ladder` |
| Model Validation QuantLib comparison suite | ✅ | `tests/test_quantlib_comparison/` (14 modules) |
| Deterministic replay for audit | ✅ | Pure functions + integer-ordinal dates |
| Multi-curve `MarketData` (OIS + SOFR + basis + XCCY) | 📋 | Top-priority roadmap item |
| Vol-surface risk factors (grid / SABR / SVI param risk) | ✅ / 📋 | Bucketing/Jacobian machinery shipped; full grid shocks planned |
| SBA capital aggregation (inter-bucket correlations) | 📋 | Bucket-level Δ + γ computed; correlation formulas planned |
| Named factor registry (`IR.OIS.USD.5Y`, …) | 📋 | Positional layout today; registry planned |
| Real-time streaming pricing service | 📋 | Vision-tier — service layer + market-data adapters |

The daily MRM production loop and the Model Validation challenger pack are **both fully achievable today** with the shipped components. Multi-curve and SBA aggregation are the two roadmap items that would close the last gaps for a bank-wide production deployment.

---

## 11. Why the JAX foundation matters for MRM specifically

| Design choice | MRM / Validation payoff |
|---|---|
| **Autodiff Greeks** | No bump-and-reprice, no bump-width disputes with the champion, no bump noise in the RTPL–HPL residual. Full second-order ladder in one `jax.hessian` call. |
| **`vmap` over scenarios and instruments** | A 10 000-scenario VaR on a 5 000-instrument book is one JIT call. Overnight batches finish before the CRO's morning meeting. |
| **One `MarketData` pytree** | `jax.grad(V)(market)` returns sensitivities to *every* factor in one pass — no per-factor plumbing to maintain when the FRTB factor list expands. |
| **Bucketing as a matrix multiply** | Trader view, desk view, CRO view, FRTB SBA view, SIMM view, PCA view — same ladder, different `A`. |
| **Pure functions + integer-ordinal dates** | Deterministic replay for validators and auditors; the entire risk batch fits inside `@eqx.filter_jit` with no Python `datetime` on the hot path. |
| **QuantLib comparison suite in the tree** | Model Validation submissions start with a shipped, versioned, regression-tested evidence pack rather than building one from scratch. |

The net effect: MRM and Model Validation share the same code path with the trading desk's Greeks and the CRO's stress report — auditable, deterministic, and *provably* consistent across audiences.

---

## 12. The pitch, tailored to the buyer

**To the Head of MRM:**
*A production-ready middle-office analytics stack: autodiff ladders, vmapped VaR/ES, waterfall P&L explain, Basel backtesting toolkit, and FRTB PLA — all in the same code path. Your morning-after P&L dialogue with the desk is one call, and the number reconciles by construction with the FRTB submission your CFO sends the regulator.*

**To the Head of Model Validation:**
*An independent-by-construction challenger implementation, in Python + JAX rather than C++, with a QuantLib comparison suite already in the tree covering options, fixed income, Heston, SABR, caps, swaptions, MC, PDE, lattice, SLV, and Greeks. Your annual validation cycle starts from a shipped evidence pack.*

**To the CRO:**
*The same pipeline computes your trader's morning Greeks, your desk head's DV01, your firm-wide 97.5% ES, and your FRTB PLA zone — from the same array, sliced differently. When any of those numbers changes, you can walk backward through the pipeline to find out why, with no vendor black box in the middle.*

**To the Chief Model Risk Officer:**
*SR 11-7 requires independence and defensible documentation. VALAX gives you both: a different technology stack, deterministic replay, exact Greeks, and a maintained regression suite against the industry's reference open-source library. Your submission packs stop being one-off spreadsheets.*

---

## 13. Where to read next

- **The engineering view of the pipeline these functions run on** → [Risk: End-to-End](../risk-overview.md).
- **The concrete code walkthrough** → [Risk & Scenarios](../guide/risk.md).
- **The trading-book regulatory outputs** → [Regulatory: FRTB & CCAR](regulatory.md).
- **The banking-book companion** → [Treasury & ALM](treasury.md).
- **The innovation-track companion** → [Quant Research](quant-research.md).
- **The validation strategy behind the QuantLib comparison suite** → [QuantLib Validation Pyramid](../architecture/quantlib-validation-pyramid.md).
- **The factor taxonomy — what is and is not modelled** → [Risk Factors](../risk-factors.md).
- **The systems around VALAX in a bank stack** → [Where VALAX Fits](../landscape.md).
