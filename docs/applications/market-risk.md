# Market Risk & Model Validation

*How VALAX serves the two middle-office functions that share the same tooling — **Market Risk Management (MRM)** on the trading book, and **Model Validation** as the SR 11-7 / TRIM independent challenger to the front office.*

These are the [audiences](index.md) VALAX is best positioned to serve *today*. Where [Regulatory: FRTB & CCAR](regulatory.md) covers the compliance outputs and [Treasury & ALM](treasury.md) covers the banking book, this page covers the **daily production loop of the middle office** — sensitivity ladders, limits, VaR/ES, backtests, P&L explain, and the challenger benchmarking Model Validation is required to run. Both functions need an *analytics platform they own*, distinct from the front-office pricers they oversee, and VALAX is architected for that role: pure functions, deterministic replay, autodiff Greeks, `vmap` over books, and a QuantLib comparison suite already in the tree.

---

## Consumers

These are the buyers and output-consumers — the people who fund the stack and read its numbers. The hands-on **users** are a different population (risk quants and developers); see [§ 2.1](#21-market-risk-management-mrm).

**To the Head of MRM:**
*A production-ready middle-office analytics stack: autodiff ladders, vmapped VaR/ES, waterfall P&L explain, Basel backtesting toolkit, and FRTB PLA — all in the same code path. Your morning-after P&L dialogue with the desk is one call, and the number reconciles by construction with the FRTB submission your CFO sends the regulator.*

**To the Head of Model Validation:**
*An independent-by-construction challenger implementation, in Python + JAX rather than C++, with a QuantLib comparison suite already in the tree covering options, fixed income, Heston, SABR, caps, swaptions, MC, PDE, lattice, SLV, and Greeks. Your annual validation cycle starts from a shipped evidence pack.*

**To the CRO:**
*The same pipeline computes your trader's morning Greeks, your desk head's DV01, your firm-wide 97.5% ES, and your FRTB PLA zone — from the same array, sliced differently. When any of those numbers changes, you can walk backward through the pipeline to find out why, with no vendor black box in the middle.*

**To the Chief Model Risk Officer:**
*SR 11-7 requires independence and defensible documentation. VALAX gives you both: a different technology stack, deterministic replay, exact Greeks, and a maintained regression suite against the industry's reference open-source library. Your submission packs stop being one-off spreadsheets.*

---

## 1. What actually differentiates VALAX from the bank's existing risk system

Every bank already runs a large end-of-day risk engine — Murex, a RiskMetrics-lineage stack, an in-house C++ system, or some combination. It already computes sensitivities, full-reval VaR, named stresses, and a Basel backtest overnight. So the honest question a CRO will ask is not *"can VALAX compute VaR"* — it can, and so can the incumbent — but *"what does a **differentiable** risk engine give me that my enumerate-and-bump system structurally cannot"*. This section is the summary of the main features; the rest of the page is the detail. Three answers, in increasing order of novelty — the first two are shipped, the third is the frontier.

### 1.1 Independence you can't fake — the shipped commercial differentiator

For Model Validation this is decisive: a challenger in Python + JAX + XLA is *genuine* codebase and toolchain independence from a C++ champion — not a wrapper over the same library, but a mathematically-equivalent implementation compiled by a different toolchain against different linear-algebra kernels. That is exactly the boundary SR 11-7 / TRIM demand, and no incumbent can sell you independence *from itself*. The shipped [`tests/test_quantlib_comparison/`](../architecture/quantlib-validation-pyramid.md) suite makes it a maintained evidence pack rather than a one-off. This is the differentiator a buyer signs for today.

### 1.2 Exact autodiff everywhere — a different cost model, not a feature

The incumbent computes Greeks by bump-and-reprice. That single implementation choice is the root of several permanent taxes autodiff does not pay:

- **The full first-order factor ladder is one reverse pass, not O(n_factors) repricings.** A book with hundreds of risk factors needs hundreds of bumped repricings per Greek order in a bump engine; `jax.grad(V)(market)` returns the sensitivity to *every* factor at once — categorically cheaper *and* exact at book scale.
- **A clean RTPL directly helps pass FRTB PLA.** The PLA test compares risk-theoretical P&L (from sensitivities) against hypothetical P&L (from full reval). Bump noise inflates the residual and is a real, common cause of PLA failure; exact autodiff sensitivities remove that noise source, so the RTPL–HPL gap becomes a genuine model signal rather than a numerical artefact.
- **No bump-width disputes in validation.** The most common champion-vs-challenger argument — "we agree on price but disagree on delta because of bump size" — cannot arise when the challenger delta is exact by construction.
- **One ladder reconciles every audience by construction.** Trader Greeks, desk DV01, CRO cross-gamma, FRTB SBA, PLA RTPL, and the backtest are all slices of the *same* `jax.grad`/`jax.hessian` output — a property of the graph, not a nightly reconciliation job.

None of this is about running the same computation faster. It is a structurally different — cheaper, exact — way of producing the risk numbers the regulator asks for.

### 1.3 Differentiable risk *measures* and reverse stress testing — the frontier

This is the categorical capability an enumerate-and-bump engine cannot express cleanly, because it requires differentiating *through* the risk measure and the scenario set. Because `hypothetical_pnl_vector`, `value_at_risk`, and `expected_shortfall` are one differentiable graph over positions and market, you can take gradients *of the risk number itself*:

```python
# ∂ES / ∂position — exact marginal / component risk, not a finite-difference approximation
def book_es(notionals, instruments, market, scenarios):
    pnl = hypothetical_pnl_vector(pricing_fn, scale(instruments, notionals), market, scenarios)
    return expected_shortfall(pnl, confidence=0.975)

marginal_es = jax.grad(book_es)(notionals, instruments, market, scenarios)   # per-position ES contribution
```

- **Exact Euler / component risk allocation.** Marginal and component VaR/ES fall out of one `jax.grad` on the risk measure — the exact Euler allocation, not the approximate contributions vendor engines report. "Which desk contributes how much of firm ES" becomes a gradient, not a re-run.
- **Optimal hedge to minimise a tail measure.** With ∂ES/∂hedge available, the risk-minimising hedge under a coherent tail measure is a gradient-based solve (`optax`/`optimistix`), not a grid search — the risk-management cousin of the [trading-desk decision loop](trading-desk.md#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer).
- **Gradient-based reverse stress testing.** Rather than enumerating named scenarios, *ascend the loss surface through the pricer* to find the worst market move subject to a plausibility constraint (e.g. Mahalanobis distance ≤ a threshold under the factor covariance). Reverse stress testing is regulatorily encouraged and genuinely hard for an enumerate-and-bump engine; with an end-to-end differentiable pricer it is a constrained optimisation the architecture supports natively.

**Honesty gate.** The *primitives* for §1.3 are shipped and exercised — differentiable pricers, `hypothetical_pnl_vector`, autodiff-friendly `value_at_risk` / `expected_shortfall`, `optimistix` / `optax`. The *packaged* products — a component-risk allocator, a reverse-stress optimiser — are not built; they are a natural extension on top of the library, not a feature you install today. They are here because they are the honest answer to "how is this more than a Python EOD risk engine": the potential is real and categorical, but it is roadmap, not shipped. See [Vision](../vision.md).

---

## 2. What MRM and Model Validation actually do

Two distinct functions, one shared toolkit.

### 2.1 Market Risk Management (MRM)

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

> **"MRM" names a function, not a person.** The Market Risk function is staffed by risk *quants* (VaR/ES and scenario methodology), risk *developers/engineers* (the risk system itself), and risk *managers/analysts* (limits, reporting, oversight). For a **library** like VALAX the hands-on **users** are the risk-quant and risk-developer sub-team; the risk managers and the CRO are **consumers** of its outputs and the buyers who fund it. Model Validation splits the same way — validation quants and developers adopt; the CMRO and the regulator consume.

### 2.2 Model Validation (SR 11-7 / TRIM)

Second-line function, often reporting to a Chief Model Risk Officer. Owns:

| Responsibility | Cadence |
|---|---|
| Independent implementation and benchmarking of every pricing model used for capital or limits | Annual + on any material change |
| Boundary-case, edge-case, and arbitrage testing of pricers | Annual |
| Sensitivity analysis: does the Greek change smoothly across parameter space? | Annual |
| Ongoing performance monitoring: does the model still track observed prices? | Quarterly |
| Documented evidence pack for regulator inspections | Continuous |

The single defining requirement — the reason this function exists — is that the challenger implementation **must not derive from the same codebase as the champion**. That is a hard architectural constraint, not a preference.

### 2.3 Why they share a toolkit

MRM's daily production loop and Model Validation's challenger pricing consume the *same* mathematical primitives: an independent pricer, exact Greeks, deterministic replay, scenario support. They differ in cadence (EOD vs annual) and in audience (CRO vs regulator), but the code path is the same. One VALAX instance serves both.

> **A word on cadence.** Unlike the [front-office trading](trading-desk.md) case, EOD is not a limitation to argue around: **regulatory VaR/ES, the 250-day backtest, and FRTB PLA are *defined* as end-of-day / periodic computations.** EOD is the correct, mandated cadence for this audience, and it is a genuine *fit-strength* — it comfortably tolerates JIT warm-up. On top of the EOD batch, VALAX also serves **on-demand intraday what-if** and **pre-trade limit checks** cheaply, because a `jax.grad` on the augmented portfolio reuses the same compiled graph. The differentiation from the incumbent, though, is never cadence and never "we compute VaR" — it is the differentiable-engine argument of [§ 1](#1-what-actually-differentiates-valax-from-the-banks-existing-risk-system).

---

## 3. The daily MRM loop, mapped to VALAX

```
┌──────────────────────────────────────────────────────────────────┐
│  Daily MRM production loop                                       │
│                                                                  │
│  1. Refresh MarketData ────────────────► valax/market/data.py    │
│     (curves, spots, vols, dividends)     valax/curves/           │
│                                                                  │
│  2. Reprice book at today's close ────► pricing_fn +             │
│     (HPL versus prior close)              portfolio.batch        │
│                                                                  │
│  3. Compute full sensitivity ladder ──► compute_ladder           │
│     (Δ, γ, vanna, volga, DV01,           greeks/autodiff         │
│      rate γ, cross-gamma)                                        │
│                                                                  │
│  4. Waterfall P&L explain ────────────► waterfall_pnl_report     │
│     (10 rungs vs. actual repricing)                              │
│                                                                  │
│  5. Bucketed limits view ─────────────► bucket_sensitivity_      │
│     (desk / sector / tenor)              ladder                  │
│                                                                  │
│  6. Full-reval VaR / ES ──────────────► hypothetical_pnl_        │
│     (10 000 historical or MC scenarios)   vector, value_at_risk, │
│                                          expected_shortfall      │
│                                                                  │
│  7. Named stress P&L ─────────────────► stress_scenario,         │
│     (crash, steepener, credit widen)     steepener, butterfly    │
│                                                                  │
│  8. Rolling 250-day backtest ─────────► var_breaches,            │
│                                          kupiec_pof,             │
│                                          christoffersen_*,       │
│                                          basel_traffic_light     │
│                                                                  │
│  9. FRTB PLA (for desks on IMA) ──────► pla_spearman, pla_ks,    │
│                                          pla_traffic_light       │
│                                                                  │
│  10. Limits monitor + breach queue ───► arithmetic on ladder     │
│      (send breaches to trader / CRO)     + bucket outputs        │
└──────────────────────────────────────────────────────────────────┘
```

Every stage has a one-line VALAX primitive today. This is not aspirational — the guide walkthroughs in [Risk & Scenarios](../guide/risk.md) exercise this loop end-to-end on synthetic books. The core object is the `SensitivityLadder` (one `jax.grad` + `jax.hessian` pass): the trader's Greeks, the desk's DV01, the CRO's cross-gamma, the FRTB SBA number, and the PLA RTPL vector are all slices of it — **the ladder is the noun; every dashboard is a verb.**

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

Under the hood, `jax.vmap` handles both axes — 10 000 scenarios × 5 000 instruments = 50 000 000 repricings compiled into one JIT call. On a single GPU this is minutes; on CPU it's competitive with the vendor risk engines it displaces. For a faster cross-check, `parametric_var(...)` gives the delta-normal answer from the ladder and covariance — and the *gap* between full-reval and parametric is itself a useful diagnostic on portfolio convexity.

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

The zone (green / yellow / red) drives the capital multiplier directly. For desks on IMA, this is joined by the FRTB PLA test (see [Regulatory: FRTB & CCAR § 2.3](regulatory.md#23-pl-attribution-pla-the-gate-to-ima)). Because these tests run inside the same code path that produces the VaR forecast, the backtest reconciles with the forecast by construction — where a split vendor stack pays a permanent reconciliation tax.

---

## 6. P&L explain — the trader-vs-MRM dialogue

The most politically charged conversation on any trading floor happens at 09:00 the morning after a big move: the trader claims the loss was "market", MRM claims it was "the position", the CRO wants a number. VALAX's `waterfall_pnl_report` answers with 10 rungs and a residual:

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

## 7. Coverage today vs. roadmap

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
| **Differentiable risk measures** (∂VaR/∂position, exact component/marginal ES, Euler allocation) | 📋 | **Frontier — §1.3.** Primitives shipped (`hypothetical_pnl_vector`, autodiff `value_at_risk`/`expected_shortfall`); allocator not packaged |
| **Gradient-based reverse stress testing** (worst plausible scenario by ascent through the pricer) | 📋 | **Frontier — §1.3.** End-to-end differentiable pricer + `optimistix`/`optax` support it; optimiser not packaged |
| Real-time streaming pricing service | 📋 | Vision-tier — service layer + market-data adapters |

The daily MRM production loop and the Model Validation challenger pack are **both fully achievable today** with the shipped components. Multi-curve and SBA aggregation are the two roadmap items that would close the last gaps for a bank-wide production deployment.

---

## 8. Where to read next

- **The engineering view of the pipeline these functions run on** → [Risk: End-to-End](../risk-overview.md).
- **The concrete code walkthrough** → [Risk & Scenarios](../guide/risk.md).
- **The trading-book regulatory outputs** → [Regulatory: FRTB & CCAR](regulatory.md).
- **The banking-book companion** → [Treasury & ALM](treasury.md).
- **The innovation-track companion** → [Quant Research](quant-research.md).
- **The validation strategy behind the QuantLib comparison suite** → [QuantLib Validation Pyramid](../architecture/quantlib-validation-pyramid.md).
- **The systems around VALAX in a bank stack** → [Where VALAX Fits](../landscape.md).
