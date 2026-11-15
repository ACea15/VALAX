# Market Risk & Model Validation

*How VALAX serves the two middle-office functions that share the same tooling — **Market Risk Management (MRM)** on the trading book, and **Model Validation** as the SR 11-7 / TRIM independent challenger to the front office.*

These are the [audiences](index.md) VALAX is best positioned to serve *today*. Where [Regulatory: FRTB & CCAR](regulatory.md) covers the compliance outputs and [Treasury & ALM](treasury.md) covers the banking book, this page covers the **daily production loop of the middle office** — bucketed sensitivities, limits, VaR/ES, backtests, P&L explain, and the challenger benchmarking Model Validation is required to run. Both functions need an *analytics platform they own*, distinct from the front-office pricers they oversee, and VALAX is architected for that role: pure functions, deterministic replay, autodiff Greeks, `vmap` over books, and a QuantLib comparison suite already in the tree.

---

## Consumers

These are the buyers and output-consumers — the people who fund the stack and read its numbers. The hands-on **users** are a different population (risk quants and developers); see [§ 2.1](#21-market-risk-management-mrm).

**To the Head of MRM:**
*A production-ready middle-office analytics stack: autodiff bucketed sensitivities, vmapped VaR/ES, waterfall P&L explain, Basel backtesting toolkit, and FRTB PLA — all in the same code path. Your morning-after P&L dialogue with the desk is one call, and the number reconciles by construction with the FRTB submission your CFO sends the regulator.*

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

The incumbent computes Greeks by bump-and-reprice; VALAX computes them by autodiff — not "the same numbers, faster" but a structurally cheaper *and* exact cost model. The generic capability and its `MRM` payoff — every factor sensitivity in one reverse pass, a bump-free RTPL that removes a common cause of FRTB PLA failure, no bump-width disputes in validation, and one set of sensitivities every audience slices — is set out in [Applications § 2.1](index.md#21-exact-autodiff-risk-no-bump-and-reprice) and [§ 2.3](index.md#23-one-market-state-object-one-code-path).

None of this is about running the same computation faster. It is a structurally different — cheaper, exact — way of producing the risk numbers the regulator asks for.

### 1.3 Differentiable risk *measures* and reverse stress testing — the frontier

This is the categorical capability an enumerate-and-bump engine cannot express cleanly, because it requires differentiating *through* the risk measure and the scenario set. Because `group_scenario_pnl` (full revaluation through `price()` under each keyed scenario), `value_at_risk`, and `expected_shortfall` are one differentiable graph over positions and market, you can take gradients *of the risk number itself*:

```python
# ∂ES / ∂position — exact marginal / component risk, not a finite-difference approximation
def book_es(notionals, instruments, model, market, scenarios):
    pnl = group_scenario_pnl(scale(instruments, notionals), model, market, scenarios)
    return expected_shortfall(pnl, confidence=0.975)

marginal_es = jax.grad(book_es)(notionals, instruments, model, market, scenarios)   # per-position ES contribution
```

- **Exact Euler / component risk allocation.** Marginal and component VaR/ES fall out of one `jax.grad` on the risk measure — the exact Euler allocation, not the approximate contributions vendor engines report. "Which desk contributes how much of firm ES" becomes a gradient, not a re-run.
- **Optimal hedge to minimise a tail measure.** With ∂ES/∂hedge available, the risk-minimising hedge under a coherent tail measure is a gradient-based solve (`optax`/`optimistix`), not a grid search — the risk-management cousin of the [trading-desk decision loop](trading-desk.md#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer).
- **Gradient-based reverse stress testing.** Rather than enumerating named scenarios, *ascend the loss surface through the pricer* to find the worst market move subject to a plausibility constraint (e.g. Mahalanobis distance ≤ a threshold under the factor covariance). Reverse stress testing is regulatorily encouraged and genuinely hard for an enumerate-and-bump engine; with an end-to-end differentiable pricer it is a constrained optimisation the architecture supports natively.

**Honesty gate.** The *primitives* for §1.3 are shipped and exercised — differentiable pricers, `group_scenario_pnl`, autodiff-friendly `value_at_risk` / `expected_shortfall`, `optimistix` / `optax`. One allocation is packaged: `keyed_component_var` splits a delta-normal VaR across risk-factor blocks (per curve, basis, surface, asset) by Euler allocation. The other *packaged* products — a per-position full-revaluation ES allocator, a reverse-stress optimiser — are not built; they are a natural extension on top of the library, not a feature you install today. They are here because they are the honest answer to "how is this more than a Python EOD risk engine": the potential is real and categorical, but it is roadmap, not shipped. See [Vision](../vision.md).

---

## 2. What MRM and Model Validation actually do

Two distinct functions, one shared toolkit.

### 2.1 Market Risk Management (MRM)

Middle-office function reporting to the CRO. Owns:

| Responsibility | Cadence |
|---|---|
| End-of-day sensitivity computation across the trading book (DV01, delta, vega, gamma, bucketed per factor) | EOD |
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
│  1. Build today's MarketState ────────► calibrate_market,        │
│     (curves, spots, vols, dividends)     bootstrap_curve_graph   │
│                                                                  │
│  2. Reprice book at today's close ────► price() via group_pv     │
│     (HPL versus prior close)                                     │
│                                                                  │
│  3. Compute bucketed sensitivities ───► keyed_sensitivities      │
│     (Δ, γ, vanna, volga, DV01 per        (market_risk per trade) │
│      curve and basis, cross-gamma)                               │
│                                                                  │
│  4. Waterfall P&L explain ────────────► keyed_waterfall_report   │
│     (one term per curve / surface /                              │
│      asset vs. actual repricing)                                 │
│                                                                  │
│  5. Bucketed limits view ─────────────► KeyedSensitivities.      │
│     (desk / sector / tenor)              key_rate_dv01,          │
│                                          block_totals, BucketMap │
│                                                                  │
│  6. Full-reval VaR / ES ──────────────► keyed_historical_        │
│     (10 000 historical or MC scenarios)   scenarios,             │
│                                          group_scenario_pnl,     │
│                                          value_at_risk,          │
│                                          expected_shortfall      │
│                                                                  │
│  7. Named stress P&L ─────────────────► keyed_stress_scenario,   │
│     (crash, steepener, basis widen)      keyed_curve_stress,     │
│                                          keyed_basis_shift       │
│                                                                  │
│  8. Rolling 250-day backtest ─────────► var_breaches,            │
│                                          kupiec_pof,             │
│                                          christoffersen_*,       │
│                                          basel_traffic_light     │
│                                                                  │
│  9. FRTB PLA (for desks on IMA) ──────► pla_spearman, pla_ks,    │
│                                          pla_traffic_light       │
│                                          (HPL/RTPL vectors still │
│                                           on the flat path)      │
│                                                                  │
│  10. Limits monitor + breach queue ───► arithmetic on the        │
│                                          sensitivities           │
│      (send breaches to trader / CRO)     + bucket outputs        │
└──────────────────────────────────────────────────────────────────┘
```

Every stage has a one-line VALAX primitive today. This is not aspirational — the guide walkthroughs in [Scenarios & Shocks](../risk-scenarios.md) and `examples/19_unified_book_workflow.py` exercise this loop end-to-end on synthetic books. The core object is the `KeyedSensitivities` (one `jax.grad` + `jax.hessian` pass through `price()`, one block per named curve, surface and asset): the trader's Greeks, the desk's DV01 per curve, the basis DV01, the CRO's cross-gamma and the per-curve VaR contributions are all slices of it — **the sensitivities are the noun; every dashboard is a verb.** One exception remains: the FRTB PLA vectors and SBA bucketing (`bucket_sensitivities`) still read the legacy flat `FlatSensitivities` until their keyed port lands ([roadmap](../roadmap.md#remaining-work)).

---

## 4. Full-revaluation VaR and ES

MRM's regulatory VaR is full-revaluation on 250 days of historical scenarios (Basel) or on parametric Monte Carlo (internal). Both are one call:

```python
from valax.risk import (
    KeyedFactorLayout, keyed_historical_scenarios, group_scenario_pnl,
    value_at_risk, expected_shortfall,
)

# One block per named curve, surface and asset of today's market
layout = KeyedFactorLayout.from_market_state(market)

# 250-day historical scenarios (columns ordered as layout.factor_ids())
scenarios = keyed_historical_scenarios(returns_matrix, layout)

# Full revaluation of the book through price() under each scenario
pnl_vec = group_scenario_pnl(instruments, model, market, scenarios)
# pnl_vec.shape == (n_scenarios,)

var_99 = value_at_risk(pnl_vec, confidence=0.99)
es_975 = expected_shortfall(pnl_vec, confidence=0.975)
```

Under the hood, `jax.vmap` handles both axes — 10 000 scenarios × 5 000 instruments = 50 000 000 repricings compiled into one JIT call. On a single GPU this is minutes; on CPU it's competitive with the vendor risk engines it displaces. For a faster cross-check, `keyed_parametric_var(sens, cov)` gives the delta-normal answer from the keyed sensitivities and a per-curve `block_covariance` — and `keyed_component_var` splits it by curve, basis and surface — and the *gap* between full-reval and parametric is itself a useful diagnostic on portfolio convexity.

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

The most politically charged conversation on any trading floor happens at 09:00 the morning after a big move: the trader claims the loss was "market", MRM claims it was "the position", the CRO wants a number. VALAX's `keyed_waterfall_report` answers with one term per named market object and a residual:

```python
from valax.market import KeyedScenario
from valax.risk import keyed_sensitivities, keyed_waterfall_report

# Yesterday's actual moves, addressed by identifier
overnight = KeyedScenario(
    spot_shocks={"SPX": spx_today - spx_yesterday},
    vol_shocks={"SPX.VOL": atm_vol_today - atm_vol_yesterday},
    curve_shocks={
        "USD.OIS": ois_zeros_today - ois_zeros_yesterday,
        "USD.SOFR.3M": sofr_zeros_today - sofr_zeros_yesterday,
    },
)

sens = keyed_sensitivities(positions, model, yesterday_market, second_order=True)
wf = keyed_waterfall_report(positions, model, yesterday_market, overnight, sensitivities=sens)

# The answer the CRO wants: first-order P&L per curve / surface / asset,
# then the second-order terms, the actual repricing and the residual
for block, pnl in wf.delta_pnl.items():
    print(f"{block}: {pnl:+.0f}")
print(f"second order:{wf.total_second_order:+.0f} "
      f"actual:{wf.actual:+.0f} unexplained:{wf.unexplained:+.0f}")
```

Every term has a name a trader recognises; the `unexplained` residual is the honest measure of "how well the risk engine understands this book". A systematically large or biased unexplained is the single most actionable model-improvement signal MRM produces.

---

## 7. Coverage today vs. roadmap

The full, cross-audience coverage matrix now lives in one place — **[Applications → Coverage](index.md#4-coverage-what-ships-today-vs-roadmap)** (scan the *Business* column for `MRM`).

In brief: the daily MRM production loop and the Model Validation challenger pack are **both fully achievable today** with shipped components. The multi-curve market has shipped; **porting the PLA vectors onto the keyed sensitivities** and **SBA aggregation** are the roadmap items that would close the last gaps for a bank-wide deployment, and the differentiable **risk-measure / reverse-stress** capabilities of [§ 1.3](#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier) are the categorical frontier.

---

## 8. Where to read next

- **The engineering view of the pipeline these functions run on** → [Risk: End-to-End](../risk-overview.md).
- **The concrete code walkthrough** → [Scenarios & Shocks](../risk-scenarios.md).
- **The trading-book regulatory outputs** → [Regulatory: FRTB & CCAR](regulatory.md).
- **The banking-book companion** → [Treasury & ALM](treasury.md).
- **The innovation-track companion** → [Quant Research](quant-research.md).
- **The validation strategy behind the QuantLib comparison suite** → [QuantLib Validation Pyramid](../architecture/quantlib-validation-pyramid.md).
- **The systems around VALAX in a bank stack** → [Positioning & Design Rationale](../positioning/index.md).
