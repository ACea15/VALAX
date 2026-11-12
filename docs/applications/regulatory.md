# Regulatory Applications: FRTB & CCAR

[← Applications overview](index.md#3-the-audience-map-ranked-by-fit-today) · one of five audience pages

*How VALAX supports bank market-risk capital programmes — the Basel III.1 **Fundamental Review of the Trading Book (FRTB)** migration and the Federal Reserve's annual **Comprehensive Capital Analysis and Review (CCAR / DFAST)** stress tests.*

This page is the **outward-facing** view of the risk stack described in [Risk: End-to-End](../risk-overview.md), the Risk how-to pages (starting with [Scenarios & Shocks](../risk-scenarios.md)), and [Models & Theory § 7](../theory/risk-measures.md#7-risk-measures). Read those first for the mathematical foundations and the code walkthrough; read *this* to see how they map onto a real bank regulatory programme.

---

## 1. Why these two workflows

FRTB and CCAR are, at their core, the *same computational problem* dressed in different regulatory clothing:

> **Reprice the trading book — many times — under prescribed market conditions, aggregate the P&L into capital numbers, and defend every number in an audit trail.**

VALAX is built around exactly that primitive. Every price comes from the pure function `price(instrument, model, market: MarketState) -> PricingResult`, and every downstream number — Greeks, sensitivity ladders, VaR/ES, backtests, the FRTB PLA test, bucketed capital views — is a *derived* quantity on top of it. The [Applications overview § 1](index.md#1-the-sixty-second-story-one-pipeline-many-audiences) sets out that "one pipeline, many audiences" decomposition once, and [Risk: End-to-End § 3](../risk-overview.md#3-the-end-to-end-pipeline) shows the pipeline end-to-end; here that single pipeline serves **two regulatory audiences**.

---

## 2. FRTB migration

FRTB is where the mapping is tightest — the library was designed with it in mind, and the risk overview is explicit that Basel III.1 / FRTB defines two parallel methodologies that **both start from the same underlying pipeline** ([Risk: End-to-End § 4.5](../risk-overview.md#45-how-much-capital-does-the-bank-need)).

### 2.1 SBA and IMA share one pipeline

| FRTB approach | What it consumes | VALAX components |
|---|---|---|
| **Standardised Approach (SBA)** | Bucketed sensitivities at FRTB tenor / sector / rating vertices, aggregated via prescribed correlations | Delta: `keyed_ladder` → `curve_ladder` per curve → `tenor_bucket_map` / `aggregate`. Curvature and cross-bucket terms: `compute_ladder` → `bucket_sensitivity_ladder` (legacy flat path, keyed port pending) → (📋 planned) SBA aggregation formula |
| **Internal Models Approach (IMA)** | Expected Shortfall on a stressed period + stress add-on for non-modellable risk factors, conditional on passing PLA + VaR backtest | `risk_theoretical_pnl_vector` / `hypothetical_pnl_vector` (legacy flat path, keyed port pending) → `expected_shortfall` → PLA check → (📋 planned) capital scaling |

Operationally, a migrating desk can prototype *both* approaches against the same instruments and the same `MarketState` without maintaining two pricing stacks.

### 2.2 Bucketed sensitivities at FRTB vertices (SBA)

FRTB SBA requires sensitivities delivered at **regulatory** vertices (10 IR tenors `{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, 11 equity sector buckets, 18 credit buckets, …) rather than the bank's internal pillar grid. VALAX exposes this as a *coordinate change* on top of autodiff sensitivities. For delta, the keyed ladder is the current route: `keyed_ladder(instruments, model, market)` gives each named curve its own pillar ladder (`curve_ladder(curve_id)`), which `aggregate` pushes onto the FRTB grid. The full second-order bucketing below, including the cross-gamma blocks SBA curvature needs, still runs on the legacy flat `SensitivityLadder` built from a `MarketData` until its keyed port lands ([roadmap](../roadmap.md#remaining-work)):

- [`compute_ladder(pricing_fn, instruments, base_market)`](../risk-ladders.md#computing-a-ladder) produces a full first- and second-order `SensitivityLadder` — delta_spot, vega, DV01 per pillar, gamma_spot, gamma_rate, vanna, volga, cross_spot_rate, cross_vol_rate — via a single `jax.grad` + `jax.hessian` pass, with no bump-and-reprice.
- [`tenor_bucket_map(pillar_times, frtb_vertices, weight="indicator" | "linear")`](../risk-bucketing.md#standard-bucket-builders) builds the aggregation matrix that pushes internal pillars onto the FRTB tenor grid.
- `equal_weight_bucket_map` does the same for equity sector and credit rating buckets.
- [`bucket_sensitivity_ladder`](../risk-bucketing.md#bucketing-a-full-sensitivity-ladder) applies bucket maps to every component of the ladder *including bilateral aggregation of the cross-gamma blocks* — which is exactly what SBA curvature and cross-bucket correlation terms need.

Because bucketing is "a matrix multiply, not a separate computation" ([Risk: End-to-End § 3](../risk-overview.md#3-the-end-to-end-pipeline)), the internal risk view and the regulatory FRTB view are literally the same ladder × a different `A`. That property is what makes an FRTB migration tractable without doubling code.

### 2.3 P&L Attribution (PLA) — the gate to IMA

A desk that fails PLA loses its right to use IMA and gets pushed to the (usually more punitive) SBA. This is *the* single largest financial incentive in the framework. VALAX ships the full BCBS d558 machinery. The HPL / RTPL vectors below still run on the legacy flat path — a `pricing_fn(instrument, MarketData)` and a flat `SensitivityLadder` — so they see one discount curve, not the per-curve split; their port onto the keyed ladder is [open](../roadmap.md#remaining-work). The tests themselves (`pla_spearman`, `pla_ks`, `pla_traffic_light`) work on plain arrays and are not affected:

```python
from valax.risk import (
    compute_ladder, risk_theoretical_pnl_vector, hypothetical_pnl_vector,
    pla_spearman, pla_ks, pla_traffic_light,
)

ladder = compute_ladder(pricing_fn, instruments, base_market)
rtpl   = risk_theoretical_pnl_vector(ladder, scenarios, base_market)              # ladder-based
hpl    = hypothetical_pnl_vector(pricing_fn, instruments, base_market, scenarios) # full reval

rho  = pla_spearman(rtpl, hpl)                              # monotonic agreement
D    = pla_ks(rtpl, hpl)                                     # distributional agreement
zone = pla_traffic_light(rho, D, n_obs=rtpl.shape[0])        # green / amber / red
```

The traffic light applies the d558 thresholds directly: green if Spearman ≥ 0.80 *and* KS $p$-value ≥ 0.264; red if either Spearman < 0.70 or KS $p$-value < 0.055; amber otherwise (see [FRTB P&L Attribution Test](../risk-backtesting.md#frtb-pl-attribution-test)).

If a desk lands amber or red, [`explained_unexplained_vector`](../risk-var.md#comparing-both-at-once) gives the per-scenario residual — a direct diagnostic pointing at missing risk factors or third-order convexity the ladder didn't capture.

!!! tip "Practical FRTB workflow"
    Compute one ladder, sweep the 250-day historical window, read the zone. If it fails, look at the unexplained vector to prioritise *which* factors to add before re-submission.

### 2.4 VaR backtesting — the other IMA gate

FRTB retains a 99% one-day VaR backtest on a rolling 250-day window with capital multipliers driven by the breach count. VALAX ships the standard Basel toolkit:

```python
from valax.risk import (
    var_breaches, kupiec_pof, christoffersen_conditional_coverage, basel_traffic_light,
)

breaches = var_breaches(actual_pnl, var_forecast)
kupiec_pof(breaches, confidence=0.99)                    # LR_uc, p-value
christoffersen_conditional_coverage(breaches, 0.99)      # joint rate + independence
basel_traffic_light(n_breaches, n_obs=250, confidence=0.99)
```

For a 250-day window at 99% VaR the mapping is standard: 0–4 breaches green, 5–9 yellow, ≥10 red, with the capital multiplier scaling accordingly ([VaR Backtesting](../risk-backtesting.md#var-backtesting)).

### 2.5 Expected Shortfall for IMA capital

FRTB IMA replaces VaR with 97.5% ES on a stressed calibration period. On the VALAX pipeline this is a one-liner over the same P&L vector:

```python
es_rtpl = expected_shortfall(rtpl, confidence=0.975)
es_hpl  = expected_shortfall(hpl,  confidence=0.975)
```

The gap `es_rtpl − es_hpl` is the **model-induced ES bias** — how much the Taylor approximation under- or over-states the tail. For well-hedged portfolios it should be near zero; when it isn't, that gap is a concrete signal on where to invest in richer risk factors before submission.

### 2.6 Multi-curve and basis risk

FRTB IR bucketing distinguishes OIS, SOFR/ESTR, basis, and cross-currency basis. VALAX today ships:

- Fully differentiable curve shocks — `jax.grad` through a bumped curve *is* the key-rate DV01 by construction. Critically for PLA: no bump noise contaminating the RTPL vs HPL comparison.
- Pre-built structured shocks: parallel, steepener, flattener, butterfly, key-rate.
- A multi-curve market: OIS discounting, projection curves per tenor and cross-currency curves live in one `CurveGraph` inside the `MarketState`, built jointly by `bootstrap_curve_graph`. Each named curve is its own risk factor, and the basis can be its own coordinate (`as_basis_layout`), so `keyed_ladder` reports a DV01 per curve and per basis (`KeyedLadder.basis_dv01`) and `keyed_component_var` gives each its VaR share ([Why Multi-Curve?](../guide/why-multicurve.md)).

What is still open: the FRTB PLA vector pair (`hypothetical_pnl_vector`, `risk_theoretical_pnl_vector`) still runs on the legacy flat `MarketData` path, so PLA does not yet see the per-curve split, and the vol-surface risk factors and SBA aggregation matrices remain to be built (see [Risk: End-to-End § 7](../risk-overview.md#7-the-state-of-the-risk-section-and-whats-next) and the [Roadmap](../roadmap.md#remaining-work)).

---

## 3. CCAR / DFAST annual stress tests

CCAR is fundamentally a **scenario stress exercise**: reprice everything under Fed-supplied Baseline / Adverse / Severely Adverse macro paths, then report P&L and capital impacts. This is exactly the shape of the VALAX scenario / VaR pipeline, just with a small number of very deep scenarios instead of 10 000 shallow ones.

### 3.1 Scenario definition matches the CCAR format

A [`KeyedScenario`](../risk-scenarios.md#keyed-scenarios-on-a-marketstate) represents **additive changes** to named risk factors — spot shocks (or multiplicative returns) by asset, parallel vol shifts by surface, per-pillar rate shocks by curve, dividend shocks — bundled as one pytree. This is the natural container for encoding a Fed shock table:

```python
from valax.market import stack_keyed_scenarios
from valax.risk import KeyedFactorLayout, keyed_stress_scenario, keyed_steepener

layout = KeyedFactorLayout.from_market_state(market)

baseline         = keyed_stress_scenario(layout, spot_shock=+0.05, vol_shock=+0.02, ...)
adverse          = keyed_stress_scenario(layout, spot_shock=-0.15, vol_shock=+0.08, ...)
severely_adverse = keyed_stress_scenario(layout, spot_shock=-0.35, vol_shock=+0.20, ...)
ccar_twist       = keyed_steepener(layout, "USD.OIS", short_bump=+0.02, long_bump=-0.01)

ccar_set = stack_keyed_scenarios([baseline, adverse, severely_adverse, ccar_twist])
```

Every named CCAR trajectory on curves, surfaces, spots and dividends (equity crash, curve twist, basis widening, vol spike, dividend shock) has a direct keyed primitive in `valax/risk/scenarios.py`. Credit-spread widening is the exception: the survival-curve shocks exist in `valax/risk/shocks.py` but are not yet addressable from a `KeyedScenario` ([roadmap](../roadmap.md#remaining-work)).

### 3.2 One-shot revaluation of the entire book

```python
from valax.risk import group_scenario_pnl

ccar_pnl = group_scenario_pnl(instruments, model, market, ccar_set)
```

Under the hood `group_scenario_pnl` `jax.vmap`s over the scenario axis, applies each scenario with `shock`, and reprices the whole group through `price()` — so **10 000 scenarios × 100 instruments = 1 000 000 repricings compile down to a single JIT-compiled call**, no Python loops ([Repricing through `price()`](../risk-var.md#repricing-through-price)).

CCAR itself only has a handful of macro scenarios, but the same machinery lets a bank overlay Monte Carlo sensitivity variations around each Fed scenario (e.g. path perturbations to probe non-linear regions) without changing any code — the bottleneck stays on the GPU rather than in Python.

### 3.3 Explaining the stress P&L to the CRO and the Fed

CCAR submissions require rigorous **P&L attribution** to defend the number. [`keyed_waterfall_report`](../risk-ladders.md#keyed-ladders-on-a-marketstate) decomposes each scenario into one first-order rung per named curve, surface and asset, each block's own second-order term, the cross terms, and `actual`, `predicted` and `unexplained`:

```python
ladder = keyed_ladder(instruments, model, market, layout, second_order=True)
wf = keyed_waterfall_report(instruments, model, market, severely_adverse, ladder=ladder)
# wf.delta_pnl   {"IR.CURVE.USD.OIS": ..., "EQ.SPOT.SPX": ..., ...}  first order per block
# wf.gamma_pnl   per-block second order;  wf.cross_pnl  every cross term
# wf.actual, wf.predicted, wf.unexplained
```

For the large moves in CCAR Severely Adverse the second-order terms (rate gamma, vanna, volga, cross-gamma) significantly reduce the unexplained residual — which is exactly what internal validators, model-risk teams, and regulators ask about first.

### 3.4 Bucketed capital views (limits, concentration, ALM)

CCAR reporting needs sensitivities and stress P&L expressed in the bank's internal capital-planning taxonomy — desk, sector, currency, tenor bucket. The same `BucketMap` machinery that produces FRTB tenor vertices produces "short / belly / wings" for a trader, "tech / energy / financials" for equities, or PCA level/slope/curvature for a rates ALM view ([Risk Bucketing](../risk-bucketing.md#risk-bucketing) and [PCA Curve Shocks](../guide/pca-rates.md)). One ladder, many audiences.

### 3.5 Full-revaluation and parametric VaR side by side

For CCAR narrative documentation it is standard to compare a full-revaluation VaR/ES on the stressed calibration set with a fast parametric (delta-normal) result — `keyed_parametric_var` on the keyed ladder, split per curve by `keyed_component_var` (the flat [`parametric_var`](../risk-var.md#parametric-var-delta-normal) is the legacy equivalent) — and to defend any gap. When they diverge, the gap is a diagnostic on convexity that the CCAR narrative typically needs to address.

### 3.6 Reproducibility for the audit trail

CCAR results are heavily audited. Every VALAX price comes from `price(instrument, model, market)`, a **pure function** with no mutable state — this is called out explicitly for the auditor use case: *same inputs ⇒ same outputs*, so an auditor can replay yesterday's books to the bit ([Risk: End-to-End § 2](../risk-overview.md#2-who-consumes-risk-numbers-and-what-they-actually-want) and [§ 6](../risk-overview.md#6-why-valax-is-built-the-way-it-is)). Deterministic autodiff Greeks and JIT-compiled scenario runs give the same answer on the same input on any hardware.

---

## 4. Coverage today vs. roadmap

The full, cross-audience coverage matrix now lives in one place — **[Applications → Coverage](index.md#4-coverage-what-ships-today-vs-roadmap)** (scan the *Business* column for `REG`).

The multi-curve market, once first on the critical path, has shipped. The shortest remaining path to a desk-usable, bank-style FRTB/CCAR system is **(a) port the PLA vector pair onto the keyed ladder → (b) node-level vol-surface risk factors → (c) FRTB SBA capital aggregation**. Full priority order lives in [Risk Factors § 4](../risk-factors.md#4-roadmap) and the project [Roadmap](../roadmap.md).

---

## 5. Why the JAX foundation matters for these programmes specifically

FRTB and CCAR are *volume* problems as much as they are correctness problems — thousands of instruments × hundreds of risk factors × thousands of scenarios × 250-day windows. The library's design decisions land directly on that: autodiff instead of bump-and-reprice, `vmap` over scenarios and instruments, one `MarketState` pytree, bucketing as a matrix multiply, and deterministic pure functions — each with its `REG` payoff enumerated in [Applications § 2](index.md#2-what-makes-valax-distinctive-capability-by-capability) (deeper *why* in [Positioning § 4](../positioning/index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys) and [Risk: End-to-End § 6](../risk-overview.md#6-why-valax-is-built-the-way-it-is)).

The net effect: the derivatives pricing engine and the regulatory reporting engine are the *same engine*, differenced or bucketed differently. One library, one mental model, one code path from a trader's delta at 14:31 to the firm-wide 97.5% ES on 250 stressed-period scenarios overnight — and to the CCAR Severely Adverse waterfall the following morning.

---

## 6. Where to read next

- **The pipeline these workflows sit on** → [Risk: End-to-End](../risk-overview.md).
- **The mathematics of VaR / ES / backtesting / PLA / bucketing** → [Models & Theory § 7](../theory/risk-measures.md#7-risk-measures).
- **The concrete code walkthrough** → [Scenarios & Shocks](../risk-scenarios.md).
- **The factor taxonomy — what is and is not modelled** → [Risk Factors](../risk-factors.md).
- **The horizontal view of the systems around VALAX in a bank stack** → [Positioning & Design Rationale](../positioning/index.md).
- **The forward direction, including vol-surface risk factors and SBA aggregation** → [Vision](../vision.md) and [Roadmap](../roadmap.md).
