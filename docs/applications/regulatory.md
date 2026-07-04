# Regulatory Applications: FRTB & CCAR

*How VALAX supports bank market-risk capital programmes — the Basel III.1 **Fundamental Review of the Trading Book (FRTB)** migration and the Federal Reserve's annual **Comprehensive Capital Analysis and Review (CCAR / DFAST)** stress tests.*

This page is the **outward-facing** view of the risk stack described in [Risk: End-to-End](../risk-overview.md), the [Risk & Scenarios guide](../guide/risk.md), and [Models & Theory § 7](../theory.md#7-risk-measures). Read those first for the mathematical foundations and the code walkthrough; read *this* to see how they map onto a real bank regulatory programme.

---

## 1. Why these two workflows

FRTB and CCAR are, at their core, the *same computational problem* dressed in different regulatory clothing:

> **Reprice the trading book — many times — under prescribed market conditions, aggregate the P&L into capital numbers, and defend every number in an audit trail.**

VALAX is built around exactly that primitive. Every pricing function has the pure signature `V(instrument, MarketData) -> price`, and every downstream number — Greeks, sensitivity ladders, VaR/ES, backtests, the FRTB PLA test, bucketed capital views — is a *derived* quantity on top of that primitive (see [Risk: End-to-End § 3](../risk-overview.md#3-the-end-to-end-pipeline)). One pipeline, two regulatory audiences.

---

## 2. FRTB migration

FRTB is where the mapping is tightest — the library was designed with it in mind, and the risk overview is explicit that Basel III.1 / FRTB defines two parallel methodologies that **both start from the same underlying pipeline** ([Risk: End-to-End § 4.5](../risk-overview.md#45-how-much-capital-does-the-bank-need)).

### 2.1 SBA and IMA share one pipeline

| FRTB approach | What it consumes | VALAX components |
|---|---|---|
| **Standardised Approach (SBA)** | Bucketed sensitivities at FRTB tenor / sector / rating vertices, aggregated via prescribed correlations | `compute_ladder` → `tenor_bucket_map` / `equal_weight_bucket_map` → `bucket_sensitivity_ladder` → (📋 planned) SBA aggregation formula |
| **Internal Models Approach (IMA)** | Expected Shortfall on a stressed period + stress add-on for non-modellable risk factors, conditional on passing PLA + VaR backtest | `risk_theoretical_pnl_vector` / `hypothetical_pnl_vector` → `expected_shortfall` → PLA check → (📋 planned) capital scaling |

Operationally, a migrating desk can prototype *both* approaches against the same instruments and the same `MarketData` container without maintaining two pricing stacks.

### 2.2 Bucketed sensitivities at FRTB vertices (SBA)

FRTB SBA requires sensitivities delivered at **regulatory** vertices (10 IR tenors `{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, 11 equity sector buckets, 18 credit buckets, …) rather than the bank's internal pillar grid. VALAX exposes this as a *coordinate change* on top of autodiff sensitivities:

- [`compute_ladder(pricing_fn, instruments, base_market)`](../guide/risk.md#computing-a-ladder) produces a full first- and second-order `SensitivityLadder` — delta_spot, vega, DV01 per pillar, gamma_spot, gamma_rate, vanna, volga, cross_spot_rate, cross_vol_rate — via a single `jax.grad` + `jax.hessian` pass, with no bump-and-reprice.
- [`tenor_bucket_map(pillar_times, frtb_vertices, weight="indicator" | "linear")`](../guide/risk.md#standard-bucket-builders) builds the aggregation matrix that pushes internal pillars onto the FRTB tenor grid.
- `equal_weight_bucket_map` does the same for equity sector and credit rating buckets.
- [`bucket_sensitivity_ladder`](../guide/risk.md#bucketing-a-full-sensitivity-ladder) applies bucket maps to every component of the ladder *including bilateral aggregation of the cross-gamma blocks* — which is exactly what SBA curvature and cross-bucket correlation terms need.

Because bucketing is "a matrix multiply, not a separate computation" ([Risk: End-to-End § 3](../risk-overview.md#3-the-end-to-end-pipeline)), the internal risk view and the regulatory FRTB view are literally the same ladder × a different `A`. That property is what makes an FRTB migration tractable without doubling code.

### 2.3 P&L Attribution (PLA) — the gate to IMA

A desk that fails PLA loses its right to use IMA and gets pushed to the (usually more punitive) SBA. This is *the* single largest financial incentive in the framework. VALAX ships the full BCBS d558 machinery:

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

The traffic light applies the d558 thresholds directly: green if Spearman ≥ 0.80 *and* KS $p$-value ≥ 0.264; red if either Spearman < 0.70 or KS $p$-value < 0.055; amber otherwise (see [FRTB P&L Attribution Test](../guide/risk.md#frtb-pl-attribution-test)).

If a desk lands amber or red, [`explained_unexplained_vector`](../guide/risk.md#comparing-both-at-once) gives the per-scenario residual — a direct diagnostic pointing at missing risk factors or third-order convexity the ladder didn't capture.

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

For a 250-day window at 99% VaR the mapping is standard: 0–4 breaches green, 5–9 yellow, ≥10 red, with the capital multiplier scaling accordingly ([VaR Backtesting](../guide/risk.md#var-backtesting)).

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
- A single `DiscountCurve` today with a documented path to full multi-curve `MarketData` on the [Roadmap](../roadmap.md) — this is called out as the top-priority gap for a "single-desk production deployment", along with vol-surface risk factors and the SBA aggregation matrices (see [Risk: End-to-End § 7](../risk-overview.md#7-the-state-of-the-risk-section-and-whats-next) and [Why Multi-Curve?](../guide/why-multicurve.md)).

Migration implication: you can prototype end-to-end today on a single-curve book, and the same code paths extend when multi-curve lands — no rewrite of the risk engine.

---

## 3. CCAR / DFAST annual stress tests

CCAR is fundamentally a **scenario stress exercise**: reprice everything under Fed-supplied Baseline / Adverse / Severely Adverse macro paths, then report P&L and capital impacts. This is exactly the shape of the VALAX scenario / VaR pipeline, just with a small number of very deep scenarios instead of 10 000 shallow ones.

### 3.1 Scenario definition matches the CCAR format

[`MarketScenario`](../guide/risk.md#scenarios) represents **additive changes** to risk factors — spot shocks (or multiplicative returns), vol shocks, per-pillar rate shocks, dividend shocks — bundled as one pytree. This is the natural container for encoding a Fed shock table:

```python
from valax.risk import stress_scenario, steepener
from valax.market import stack_scenarios

baseline         = stress_scenario(n_assets, n_pillars, spot_shock=+0.05, vol_shock=+0.02, ...)
adverse          = stress_scenario(n_assets, n_pillars, spot_shock=-0.15, vol_shock=+0.08, ...)
severely_adverse = stress_scenario(n_assets, n_pillars, spot_shock=-0.35, vol_shock=+0.20, ...)
ccar_twist       = steepener(n_assets, n_pillars, short_bump=+0.02, long_bump=-0.01)

ccar_set = stack_scenarios([baseline, adverse, severely_adverse, ccar_twist])
```

Every named CCAR trajectory (equity crash, credit spread widening, curve twist, vol spike, dividend shock) has a direct primitive in `valax/risk/scenarios.py` and `valax/risk/shocks.py`.

### 3.2 One-shot revaluation of the entire book

```python
from valax.risk import hypothetical_pnl_vector

ccar_pnl = hypothetical_pnl_vector(pricing_fn, instruments, base_market, ccar_set)
```

Under the hood `portfolio_pnl` `jax.vmap`s over the scenario axis, and inside each scenario `jax.vmap`s over instruments — so **10 000 scenarios × 100 instruments = 1 000 000 repricings compile down to a single JIT-compiled call**, no Python loops ([Full-Revaluation VaR](../guide/risk.md#full-revaluation-var)).

CCAR itself only has a handful of macro scenarios, but the same machinery lets a bank overlay Monte Carlo sensitivity variations around each Fed scenario (e.g. path perturbations to probe non-linear regions) without changing any code — the bottleneck stays on the GPU rather than in Python.

### 3.3 Explaining the stress P&L to the CRO and the Fed

CCAR submissions require rigorous **P&L attribution** to defend the number. [`waterfall_pnl_report`](../guide/risk.md#waterfall-pl-decomposition) decomposes each scenario into 10 rungs — delta_spot / vega / DV01 / delta_div, spot γ, rate γ, vanna, volga, cross spot×rate, cross vol×rate — plus `actual`, `predicted`, and `unexplained`:

```python
ladder = compute_ladder(pricing_fn, instruments, base_market)
wf = waterfall_pnl_report(pricing_fn, instruments, base_market, severely_adverse, ladder=ladder)
# wf.delta_spot, wf.delta_vol, wf.delta_rate, wf.gamma_spot, wf.vanna_pnl, wf.volga_pnl,
# wf.cross_spot_rate_pnl, ..., wf.actual, wf.predicted, wf.unexplained
```

For the large moves in CCAR Severely Adverse the second-order rungs (rate gamma, vanna, volga, cross-gamma) significantly reduce the unexplained residual — which is exactly what internal validators, model-risk teams, and regulators ask about first.

### 3.4 Bucketed capital views (limits, concentration, ALM)

CCAR reporting needs sensitivities and stress P&L expressed in the bank's internal capital-planning taxonomy — desk, sector, currency, tenor bucket. The same `BucketMap` machinery that produces FRTB tenor vertices produces "short / belly / wings" for a trader, "tech / energy / financials" for equities, or PCA level/slope/curvature for a rates ALM view ([Risk Bucketing](../guide/risk.md#risk-bucketing) and [PCA Curve Shocks](../guide/pca-rates.md)). One ladder, many audiences.

### 3.5 Full-revaluation and parametric VaR side by side

For CCAR narrative documentation it is standard to compare a full-revaluation VaR/ES on the stressed calibration set with a fast parametric (delta-normal) [`parametric_var`](../guide/risk.md#parametric-var-delta-normal) result, and to defend any gap. When they diverge, the gap is a diagnostic on convexity that the CCAR narrative typically needs to address.

### 3.6 Reproducibility for the audit trail

CCAR results are heavily audited. Every VALAX pricing function has the signature `(instrument, MarketData) → price` and is a **pure function** with no mutable state — this is called out explicitly for the auditor use case: *same inputs ⇒ same outputs*, so an auditor can replay yesterday's books to the bit ([Risk: End-to-End § 2](../risk-overview.md#2-who-consumes-risk-numbers-and-what-they-actually-want) and [§ 6](../risk-overview.md#6-why-valax-is-built-the-way-it-is)). Deterministic autodiff Greeks and JIT-compiled scenario runs give the same answer on the same input on any hardware.

---

## 4. Coverage today vs. roadmap

Being honest about scope, because a bank programme has to know where the gaps are:

| FRTB / CCAR need | Status | Component(s) |
|---|---|---|
| Autodiff Greeks (delta, gamma, vega, vanna, volga, DV01, KRD) | ✅ | First- and second-order `SensitivityLadder` |
| Scenario generation (parametric, historical, named stress) | ✅ | `parametric_scenarios`, `historical_scenarios`, `stress_scenario`, `steepener`, `butterfly`, `flattener` |
| Full-revaluation portfolio P&L under scenarios | ✅ | `hypothetical_pnl_vector` / `portfolio_pnl` (vmapped) |
| Ladder-based P&L prediction (RTPL) | ✅ | `risk_theoretical_pnl_vector` |
| VaR and Expected Shortfall | ✅ | `value_at_risk`, `expected_shortfall`, `parametric_var` |
| Basel VaR backtest (Kupiec, Christoffersen, traffic light) | ✅ | `kupiec_pof`, `christoffersen_conditional_coverage`, `basel_traffic_light` |
| FRTB PLA test (Spearman + KS + d558 zone) | ✅ | `pla_spearman`, `pla_ks`, `pla_traffic_light` |
| Bucketing to FRTB / SIMM vertices | ✅ | `tenor_bucket_map`, `equal_weight_bucket_map`, `bucket_sensitivity_ladder` |
| SBA capital aggregation (inter-bucket correlations + curvature add-on) | 📋 | Bucket-level Δ + γ computed today; prescribed correlation formulas planned |
| Vol-surface risk factors (grid / SABR / SVI parameter risk) | ✅ / 📋 | Bucketing/Jacobian machinery shipped; full grid shocks planned |
| Multi-curve `MarketData` (OIS + SOFR + basis + XCCY) | 📋 | Single-curve today; extension is architectural, not disruptive |
| IMA stressed-period selection (automated) | 📋 | Caller supplies the 250-day window today |
| XVA (CVA / FVA / KVA) for CCAR trading-book add-ons | 📋 | Needs planned CDS pricer + credit-IR joint MC |

The shortest critical path to a desk-usable, bank-style FRTB/CCAR system is: **(a) multi-curve `MarketData` → (b) vol-surface risk factors → (c) FRTB SBA capital aggregation** — which together close roughly 90% of the gap to a single-desk production deployment. Full priority order lives in [Risk Factors § 4](../risk-factors.md#4-roadmap) and the project [Roadmap](../roadmap.md).

---

## 5. Why the JAX foundation matters for these programmes specifically

FRTB and CCAR are *volume* problems as much as they are correctness problems — thousands of instruments × hundreds of risk factors × thousands of scenarios × 250-day windows. The library's design decisions land directly on that ([Risk: End-to-End § 6](../risk-overview.md#6-why-valax-is-built-the-way-it-is), [Design Rationale](../design-rationale.md)):

| Design choice | Regulatory payoff |
|---|---|
| **Autodiff instead of bump-and-reprice** | Ladder + PLA Greeks in one reverse-mode pass at ~3× a single pricing call. No 50 000-line bump framework to maintain. Critically for PLA: no bump noise biasing the RTPL–HPL comparison. |
| **`vmap` over scenarios and instruments** | A 10 000-scenario ES run or a portfolio KRD sweep is one JIT-compiled call, not a Python loop. Bank-scale portfolios become feasible on a single GPU. |
| **One `MarketData` pytree** | `jax.grad(V)(market)` returns sensitivities to *all* risk factors in one call — no per-factor plumbing to maintain when the FRTB factor list expands. |
| **Bucketing as a matrix multiply** | Trader view, MRM view, FRTB SBA view, ISDA SIMM view, and PCA/ALM view are the same ladder × a different `A` or `J`. Adding a new regulatory view is a five-line change, not a new subsystem. |
| **Pure functions + integer-ordinal dates** | Deterministic replay for auditors and validators; the calibration/risk loop fits entirely inside `jit` with no Python `datetime` round-trips on the hot path — which is what makes overnight CCAR runs realistic. |

The net effect: the derivatives pricing engine and the regulatory reporting engine are the *same engine*, differenced or bucketed differently. One library, one mental model, one code path from a trader's delta at 14:31 to the firm-wide 97.5% ES on 250 stressed-period scenarios overnight — and to the CCAR Severely Adverse waterfall the following morning.

---

## 6. Where to read next

- **The pipeline these workflows sit on** → [Risk: End-to-End](../risk-overview.md).
- **The mathematics of VaR / ES / backtesting / PLA / bucketing** → [Models & Theory § 7](../theory.md#7-risk-measures).
- **The concrete code walkthrough** → [Risk & Scenarios guide](../guide/risk.md).
- **The factor taxonomy — what is and is not modelled** → [Risk Factors](../risk-factors.md).
- **The horizontal view of the systems around VALAX in a bank stack** → [Where VALAX Fits](../landscape.md).
- **The forward direction, including multi-curve, vol-surface, and SBA aggregation** → [Vision](../vision.md) and [Roadmap](../roadmap.md).
