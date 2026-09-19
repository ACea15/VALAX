# Applications

*A map of who inside a financial institution actually uses VALAX, why they use it, and where in the library to find the tools they need — organised by audience rather than by module.*

The rest of the documentation is organised by *what* the library does — pricing, curves, risk, calibration. This section is organised by *who* uses it. If you are evaluating VALAX for a specific function inside a bank, an asset manager, a hedge fund, or an academic group, start with the distinctive **capabilities** in [§ 2](#2-what-makes-valax-distinctive-capability-by-capability), then jump to the **profile** that fits your role.

---

## 1. The sixty-second story: one pipeline, many audiences

Every function of the library reduces to one primitive:

```
V(instrument, MarketData) → price
```

— a pure JAX function you can differentiate, JIT-compile, and `vmap`. Every downstream capability is a derived quantity on top of that primitive:

- **Greeks** are `jax.grad(V)`.
- **Portfolios** are `jax.vmap(V)`.
- **VaR / ES** are sample statistics on a P&L vector produced by scenario `vmap`.
- **Regulatory capital** is a bucketed contraction of the sensitivity ladder.
- **Neural surrogates** train against the pricer's own gradient as ground truth.
- **Deep-hedging policies** learn against the gradient of P&L through the SDE simulation.

*The same code path* serves the trader's morning delta, the risk manager's VaR, the model validator's challenger benchmark, the treasurer's IRRBB report, the regulator's FRTB submission, and the researcher's neural-SDE calibration. That "one pipeline, many audiences" property is what these application pages document.

For the systems around VALAX in the bank stack (order-management, market data, settlement), see [Where VALAX Fits](../landscape.md). For the risk engine's internal architecture, see [Risk: End-to-End](../risk-overview.md). For the deep motivation behind each design decision, see [Design Rationale](../design-rationale.md).

---

## 2. What makes VALAX distinctive — capability by capability

The five profiles below tell one story from five seats. The story itself is a short list of *capabilities* that a pure-functional, JAX-native engine has and a bump-and-reprice C++ stack structurally does not — stated **once** here, then applied per business. For *why* the architecture is built this way, see [Design Rationale § 3](../design-rationale.md#3-the-seven-choices-valax-commits-to-and-what-each-one-buys); for who ranks where, the [audience map](#3-the-audience-map-ranked-by-fit-today) below.

Business tags: `MRM` Market Risk & Model Validation · `QR` Quant Research · `FO` Front-Office Trading · `REG` Regulatory (FRTB/CCAR) · `TRE` Treasury & ALM.

### 2.1 Exact autodiff risk — no bump-and-reprice

Every Greek, and the full first-/second-order sensitivity ladder, comes from one reverse-mode pass over the pricer (`jax.grad` / `jax.hessian`, `compute_ladder`) — exact to machine precision, with no separate bump framework to drift out of sync with the pricer it shadows.

**Who it serves:** `MRM` the entire factor ladder in one pass, and a bump-free RTPL that stops finite-difference noise from failing FRTB PLA · `FO` exact hedge deltas with none of the noise that is worst near barriers and expiry · `REG` the sensitivities FRTB asks for *are* `jax.grad` output — no 50k-line bump engine · `TRE` exact key-rate durations across 20+ pillars, no bump-width dispute with the auditor · `QR` every new model ships with all Greeks the moment its price function exists.

### 2.2 Batch at scale — `vmap` + JIT + GPU

`jax.vmap` over instruments and scenarios collapses the Python loop of a C++ stack into a single JIT-compiled kernel; the batch *is* the unit of computation, and the identical code runs on CPU, GPU, or TPU.

**Who it serves:** `MRM`/`REG` a 10k-scenario × 5k-instrument full-reval VaR/ES compiles to one call — minutes on a GPU · `TRE` a 500k-position banking book across the six IRRBB scenarios is one call, not an overnight batch · `FO` a 50k-name quote universe reprices sub-second on a warm cache · `QR` parameter sweeps and MC experiments are one `vmap` at GPU speed.

### 2.3 One market-state object, one code path

A single `MarketData` pytree carries every risk factor, and the *same* JIT-compiled `V` serves the trader's hot price, the EOD risk batch, the regulator's backtest, and the audit replay. Internal numbers reconcile by construction rather than through a nightly reconciliation job.

**Who it serves:** `MRM` trader Greeks, desk DV01, CRO cross-gamma, FRTB SBA, and PLA RTPL are all slices of one ladder · `REG` SBA and IMA prototype against the same instruments and the same `MarketData`, with no second stack · `TRE` the HQLA duration in the ALM report and in the CFO's return are the same array, sliced differently · `FO` a pre-trade what-if runs the same compiled function as the overnight batch.

### 2.4 Determinism & independent validation

Pure functions plus integer-ordinal dates make every result bit-exactly replayable; the synthetic-market harness and the shipped QuantLib comparison suite make the engine an independent, evidence-backed challenger.

**Who it serves:** `MRM` an independent-by-construction challenger in Python + JAX (not a wrapper over the C++ champion) — exactly what SR 11-7 / TRIM demand · `REG`/`TRE` same inputs ⇒ same outputs, so an auditor replays yesterday's books to the bit · `QR` publish with a git tag and reviewers re-run the notebook.

### 2.5 Differentiable into the decision layer ⚑ (frontier)

Because the whole pipeline is one differentiable graph, the derivative does not stop at model inputs — it reaches the quantities a desk or risk team can act on. The *primitives* are shipped and exercised; the *packaged products* are prototype-on-top, not install-and-go.

**Who it serves:** `FO` exact hedges in *tradable instruments* via implicit diff through calibration, and gradient-based intraday optimal hedging / quoting ([Front-Office § 2](trading-desk.md#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer)) · `MRM` gradients *of the risk measure* — exact component/marginal ES (Euler allocation) and gradient-based reverse stress testing ([Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier)) · `QR` deep hedging: the gradient of hedged P&L straight through the SDE simulation ([Quant Research § 6](quant-research.md#6-deep-hedging-the-flagship-differentiable-finance-application)).

### 2.6 JAX-ecosystem composability

The pricer is a plain JAX function, so `diffrax` (SDE simulation), `optimistix` (calibration / root-finding), `optax` (gradient optimisation), and `equinox` (pytree models & networks) compose with no serialisation boundary — and `flax` / `numpyro` / `blackjax` are one import away.

**Who it serves:** `QR` SDE research, gradient-based calibration, neural surrogates, and MCMC calibration in one environment — paper to calibrated prototype in hours · `FO` warm-started `optimistix` recalibration keeps intraday surfaces stable · `TRE` behavioural models (deposit pass-through, prepayment) become `optax`-calibrated `equinox.Module`s on top of the same kernel.

---

## 3. The audience map — ranked by fit today

Not every function in a financial institution is a good fit for VALAX today. The grade is the strength of the *capability fit*: 🟢🟢🟢 an immediate, near-complete fit; 🟢🟢 a strong fit with a known roadmap gap; some are 🟡 potentially doable; 🔴 out of scope by design. The honest ranking:

| Rank | Audience | Fit today | Why | Sub-page |
|---|---|---|---|---|
| 1 | **Market Risk & Model Validation** | 🟢🟢🟢 | Regulator-mandated need; no incumbent lock-in; batch cadence | [Market Risk & Model Validation](market-risk.md) |
| 2 | **Quant Research / Structuring / Model R&D** | 🟢🟢🟢 | Differentiable pricing is a categorical advantage — enables deep hedging, neural surrogates, gradient-based calibration that C++ analytics libraries structurally cannot support | [Quant Research](quant-research.md) |
| 3 | **Front-office trading (pricing core)** | 🟢🟢 | The pricing/Greeks/calibration core is a genuine strength and the differentiator is *categorical* — differentiability into the decision layer (exact hedges in tradable instruments; frontier gradient-based hedging/quoting). Held back from 🟢🟢🟢 only by the intraday *service layer* being Vision-tier. Valuation core feeding a *separate* strategy/execution system — not the alpha, not the OMS | [Front-Office Trading](trading-desk.md) |
| 4 | **Regulatory reporting (FRTB / CCAR)** | 🟢🟢 | Bucketed sensitivities, PLA test, VaR backtesting all shipped; SBA aggregation on roadmap | [Regulatory: FRTB & CCAR](regulatory.md) |
| 5 | **Treasury / ALM** | 🟡 | IRRBB Standardised Approach is a near-perfect fit; HQLA pricing is turnkey. | [Treasury & ALM](treasury.md) |
| 6 | **Back office / operations** | 🔴 | Wrong domain — not what the library does, by design | — |

The top two audiences (positions 1–2) represent two different theses about the library:

- **Market Risk & Model Validation** is the strongest **commercial** adoption case — regulator-mandated need, no incumbent lock-in, budget in hand.
- **Quant Research** is the strongest **technical** adoption case — differentiable pricing enables research directions that are structurally out of reach for C++ analytics libraries.


---

## 3. The five sub-pages

Each sub-page below is written for a specific audience — role, cadence, KPIs, and the buyer-tailored pitch. Read the one that matches your function first; skim the others for context.

### 🥇 [Market Risk & Model Validation](market-risk.md)

**For:** Middle-office Market Risk Management (MRM) and second-line Model Validation (SR 11-7 / TRIM).

The daily MRM production loop — sensitivity ladders, waterfall P&L explain, full-revaluation VaR/ES, Basel backtesting, limits monitoring, FRTB PLA — all in one code path. Plus the Model Validation angle: an independent-by-construction challenger pricer in Python + JAX rather than C++, with a QuantLib comparison suite already in the tree covering options, fixed income, Heston, SABR, caps, swaptions, MC, PDE, lattice, SLV, and Greeks. Deterministic replay for supervisory inspections.

### 🥇 [Quant Research](quant-research.md)

**For:** Front-office quants, structuring desks, model R&D teams, hedge-fund systematic PMs, academic researchers.

The argument that differentiable pricing is not a "nice implementation choice" but the primary source of research-productivity leverage: every new pricer ships with all Greeks; a new payoff is a Python function, not a class subtree; calibration is `optimistix.least_squares` on the autodiff Jacobian; neural surrogates and deep hedging become one-file prototypes rather than multi-quarter engineering projects. The paper-to-calibrated-prototype loop collapses from months to hours. This is the innovation-track adoption case and — arguably — the strongest technical fit of any audience.

### 🥈 [Front-Office Trading](trading-desk.md)

**For:** front-office desks running an in-house intraday trading platform — desk quants, desk technology, desk risk.

The honest case for VALAX as the differentiable **pricing-and-risk core** of an intraday (not high-frequency) trading system — and, crucially, the honest case for *what is actually novel* about it. EOD batch repricing and bumped Greeks are a commodity every bank already has; intraday cadence is a useful but incremental improvement. The categorical claim is that the pricing graph is **differentiable into the decision layer**: exact hedges in tradable instruments via implicit diff through calibration (near-shipped), and gradient-based intraday optimal hedging / quoting where the control problem shares the pricer's graph (frontier — primitives shipped, product not). Supported by the three enabling strengths (fast `vmap`+`filter_jit` kernels, exact autodiff Greeks, warm-started robust calibration). Deliberately draws a hard boundary — strategy, sizing, and execution live in a *separate* system that consumes VALAX prices, Greeks, and Jacobians; the pricer is not the alpha and not the OMS.

### 🥈 [Regulatory: FRTB & CCAR](regulatory.md)

**For:** Regulatory reporting teams, capital planning, CROs, supervisory dialogue.

The two flagship trading-book regulatory workflows — Basel III.1 FRTB (SBA and IMA, PLA, backtesting) and the Fed's annual CCAR / DFAST stress tests. Both start from the same VALAX pipeline; they differ only in how bucketing and scenarios are configured. Coverage-vs-roadmap tables are explicit about what ships today (autodiff Greeks, PLA test, bucketing, VaR backtest, ES) and what is on the roadmap (SBA correlation aggregation, multi-curve `MarketData`, automated IMA stress-period selection).

### 🥈 [Treasury & ALM](treasury.md)

**For:** Head of ALM, Treasurers, banking-book risk teams, LCR/NSFR reporting teams, HQLA portfolio managers.

The IRRBB Standardised Approach walkthrough — the six BCBS 368 scenarios map 1-to-1 to VALAX primitives (`parallel_shift`, `steepener`, `flattener`, `key_rate_bump`) and ΔEVE + SOT + KRD bucketing are one call each. Plus HQLA / investment-portfolio pricing, cash-flow bucketing for LCR / NSFR, and the honest gap on behavioural models (deposit pass-through, prepayment) — which are the natural home for the JAX moat once someone builds them.

---

## 5. Coverage — what ships today vs. roadmap

This is the single, consolidated coverage reference for every application above; each sub-page links here instead of repeating its own matrix. A capability that serves several businesses appears **once**, with all of them listed in the *Business* column.

**Status:** ✅ shipped & tested · 🟡 primitives shipped, orchestration not yet packaged · 📋 roadmap · ⚑ differentiable frontier (primitives shipped, product not packaged) · 🔴 out of scope by design.

**Business:** `MRM` Market Risk & Model Validation · `QR` Quant Research · `FO` Front-Office Trading · `REG` Regulatory (FRTB/CCAR) · `TRE` Treasury & ALM · `All` every audience.

| Capability | Status | Business | Component(s) / notes |
|---|---|---|---|
| Differentiable pricer `V(instrument, market) → price` | ✅ | All | Every `valax/pricing/*` function |
| Autodiff Greeks & sensitivity ladders (1st/2nd order: Δ, Γ, vega, vanna, volga, DV01, KRD) | ✅ | All | `jax.grad`/`jax.hessian`, `compute_ladder`, `SensitivityLadder`, `valax/greeks/` |
| Whole-book / universe batched revaluation | ✅ | All | `jax.vmap` + `@eqx.filter_jit`, `portfolio_pnl` |
| GPU / TPU execution (same code) | ✅ | All | JAX / XLA |
| VaR & Expected Shortfall (full-reval + parametric delta-normal) | ✅ | MRM, REG | `hypothetical_pnl_vector`, `value_at_risk`, `expected_shortfall`, `parametric_var` |
| Scenario generation (historical, parametric, named stress) | ✅ | MRM, REG, TRE | `historical_scenarios`, `parametric_scenarios`, `stress_scenario`, `steepener`, `flattener`, `butterfly` |
| Scenario / pre-trade "what-if" P&L (incl. intraday grids) | ✅ | FO, MRM, REG, TRE | nested `vmap`, `apply_scenario`, `hypothetical_pnl_vector` |
| Ladder-based P&L prediction (RTPL) | ✅ | MRM, REG | `risk_theoretical_pnl_vector` |
| Waterfall P&L explain (10 rungs + unexplained) | ✅ | MRM, REG | `waterfall_pnl_report`, `waterfall_pnl` |
| Basel VaR backtest (Kupiec, Christoffersen, traffic light) | ✅ | MRM, REG | `var_breaches`, `kupiec_pof`, `christoffersen_conditional_coverage`, `basel_traffic_light` |
| FRTB PLA test (Spearman + KS + d558 zone) | ✅ | MRM, REG | `pla_spearman`, `pla_ks`, `pla_traffic_light` |
| Bucketing (FRTB/SIMM vertices, sector, PCA, Basel tenor) | ✅ | MRM, REG, TRE | `tenor_bucket_map`, `equal_weight_bucket_map`, `pca_jacobian`, `bucket_sensitivity_ladder` |
| Differentiable SDE path simulation | ✅ | QR | `diffrax` — GBM, Heston, LMM, SABR, local vol, SLV |
| Robust gradient-based calibration (Heston, SABR, SLV) | ✅ | QR, FO | `optimistix` LM + `optax`, `valax/calibration/` |
| Vol-surface models (SABR, SVI, SLV, Dupire) | ✅ | QR | `valax/surfaces/`, `valax/models/slv.py`, `valax/models/local_vol.py` |
| Neural-network container | ✅ | QR | `equinox.Module` |
| Quote-level hedge Jacobian (risk in tradable instruments; implicit diff through calibration) | ✅ | FO | `quote_jacobian` (`optimistix.ImplicitAdjoint`), `compute_ladder` |
| Calibration-stability monitoring (`vmap` robustness sweep) | ✅ | FO | `jax.vmap` over perturbed quotes |
| IRRBB 6-scenario ΔEVE (BCBS 368 Standardised) | ✅ | TRE | `parallel_shift`, `steepener`, `flattener`, `key_rate_bump`, `hypothetical_pnl_vector` |
| HQLA / investment-portfolio pricing (fixed, floating, callable) | ✅ | TRE | `valax/instruments/bonds.py`, `valax/pricing/analytic/bonds.py`, callable-bond lattice |
| QuantLib comparison suite (14 modules) | ✅ | MRM, QR, FO | `tests/test_quantlib_comparison/` |
| Golden tests / deterministic replay for audit | ✅ | All | `tests/golden/` + `golden_manifest.json`; pure functions + integer-ordinal dates |
| Vol-surface risk factors (grid / SABR / SVI parameter risk) | ✅ / 📋 | MRM, REG | Bucketing/Jacobian machinery shipped; full grid shocks planned |
| Intraday `MarketState` build + calibration diagnostics | 🟡 | FO | [Productionisation § 5](../architecture/production.md#5-layer-2-3-build-and-calibration-workflow); not fully packaged |
| Long-lived pricing service (gRPC/REST, resident state) | 🟡 | FO, MRM | [Productionisation § 8](../architecture/production.md#8-layer-6-service-layer-deferred) — deferred |
| NII 12-month simulation engine (position walker) | 🟡 | TRE | All primitives present; orchestration layer not yet packaged |
| LCR / NSFR contractual cash-flow bucketing | 🟡 | TRE | Schedule + bucketing primitives present; run-off-factor library not shipped |
| Multi-curve `MarketData` (OIS + SOFR + basis + XCCY) | 📋 | MRM, REG, TRE | Top-priority roadmap item; single-curve today |
| SBA capital aggregation (inter-bucket correlations + curvature) | 📋 | MRM, REG | Bucket-level Δ + γ today; prescribed correlation formulas planned |
| Named factor registry (`IR.OIS.USD.5Y`, …) | 📋 | MRM | Positional layout today; registry planned |
| IMA stressed-period selection (automated) | 📋 | REG | Caller supplies the 250-day window today |
| XVA (CVA / FVA / KVA) | 📋 | REG, QR | Needs planned CDS pricer + credit-IR joint MC |
| Streaming / incremental market-data updates | 📋 | FO | [Roadmap → Real-Time Risk](../roadmap.md#63-real-time-risk) |
| JIT warm-up / shape-bucketing service harness | 📋 | FO | Vision-tier; prerequisite for quote-response latency |
| FTP curve arithmetic (base + liquidity + credit add-ons) | 📋 | TRE | Depends on multi-curve |
| Behavioural models — prepayment, deposit pass-through, non-maturity-deposit life curves | 📋 | TRE | Not on roadmap; natural JAX extension (`equinox.Module` + `jax.grad` + `optax`) |
| Credit-adjusted own-funding discount curve | 📋 | TRE | Depends on multi-curve |
| Worked neural-surrogate & deep-hedging examples | 📋 | QR | All primitives shipped; canonical notebooks planned |
| Differentiable portfolio construction | 📋 | QR | P5 frontier item — see [Vision](../vision.md) |
| Rough-volatility / neural-SDE calibration | 📋 | QR | `diffrax` supports the paths; canonical example planned |
| Differentiable risk *measures* (∂VaR/∂position, component/marginal ES, Euler allocation) | ⚑ | MRM | Primitives shipped; allocator not packaged — see [Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier) |
| Gradient-based reverse stress testing | ⚑ | MRM | Differentiable pricer + `optimistix`/`optax`; optimiser not packaged — [Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier) |
| Gradient-based intraday optimal hedging / quoting | ⚑ | FO | Primitives shipped (`diffrax`, `optax`, `eqx.filter_grad`); product not packaged — [Front-Office § 2.2](trading-desk.md#22-gradient-based-intraday-decisions-the-pricer-and-the-control-problem-share-one-graph-frontier) |
| Strategy / signal generation (alpha) | 🔴 | FO | Separate system by design |
| Order management / execution / fills | 🔴 | FO | Separate system by design |
| Back office / operations | 🔴 | — | Wrong domain by design |

> **Reading the frontier (⚑).** These are the categorical advantages a differentiable engine has over an enumerate-and-bump stack: the *primitives* ship and are exercised, but the *packaged product* is a prototype-on-top, not an install-and-go feature. They are the honest answer to "how is this more than a fast EOD engine" — real potential, roadmap delivery. The [adoption order in § 6](#6-the-strategic-order-of-operations) sequences all of this.

---

## 6. The strategic order-of-operations

This is a different axis from the [fit ranking of § 3](#3-the-audience-map-ranked-by-fit-today): that ranks *capability fit today*, this ranks *adoption sequence*. Front-office trading can be a strong capability fit (🟢🟢) yet sit late here, because deploying it as an intraday service is Vision-tier. For a bank considering serious adoption, the sequence that maximises return and minimises risk:

1. **Model Validation first.** SR 11-7 / TRIM already require an independent challenger; VALAX is architected to be one. The QuantLib comparison suite is a shipped evidence pack. Adoption here is the lowest-friction, highest-defensibility entry point.
2. **Simultaneously: Quant Research adoption** — different budget, different buyer, provides the frontier-capability story (neural surrogates, deep hedging, differentiable calibration). The two teams cross-pollinate: research prototypes graduate into validated production without a rewrite because it is the same library.
3. **Expand into MRM** once Model Validation is a happy internal reference. Same code path, adjacent workflow, natural cross-sell to the CRO.
4. **Treasury / ALM** once multi-curve `MarketData` ships (roadmap top priority).
5. **Regulatory reporting** falls out as an *output* of steps 1–4 — the FRTB PLA, VaR backtest, and CCAR waterfall are computations on top of the same pipeline.
6. **Front-office trading** — Vision-tier as a *deployed intraday service*, though the pricing/Greeks/calibration core is already strong. Pursue only after the service layer, market-data adapters, and JIT warm-up strategy described in [architecture/production.md § 8.1](../architecture/production.md#81-intraday-warm-serving-considerations-deferred) are in place — and only ever as the valuation core feeding a *separate* strategy/execution system. See [Front-Office Trading](trading-desk.md) for the honest boundary.
7. **Back office** — never. Out of scope by design.

The through-line: **do not start with trading, and do not start with regulatory reporting**. Start with the two functions that are structurally set up to benefit (Model Validation, Quant Research) and let the rest of the stack pull adoption naturally as the library becomes an internal reference.

---

## 7. Where to read next

- **Every instrument, its pricing method, vol input, and calibration** → [Instrument Pricing](instrument-pricing.md).
- **The systems around VALAX in a bank stack** → [Where VALAX Fits](../landscape.md).
- **The end-to-end risk pipeline these applications sit on** → [Risk: End-to-End](../risk-overview.md).
- **The mathematical foundations** → [Models & Theory](../theory/index.md).
- **The concrete code walkthroughs** → [User Guide](../guide/tutorial-rates.md).
- **The forward direction** → [Vision](../vision.md), [Roadmap](../roadmap.md).
- **Why the library makes the architectural choices it does** → [Design Rationale](../design-rationale.md).
.md).
