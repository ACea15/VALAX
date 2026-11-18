# Applications

*A map of who inside a financial institution actually uses VALAX, why they use it, and where in the library to find the tools they need — organised by audience rather than by module.*

The rest of the documentation is organised by *what* the library does — pricing, curves, risk, calibration. This section is organised by *who* uses it. If you are evaluating VALAX for a specific function inside a bank, an asset manager, a hedge fund, or an academic group, start with the distinctive **capabilities** in [§ 2](#2-what-makes-valax-distinctive-capability-by-capability), then jump to the **profile** that fits your role.

---

## 1. The sixty-second story: one pipeline, many audiences

Every function of the library reduces to one primitive:

```
price(instrument, model, market: MarketState) → PricingResult
```

— a pure JAX function you can differentiate, JIT-compile, and `vmap`. The `MarketState` is one pytree holding every curve of the multi-curve graph, every vol surface, spot and dividend; the `model` says how the instrument is valued against it. Every downstream capability is a derived quantity on top of that primitive:

- **Greeks** are `jax.grad` of `price` with respect to the market (`market_risk`), back to the quoted instruments through the calibration (`chain_adjoint`).
- **Portfolios** are `jax.vmap` of `price` over a batch of trades.
- **VaR / ES** are sample statistics on a P&L vector produced by `vmap` over keyed scenarios.
- **Regulatory capital** is a bucketed contraction of the sensitivities.
- **Neural surrogates** train against the pricer's own gradient as ground truth.
- **Deep-hedging policies** learn against the gradient of P&L through the SDE simulation.

*The same code path* serves the trader's morning delta, the risk manager's VaR, the model validator's challenger benchmark, the treasurer's IRRBB report, the regulator's FRTB submission, and the researcher's neural-SDE calibration. That "one pipeline, many audiences" property is what these application pages document.

For the systems around VALAX in the bank stack (order-management, market data, settlement) and the motivation behind each design decision, see [Positioning & Design Rationale](../positioning/index.md). For the risk engine's internal architecture, see [Risk: End-to-End](../risk-overview.md).

---

## 2. What makes VALAX distinctive — capability by capability

The five profiles below tell one story from five seats. The story itself is a short list of *capabilities* that a pure-functional, JAX-native engine has and a bump-and-reprice C++ stack structurally does not — stated **once** here, then applied per business. For *why* the architecture is built this way, see [Positioning § 4](../positioning/index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys); for who ranks where, the [audience map](#3-the-audience-map-ranked-by-fit-today) below.

Business tags: `MRM` Market Risk & Model Validation · `QR` Quant Research · `FO` Front-Office Trading · `REG` Regulatory (FRTB/CCAR) · `TRE` Treasury & ALM.

### 2.1 Exact autodiff risk — no bump-and-reprice

Every first-order Greek and bucketed sensitivity comes from one reverse-mode (AAD) pass over the pricer, and the second-order set from one `jax.hessian` on the same function (`jax.grad` / `jax.hessian` through `price()`; `market_risk` for a trade, `keyed_sensitivities` for a book) — exact to machine precision, with no separate bump framework to drift out of sync with the pricer it shadows.

**Who it serves:** `MRM` every factor sensitivity in one pass, and a bump-free RTPL that stops finite-difference noise from failing FRTB PLA · `FO` exact hedge deltas with none of the noise that is worst near barriers and expiry · `REG` the sensitivities FRTB asks for *are* `jax.grad` output — no 50k-line bump engine · `TRE` exact key-rate durations across 20+ pillars, no bump-width dispute with the auditor · `QR` every new model ships with all Greeks the moment its price function exists.

### 2.2 Batch at scale — `vmap` + JIT + GPU

`jax.vmap` over instruments and scenarios collapses the Python loop of a C++ stack into a single JIT-compiled kernel; the batch *is* the unit of computation, and the identical code runs on CPU, GPU, or TPU.

**Who it serves:** `MRM`/`REG` a 10k-scenario × 5k-instrument full-reval VaR/ES compiles to one call — minutes on a GPU · `TRE` a 500k-position banking book across the six IRRBB scenarios is one call, not an overnight batch · `FO` a 50k-name quote universe reprices sub-second on a warm cache · `QR` parameter sweeps and MC experiments are one `vmap` at GPU speed.

### 2.3 One market-state object, one code path

A single `MarketState` pytree carries every risk factor, each named by identifier (`USD.OIS`, `USD.SOFR.3M`, a surface id, an asset id), and the *same* JIT-compiled `price` serves the trader's hot price, the EOD risk batch, the regulator's backtest, and the audit replay. Internal numbers reconcile by construction rather than through a nightly reconciliation job.

**Who it serves:** `MRM` trader Greeks, desk DV01, CRO cross-gamma, FRTB SBA, and PLA RTPL are all slices of one set of sensitivities · `REG` SBA and IMA prototype against the same instruments and the same `MarketState`, with no second stack · `TRE` the HQLA duration in the ALM report and in the CFO's return are the same array, sliced differently · `FO` a pre-trade what-if runs the same compiled function as the overnight batch · **product control / finance** explain the day-over-day move in official P&L with `keyed_waterfall_report`, one term per named curve, surface and asset, on the same sensitivities the desk sees.

### 2.4 Determinism & independent validation

Pure functions plus integer-ordinal dates make every result bit-exactly replayable; the synthetic-market harness and the shipped QuantLib comparison suite make the engine an independent, evidence-backed challenger.

**Who it serves:** `MRM` an independent-by-construction challenger in Python + JAX (not a wrapper over the C++ champion) — exactly what SR 11-7 / TRIM demand · `REG`/`TRE` same inputs ⇒ same outputs, so an auditor replays yesterday's books to the bit · **internal / external audit** reproduce yesterday's risk numbers from yesterday's market data, with each quote traced to its snapshot (`snapshot_id` provenance) · `QR` publish with a git tag and reviewers re-run the notebook.

### 2.5 Differentiable into the decision layer ⚑ (frontier)

Because the whole pipeline is one differentiable graph, the derivative does not stop at model inputs — it reaches the quantities a desk or risk team can act on. The *primitives* are shipped and exercised; the *packaged products* are prototype-on-top, not install-and-go.

**Who it serves:** `FO` exact hedges in *tradable instruments* via implicit diff through calibration, and gradient-based intraday optimal hedging / quoting ([Front-Office § 2](trading-desk.md#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer)) · `MRM` gradients *of the risk measure* — exact component/marginal ES (Euler allocation) and gradient-based reverse stress testing ([Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier)) · `QR` deep hedging: the gradient of hedged P&L straight through the SDE simulation ([Quant Research § 6](quant-research.md#6-deep-hedging-the-flagship-differentiable-finance-application)).

### 2.6 JAX-ecosystem composability

The pricer is a plain JAX function, so `diffrax` (SDE simulation), `optimistix` (calibration / root-finding), `optax` (gradient optimisation), and `equinox` (pytree models & networks) compose with no serialisation boundary — and `flax` / `numpyro` / `blackjax` are one import away.

**Who it serves:** `QR` SDE research, gradient-based calibration, neural surrogates, and MCMC calibration in one environment — paper to calibrated prototype in hours · `FO` warm-started `optimistix` recalibration keeps intraday surfaces stable · `TRE` behavioural models (deposit pass-through, prepayment) become `optax`-calibrated `equinox.Module`s on top of the same kernel.

---

## 3. The audience map — ranked by fit today

Each audience has its own page covering role, cadence, KPIs and the pitch written for that buyer. This section is the only entry point to those pages. Read the one that matches your function first, and skim the others for context.

Not every function in a financial institution is a good fit for VALAX today. The grade shows how strong the *capability fit* is: 🟢🟢🟢 an immediate, near-complete fit; 🟢🟢 a strong fit with a known roadmap gap; 🟡 potentially doable; 🔴 out of scope by design. The ranking:

| Rank | Audience | Fit today | Why |
|---|---|---|---|
| 1 | [**Market Risk & Model Validation**](#31-market-risk-model-validation) | 🟢🟢🟢 | Regulator-mandated need; no incumbent lock-in; batch cadence |
| 2 | [**Quant Research / Structuring / Model R&D**](#32-quant-research) | 🟢🟢🟢 | Differentiable pricing is a categorical advantage |
| 3 | [**Front-office trading (pricing core)**](#33-front-office-trading) | 🟢🟢 | Strong pricing core; intraday service layer is Vision-tier |
| 4 | [**Regulatory reporting (FRTB / CCAR)**](#34-regulatory-frtb-ccar) | 🟢🟢 | Sensitivities, PLA, backtests shipped; SBA aggregation on roadmap |
| 5 | [**Treasury / ALM**](#35-treasury-alm) | 🟡 | IRRBB Standardised Approach fits; behavioural models missing |
| 6 | **Back office / operations** | 🔴 | Wrong domain — not what the library does, by design |

The top two audiences stand for two different theses about the library:

- **Market Risk & Model Validation** is the strongest **commercial** adoption case: regulator-mandated need, no incumbent lock-in, budget in hand.
- **Quant Research** is the strongest **technical** adoption case: differentiable pricing opens research directions that C++ analytics libraries structurally cannot reach.

### 3.1 Market Risk & Model Validation

**[→ Market Risk & Model Validation](market-risk.md)** · 🟢🟢🟢 · **For:** middle-office Market Risk Management (MRM) and second-line Model Validation (SR 11-7 / TRIM).

**Asks:** *"Is this desk within its DV01 / vega / VaR / stress limit? Did the model under-predict yesterday's loss? Does the engine agree with QuantLib, and is the calibration stable?"*

The daily MRM production loop runs in one code path: bucketed sensitivities, waterfall P&L explain, full-revaluation VaR/ES, Basel backtesting, limits monitoring and FRTB PLA. For Model Validation, VALAX is an independent-by-construction challenger pricer in Python + JAX rather than C++. A QuantLib comparison suite is already in the tree, covering options, fixed income, Heston, SABR, caps, swaptions, MC, PDE, lattice, SLV and Greeks. Deterministic replay supports supervisory inspections.

### 3.2 Quant Research

**[→ Quant Research](quant-research.md)** · 🟢🟢🟢 · **For:** front-office quants, structuring desks, model R&D teams, hedge-fund systematic PMs, academic researchers.

**Asks:** *"What is the sensitivity of CVA to a tweak in the calibration weights? Can I train a neural surrogate for autocallables?"*

Here differentiable pricing is the main source of research-productivity leverage, not just a nice implementation choice. Every new pricer ships with all its Greeks. A new payoff is a Python function, not a class subtree. Calibration is `optimistix.least_squares` on the autodiff Jacobian. Neural surrogates and deep hedging become one-file prototypes rather than multi-quarter engineering projects, so the path from paper to calibrated prototype shrinks from months to hours.

### 3.3 Front-Office Trading

**[→ Front-Office Trading](trading-desk.md)** · 🟢🟢 · **For:** front-office desks running an in-house intraday trading platform — desk quants, desk technology, desk risk.

**Asks:** *"What's the fair price? What's my delta? If SPX drops 2% what do I lose? Quote me an exotic in seconds, with Greeks."*

VALAX can serve as the differentiable **pricing-and-risk core** of an intraday (not high-frequency) trading system. EOD repricing and bumped Greeks are a commodity, and intraday cadence is only an incremental improvement. What is new is that the pricing graph is **differentiable into the decision layer**. That gives exact hedges in tradable instruments via implicit differentiation through calibration (near-shipped), and gradient-based intraday optimal hedging / quoting (frontier: primitives shipped, product not). The grade is held back from 🟢🟢🟢 only because the intraday *service layer* is Vision-tier. There is a hard boundary: strategy, sizing and execution live in a *separate* system that consumes VALAX prices, Greeks and Jacobians.

### 3.4 Regulatory: FRTB & CCAR

**[→ Regulatory: FRTB & CCAR](regulatory.md)** · 🟢🟢 · **For:** regulatory reporting teams, capital planning, CROs, supervisory dialogue.

**Asks:** *"Capital number. Backtest pass/fail. PLA pass/fail. Sensitivities at FRTB / SIMM vertices."*

This page covers the two flagship trading-book regulatory workflows: Basel III.1 FRTB (SBA and IMA, PLA, backtesting) and the Fed's annual CCAR / DFAST stress tests. Both run on the same VALAX pipeline and differ only in how bucketing and scenarios are configured. Autodiff Greeks, the PLA test, bucketing, the VaR backtest and ES ship today. SBA correlation aggregation and automated IMA stress-period selection are on the roadmap.

### 3.5 Treasury & ALM

**[→ Treasury & ALM](treasury.md)** · 🟡 · **For:** Head of ALM, Treasurers, banking-book risk teams, LCR/NSFR reporting teams, HQLA portfolio managers.

**Asks:** *"IR sensitivity by bucket? Funding gap by tenor? FX exposure overall?"*

The page walks through the IRRBB Standardised Approach. The six BCBS 368 scenarios map one-to-one onto VALAX primitives (`parallel_shift`, `steepener`, `flattener`, `key_rate_bump`), and ΔEVE, the supervisory outlier test (SOT) and KRD bucketing are one call each. It also covers HQLA / investment-portfolio pricing and cash-flow bucketing for LCR / NSFR. The gap is behavioural models (deposit pass-through, prepayment). Nobody has built them yet, and they are where JAX would add the most once someone does.

---

## 4. Coverage — what ships today vs. roadmap

<!-- coverage-map -->

This is the single, consolidated coverage reference for every application above; each sub-page links here instead of repeating its own matrix. A capability that serves several businesses appears **once**, with all of them listed in the *Business* column.

**Status:** ✅ shipped & tested · 🟡 primitives shipped, orchestration not yet packaged · 📋 roadmap · ⚑ differentiable frontier (primitives shipped, product not packaged) · 🔴 out of scope by design.

**Business:** `MRM` Market Risk & Model Validation · `QR` Quant Research · `FO` Front-Office Trading · `REG` Regulatory (FRTB/CCAR) · `TRE` Treasury & ALM · `All` every audience.

| Capability | Status | Business | Component(s) / notes |
|---|---|---|---|
| Differentiable pricer `price(instrument, model, market) → PricingResult` | ✅ | All | Every route on the `price()` front door (`valax/core/`); the `valax/pricing/*` kernels it dispatches to are public API too |
| Autodiff Greeks & bucketed sensitivities (1st/2nd order: Δ, Γ, vega, vanna, volga, DV01 per curve, KRD) | ✅ | All | `market_risk` (single trade), `keyed_sensitivities` / `KeyedSensitivities` (book, one block per named curve, surface, asset); `jax.grad`/`jax.hessian` through `price()` |
| Whole-book / universe batched revaluation | ✅ | All | `group_pv` / `group_reprice` through `price()`, `jax.vmap` + `@eqx.filter_jit` |
| GPU / TPU execution (same code) | ✅ | All | JAX / XLA |
| VaR & Expected Shortfall (full-reval + parametric delta-normal, with per-curve Euler decomposition) | ✅ | MRM, REG | `group_scenario_pnl` + `value_at_risk` / `expected_shortfall`; `keyed_parametric_var`, `keyed_component_var` |
| Scenario generation (historical, parametric, named stress, per-curve and basis) | ✅ | MRM, REG, TRE | `keyed_historical_scenarios`, `keyed_parametric_scenarios` over a per-curve `block_covariance`; `keyed_stress_scenario`, `keyed_curve_stress` (steepener / flattener / butterfly), `keyed_basis_shift` |
| Scenario / pre-trade "what-if" P&L | ✅ | FO, MRM, REG, TRE | `KeyedScenario` + `shock`, `group_scenario_pnl`, nested `vmap` |
| Revaluation grids / ladders at shifted levels (spot, vol, rate): P&L to all orders per direction | ✅ / 📋 | FO, MRM, REG | `revaluation_grid` → `RevaluationGrid`, `revaluation_grid_pnl` ([guide](../risk-revaluation-grids.md)); 2-D product grids and the FRTB curvature charge on top are [planned](../roadmap.md#44-frtb-and-desk-risk-programme) |
| FRTB PLA P&L vectors (HPL / RTPL) | ✅ / 🟡 | MRM, REG | `hypothetical_pnl_vector`, `risk_theoretical_pnl_vector` — shipped on the legacy flat `MarketData` path; port to the keyed sensitivities is [open](../roadmap.md#remaining-work) |
| Waterfall P&L explain (one term per market object + unexplained) | ✅ | MRM, REG | `keyed_waterfall_report`, `keyed_waterfall_pnl` (a pure basis move lands on the basis term only) |
| Basel VaR backtest (Kupiec, Christoffersen, traffic light) | ✅ | MRM, REG | `var_breaches`, `kupiec_pof`, `christoffersen_conditional_coverage`, `basel_traffic_light` |
| FRTB PLA test (Spearman + KS + d558 zone) | ✅ | MRM, REG | `pla_spearman`, `pla_ks`, `pla_traffic_light` |
| Bucketing (FRTB/SIMM vertices, sector, PCA, Basel tenor) | ✅ | MRM, REG, TRE | `tenor_bucket_map`, `equal_weight_bucket_map`, `pca_jacobian`; `bucket_sensitivities` still reads the flat sensitivities |
| Differentiable SDE path simulation | ✅ | QR | `diffrax` — GBM, Heston, LMM, SABR, local vol, SLV |
| Robust gradient-based calibration (Heston, SABR, SLV) | ✅ | QR, FO | `optimistix` LM + `optax`, `valax/calibration/` |
| Vol-surface models (SABR, SVI, SLV, Dupire) | ✅ | QR | `valax/surfaces/`, `valax/models/slv.py`, `valax/models/local_vol.py` |
| Neural-network container | ✅ | QR | `equinox.Module` |
| Quote-level hedge Jacobian (risk in tradable instruments; implicit diff through calibration) | ✅ | FO | `curve_quote_deltas`, `chain_adjoint` (whole chain: curves → vols → model), `quote_jacobian` (`optimistix.ImplicitAdjoint`) |
| Calibration-stability monitoring (`vmap` robustness sweep) | ✅ | FO | `jax.vmap` over perturbed quotes |
| IRRBB 6-scenario ΔEVE (BCBS 368 Standardised) | ✅ | TRE | `keyed_curve_stress` (parallel / steepener / flattener) and `KeyedScenario.curve_shocks` (short-rate shocks) on any named curve, repriced by `group_scenario_pnl` |
| HQLA / investment-portfolio pricing (fixed, floating, callable) | ✅ | TRE | `valax/instruments/bonds.py`, `valax/pricing/analytic/bonds.py`, callable-bond lattice |
| QuantLib comparison suite (14 modules) | ✅ | MRM, QR, FO | `tests/test_quantlib_comparison/` |
| Golden tests / deterministic replay for audit | ✅ | All | `tests/golden/` + `golden_manifest.json`; pure functions + integer-ordinal dates |
| Vol-surface risk factors (grid / SABR / SVI parameter risk) | ✅ / 📋 | MRM, REG | Per-surface parallel vega (`shift_surface`) and quote-space vega for rates cubes shipped; node-level vega buckets planned |
| Intraday `MarketState` build + calibration diagnostics | 🟡 | FO | [Productionisation § 5](../architecture/production.md#5-layer-2-3-build-and-calibration-workflow); not fully packaged |
| Long-lived pricing service (gRPC/REST, resident state) | 🟡 | FO, MRM | [Productionisation § 8](../architecture/production.md#8-layer-6-service-layer-deferred) — deferred |
| NII 12-month simulation engine (position walker) | 🟡 | TRE | All primitives present; orchestration layer not yet packaged |
| LCR / NSFR contractual cash-flow bucketing | 🟡 | TRE | Schedule + bucketing primitives present; run-off-factor library not shipped |
| Multi-curve market (OIS + SOFR + basis + XCCY) | ✅ | MRM, REG, TRE | `CurveGraph` (`bootstrap_curve_graph`) inside `MarketState`; per-curve shocks, sensitivities and VaR, basis as its own factor (`as_basis_layout`, `KeyedSensitivities.basis_dv01`) |
| Keyed shocks for credit / inflation / FX on `MarketState` | 📋 | MRM, REG, TRE | Curves and spots are priced from `MarketState`; not yet addressable by `KeyedScenario` — [roadmap](../roadmap.md#remaining-work) |
| SBA capital aggregation (inter-bucket correlations + curvature) | 📋 | MRM, REG | Bucket-level Δ + γ today; prescribed correlation formulas planned — component status in the [FRTB capability matrix](regulatory.md#frtb-capability-matrix) |
| Named risk factors (`IR.CURVE.USD.OIS.5Y`, …) | ✅ | MRM | `KeyedFactorLayout` names every coordinate ([Risk Factors § 1.2](../risk-factors.md#12-factor-identifier-convention)); the legacy flat path stays positional |
| IMA stressed-period selection (automated) | 📋 | REG | Caller supplies the 250-day window today |
| XVA (CVA / FVA / KVA) | 📋 | REG, QR | Needs planned CDS pricer + credit-IR joint MC |
| Streaming / incremental market-data updates | 📋 | FO | [Roadmap → Real-Time Risk](../roadmap.md#63-real-time-risk) |
| JIT warm-up / shape-bucketing service harness | 📋 | FO | Vision-tier; prerequisite for quote-response latency |
| FTP curve arithmetic (base + liquidity + credit add-ons) | 📋 | TRE | Multi-curve container shipped (an FTP curve is one more `CurveGraph` node); the add-on arithmetic layer is not built |
| Behavioural models — prepayment, deposit pass-through, non-maturity-deposit life curves | 📋 | TRE | Not on roadmap; natural JAX extension (`equinox.Module` + `jax.grad` + `optax`) |
| Credit-adjusted own-funding discount curve | 📋 | TRE | Multi-curve container shipped; own-credit spread construction not built |
| Worked neural-surrogate & deep-hedging examples | 📋 | QR | All primitives shipped; canonical notebooks planned |
| Differentiable portfolio construction | 📋 | QR | P5 frontier item — see [Vision](../vision.md) |
| Rough-volatility / neural-SDE calibration | 📋 | QR | `diffrax` supports the paths; canonical example planned |
| Differentiable risk *measures* (∂VaR/∂position, component/marginal ES, Euler allocation) | ⚑ | MRM | Primitives shipped; allocator not packaged — see [Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier) |
| Gradient-based reverse stress testing | ⚑ | MRM | Differentiable pricer + `optimistix`/`optax`; optimiser not packaged — [Market Risk § 1.3](market-risk.md#13-differentiable-risk-measures-and-reverse-stress-testing-the-frontier) |
| Gradient-based intraday optimal hedging / quoting | ⚑ | FO | Primitives shipped (`diffrax`, `optax`, `eqx.filter_grad`); product not packaged — [Front-Office § 2.2](trading-desk.md#22-gradient-based-intraday-decisions-the-pricer-and-the-control-problem-share-one-graph-frontier) |
| Strategy / signal generation (alpha) | 🔴 | FO | Separate system by design |
| Order management / execution / fills | 🔴 | FO | Separate system by design |
| Back office / operations | 🔴 | — | Wrong domain by design |

> **Reading the frontier (⚑).** These are the categorical advantages a differentiable engine has over an enumerate-and-bump stack: the *primitives* ship and are exercised, but the *packaged product* is a prototype-on-top, not an install-and-go feature. They are the honest answer to "how is this more than a fast EOD engine" — real potential, roadmap delivery. The [adoption order in § 5](#5-the-strategic-order-of-operations) sequences all of this.

---

## 5. The strategic order-of-operations

This is a different axis from the [fit ranking of § 3](#3-the-audience-map-ranked-by-fit-today): that ranks *capability fit today*, this ranks *adoption sequence*. Front-office trading can be a strong capability fit (🟢🟢) yet sit late here, because deploying it as an intraday service is Vision-tier. For a bank considering serious adoption, the sequence that maximises return and minimises risk:

1. **Model Validation first.** SR 11-7 / TRIM already require an independent challenger; VALAX is architected to be one. The QuantLib comparison suite is a shipped evidence pack. Adoption here is the lowest-friction, highest-defensibility entry point.
2. **Simultaneously: Quant Research adoption** — different budget, different buyer, provides the frontier-capability story (neural surrogates, deep hedging, differentiable calibration). The two teams cross-pollinate: research prototypes graduate into validated production without a rewrite because it is the same library.
3. **Expand into MRM** once Model Validation is a happy internal reference. Same code path, adjacent workflow, natural cross-sell to the CRO.
4. **Treasury / ALM** next. The multi-curve market it was waiting on has shipped, so the remaining gap is behavioural models and a packaged NII engine (§ 3.5).
5. **Regulatory reporting** falls out as an *output* of steps 1–4 — the FRTB PLA, VaR backtest, and CCAR waterfall are computations on top of the same pipeline.
6. **Front-office trading** — Vision-tier as a *deployed intraday service*, though the pricing/Greeks/calibration core is already strong. Pursue only after the service layer, market-data adapters, and JIT warm-up strategy described in [architecture/production.md § 8.1](../architecture/production.md#81-intraday-warm-serving-considerations-deferred) are in place — and only ever as the valuation core feeding a *separate* strategy/execution system. See [Front-Office Trading](trading-desk.md) for the honest boundary.
7. **Back office** — never. Out of scope by design.

The through-line: **do not start with trading, and do not start with regulatory reporting**. Start with the two functions that are structurally set up to benefit (Model Validation, Quant Research) and let the rest of the stack pull adoption naturally as the library becomes an internal reference.

---

## 6. Where to read next

- **Every instrument, its pricing method, vol input, and calibration** → [Instrument Pricing](instrument-pricing.md).
- **The systems around VALAX in a bank stack, and why the library makes the architectural choices it does** → [Positioning & Design Rationale](../positioning/index.md).
- **The end-to-end risk pipeline these applications sit on** → [Risk: End-to-End](../risk-overview.md).
- **The mathematical foundations** → [Models & Theory](../theory/index.md).
- **The concrete code walkthroughs** → [User Guide](../guide/tutorial-rates.md).
- **The forward direction** → [Vision](../vision.md), [Roadmap](../roadmap.md).
