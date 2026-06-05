# Risk: End-to-End

*A narrative guide to why risk exists, who consumes it, and how every number on a risk report falls out of one tiny pure-functional primitive.*

This page sits between the abstract maths in [Models & Theory § 7](theory.md#7-risk-measures) and the concrete code in the [Risk & Scenarios guide](guide/risk.md). Read it first if you have not built a bank risk system before. Read it again if you are about to extend the engine — it is the map that tells you where each new factor, instrument, or test fits in.

---

## 1. The sixty-second story

A bank holds a portfolio worth tens of billions of dollars. Every night the portfolio is **revalued** to today's market. The change in value from yesterday — the **P&L** — is split into pieces and explained to a handful of audiences:

- **Traders** want to know whether their hedges worked.
- **The bank's Market Risk Management (MRM)** team wants to know whether the day's loss was unusual *given the model* — and whether the trader is within their assigned limit.
- **The bank's Chief Risk Officer** wants to know whether the firm-wide loss distribution still fits inside the firm's risk appetite.
- **The regulator** wants to know that the bank holds enough capital to survive a 1-in-100 day, and that the model the bank uses to compute that capital is honest.
- **The internal and external auditor** wants to be able to reproduce every number on demand.

Each of these audiences asks different questions, but the answers come from the **same pipeline**. That pipeline is the subject of this page.

---

## 2. Who consumes risk numbers, and what they actually want

Real banks do not have "a risk department". They have at least five distinct functions that share data but have different timescales, different vocabularies, and different stakes.

| Audience | Cares about… | Cadence | VALAX outputs they would use |
|---|---|---|---|
| **Trader** | "What's my P&L? What's my delta? If SPX drops 2%, what do I lose?" | Intraday + EOD | Greeks ladder; stress scenarios; `pnl_attribution`; `waterfall_pnl_report`. |
| **Risk Manager (MRM)** | "Is this desk within its DV01 / vega / VaR / stress limit? Did our risk model under-predict yesterday's loss?" | EOD + weekly | `compute_ladder`; `value_at_risk`; `var_breaches`; `kupiec_pof`; `bucket_sensitivity_ladder`. |
| **Model Validator / Quant** | "Does our pricing engine agree with the market? Does our risk model explain actual P&L (FRTB PLA)?" | Monthly + ad-hoc | `explained_unexplained_vector`; `pla_spearman`; `pla_ks`; calibration residuals. |
| **Treasury / ALM** | "What's our IR sensitivity by bucket? Funding gap by tenor? CVA on the dealer book?" | Daily | Bucketed `delta_rate`; multi-curve KRDs; (planned) credit/CVA aggregation. |
| **Chief Risk Officer** | "Top-of-house VaR. Concentration risk. Stress P&L under named scenarios." | EOD + monthly board pack | Portfolio VaR/ES; named stress (`steepener`, `crash`, …); CRO dashboard built on bucketed ladders. |
| **Regulator** (Basel / national supervisor) | "Capital. Backtest pass/fail. PLA pass/fail. Bucketed sensitivities at the standard vertices." | Quarterly | Bucketed Greeks at FRTB / SIMM vertices; `basel_traffic_light`; `pla_traffic_light`. |
| **Internal & external auditor** | "Reproduce yesterday's risk numbers from yesterday's market data." | Annual + ad-hoc | The fact that every VALAX function is *pure* — same inputs ⇒ same outputs. |

A useful mental model: **traders and MRM disagree about how much risk a position has, regulators disagree about how much capital the bank should hold against it, and auditors disagree about whether the disagreement was recorded correctly.** The risk engine has to feed all three conversations consistently — that is why everything in VALAX is deterministic, autodiff-derived, and built on the same `MarketData` primitive.

---

## 3. The end-to-end pipeline

Every number in §2 is produced by the same chain of transformations. From left to right:

```
   Raw market quotes
         │
         ▼
  ┌────────────────┐    bootstrap / fit
  │  Curves &      │◄─────────────────────  (valax/curves, valax/surfaces,
  │  surfaces      │                         valax/calibration)
  └────────┬───────┘
           │
           ▼
  ┌────────────────┐
  │  MarketData    │   ←—— one pytree carrying every risk factor's level
  │  (pytree)      │       (valax/market/data.py)
  └────────┬───────┘
           │
           ▼
  ┌────────────────┐
  │  Pricing fn    │   V(instrument, MarketData) -> price
  │  V(·)          │       (valax/pricing/*)
  └────────┬───────┘
           │   one reverse-mode pass
           ▼
  ┌────────────────┐
  │ Sensitivity    │   δ_x = ∇_x V       (autodiff, no bumps)
  │ ladder         │   γ_x = ∇²_x V
  └────────┬───────┘       (valax/risk/ladders.py)
           │
           ├──────────────► bucket via A or J ──► bucketed Δ, γ ──► limits / capital
           │                (valax/risk/bucketing.py)
           ▼
  ┌────────────────┐                                  ┌────────────────┐
  │ Scenarios      │ ── apply_scenario ──► many ──►  │   P&L vector  │
  │ (historical /  │                       reprices   │  (n_scenarios)│
  │  parametric /  │   or                              └───────┬───────┘
  │  stress)       │ ── waterfall rungs ──►  RTPL ──►          │
  └────────────────┘  (cheap arithmetic)                       │
           │                                                    ▼
           │                                          ┌────────────────┐
           │                                          │   VaR  / ES   │  ←—— sample stats
           │                                          │   on PnL vec  │       (theory §7.1-7.2)
           │                                          └───────┬───────┘
           │                                                    │
           ▼                                                    ▼
   Backtest:   var_breaches → kupiec_pof → traffic light
   PLA:        rtpl vs hpl → spearman + ks → FRTB zone
   Capital:    bucketed Δ at FRTB vertices → SBA formula (planned)
```

Three things are worth stressing about this picture:

1. **There is only one pricing function.** Everything else (Greeks, ladders, scenarios, VaR, backtests) is a *derived* quantity. If `V(market)` is wrong, every downstream number is wrong; if `V(market)` is right and differentiable, every downstream number is right by construction.
2. **There is only one P&L vector primitive.** VaR is a quantile of it. ES is a tail-mean of it. Backtests count its breaches. The FRTB PLA test compares two of them (HPL vs RTPL). All four are sample statistics over the same array shape.
3. **Bucketing is a coordinate change, not a separate computation.** The same sensitivity that lives on `MarketData.discount_curve.discount_factors` becomes a "DV01 at 5Y" or a "first PCA score" purely by multiplying by an aggregation matrix or a Jacobian.

---

## 4. The five questions risk has to answer

A working risk system answers five questions. The vocabulary is sometimes intimidating; the underlying ideas are not.

### 4.1 What is my book worth right now?

This is just **pricing**: evaluate every position under today's market and sum.

```
V_portfolio(market) = Σ V_i(instrument_i, market)
```

In VALAX, every pricing function in `valax/pricing/` has the shape `V(instrument, market) → price`, with `market` carrying curves, spots, vols, and any other state. Today's `V` minus yesterday's `V` (with **today's portfolio**, not yesterday's) is the **hypothetical P&L (HPL)** — the cleanest definition of P&L because it strips out intraday trading and new deals. (See [Theory § 7.5](theory.md#75-pl-vectors-hypothetical-risk-theoretical-actual) for HPL vs APL vs RTPL.)

### 4.2 How does my book move when markets move?

This is **sensitivities** — the Greeks. Trader-level Greeks (delta, gamma, vega) and curve-level Greeks (DV01, key-rate durations, vol-of-vol convexities) are all just partial derivatives of `V`:

$$
\delta_x = \frac{\partial V}{\partial x}, \qquad \gamma_{xy} = \frac{\partial^2 V}{\partial x \partial y}
$$

In VALAX they come from `jax.grad` and `jax.hessian` on the pricing function — one reverse-mode pass per order — and are collected into a `SensitivityLadder` by [`compute_ladder`](guide/risk.md#sensitivity-ladders-and-waterfall-pl). No bump-and-reprice. No finite differences. **The "Greeks" a trader sees and the "key-rate DV01" a regulator wants are literally the same array, sliced and bucketed differently** — see [Theory § 6](theory.md#6-greeks-and-automatic-differentiation) for the autodiff derivation and [Theory § 7.4](theory.md#74-sensitivity-ladders) for ladders.

### 4.3 What's a bad day for my book?

This is **VaR and Expected Shortfall**. A "bad day" is the left tail of the distribution of $\Delta V$:

- $\text{VaR}_\alpha$ = the loss exceeded with probability $1-\alpha$.
- $\text{ES}_\alpha$ = the average loss conditional on exceeding $\text{VaR}_\alpha$.

The distribution comes from a **P&L vector**: full-revaluation under many scenarios (historical or Monte Carlo) gives **HPL**; passing the same scenarios through the second-order ladder gives **RTPL** much faster. Both vectors plug straight into [`value_at_risk`](api/risk.md) and [`expected_shortfall`](api/risk.md). The conceptual derivation lives in [Theory § 7.1](theory.md#71-value-at-risk-var) and [§ 7.2](theory.md#72-expected-shortfall-cvar); the workflow code is in the [guide](guide/risk.md#full-revaluation-var).

### 4.4 Is the risk model honest?

This is **backtesting**. A model that consistently under-predicts losses is dangerous; a model that wildly over-predicts is wasteful of capital. Two regulator-mandated tests, both shipped in VALAX:

- **VaR backtest** — count how often realised P&L breaches the model's forecast, and apply the Basel traffic-light zoning (`basel_traffic_light`) plus the Kupiec POF and Christoffersen independence tests for finer power. ([Theory § 7.6](theory.md#76-var-backtesting).)
- **FRTB P&L Attribution test (PLA)** — compare *risk-theoretical* P&L (from the ladder) to *hypothetical* P&L (from full revaluation) over 250 days. Use Spearman rank correlation for monotonic agreement and a Kolmogorov-Smirnov test for distributional agreement, then read the BCBS d558 zone with `pla_traffic_light`. ([Theory § 7.7](theory.md#77-frtb-pl-attribution-test).)

When PLA fails red, the desk loses its right to use the **Internal Models Approach** for capital and has to fall back to the (usually more punitive) **Standardised Approach**. This is the largest single financial incentive in the entire framework: keep the ladder sharp.

### 4.5 How much capital does the bank need?

This is where every prior question becomes a real dollar number that hits the bank's balance sheet. Basel III.1 / FRTB defines two parallel methodologies:

| Approach | What it consumes | Where in VALAX |
|---|---|---|
| **Standardised Approach (SBA)** | Bucketed sensitivities at *standard vertices* (FRTB IR tenor grid, equity sectors, …), aggregated via prescribed correlations into a capital number. | `compute_ladder` → `tenor_bucket_map` / `equal_weight_bucket_map` → `bucket_sensitivity_ladder` → (planned) SBA aggregation formulas. |
| **Internal Models Approach (IMA)** | Expected Shortfall on a stressed-period scenario set, plus a stress-add-on for non-modellable risk factors. Subject to passing PLA + VaR backtest. | `risk_theoretical_pnl_vector` / `hypothetical_pnl_vector` → `expected_shortfall` → PLA check → (planned) capital scaling. |

The crucial point: **both methodologies start from the same sensitivities and the same P&L vectors that the trader and MRM already use**. The regulator's capital number, the trader's Greek, and the risk officer's VaR are not three separate systems — they are three slices of one pipeline. (Bucketing is what aligns the slices; see [§ 5](#5-the-standardization-layer-why-bucketing-matters) below.)

---

## 5. The standardization layer: why bucketing matters

A regulator looking at fifty banks needs to compare them. A clearing house computing margin for one trade between two banks needs both sides to agree on what they exchange. Neither is possible if each bank speaks its own internal risk language — one quotes DV01 on 30 pillars, another on 12, a third on PCA scores.

**Bucketing is the bridge.** It collapses each bank's high-dimensional, internally-chosen factor space onto a fixed external grid:

| Audience | Standard buckets |
|---|---|
| FRTB SBA capital | 10 IR tenor vertices `{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, 11 equity sector buckets, 18 credit buckets, … |
| ISDA SIMM (bilateral margin) | A different but similar tenor grid and sector taxonomy. |
| Internal limits | Whatever the bank chooses, often coarser (e.g. "short / belly / wings"). |
| PCA factor risk | Three to five data-driven components per curve. |

VALAX exposes both transformation flavours — **linear** (`aggregate`, `BucketMap`) for FRTB / SIMM-style summation, and **Jacobian** (`pushforward_sensitivities`, `pca_jacobian`, `level_slope_curvature_jacobian`) for smooth reparameterizations like PCA or SVI/SABR parameter Greeks. The mathematics is in [Theory § 7.8](theory.md#78-risk-bucketing-linear-and-jacobian-transformations); the practical recipes are in the [guide](guide/risk.md#risk-bucketing); the full registry of which factors live where is in [Risk Factors § 7](risk-factors.md#7-bucketing).

A useful aphorism for engineers new to the area: **internal risk models live in autodiff-natural raw-factor space; the outside world lives in bucket space. The job of the bucketing layer is to be the FX desk between the two.**

---

## 6. Why VALAX is built the way it is

Every design choice in the library is in service of the pipeline above. The full argument — what is actually hard about quant systems at scale and which architectural choices target which pain points — lives in the dedicated [Design Rationale](design-rationale.md) page. The short version, in the context of the risk pipeline:

| Choice | Pipeline payoff |
|---|---|
| **`MarketData` is one pytree** | A single object carries every risk factor's level. `jax.grad(V)(market)` returns sensitivities to *all of them* in one call — no plumbing per factor. |
| **All pricing functions are pure** (`(instrument, market) → price`) | Determinism: an auditor can replay yesterday's books to the bit. Composability: every downstream layer is just function composition. |
| **`equinox.Module` pytrees** | Curves, surfaces, models, scenarios, ladders, bucket maps — every type is a pytree, so `jit` / `vmap` / `grad` work uniformly across the stack. |
| **Autodiff Greeks** | First-order ladder: one reverse pass, ~3× one pricing call. Hessian: one forward-over-reverse pass, ~3N× one pricing call. *Same code path for vanilla and exotic instruments.* No 50 000-line bump framework to maintain. |
| **`vmap` over scenarios** | A 10 000-scenario VaR run is one JIT-compiled call, not a Python loop. Bank-scale portfolios become feasible on a single GPU. |
| **Bucketing as a matrix multiply** | Trader view, MRM view, and regulator view are all the same ladder × a different `A` or `J`. New views are five-line additions. |
| **Dates as integer ordinals** | The calibration and risk loops fit inside `jit` end-to-end — no Python `datetime` round-trips on the hot path. |

The combined effect: the *same* code path handles "compute a delta for a trader at 14:31" and "compute the firm-wide 99% ES on 250 stressed-period scenarios overnight". One library, one mental model, two orders of magnitude difference in scale and zero difference in semantics.

---

## 7. The state of the risk section, and what's next

What is in the library **today** (✅) and what is signposted for the next iterations (🚧 / 📋), grouped by pipeline stage:

| Stage | Status | Components |
|---|---|---|
| Market data containers | ✅ | `MarketData`, `MarketScenario`, `ScenarioSet`, `MultiCurveSet`, `SurvivalCurve`, `InflationCurve`, vol surfaces. |
| Pricing engines | ✅ (broad) / 📋 (XVA, CDS) | Equity, FX, rates, inflation, callable / puttable bonds, credit CDS pricer planned. |
| Sensitivity ladders | ✅ | First- and second-order ladder, ten-rung waterfall, autodiff throughout. |
| Scenario generation | ✅ | Historical, parametric Gaussian / t, named stress (parallel, steepener, butterfly, …). |
| Shock primitives | ✅ | Single-curve, multi-curve (basis), credit hazard. 📋 Vol surface, inflation, FX vol. |
| P&L vectors | ✅ | HPL (`hypothetical_pnl_vector`), RTPL (`risk_theoretical_pnl_vector`), unexplained. |
| VaR / ES | ✅ | Full-revaluation and parametric delta-normal. |
| Backtesting | ✅ | Breach detection, Kupiec POF, Christoffersen independence, Basel traffic light. |
| FRTB PLA | ✅ | Spearman + KS, BCBS d558 zoning. |
| Bucketing | ✅ | Linear `BucketMap` ops, Jacobian reparameterization, FRTB tenor / sector / PCA / L-S-C / autodiff Jacobian builders. |
| **SBA capital aggregation** | 📋 | Bucket-level Δ + γ are computed today; the prescribed inter-bucket correlation matrices and curvature add-ons are next. |
| **XVA (CVA / FVA / KVA)** | 📋 | Needs the planned CDS pricer + a credit-IR joint Monte Carlo path generator. |
| **IMA stress-period selection** | 📋 | The 250-day historical window is currently chosen by the caller; automated stress-period identification will land alongside the SBA work. |
| **Named factor registry** | 📋 | Factor IDs (`IR.OIS.USD.5Y`, …) are documented but the engine still uses positional layouts; the migration is roadmap item 7. |

The full priority order, with dependencies, is in [Risk Factors § 4 Roadmap](risk-factors.md#4-roadmap) and the project-wide [Roadmap](roadmap.md). The shortest critical path to a desk-usable bank-style risk system is: **(a) multi-curve `MarketData` → (b) vol-surface risk factors → (c) FRTB SBA capital aggregation**, which between them close roughly 90 % of the gap to a single-desk production deployment.

---

## 8. Where to read next

Different audiences will follow different paths from here:

- **Quants / pricing developers** → [Models & Theory](theory.md) for the math, [Risk Factors](risk-factors.md) for what's in the engine, [API: Risk](api/risk.md) for signatures.
- **Risk system developers / engineers** → [Risk & Scenarios guide](guide/risk.md) for the workflow, [Risk Factors § 5 Instrument → Factor matrix](risk-factors.md#5-instrument-factor-matrix) for coverage, [Roadmap](roadmap.md) for the priorities.
- **Front-office (trader / structurer)** → [Greeks guide](guide/greeks.md) and the *Sensitivity Ladders* section of the [Risk guide](guide/risk.md#sensitivity-ladders-and-waterfall-pl).
- **Regulators / validators** → [Theory § 7.6–7.8](theory.md#76-var-backtesting) for the backtesting + PLA + bucketing maths, [Risk Factors](risk-factors.md) for the audit-trail of what is and is not modelled.

---

## 9. Glossary

A pocket dictionary of the terms used above and across the risk section.

| Term | Plain-English meaning | Where in VALAX |
|---|---|---|
| **APL** | *Actual P&L.* The end-of-day P&L from the books — includes trading, fees, valuation adjustments. Not produced by the engine; computed by Finance. | (input, not generated) |
| **HPL** | *Hypothetical P&L.* Today's portfolio repriced under tomorrow's market — strips out intraday flow. | `hypothetical_pnl_vector` |
| **RTPL** | *Risk-theoretical P&L.* The risk engine's prediction from the ladder. | `risk_theoretical_pnl_vector` |
| **Greek** | Generic name for a partial derivative of price w.r.t. a market input (delta, gamma, vega, rho, …). | `compute_ladder`, `jax.grad(V)` |
| **DV01** | *Dollar value of 01.* P&L change for +1 bp parallel rate move. | `ladder.delta_rate` summed |
| **KRD** | *Key-rate duration.* DV01 sliced per tenor pillar. | `ladder.delta_rate` |
| **VaR** | *Value at Risk.* Loss exceeded with probability $1-\alpha$ over a horizon. | `value_at_risk` |
| **ES / CVaR** | *Expected Shortfall.* Mean loss conditional on exceeding VaR. | `expected_shortfall` |
| **Backtest** | Comparing forecast VaR to realised P&L over a window. | `var_breaches`, `kupiec_pof`, `christoffersen_*`, `basel_traffic_light` |
| **PLA** | *P&L Attribution* test (FRTB). RTPL vs HPL agreement check. | `pla_spearman`, `pla_ks`, `pla_traffic_light` |
| **FRTB** | Basel III.1 *Fundamental Review of the Trading Book.* The current market-risk capital regime. | Backtesting, PLA, SBA, IMA |
| **SBA** | *Standardised Approach* (FRTB). Capital from bucketed sensitivities + prescribed correlations. | Bucketing + (planned) aggregation |
| **IMA** | *Internal Models Approach* (FRTB). Capital from the bank's own ES model, conditional on backtest + PLA pass. | ES on RTPL/HPL + PLA |
| **SIMM** | *Standard Initial Margin Model* (ISDA). Bilateral OTC margin from bucketed sensitivities. | Bucketing (different vertices) |
| **MRM** | *Market Risk Management.* The internal bank function that owns limits, VaR, and capital reporting. | (audience, not generated) |
| **CSA** | *Credit Support Annex.* Bilateral collateral agreement; drives which discount curve a swap uses. | `DiscountCurve` (OIS) |
| **Survival probability** | $S(t) = \Pr(\tau > t)$ — probability the issuer is still alive at $t$. | `SurvivalCurve` |
| **Hazard rate** | Instantaneous default intensity; $S(t) = \exp(-\int_0^t h)$. | `hazard_rate`, `piecewise_hazards` |
| **Recovery rate** | Fraction of notional recovered on default (≈40% senior unsecured). | (parameter) |
| **Basis** | Spread between two related curves (3M vs OIS, SOFR vs Fed Funds, FX cross-currency). | `bump_forward_curve`, `parallel_basis_shift` |
| **Bucketing** | Aggregating raw-factor sensitivities into a coarser standardised grid. | `BucketMap`, `bucket_sensitivity_ladder` |
| **PCA factors** | Principal components of factor returns; first three on a yield curve are level / slope / curvature. | `pca_jacobian` |
| **Ladder** | A bucketed vector of sensitivities across a risk-factor dimension. | `SensitivityLadder` |
| **Waterfall** | The rung-by-rung P&L decomposition built from a ladder. | `waterfall_pnl`, `waterfall_pnl_report` |

---

*Last updated alongside the bucketing release. As the risk section grows, this page is the index — every new component should appear in §3 (pipeline), §5 (bucketing), or §7 (status table) before it lands in code.*
