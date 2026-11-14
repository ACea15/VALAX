# Treasury & ALM

*How VALAX supports bank Treasury — the function that runs the balance sheet as a portfolio, owns interest-rate risk in the **banking book**, and is measured on Net Interest Income (NII) stability across rate cycles.*

This page is the companion to [Regulatory: FRTB & CCAR](regulatory.md). Where that page maps VALAX onto the **trading-book** capital regimes (FRTB, CCAR), this page maps it onto the **banking-book** regimes owned by Treasury: **IRRBB**, **NII simulation**, **LCR / NSFR**, **FTP**, and the bank's own **HQLA / investment portfolio**.

The theme: Treasury has one of the most poorly-served analytics spaces in bank software — front-office vendors are trading-book focused, ALM vendors are expensive and opinionated, and most Treasury desks quietly run their IRRBB engine in Excel + Python. VALAX offers a JAX-native, differentiable, `vmap`-scalable alternative that fits Treasury's regulatory workflow almost primitive-for-primitive.

---

## 1. What Bank Treasury actually does

Distinct from **Market Risk Management** (which owns the trading book — see [Risk: End-to-End § 2](../risk-overview.md#2-who-consumes-risk-numbers-and-what-they-actually-want)) and from **CFO / Finance** (which owns the general ledger), Treasury manages the bank's own balance sheet. It wears five hats:

| Sub-function | Owns | Cadence | Regulatory driver |
|---|---|---|---|
| **1. Asset-Liability Management (ALM)** | Interest-rate profile of assets vs. liabilities on the banking book; NII and Economic Value of Equity (EVE) sensitivity | Daily / weekly / monthly | **IRRBB** (BCBS 368 / EBA IRRBB) |
| **2. Funding & Liquidity** | Cash-flow ladders, cash & collateral inventory, wholesale funding programme | Intraday + daily | **LCR** (30-day stress), **NSFR** (1-year stable funding) |
| **3. Funds Transfer Pricing (FTP)** | Internal curve for pricing liquidity across desks and businesses | Daily curve refresh | Internal governance |
| **4. Investment / HQLA portfolio** | The bank's own book of high-quality liquid assets (sovereigns, covered bonds, agency MBS) — held to meet LCR and to earn a spread | Daily mark, monthly rebalance | LCR eligibility rules |
| **5. Structural hedging** | Hedges of the "equity duration" arising from non-maturity deposits, current accounts, retail savings — the true home of the *deposit franchise* | Monthly | Board risk appetite |

The single overarching KPI is **Net Interest Income (NII)**: the difference between what the bank earns on assets and pays on liabilities. Treasury exists to make that number predictable across rate cycles, subject to the constraint that liquidity ratios and capital ratios never break.

---

## 2. The daily Treasury loop, mapped to VALAX

Here is Treasury's daily production loop in one picture, with each stage annotated by the VALAX primitive that serves it:

```
┌──────────────────────────────────────────────────────────────────┐
│  Daily Treasury loop                                             │
│                                                                  │
│  1. Refresh curves ────────────────────► bootstrap_curve_graph   │
│     (OIS, SOFR/€STR, funding)            → MarketState           │
│                                                                  │
│  2. Reprice banking book ─────────────► price() via group_pv    │
│     (bonds, FRNs, swaps, caps)                                   │
│                                                                  │
│  3. Compute EVE + NII sensitivity ────► keyed_ladder             │
│     (KRD ladder, DV01 per curve)        (curve_ladder, dv01)     │
│                                                                  │
│  4. Apply IRRBB 6 scenarios ──────────► keyed_curve_stress      │
│                                          (parallel / steepener / │
│                                          flattener), KeyedScenario│
│                                                                  │
│  5. ΔEVE / ΔNII per scenario ─────────► group_scenario_pnl      │
│     (vmap across scenarios)                                      │
│                                                                  │
│  6. Bucket to Basel tenor grid ───────► tenor_bucket_map,        │
│                                          aggregate on            │
│                                          curve_ladder            │
│                                                                  │
│  7. SOT check vs. Tier 1 ─────────────► arithmetic on ΔEVE       │
│                                                                  │
│  8. Cash-flow projection ─────────────► valax/dates/schedule    │
│     (LCR / NSFR bucketing)              + BucketMap              │
│                                          🟡 behavioural overlays │
│                                             are the honest gap    │
│                                                                  │
│  9. Investment portfolio mark ────────► group_pv / keyed_ladder  │
│     + duration/KRD                       over HQLA instruments   │
└──────────────────────────────────────────────────────────────────┘
```

Six of nine stages have a one-line VALAX primitive today. Stage 8 is partial. Behavioural models (prepayment, deposit pass-through) are the honest gap — addressed in [§ 8](#8-the-honest-gap-behavioural-models) below.

---

## 3. IRRBB — the killer application

IRRBB (Interest Rate Risk in the Banking Book, BCBS 368) is Treasury's largest single regulatory workflow, and it is a near-perfect match for what VALAX already ships. Every bank subject to Basel III has to run this, quarterly at minimum, and the results feed straight into SREP dialogue with the supervisor.

### 3.1 What BCBS 368 requires

Banks must compute the change in **Economic Value of Equity (ΔEVE)** under **six prescribed interest-rate scenarios**, per currency:

| # | Scenario | Description |
|---|---|---|
| 1 | Parallel up | All tenors bumped +Δr |
| 2 | Parallel down | All tenors bumped −Δr |
| 3 | Steepener | Short end down, long end up |
| 4 | Flattener | Short end up, long end down |
| 5 | Short-rate shock up | Front-end tenors bumped up |
| 6 | Short-rate shock down | Front-end tenors bumped down |

Plus **ΔNII** under parallel up/down over a rolling 12-month horizon. Then the **Supervisory Outlier Test (SOT)**: if the worst-case ΔEVE > 15% of Tier 1 capital, the supervisor investigates.

The scenario magnitudes are prescribed per currency (e.g. USD: parallel ±200 bp, short-shock ±300 bp; EUR: ±200 bp / ±250 bp). Everything else — the pricing, the aggregation, the reporting — is up to the bank.

### 3.2 The 1-to-1 mapping to VALAX primitives

| BCBS 368 requirement | VALAX primitive |
|---|---|
| Parallel up / down | `keyed_curve_stress(layout, "USD.OIS", "parallel", dr, dr)` (or `keyed_stress_scenario(layout, parallel_shift=dr)` for every curve) |
| Steepener | `keyed_steepener(layout, "USD.OIS", short_bump, long_bump)` |
| Flattener | `keyed_flattener(layout, "USD.OIS", short_bump, long_bump)` |
| Short-rate shock up / down | `KeyedScenario(curve_shocks={"USD.OIS": front_end_bumps})` — the BCBS short-rate decay shape as a per-pillar vector |
| Banking-book instruments (fixed / floating bonds, FRNs, swaps, caps, callable bonds) | [`valax/instruments/bonds.py`](../guide/fixed-income.md), [`valax/instruments/rates.py`](../guide/rates-exotics.md), [callable bonds guide](../guide/callable-bonds.md) |
| ΔEVE via re-discounting under each scenario | `group_scenario_pnl(positions, model, market, irrbb_scenarios)` — full revaluation through `price()` |
| Key-rate durations, DV01 ladder | `keyed_ladder(...).curve_ladder("USD.OIS")` — pillar-level KRDs per named curve from one autodiff pass; `basis_dv01` for the projection-vs-OIS basis |
| Bucket to Basel tenor grid | `tenor_bucket_map(pillar_times, basel_vertices)` + `aggregate` on the curve's ladder |
| PCA level/slope/curvature (industry-standard factor reduction) | `pca_jacobian` + `pushforward_sensitivities` — see [PCA Curve Shocks](../guide/pca-rates.md) |
| SOT check | `jnp.max(jnp.abs(delta_eve)) / tier1_capital` |

That is not a marketing table. Four of the six BCBS scenarios are named functions in `valax.risk`, and the two short-rate shocks are one `KeyedScenario` each. A Treasury team can stand up a production-grade IRRBB Standardised Approach calculator using primitives that exist today.

### 3.3 End-to-end IRRBB code sketch

```python
import jax.numpy as jnp
from valax.market import KeyedScenario, stack_keyed_scenarios
from valax.risk import (
    KeyedFactorLayout, keyed_curve_stress, keyed_steepener, keyed_flattener,
    group_scenario_pnl, keyed_ladder, tenor_bucket_map, aggregate,
)

OIS = "USD.OIS"
layout = KeyedFactorLayout.from_market_state(market)

# 1. BCBS 368 USD scenarios (magnitudes for USD), on the named OIS curve
DR_PARALLEL = 0.02   # ±200 bp
DR_SHORT    = 0.03   # ±300 bp

irrbb = stack_keyed_scenarios([
    keyed_curve_stress(layout, OIS, "parallel", +DR_PARALLEL, +DR_PARALLEL),
    keyed_curve_stress(layout, OIS, "parallel", -DR_PARALLEL, -DR_PARALLEL),
    keyed_steepener(layout, OIS, short_bump=-DR_SHORT, long_bump=+DR_PARALLEL),
    keyed_flattener(layout, OIS, short_bump=+DR_SHORT, long_bump=-DR_PARALLEL),
    KeyedScenario(curve_shocks={OIS: +short_shock_vector}),   # scenario 5
    KeyedScenario(curve_shocks={OIS: -short_shock_vector}),   # scenario 6
])

# 2. ΔEVE per scenario — full revaluation of the banking book through price()
delta_eve = group_scenario_pnl(positions, model, market, irrbb)

# 3. Worst-case and SOT check
worst = jnp.min(delta_eve)           # most negative EVE change
sot_ratio = jnp.abs(worst) / tier1_capital
sot_pass  = sot_ratio < 0.15         # BCBS 368 threshold

# 4. Ladder for KRD / DV01 attribution — where the sensitivity is concentrated
ladder = keyed_ladder(positions, model, market, layout)
krd = ladder.curve_ladder(OIS)       # per-pillar KRDs of the OIS curve

# 5. Bucket to Basel 19-tenor IRRBB grid for the ITS / EBA return
basel_vertices = jnp.array([
    0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35
])
tenor_bm = tenor_bucket_map(pillar_times, basel_vertices, weight="linear")
krd_basel = aggregate(tenor_bm, krd)
# krd_basel is the KRD ladder on the Basel tenor grid — ready for the ITS return.
```

Under the hood, `group_scenario_pnl` uses `jax.vmap` over scenarios and — inside each scenario — over instruments. For a 500 000-position retail banking book with six IRRBB scenarios, that is **3 000 000 repricings compiled into a single JIT call** rather than a Python loop over positions. On a single GPU this runs in seconds; on CPU it is still competitive with vendor ALM engines that need overnight batches.

### 3.4 Why this matters commercially

IRRBB is one of the highest-priced, worst-served software segments in bank technology:

- Legacy ALM vendors (QRM, Bancware, Moody's ALM, FIS Ambit) cost seven figures per year for a mid-tier bank, are inflexible to model, and are opaque to the auditor.
- Excel-based Treasury calculations dominate at Tier 2/3 banks despite being fragile and hard to defend at supervisory inspection.
- Every rate-cycle turn (2022 inflation shock; 2023 Silicon Valley Bank; ongoing supervisory focus on unrealised losses) puts IRRBB back on the CRO and CFO agenda.

VALAX gives a Treasury team a modern, transparent, auditable, differentiable alternative that a small internal analytics team can extend. That combination is genuinely rare in this space.

---

## 4. NII simulation

The second half of IRRBB — and arguably the number Treasury *actually manages* day-to-day — is **projected Net Interest Income** over the next 12 months under each rate scenario.

**What NII needs:**
1. Project contractual cashflows on the banking book out 12 months, month-by-month.
2. For floating legs, apply the scenario's projected forward rates.
3. For maturing positions, apply a **rollover assumption** (typically: reinvest at the new curve).
4. Sum interest received minus interest paid per month; aggregate to 12-month NII.

**What VALAX provides for this today:**
- Schedule generation and day-count arithmetic ([`valax/dates/schedule.py`](../guide/tutorial-rates.md), [`valax/dates/daycounts.py`](../api/dates.md)) — all JIT-compatible integer ordinals.
- Floating-rate cash-flow computation (FRN and floating swap leg pricers).
- Curve-based forward-rate projection (`DiscountCurve` interpolation → forward rates).
- Scenario propagation: a `KeyedScenario` applied by `shock` → shocked `MarketState` → NII delta, on any named curve.

**What is not yet turnkey:**
- A first-class NII projection engine that walks the banking book position-by-position and aggregates monthly. Every piece needed is there; the orchestration layer isn't packaged. This is a natural addition and does not require any architectural change — it is a few hundred lines composing existing primitives.

For a Treasury team evaluating VALAX, the pragmatic message: **IRRBB ΔEVE is a today feature; NII simulation is a natural extension that a small internal team can build on top in weeks.**

---

## 5. HQLA / investment portfolio

The bank's own portfolio of sovereigns, covered bonds, agency MBS, and supranational debt — held to meet LCR and to earn spread. This is pure fixed-income pricing, and it is a 🟢 immediate fit:

| Task | VALAX support |
|---|---|
| Bond present-value (fixed, floating, callable) | [`valax/pricing/analytic/bonds.py`](../guide/fixed-income.md), [callable bonds guide](../guide/callable-bonds.md) |
| Yield-to-maturity, duration, convexity | `jax.grad` / `jax.hessian` on the pricer |
| Key-rate durations at Treasury's chosen bucket grid | `keyed_ladder(...).curve_ladder(curve_id)` + `tenor_bucket_map` / `aggregate` |
| Portfolio-level stress (parallel, steepener, basis) | `keyed_curve_stress` / `keyed_basis_shift` + `group_scenario_pnl` (credit-spread shocks are not yet keyed — [roadmap](../roadmap.md#remaining-work)) |
| Multi-scenario ES for HQLA VaR | `expected_shortfall` on the `group_scenario_pnl` vector |
| Callable-bond OAS (option-adjusted spread) | Callable-bond lattice pricer (guide/callable-bonds) |

Treasury typically runs this in Excel + Bloomberg SDK today. Replacing that with a differentiable batch that ties into the same IRRBB engine is a natural upgrade — and it means the *HQLA duration in the ALM report* and the *HQLA duration in the investment report* are provably the same number, because they come from the same code path.

---

## 6. LCR / NSFR mechanics

LCR (Liquidity Coverage Ratio, 30-day stress) and NSFR (Net Stable Funding Ratio, 1-year stable funding) are fundamentally **bucketing exercises on projected cash flows**:

1. Project every asset's and liability's contractual cash flows onto a tenor grid.
2. Apply regulator-prescribed run-off factors (retail deposits 5–10%; wholesale 25–100%; committed lines 10–40%) or stable-funding factors (retained earnings 100%; retail deposits 90–95%; wholesale < 6M 50%).
3. Sum to get the LCR / NSFR ratio.

**What VALAX offers for this today:**
- Contractual cash-flow projection for bonds, FRNs, swaps, caps, and floors — schedule generation is in `valax/dates/schedule.py`.
- The bucketing layer (`BucketMap`, `tenor_bucket_map`) is exactly the right primitive for the LCR tenor grid — see [Risk Bucketing](../risk-bucketing.md#risk-bucketing).
- `vmap` over a 500 000-loan banking book is what JAX is built for.

**What it doesn't offer today:**
- The run-off / stable-funding **factor tables** are regulatory constants — trivial to add as data, but they're not "in the library" today.
- Behavioural cash-flow models for non-maturity deposits, prepayable mortgages, and credit-line drawdowns are **not** in the library today. These matter enormously for a realistic LCR/NSFR — see [§ 8](#8-the-honest-gap-behavioural-models) below.

**Verdict:** the mechanical bones are there for a research-grade LCR calculator; a production LCR engine needs the behavioural overlays that aren't yet built.

---

## 7. Funds Transfer Pricing (FTP)

FTP is Treasury's internal-pricing curve: every trade booked anywhere in the bank is charged (or credited) at *"base rate + liquidity add-on + credit add-on"* against the FTP curve, so that P&L attribution to individual desks reflects the true cost of funding.

**Mechanically:** it is discount-curve pricing — VALAX's bread and butter. Where the gap sits:

- **Today:** the [multi-curve market](../guide/why-multicurve.md) lets a Treasury team carry OIS, SOFR projection, an FTP / funding curve and XCCY curves side by side in one `CurveGraph` inside the `MarketState`, each shockable and reported as its own risk factor.
- **Not yet built:** an FTP-specific curve arithmetic layer (base + tenor liquidity premium + credit add-on + behavioural overlay) and a credit-adjusted own-funding curve construction. Both compose existing curve primitives; neither is packaged.

**Verdict:** the curve container and pricing mechanics are there; the FTP add-on arithmetic is the pending piece.

---

## 8. The honest gap: behavioural models

The hardest problem in Treasury is that the biggest single item on the liability side — **non-maturity deposits (current accounts, retail savings)** — has no contractual maturity, no contractual rate, and behaviour that is model-driven, not pricing-driven. Similarly, retail mortgages on the asset side have prepayment optionality that dominates their duration.

**What Treasury needs (and VALAX does not yet ship):**
- A **deposit rate pass-through model** — how retail deposit rates track central-bank rates, with lags and asymmetries.
- A **prepayment model** for retail mortgages (Andrew Davidson / OAS-style).
- **Behavioural life curves** for current accounts (typical assumption: 30% at 1 day, 70% amortising over 5 years).

But here is the interesting part — **a JAX-native, differentiable, `optax`-calibrated deposit-rate model or prepayment model is exactly the kind of thing VALAX's architecture makes easy to add**:

- `equinox.Module` for the model parameters (pass-through betas, lag structure, prepayment ramp).
- `jax.grad` for calibration to historical deposit-rate and prepayment data.
- `vmap` for scenario propagation across the entire retail book.
- Composability with the existing pricing kernel — the deposit model outputs a *behavioural cashflow schedule* that the existing curve-based discounting consumes unchanged.

The [Vision](../vision.md) doc's mention of *"neural surrogates"* and the [Roadmap](../roadmap.md)'s emphasis on differentiable modelling both apply directly here. Behavioural models are a natural home for the JAX moat — every incumbent ALM vendor treats them as opaque black boxes with proprietary calibration; a differentiable, open-source-style implementation is a genuine competitive edge.

For a Treasury team evaluating VALAX, the honest framing: **the deterministic ALM machinery is here today; the behavioural layer is the internal build project — but it is architecturally the easiest kind of thing to build on top of what exists.**

---

## 9. Coverage today vs. roadmap

The full, cross-audience coverage matrix now lives in one place — **[Applications → Coverage](index.md#4-coverage-what-ships-today-vs-roadmap)** (scan the *Business* column for `TRE`).

**The through-line:** the *pricing and sensitivity* primitives Treasury needs are all there today; the *behavioural and structural* modelling — deposit pass-through, prepayment, non-maturity-deposit life curves, FTP — is not, a genuine boundary of current scope, though architecturally the easiest kind of thing to add on top of `equinox.Module` + `jax.grad`.

---

## 10. Why the JAX foundation matters for Treasury specifically

Treasury has properties that make JAX a genuinely good fit — not just a "modern Python stack" but a specific technical advantage: autodiff KRDs and durations, `vmap` over a 500 000-position book, one `MarketState` pytree, bucketing as a matrix multiply, `optax`-calibrated behavioural models, and deterministic pure functions. Each capability and its `TRE` payoff is enumerated in [Applications § 2](index.md#2-what-makes-valax-distinctive-capability-by-capability) (deeper *why* in [Positioning § 4](../positioning/index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys)).

The net effect: the same code path that produces a trader's DV01 on a swap at 14:31 produces the firm-wide EVE waterfall under the IRRBB Severely-Adverse scenario overnight, and the CFO's morning ITS return two hours later. One library, one mental model, three regulatory audiences.

---

## 11. So — the pitch, tailored to the buyer

**To the Head of ALM:**
*A production-grade IRRBB Standardised Approach engine using primitives that exist today. The six BCBS 368 scenarios are already named functions; ΔEVE + KRD ladders are one call each; the Basel tenor bucketing is a matrix multiply. Stand up a challenger IRRBB batch in weeks, defend it against your incumbent vendor with a QuantLib comparison, and use the same code path for your CFO's ITS return and your own management dashboard.*

**To the Treasurer:**
*Your HQLA book, your IRRBB report, your NII sensitivity, and your (future) FTP curve all live in the same differentiable pricing kernel. Numbers reconcile by construction — the HQLA duration on your risk report and the HQLA duration in the CFO's return are the same array, sliced differently.*

**To the Head of Balance Sheet / Structural Hedging:**
*The deterministic ALM machinery is here today. The behavioural layer — deposit pass-through, prepayment, non-maturity deposit life curves — is where JAX gives you a genuine analytical edge over opaque vendor models: gradient-calibrated to your own history, transparent to your validators, and composable with the same pricing kernel.*

**To the CFO / CRO:**
*Treasury, MRM, and Model Validation all consume different slices of the same pipeline. The banking-book IRRBB ΔEVE, the trading-book FRTB ES, and the model-validation challenger price are three views of one code path — auditable, deterministic, and defensible in supervisory dialogue.*

---

## 12. Where to read next

- **The pipeline these workflows sit on** → [Risk: End-to-End](../risk-overview.md).
- **The trading-book companion to this page** → [Regulatory: FRTB & CCAR](regulatory.md).
- **Fixed-income pricing (bonds, FRNs, swaps, caps)** → [Fixed Income](../guide/fixed-income.md), [Rates Exotics](../guide/rates-exotics.md), [Callable / Puttable Bonds](../guide/callable-bonds.md).
- **Curve construction and multi-curve rationale** → [Yield Curves & Bootstrapping](../guide/curves.md), [Why Multi-Curve?](../guide/why-multicurve.md).
- **Scenario framework and bucketing** → [Scenarios & Shocks](../risk-scenarios.md), [PCA Curve Shocks](../guide/pca-rates.md).
- **The forward direction — behavioural models, FTP** → [Vision](../vision.md) and [Roadmap](../roadmap.md).
