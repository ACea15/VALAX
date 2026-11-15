# Positioning & Design Rationale

*Where VALAX sits among the systems around a pricing-and-risk engine at a bank or a fund, what is actually hard about building that layer, and which architectural choices VALAX makes to attack it — plus an honest account of what it deliberately does not try to be or fix.*

This page sits beside [Vision](../vision.md) (the destination) and the [Roadmap](../roadmap.md) (the order in which features land). It looks both *outward* — the systems around VALAX — and *inward* — why the engine is built the way it is. It is deliberately opinionated. For who uses VALAX and why, see [Applications](../applications/index.md); for how VALAX compares with QuantLib, vendor platforms, bank libraries and prop / buy-side stacks, see [Peer Tools & Market Segments](peers.md).

---

## 1. The thesis

A trading floor is not one piece of software. It is **twenty to forty distinct systems** wired together by contracts on identifiers, timestamps, calendars and conventions. Some are commodity (the general ledger, collateral management); some are competitive moats every firm builds in-house (alpha signals, execution algos). In the middle sits a small cluster that is neither commodity nor signal-bearing but absolutely load-bearing: **the pricing-and-risk engine**. It computes the fair price, the Greek, the VaR number, the CVA charge and the FRTB sensitivity, and every conversation a desk has with risk, finance, the regulator and its own P&L attribution flows through it. **VALAX is exactly this cluster** — not a market-data vendor, an OMS, an alpha library or a settlement platform, but the differentiable kernel those systems feed or consume.

The hardest part of building that cluster is **not** the maths in any individual component; the maths is mostly solved. The hardest part is **keeping ~500 mutually-aware components consistent over twenty years while every one of them is being changed by a different team under regulatory pressure**. VALAX's architecture is designed to minimise the *coordination tax* this imposes — the half of a bank quant's day spent reconciling identifiers, calendars, sign conventions, snapshot times and bump conventions across systems. Every design choice in the library serves that consistency, so that a single JIT-compiled function call serves the front-office quote, the EOD risk batch, the XVA inner loop and the regulator's IMA backtest from the same line of code.

---

## 2. The bank / hedge-fund stack at a glance

A useful mental picture is six concentric rings around the core book of trades. VALAX lives in exactly one ring.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  6. Regulatory & external                                       │
   │     Basel/FRTB reporting · SEC/CFTC filings · trade repositories│
   │  ┌───────────────────────────────────────────────────────────┐  │
   │  │  5. Finance & control                                     │  │
   │  │     General ledger · P&L attribution · product control    │  │
   │  │  ┌─────────────────────────────────────────────────────┐  │  │
   │  │  │  4. Risk & treasury                                 │  │  │
   │  │  │     Market risk · credit/XVA · liquidity · capital  │  │  │
   │  │  │  ┌───────────────────────────────────────────────┐  │  │  │
   │  │  │  │  3. Pricing & analytics  ◄── VALAX lives HERE │  │  │  │
   │  │  │  │     Curves · surfaces · pricers · Greeks ·    │  │  │  │
   │  │  │  │     scenarios · VaR · XVA inner loop          │  │  │  │
   │  │  │  │  ┌─────────────────────────────────────────┐  │  │  │  │
   │  │  │  │  │  2. Trade lifecycle                     │  │  │  │  │
   │  │  │  │  │     Booking · golden source · STP ·     │  │  │  │  │
   │  │  │  │  │     confirmations · settlement          │  │  │  │  │
   │  │  │  │  │  ┌───────────────────────────────────┐  │  │  │  │  │
   │  │  │  │  │  │  1. Front office & venues         │  │  │  │  │  │
   │  │  │  │  │  │     OMS · EMS · algos · venues ·  │  │  │  │  │  │
   │  │  │  │  │  │     market data feeds             │  │  │  │  │  │
   │  │  │  │  │  └───────────────────────────────────┘  │  │  │  │  │
   │  │  │  │  └─────────────────────────────────────────┘  │  │  │  │
   │  │  │  └───────────────────────────────────────────────┘  │  │  │
   │  │  └─────────────────────────────────────────────────────┘  │  │
   │  └───────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────┘
                    Reference data + market data flow ↕
                    Trade IDs + position state flow ↕
```

VALAX is the **third ring**. It is fed by rings 1–2 (trades, market data) and consumed by rings 4–6 (risk, capital, finance, regulators). Every ring above and below is either commodity infrastructure a firm already has, or a moat the firm builds itself. The third ring is where most firms pay the highest *consistency tax* (§ 3) — and where a JAX-native, differentiable, deterministic engine has the most leverage. Every consumer in rings 4–6, from the trader to the auditor, needs **numerically identical** answers from the **same** pricing function; the cost of the opposite world, where the trader's, the risk system's and the regulator's pricers have drifted apart, is the largest hidden line item in a tier-1 bank's quant budget.

---

## 3. What actually kills bank programmes, ranked

When a quant systems programme overruns by years or fails to land, the post-mortem almost never reads "the Heston calibrator wasn't accurate enough." It almost always reads one of the things below, ordered by how often they sink a programme, not by how intellectually deep they are.

### 3.1 Data and identifier consistency — the silent killer

Not "data infrastructure" (storage, throughput — those are solved with money). **Semantic consistency across the firm.**

Pick a single trade: a 5Y USD payer swap with JPM. Now ask, for each system that touches it:

- What's its ID in the front-office booking system?
- In the firm's golden-source trade store?
- In the collateral / margin system?
- In the regulatory reporting feed?
- In the CVA engine's netting-set definition?
- In the general ledger?

The answer is typically six different IDs, mapped through six different reference tables, owned by six different teams, each with its own change-control board. Multiply by tens of millions of live trades and two decades of history. Now add: reference data on issuers, calendars per market, day-count conventions per leg, fixing sources, corporate-action handling, bitemporal versioning of every overrideable value.

The pricing maths for that swap is 200 lines of code. The identifier plumbing is a 30-person team for a decade.

### 3.2 Model governance and regulatory permission — the second killer

A bank cannot deploy a new pricer the way a startup ships a service. Under SR 11-7 (US), SS1/23 (UK), TRIM (EU), and FRTB IMA, every model carries an inventory entry, a tier classification, independent validation by a team that did not build it, sign-off by a model risk committee, documented limitations and approved use perimeter, annual re-validation, and re-validation on every material change. IMA capital approval for a new product class is typically an 18–36 month project.

You can build a state-of-the-art calibrator in a week. Getting it into production for capital use takes years. That asymmetry is the single biggest constraint on a bank's ability to ship quant code.

### 3.3 Joint factor consistency, XVA, and what-if at trader speed

Risk *is* hard, but in a specific way. The univariate maths is mostly known. What's hard at scale is:

- **Joint factor dynamics across asset classes.** Equity surfaces, IR curves, credit curves, FX surfaces all evolving under *one* consistent joint distribution for CVA. A 10,000-factor joint covariance that is PSD, calibrated to vanillas, and stable under shrinkage is a real research problem.
- **XVA** (CVA / FVA / KVA / MVA / ColVA). Each is a portfolio-level Monte Carlo on the *entire* derivatives book under joint market+credit scenarios. Computationally savage; the governance of which CSAs, netting sets, funding curves, recoveries, and wrong-way-risk correlations to use is harder than the computation.
- **What-if at trader speed.** Pre-trade: "if I add this hedge, what does it do to my desk's IMA capital?" needs full-revaluation risk on a *hypothetical* book in seconds.

Hard, but a known playbook. Bounded scope.

### 3.4 Pricing engines

Lower than people expect. 90% of the trading book is vanilla products with mature, 30-year-old pricers. The hard 10% (autocallables, Bermudans, callable range accruals, SLV calibration, pathwise Greeks for barriers) is known research with bounded scope.

Where it remains hard:

- Calibration robustness across regimes (Heston blowing up in crisis vol).
- Greek stability day-over-day for hedge ratios that don't jitter.
- Cross-desk model consistency (swaption Bachelier vol reconcilable with cap Black vol).

All intellectually tractable. Hire the right four PhDs and ship.

### 3.5 Execution algorithms

Technically intense (microstructure, queue position, adverse selection, microsecond latency) but **organisationally the simplest** thing on the list — localised to one desk, independently testable via historical replay, self-policing economics (bad algos lose money visibly and are killed in days). The smartest people on a bank's tech floor work here, but the *programme management* is light compared to risk, capital, or finance integration.

### 3.6 The meta-point

The pattern is unmistakeable: **the hard things are not the components, they are the contracts between the components.** Every arrow between two systems is a bilateral agreement about identifiers, timestamps, calendars, sign conventions, currency precision, settlement, fixings, and corporate-action handling. There are typically 200–500 such contracts in a tier-1 bank. Each is owned by a different team with a different release cadence. The cost of changing any one of them grows faster than linearly in the number of others it touches.

That web of contracts is the moat that protects incumbent bank software. It is also the cost that grinds new programmes to a halt.

---

## 4. The seven choices VALAX commits to, and what each one buys

The architecture is not "JAX because JAX is cool." Each choice is a targeted response to one of the pain points in § 3. The table below is the load-bearing claim of this page.

| Architectural choice | Pain it targets | What it buys you |
|---|---|---|
| **Pure functional, no mutable state.** Every pricing call is `price(instrument, model, market) → PricingResult` with no side effects. | § 3.1 (reproducibility / lineage / audit) and § 3.2 (model governance) | Determinism by construction. Yesterday's risk number can be replayed bit-exactly from yesterday's inputs. The auditor's reconciliation question becomes trivial. The model risk team's "reproduce the validation pack" request becomes a function call. |
| **Single `MarketState` pytree.** One object carries every curve of the multi-curve graph, every surface, spot and dividend, each named by identifier. | § 3.1 (cross-system contracts) | Replaces dozens of bilateral "which version of which curve at which timestamp" contracts with a single typed object. Any downstream consumer (pricer, risk, capital, what-if) sees the *same* market state by construction. |
| **Autodiff Greeks via `jax.grad`.** No bump-and-reprice framework. | § 3.1 (drift between pricing and risk) and § 3.3 (XVA Greek consistency) | The single largest source of front-office vs middle-office disagreement is bump frameworks drifting out of sync with the pricer they shadow. With autodiff there is no separate framework — the pricer *is* the Greek engine. The trader's delta and the regulator's DV01 come from the same line of code. |
| **One JIT-compiled code path for the trader's hot price and the overnight risk run.** | § 3.3 (what-if at trader speed) | Eliminates the "research vs production" duality. A pre-trade what-if uses the same compiled pricing function as the EOD risk batch; the only difference is the input book. Whatever speed-up you buy on the batch flows straight into the trader's quote latency. |
| **Integer ordinals + static day-count conventions in JAX-traced code.** | § 3.1 (calendar / timestamp ambiguity) | Calendar bugs are the #2 source of cross-system inconsistency after identifier bugs. By forcing dates into JIT-safe integer ordinals with day-count conventions as static module fields, every pricing function declares its date semantics structurally — and the type system catches mismatches before they reach production. |
| **`equinox.Module` pytrees for every domain type** — curves, surfaces, models, scenarios, sensitivities, bucket maps. | § 3.3 (joint factor consistency) | Uniform composability. `jit` / `vmap` / `grad` work without ceremony across the entire stack. Risk types and pricing types and calibration types all participate in the same pytree machinery, so a joint scenario across asset classes is one `KeyedScenario` pytree, not a hand-rolled struct per asset class. |
| **GPU-first `vmap` over scenarios and instruments.** | § 3.3 (XVA, full-revaluation what-if) | A 10,000-scenario × 10,000-instrument VaR is one JIT-compiled call, not a Python loop. The CVA nested-MC inner loop that takes a bank's grid overnight collapses to a daytime run on a single GPU. This is the single largest *quantitative* advantage VALAX has over QuantLib-style stacks. |

The crucial property is that these choices **compose multiplicatively, not additively**. Determinism + autodiff + single JIT path means an audit trail of a stress scenario is identical to a trader's pre-trade what-if is identical to the regulator's IMA backtest. They are not three different systems with reconciliation jobs between them — they are three call sites of the same function. How each choice pays off per business line is set out in [Applications § 2](../applications/index.md#2-what-makes-valax-distinctive-capability-by-capability); the rules contributors follow to keep these properties are in [Architecture → Invariants for contributors](../architecture/overview.md#invariants-for-contributors).

### 4.1 The compound effect

In a traditional bank stack:

```
Trader's price        =  pricer_v3.cpp + cache_v7 + fixings_db_at_now
EOD risk number       =  pricer_v3.cpp + bump_framework_v12 + curves_at_4pm
Regulatory capital    =  pricer_v3_archived.cpp + frtb_engine + curves_at_eod_t-1
What-if               =  separate "what-if" service with its own model loader
Audit replay          =  best-effort reconstruction from logs
```

Each row is a different system, with a different snapshot mechanism, a different cache, and a different reason it might disagree with the others. Most of the firm's quant headcount goes into making them agree. The reconciliation cost dominates the build cost over the lifetime of the system.

In VALAX, every row is *the same call*:

```python
# Trader hot price
pv = price(instrument, model, market_now).price

# EOD risk: every curve, surface and asset in one set of sensitivities
sens = keyed_sensitivities(book, model, market_eod, second_order=True)

# Regulatory capital: the same sensitivities in regulatory coordinates
frtb_krd = aggregate(frtb_tenor_map, sens.key_rate_dv01("USD.OIS"))

# What-if
pv_after = price(instrument, model, shock(market_now, hypothetical_shock)).price

# Audit replay: recalibrate from the stored quotes of that close
market_then = calibrate_market(chain, quotes_2024_03_14, fixed_2024_03_14).market
pv_then = price(instrument, model, market_then).price
```

Same `price`. Same `MarketState` type. Same JIT-compiled binary. The reconciliation problem doesn't go away in the real world — there are still booking systems and golden sources to reconcile — but the *internal* reconciliation surface area is reduced to a single boundary: did the right quotes arrive at the input? Everything downstream is mechanical.

This is the architectural payoff in one sentence: **VALAX is designed so that the consistency tax inside the engine is zero, leaving the firm free to spend its quant budget on the consistency tax at its boundaries — where it actually has to be paid.** The tests that defend this promise are described in [Validation Strategy](../architecture/validation-strategy.md).

---

## 5. Integration map — what couples to VALAX

VALAX is a deliberately narrow tool. The tables below list the systems that surround it in a real deployment — what VALAX expects to receive from them and what it returns.

### 5.1 Upstream — what VALAX consumes

| System category | Examples (illustrative) | What VALAX needs from it |
|---|---|---|
| **Market data vendors / feeds** | Bloomberg, Refinitiv, ICE, exchange direct feeds, Pico/Exegy for low-latency | Raw quotes (deposits, FRAs, swaps, swaptions, vol grids, FX forwards, CDS spreads) keyed by snapshot timestamp. Roadmap P4.2. |
| **Reference data** | Bloomberg OpenSymbology, GLEIF (LEI), internal product master | Issuer hierarchies, calendar definitions, day-count tables, holiday lists, instrument static. |
| **Golden source / trade store** | Murex, Calypso, Summit, internal trade DBs (Athena/SecDB/Quartz) | The book of positions in a form VALAX can map onto its instrument pytrees. |
| **Booking / OMS** | Murex front office, Calypso, FlexTrade, internal | Trade inception events, lifecycle events (fixings, resets, novations). |
| **Counterparty / collateral** | TriOptima, Acadia, internal CSA store | Netting-set definitions, CSA terms, collateral balances — needed by the XVA workstream (roadmap P3.1). |

### 5.2 Downstream — what VALAX feeds

| System category | Examples (illustrative) | What VALAX hands them |
|---|---|---|
| **Risk aggregation / dashboards** | OpenGamma, internal CRO dashboards, Tableau/PowerBI on a P&L mart | Portfolio Greeks, VaR/ES, scenario P&L vectors, FRTB sensitivities. |
| **General ledger / product control** | SAP, Oracle Financials, internal G/L | Marks, accrued cashflows, official P&L by book. |
| **Regulatory reporting** | Axiom, AxiomSL, Wolters Kluwer OneSumX, internal | Bucketed sensitivities at FRTB / SIMM vertices, backtest results, PLA outputs. |
| **XVA / capital engines** | Internal CVA engines, Numerix CrossAsset XVA, Murex MX.3 XVA | Path-wise revaluations consumed by the outer simulation. *Long-term, VALAX-on-GPU is the XVA engine itself (see [Vision § XVA](../vision.md#xva-and-counterparty-risk)); in the medium term it can also feed an existing one.* |
| **Front-office UIs / quote engines** | RFQ blotters, AMM engines, internal trader cockpits | Sub-millisecond pricing and Greeks via the (planned) gRPC service. |

---

## 6. Scope — what VALAX is not, and what it does not fix

Two kinds of thing sit outside VALAX: **adjacent systems** it couples to but deliberately does not try to be, and **problems** that are real and important but outside the reach of any pricing library, this one included. Trying to absorb either is out of scope; coupling cleanly to them is the point.

| Out of scope | Why |
|---|---|
| ***Adjacent systems*** | |
| **Order management (OMS) and execution (EMS)** | Order routing, venue connectivity, smart-order-routing logic, exchange certifications. VALAX has no view on *where* a trade is placed; it is the pricing-and-risk layer underneath such systems, not a replacement. |
| **Execution algorithms and market-making strategy** | VWAP, TWAP, POV, liquidity-seeking, quote skewing, inventory policy. These live in latency-sensitive C++/FPGA stacks and consume *prices* from VALAX, not the other way around. |
| **Microsecond-latency execution paths** | Code written in C++ or pinned to FPGA for HFT. VALAX targets sub-millisecond pricing on GPU, not microsecond execution. |
| **Signal generation / alpha** | Stat-arb signals, factor models, ML alphas. These produce *positions* that VALAX then values. Differentiable strategy research ([Vision](../vision.md)) uses VALAX as the *evaluation* substrate, not as an alpha library. |
| **Backtesting frameworks for systematic strategies** | Zipline, Backtrader, QuantConnect, Lopez de Prado-style ML pipelines. These are *position simulators*; VALAX is the *valuer* they call. |
| **Portfolio optimisation / robo-advice** | Mean-variance, Black-Litterman, risk-parity engines. VALAX can be the differentiable mark-to-market *inside* a gradient-based optimiser (roadmap P5), but it does not own the optimiser. |
| **Trade lifecycle, settlement, clearing, custody, collateral movements** | DTCC, Euroclear, CCPs, custodians, corporate-action processing. VALAX values positions and computes their risk; it does not manage the trade after the value is computed. |
| **Market data distribution / tick storage** | kdb+, OneTick, Solace, Refinitiv RDP. VALAX consumes snapshots; it is not a tick store. |
| **Trade repositories / regulatory feeds** | DTCC GTR, EMIR feeds. Compliance plumbing. |
| ***Problems no pricing library fixes*** | |
| **Reference-data identifier mapping** | Lives upstream of any pricer (§ 3.1); the firm needs a golden-source identifier service regardless of what library prices the trade. |
| **Model risk governance (SR 11-7, SS1/23, TRIM)** | Process, not code (§ 3.2). VALAX makes validation cheaper (pure functions, deterministic outputs) but does not replace MRM. |
| **Regulatory approval cycles for IMA / new product classes** | Calendar-driven and human-driven. The most VALAX can do is shorten the *internal* preparation: validation packs, backtesting evidence, PLA tracking. |
| **Joint factor calibration across the entire firm** | The *primitives* are in place (autodiff, joint pytree scenarios, PCA bucketing). The actual joint model — what factors, what dependence, what stress library — is a multi-year quant programme that consumes the library; it doesn't ship with it. |

The design rationale in § 4 is a *targeted* claim, not a universal one. VALAX wins on the half of the problem that is amenable to architectural attack. The other half still requires a bank, with bank people, on a bank's timeline.

---

## 7. Where VALAX sits today

VALAX is not a finished competitor to the systems in [Peer Tools & Market Segments](peers.md). Today it is:

- **Operationally**, a pricing kernel and risk-sensitivity library covering standard equity options, vanilla and exotic rates, FX vanillas/forwards, inflation, credit-curve scaffolding, and four pricing methods (analytical / MC / PDE / lattice) with autodiff Greeks throughout — roughly the inner box of [Vision § "The shape of the mature engine"](../vision.md). The authoritative per-instrument status is the [coverage matrix](../applications/instrument-pricing.md).
- **Structurally**, the only stack the authors are aware of in this slot that is JAX-native, pure-functional, GPU-first, and differentiable end-to-end.
- **Strategically**, with `MarketState` and the multi-curve graph now shipped, on a roadmap whose next priorities (XVA + FRTB capital, then the service layer and market-data persistence) move VALAX from "library" to "deployable third-ring component" — see [Roadmap § "Production Priority Roadmap"](../roadmap.md#production-priority-roadmap-snapshot-reconciled-against-the-codebase).

The deliberate non-goals of § 6 stay constant. The whole bet is that there is exactly one shape of system the next generation of trading desks and risk teams need in the third ring, and that doing only that shape — well, and with the JAX ecosystem behind it — is more valuable than any of the adjacent boxes we could try to absorb.

---

## 8. Where to read next

- **The destination** these choices serve → [Vision](../vision.md); **the time-ordered build-out** → [Roadmap](../roadmap.md).
- **Who uses VALAX, and the per-business payoff of each choice** → [Applications](../applications/index.md).
- **Peer tools, prop houses, HFT and the buy side** → [Peer Tools & Market Segments](peers.md).
- **How the rationale plays out in the risk pipeline** → [Risk: End-to-End § 6](../risk-overview.md#6-why-valax-is-built-the-way-it-is).
- **The production scaffolding** that operationalises it → [Architecture: Productionisation Design](../architecture/production.md).
- **The tests that defend it** → [Validation Strategy](../architecture/validation-strategy.md); **the rules contributors follow** → [Invariants for contributors](../architecture/overview.md#invariants-for-contributors).

---

*This page changes slowly. It updates when the underlying claim about what is and is not hard about bank-scale quant systems changes — which, on the evidence of the last twenty years, is rarely.*
