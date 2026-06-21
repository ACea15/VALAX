# Where VALAX Fits: The Quant-Finance Toolchain

*A map of the systems that surround a pricing-and-risk engine at a sell-side bank or a buy-side hedge fund, and an honest account of which boxes VALAX fills, which boxes it couples to, and which boxes it deliberately does not try to be.*

This page is the **horizontal** complement to [Vision](vision.md) (the destination), [Design Rationale](design-rationale.md) (why the architecture is what it is), and [Roadmap](roadmap.md) (the order in which features land). Those documents look *inward* — what VALAX is, why, and when. This one looks *outward* — the universe of tools around VALAX and where in it we sit today.

If you are evaluating VALAX for adoption, this is the page that answers *"what else would I need to run a desk on this?"*.

---

## 1. The sixty-second story

A trading floor is not one piece of software. It is **twenty to forty distinct systems** wired together by contracts on identifiers, timestamps, calendars, and conventions. Some of those systems are commodity (the general ledger; collateral management). Some are competitive moats that every firm builds in-house (alpha signals; execution algos). And in the middle sits a small cluster of systems that are neither commodity nor signal-bearing, but absolutely load-bearing: **the pricing-and-risk engine**.

That cluster is what computes a fair price, a Greek, a VaR number, a CVA charge, and an FRTB sensitivity. Every conversation a trading desk has with risk, with finance, with the regulator, and with its own P&L attribution flows through it. **VALAX is exactly this cluster** — and nothing else.

We are not a market-data vendor. We are not an order-management system. We are not an alpha library. We are not a settlement platform. We are the differentiable pricing-and-risk kernel that all of those systems either feed or consume, designed so a single JIT-compiled function call serves the front-office quote, the EOD risk batch, the XVA inner loop, and the regulator's IMA backtest from the same line of code.

---

## 2. The bank/hedge-fund stack at a glance

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

VALAX is the **third ring**. It is fed by rings 1–2 (trades, market data) and consumed by rings 4–6 (risk, capital, finance, regulators). Every ring above and below VALAX is either commodity infrastructure that already exists in a firm, or it is a competitive moat that the firm builds itself. The third ring is the one where most firms today are paying the highest *consistency tax* — see [Design Rationale §2](design-rationale.md) — and where a JAX-native, differentiable, deterministic engine has the most leverage to offer.

---

## 3. What each department wants from the pricing-and-risk core

The same engine is queried by very different people on very different cadences. The table below is the load-bearing claim for *why* the third ring has to be a single coherent stack rather than the dozen overlapping ones that most banks accumulate.

| Department | Headline question | Cadence | What VALAX gives them |
|---|---|---|---|
| **Trading desk** | "What's the fair price? What's my delta? If SPX drops 2% what do I lose?" | Tick / intraday | Analytical, MC, PDE, and lattice pricers with autodiff Greeks (`valax/pricing`, `valax/greeks`). |
| **Structuring / sales** | "Quote me an exotic in seconds, with Greeks." | Pre-trade | Same pricers via the (planned) `PriceService.MarkToMarket` gRPC endpoint — sub-millisecond on GPU once compiled. |
| **Automated market making** | "Give me a fair mid and a confidence interval at thousands of quotes/sec." | Microsecond–millisecond | JIT-compiled pricing kernels, `vmap` across strike/tenor grids, GPU residency for hot books. (VALAX does *not* place the quote — see §5.) |
| **Market risk (MRM)** | "Is this desk within its DV01 / vega / VaR / stress limit? Did the model under-predict yesterday's loss?" | EOD + intraday | `valax/risk` — ladders, scenario engines, VaR/ES, backtests, PLA. The portfolio is differentiated *as a whole*. |
| **XVA desk** | "What's CVA / DVA / FVA on this netting set? Hedge ratios to CDS / IR / FX?" | Daily, with intraday what-if | Nested MC where the inner repricing is the same JIT'd function used by the front office. CVA Greeks come from `jax.grad`, not finite differences. (Roadmap P3.1.) |
| **Treasury / ALM** | "IR sensitivity by bucket? Funding gap by tenor? FX exposure overall?" | Daily | Bucketed Greeks at standard vertices, multi-curve key-rate durations, FX exposures from the same `MarketData` pytree. |
| **Model validation / model risk** | "Does the engine agree with QuantLib / market consensus? Is the calibration stable?" | Monthly + ad-hoc | The [QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md), the synthetic-market harness, and reproducible snapshots. |
| **Product control / finance** | "Reconcile yesterday's official P&L to today's. Explain the move." | EOD | `pnl_attribution`, `waterfall_pnl_report`, and bit-exact snapshot replay. |
| **Regulator (Basel, national supervisor)** | "Capital number. Backtest pass/fail. PLA pass/fail. Sensitivities at FRTB / SIMM vertices." | Quarterly | Bucketed Greeks at FRTB vertices, Basel traffic-light backtests, PLA Spearman/KS. (Full FRTB SA on roadmap P3.2.) |
| **Internal / external audit** | "Reproduce yesterday's risk numbers from yesterday's market data." | Annual + ad-hoc | Determinism by construction: same `(quotes, spec, version)` → same `snapshot_id` → same P&L. |
| **Quant research** | "Sensitivity of CVA to a tweak in the calibration weights? Train a neural surrogate for autocallables?" | Continuous | Any pricing function is a pure differentiable function — composable with NumPyro, Optax, Flax/Equinox NNs out of the box. |

The point of the table is not the list of consumers. It is that they all need to receive **numerically identical** answers derived from the **same** pricing function. The cost of the *opposite* world — where the trader's pricer, the risk system's pricer, and the regulatory pricer have drifted apart — is the largest hidden line item in a tier-1 bank's quant budget. See [Design Rationale §2.1](design-rationale.md) for the full argument.

---

## 4. The flagship use cases, by what they stress in the engine

VALAX's architecture is a bet on three workflows being the highest-leverage ones for a JAX-native engine. Each stresses a different property of the stack.

### 4.1 Trading — fair price, fast and consistent

**What's needed.** A price that the desk trusts, a Greek that hedges cleanly, and a reproducible mark at EOD that the risk team also trusts.

**What VALAX provides.** Pure-function pricers covering the standard book (equity vanillas/exotics, rates vanillas/exotics, FX, inflation, multi-asset), with autodiff Greeks that are *the same numbers* the risk system uses. The "trader's delta vs the risk system's delta" reconciliation conversation — the one that consumes a chunk of every middle-office team's day — does not exist when both come from the same `jax.grad`.

**What's exercised.** Pricing breadth (`valax/pricing/*`), instrument coverage (`valax/instruments`), autodiff stability (no double-where leaks, no NaN-poisoning), JIT cache discipline.

### 4.2 Automated market making — speed of evaluation

**What's needed.** Tens of thousands of revaluations per second across a quoting grid (strike × tenor × side), with fair-value updates as market data ticks, plus a Greek for inventory management.

**What VALAX provides.** JIT compilation, `vmap` across the quoting grid, GPU residency for the hot book, and analytical / fast-Fourier pricers where MC is too slow (Black-Scholes, Black-76, Bachelier, Heston Fang-Oosterlee COS, SABR analytic). For exotics on the same desk, neural surrogate pricers (roadmap P5.1) plug in as drop-in pure functions that inherit autodiff Greeks.

**What's exercised.** JIT compilation latency, `vmap` shape stability, accelerator residency, surrogate-pricer training infrastructure.

**What VALAX is *not*.** The market-making *strategy* — quote skewing, inventory limits, queue position, adverse-selection avoidance — is the firm's competitive moat and lives in the AMM engine itself. VALAX answers the question "what is fair?" so the AMM strategy can decide where to quote relative to it.

### 4.3 Market risk — derivatives of portfolios

**What's needed.** A 10,000-trade book differentiated *as a whole* against thousands of market factors, repriced under thousands of scenarios, all before the EOD batch window closes.

**What VALAX provides.** Reverse-mode autodiff over the entire portfolio: one backward pass computes every Greek of every trade against every factor. `vmap` across scenarios collapses what is a Python loop in C++ stacks into a single JIT-compiled call. On GPU this is the **single largest practical win over QuantLib-style stacks** — see [Vision §2](vision.md).

**What's exercised.** The full sensitivity ladder (`valax/risk`), scenario engine, bucketing, VaR/ES on a vectorised P&L array.

### 4.4 XVA and counterparty risk — nested Monte Carlo at GPU speed

**What's needed.** Exposure simulation = outer-loop market paths × inner-loop full-portfolio repricing at each timestep. A bank CVA engine today is a multi-million-dollar overnight grid job.

**What VALAX provides.** The inner loop is the same JIT'd pricing function used by the front office. The outer loop is `vmap` across scenarios. CVA Greeks come from `jax.grad`, not a separate finite-difference engine, so the XVA desk hedges with the *same numbers used to mark*. This is the **strategic crown jewel** on the roadmap (P3.1).

### 4.5 Research and what-if

**What's needed.** A pre-trade what-if ("if I add this hedge, what does it do to my desk's IMA capital?") that has trader-grade latency, plus the freedom to swap in a Bayesian posterior, an NN surrogate, or a calibrated SLV without rewriting the stack.

**What VALAX provides.** No "research vs production" duality. The research stack *is* the production stack — see [Design Rationale §3](design-rationale.md). Same pricers, same `MarketData`, same Greeks. Any pure function can be wrapped in NumPyro for uncertainty, Optax for calibration, or Flax/Equinox for surrogates.

---

## 5. What VALAX is *not*, and what couples to it

VALAX is a deliberately narrow tool. The list below is the set of systems that surround it in a real deployment — what VALAX expects to receive from them, what it returns, and where they live in a firm's organisation. Treat this as the **integration map**.

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
| **XVA / capital engines** | Internal CVA engines, Numerix CrossAsset XVA, Murex MX.3 XVA | Path-wise revaluations consumed by the outer simulation. *Long-term, VALAX-on-GPU is the XVA engine itself; in the medium term it can also feed an existing one.* |
| **Front-office UIs / quote engines** | RFQ blotters, AMM engines, internal trader cockpits | Sub-millisecond pricing and Greeks via the (planned) gRPC service. |

### 5.3 Adjacent — what VALAX deliberately does not try to be

These are the boxes around VALAX on the floor. Trying to absorb any of them is out of scope; coupling to them is the whole point.

| Adjacent capability | Why it is out of scope |
|---|---|
| **Order management (OMS) and execution (EMS)** | Order routing, venue connectivity, smart-order-routing logic, exchange certifications. VALAX has no view on *where* a trade is placed. |
| **Execution algorithms** | VWAP, TWAP, POV, liquidity-seeking, market-making strategies. These live in latency-sensitive C++/FPGA stacks and consume *prices* from VALAX, not the other way around. |
| **Signal generation / alpha** | Stat-arb signals, factor models, ML alphas. These produce *positions* that VALAX then values. (Differentiable strategy research from §4.5 is the *evaluation* substrate, not an alpha library.) |
| **Backtesting frameworks for systematic strategies** | Zipline, Backtrader, QuantConnect, Lopez de Prado-style ML pipelines. These are *position simulators*; VALAX is the *valuer* they call. |
| **Portfolio optimisation / robo-advice** | Mean-variance, Black-Litterman, risk parity engines. VALAX can be the differentiable mark-to-market *inside* a gradient-based portfolio optimiser (roadmap P5), but it does not own the optimiser. |
| **Settlement, clearing, custody** | DTCC, Euroclear, CCPs, custodian banks. Pure operations infrastructure. |
| **Market data distribution / tick storage** | kdb+, OneTick, Solace, Refinitiv RDP. VALAX consumes snapshots; it is not a tick store. |
| **Trade repositories / regulatory feeds** | DTCC GTR, EMIR feeds. Compliance plumbing. |
| **Microsecond-latency execution paths** | The kind of code written in C++ or pinned to FPGA for HFT. VALAX targets sub-millisecond pricing on GPU, not microsecond execution. |

---

## 6. Peer tools — what else fills the third ring today

VALAX competes for the same job as a small set of well-known systems. The honest comparison is:

| Tool | Origin | Strengths | Where VALAX is structurally different |
|---|---|---|---|
| **QuantLib** | Open source, C++ (1999–) | Enormous instrument and convention breadth; the de-facto cross-check oracle. | Bumped Greeks, mutable engine pattern, CPU-only, no autodiff. VALAX validates against it (see [QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md)) but does not inherit its architecture. |
| **Murex MX.3 / Calypso / Summit / Numerix CrossAsset / FINCAD** | Commercial sell-side platforms | Battle-tested coverage of structured products, lifecycle, integrations, XVA modules. | Closed, vendor-bound, finite-difference Greeks, no GPU/autodiff story. Multi-million-dollar licences. |
| **Bank internal libraries** (JPM Athena, GS SecDB/Slang, BAML Quartz, MS Hydra) | In-house, Python or Slang/Python | Tight integration with golden trade source; firm-specific governance baked in. | Built over 20+ years on mutable object graphs; the *consistency tax* described in [Design Rationale §2](design-rationale.md) is exactly what motivates VALAX. |
| **Hedge-fund / prop "quant kitchens"** (e.g. Bloomberg DLib, hedge-fund internal pricers) | Mixed, often Python over C++ kernels | Lean, focused on the strategies actually traded; fast iteration. | Often missing rigorous calibration, audit, governance for sell-side use — and almost never autodiff-native or GPU-vectorised across instruments × scenarios. **The picture is more nuanced than the table suggests — see §6.1 below.** |
| **Domain-specific JAX libraries** (`diffrax`, `optimistix`, `equinox`, `numpyro`, `flax`) | Open source, JAX ecosystem | The libraries VALAX is built *on*. | Not pricers; they are the differentiable maths VALAX composes into pricers. |

The pattern is consistent: every existing tool in the third ring made architectural choices that pre-date modern autodiff and accelerators. VALAX's bet is that a pure-functional, differentiable, GPU-first kernel — written today, with twenty-five years of QuantLib coverage as a validation oracle — is a strictly better foundation for the next twenty years of pricing-and-risk work than retrofitting any of the above.

For the full *why* of that bet, see [Design Rationale](design-rationale.md).

### 6.1 A note on prop houses and HFT — narrower scope, different economics

A caveat up front: **prop trading and HFT are deliberately secretive industries**, and almost nothing about what's actually inside the major firms is public. What follows is a synthesis of what is plausibly inferable from public job postings, conference talks, alumni-authored blog posts, court filings, and the trading behaviour observable in the market itself. Treat it as informed guesswork, not insider knowledge. Specific firms are named as *representatives of a category*, not as claims about their internal stacks.

With that caveat, the popular picture — *"the big banks have decade-old libraries with Heston and Black-Scholes, but so do the prop houses and the HFT firms"* — is likely partly right and mostly misleading. The space appears to be sharply stratified, and the stratification matters for placing VALAX honestly. Roughly four groups:

**1. Bank-style libraries.** Athena (JPM), SecDB/Slang (Goldman), Quartz (BAML), Hydra (Morgan Stanley); or the vendor equivalents Murex MX.3, Calypso, Numerix CrossAsset. These are the best-documented of the four groups because banks publish papers, alumni write books (e.g. Joshi, Hull, the Athena/SecDB literature), and regulators force disclosure of broad architectural shape. They cover hundreds of instrument types across every asset class, with decades of accumulated convention handling, model governance baked in, and integration into golden-source trade stores. They are slow to change, expensive to run, and structurally what VALAX is designed to challenge. The §6 comparison stands for this group.

**2. Prop options market makers — likely narrow, fast, and proprietary.** Optiver, IMC, SIG, Akuna, Wolverine, CTC, Flow Traders, the listed-options books at Jane Street and Citadel Securities. These firms make markets in listed options at scale, and several things about the business *suggest* they build their own pricing infrastructure rather than license a bank-style library:

- Their product universe — listed equity / index / ETF / futures options — is narrow compared to a bank's cross-asset book. The maths needed (Black-Scholes, CRR or trinomial for American, a smile model, dividends, borrow rates, maybe a jump term for short-dated) is a small, stable syllabus that has not fundamentally changed since the 1990s.
- Their latency requirements are well below what any bank or vendor library targets, which makes a tuned in-house pricer plausibly easier than retrofitting.
- Public job postings at these firms routinely advertise C++ / quant-developer roles building "low-latency pricing infrastructure", "vol surface fitting", and "Greeks computation". This is consistent with in-house libraries; it isn't proof of them.
- The economics appear to support building rather than licensing — these are profitable firms with quant headcounts in the tens to low hundreds, against revenue lines large enough to absorb the investment.

What is much less clear from the outside is *how much* of their stack is bespoke versus glued together from open-source components and proprietary calibration on top. **VALAX is most likely not a direct hot-path competitor for this group** — firms that have already invested in a tuned, latency-optimised pricer are unlikely to swap it for a general-purpose library. The more honest pitch is *augmentation*: VALAX for the **batch overnight risk run**, **portfolio-level Greeks across the book**, and the **what-if / scenario layer** that a latency-tuned hot-path pricer is typically not designed for. The hot path stays theirs; the slow path is where an open, differentiable, GPU-vectorised engine has something distinctive to offer.

**3. Pure HFT — probably a different conversation entirely.** Virtu, Tower, Hudson River Trading, Two Sigma Securities, XTX, Jump's HFT book, the cash-equity / futures / ETF businesses inside Citadel Securities. For firms in this group whose primary business is cash equities, futures basis, ETF-vs-constituent arbitrage, or rates cash-vs-futures arb, public discussion of their stacks (conference talks, ex-employee accounts, court documents from various lawsuits) consistently points away from option-pricing-model machinery and towards *fair-value models* — predictors of the next mid from order-book state and microstructure features, often implemented in lean C++ or pinned to FPGA. Risk in this world tends to be positional ("flat by close") rather than Greek-based, and the "valuation" of a position is typically the venue mid rather than a model output. **It is hard to see a natural role for VALAX in this style of business**, and we should not claim one. When a firm in this category also runs a listed-options book — Virtu and Citadel Securities both do — that book is widely understood to be run separately, on infrastructure that looks much more like group 2.

**4. The buy-side spectrum — heterogeneous and the most VALAX-relevant.** Hedge funds and asset managers vary enormously, and unlike banks they face essentially no public-disclosure pressure on their internal stacks. The picture below is a rough sketch of categories rather than a claim about any individual firm:

- **Vol-arb / dispersion / structured-note funds** (Capstone, Parallax, the vol pods inside multi-strats) — likely run proprietary pricers, narrowly scoped, with the same economics as group 2.
- **Rates RV and macro funds** (Brevan, Rokos, Element, Caxton) — plausibly a Python calibration layer on top of QuantLib or a wrapped bank library, plus a handful of proprietary models for the specific trades they put on, though the mix varies.
- **Convertible arb, distressed credit, structured-credit funds** — coverage ranges from senior-quant hand-rolled pricers to licensed Numerix / Kamakura installs to thin wrappers around vendor analytics. Smaller funds in this space generally cannot afford to build *or* license fully and likely live with the resulting precision loss.
- **Multi-strat pod shops** (Millennium, P72, Balyasny, ExodusPoint, Schonfeld) — the public picture, reinforced by their hiring patterns, is that there is no firm-wide pricing library; each pod brings or builds its own tools, and the firm provides risk aggregation on top, often based on pod-reported sensitivities rather than independent repricing.
- **Systematic equity / CTA funds** (Two Sigma, AQR, Renaissance, Winton, Man AHL) — to the extent these firms trade options at all, the pricer is generally not the part of the stack they invest in; their proprietary code is the *signal*, not the valuation.

**Where VALAX likely fits.** Pulling the four groups together: banks have built their libraries over decades because regulation and product breadth force them to cover everything; prop options market makers appear to have built narrower ones because a small team can plausibly own the whole syllabus for a focused product set; pure HFT firms don't seem to need pricing libraries in the bank sense at all; and the buy-side picture is patchy. The audience for whom **building** a bank-style stack is uneconomical and **licensing** Numerix or Murex is overkill — multi-strat hedge-fund pods, mid-sized vol funds, sophisticated family offices, new prop desks spinning up a derivatives book, and arguably the *slow-path risk and what-if layer* sitting behind a group-2 prop options market maker — is the audience where a credible open, differentiable, GPU-vectorised third-ring component appears most likely to be the missing option today. That is where we expect VALAX to have its strongest pull. Whether any specific firm sees it that way is, of course, something only that firm can tell us.

---

## 7. Buy-side counterpart — what changes at a hedge fund or asset manager

The map above is bank-flavoured because banks publish the most about their stacks. The same picture holds on the buy side with three substitutions:

1. **Trade volumes and product zoo are narrower.** A vol-arb fund cares about a few hundred listed options and variance swaps; an LDI manager cares about IR/inflation swaps and bonds. Coverage requirements are smaller, but the maths is the same.
2. **Regulatory ring is lighter but not empty.** No FRTB IMA, but SIMM for uncleared margin, SEC/CFTC reporting for swaps, and AIFMD/UCITS risk reporting in Europe. The bucketed-sensitivities outputs from VALAX feed these the same way they feed bank reports.
3. **The biggest win shifts from XVA to differentiable strategy research.** Buy-side workflows that today are prohibitively expensive — gradient-based portfolio construction with derivatives in the loop, deep hedging of structured-note books, Bayesian posteriors on calibrated parameters — are the natural consumers of VALAX's frontier layer (roadmap P5). See [Vision §3 and §4](vision.md) for the table mapping buy-side workflows to VALAX features.

---

## 8. Where we sit *today*, honestly

VALAX is not a finished competitor to the systems in §6. Today it is:

- **Operationally**, a pricing kernel and risk-ladder library that already covers standard equity options, vanilla rates, FX vanillas/forwards, inflation, credit-curve scaffolding, and four pricing methods (analytical / MC / PDE / lattice) with autodiff Greeks throughout — roughly the inner box of [Vision § "The shape of the mature engine"](vision.md).
- **Structurally**, the only stack the authors are aware of in this slot that is JAX-native, pure-functional, GPU-first, and differentiable end-to-end.
- **Strategically**, on a roadmap whose next two priorities (`MarketState` + multi-curve graph; XVA + FRTB) move VALAX from "library" to "deployable third-ring component" — see [Roadmap §"Production Priority Roadmap"](roadmap.md#production-priority-roadmap-snapshot-june-2025).

The deliberate non-goals stay constant: we do not become an OMS, an alpha library, a tick store, or a microsecond execution path. The whole bet is that there is exactly one shape of system that the next generation of trading desks and risk teams need in the third ring, and that doing only that shape — well, and with the JAX ecosystem behind it — is more valuable than any one of the adjacent boxes we could try to absorb.

---

## See also

- [Vision](vision.md) — what VALAX is becoming and which use cases unlock when.
- [Design Rationale](design-rationale.md) — why pure-functional + autodiff + GPU is the right bet for the third ring.
- [Roadmap](roadmap.md) — the order in which the capabilities described here ship.
- [Risk: End-to-End](risk-overview.md) — the audience-by-audience tour of the risk side of the engine.
- [Architecture → Productionisation Design](architecture/production.md) — how the third-ring component becomes a deployable service.
