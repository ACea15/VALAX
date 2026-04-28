# Vision: VALAX as a Trading & Investment Platform

This document captures the long-term ambition for VALAX: the kinds of workflows it would support, the users it would serve, and the structural advantages of its JAX-native architecture once mature.

It is **not** a feature list (see [`roadmap.md`](roadmap.md) for time-ordered tasks and tier breakdowns), and it is **not** a structural design (see [`architecture/production.md`](architecture/production.md) for the data model, layering, and phased delivery plan). It is the **why** that the roadmap and the architecture documents serve. It is deliberately untimed: it should still read true after several roadmap tiers have shipped.

VALAX today is a pricing kernel — option pricers, curve bootstrapping, autodiff Greeks, vmap portfolio risk. The destination described here is a **differentiable valuation, risk, and research engine** usable end-to-end by trading desks, asset managers, and quant researchers.

---

## The shape of the mature engine

By the time the production scaffolding (Workstream A in `architecture/production.md`) and the priority tiers in `roadmap.md` are in place, VALAX is no longer a library — it is a stack with three concentric layers of usage:

```
       ┌──────────────────────────────────────────┐
       │  Frontier (P5):                          │
       │  Neural surrogates, deep hedging,        │
       │  differentiable portfolio construction   │
       │  ← JAX-native moat                       │
       │                                          │
       │   ┌────────────────────────────────┐     │
       │   │  Strategic (P3, P4):           │     │
       │   │  XVA, FRTB, real-time risk,    │     │
       │   │  service APIs, market data     │     │
       │   │                                │     │
       │   │   ┌──────────────────────┐     │     │
       │   │   │  Operational (P1,P2):│     │     │
       │   │   │  EOD MTM, calibration│     │     │
       │   │   │  pricing, Greeks,    │     │     │
       │   │   │  scenario VaR        │     │     │
       │   │   └──────────────────────┘     │     │
       │   └────────────────────────────────┘     │
       └──────────────────────────────────────────┘
```

Each ring is a different kind of customer doing a different kind of work. The outer rings consume the inner rings as pure functions — there is no separate "research" stack and "production" stack.

---

## 1. Operational uses

The first thing VALAX gives a trading or investment shop is the unglamorous-but-essential daily pipeline. Once `MarketState`, `MarketSpec`, the snapshot store, and the multi-curve graph solver are landed, the engine supports:

### End-of-day mark-to-market
- `valax build-state` ingests vendor quotes → calibrated `MarketState`.
- `valax mtm` revalues the book → P&L parquet + per-trade Greeks.
- The same snapshot replays years later, bitwise-identical.

**Users:** middle office, product control, risk reporting.
**Why VALAX wins:** snapshot determinism is enforced *by construction* (see [`production.md` §9](architecture/production.md)), not by ops discipline. Same `(quotes, spec, version)` → same `snapshot_id` → same P&L. Regulator-grade reproducibility without a separate audit system.

### Curve and surface marks for traders
- The graph solver produces OIS / tenor / cross-currency basis curves with diagnostics per quote.
- Vol surfaces (SABR, SVI, FX delta-space) are arbitrage-free and differentiable.
- The `quote_jacobian` is the desk's **risk-transfer matrix**: how to hedge curve risk with liquid instruments.

**Users:** rates traders, vol traders, structuring desks pricing client requests.

### Pre-trade pricing and quote generation
- Sales / structuring desks call the pricer with hypothetical trades and see PV, Greeks, and risk decomposition before sending a quote.
- With the gRPC service (P4.1) this is a sub-millisecond call from a front-office app.
- Vega / vanna / volga on an exotic are *exact* (autodiff), not bumped.

**Users:** structuring, sales, market makers.

### Calibration as a service
- Any user can re-calibrate to a perturbed surface ("show me my book if I refit excluding the 25Δ wing point") because calibration is a pure function.
- Calibration is itself differentiable: sensitivity of fitted parameters to input quotes is available, not approximated.

---

## 2. Strategic uses

This tier is where VALAX stops being a pricing tool and becomes the **pricing-and-risk core of the firm**.

### Risk and regulatory
- **Full-revaluation VaR / ES** — already implemented; with `vmap` + GPU, 10k-trade books × 1k scenarios run in seconds, not hours. This is the single largest practical win over QuantLib-style stacks.
- **FRTB SA & IMA** — the sensitivities required by FRTB are exactly what `jax.grad` produces. The standardized approach reduces to a registry of risk-weight tables on top of existing Greeks.
- **Reverse stress testing** — `optimistix.minimize` over `MarketScenario` parameters to find the smallest market move that breaches a P&L threshold. Hard in C++ stacks; trivial here because the entire pricing chain is differentiable.

### XVA and counterparty risk
This is the **single highest-value structural-advantage use case**, and the one where the JAX architecture creates a moat that traditional bank stacks cannot replicate cheaply:

- Exposure simulation = nested Monte Carlo. Outer loop: market paths. Inner loop: full-portfolio repricing at each timestep.
- A bank XVA system today is a multi-million-dollar overnight grid job.
- VALAX-on-GPU collapses that into a daytime run with **fully-differentiable CVA, DVA, FVA, MVA per trade**.
- **CVA Greeks** (sensitivity of CVA to market factors) come from `jax.grad`, not from a separate finite-difference engine, so the XVA desk hedges dynamically with the same numbers used to mark.

### Real-time desk integration
With the service layer, audit logging, and market-data adapters in place:
- Front-office GUIs call `PriceService.MarkToMarket` for live P&L tickers.
- Risk dashboards subscribe to streaming Greek updates as market data ticks.
- Every pricing call is auditable: `(snapshot_id, code_version, run_id)` → reproducible decision log.

This is the moment VALAX becomes embeddable in a real bank or hedge fund stack.

---

## 3. Investment-management uses (cross-cutting)

The roadmap is bank-flavoured, but the same engine serves a buy-side audience naturally. The following table maps representative buy-side workflows to the roadmap tiers that unlock them:

| Buy-side use case | What it needs from VALAX | Roadmap tier |
|---|---|---|
| LDI / pension liability hedging | Inflation curves, key-rate durations, swap pricing | Already done |
| Convertible arbitrage | Convertible bond pricer, equity-credit hybrid PDE | 3.8 |
| Vol arbitrage / dispersion | Multi-asset MC, variance swaps, correlation Greeks | Already done |
| Structured product evaluation (buy-side) | Autocallable / phoenix pricing under SLV | 3.6 + 2.4 |
| Tail-hedge programs | Barrier options, lookbacks, jump-diffusion | 3.6 + 2.5 |
| Systematic credit | CDS, survival curves, tranche correlation | 3.4 |
| Macro / rates RV | Multi-curve graph, swaptions, CMS spreads, Bermudans | P1, 2.1, 3.7 |
| Portfolio construction with derivatives | Differentiable mark-to-market over allocations | P5.x |

The last row is the conceptually interesting one. Most asset managers optimize portfolios on linear assets and treat derivatives as a side-car. VALAX's autodiff backbone means a single `jax.grad` over `mark_to_market(...)` returns sensitivities of total book P&L to *any* parameter — allocation, hedge ratio, structured-note coupon, anything. That is the foundation for **gradient-based portfolio construction with derivatives in the loop** — work that is prohibitively expensive in non-differentiable stacks.

---

## 4. Frontier uses — the JAX-native moat

These are the use cases that justify building VALAX in JAX rather than wrapping QuantLib. They become accessible *for free* once the operational layer exists, because they consume the same pure pricing functions.

### Neural surrogate pricers
- Train a small NN on `(market_state, instrument) → price` for slow exotics (Bermudans, autocallables, MBS).
- **Differential ML**: train on Greeks too, using `jax.grad` of the slow ground-truth pricer as auxiliary labels.
- Result: real-time pricing of products that today take seconds, with a continuous Greeks surface inherited from autodiff through the NN.
- Use case: a structuring desk quoting autocallables to clients in milliseconds; an XVA system using surrogates in the inner loop to crush nested-MC cost.

### Deep hedging
- An RL agent trained inside VALAX learns hedging policies that respect transaction costs, market impact, and discrete rebalancing.
- The entire training loop is differentiable: market simulator → portfolio dynamics → cost function → backprop.
- Use case: optimal hedging of a structured note book; market-making policies for a vol desk; replicating exotic payoffs under realistic frictions.

### Differentiable trading research
- `MarketState` → `Portfolio` → `mark_to_market` is one composable pure function.
- `jax.grad` of *firm P&L* with respect to any control input — trade sizes, hedge ratios, rebalance frequency, calibration weights — is just a function call.
- This is the foundation for **strategy research with derivatives in the loop**, a class of work that is prohibitively expensive elsewhere.

### Bayesian and uncertainty-aware valuation
- JAX integrates with NumPyro and blackjax. Posterior distributions over calibrated parameters can be propagated through pricing.
- Use case: model-risk reserves; uncertainty quantification on illiquid marks; robust XVA.

---

## The arc, in one paragraph

**VALAX today** is a pricing kernel. **VALAX after Workstream A and Priorities 1–2** is a deterministic EOD valuation engine that any trading shop can adopt for marks, calibration, and Greeks. **VALAX after Priorities 3–4** is a full risk-and-regulatory backbone — real-time XVA, FRTB capital, service APIs, market-data integration — at which point it competes head-on with internal bank quant libraries on cost and beats them on GPU XVA. **VALAX after Priority 5** is something no major bank library has today: a fully-differentiable trading research environment where pricing, calibration, hedging, and portfolio construction are all gradients of one composition.

---

## Implications for sequencing

The vision implies a small set of durable principles for prioritization. Specific time-ordered tasks live in [`roadmap.md`](roadmap.md); the principles are:

1. **`MarketState` is the highest-leverage prerequisite.** It is a hard dependency for the service layer (P4), XVA (P3.1), and neural surrogates (P5.1). Roughly 80% of everything downstream blocks on it.
2. **The multi-curve graph unlocks three of the four strategic use cases.** It is independently shippable and should be progressed in parallel with `MarketState`.
3. **Every new pricer must remain `jax.grad`-able and `vmap`-able.** This single discipline keeps the entire frontier (P5) reachable for free. The acceptance test for a new module is: *can I differentiate it, vectorize it, and sample its inputs from a posterior?*
4. **The XVA prototype is the strategic crown jewel.** Once `MarketState`, the multi-curve graph, and a credit survival curve are in place, a CVA demo on GPU is the single artefact most likely to make a bank seriously evaluate VALAX. Plan its dependencies (netting sets, exposure harness) early so it isn't a re-architecture later.
5. **Defer P5, but design for it.** Don't build neural surrogates yet, but never compromise the purity of pricing functions. The frontier is paid for by discipline at the operational layer.

---

## Non-goals

To keep scope honest, this vision **does not** describe:

- A trading or order-management system. VALAX values and computes risk on positions; it does not place orders, manage execution, or interact with venues.
- A market-making or signal-generation system. It is the *pricing and risk* layer underneath such systems, not a replacement for them.
- A portfolio recommendation or robo-advisor product.
- A real-time tick-by-tick low-latency engine of the kind written in C++ or FPGA. VALAX targets sub-millisecond pricing on the GPU, not microsecond execution paths.
- A delivery commitment. The capabilities described here are the *destination*, not promises tied to a date.

---

This document is expected to evolve slowly. When a roadmap tier ships, `roadmap.md` updates; this document only updates if the *kind* of thing VALAX is becoming changes.
