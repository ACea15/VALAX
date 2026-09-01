# Front-Office Trading — the Differentiable Pricing Core

*How VALAX serves a front-office desk running an **in-house trading platform**: not as the strategy and not as the execution gateway, but as the differentiable **pricing-and-risk core** that values a universe of products consistently and — the part that is actually novel — exposes exact **gradients that cross into the decision layer**, at intraday cadence, on CPU or GPU.*

Before the pitch, the honest framing of *what is and isn't novel here*, because it changes what the page is even about:

> **End-of-day batch valuation and risk is a commodity.** Every bank already reprices its book overnight and computes bumped Greeks and VaR. Doing the same computations *intraday* is a useful cadence improvement — real-time risk exists, it is just expensive and painful on C++ analytics stacks — but it is *incremental*, not categorical. If VALAX's only claim were "the same EOD numbers, faster," it would not earn a place on a desk.
>
> **The categorical claim is different: the pricing graph is differentiable all the way into the decision layer.** The same computation that produces a price yields — exactly, cheaply — the Jacobian of prices and P&L with respect to the *tradable hedge instruments* and, at the frontier, with respect to the desk's own *decision variables* (hedge notionals, quote skews, inventory). An EOD C++ batch structurally cannot do this; it is the one thing this architecture offers that the incumbent does not. That is the axis on which this page argues, and [§ 2](#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer) is its centre of gravity.

This is also the audience the [Applications overview](index.md) ranks 🟢🟢 — a genuine strength *as a pricing-and-risk core*, held back from 🟢🟢🟢 only because the intraday *service layer*, streaming market data, and JIT warm-up strategy are [Productionisation § 8.1](../architecture/production.md#81-intraday-warm-serving-considerations-deferred) / Vision-tier items, and the decision-layer differentiability is genuine but *frontier* (primitives shipped, product not packaged). The page stays honest about each.

So it also draws a hard line and stays on the honest side of it:

> **VALAX is a candidate for the valuation and risk core of an intraday trading system. It is not the alpha, and it is not the execution stack.** Strategy signal generation, position sizing, order routing, and fill handling live in a *separate system* (typically a separate repository) that *consumes* VALAX prices, Greeks, and Jacobians. This page documents what the pricing core does well and where its boundary sits.

If you keep that separation, the fit is genuinely good. If you blur it — "the engine says fair value is 1.02, market is 1.00, so buy" — you are one step from the classic ways a model-vs-market residual loses money. The [§ 7 honesty section](#7-the-honest-boundary-what-the-pricer-is-not) is not boilerplate; it is the most important part of the page.

---

## 1. The shape of the system — where VALAX sits

An in-house intraday trading platform is at least three subsystems with three different owners, three different tolerances for latency, and three different failure modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│  market data feed  ──▶  curve / surface builders  ──▶  VALAX PRICER     │
│  (streaming quotes)      (calibration, intraday)        (jit + vmap)     │
│                                                              │           │
│                              prices + Greeks + scenario P&L ◀┘           │
│                                                              │           │
│           ┌──────────────────────────────────────────────────┴───────┐  │
│           ▼                                                           ▼  │
│   STRATEGY / SIGNAL LAYER   (separate repo)              RISK & LIMITS   │
│   edge? sizing? cost model? ─────────────┐               Greeks → hedge, │
│                                          │               limit checks    │
│                                          ▼                               │
│                            EXECUTION / OMS  (separate repo)              │
│                            spreads, market impact, fills, fees          │
└────────────────────────────────────────────────────────────────────────┘
```

VALAX owns exactly one box in that diagram — the **pricer** — plus a strong assist on the **risk & limits** box (Greeks and scenario P&L). It owns *none* of the strategy box and *none* of the execution box. What makes the interface at the arrow more than a number, though, is that it carries **exact gradients**, not just prices: the strategy and hedging layers receive the Jacobian of value and P&L, which is the substance of [§ 2](#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer). The three enabling strengths — **calibration robustness, easy Greeks, fast JIT/GPU** — all live inside that one box and are what make that differentiable interface trustworthy and fast.

The cadence assumption throughout this page is **intraday, not high-frequency**: revalue a book or a quote universe on a seconds-to-minutes loop, recalibrate surfaces as the market moves, refresh Greeks and Jacobians for hedging and limits. Intraday is the cadence that makes the differentiable interface *operationally live* — but note that cadence is the enabler, not the novelty. JAX's trace-compile-cache execution model fits that cadence well; it is the wrong tool for microsecond-latency HFT, and this page never pretends otherwise (see [§ 6](#6-latency-the-honest-numbers)).

---

## 2. The actual novelty — differentiability that crosses into the decision layer

Here is the claim that separates VALAX from any bank's existing EOD stack. It is not "we price faster." It is: **because the whole pipeline is one differentiable graph, the exact derivative of value and P&L is available at the boundary where trading decisions are made — with respect to the things a desk can actually act on.** There are two rungs to this, and they should be sold at very different confidence levels.

### 2.1 Exact hedges in *tradable instruments* — through calibration (near-shipped)

A bank's risk system tells you delta to the *model's* inputs. What a desk actually needs to hedge is sensitivity to the *liquid instruments it can trade* — the swaps, futures, and options it calibrated the curves and surfaces from. Getting there means differentiating **through the calibration**, which is exactly where finite-difference stacks become expensive and noisy and where autodiff becomes a categorical advantage:

```python
# ∂(calibrated curve/surface) / ∂(market quotes) — one linear solve, not a bump ladder.
# optimistix.ImplicitAdjoint differentiates the *fixed point* of the calibration,
# so the cost is independent of the solver's iteration count.
J = quote_jacobian(curve_graph, instruments, by="zero_rate")   # (n_outputs, n_quotes)
```

- **The hedge is expressed in instruments you can trade**, because the chain rule composes the pricer's sensitivities with the calibration Jacobian automatically. No separate "risk-in-hedge-instruments" transform to build and maintain.
- **It is exact and cheap.** Implicit differentiation through the calibration fixed point (`optimistix.ImplicitAdjoint`) costs one linear solve regardless of how many Newton steps the bootstrap took — where a bump-through-recalibration approach costs a full re-solve per quote and carries finite-difference noise.
- **An EOD C++ analytics batch approximates this with a bumped delta ladder.** VALAX produces the true Jacobian as a byproduct of the graph it already evaluated. This rung is essentially *shipped* — the machinery lives in the curve build (`quote_jacobian`, [Productionisation § 11.9](../architecture/production.md#11-multi-curve-framework)).

### 2.2 Gradient-based intraday decisions — the pricer and the control problem share one graph (frontier)

The second rung is the genuinely new territory, and it is honestly *frontier*, not product. Because the pricer, the SDE path simulation (`diffrax`), and the payoff are all one differentiable computation, a desk's *decision* problems become gradient-based optimizations whose objective flows straight through the pricer:

```python
import equinox as eqx
import optax

def objective(decision, market, book):
    # decision = hedge notionals / quote skews / inventory targets (the desk's controls)
    hedged_pnl = simulate_pnl(book, decision, market)      # diffrax paths, differentiable
    return -risk_adjusted(hedged_pnl) + transaction_cost(decision)

grad_obj = eqx.filter_grad(objective)   # exact gradient of the decision objective
# ... optax step on `decision` — the control tunes itself against the true P&L gradient
```

That single pattern is the trading-desk incarnation of deep hedging ([Quant Research § 6](quant-research.md)), and it unlocks a family of things an EOD batch structurally cannot express:

- **Optimal hedging under transaction costs** — solve for the hedge that minimises risk *net of* trading cost, not the naive zero-delta hedge, using the exact gradient of hedged P&L rather than a grid search.
- **Inventory / skew management and optimal quoting** — differentiate a quoting objective (spread capture vs. inventory risk vs. adverse selection penalty) with respect to the quote skews and act on the gradient.
- **One framework, no serialisation boundary** — the control optimizer lives in the same array framework as the pricer (`filter_grad` + `optax`), so there is no C++/Python gradient boundary to marshal across, which is precisely where this idea dies on incumbent stacks.

**Honesty gate.** The *primitives* for this rung are shipped and battle-tested (`diffrax`, `optax`, `eqx.filter_grad`, differentiable payoffs). The *packaged* intraday optimal-hedging / quoting product is not — it is a prototype-in-a-few-days effort on top of the library, not a feature you install. It is on this page because it is the honest answer to "where is the novelty," and because the potential is genuinely there in the architecture — not because it ships today. The [Vision](../vision.md) doc is where this lives on the roadmap.

**Where this still stops.** Even at rung two, VALAX supplies the *gradient*; the desk's separate system still owns the *objective* — what counts as edge, what the cost model is, what risk aversion to use, and whether the whole loop is a good idea on a given product. Differentiability makes the optimization exact; it does not make the objective correct. That boundary is [§ 7](#7-the-honest-boundary-what-the-pricer-is-not).

---

## 3. Fast JIT + GPU: pricing a whole product universe in one call

The intraday loop's core operation is: *given the current market, price every product I might quote or hold, and do it again a few seconds later.* That is precisely the operation JAX is built to make cheap.

A pricer written as a pure function `V(instrument, market) → price` composes into a batched, compiled kernel:

```python
import jax
import equinox as eqx
from jaxtyping import Array, Float

@eqx.filter_jit
def price_book(instruments, market) -> Float[Array, " n_products"]:
    # one compiled kernel prices the entire universe
    return jax.vmap(price_one, in_axes=(0, None))(instruments, market)
```

- **`vmap` collapses the product loop into an array axis.** A universe of 50 000 vanillas and light exotics is one kernel launch, not a Python loop over 50 000 objects.
- **`@eqx.filter_jit` compiles once, runs many.** After the first (warm-up) call, every subsequent revaluation reuses the compiled XLA executable — the per-tick cost is the kernel runtime, not tracing.
- **The same code runs on GPU with no changes.** JAX/XLA targets CPU, GPU, and TPU from one source; a desk can develop on CPU and deploy the identical kernel on a GPU when the universe grows.
- **Scenario grids are just another `vmap` axis.** Pricing the universe across a grid of market perturbations — for pre-trade "what if", limit headroom, or intraday stress — nests one `vmap` inside another and still compiles to a single kernel.

The practical payoff: the revaluation that a scalar C++ pricer runs as an overnight or minutes-long batch becomes a sub-second-to-seconds intraday refresh, because the batch *is* the unit of computation rather than an outer loop around it.

Two hard caveats, stated up front (expanded in [§ 6](#6-latency-the-honest-numbers)):

1. **First call compiles.** Cold-start compilation is seconds. You must pre-warm every kernel shape at start-up and keep the compiled functions resident — this is the "JIT warm-up strategy" the overview flags as a prerequisite.
2. **Shape changes retrace.** If the product count or path count changes shape between calls, XLA recompiles. The service layer must bucket work into stable shapes (fixed batch sizes, padding) so the hot path never retraces.

---

## 4. Easy Greeks: exact hedge ratios as a byproduct of the price

For a desk that holds risk, the pricer's most valuable output is often not the price — it is the *derivative* of the price. VALAX's headline advantage here is categorical: **every sensitivity is an autodiff call on the pricer you already wrote, exact to machine precision, with no bump-and-reprice infrastructure.**

```python
# Delta / rho / vega: gradient against whichever market leaves the pricer consumes
greeks = jax.grad(price_one)(instrument, market)

# Gamma / vanna / volga: second-order, same one-liner
hedge_convexity = jax.hessian(price_one)(instrument, market)

# Batched: the full Greek ladder for the whole book in one compiled call
book_greeks = jax.vmap(jax.grad(price_one), in_axes=(0, None))(instruments, market)
```

Why this matters specifically for an intraday desk:

- **Hedging is driven by exact deltas, not noisy bumps.** Bump-and-reprice deltas carry finite-difference noise that is worst exactly where it hurts — near barriers, near expiry, in the wings. Autodiff deltas have none of it. That directly reduces hedge slippage and P&L attribution noise.
- **The hedge ratio and the price come from the same graph.** There is no risk of the price and the delta being computed by two subtly different code paths — a recurring source of P&L-explain breaks in split analytics stacks.
- **Cross-Greeks are free.** `jax.hessian` gives the full second-order matrix (gamma, vanna, volga, cross-gamma) with no per-Greek engineering. A desk running gamma/vega limits gets the whole ladder without building a bump matrix per risk factor.
- **Sensitivities to *calibrated* inputs.** Because calibration is itself differentiable (see [§ 5](#5-calibration-robustness-the-differentiable-inner-loop)), you can propagate to sensitivities against the *raw market quotes* — the hedge instruments you can actually trade — via the quote Jacobian (`quote_jacobian` for curves), not just against model parameters.

The [Greeks guide](../guide/greeks.md) and [`valax/greeks/`](../api/greeks.md) document the wrappers; the point for this audience is that a new product ships with its complete risk ladder the moment its price function exists.

---

## 5. Calibration robustness: the differentiable inner loop

An intraday pricer is only as trustworthy as the market it is calibrated to. Surfaces and curves move through the session; a mis-calibrated or unstably-recalibrated surface produces confident, wrong "fair" values and injects false signal into whatever consumes them. Calibration robustness is therefore not a nicety for this audience — it is the difference between a usable engine and a dangerous one.

VALAX's differentiable design attacks this directly:

```python
import optimistix as optx

def residual(params, args):
    quotes, market = args
    model = jax.vmap(price_under_model)(instruments, params, market)
    return model - quotes                      # residual vector

solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
fit = optx.least_squares(residual, solver, initial_params, args=(quotes, market))
```

- **Exact Jacobians make the solve well-conditioned and fast.** `optimistix` differentiates the residual *through the pricer* by autodiff, so Levenberg–Marquardt gets the true Jacobian rather than a finite-difference approximation. That is both faster to converge and far less prone to the noise-driven non-convergence that plagues bumped calibrations — the property this page calls "calibration robustness".
- **Warm-starting keeps intraday recalibration stable.** Seeding each recalibration from the previous tick's solution (a natural pattern with a pure `fit(initial=previous)` call) keeps parameters on the same local solution as the market drifts, avoiding the parameter *jumps* that otherwise show up as spurious P&L and spurious signal.
- **Robustness is measurable, not asserted.** Because the whole pipeline is `vmap`-able, you can stress the calibration — `vmap` it over hundreds of small perturbations of the input quotes and inspect how much the fitted parameters (and downstream prices) move. A desk can *monitor* calibration stability intraday as a first-class health metric, not discover instability in the P&L the next morning.
- **Shipped, validated calibrations.** Heston, SABR, and SLV calibrations are in the tree today, each with an autodiff Jacobian and a QuantLib cross-check where a counterpart exists — see [Calibration](../guide/calibration.md), [Vol Surfaces](../guide/vol-surfaces.md), [SLV](../guide/slv.md).

The [Productionisation design](../architecture/production.md) formalises the surrounding machinery — `build_market_state`, per-artifact calibration diagnostics (RMSE, max fit error, Jacobian condition number), and deterministic snapshots — so an intraday build can report *why* it trusts (or rejects) the surface it just fit. Those diagnostics are the objective, auditable side of "calibration robustness".

---

## 6. Latency: the honest numbers

The single most common way this application disappoints is a latency expectation mismatch. The honest picture:

| Regime | Fit | Why |
|---|---|---|
| **Overnight / EOD batch** | 🟢 | The kernel *is* the batch; this is the library's comfort zone today. |
| **Intraday refresh (seconds–minutes)** | 🟢 with warm-up | Compiled kernels reprice a universe in sub-second-to-seconds on warm cache. This is the target regime of this page. |
| **Quote-response (tens–hundreds of ms)** | 🟡 | Achievable for bounded universes with a resident, pre-warmed service and stable shapes; needs the [service layer](../architecture/production.md#8-layer-6-service-layer-deferred). |
| **High-frequency (µs–low-ms)** | 🔴 | Wrong tool. JAX's trace/dispatch overhead and Python-hosted control flow are not built for this. Use a bespoke low-latency stack. |

The engineering that moves an intraday deployment from "works in a notebook" to "works on the desk" is entirely in the imperative shell, not the pricer:

1. **Pre-warm every kernel shape at start-up** so no client request ever pays cold-compilation cost.
2. **Bucket work into fixed shapes** (fixed batch sizes, padding) so the hot path never retraces.
3. **Keep compiled functions and the `MarketState` resident** in a long-lived service process ([Productionisation § 8](../architecture/production.md#8-layer-6-service-layer-deferred)).
4. **Pin dtypes and backends**, and note the GPU determinism caveats (cuBLAS non-determinism is real) if bitwise reproducibility across nodes matters.

None of these are shipped turnkey today — which is exactly why the overview ranks this audience 🟢🟢 (a strong pricing core) rather than 🟢🟢🟢.

---

## 7. The honest boundary — what the pricer is *not*

This is the section to read twice. A differentiable pricing engine is a *component*. Treating its output as a trading decision is a category error that the JAX-ness does nothing to fix.

- **"Fair value" means "consistent with your model and your inputs" — not "correct".** If the vol surface, discount curve, borrow, or dividend assumption is stale or wrong, the engine produces a confident, differentiable, wrong price. Garbage in, differentiable garbage out.
- **A model-vs-market gap is a hypothesis, not a signal.** Most of the time the market is not wrong — your model is missing something (a borrow cost, a skew premium, an event, a liquidity or credit adjustment). Systematically buying whatever your model calls "cheap" is a well-known way to accumulate exactly the positions informed counterparties are unloading. That is **adverse selection**, and no pricing accuracy removes it.
- **Costs usually dominate model edge on liquid products.** Spreads, market impact, fees, and hedging costs routinely exceed any model-vs-market residual on the products where the residual is measurable. The strategy layer — not the pricer — must model these, which is one reason it belongs in a separate system.
- **Pricing, strategy, and execution are three separate engineering problems.** VALAX is a strong answer to the first and a useful input to the risk side of the third (hedging). It is *not* an answer to "what is my edge, how big do I size, and how do I get filled". Keep those in the separate repo where they belong.
- **Calibration drift can masquerade as signal.** Intraday recalibration can jump; the resulting price change is model noise, not information. Monitor calibration stability (see [§ 5](#5-calibration-robustness-the-differentiable-inner-loop)) so the strategy layer never mistakes a recalibration artefact for a move.

The through-line: **make the engine the trusted, boring, well-tested valuation-and-risk core, and put all skepticism, edge estimation, and cost modelling in the separate strategy system that consumes it.** That division is what makes the whole thing safe.

---

## 8. Coverage today vs. roadmap

| Front-office need | Status | Component(s) |
|---|---|---|
| Differentiable pricer `V(instrument, market) → price` | ✅ | Every `valax/pricing/*` function |
| Autodiff Greeks (delta/vega/rho + gamma/vanna/volga) | ✅ | `jax.grad` / `jax.hessian`, `valax/greeks/` |
| Whole-universe batched revaluation | ✅ | `jax.vmap` + `@eqx.filter_jit` |
| GPU / TPU execution (same code) | ✅ | JAX / XLA |
| Intraday scenario / pre-trade "what-if" grids | ✅ | nested `vmap`, `apply_scenario`, `hypothetical_pnl_vector` |
| Robust differentiable calibration (Heston, SABR, SLV) | ✅ | `optimistix` LM + `optax`, `valax/calibration/` |
| Quote-level hedge Jacobian (risk in tradable instruments), via implicit diff through calibration | ✅ | `quote_jacobian` (`optimistix.ImplicitAdjoint`), `compute_ladder` |
| Calibration-stability monitoring (`vmap` robustness sweep) | ✅ | pattern via `jax.vmap` over perturbed quotes |
| Validation evidence (QuantLib cross-check, golden tests) | ✅ | `tests/test_quantlib_comparison/`, `tests/golden/` |
| Intraday `MarketState` build + calibration diagnostics | 🟡 | Designed in [Productionisation § 5](../architecture/production.md#5-layer-2-3-build-and-calibration-workflow); not fully packaged |
| Long-lived pricing **service** (gRPC/REST, resident state) | 🟡 | [Productionisation § 8](../architecture/production.md#8-layer-6-service-layer-deferred) — deferred |
| Streaming / incremental market-data updates | 📋 | [Roadmap Real-Time Risk](../roadmap.md#63-real-time-risk) — out of scope for the current design |
| JIT warm-up / shape-bucketing service harness | 📋 | Vision-tier; the productionisation prerequisite for quote-response latency |
| **Gradient-based intraday optimal hedging / quoting** (control shares the pricer graph) | 📋 | **Frontier — the §2.2 novelty.** Primitives shipped (`diffrax`, `optax`, `eqx.filter_grad`, differentiable payoffs); packaged product not built. See [Vision](../vision.md) |
| Strategy / signal generation | 🔴 | Out of scope by design — separate system |
| Order management / execution / fills | 🔴 | Out of scope by design — separate system |

The pattern mirrors every other application page: **the pricing, Greek, and calibration primitives are shipped and strong; the intraday *service harness* around them is the build project.**

---

## 9. The pitch, tailored to the buyer

**To the Desk Head / Trading Quant:**
*EOD repricing is table stakes — every bank has it. What you can't get from a C++ batch is this: one differentiable graph that hands your hedging and quoting logic the exact Jacobian of P&L with respect to the instruments you actually trade (and, at the frontier, with respect to your own hedge notionals and skews). One compiled kernel prices the whole universe and returns exact deltas, gammas, and vegas in the same call; warm-started calibration keeps intraday surfaces stable. Wire the prices, Greeks, and Jacobians into your own strategy and OMS; VALAX stays firmly on the valuation side of the line.*

**To the Head of Desk Technology:**
*The pricing core is a pure JAX function — `vmap` for the universe, `filter_jit` for speed, GPU with no code change. The work to make it an intraday service is a pre-warmed, shape-stable, resident process around that kernel (Productionisation § 8), not a rewrite of the analytics. You own that shell; the pricing math is done and validated against QuantLib.*

**To the CRO / Desk Risk:**
*Every price carries its exact sensitivities from the same computation graph, so hedge ratios and P&L-explain reconcile by construction. Calibration ships objective health metrics — fit RMSE, max error, Jacobian condition number, and a `vmap` stability sweep — so an unstable surface is caught intraday, not in tomorrow's P&L.*

**To the CIO / Sponsor deciding scope:**
*Adopt VALAX as the pricing-and-risk core, and keep strategy and execution as a separate system that consumes it. That boundary is deliberate: it keeps the valuation engine boring, testable, and reusable across desks — and it keeps the parts that actually lose money (edge estimation, adverse selection, transaction costs) where a skeptical, market-structure-aware team owns them.*

---

## 10. Where to read next

- **The applications overview and audience ranking** → [Applications](index.md).
- **The strongest technical-adoption companion (differentiable research)** → [Quant Research](quant-research.md).
- **The deep-hedging pattern behind the § 2.2 decision loop** → [Quant Research § 6](quant-research.md), [Vision](../vision.md).
- **The risk / hedging pipeline the Greeks feed** → [Risk: End-to-End](../risk-overview.md).
- **What it takes to make this a production service** → [Productionisation Design](../architecture/production.md).
- **Where VALAX sits in the surrounding bank / desk stack** → [Where VALAX Fits](../landscape.md).
- **Greeks, calibration, and vol-surface mechanics** → [Greeks](../guide/greeks.md), [Calibration](../guide/calibration.md), [Vol Surfaces](../guide/vol-surfaces.md).
- **The forward direction — real-time risk, service layer** → [Vision](../vision.md), [Roadmap](../roadmap.md).
