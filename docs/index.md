# VALAX

**JAX-native quantitative finance valuation engine.**

VALAX is a quantitative finance library built entirely on JAX and its ecosystem. It provides option pricing, fixed income analytics, and risk computation with automatic differentiation for Greeks — no finite differences needed.

**The pricing engine runs on the same stack that builds frontier machine-learning models.** JAX, XLA, `equinox` and `optax` are what train large neural networks today, and they are the backend of VALAX's pricers, calibrators and risk. That has two consequences:

- **Pricing and ML can be blended.** A pricer, a calibration and a neural network are all differentiable functions on the same pytrees, so they compose in one program. Gradients flow through a price into a network's weights, or from a network's output into a calibrated model. You can train on pricing-model outputs and autodiff Greeks, or put a learned component inside a pricing pipeline, without a second ML stack or a bridge between two runtimes.
- **The engine is compatible with modern hardware.** Because pricing compiles through XLA, the same code targets CPUs, GPUs and TPUs. Batching is `vmap` and compilation is `jit`, the primitives accelerator workloads are built on, so scenario, portfolio and Monte Carlo workloads map onto accelerators without a rewrite.

The ML-native pieces already in the library are gradient-based calibration (`optax`, `optimistix`, `lineax`) and implicit differentiation through the calibration chain. Neural surrogate pricers, published GPU benchmarks and multi-device sharding are scheduled on the [Roadmap](roadmap.md).

For the long-term direction — how VALAX is intended to be used by trading desks, asset managers, and quant researchers — see [Vision](vision.md). For time-ordered feature work, see [Roadmap](roadmap.md). For unprioritized, uncommitted explorations that may or may not graduate into roadmap work, see [Research Ideas](research-ideas.md). For where VALAX sits among the twenty-plus other systems that surround a pricing engine on a real trading floor, and the architectural logic that ties everything together — what is actually hard about quant systems at scale and which pieces of that hardness VALAX is built to attack — see [Positioning & Design Rationale](positioning/index.md).

## Why this exists

The hardest part of building a bank-scale quant system isn't the maths in any individual component — it's keeping ~500 mutually-aware components consistent over twenty years while every one of them is being changed by a different team under regulatory pressure. VALAX is designed to drive the *internal* consistency cost as close to zero as a modern language and accelerator stack will allow. That frees a firm's quant budget for two things. The first is the consistency problems that genuinely require humans (reference data, model governance, regulatory approval), as opposed to the ones that don't (bump frameworks drifting out of sync with pricers, snapshot reconciliation between front-office and risk, audit replay). The second, and more valuable, is the work that actually differentiates a desk: research, alpha generation, new products and models, and shortening the path from idea to production. Every engineer-hour not spent reconciling two implementations of the same Greek can go into one of those instead.
The full argument — what actually sinks bank programmes, which choices in the library are targeted at which pain points, and what VALAX deliberately does *not* fix — is in [Positioning & Design Rationale](positioning/index.md).

## Key Features

- **Autodiff Greeks** — `jax.grad` gives exact delta, gamma, vega, vanna, volga, rho and more. Higher-order Greeks via nested differentiation.
- **Fixed income** — Discount curves, bond pricing, yield-to-maturity, and autodiff risk measures (duration, convexity, key-rate durations).
- **Multiple pricing methods** — Analytical (Black-Scholes, Black-76, Bachelier), Monte Carlo (GBM, Heston via diffrax), PDE (Crank-Nicolson), and lattice (CRR binomial with American exercise).
- **Volatility surfaces** — Grid, SABR, and SVI surface constructions with calibration. All differentiable for smile-aware Greeks.
- **Risk & scenarios** — Multi-curve shocks, parametric/historical scenario generation, VaR and Expected Shortfall via vmapped repricing.
- **Portfolio vectorization** — `jax.vmap` prices thousands of instruments in a single call.
- **Modern-hardware ready** — Pricing compiles through XLA, so the same code targets CPU, GPU and TPU with no changes.
- **Built on the ML stack** — Pricers, calibrations and risk share JAX, `equinox` and `optax` with frontier ML models. Differentiable end to end, so ML components and pricing models compose in one program. Neural surrogate pricers are on the [Roadmap](roadmap.md).
- **Pure functional** — No mutable state. Every data structure is a JAX pytree via equinox.

## Accuracy and validation

A valuation engine is only useful if its numbers can be trusted, so validation
is a first-class part of VALAX, not an afterthought. Every pricer is checked
against something *independent* of itself, in four ways:

| Check | What it proves | Representative evidence |
|---|---|---|
| **Analytic vs numerical** | The lattice, PDE and Monte Carlo engines converge to the closed form | Crank-Nicolson PDE matches Black-Scholes to ~6e-5 on an ATM call; Heston PDE vs the COS semi-analytic price to `rel < 1e-3`; Hull-White PDE vs Jamshidian to `abs < 1e-4`; Monte Carlo within `3 × stderr` of the analytic price |
| **Against QuantLib** | Agreement with the industry-standard open-source library | Black-Scholes prices and Greeks, SABR smiles and curve-bootstrap discount factors to `abs < 1e-10`; Hull-White and G2++ swaptions against Jamshidian / Brigo-Mercurio engines and independent QuantLib finite-difference engines |
| **Autodiff vs finite differences** | The Greeks are right, not just the prices | Autodiff delta, gamma, vega, rho and theta are compared with QuantLib's Greeks, and autodiff derivatives through PDE and calibration code are checked against finite differences (`rtol ≈ 1e-4`) |
| **Invariants and audit** | The properties a price must satisfy hold for *any* market, not just the test case | Put-call parity and monotonicity via `hypothesis`; an audit-conformance suite for multi-curve and adjoint risk; versioned golden datasets and deliberate arbitrage injection |

The comparison is not run on one hand-picked market. The QuantLib suite is
parametrised over **sampled synthetic markets**, so a convention that happens to
agree at the canonical ATM case still gets caught elsewhere. It is organised as
a three-stage pyramid: point pricers, then calibration outputs (curves, smiles),
then exotics priced on a calibrated surface. A failure at a higher stage with
the lower stages green therefore localises to the calibrator-to-pricer boundary.
That is the full quote → surface → exotic chain a desk actually runs.

The suite has found real problems. Comparing a calibrated Heston model against
QuantLib exposed a discretisation bias in the Monte Carlo scheme when the Feller
condition is violated, which is common in fitted parameters. It was fixed by
moving to Andersen's quadratic-exponential scheme, and the previously failing
cases now sit inside one standard error
([details](architecture/quantlib-validation-pyramid.md)).

Over 5,500 tests run in the suite, including 1,500+ QuantLib comparisons across
29 files and 330+ audit-conformance tests. Tolerances are set by comparison type
(machine precision for analytic vs analytic, a 3σ band for Monte Carlo), and the
policy is that any deviation carries a justification in the test's docstring.

Where no external reference exists, we say so rather than imply otherwise. The
Cheyette model has no QuantLib engine, so it is validated through its exact
Gaussian reduction (zero skew gives Hull-White and G2++), which *is* checked
against QuantLib. Inflation pricers and spread options have no QuantLib
comparison today, and risk ladders are validated against internal identities.
Performance benchmarks against QuantLib are not yet published. The current
picture is in
[Benchmarks and QuantLib Validation](benchmarks.md), the
[QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md), the
[Audit Conformance Map](architecture/aad-audit-conformance.md) and
[Reproducibility & Arbitrage Tests](guide/reproducibility_and_arbitrage_tests.md).

## Why JAX?

Traditional quant libraries compute Greeks via finite differences (bump-and-reprice) or hand-derived closed-form expressions. VALAX uses JAX's automatic differentiation instead:

| Approach | Effort | Accuracy | Speed |
|---|---|---|---|
| Finite differences | Low | O(h) error, tuning needed | 2N+1 evaluations for N Greeks |
| Closed-form | High (per model) | Exact | Fast, but limited to simple models |
| **Autodiff (VALAX)** | **Zero** | **Machine precision** | **One backward pass for all Greeks** |

This means every pricing function — analytical, Monte Carlo, PDE, or lattice — automatically supports Greeks with no additional code.

For the JAX idioms VALAX relies on — static trace-time dispatch, the double-where autodiff-safety pattern, `eqx.field(static=True)` semantics, JIT cache keys, autodiff-through-autodiff — see [JAX Patterns in VALAX](architecture/jax-patterns.md). Every pattern is anchored on a real file:line in the codebase so the doc and the implementation stay in sync.

## Quick Example

Start from **market quotes**, not a hand-built curve: bootstrap a SOFR curve from
deposits and swap rates, price a bond off it, and get the hedge ratios a desk
actually trades — the sensitivity to each *input quote* — through the same
differentiable pipeline.

```python
import jax
jax.config.update("jax_enable_x64", True)   # curve work needs float64

import jax.numpy as jnp
from valax.core import MarketState, price
from valax.curves import DepositRate, SwapRate
from valax.curves.bootstrap_graph import bootstrap_curve_graph
from valax.curves.graph import CurveSpec
from valax.dates import ymd_to_ordinal
from valax.greeks.adjoint import curve_adjoint, curve_quote_deltas
from valax.instruments import FixedRateBond

REF = jnp.int32(int(ymd_to_ordinal(2026, 3, 16)))
yrs = lambda *n: REF + jnp.array([365 * k for k in n], dtype=jnp.int32)

# 1. Market quotes -> bootstrapped curve (a Newton solve, itself differentiable)
spec = CurveSpec(curve_id="USD.SOFR.OIS", currency="USD", pillar_dates=yrs(1, 2, 3, 5))
quotes = [
    DepositRate(start_date=REF, end_date=yrs(1)[0], rate=jnp.array(0.0305),
                curves_touched=("USD.SOFR.OIS",)),
    SwapRate(start_date=REF, fixed_dates=yrs(1, 2), rate=jnp.array(0.0310),
             curves_touched=("USD.SOFR.OIS",)),
    SwapRate(start_date=REF, fixed_dates=yrs(1, 2, 3), rate=jnp.array(0.0315),
             curves_touched=("USD.SOFR.OIS",)),
    SwapRate(start_date=REF, fixed_dates=yrs(1, 2, 3, 4, 5), rate=jnp.array(0.0320),
             curves_touched=("USD.SOFR.OIS",)),
]
build = bootstrap_curve_graph(REF, [spec], quotes)
market = MarketState(curves=build.check().value)   # raises if the solve did not converge

# 2. Price a 5-year 4% bond through the one pricing contract
bond = FixedRateBond(
    payment_dates=yrs(1, 2, 3, 4, 5), settlement_date=REF,
    coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0), frequency=1,
    discount_id="USD.SOFR.OIS",
)
print(float(price(bond, None, market).price))        # 103.4399

# 3. Hedge ratios per input quote: one adjoint solve through the bootstrap,
#    no re-bootstrapping and no bumping
adj = curve_adjoint(market.curves, REF, [spec], quotes)
print(curve_quote_deltas(bond, None, market, adjoint=adj))
# [  -0.635   -1.290   -2.961  -466.649]   (1y depo, 2y, 3y, 5y swap)
```

The same `price(instrument, model, market)` call covers equity, FX, inflation,
credit and rates trades, and switches numerical engine with one argument. Here an
American put goes through a binomial lattice and a Crank-Nicolson PDE, and the
delta is taken from the same call:

```python
from valax.curves import DiscountCurve
from valax.greeks import market_risk
from valax.instruments import AmericanOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.pde import PDEConfig

days = jnp.arange(0, 21 * 365, 365, dtype=jnp.int32)
curve = DiscountCurve(pillar_dates=REF + days, discount_factors=jnp.exp(-0.05 * days / 365.0),
                      reference_date=REF, day_count="act_365")
mkt = MarketState.single_curve(curve, spot=jnp.array(100.0), vol=jnp.array(0.20))
model = BlackScholesModel(vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0))
put = AmericanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)

print(float(price(put, model, mkt, method="lattice").price))   # 6.0864
print(float(price(put, model, mkt, method="pde").price))       # 6.0895
risk = market_risk(put, model, mkt, method="pde", config=PDEConfig())
print(float(risk.spots["_default_"]))                           # -0.4110 (delta)
```

For the full contract — multi-curve selectors, calibrated models, Monte Carlo,
scenarios — see the [Unified Pricing Tutorial](guide/tutorial-unified-pricing.md).

## Examples

Runnable, self-checking Python scripts in `examples/` demonstrate the full library. Each file uses `# %%` cell markers for interactive IDE execution.

**Start with the unified workflow (10–22).** These run noisy synthetic quotes through a calibration chain, price through `price(instrument, model, market)`, and push risk back to the quotes with one adjoint sweep, reconciled against autodiff and finite differences.

| Example | What it shows |
|---------|---------------|
| `10_unified_market_tiers.py` | One option on three markets (flat, curve + SABR smile, + Heston) through the same workflow |
| `11_unified_multicurve_rates.py` | OIS + 3M joint bootstrap, swap book quote risk in one sweep |
| `12_unified_equity_smile_and_model.py` | Equity curve, SABR smiles, Heston; barrier / Asian Monte Carlo with quote risk |
| `13_unified_quote_scenarios.py` | Correlated quote scenarios, `vmap` recalibration + repricing, P&L explain, VaR / ES |
| `14`, `18` | Per-trade Hull–White, G2++ and Cheyette calibrated on each Bermudan's co-terminal basket; model risk between them |
| `15_unified_rates_caps_optionlet_chain.py` | Cap → caplet strip → optionlet SABR → caps and floors, with quote risk |
| `16_unified_calibration_diagnostics.py` | Bad quotes named by cell or bucket, conditioning alerts, engine provenance |
| `17_unified_slv_fixed_point.py` | Stochastic-local-vol leverage as a fixed point; barrier under LV / Heston / SLV |
| `19_unified_book_workflow.py` | Rates book end to end: bucketed DV01s, PCA scenarios, VaR / ES |
| `20_unified_fx_inflation_xccy.py` | FX, cross-currency, basis and inflation trades with declared-entry risk |
| `21_unified_rates_engine_ladder.py` | One swaption under every model and engine (Hull–White, G2++, Cheyette, LMM) |
| `22_unified_multi_asset_exotics.py` | Spread / worst-of and a barrier / Asian / lookback / variance-swap ladder across BS, Heston, LV, SLV |

**The kernel layer (01–06)** calls the pricing functions directly, a supported layer you can compose freely: equity options and autodiff Greeks (`01`), SABR smiles and calibration (`02`), curves, bonds and key-rate durations (`03`), caps, swaps and swaptions (`04`), Monte Carlo paths and exotics (`05`), and PDE and lattice pricing (`06`).

Also in the repository:

- `examples/comparisons/`: the same problem across methods, such as Heston smile, local-vol PDE, and Hull–White, G2++ and Cheyette MC vs tree vs PDE vs analytic.
- `examples/pitfalls/`: deliberately broken reproductions of subtle numerical traps, each with its symptom, root cause and way out.

See the [Examples](examples.md) page for the full index, per-example walkthroughs and code snippets, and [Two layers, both supported](examples.md#two-layers-both-supported) for how the two layers relate.
