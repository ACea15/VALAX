# VALAX

**JAX-native quantitative finance valuation engine.**

VALAX is a quantitative finance library built entirely on JAX and its ecosystem. One `price(instrument, model, market, config)` contract covers rates, equity, FX, inflation and credit trades across analytic, Monte Carlo, PDE and lattice engines. Curve building, calibration, pricing and Greeks run as one differentiable program.

**The pricing engine runs on the same stack that builds frontier machine-learning models.** JAX, XLA, `equinox` and `optax` train large neural networks today, and they are also the backend of VALAX's pricers, calibrators and risk. A pricer, a calibration and a neural network are therefore all differentiable functions on the same pytrees, and they compose in one program. Gradients flow through a price into a network's weights, or from a network's output into a calibrated model. You can train on pricing-model outputs and autodiff Greeks, or put a learned component inside a pricing pipeline, without a second ML stack or a bridge between two runtimes. The same compilation path puts pricing on GPUs and TPUs (see *Fast by construction* below).

The ML-native pieces already in the library are gradient-based calibration (`optax`, `optimistix`, `lineax`) and implicit differentiation through the calibration chain. Neural surrogate pricers, published GPU benchmarks and multi-device sharding are scheduled on the [Roadmap](roadmap.md).

## Why this exists

In a bank-scale quant system, the maths of any single model is rarely the hard part. The hard part is keeping many interdependent components consistent for years while different teams change them. In practice that means a bump-and-reprice risk framework drifting out of sync with the pricer, front-office and risk numbers that have to be reconciled, and past results that cannot be replayed for audit.

VALAX is designed so those problems do not arise inside the library:

- **One implementation per instrument.** Price, Greeks and scenario P&L all come from the same pricing function. Greeks are its derivatives, not the output of a separate bump framework that can drift.
- **Pure functions on immutable pytrees.** The same inputs always give the same number, so any result can be replayed.
- **Calibration in the same differentiable graph.** Sensitivities to market quotes flow back through the curve and model calibration, with no separate re-bootstrap to keep aligned.

Some consistency work still needs people (reference data, model governance, regulatory approval), and VALAX does not remove it. What it removes is the engineering time spent reconciling two implementations of the same Greek. That time can go into research, new products and models, and a shorter path from idea to production.

The full argument — what actually sinks bank programmes, which choices in the library target which pain points, and what VALAX deliberately does *not* fix — is in [Positioning & Design Rationale](positioning/index.md).

## Key Features

- **Autodiff Greeks** — `jax.grad` gives exact delta, gamma, vega, vanna, volga, rho and more, with no bump-size tuning. Higher-order Greeks via nested differentiation. This works for every engine (analytic, Monte Carlo, PDE and lattice) without per-pricer Greek code, and reverse mode returns all first-order sensitivities in one backward pass.
- **Fast by construction** — `jit` traces a pricer once and XLA compiles it into fused native kernels, so repeat calls skip Python entirely. `vmap` batches across instruments, scenarios and Monte Carlo paths, and the compiled program (including its gradients) runs on CPU, GPU or TPU. Published benchmarks are on the [Roadmap](roadmap.md).
- **Strong scientific libraries, not hand-rolled numerics** — SDE paths use `diffrax`, root-finding and least-squares use `optimistix`, structured linear solves use `lineax`, and gradient-based optimisation uses `optax`. These are maintained, tested and differentiable, and the calibrations and PDE/MC engines inherit that.
- **Four engine families** — analytic, Monte Carlo, PDE and lattice. Which instruments each one covers is in the [coverage matrix](applications/instrument-pricing.md).
    - **Analytic and semi-analytic:** Black-Scholes, Black-76, Bachelier, Garman-Kohlhagen, Heston (COS), SABR, Jamshidian (Hull-White) and Brigo-Mercurio (G2++) swaptions, Margrabe and Kirk spread options.
    - **Monte Carlo:** GBM, Heston (Andersen QE), SABR, local vol, SLV, correlated multi-asset, Hull-White, G2++, Cheyette 1F/2F and LMM, with Longstaff-Schwartz for Bermudan swaptions.
    - **PDE:** 1D Crank-Nicolson with Rannacher start-up and American exercise, and 2D ADI (Douglas, Craig-Sneyd, Hundsdorfer-Verwer) for Heston, G2++ and Cheyette 2F, on `lineax` tridiagonal solves.
    - **Lattice:** CRR binomial with American exercise, and Hull-White trinomial trees (including time-dependent volatility) for callable and puttable bonds and Bermudan swaptions.
- **Curves, surfaces and calibration** — joint multi-curve bootstrap, plus CDS survival and inflation curves. Vol surfaces include SABR, SVI, caplet strips, optionlet surfaces and swaption cubes. Heston, Hull-White, G2++, Cheyette and SLV models calibrate to market, including per-trade co-terminal baskets. Calibrations compose into a chain with implicit differentiation, so sensitivities to market quotes flow through every level. Each calibration's assumptions and limitations are in the [Calibration Register](applications/calibration-register.md).
- **Pure functional** — No mutable state. Every data structure is a JAX pytree via equinox. The JAX idioms this relies on, each anchored on real code, are in [JAX Patterns in VALAX](architecture/jax-patterns.md).

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
comparison today, and bucketed risk sensitivities are validated against internal identities.
Performance benchmarks against QuantLib are not yet published. The current
picture is in
[Benchmarks and QuantLib Validation](benchmarks.md), the
[QuantLib Validation Pyramid](architecture/quantlib-validation-pyramid.md), the
[Audit Conformance Map](architecture/aad-audit-conformance.md) and
[Reproducibility & Arbitrage Tests](guide/reproducibility_and_arbitrage_tests.md).

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

## Where to go next

- [Vision](vision.md): how VALAX is meant to be used by trading desks, asset managers and quant researchers.
- [Roadmap](roadmap.md): what is built and what is planned, tier by tier.
- [Research Ideas](research-ideas.md): uncommitted explorations that may or may not become roadmap work.
- [Positioning & Design Rationale](positioning/index.md): where VALAX sits among the other systems on a trading floor, and which parts of the hard problems it is built to solve.
