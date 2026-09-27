# Tutorial — The Unified Pricing Contract

Every VALAX trade — an equity call, a fixed-rate bond, a dual-curve swap, a
Bermudan swaption on a two-factor Gaussian model — is priced through **one
call**:

```python
price(instrument, model, market, *, method="auto", config=None, key=None)
```

This tutorial teaches that contract from the outside in: what the four
arguments mean, how the engine is chosen, how a trade names the curves it
discounts and projects on, how sensitivities come out of the same call, and
how to plug your own pricer in. Every snippet below was executed to produce
the numbers quoted in the text; copy them in order and you will reproduce
them.

| Stage | What you learn | Symbols |
|---|---|---|
| [1](#1-the-contract) | The contract: four nouns, one result | `price`, `PricingResult` |
| [2](#2-one-trade-four-engines) | One trade through analytic / lattice / PDE / MC | `method=`, `config=`, `key=` |
| [3](#3-how-the-engine-is-chosen) | Method availability vs preference | `available_methods`, `auto_method` |
| [4](#4-no-model-needed-curve-discounted-trades) | Curve-discounted trades | `model=None` |
| [5](#5-selectors-single-curve-for-free-multi-curve-on-purpose) | Selectors and resolution | `discount_id`, `forward_id`, `resolve` |
| [6](#6-many-assets-one-market) | Keyed spots and surfaces | `underlying_id`, surface alphabet |
| [7](#7-models-that-reference-the-market) | Binding a model to a market curve | `from_market`, `with_market_curve` |
| [8](#8-sensitivities-through-the-same-call) | Market / model / contract greeks | `market_greeks`, `curve_pillar_deltas` |
| [9](#9-building-the-market-from-quotes) | Quotes → curves → `MarketState` | `bootstrap_curve_graph` |
| [10](#10-extending-register-your-own-pricer) | Registering a new pricer | `register_pricer`, `set_auto_method` |
| [11](#11-failure-modes-and-what-they-mean) | Reading the error messages | — |

!!! tip "Enable float64 first"
    Curve and rates work is intolerant of single precision. Put
    `jax.config.update("jax_enable_x64", True)` at the top of any script
    before importing anything that builds arrays — every number on this page
    assumes it.

## 0. Setup

```python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from valax.core import (
    MarketState, PricingResult, auto_method, available_methods,
    price, register_pricer, registered_pricers, resolve,
)
from valax.curves import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates import ymd_to_ordinal
from valax.instruments import AmericanOption, EuropeanOption, FixedRateBond
from valax.instruments.rates import InterestRateSwap, Swaption
from valax.models.black_scholes import BlackScholesModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.mc import MCConfig
from valax.pricing.pde import PDEConfig

REF = jnp.int32(int(ymd_to_ordinal(2026, 3, 16)))


def flat_curve(rate, ref=REF, n_years=21):
    """A flat continuously-compounded curve on annual pillars."""
    days = jnp.arange(0, n_years * 365, 365, dtype=jnp.int32)
    return DiscountCurve(
        pillar_dates=ref + days,
        discount_factors=jnp.exp(-rate * days / 365.0),
        reference_date=ref,
        day_count="act_365",
    )
```

A flat curve keeps the arithmetic checkable. Stage 9 replaces it with a curve
bootstrapped from quotes, which is what you would do for real.

## 1. The contract

Four nouns go in, one result comes out.

| Argument | What it is | What it must **not** contain |
|---|---|---|
| `instrument` | Contract data — dates, strike, notional, conventions — plus static **market selectors** naming the curves/underlying *this trade* uses. | Market data. An instrument is data-only; it has no `.npv()`. |
| `model` | Dynamics choice + calibrated parameters (`BlackScholesModel`, `HullWhiteModel`, `HestonModel`, …), or `None`. | Numerical settings. Those are `config`. |
| `market` | A `MarketState`: a keyed `CurveGraph`, keyed vol surfaces, spots, dividends, fixings, credit/inflation/FX corners. | Contract or model parameters. |
| `method` / `config` / `key` | *How* to compute: engine name, its numerical settings, and a PRNG key for Monte Carlo. | Anything that changes the trade's value. |

The return value is always a `PricingResult` — `price`, the `method` actually
used, `stderr` (`None` for methods that have no sampling error), and optional
`diagnostics`:

```python
market = MarketState.single_curve(
    flat_curve(0.05), spot=jnp.array(100.0), vol=jnp.array(0.20)
)
model = BlackScholesModel(
    vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
)
call = EuropeanOption(
    strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True
)

result = price(call, model, market)
print(result.price, result.method, result.stderr)
# 10.450583572185579 analytic None
print(float(result))                # 10.450583572185579
```

Three things are worth internalising now:

1. **This trade named nothing.** No curve id, no underlying id, no engine, no
   config. `MarketState.single_curve` files everything under the `"_default_"`
   sentinel and a bare instrument resolves to exactly that, so the
   single-curve, single-asset case is configuration-free — and numerically
   identical to calling the low-level pricer directly.
2. **`float(result)` works**, so a `PricingResult` drops into scalar-expecting
   code without unpacking.
3. **`price` is a pure function** of `(instrument, model, market)`, which is
   what makes stage 8's sensitivities a single `jax.grad`.

!!! note "Where the boundary sits"
    `model` answers *how the world moves*; `market` answers *what the world is
    right now*; `config` answers *how hard we compute*. When you cannot decide
    where a number belongs, ask which of those three questions it answers — a
    Heston `v0` is a model parameter, today's SPX level is market, the number
    of MC paths is config.

## 2. One trade, four engines

Only `method` and `config` change below — the trade and the market are
untouched:

```python
res_a = price(call, model, market)                                  # analytic
res_l = price(call, model, market, method="lattice")                # CRR tree
res_p = price(call, model, market, method="pde", config=PDEConfig())
res_m = price(
    call, model, market,
    method="mc", config=MCConfig(n_paths=100_000, n_steps=50),
    key=jax.random.PRNGKey(0),
)
```

| `method` | Price | `stderr` | Notes |
|---|---|---|---|
| `"analytic"` (auto) | 10.450584 | `None` | Black-Scholes closed form |
| `"lattice"` | 10.440591 | `None` | default `BinomialConfig`; refine with `n_steps` |
| `"pde"` | 10.450648 | `None` | Crank-Nicolson, matches the closed form to ~6e-5 |
| `"mc"` | 10.521193 | 0.046521 | within 1.5 standard errors of 10.450584 |

That table is the cross-validation VALAX runs in its own test suite, written
as four lines of user code. The engines' internals — grids, Greeks stability,
variance reduction — are documented in the solver guides:
[Analytical](analytical.md), [Lattice](lattice.md), [PDE](pde.md),
[Monte Carlo](monte-carlo.md).

!!! warning "`method="mc"` requires a `key`"
    Randomness is explicit in JAX and VALAX never invents a seed for you.
    Omitting `key` raises `ValueError: method='mc' needs a PRNG key`. Pass the
    *same* key twice and you get the same number twice — common random numbers
    for free when you compare two markets.

## 3. How the engine is chosen

Two separate questions: what *can* price this pair, and what *should*.

```python
available_methods(call, model)   # ('analytic', 'lattice', 'pde', 'mc')
auto_method(call, model)         # 'analytic'

american = AmericanOption(
    strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False
)
auto_method(american, model)                      # 'lattice'
float(price(american, model, market).price)       # 6.086383
```

- **Availability** comes from the registry, keyed
  `(instrument_type, model_type, method)`. It includes every MC and PDE recipe
  in `valax/pricing/{mc,pde}/`, because those registries are *bridged* rather
  than duplicated — a recipe registered there is reachable through `price` the
  moment it lands.
- **Preference** comes from a separate policy table. `method="auto"` takes the
  policy entry when there is one and it is available, otherwise the first
  available method in `analytic → lattice → pde → mc` (exact and cheap first).
- An explicit `method=` always wins over the policy.

`registered_pricers()` lists every reachable triple — 88 of them at the time of
writing — which makes it a useful "what can this library price?" probe:

```python
len(registered_pricers())    # 88
registered_pricers()[:3]
# [('AmericanOption', 'BlackScholesModel', 'lattice'),
#  ('AmericanOption', 'BlackScholesModel', 'pde'),
#  ('AsianOption', 'BlackScholesModel', 'mc')]
```

For the curated, instrument-by-instrument view (with model inputs and
calibration routes), read the
[coverage matrix](../applications/instrument-pricing.md) — that page, not this
one, is the authoritative map of what is priced by what.

## 4. No model needed: curve-discounted trades

A fixed-rate bond has no dynamics: its value is discounted cashflows.
`model=None` is a first-class case, not a placeholder.

```python
rates_market = MarketState.single_curve(flat_curve(0.03))

bond = FixedRateBond(
    payment_dates=REF + jnp.array([365, 730, 1095, 1460, 1825], jnp.int32),
    settlement_date=REF,
    coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0),
    frequency=1,
)

float(price(bond, None, rates_market).price)   # 104.365876
available_methods(bond, None)                  # ('analytic',)
```

Vanilla swaps, OIS swaps and FRNs work the same way. A 4 % coupon discounted
at 3 % is worth a premium, as it should be. When you later want the *same*
bond under a short-rate model — for an embedded call, say — you pass the model
instead of `None` and the dispatch takes you to the PDE recipe; nothing about
the instrument changes. See [Fixed Income](fixed-income.md) and
[Callable Bonds](callable-bonds.md).

## 5. Selectors: single-curve for free, multi-curve on purpose

An instrument carries **contractual** selectors — static strings naming the
market pieces *this trade* uses:

| Selector | Meaning | Default |
|---|---|---|
| `discount_id` | Curve this trade discounts on | `"_default_"` |
| `forward_id` | Projection curve for floating legs | `None` — *project off the discount curve* |
| `underlying_id` | Asset the option is written on | `"_default_"` |

Models carry **modelling** references instead (which surface they read, which
curve they were calibrated to). Both kinds are `eqx.field(static=True)`
strings: they are bound at trace time by `resolve`, never traced, and never in
the gradient path.

Here is the same swap priced single-curve and dual-curve. The only difference
is two strings:

```python
ois, proj = flat_curve(0.030), flat_curve(0.035)
dates = REF + jnp.array([365, 730, 1095, 1460, 1825], jnp.int32)

swap_single = InterestRateSwap(
    start_date=REF, fixed_dates=dates, fixed_rate=jnp.array(0.032),
    notional=jnp.array(10_000_000.0), day_count="act_365",
)
swap_dual = InterestRateSwap(
    start_date=REF, fixed_dates=dates, fixed_rate=jnp.array(0.032),
    notional=jnp.array(10_000_000.0), day_count="act_365",
    discount_id="USD.SOFR.OIS",      # discount on OIS
    forward_id="USD.SOFR.3M",        # project 3M forwards
)

single_market = MarketState.single_curve(ois)
graph_market = MarketState(
    curves=CurveGraph(curves={"USD.SOFR.OIS": ois, "USD.SOFR.3M": proj})
)

float(price(swap_single, None, single_market).price)   #  -70,686.06
float(price(swap_dual,   None, graph_market).price)    #  165,557.14
```

The receiver-of-3M pays 3.2 % fixed. Projecting off a curve 50 bp above OIS
lifts the floating leg and turns the NPV positive — the entire economics of
the basis, expressed by naming a second curve.

You can inspect the binding directly, which is the fastest way to debug a
"wrong curve" suspicion:

```python
resolve(swap_dual, None, graph_market).forward_curve    # the 3M curve
resolve(swap_single, None, single_market).forward_curve # None ⇒ single-curve
```

!!! note "`forward_curve is None` is a meaning, not a missing value"
    `None` is the signal every VALAX rates pricer already reads as "project off
    the discount curve". It is a *trace-time* branch, so a single-curve trade
    runs the pre-existing single-curve code path and reproduces the legacy
    price bit-for-bit. Multi-curve is opt-in; it is never imposed on you.

Curve ids follow the frozen alphabet `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]`
(`USD.SOFR.OIS`, `USD.LIBOR.3M`). Why the discount/projection split matters
economically is [Why Multi-Curve?](why-multicurve.md); how the graph is solved
is [Curves](curves.md#5-multi-curve-bootstrap).

## 6. Many assets, one market

Scale past one asset by keying spots, dividends and surfaces, then naming the
underlying on the trade:

```python
from valax.surfaces.constant import ConstantVol

equities = MarketState(
    curves=CurveGraph(curves={"_default_": flat_curve(0.05, n_years=6)}),
    spots={"SPX": jnp.array(5000.0), "AAPL": jnp.array(190.0)},
    dividends={"SPX": jnp.array(0.015), "AAPL": jnp.array(0.005)},
    surfaces={"SPX.VOL": ConstantVol(vol=jnp.array(0.18)),
              "AAPL.VOL": ConstantVol(vol=jnp.array(0.28))},
    asset_ids=("SPX", "AAPL"),
)

spx_call = EuropeanOption(
    strike=jnp.array(5100.0), expiry=jnp.array(1.0), underlying_id="SPX"
)
aapl_call = EuropeanOption(
    strike=jnp.array(200.0), expiry=jnp.array(0.5), underlying_id="AAPL"
)

spx_model = BlackScholesModel(
    vol=jnp.array(0.18), rate=jnp.array(0.05), dividend=jnp.array(0.015)
)
aapl_model = BlackScholesModel(
    vol=jnp.array(0.28), rate=jnp.array(0.05), dividend=jnp.array(0.005)
)

float(price(spx_call, spx_model, equities).price)    # 388.951996
float(price(aapl_call, aapl_model, equities).price)  #  12.555190
```

Surface ids are validated at construction against
`<UNDERLYING>.<KIND>[.<QUALIFIER>]` with `KIND` one of `VOL`, `LV`,
`LEVERAGE`, `CUBE`, `OPTIONLET`:

```python
MarketState(
    curves=CurveGraph(curves={"_default_": flat_curve(0.05, n_years=6)}),
    surfaces={"spx vol": ConstantVol(vol=jnp.array(0.2))},
)
# ValueError: Invalid surface identifier 'spx vol'. Expected the pattern
# <UNDERLYING>.<KIND>[.<QUALIFIER>] with KIND in ('VOL', 'LV', 'LEVERAGE',
# 'CUBE', 'OPTIONLET') (e.g. 'SPX.VOL', 'EURUSD.VOL', 'USD.SOFR.CUBE') …
```

Failing at construction rather than at price time is deliberate: a typo'd
surface key that silently resolves to the wrong smile is the expensive kind of
bug. Surfaces themselves stay duck-typed — grids, SVI, SABR, leverage grids,
swaption cubes and `ConstantVol` all coexist in the same container
([Volatility Surfaces](vol-surfaces.md)).

!!! tip "Coming from `MarketData`?"
    `MarketState.from_market_data(md, asset_ids=…)` and
    `state.to_market_data()` are a lossless two-way view for everything
    `MarketData` can express, so the risk/VaR framework keeps working while
    you migrate ([API → Market Data](../api/market.md)).

## 7. Models that reference the market

A model is *calibrated to* a market, so the two are not freely combinable. The
consistency rule is worth stating plainly:

> A model is only valid paired with the `MarketState` it was calibrated
> against (or a scenario-shocked version reusing the same ids).

Short-rate models therefore record the curve they are bound to, and can be
built from — and re-bound to — a market:

```python
hw = HullWhiteModel.from_market(
    graph_market,
    mean_reversion=jnp.array(0.10),
    volatility=jnp.array(0.01),
    curve_id="USD.SOFR.OIS",
)
hw.curve_id                  # 'USD.SOFR.OIS'

shocked = MarketState(curves=CurveGraph(curves={
    "USD.SOFR.OIS": flat_curve(0.04), "USD.SOFR.3M": flat_curve(0.045),
}))
hw_shocked = hw.with_market_curve(shocked)   # same params, re-fitted curve
```

`from_market` records the linkage; `with_market_curve` re-binds under a shocked
market while preserving the intrinsic parameters (mean reversion, vol). That is
the pattern for scenario repricing: shock the `MarketState`, re-bind, reprice.

The selectors of stage 5 carry all the way into the MC and PDE engines. A
payer swaption on the graph market, discounting OIS and projecting 3M:

```python
payer = Swaption(
    expiry_date=REF + jnp.int32(2 * 365),
    fixed_dates=REF + jnp.arange(3 * 365, 8 * 365 + 1, 365, dtype=jnp.int32),
    strike=jnp.array(0.032),
    notional=jnp.array(10_000_000.0),
    is_payer=True,
    day_count="act_365",
    discount_id="USD.SOFR.OIS",
    forward_id="USD.SOFR.3M",
)
cfg = PDEConfig(n_spot=301, n_time=300, spot_range=6.0)

float(price(payer, hw, graph_market, method="pde", config=cfg).price)
# 309,644.49   — dual-curve

# …the same trade with no selectors, on a single-curve market:
# 167,187.75
```

The dual-curve price *exceeds* both single-curve legs here: with projection
above OIS, the trade keeps the higher OIS discount factors **and** gains the
higher projected forwards, so both effects push the same way. (A useful
sanity-check habit: predict the sign of a basis change before you run it.)

!!! warning "The tenor basis is deterministic, and that has consequences"
    Hull-White and G2++ are exact-fit-to-*one*-curve models, so multi-curve
    here means a **frozen** tenor-basis overlay: discount on the resolved
    curve, project off the forward curve, treat the basis as a deterministic
    spread. A basis swaption or tenor-spread option is therefore worth **zero**
    under this overlay, and the CMS-spread recipes *reject* a distinct
    projection curve rather than silently ignore it. Cheyette and LMM recipes
    remain single-curve. The modelling choice and its limits are derived in
    [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md); the engine
    recipes are in [PDE](pde.md) and [Monte Carlo](monte-carlo.md).

## 8. Sensitivities through the same call

Because `price` is pure, every Greek is one backward pass through it. Three
families, all first-class:

| Family | Differentiate w.r.t. | Examples |
|---|---|---|
| **Market** | `MarketState` leaves | delta, vega, curve key-rate deltas |
| **Model** | model parameters | ∂/∂v0, ∂/∂κ, ∂/∂leverage |
| **Contract** | instrument leaves | theta (∂/∂expiry), dual-delta (∂/∂strike) |

```python
from valax.greeks.market_greeks import (
    bucketed_curve_deltas, curve_pillar_deltas, dual_delta,
    market_gradient, market_greeks, model_gradient, spot_delta, theta,
)

market_greeks(call, model, market)
# {'price':        10.450584,
#  'delta':         0.636831,
#  'vega':         37.524035,
#  'rho':          53.232482,
#  'dividend_rho': -63.683065,
#  'theta':        -6.414028}

float(dual_delta(call, model, market))    # -0.532325  (∂price/∂strike)
```

The raw gradient is a pytree shaped like its input, which is what makes an
all-at-once risk pass cheap:

```python
grad = market_gradient(call, model, market)     # a MarketState of derivatives
float(grad.spots["_default_"])                  # 0.636831  — delta
grad.curves["_default_"].discount_factors.shape # (21,)     — per-pillar
float(model_gradient(call, model, market).vol)  # 37.524035 — vega
```

### Theta: autodiff where it is valid, a bump where it is not

`theta` is not one mechanism but a policy, because `expiry` is not always a
smooth number:

- **analytic / lattice** — `expiry` is a smooth year-fraction float, so
  `theta = -∂V/∂T` by exact autodiff → `-6.414028`.
- **PDE / MC** — `expiry` also fixes the time grid (a static discretisation you
  cannot differentiate through), so a finite-difference calendar bump is used
  → `-6.417010` for the same option.
- **Rates instruments** — dates are integer ordinals, so "theta" is a
  valuation-date *roll*, not a derivative. `theta` raises
  `NotImplementedError` rather than return a plausible wrong number.

```python
float(theta(call, model, market))                                   # -6.414028
float(theta(call, model, market, method="pde", config=PDEConfig()))  # -6.417010
```

### Curve risk: pillars → buckets → quotes

`curve_pillar_deltas` is the raw `∂price/∂DF(pillar)` vector. For the 5-year
10 mm swap of stage 5 on a flat 3 % curve:

```python
pillars = curve_pillar_deltas(swap_single, None, single_market)
# [+10,000,000, -320,000, -320,000, -320,000, -320,000, -10,320,000, 0, …]
```

Readable by hand, which is the point: `+10 mm` is the floating leg's dependence
on the start-date discount factor, `-320 k` is each 3.2 % coupon on 10 mm, and
`-10.32 mm` is the final coupon plus notional. Aggregate onto FRTB-style tenor
vertices with a `BucketMap`:

```python
from valax.risk.bucketing import tenor_bucket_map

pillar_times = jnp.arange(0, 21 * 365, 365) / 365.0
bmap = tenor_bucket_map(
    pillar_times, jnp.array([1.0, 2.0, 3.0, 5.0, 10.0]), weight="indicator"
)
bucketed_curve_deltas(pillars, bmap)
# [9,680,000, -320,000, -640,000, -10,320,000, 0]
```

With an indicator map the total is conserved (`sum(buckets) == sum(pillars)`),
which is the invariant to assert in your own tests. Bucketing is just a linear
coordinate change on the raw gradient — PCA and **quote space** are siblings of
the same operation, not separate calculations
([Risk Bucketing](../risk-bucketing.md)). Quote-space risk needs the
bootstrap's Jacobian as well as the pricer's, so it waits for
[stage 9](#from-quotes-to-hedge-ratios), where the curve is built from quotes.

### One pass, many Greeks; and it all jits

```python
@eqx.filter_jit
def npv(market, model):
    return price(call, model, market).price

float(npv(market, model))          # 10.450584


def value(pair):
    market, model = pair
    return price(call, model, market).price

d_market, d_model = eqx.filter_grad(value)((market, model))
float(d_market.spots["_default_"]), float(d_model.vol)   # (0.636831, 37.524035)
```

Delta and vega from a *single* backward pass. Use `eqx.filter_jit` /
`eqx.filter_grad` (not raw `jax.jit`) because instruments and models carry
static fields. Full reference: [API → Greeks](../api/greeks.md).

## 9. Building the market from quotes

Real markets are not flat curves. The build verb is the bootstrap, and its
output drops straight into `MarketState`:

```python
from valax.curves import DepositRate, SwapRate
from valax.curves.bootstrap_graph import bootstrap_curve_graph
from valax.curves.graph import CurveSpec

pillars = REF + jnp.array([182, 365, 730, 1825], dtype=jnp.int32)
spec = CurveSpec(curve_id="USD.SOFR.OIS", currency="USD", pillar_dates=pillars)
quotes = [
    DepositRate(start_date=REF, end_date=pillars[0], rate=jnp.array(0.0300),
                curves_touched=("USD.SOFR.OIS",)),
    DepositRate(start_date=REF, end_date=pillars[1], rate=jnp.array(0.0305),
                curves_touched=("USD.SOFR.OIS",)),
    SwapRate(start_date=REF, fixed_dates=pillars[1:3], rate=jnp.array(0.0310),
             curves_touched=("USD.SOFR.OIS",)),
    SwapRate(start_date=REF,
             fixed_dates=REF + jnp.array([365, 730, 1095, 1460, 1825], jnp.int32),
             rate=jnp.array(0.0320), curves_touched=("USD.SOFR.OIS",)),
]

graph, diag = bootstrap_curve_graph(REF, [spec], quotes)
bool(diag.converged), float(diag.max_abs_residual)   # (True, 2.8e-17)

market = MarketState(curves=graph)

bond_ois = FixedRateBond(
    payment_dates=REF + jnp.array([365, 730, 1095, 1460, 1825], jnp.int32),
    settlement_date=REF, coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0), frequency=1,
    discount_id="USD.SOFR.OIS",
)
float(price(bond_ois, None, market).price)     # 103.439757
```

### From quotes to hedge ratios

Now that the curve came *from* quotes, there is a Jacobian between them, and
stage 8's pillar deltas can be pushed into the space a desk actually trades:
`∂price/∂quote = J_buildᵀ · ∇price`.

```python
from valax.curves.bootstrap_graph import quote_jacobian

J = quote_jacobian(REF, [spec], quotes, by="df")     # (n_pillars, n_quotes)
pillar_deltas = curve_pillar_deltas(bond_ois, None, market,
                                    curve_id="USD.SOFR.OIS")
quote_deltas = J.T @ pillar_deltas[1:]
# [0.0, -0.6356, -2.6041, -468.272]
```

(`[1:]` drops the reference-date pillar, whose discount factor is 1 by
construction and which `quote_jacobian` therefore omits.) Read the result as
hedge ratios: the 5-year bond has **zero** sensitivity to the 6-month deposit
quote, and −468 per unit of the 5-year swap rate — about −4.7 cents per basis
point. Because the bootstrap is differentiated by implicit adjoint, this costs
one linear solve, not a re-bootstrap per quote.

The full pipeline, then, is four verbs:

```text
quotes ──build──▶ MarketState ──calibrate──▶ model ──price──▶ PricingResult
                        │                                          │
                        └────────────── greeks ─────────────────────┘
                                          │
                                   bucket / quote-space
```

`build` is [Curves](curves.md) (single- and multi-curve) plus
[Vol Surfaces](vol-surfaces.md); `calibrate` is
[Calibration](calibration.md) and returns `(model, solution)`; `price` and
`greeks` are this page. For a full worked rates pipeline on synthetic data —
including calibration diagnostics and residual analysis — see
[Tutorial — Rates End-to-End](tutorial-rates.md).

## 10. Extending: register your own pricer

A pricer is registered against `(instrument, model, method)`. Adapters are
meant to be thin — unpack the `ResolvedMarket`, call the numerics:

```python
class ZeroCouponNote(eqx.Module):
    maturity_date: jnp.ndarray
    face_value: jnp.ndarray
    discount_id: str = eqx.field(static=True, default="_default_")


@register_pricer(ZeroCouponNote, type(None), "analytic")
def _price_zcn(*, instrument, model, market, resolved, config, key):
    return PricingResult(
        price=instrument.face_value
        * resolved.discount_curve(instrument.maturity_date),
        method="analytic",
    )


note = ZeroCouponNote(
    maturity_date=REF + jnp.int32(5 * 365), face_value=jnp.array(100.0)
)
float(price(note, None, rates_market).price)    # 86.070798
```

Rules that keep the registry honest:

- The adapter signature is fixed: keyword-only `instrument`, `model`, `market`,
  `resolved`, `config`, `key`. Ignore what you do not need (`del model, key`).
- Take market data from `resolved`, never from raw ids — that is what makes
  your pricer work on a single-curve *and* a multi-curve market unchanged. Note
  `ZeroCouponNote` above became multi-curve-capable purely by declaring
  `discount_id`.
- Registering a duplicate raises unless you pass `overwrite=True`; state a
  preferred engine with `set_auto_method(MyInstrument, MyModel, "pde")`.
- **Subclasses inherit pricers.** Resolution walks the MRO, so a subclass that
  only *adds* selectors reaches its base class's pricers, and a registration on
  the subclass still wins.
- Growth belongs in the dispatch, never in the signature. If you find yourself
  wanting a new `price(...)` argument, you probably want a `config` field or a
  new `method`.

New numerics also need a home in the docs: add the row to the
[coverage matrix](../applications/instrument-pricing.md), the recipe to the
owning guide, and the symbol to the roadmap — see `AGENTS.md`.

## 11. Failure modes and what they mean

Every message below is raised eagerly, before any numerics, and names what is
available.

| What you did | What you get |
|---|---|
| `method="mc"` with no `key` | `ValueError: method='mc' needs a PRNG key: pass key=jax.random.PRNGKey(...).` |
| Misspelled the engine | `ValueError: Unknown pricing method 'quadrature'. Choose one of ('analytic', 'lattice', 'pde', 'mc') or 'auto'.` |
| Asked for an engine that pair has no recipe for | `ValueError: No mc recipe registered for (FixedRateBond, NoneType).` + the list of what is |
| Named a curve the market lacks | `KeyError: Curve 'USD.SOFR.OIS' not in MarketState (available: ['_default_']).` |
| Equity trade on a rates-only market | `ValueError: Black-Scholes pricing needs a spot: the MarketState carries none for this instrument's underlying. …` |
| Unregistered `(instrument, model)` pair | `ValueError: No pricer registered for (…)` + every registered triple |

Two more that are quieter but worth knowing:

- **Day-count mismatch.** Short-rate recipes build their time axis from the
  *instrument's* `day_count`, while a `DiscountCurve` uses its own. An
  `act_360` trade on an `act_365` curve differs by ~1.4 %. Keep conventions
  aligned (that is why every trade on this page says `act_365`).
- **A `MarketState` richer than `MarketData`.** `to_market_data()` raises
  rather than flatten a real smile into one scalar vol per asset.

## Where to go next

- [API → Core](../api/core.md) — full reference for `price`, `MarketState`,
  `resolve`, `PricingResult`, `register_pricer`.
- [API → Greeks](../api/greeks.md) — the unified sensitivities surface.
- [Coverage matrix](../applications/instrument-pricing.md) — what is priced by
  what, per instrument.
- [Tutorial — Rates End-to-End](tutorial-rates.md) — the build/calibrate half
  of the pipeline in full.
- [Scaling Up](../scaling/batching.md) — `vmap`, batching and JIT once the
  single-trade call is familiar.

!!! note "Design record"
    The rationale behind this contract — the conventions it replaced, the
    rejected alternatives, the phase-by-phase migration ledger, and the
    outstanding work — lives in `PRICING_ONTOLOGY.md` at the repository root.
    It is a contributor artefact, deliberately outside the documentation; this
    page is the user-facing contract.
