# Core — Market State & Unified Pricing

`valax.core` owns the two objects every other package leans on: the
**calibrated market container** (`MarketState`) and the **unified pricing
verb** (`price`) — one call convention replacing three, with *no pricer
rewritten*.

!!! tip "Learning it rather than looking it up?"
    This page is the reference. The guided walkthrough — the contract, the
    selectors, worked multi-curve and sensitivity examples, and how to
    register your own pricer — is
    [Tutorial — The Unified Pricing Contract](../guide/tutorial-unified-pricing.md).

```python
from valax.core import MarketState, price

market = MarketState.single_curve(curve, spot=spot, vol=vol)
result = price(option, model, market)          # method="auto"
result.price, result.method                    # 10.4506, "analytic"
```

## The unified signature

Growth happens in the **dispatch**, never in the signature: new methods
and new `(instrument, model)` pairs register; `price`'s parameters do not
change. What the four arguments own, why single-curve trades need zero
configuration, and how `model=None` covers curve-discounted instruments are
walked through in
[the tutorial](../guide/tutorial-unified-pricing.md#1-the-contract).

### Method selection

Availability and preference are separate: the registry answers *what can be
priced*, the policy table answers *what should be*. `method="auto"` takes the
policy entry when there is one and it is available, else the first available
method in `analytic → lattice → pde → mc`.

::: valax.core.price

::: valax.core.PricingResult

::: valax.core.auto_method

::: valax.core.available_methods

::: valax.core.registered_pricers

### Registering a pricer

Adapters are thin: unpack the [`ResolvedMarket`](#valax.core.ResolvedMarket)
and call a pricer that already exists. A subclass that only *adds* market
selectors inherits its base class's pricers — resolution walks the MRO, so the
most specific registration wins. Worked example and the rules that keep the
registry honest:
[tutorial § 10](../guide/tutorial-unified-pricing.md#10-extending-register-your-own-pricer).

::: valax.core.register_pricer

::: valax.core.set_auto_method

## MarketState

The calibrated market: a [`CurveGraph`](curves.md#curve-graph), keyed vol
surfaces, spots, dividends, fixings, and the credit / inflation / FX
corners. It **generalises** [`MarketData`](market.md#core-market-state)
rather than renaming it — one curve becomes a keyed graph, one scalar vol
per asset becomes a dict of full surfaces — and keeps a two-way view onto
it for one deprecation cycle (`from_market_data` / `to_market_data`).

Surface identifiers follow the frozen alphabet
`<UNDERLYING>.<KIND>[.<QUALIFIER>]` with `KIND` one of `VOL`, `LV`,
`LEVERAGE`, `CUBE`, `OPTIONLET` (e.g. `SPX.VOL`, `EURUSD.VOL`,
`USD.SOFR.CUBE`), validated at construction. The `"_default_"` sentinel
is accepted everywhere, which is what makes the single-asset,
single-curve case configuration-free.

::: valax.core.MarketState

::: valax.core.validate_surface_id

## Resolution

`resolve` is the one place market identifiers become objects. Instruments
own **contractual** selectors (which curve *this trade* discounts on,
which index it projects, its underlying); models own **modelling**
references (which surface they read). Both are static strings, so
resolution happens at trace time and stays out of the gradient path.

::: valax.core.resolve

::: valax.core.ResolvedMarket

::: valax.core.resolve_curves

::: valax.core.resolve_spot

::: valax.core.resolve_dividend

::: valax.core.resolve_surface

## Model binding — constants or market-sourced

Pricing is four layers, and a model parameter may be supplied at either of
two of them:

| Layer | Example | Owns |
|---|---|---|
| **Kernel** | `black_scholes_price(option, spot, vol, rate, dividend)` | the closed form; pure scalars, no market |
| **Model** | `BlackScholesModel(vol=0.2, rate=0.05, dividend=0.0)` | calibrated parameters |
| **Binding** | `bind_model(model, instrument, resolved)` | market → scalar, the **only** place that conversion lives |
| **Front door** | `price(instrument, model, market)` | resolution, binding, dispatch |

A model whose parameters are all given is self-contained — the textbook
form, and what every closed-form derivation in
[Theory](../theory/foundations.md) assumes. Leaving one `None` sources it
from the `MarketState` instead:

```python
BlackScholesModel(vol=0.2, rate=0.05, dividend=0.0)  # all constant
BlackScholesModel(vol=0.2)                # rate + dividend from market
BlackScholesModel(rate=0.05)              # vol from the surface
BlackScholesModel()                       # everything from the market
```

A market-sourced `rate` is the continuously-compounded zero rate at **the
instrument's own maturity** (day-count aware, via the curve's own
`day_count`); a market-sourced `vol` is the surface sampled at
`(strike, expiry)`; a market-sourced `dividend` is the market's yield for
the underlying.

Three properties worth knowing:

- **Constant mode is bit-identical** to pricing before this layer existed.
  A fully-specified model is returned by `bind_model` unchanged — the same
  object, not a copy.
- **Market mode carries risk.** Binding happens *inside* the
  differentiated path, so a sourced parameter traces back to the curve or
  surface. That is what lets an equity book report curve pillar deltas
  rather than a structural zero — see
  [VaR & P&L](../risk-var.md#repricing-through-price).
- **It is still a constant-parameter model.** Projecting at each trade's
  own maturity makes its discount factor right for that maturity, so a
  book of mixed expiries uses different rates — correct per trade, but not
  a term-structure model.

::: valax.core.binding.bind_model

::: valax.core.binding.project_rate

::: valax.core.binding.sample_vol

::: valax.core.binding.instrument_horizon

## See also

- [Tutorial — The Unified Pricing Contract](../guide/tutorial-unified-pricing.md)
  — the same API as a walkthrough, with runnable examples and the failure
  modes explained.
- [API → Pricing](pricing.md) — the pricers `price` dispatches to.
- [API → Market Data](market.md) — the legacy `MarketData` container and
  the synthetic-market generators.
- [API → Curves](curves.md) — `CurveGraph` and the joint bootstrap that
  builds it.
