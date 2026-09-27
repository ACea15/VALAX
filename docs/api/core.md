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

::: valax.core.resolve_surface

## See also

- [Tutorial — The Unified Pricing Contract](../guide/tutorial-unified-pricing.md)
  — the same API as a walkthrough, with runnable examples and the failure
  modes explained.
- [API → Pricing](pricing.md) — the pricers `price` dispatches to.
- [API → Market Data](market.md) — the legacy `MarketData` container and
  the synthetic-market generators.
- [API → Curves](curves.md) — `CurveGraph` and the joint bootstrap that
  builds it.
