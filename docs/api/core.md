# Core — Market State & Unified Pricing

`valax.core` owns the two objects every other package leans on: the
**calibrated market container** (`MarketState`) and the **unified pricing
verb** (`price`). Together they are phase P2 of the
[Pricing Ontology](../architecture/pricing-ontology.md) — one call
convention replacing three, with *no pricer rewritten*.

```python
from valax.core import MarketState, price

market = MarketState.single_curve(curve, spot=spot, vol=vol)
result = price(option, model, market)          # method="auto"
result.price, result.method                    # 10.4506, "analytic"
```

## The unified signature

```python
price(
    instrument,          # contract data + static market selectors
    model,               # dynamics + calibrated params (None ⇒ pure discounting)
    market,              # MarketState
    *,
    method="auto",       # "analytic" | "lattice" | "pde" | "mc" | "auto"
    config=None,         # PDEConfig / MCConfig / BinomialConfig
    key=None,            # PRNG key, required for method="mc"
) -> PricingResult
```

Growth happens in the **dispatch**, never in the signature: new methods
and new `(instrument, model)` pairs register; `price`'s parameters do
not change.

Three properties worth knowing before you use it:

- **Single-curve needs zero configuration.** An instrument that declares
  no selectors resolves to the `"_default_"` curve with no projection
  curve — the pre-existing single-curve semantics, bit-for-bit.
- **The MC and PDE registries are bridged, not duplicated.** Every
  recipe in `valax/pricing/{mc,pde}/` is reachable through `price` the
  moment it is registered there, so the registries cannot drift apart.
- **`model=None` is a first-class case** for instruments priced by curve
  discounting alone (fixed-rate bonds, vanilla and OIS swaps, FRNs).

### Method selection

Availability and preference are separate (resolved question Q2 of the
ontology): the registry answers *what can be priced*, the policy table
answers *what should be*. `method="auto"` takes the policy entry when
there is one and it is available, else the first available method in
`analytic → lattice → pde → mc`.

::: valax.core.price

::: valax.core.PricingResult

::: valax.core.auto_method

::: valax.core.available_methods

::: valax.core.registered_pricers

### Registering a pricer

Adapters are thin: unpack the [`ResolvedMarket`](#valax.core.ResolvedMarket)
and call a pricer that already exists.

```python
from valax.core import PricingResult, register_pricer

@register_pricer(MyInstrument, MyModel, "analytic")
def _my_pricer(*, instrument, model, market, resolved, config, key):
    return PricingResult(
        price=my_closed_form(instrument, resolved.discount_curve, model),
        method="analytic",
    )
```

A subclass that only *adds* market selectors inherits its base class's
pricers — resolution walks the MRO, so the most specific registration
wins.

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

- [Pricing Ontology & Unified Signature](../architecture/pricing-ontology.md)
  — the design this implements, its invariants, and the remaining phases.
- [API → Pricing](pricing.md) — the pricers `price` dispatches to.
- [API → Market Data](market.md) — the legacy `MarketData` container and
  the synthetic-market generators.
- [API → Curves](curves.md) — `CurveGraph` and the joint bootstrap that
  builds it.
