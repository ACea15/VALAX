# Analytical Pricing

This page covers the closed-form vanilla engines — Black-Scholes-Merton,
Black-76, Bachelier, SABR and Garman-Kohlhagen — and how to pick between them.
**Which instruments have an analytic pricer**, routed through `price()` (✅)
or called directly (◇), is stated once, in the *Analytic* column of the
[coverage matrix](../applications/instrument-pricing.md#coverage-matrix); an
exotic's closed form (digital, Margrabe / Kirk, Jamshidian, CMS, range
accrual, …) is documented on its instrument guide, and the Heston COS
semi-analytic pricer in [theory](../theory/stochastic-models.md) and
[API: Pricing](../api/pricing.md).

## Black-Scholes-Merton

For European options on equities with continuous dividends.

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where $d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$ and $d_2 = d_1 - \sigma\sqrt{T}$.

```python
from valax.instruments import EuropeanOption
from valax.pricing.analytic import black_scholes_price
import jax.numpy as jnp

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
price = black_scholes_price(option, jnp.array(100.0), jnp.array(0.20),
                             jnp.array(0.05), jnp.array(0.02))
```

**Arguments**: `(option, spot, vol, rate, dividend)`

### Implied Volatility

Newton-Raphson solver using autodiff for vega — no hand-coded derivative needed:

```python
from valax.pricing.analytic.black_scholes import black_scholes_implied_vol

iv = black_scholes_implied_vol(option, spot=jnp.array(100.0), rate=jnp.array(0.05),
                                dividend=jnp.array(0.02), market_price=jnp.array(9.23))
```

## Black-76

For European options on forwards and futures. No dividend yield — the forward price already accounts for carry.

$$C = e^{-rT} \left[ F N(d_1) - K N(d_2) \right]$$

```python
from valax.pricing.analytic import black76_price

price = black76_price(option, forward=jnp.array(102.0), vol=jnp.array(0.25),
                       rate=jnp.array(0.03))
```

**Arguments**: `(option, forward, vol, rate)`

## Bachelier (Normal Model)

For options where the underlying follows arithmetic Brownian motion ($dF = \sigma\, dW$). Common in interest rate markets where negative rates are possible.

$$C = e^{-rT} \sigma\sqrt{T} \left[ d\, N(d) + n(d) \right]$$

where $d = (F - K) / (\sigma\sqrt{T})$ and $\sigma$ is the **normal (absolute) volatility**.

```python
from valax.pricing.analytic import bachelier_price

# Normal vol is in absolute terms (e.g., 20 bps annualized)
price = bachelier_price(option, forward=jnp.array(100.0), vol=jnp.array(20.0),
                         rate=jnp.array(0.02))
```

**Arguments**: `(option, forward, vol, rate)`

## SABR Stochastic Volatility

The SABR model generates a volatility smile from four parameters ($\alpha$, $\beta$, $\rho$, $\nu$). Pricing uses Hagan's implied vol formula fed into Black-76.

$$dF = \alpha F^\beta\, dW_1, \quad d\alpha_t = \nu \alpha_t\, dW_2, \quad \text{Corr}(dW_1, dW_2) = \rho$$

```python
from valax.models import SABRModel
from valax.pricing.analytic import sabr_implied_vol, sabr_price

model = SABRModel(alpha=jnp.array(0.3), beta=jnp.array(0.5),
                  rho=jnp.array(-0.3), nu=jnp.array(0.4))

# Implied vol at a given strike
vol = sabr_implied_vol(model, forward=jnp.array(100.0),
                       strike=jnp.array(105.0), expiry=jnp.array(1.0))

# Full option price via Black-76
price = sabr_price(option, forward=jnp.array(100.0),
                   rate=jnp.array(0.05), model=model)
```

**Arguments**: `sabr_price(option, forward, rate, model)`

The implied vol formula is autodiff-safe — you can compute Greeks via `jax.grad` through the full SABR → Black-76 chain.

## Choosing a Model

| Model | Underlying | Vol type | Negative rates? |
|---|---|---|---|
| Black-Scholes | Equity spot | Lognormal | No |
| Black-76 | Forward/futures | Lognormal | No |
| Bachelier | Forward | Normal (absolute) | Yes |
| SABR | Forward | Stochastic (smile) | Depends on $\beta$ |
| Garman-Kohlhagen | FX spot | Lognormal | No |

---

## FX Options (Garman-Kohlhagen)

Garman-Kohlhagen is the standard model for FX vanilla options. It is algebraically identical to Black-Scholes with the **foreign risk-free rate** playing the role of the dividend yield.

### FX Forwards

The fair forward rate follows from covered interest rate parity:

```python
import jax.numpy as jnp
from valax.pricing.analytic.fx import fx_forward_rate, fx_forward_price
from valax.instruments.fx import FXForward

spot = jnp.array(1.10)          # EUR/USD
r_usd = jnp.array(0.05)        # domestic (USD) rate
r_eur = jnp.array(0.03)        # foreign (EUR) rate
expiry = jnp.array(0.5)        # 6 months

F = fx_forward_rate(spot, r_usd, r_eur, expiry)
# F ≈ 1.1110 (USD weakens because r_usd > r_eur)

# Price an FX forward contract
fwd = FXForward(
    strike=jnp.array(1.12),     # delivery rate
    expiry=expiry,
    notional_foreign=jnp.array(1e6),  # 1M EUR
    is_buy=True,                # buy EUR / sell USD
    currency_pair="EUR/USD",
)
npv = fx_forward_price(fwd, spot, r_usd, r_eur)
# NPV in USD (negative if strike > fair forward)
```

### FX Vanilla Options

```python
from valax.instruments.fx import FXVanillaOption
from valax.pricing.analytic.fx import garman_kohlhagen_price

vol = jnp.array(0.08)  # 8% implied vol

call = FXVanillaOption(
    strike=jnp.array(1.12),
    expiry=expiry,
    notional_foreign=jnp.array(1e6),
    is_call=True,
    currency_pair="EUR/USD",
)

price = garman_kohlhagen_price(call, spot, vol, r_usd, r_eur)
# Price in USD (domestic currency)
```

Put-call parity holds: $C - P = N \cdot (S \cdot e^{-r_f T} - K \cdot e^{-r_d T})$.

### FX Delta Conventions

FX markets use three delta conventions. This is one of the key differences from equity options:

```python
from valax.pricing.analytic.fx import fx_delta

# Three conventions give different numbers
spot_d = fx_delta(call, spot, vol, r_usd, r_eur, "spot")
fwd_d  = fx_delta(call, spot, vol, r_usd, r_eur, "forward")
pa_d   = fx_delta(call, spot, vol, r_usd, r_eur, "premium_adjusted")
# forward_delta > spot_delta > premium_adjusted_delta (for calls)
```

| Convention | Formula (call) | When to use |
|---|---|---|
| `"spot"` | $e^{-r_f T}\Phi(d_1)$ | G10 pairs (EUR/USD, USD/JPY) |
| `"forward"` | $\Phi(d_1)$ | Some interbank markets |
| `"premium_adjusted"` | $e^{-r_f T}\Phi(d_1) - V/(S \cdot N)$ | EM pairs (USD/BRL, USD/TRY) |

### Building a Smile from Delta Quotes

FX vol surfaces are quoted at standard delta points. Use `delta_to_strike` to convert delta quotes to strikes for surface construction:

```python
from valax.pricing.analytic.fx import delta_to_strike

# Market quotes: 25-delta call vol and 25-delta put vol
vol_25d_call = jnp.array(0.085)
vol_25d_put = jnp.array(0.090)

# Convert to strikes
K_25d_call = delta_to_strike(
    jnp.array(0.25), spot, vol_25d_call, r_usd, r_eur,
    expiry, is_call=True, convention="spot",
)
K_25d_put = delta_to_strike(
    jnp.array(-0.25), spot, vol_25d_put, r_usd, r_eur,
    expiry, is_call=False, convention="spot",
)
# K_25d_call > F > K_25d_put
```

### FX Greeks via Autodiff

All Greeks are free via `jax.grad`, including FX-specific sensitivities:

```python
import jax

# Delta (sensitivity to spot)
delta = jax.grad(lambda s: garman_kohlhagen_price(call, s, vol, r_usd, r_eur))(spot)

# Vega
vega = jax.grad(lambda v: garman_kohlhagen_price(call, spot, v, r_usd, r_eur))(vol)

# Domestic rho (positive for calls — higher r_d raises the forward)
rho_dom = jax.grad(lambda r: garman_kohlhagen_price(call, spot, vol, r, r_eur))(r_usd)

# Foreign rho (negative for calls — higher r_f lowers the forward)
rho_for = jax.grad(lambda r: garman_kohlhagen_price(call, spot, vol, r_usd, r))(r_eur)

# Gamma, vanna, volga — nested jax.grad as usual
gamma = jax.grad(jax.grad(
    lambda s: garman_kohlhagen_price(call, s, vol, r_usd, r_eur)
))(spot)
```

### Through `price()`

On a `MarketState`, `FXForward` (`model=None`) and `FXVanillaOption`
(`ImpliedVolModel(surface_id)`) take their inputs from the market by their
static selectors: the FX spot from `fx_spots` under `currency_pair` without
the `/`, the domestic rate off `discount_id` and the foreign rate off
`foreign_discount_id` (each the curve's zero rate to the expiry, so the
`exp(-r T)` discount factors are the curves' own), and — for the option — the
lognormal vol off the named surface at `(strike, expiry)`. `market_risk`
then reports both curves' pillar sensitivities, the FX delta and the vega
under their ids:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from valax.core import MarketState, price
from valax.curves.discount import flat_curve
from valax.curves.graph import CurveGraph
from valax.greeks import market_risk
from valax.instruments import FXVanillaOption
from valax.models import ImpliedVolModel
from valax.surfaces import ConstantVol

market = MarketState(
    curves=CurveGraph(curves={"USD.SOFR.OIS": flat_curve(0.05),
                              "EUR.ESTR.OIS": flat_curve(0.03)}),
    fx_spots={"EURUSD": jnp.asarray(1.10)},
    surfaces={"EURUSD.VOL": ConstantVol(jnp.asarray(0.08))},
)
call = FXVanillaOption(
    strike=jnp.array(1.12), expiry=jnp.array(0.5),
    notional_foreign=jnp.array(1_000_000.0), currency_pair="EUR/USD",
    discount_id="USD.SOFR.OIS", foreign_discount_id="EUR.ESTR.OIS",
)
model = ImpliedVolModel("EURUSD.VOL")
print(f"price (USD): {float(price(call, model, market).price):,.2f}")
risk = market_risk(call, model, market)
print(f"FX delta   : {float(risk.fx_spots['EURUSD']):,.0f}")
print(f"vega       : {float(risk.surfaces['EURUSD.VOL'].vol):,.0f}")
print(sorted(risk.curves))       # ['EUR.ESTR.OIS', 'USD.SOFR.OIS']
```

!!! note "FX vs Equity: key differences"
    - **Two interest rates**: domestic $r_d$ and foreign $r_f$ (equity has one rate + dividend yield)
    - **Delta conventions**: FX uses spot, forward, and premium-adjusted (equity uses only spot delta)
    - **Premium currency**: FX premiums can be in either currency, affecting the delta
    - **Vol surface quoting**: FX quotes in delta space (25Δ, 10Δ); equity quotes in strike space
    - **Domestic rho sign**: For FX calls, $\partial C / \partial r_d > 0$ (higher domestic rate raises the forward). This is opposite to the equity intuition where higher rates reduce call prices via discounting

---

## Short-Rate Swaptions (Jamshidian, Brigo–Mercurio)

The model-consistent European swaption closed forms — Jamshidian for
Hull-White (`hw_swaption_price`, `hw_zcb_option_price`) and the
Brigo–Mercurio one-dimensional integral for G2++ (`g2pp_swaption_price`) —
are the calibration workhorses for those models. The derivations are in
[Jamshidian Decomposition](../theory/hull-white-swaptions.md) and
[G2++ §3](../theory/g2pp.md#3-european-swaptions-brigomercurio-integral); the
signatures are in
[API → Pricing](../api/pricing.md#hullwhite-jamshidian).

**Time-dependent volatility.** Both accept a piecewise-constant volatility:
`HullWhiteModel(a, volatility=[...], volatility_times=[...])`, and
`G2PPModel(..., volatility_x=[...], volatility_ratio=[...], ...,
volatility_times=[...])` with second-factor volatility
$\eta(t) = \text{ratio}(t)\,\sigma(t)$ (read it back as the property
`model.volatility_y`; it is not a constructor argument). Only the variance
integrals change — they become exact sums over the buckets — so a one-bucket
model runs the original constant-parameter formula. See
[Time-Dependent Volatility](../theory/time-dependent-volatility.md).

**Root-finding guard: NaN, not a raise.** Each pricer nests a Newton solve —
the Jamshidian critical rate, the Brigo–Mercurio exercise boundary — that is
differentiated implicitly. Unguarded, an absurd or non-finite parameter made
the inner solve fail and the implicit derivative then hit a non-finite linear
system, which **raises at runtime** inside `jit` / `vmap` and aborts a whole
calibration or book batch for one bad trade. Both solves are now guarded: a
detached probe solve decides whether the solve succeeds, and the
differentiable solve runs on the real inputs when it does and on a benign
stand-in when it does not. A failed entry then prices to **NaN with a NaN
gradient**, and only that entry; on success the value and gradient are those
of the unguarded solve (`tests/test_pricing/test_root_guard.py`).

Measured overhead of the guard (price + gradient, median wall time):

| Workload | Hull-White (before → after) | G2++ (before → after) |
|---|---|---|
| single trade | 0.019 → 0.020 ms | 0.074 → 0.077 ms |
| vmapped basket of 10 | 0.020 → 0.052 ms | 0.17 → 0.21 ms |
| book of 1000 | 0.26 → 0.42 ms | 4.2 → 5.7 ms |
| compile time | +15–50% | +15–50% |

This was judged immaterial next to a whole batch aborting, so the guard is
always on and has no switch.

```python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models import G2PPModel, HullWhiteModel
from valax.pricing.analytic import g2pp_swaption_price, hw_swaption_price

ref = int(ymd_to_ordinal(2026, 1, 1))
pillars = jnp.array([ref + 365 * k for k in range(21)], dtype=jnp.int32)
curve = DiscountCurve(
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.035 * (pillars - ref) / 365.0),
    reference_date=jnp.int32(ref),
    day_count="act_365",
)
swaption = Swaption(                       # 2y x 5y payer
    expiry_date=jnp.int32(ref + 730),
    fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
    strike=jnp.array(0.035), notional=jnp.array(1.0), is_payer=True,
    day_count="act_365",
)
breaks = jnp.array([1.0, 3.0])
hw = bind_curves(
    HullWhiteModel(jnp.array(0.05), jnp.array([0.012, 0.008, 0.015]),
                   volatility_times=breaks),
    curve,
)
g2 = bind_curves(
    G2PPModel(
        mean_reversion_x=jnp.array(0.10), mean_reversion_y=jnp.array(0.50),
        volatility_x=jnp.array([0.010, 0.008, 0.012]),
        volatility_ratio=jnp.array(0.8),   # one bucket, broadcast to three
        correlation=jnp.array(-0.5), volatility_times=breaks,
    ),
    curve,
)
print("HW  sigma(t) Jamshidian:     ", float(eqx.filter_jit(hw_swaption_price)(swaption, hw)))
print("G2++ sigma(t) Brigo-Mercurio:", float(eqx.filter_jit(g2pp_swaption_price)(swaption, g2)))
print("G2++ eta(t) = ratio * sigma: ", g2.volatility_y)


# A batch with one absurd bucket: that entry is NaN, the others are untouched,
# and nothing raises inside jit / vmap.
@eqx.filter_jit
def price_and_vega(vols):
    def one(v):
        return hw_swaption_price(swaption, eqx.tree_at(lambda m: m.volatility, hw, v))
    return jax.vmap(jax.value_and_grad(one))(vols)


vols = jnp.array([[0.012, 0.008, 0.015], [0.012, 10.0, 0.015], [0.010, 0.010, 0.010]])
prices, vegas = price_and_vega(vols)
print("prices:", prices)
print("vega finite per trade:", jnp.all(jnp.isfinite(vegas), axis=1))
```

```text
HW  sigma(t) Jamshidian:      0.02227938010802243
G2++ sigma(t) Brigo-Mercurio: 0.01558790734400638
G2++ eta(t) = ratio * sigma:  [0.008  0.0064 0.0096]
prices: [0.02227938        nan 0.0220731 ]
vega finite per trade: [ True False  True]
```

Screen a batch with `jnp.isfinite(prices)` before aggregating: a NaN is the
signal that an entry's parameters are outside the region where the solve
converges.
