# Analytical Pricing

VALAX provides three closed-form pricing models for European options.

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

!!! note "FX vs Equity: key differences"
    - **Two interest rates**: domestic $r_d$ and foreign $r_f$ (equity has one rate + dividend yield)
    - **Delta conventions**: FX uses spot, forward, and premium-adjusted (equity uses only spot delta)
    - **Premium currency**: FX premiums can be in either currency, affecting the delta
    - **Vol surface quoting**: FX quotes in delta space (25Δ, 10Δ); equity quotes in strike space
    - **Domestic rho sign**: For FX calls, $\partial C / \partial r_d > 0$ (higher domestic rate raises the forward). This is opposite to the equity intuition where higher rates reduce call prices via discounting
