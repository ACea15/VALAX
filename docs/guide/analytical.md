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
