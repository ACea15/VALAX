# Pricing Functions

All pricing functions are pure functions with no side effects. They take an instrument and market data, and return a scalar price.

## Analytical

### `black_scholes_price`

```python
black_scholes_price(option, spot, vol, rate, dividend) -> Float[Array, ""]
```

Black-Scholes-Merton closed-form for European options on equities.

### `black76_price`

```python
black76_price(option, forward, vol, rate) -> Float[Array, ""]
```

Black-76 for European options on forwards/futures.

### `bachelier_price`

```python
bachelier_price(option, forward, vol, rate) -> Float[Array, ""]
```

Bachelier (normal) model. `vol` is absolute (normal) volatility.

### `black_scholes_implied_vol`

```python
black_scholes_implied_vol(option, spot, rate, dividend, market_price,
                           n_iterations=20) -> Float[Array, ""]
```

Newton-Raphson implied volatility using autodiff vega.

### Bond Pricing

#### `zero_coupon_bond_price`

```python
zero_coupon_bond_price(bond, curve) -> Float[Array, ""]
```

Price a zero-coupon bond from a discount curve. Returns `face_value * DF(maturity)`.

#### `fixed_rate_bond_price`

```python
fixed_rate_bond_price(bond, curve) -> Float[Array, ""]
```

Price a fixed-rate coupon bond by discounting each future coupon and the face value redemption using the curve.

#### `fixed_rate_bond_price_from_yield`

```python
fixed_rate_bond_price_from_yield(bond, ytm) -> Float[Array, ""]
```

Standard yield-based bond pricing: $P = \sum_i \frac{C}{(1+y/f)^i} + \frac{F}{(1+y/f)^n}$.

#### `yield_to_maturity`

```python
yield_to_maturity(bond, market_price, n_iterations=50) -> Float[Array, ""]
```

Newton-Raphson YTM solver using autodiff for the price-yield derivative.

#### `modified_duration`

```python
modified_duration(bond, ytm) -> Float[Array, ""]
```

$-\frac{1}{P}\frac{dP}{dy}$ computed via `jax.grad`.

#### `convexity`

```python
convexity(bond, ytm) -> Float[Array, ""]
```

$\frac{1}{P}\frac{d^2P}{dy^2}$ computed via nested `jax.grad`.

#### `key_rate_durations`

```python
key_rate_durations(bond, curve) -> Float[Array, "n_pillars"]
```

Sensitivity of bond price to each curve pillar's zero rate. One backward pass gives all sensitivities.

## Monte Carlo

### `mc_price`

```python
mc_price(option, spot, model, config, key, payoff_fn=european_payoff) -> Float[Array, ""]
```

Monte Carlo pricing. Dispatches path generation based on model type (`BlackScholesModel` or `HestonModel`).

### `mc_price_with_stderr`

```python
mc_price_with_stderr(...) -> tuple[Float[Array, ""], Float[Array, ""]]
```

Same as `mc_price` but also returns the standard error estimate.

### `MCConfig`

```python
MCConfig(n_paths: int, n_steps: int)
```

## PDE

### `pde_price`

```python
pde_price(option, spot, vol, rate, dividend, config=PDEConfig()) -> Float[Array, ""]
```

Crank-Nicolson finite difference solver in log-spot space.

### `PDEConfig`

```python
PDEConfig(n_spot: int = 200, n_time: int = 200, spot_range: float = 4.0)
```

## Lattice

### `binomial_price`

```python
binomial_price(option, spot, vol, rate, dividend, config=BinomialConfig()) -> Float[Array, ""]
```

CRR binomial tree. Supports both European and American exercise.

### `BinomialConfig`

```python
BinomialConfig(n_steps: int = 200, american: bool = False)
```
