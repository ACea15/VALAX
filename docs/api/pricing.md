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
