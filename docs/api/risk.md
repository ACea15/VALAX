# Risk

Scenario generation, shock application, and risk measures.

## Market Data

### `MarketData`

```python
class MarketData(eqx.Module):
    spots: Float[Array, "n_assets"]
    vols: Float[Array, "n_assets"]
    dividends: Float[Array, "n_assets"]
    discount_curve: DiscountCurve
```

Complete market state. All fields are differentiable.

### `MarketScenario`

```python
class MarketScenario(eqx.Module):
    spot_shocks: Float[Array, "n_assets"]
    vol_shocks: Float[Array, "n_assets"]
    rate_shocks: Float[Array, "n_pillars"]
    dividend_shocks: Float[Array, "n_assets"]
    multiplicative: bool = False  # static
```

Additive risk factor changes. When `multiplicative=True`, spot shocks are returns.

### `ScenarioSet`

Same fields as `MarketScenario` but with a leading `(n_scenarios, ...)` axis on every leaf. Designed for `jax.vmap`.

### `zero_scenario`

```python
zero_scenario(n_assets, n_pillars) -> MarketScenario
```

No-op scenario (all shocks zero).

### `stack_scenarios`

```python
stack_scenarios(scenarios: list[MarketScenario]) -> ScenarioSet
```

Stack a list of single scenarios into a batched `ScenarioSet`.

---

## Shock Application

### `bump_curve_zero_rates`

```python
bump_curve_zero_rates(curve, rate_bumps) -> DiscountCurve
```

Apply additive zero-rate bumps at each pillar: $DF_{\text{new}}(t_i) = DF_{\text{old}}(t_i) \cdot e^{-\Delta r_i \cdot t_i}$.

### `parallel_shift`

```python
parallel_shift(curve, bump) -> DiscountCurve
```

Uniform bump to all pillar zero rates.

### `key_rate_bump`

```python
key_rate_bump(curve, pillar_index, bump) -> DiscountCurve
```

Bump a single pillar's zero rate.

### `apply_scenario`

```python
apply_scenario(base: MarketData, scenario: MarketScenario) -> MarketData
```

Apply all shocks (spot, vol, rate, dividend) to produce a new market state. Fully differentiable.

---

## Scenario Generation

### `parametric_scenarios`

```python
parametric_scenarios(key, cov, n_scenarios, n_assets, n_pillars,
                     distribution="normal", df=5.0) -> ScenarioSet
```

Correlated samples via Cholesky decomposition. Column ordering in `cov`: `[spots, vols, rates, dividends]`. Supports `"normal"` and `"t"` distributions.

### `historical_scenarios`

```python
historical_scenarios(returns, n_assets, n_pillars) -> ScenarioSet
```

Slice observed daily changes (shape `(n_obs, n_factors)`) into a `ScenarioSet`.

### `stress_scenario`

```python
stress_scenario(n_assets, n_pillars, spot_shock=0.0, vol_shock=0.0,
                parallel_rate_shift=0.0, rate_shocks=None,
                dividend_shock=0.0) -> MarketScenario
```

Deterministic stress with optional custom per-pillar rate shocks.

### `steepener`

```python
steepener(n_assets, n_pillars, short_bump, long_bump) -> MarketScenario
```

Linear rate profile from `short_bump` to `long_bump`.

### `flattener`

```python
flattener(n_assets, n_pillars, short_bump, long_bump) -> MarketScenario
```

Linear rate profile (typically short up, long down).

### `butterfly`

```python
butterfly(n_assets, n_pillars, wing_bump, belly_bump) -> MarketScenario
```

Quadratic rate profile: wings at `wing_bump`, belly at `belly_bump`.

---

## Risk Measures

### `portfolio_pnl`

```python
portfolio_pnl(pricing_fn, instruments, base, scenarios) -> Float[Array, "n_scenarios"]
```

Compute P&L under each scenario via `jax.vmap`. Returns `portfolio_value(scenario_i) - portfolio_value(base)`.

### `reprice_under_scenario`

```python
reprice_under_scenario(pricing_fn, instruments, base, scenario) -> Float[Array, ""]
```

Reprice a portfolio under a single scenario. Applies the scenario, extracts market args, prices via `jax.vmap` over instruments, and sums.

### `value_at_risk`

```python
value_at_risk(pnl, confidence=0.99) -> Float[Array, ""]
```

VaR: negative of the $(1 - \alpha)$ quantile. Positive VaR indicates a loss threshold.

### `expected_shortfall`

```python
expected_shortfall(pnl, confidence=0.99) -> Float[Array, ""]
```

CVaR: mean of losses beyond VaR. Always $\geq$ VaR.
