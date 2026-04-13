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

---

## FX Pricing

### `fx_forward_rate`

```python
fx_forward_rate(spot, r_domestic, r_foreign, T) -> Float[Array, ""]
```

Covered-interest-rate-parity forward rate: $F = S \cdot e^{(r_d - r_f) T}$.

### `fx_forward_price`

```python
fx_forward_price(forward: FXForward, spot, r_domestic, r_foreign) -> Float[Array, ""]
```

NPV of an FX forward contract. Positive = forward buyer is in-the-money.

### `garman_kohlhagen_price`

```python
garman_kohlhagen_price(option: FXVanillaOption, spot, vol, r_domestic, r_foreign) -> Float[Array, ""]
```

Garman-Kohlhagen (modified Black-Scholes) for European FX options. The foreign rate acts as a continuous dividend yield.

### `fx_implied_vol`

```python
fx_implied_vol(option: FXVanillaOption, spot, r_domestic, r_foreign,
               market_price, n_iterations=20) -> Float[Array, ""]
```

Newton-Raphson implied volatility using autodiff vega.

### `fx_delta`

```python
fx_delta(option: FXVanillaOption, spot, vol, r_domestic, r_foreign,
         convention="spot") -> Float[Array, ""]
```

FX delta in one of three market conventions:

- `"spot"` — standard spot delta: $e^{-r_f T} \Phi(d_1)$
- `"forward"` — forward delta: $\Phi(d_1)$
- `"premium_adjusted"` — premium-adjusted spot delta: $e^{-r_f T} \Phi(d_1) - P/S$

### `strike_to_delta`

```python
strike_to_delta(strike, spot, vol, r_domestic, r_foreign, T, is_call,
                convention="spot") -> Float[Array, ""]
```

Convert a strike to its delta value under the specified convention.

### `delta_to_strike`

```python
delta_to_strike(delta, spot, vol, r_domestic, r_foreign, T, is_call,
                convention="spot", n_iterations=20) -> Float[Array, ""]
```

Invert delta to find the corresponding strike (Newton-Raphson via autodiff).

---

## Variance Swap Pricing

### `variance_swap_fair_strike`

```python
variance_swap_fair_strike(vol) -> Float[Array, ""]
```

BSM fair variance strike: $K_{\text{var}} = \sigma^2$. Under Black-Scholes the fair variance is the squared implied vol.

### `variance_swap_price`

```python
variance_swap_price(swap: VarianceSwap, vol, rate) -> Float[Array, ""]
```

Mark-to-market of a variance swap under BSM. NPV = $N_{\text{var}} \cdot (\sigma^2 - K_{\text{var}}) \cdot e^{-rT}$.

### `variance_swap_price_seasoned`

```python
variance_swap_price_seasoned(swap: VarianceSwap, vol, rate,
                              realized_var, elapsed_fraction) -> Float[Array, ""]
```

Seasoned variance swap accounting for the portion already accrued: blends realized variance over the elapsed period with implied variance over the remaining period.

---

## Floating Rate Instruments

### `floating_rate_bond_price`

```python
floating_rate_bond_price(bond: FloatingRateBond, curve: DiscountCurve) -> Float[Array, ""]
```

Price a floating-rate note under the single-curve assumption. Projects forward rates from the curve (or uses known fixings from `bond.fixing_rates` where finite). Satisfies the par-at-reset invariant: a zero-spread FRN on its first reset date prices to face value.

### `ois_swap_price`

```python
ois_swap_price(swap: OISSwap, curve: DiscountCurve) -> Float[Array, ""]
```

NPV of an Overnight Index Swap. Float leg uses the telescoping identity $N \cdot (DF(T_0) - DF(T_n))$; fixed leg is the standard annuity. Sign follows `pay_fixed`.

### `ois_swap_rate`

```python
ois_swap_rate(swap: OISSwap, curve: DiscountCurve) -> Float[Array, ""]
```

Par OIS rate: $K^* = (DF(T_0) - DF(T_n)) / A$.

---

## Rates Exotics

### `cross_currency_swap_price`

```python
cross_currency_swap_price(swap: CrossCurrencySwap, domestic_curve: DiscountCurve,
                          foreign_curve: DiscountCurve, spot) -> Float[Array, ""]
```

NPV (in domestic currency) of a cross-currency basis swap. Two-curve telescoping with spot conversion. With `exchange_notional=True`, the NPV collapses to $N_d \cdot s \cdot A_d$.

### `cross_currency_basis_spread`

```python
cross_currency_basis_spread(swap: CrossCurrencySwap, domestic_curve: DiscountCurve,
                            foreign_curve: DiscountCurve, spot) -> Float[Array, ""]
```

Par basis spread that zeroes the XCCY NPV.

### `total_return_swap_price`

```python
total_return_swap_price(swap: TotalReturnSwap, curve: DiscountCurve,
                        unrealized_return=None) -> Float[Array, ""]
```

NPV under the self-financing asset assumption. At reset: $\text{NPV}_{\text{receiver}} = -N \cdot s \cdot A$. Optional `unrealized_return` adds accrued mark-to-market.

### `cms_swap_price`

```python
cms_swap_price(swap: CMSSwap, curve: DiscountCurve) -> Float[Array, ""]
```

CMS swap NPV using per-period forward par swap rates on a synthetic annual underlying swap. **No convexity adjustment** — see the guide for caveats.

### `cms_cap_floor_price_black76`

```python
cms_cap_floor_price_black76(cap: CMSCapFloor, curve: DiscountCurve,
                            vol) -> Float[Array, ""]
```

Black-76 on the unadjusted forward CMS rate. `vol` is scalar or per-period. Floor via put-call parity. Same convexity caveat as `cms_swap_price`.

### `range_accrual_price_black76`

```python
range_accrual_price_black76(accrual: RangeAccrual, curve: DiscountCurve,
                            vol) -> Float[Array, ""]
```

Digital-replication range accrual: per-period snapshot probability $P(L < F < U)$ under Black-76. `vol` is scalar or per-period.

---

## Lattice — Hull-White Trinomial Tree

### `build_hull_white_tree`

```python
build_hull_white_tree(model: HullWhiteModel, T: float, n_steps: int = 100) -> HullWhiteTree
```

Construct a Hull-White recombining trinomial tree from $t = 0$ to $T$. Per-step $\alpha$ calibration matches market discount factors exactly. Returns a `HullWhiteTree` pytree containing rates, Arrow-Debreu prices, probabilities, and target indices.

### `callable_bond_price`

```python
callable_bond_price(bond: CallableBond, model: HullWhiteModel,
                    n_steps: int = 100) -> Float[Array, ""]
```

Price a callable bond via backward induction on the Hull-White tree. The issuer calls when continuation value exceeds the call price. Result < straight bond price.

### `puttable_bond_price`

```python
puttable_bond_price(bond: PuttableBond, model: HullWhiteModel,
                    n_steps: int = 100) -> Float[Array, ""]
```

Price a puttable bond via backward induction. The holder puts when continuation value falls below the put price. Result > straight bond price.

### `HullWhiteTree`

```python
class HullWhiteTree(eqx.Module):
    dt: Float[Array, ""]
    dx: Float[Array, ""]
    n_steps: int           # static
    j_max: int             # static
    alpha: Float[Array, "n_steps"]
    rates: Float[Array, "n_steps_plus1 n_states"]
    probs: Float[Array, "n_states 3"]
    targets: Int[Array, "n_states 3"]
```

Pre-built tree data structure. `n_states = 2 * j_max + 1`.

---

## Inflation Derivatives

### `zcis_price`

```python
zcis_price(swap: ZeroCouponInflationSwap, inflation_curve: InflationCurve,
           discount_curve: DiscountCurve) -> Float[Array, ""]
```

NPV of a zero-coupon inflation swap. Inflation leg = $N \cdot (CPI(T)/CPI(0) - 1) \cdot DF(T)$; fixed leg = $N \cdot ((1+K)^T - 1) \cdot DF(T)$. Sign follows `is_inflation_receiver`.

### `zcis_breakeven_rate`

```python
zcis_breakeven_rate(swap: ZeroCouponInflationSwap,
                    inflation_curve: InflationCurve) -> Float[Array, ""]
```

Par (breakeven) rate $K^* = (CPI(T)/CPI(0))^{1/T} - 1$. Independent of the discount curve.

### `yyis_price`

```python
yyis_price(swap: YearOnYearInflationSwap, inflation_curve: InflationCurve,
           discount_curve: DiscountCurve) -> Float[Array, ""]
```

NPV of a year-on-year inflation swap. Per-period YoY forward rate from the inflation curve ratio, discounted. No convexity adjustment.

### `inflation_cap_floor_price_black76`

```python
inflation_cap_floor_price_black76(cap: InflationCapFloor,
                                  inflation_curve: InflationCurve,
                                  discount_curve: DiscountCurve,
                                  vol) -> Float[Array, ""]
```

Black-76 on the YoY forward inflation rate. `vol` is scalar or per-period. Floor via put-call parity.

---

## Spread Options

### `margrabe_price`

```python
margrabe_price(option: SpreadOption, s1, s2, vol1, vol2, rho,
               q1=0.0, q2=0.0) -> Float[Array, ""]
```

Margrabe's exact formula for exchange options ($K = 0$). Price is independent of the risk-free rate. Put via Margrabe parity.

### `kirk_price`

```python
kirk_price(option: SpreadOption, s1, s2, vol1, vol2, rho, rate,
           q1=0.0, q2=0.0) -> Float[Array, ""]
```

Kirk's approximation for spread options ($K \neq 0$). Treats $S_2 + K$ as a single asset with adjusted vol. Degenerates to Margrabe when $K = 0$. Put via put-call parity.

### `spread_option_price`

```python
spread_option_price(option: SpreadOption, s1, s2, vol1, vol2, rho, rate,
                    q1=0.0, q2=0.0) -> Float[Array, ""]
```

Convenience wrapper — dispatches to `kirk_price` (handles all $K$).
