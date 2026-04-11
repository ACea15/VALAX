# Instruments

All instruments are `equinox.Module` subclasses — frozen dataclasses registered as JAX pytrees. They carry no pricing logic.

## `EuropeanOption`

```python
from valax.instruments import EuropeanOption

class EuropeanOption(eqx.Module):
    strike: Float[Array, ""]                          # strike price
    expiry: Float[Array, ""]                          # time to expiry (year fractions)
    is_call: bool = eqx.field(static=True, default=True)  # True=call, False=put
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |

**Notes**:

- `is_call` is marked `static=True` because it controls branching logic (call vs put formula). JAX traces separate code paths for `True` and `False`.
- `strike` and `expiry` are JAX arrays, so you can differentiate through them (e.g., strike sensitivity, theta via autodiff on expiry).
- For batch pricing via `vmap`, create a single `EuropeanOption` with batched arrays: `EuropeanOption(strike=jnp.array([90, 100, 110]), ...)`.

## `ZeroCouponBond`

```python
from valax.instruments import ZeroCouponBond

class ZeroCouponBond(eqx.Module):
    maturity: Int[Array, ""]     # maturity date (ordinal)
    face_value: Float[Array, ""] # par value paid at maturity
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `maturity` | `Int[Array, ""]` | No | Maturity date as ordinal (days since epoch). |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |

## `FixedRateBond`

```python
from valax.instruments import FixedRateBond

class FixedRateBond(eqx.Module):
    payment_dates: Int[Array, "n_payments"]  # coupon + maturity dates
    settlement_date: Int[Array, ""]          # valuation date
    coupon_rate: Float[Array, ""]            # annual coupon rate
    face_value: Float[Array, ""]             # par value
    frequency: int = eqx.field(static=True, default=2)   # coupons/year
    day_count: str = eqx.field(static=True, default="act_365")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Ordinal dates for coupon payments (includes maturity). |
| `settlement_date` | `Int[Array, ""]` | No | Settlement/valuation date as ordinal. |
| `coupon_rate` | `Float[Array, ""]` | No | Annual coupon rate (e.g., 0.05 for 5%). Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `frequency` | `int` | Yes | Coupons per year (1, 2, or 4). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Use `generate_schedule()` from `valax.dates` to build `payment_dates`.
- Only future cash flows (`payment_date > settlement_date`) are included in pricing.
- `frequency` and `day_count` are static because they control computation structure, not values.

## `AmericanOption`

```python
from valax.instruments import AmericanOption

class AmericanOption(eqx.Module):
    strike: Float[Array, ""]                          # exercise price
    expiry: Float[Array, ""]                          # time to expiry (year fractions)
    is_call: bool = eqx.field(static=True, default=True)  # True=call, False=put
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Exercise price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |

**Notes**:

- Exercisable at any time up to expiry. Pricing requires methods that handle early exercise (binomial trees, PDE with free boundary, or Longstaff-Schwartz MC).
- For calls on non-dividend-paying stocks, the American price equals the European price (early exercise is never optimal).
- For puts or when dividends are present, the early exercise premium is positive.

## `EquityBarrierOption`

```python
from valax.instruments import EquityBarrierOption

class EquityBarrierOption(eqx.Module):
    strike: Float[Array, ""]                          # exercise price
    expiry: Float[Array, ""]                          # time to expiry (year fractions)
    barrier: Float[Array, ""]                         # barrier price level
    is_call: bool = eqx.field(static=True, default=True)      # True=call, False=put
    is_up: bool = eqx.field(static=True, default=True)        # True if barrier above spot
    is_knock_in: bool = eqx.field(static=True, default=True)  # True=knock-in, False=knock-out
    smoothing: float = eqx.field(static=True, default=0.0)    # sigmoid smoothing width (0=hard barrier)
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Exercise price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `barrier` | `Float[Array, ""]` | No | Barrier price level. Differentiable. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |
| `is_up` | `bool` | Yes | True if barrier is above current spot (up barrier). Not differentiable. |
| `is_knock_in` | `bool` | Yes | True for knock-in, False for knock-out. Not differentiable. |
| `smoothing` | `float` | Yes | Sigmoid smoothing width for differentiable MC payoffs. Set to 0.0 for hard barriers. |

**Notes**:

- European option that activates (knock-in) or deactivates (knock-out) if spot breaches the barrier.
- Knock-in / knock-out parity: `knock_in_price + knock_out_price = vanilla_price` for same strike, barrier, and type.
- Barrier monitoring is continuous for analytical pricing, discrete (per time step) for Monte Carlo.

## `AsianOption`

```python
from valax.instruments import AsianOption

class AsianOption(eqx.Module):
    strike: Float[Array, ""]                          # exercise price
    expiry: Float[Array, ""]                          # time to expiry (year fractions)
    is_call: bool = eqx.field(static=True, default=True)          # True=call, False=put
    averaging: str = eqx.field(static=True, default="arithmetic") # "arithmetic" or "geometric"
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Exercise price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |
| `averaging` | `str` | Yes | Averaging method: `"arithmetic"` or `"geometric"`. |

**Notes**:

- Payoff depends on the average spot price over observation dates rather than terminal spot.
- No closed-form price exists for arithmetic averages under BSM; geometric averages have a closed form.
- Lower vega and gamma than equivalent vanillas because averaging reduces effective volatility.

## `LookbackOption`

```python
from valax.instruments import LookbackOption

class LookbackOption(eqx.Module):
    expiry: Float[Array, ""]                                    # time to expiry (year fractions)
    is_call: bool = eqx.field(static=True, default=True)       # True=call, False=put
    is_fixed_strike: bool = eqx.field(static=True, default=False) # True=fixed-strike, False=floating-strike
    strike: Float[Array, ""] = eqx.field(default=None)         # exercise price (used only when is_fixed_strike=True)
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |
| `is_fixed_strike` | `bool` | Yes | True for fixed-strike lookback, False for floating-strike. Not differentiable. |
| `strike` | `Float[Array, ""]` | No | Exercise price (used only when `is_fixed_strike=True`). Differentiable. |

**Notes**:

- Floating-strike: call payoff is `S_T - min(S_t)`, put payoff is `max(S_t) - S_T`.
- Fixed-strike: call payoff is `max(max(S_t) - K, 0)`, put payoff is `max(K - min(S_t), 0)`.
- Floating-strike lookbacks are always in the money (non-negative payoff by construction).

## `VarianceSwap`

```python
from valax.instruments import VarianceSwap

class VarianceSwap(eqx.Module):
    expiry: Float[Array, ""]        # observation period (year fractions)
    strike_var: Float[Array, ""]    # variance strike (K_vol^2)
    notional_var: Float[Array, ""]  # variance notional
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `expiry` | `Float[Array, ""]` | No | Time to expiry / observation period in year fractions. Differentiable. |
| `strike_var` | `Float[Array, ""]` | No | Variance strike K_var = K_vol². Differentiable. |
| `notional_var` | `Float[Array, ""]` | No | Variance notional. Differentiable. |

**Notes**:

- Payoff: `N_var × (σ_realized² − K_var)` where σ_realized² is annualized realized variance from discrete log returns.
- Under Black-Scholes, fair variance strike equals squared implied volatility.
- Vega notional relates to variance notional as: `N_vega = 2 × K_vol × N_var`.

<!-- Exotic Options -->

## `CompoundOption`

```python
from valax.instruments import CompoundOption

class CompoundOption(eqx.Module):
    outer_expiry: Float[Array, ""]   # expiry of the compound option (year fractions)
    outer_strike: Float[Array, ""]   # premium to acquire the underlying option
    inner_expiry: Float[Array, ""]   # expiry of the underlying option (must be > outer_expiry)
    inner_strike: Float[Array, ""]   # strike of the underlying option
    outer_is_call: bool = eqx.field(static=True, default=True)   # True if compound option is a call
    inner_is_call: bool = eqx.field(static=True, default=True)   # True if underlying option is a call
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `outer_expiry` | `Float[Array, ""]` | No | Expiry of the compound option in year fractions. Differentiable. |
| `outer_strike` | `Float[Array, ""]` | No | Premium to acquire the underlying option. Differentiable. |
| `inner_expiry` | `Float[Array, ""]` | No | Expiry of the underlying option (must be > `outer_expiry`). Differentiable. |
| `inner_strike` | `Float[Array, ""]` | No | Strike of the underlying option. Differentiable. |
| `outer_is_call` | `bool` | Yes | True if the compound option is a call. Not differentiable. |
| `inner_is_call` | `bool` | Yes | True if the underlying option is a call. Not differentiable. |

**Notes**:

- Priced via the Geske formula (bivariate normal CDF).
- Four combinations: call-on-call, call-on-put, put-on-call, put-on-put.
- `inner_expiry` must be strictly greater than `outer_expiry`.

## `ChooserOption`

```python
from valax.instruments import ChooserOption

class ChooserOption(eqx.Module):
    choose_date: Float[Array, ""]   # date when call/put choice is made (year fractions)
    expiry: Float[Array, ""]        # final expiry of the resulting option
    strike: Float[Array, ""]        # strike price
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `choose_date` | `Float[Array, ""]` | No | Date when the holder chooses call or put, in year fractions. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Final expiry of the resulting option. Differentiable. |
| `strike` | `Float[Array, ""]` | No | Strike price. Differentiable. |

**Notes**:

- At `choose_date`, the holder selects the more valuable of a call or put with the same strike and expiry.
- Decomposed into a call plus a put with adjusted parameters for closed-form pricing.

## `Autocallable`

```python
from valax.instruments import Autocallable

class Autocallable(eqx.Module):
    observation_dates: Int[Array, "n"]    # autocall/coupon observation dates (ordinals)
    autocall_barrier: Float[Array, ""]    # autocall trigger as fraction of initial spot
    coupon_barrier: Float[Array, ""]      # coupon payment trigger as fraction of initial
    coupon_rate: Float[Array, ""]         # periodic coupon rate
    ki_barrier: Float[Array, ""]          # knock-in put barrier as fraction of initial
    strike: Float[Array, ""]             # put strike as fraction of initial (if knocked in)
    notional: Float[Array, ""]           # note face value
    has_memory: bool = eqx.field(static=True, default=False)  # True if missed coupons are recovered later
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `observation_dates` | `Int[Array, "n"]` | No | Autocall and coupon observation dates as ordinals. |
| `autocall_barrier` | `Float[Array, ""]` | No | Autocall trigger as fraction of initial spot (e.g., 1.0). Differentiable. |
| `coupon_barrier` | `Float[Array, ""]` | No | Coupon payment trigger as fraction of initial spot. Differentiable. |
| `coupon_rate` | `Float[Array, ""]` | No | Periodic coupon rate. Differentiable. |
| `ki_barrier` | `Float[Array, ""]` | No | Knock-in put barrier as fraction of initial spot. Differentiable. |
| `strike` | `Float[Array, ""]` | No | Put strike as fraction of initial (used if knock-in triggered). Differentiable. |
| `notional` | `Float[Array, ""]` | No | Note face value. Differentiable. |
| `has_memory` | `bool` | Yes | True if missed coupons are recovered on later coupon dates. |

**Notes**:

- Phoenix autocallable: early redemption at par + coupon if spot ≥ `autocall_barrier`.
- Requires Monte Carlo simulation for path-dependent barrier logic.
- Memory feature causes accumulated missed coupons to be paid when coupon barrier is next breached.

## `WorstOfBasketOption`

```python
from valax.instruments import WorstOfBasketOption

class WorstOfBasketOption(eqx.Module):
    expiry: Float[Array, ""]       # time to expiry (year fractions)
    strike: Float[Array, ""]       # strike as return level (1.0 = ATM)
    notional: Float[Array, ""]     # notional amount
    n_assets: int = eqx.field(static=True, default=2)    # number of basket assets
    is_call: bool = eqx.field(static=True, default=False) # True for call, False for put
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `strike` | `Float[Array, ""]` | No | Strike as return level (1.0 = ATM). Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional amount. Differentiable. |
| `n_assets` | `int` | Yes | Number of basket assets. |
| `is_call` | `bool` | Yes | True for call, False for put. Not differentiable. |

**Notes**:

- Payoff is based on the worst-performing asset in the basket.
- Requires correlated multi-asset Monte Carlo simulation.
- `n_assets` is static as it determines array shapes in the simulation.

## `Cliquet`

```python
from valax.instruments import Cliquet

class Cliquet(eqx.Module):
    observation_dates: Int[Array, "n"]   # reset dates (ordinals)
    local_cap: Float[Array, ""]          # max return per period
    local_floor: Float[Array, ""]        # min return per period
    global_floor: Float[Array, ""]       # min total accumulated return
    notional: Float[Array, ""]           # notional amount
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `observation_dates` | `Int[Array, "n"]` | No | Reset/observation dates as ordinals. |
| `local_cap` | `Float[Array, ""]` | No | Maximum return credited per period. Differentiable. |
| `local_floor` | `Float[Array, ""]` | No | Minimum return credited per period. Differentiable. |
| `global_floor` | `Float[Array, ""]` | No | Minimum total accumulated return at maturity. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional amount. Differentiable. |

**Notes**:

- Ratchet option: periodic returns are individually capped and floored, then summed.
- The global floor provides a minimum payoff guarantee at maturity.
- Path-dependent — requires Monte Carlo simulation.

## `DigitalOption`

```python
from valax.instruments import DigitalOption

class DigitalOption(eqx.Module):
    strike: Float[Array, ""]    # strike price
    expiry: Float[Array, ""]    # time to expiry (year fractions)
    payout: Float[Array, ""]    # fixed payout if in-the-money
    is_call: bool = eqx.field(static=True, default=True)  # True for digital call, False for digital put
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `payout` | `Float[Array, ""]` | No | Fixed payout if option finishes in-the-money. Differentiable. |
| `is_call` | `bool` | Yes | True for digital call, False for digital put. Not differentiable. |

**Notes**:

- Binary/cash-or-nothing option: pays `payout` if spot is above (call) or below (put) strike at expiry, zero otherwise.
- Discontinuous payoff — greeks can be sensitive near the strike.

## `SpreadOption`

```python
from valax.instruments import SpreadOption

class SpreadOption(eqx.Module):
    expiry: Float[Array, ""]      # time to expiry (year fractions)
    strike: Float[Array, ""]      # spread strike (0 = exchange option)
    notional: Float[Array, ""]    # notional amount
    is_call: bool = eqx.field(static=True, default=True)  # True for call on the spread
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `strike` | `Float[Array, ""]` | No | Spread strike (0 = exchange option). Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional amount. Differentiable. |
| `is_call` | `bool` | Yes | True for call on the spread. Not differentiable. |

**Notes**:

- Payoff: `max(S1 - S2 - K, 0)` for a call, where S1 and S2 are the two asset prices.
- When `strike = 0`, reduces to a Margrabe exchange option with a closed-form solution.
- Kirk's approximation is used when `strike ≠ 0`.

<!-- Bonds -->

## `FloatingRateBond`

```python
from valax.instruments import FloatingRateBond

class FloatingRateBond(eqx.Module):
    payment_dates: Int[Array, "n"]                # coupon payment dates (ordinals)
    fixing_dates: Int[Array, "n"]                 # rate observation dates (ordinals)
    settlement_date: Int[Array, ""]               # valuation date (ordinal)
    spread: Float[Array, ""]                      # fixed spread over reference rate
    face_value: Float[Array, ""]                  # par value
    fixing_rates: Optional[Float[Array, "n"]] = None  # known past fixings (NaN for future)
    frequency: int = eqx.field(static=True, default=4)        # resets per year
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Coupon payment dates as ordinals. |
| `fixing_dates` | `Int[Array, "n"]` | No | Rate observation/fixing dates as ordinals. |
| `settlement_date` | `Int[Array, ""]` | No | Valuation date as ordinal. |
| `spread` | `Float[Array, ""]` | No | Fixed spread over the reference rate. Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `fixing_rates` | `Optional[Float[Array, "n"]]` | No | Known past fixing rates (`None` or NaN for future fixings). |
| `frequency` | `int` | Yes | Number of coupon resets per year (e.g., 4 for quarterly). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Floating rate note (FRN) with coupons that reset off a reference rate (e.g., SOFR).
- `fixing_rates` allows mixing known historical fixings with projected future rates.
- Prices to approximately par on reset dates when `spread` matches the market spread.

## `CallableBond`

```python
from valax.instruments import CallableBond

class CallableBond(eqx.Module):
    payment_dates: Int[Array, "n"]     # coupon dates (ordinals)
    settlement_date: Int[Array, ""]    # valuation date (ordinal)
    coupon_rate: Float[Array, ""]      # annual coupon rate
    face_value: Float[Array, ""]       # par value
    call_dates: Int[Array, "m"]        # dates issuer may call (ordinals)
    call_prices: Float[Array, "m"]     # redemption price at each call date
    frequency: int = eqx.field(static=True, default=2)        # coupons per year
    day_count: str = eqx.field(static=True, default="act_365")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Coupon payment dates as ordinals. |
| `settlement_date` | `Int[Array, ""]` | No | Valuation date as ordinal. |
| `coupon_rate` | `Float[Array, ""]` | No | Annual coupon rate. Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `call_dates` | `Int[Array, "m"]` | No | Dates on which the issuer may call the bond (ordinals). |
| `call_prices` | `Float[Array, "m"]` | No | Redemption price at each call date. Differentiable. |
| `frequency` | `int` | Yes | Coupons per year (1, 2, or 4). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Fixed-rate bond with an embedded issuer call schedule.
- The issuer will exercise the call when it is economically advantageous (rates fall).
- Requires a rate tree or backward-induction model for pricing.

## `PuttableBond`

```python
from valax.instruments import PuttableBond

class PuttableBond(eqx.Module):
    payment_dates: Int[Array, "n"]     # coupon dates (ordinals)
    settlement_date: Int[Array, ""]    # valuation date (ordinal)
    coupon_rate: Float[Array, ""]      # annual coupon rate
    face_value: Float[Array, ""]       # par value
    put_dates: Int[Array, "m"]         # dates holder may put (ordinals)
    put_prices: Float[Array, "m"]      # redemption price at each put date
    frequency: int = eqx.field(static=True, default=2)        # coupons per year
    day_count: str = eqx.field(static=True, default="act_365")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Coupon payment dates as ordinals. |
| `settlement_date` | `Int[Array, ""]` | No | Valuation date as ordinal. |
| `coupon_rate` | `Float[Array, ""]` | No | Annual coupon rate. Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `put_dates` | `Int[Array, "m"]` | No | Dates on which the holder may put the bond (ordinals). |
| `put_prices` | `Float[Array, "m"]` | No | Redemption price at each put date. Differentiable. |
| `frequency` | `int` | Yes | Coupons per year (1, 2, or 4). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Fixed-rate bond with an embedded holder put schedule.
- The holder will exercise the put when it is economically advantageous (rates rise).
- Requires a rate tree or backward-induction model for pricing.

## `ConvertibleBond`

```python
from valax.instruments import ConvertibleBond

class ConvertibleBond(eqx.Module):
    payment_dates: Int[Array, "n"]                        # coupon dates (ordinals)
    settlement_date: Int[Array, ""]                       # valuation date (ordinal)
    coupon_rate: Float[Array, ""]                         # annual coupon rate
    face_value: Float[Array, ""]                          # par value
    conversion_ratio: Float[Array, ""]                    # shares per unit of face value
    call_dates: Optional[Int[Array, "m"]] = None          # issuer call dates
    call_prices: Optional[Float[Array, "m"]] = None       # issuer call prices
    frequency: int = eqx.field(static=True, default=2)        # coupons per year
    day_count: str = eqx.field(static=True, default="act_365")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Coupon payment dates as ordinals. |
| `settlement_date` | `Int[Array, ""]` | No | Valuation date as ordinal. |
| `coupon_rate` | `Float[Array, ""]` | No | Annual coupon rate. Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `conversion_ratio` | `Float[Array, ""]` | No | Number of shares received per unit of face value on conversion. Differentiable. |
| `call_dates` | `Optional[Int[Array, "m"]]` | No | Optional issuer call dates (ordinals). `None` if non-callable. |
| `call_prices` | `Optional[Float[Array, "m"]]` | No | Optional issuer call prices. `None` if non-callable. Differentiable. |
| `frequency` | `int` | Yes | Coupons per year (1, 2, or 4). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Equity-credit hybrid: holder may convert bond into equity at any time.
- Value is `max(bond_value, conversion_ratio × stock_price)` at each decision point.
- Issuer call provision forces early conversion decisions.
- Requires coupled equity/credit tree or Monte Carlo for pricing.

<!-- Credit Derivatives -->

## `CDS`

```python
from valax.instruments import CDS

class CDS(eqx.Module):
    effective_date: Int[Array, ""]     # contract start date (ordinal)
    maturity_date: Int[Array, ""]      # contract end date (ordinal)
    premium_dates: Int[Array, "n"]     # premium payment dates (ordinals)
    spread: Float[Array, ""]           # annual CDS spread
    notional: Float[Array, ""]         # notional principal
    recovery_rate: Float[Array, ""]    # expected recovery rate
    is_protection_buyer: bool = eqx.field(static=True, default=True)   # True = buying protection
    premium_frequency: int = eqx.field(static=True, default=4)         # premiums per year
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `effective_date` | `Int[Array, ""]` | No | Contract start date as ordinal. |
| `maturity_date` | `Int[Array, ""]` | No | Contract end date as ordinal. |
| `premium_dates` | `Int[Array, "n"]` | No | Premium payment dates as ordinals. |
| `spread` | `Float[Array, ""]` | No | Annual CDS spread (e.g., 0.01 for 100 bps). Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `recovery_rate` | `Float[Array, ""]` | No | Expected recovery rate on default. Differentiable. |
| `is_protection_buyer` | `bool` | Yes | True if buying protection, False if selling. |
| `premium_frequency` | `int` | Yes | Number of premium payments per year. |
| `day_count` | `str` | Yes | Day count convention for premium accrual. |

**Notes**:

- Protection buyer pays periodic spread and receives `(1 - recovery_rate) × notional` on default.
- Mark-to-market value depends on the difference between contract spread and current market spread.
- Accrued premium at default is typically included in settlement.

## `CDOTranche`

```python
from valax.instruments import CDOTranche

class CDOTranche(eqx.Module):
    effective_date: Int[Array, ""]     # trade date (ordinal)
    maturity_date: Int[Array, ""]      # maturity date (ordinal)
    premium_dates: Int[Array, "n"]     # premium payment dates (ordinals)
    attachment: Float[Array, ""]       # lower attachment point
    detachment: Float[Array, ""]       # upper detachment point
    spread: Float[Array, ""]           # running premium spread
    notional: Float[Array, ""]         # total portfolio notional
    recovery_rate: Float[Array, ""]    # uniform recovery rate
    n_names: int = eqx.field(static=True, default=125)       # number of reference entities
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `effective_date` | `Int[Array, ""]` | No | Trade date as ordinal. |
| `maturity_date` | `Int[Array, ""]` | No | Maturity date as ordinal. |
| `premium_dates` | `Int[Array, "n"]` | No | Premium payment dates as ordinals. |
| `attachment` | `Float[Array, ""]` | No | Lower attachment point (e.g., 0.03 for 3%). Differentiable. |
| `detachment` | `Float[Array, ""]` | No | Upper detachment point (e.g., 0.07 for 7%). Differentiable. |
| `spread` | `Float[Array, ""]` | No | Running premium spread. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Total portfolio notional. Differentiable. |
| `recovery_rate` | `Float[Array, ""]` | No | Uniform recovery rate assumed for all names. Differentiable. |
| `n_names` | `int` | Yes | Number of reference entities in the portfolio. |
| `day_count` | `str` | Yes | Day count convention for premium accrual. |

**Notes**:

- Tranche absorbs portfolio losses between `attachment` and `detachment` points.
- Tranche notional is `(detachment - attachment) × notional`.
- Requires a portfolio loss model (e.g., Gaussian copula) for pricing.
- `n_names` is static as it determines correlation matrix dimensions.

<!-- Inflation Derivatives -->

## `ZeroCouponInflationSwap`

```python
from valax.instruments import ZeroCouponInflationSwap

class ZeroCouponInflationSwap(eqx.Module):
    effective_date: Int[Array, ""]     # swap start date (ordinal)
    maturity_date: Int[Array, ""]      # payment date (ordinal)
    fixed_rate: Float[Array, ""]       # annual break-even rate
    notional: Float[Array, ""]         # notional principal
    base_cpi: Float[Array, ""]         # CPI level at inception
    is_inflation_receiver: bool = eqx.field(static=True, default=True)  # True = receive inflation
    index_lag: int = eqx.field(static=True, default=3)          # CPI publication lag in months
    day_count: str = eqx.field(static=True, default="act_act")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `effective_date` | `Int[Array, ""]` | No | Swap start date as ordinal. |
| `maturity_date` | `Int[Array, ""]` | No | Payment date as ordinal. |
| `fixed_rate` | `Float[Array, ""]` | No | Annual break-even inflation rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `base_cpi` | `Float[Array, ""]` | No | CPI level at inception. Differentiable. |
| `is_inflation_receiver` | `bool` | Yes | True if receiving the inflation leg. |
| `index_lag` | `int` | Yes | CPI publication lag in months (typically 3). |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Single payment at maturity: inflation receiver gets `notional × (CPI_T / base_cpi - 1)`, pays `notional × ((1 + fixed_rate)^T - 1)`.
- The `index_lag` accounts for delayed CPI publication.
- Break-even rate is the fixed rate that makes the swap NPV zero at inception.

## `YearOnYearInflationSwap`

```python
from valax.instruments import YearOnYearInflationSwap

class YearOnYearInflationSwap(eqx.Module):
    effective_date: Int[Array, ""]     # swap start date (ordinal)
    payment_dates: Int[Array, "n"]     # payment dates (ordinals)
    fixed_rate: Float[Array, ""]       # annual fixed rate
    notional: Float[Array, ""]         # notional principal
    base_cpi: Float[Array, ""]         # CPI level at inception
    is_inflation_receiver: bool = eqx.field(static=True, default=True)
    index_lag: int = eqx.field(static=True, default=3)          # publication lag in months
    day_count: str = eqx.field(static=True, default="act_act")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `effective_date` | `Int[Array, ""]` | No | Swap start date as ordinal. |
| `payment_dates` | `Int[Array, "n"]` | No | Payment dates as ordinals. |
| `fixed_rate` | `Float[Array, ""]` | No | Annual fixed rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `base_cpi` | `Float[Array, ""]` | No | CPI level at inception. Differentiable. |
| `is_inflation_receiver` | `bool` | Yes | True if receiving the inflation leg. |
| `index_lag` | `int` | Yes | CPI publication lag in months (typically 3). |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Periodic payments: each period exchanges annual CPI return (`CPI_t / CPI_{t-1} - 1`) vs. fixed rate.
- Year-on-year structure avoids the compounding risk present in zero-coupon inflation swaps.
- Requires a year-on-year convexity adjustment when derived from zero-coupon inflation curves.

## `InflationCapFloor`

```python
from valax.instruments import InflationCapFloor

class InflationCapFloor(eqx.Module):
    effective_date: Int[Array, ""]     # start date (ordinal)
    payment_dates: Int[Array, "n"]     # caplet/floorlet dates (ordinals)
    strike: Float[Array, ""]           # strike inflation rate
    notional: Float[Array, ""]         # notional principal
    base_cpi: Float[Array, ""]         # CPI at inception
    is_cap: bool = eqx.field(static=True, default=True)     # True for cap, False for floor
    index_lag: int = eqx.field(static=True, default=3)
    day_count: str = eqx.field(static=True, default="act_act")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `effective_date` | `Int[Array, ""]` | No | Start date as ordinal. |
| `payment_dates` | `Int[Array, "n"]` | No | Caplet/floorlet payment dates as ordinals. |
| `strike` | `Float[Array, ""]` | No | Strike inflation rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `base_cpi` | `Float[Array, ""]` | No | CPI level at inception. Differentiable. |
| `is_cap` | `bool` | Yes | True for cap, False for floor. Not differentiable. |
| `index_lag` | `int` | Yes | CPI publication lag in months. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Option on year-on-year CPI returns: each caplet pays `max(YoY_inflation - strike, 0) × notional`.
- A floor provides protection against deflation (low inflation).
- Priced as a strip of individual caplets/floorlets on YoY inflation.

<!-- FX Derivatives -->

## `FXForward`

```python
from valax.instruments import FXForward

class FXForward(eqx.Module):
    strike: Float[Array, ""]              # delivery FX rate (domestic per foreign)
    expiry: Float[Array, ""]              # time to maturity (year fractions)
    notional_foreign: Float[Array, ""]    # amount in foreign currency
    is_buy: bool = eqx.field(static=True, default=True)        # True=buy foreign / sell domestic
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Delivery FX rate (domestic per foreign). Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to maturity in year fractions. Differentiable. |
| `notional_foreign` | `Float[Array, ""]` | No | Amount in foreign currency. Differentiable. |
| `is_buy` | `bool` | Yes | True = buy foreign / sell domestic. Not differentiable. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- Agreement to exchange `notional_foreign` units of foreign currency for `notional_foreign × strike` units of domestic currency at maturity.
- Fair forward rate: `F = S × exp((r_dom − r_for) × T)`. NPV is zero at inception when `strike = F`.
- `is_buy=True` means buying foreign / selling domestic.

## `FXVanillaOption`

```python
from valax.instruments import FXVanillaOption

class FXVanillaOption(eqx.Module):
    strike: Float[Array, ""]              # strike FX rate (domestic per foreign)
    expiry: Float[Array, ""]              # time to expiry (year fractions)
    notional_foreign: Float[Array, ""]    # notional in foreign currency
    is_call: bool = eqx.field(static=True, default=True)       # True=call (buy foreign)
    premium_currency: str = eqx.field(static=True, default="domestic")  # "domestic" or "foreign"
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike FX rate (domestic per foreign). Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `notional_foreign` | `Float[Array, ""]` | No | Notional in foreign currency. Differentiable. |
| `is_call` | `bool` | Yes | True for call (buy foreign), False for put (sell foreign). Not differentiable. |
| `premium_currency` | `str` | Yes | Premium quoted in `"domestic"` or `"foreign"` currency. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- European FX vanilla option (Garman-Kohlhagen). Call payoff: `notional × max(S_T − K, 0)`.
- `premium_currency` affects the premium-adjusted delta used in the FX market.
- A call gives the right to buy foreign currency at the strike rate; a put gives the right to sell.

## `FXBarrierOption`

```python
from valax.instruments import FXBarrierOption

class FXBarrierOption(eqx.Module):
    strike: Float[Array, ""]              # strike FX rate
    expiry: Float[Array, ""]              # time to expiry (year fractions)
    notional_foreign: Float[Array, ""]    # notional in foreign currency
    barrier: Float[Array, ""]             # barrier FX rate level
    is_call: bool = eqx.field(static=True, default=True)       # True=call, False=put
    is_up: bool = eqx.field(static=True, default=True)         # True if barrier above spot
    is_knock_in: bool = eqx.field(static=True, default=True)   # True=knock-in, False=knock-out
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike FX rate. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `notional_foreign` | `Float[Array, ""]` | No | Notional in foreign currency. Differentiable. |
| `barrier` | `Float[Array, ""]` | No | Barrier FX rate level. Differentiable. |
| `is_call` | `bool` | Yes | True for call, False for put. Not differentiable. |
| `is_up` | `bool` | Yes | True if barrier is above spot (up-barrier). Not differentiable. |
| `is_knock_in` | `bool` | Yes | True for knock-in, False for knock-out. Not differentiable. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- FX single-barrier option: activated (knock-in) or deactivated (knock-out) if spot breaches the barrier.
- Four types: up-and-in, up-and-out, down-and-in, down-and-out.
- Closed-form solutions exist for continuous barriers; discrete monitoring requires Monte Carlo.

## `QuantoOption`

```python
from valax.instruments import QuantoOption

class QuantoOption(eqx.Module):
    strike: Float[Array, ""]           # strike in foreign currency
    expiry: Float[Array, ""]           # time to expiry (year fractions)
    notional: Float[Array, ""]         # notional (domestic currency)
    quanto_fx_rate: Float[Array, ""]   # fixed FX rate (domestic per foreign)
    is_call: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike in foreign currency. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional in domestic currency. Differentiable. |
| `quanto_fx_rate` | `Float[Array, ""]` | No | Fixed FX rate (domestic per foreign unit). Differentiable. |
| `is_call` | `bool` | Yes | True for call, False for put. Not differentiable. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- Foreign asset option with payoff converted at a pre-agreed fixed FX rate.
- Quanto adjustment modifies the drift of the foreign asset by `−ρ σ_S σ_FX`.
- Eliminates FX risk for the option holder.

## `TARF`

```python
from valax.instruments import TARF

class TARF(eqx.Module):
    fixing_dates: Int[Array, "n"]            # observation dates (ordinals)
    strike: Float[Array, ""]                 # strike FX rate
    target: Float[Array, ""]                 # target accrual level
    notional_per_fixing: Float[Array, ""]    # notional per fixing period
    leverage: Float[Array, ""]               # leverage on losses
    is_buy: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `fixing_dates` | `Int[Array, "n"]` | No | Observation/fixing dates as ordinals. |
| `strike` | `Float[Array, ""]` | No | Strike FX rate. Differentiable. |
| `target` | `Float[Array, ""]` | No | Target accrual level; contract terminates when reached. Differentiable. |
| `notional_per_fixing` | `Float[Array, ""]` | No | Notional per fixing period. Differentiable. |
| `leverage` | `Float[Array, ""]` | No | Leverage multiplier applied to losses. Differentiable. |
| `is_buy` | `bool` | Yes | True if buying foreign currency at strike. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- Target Accrual Range Forward: gains accumulate toward the target; contract knocks out when target is reached.
- Losses are leveraged (typically 2×), making the structure asymmetric.
- Path-dependent — requires Monte Carlo simulation.

## `FXSwap`

```python
from valax.instruments import FXSwap

class FXSwap(eqx.Module):
    near_date: Int[Array, ""]            # near leg date (ordinal)
    far_date: Int[Array, ""]             # far leg date (ordinal)
    spot_rate: Float[Array, ""]          # near leg FX rate
    forward_rate: Float[Array, ""]       # far leg FX rate
    notional_foreign: Float[Array, ""]   # foreign currency amount
    is_buy_near: bool = eqx.field(static=True, default=True)   # buy foreign on near leg
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `near_date` | `Int[Array, ""]` | No | Near leg settlement date as ordinal. |
| `far_date` | `Int[Array, ""]` | No | Far leg settlement date as ordinal. |
| `spot_rate` | `Float[Array, ""]` | No | Near leg FX rate. Differentiable. |
| `forward_rate` | `Float[Array, ""]` | No | Far leg FX rate. Differentiable. |
| `notional_foreign` | `Float[Array, ""]` | No | Foreign currency notional amount. Differentiable. |
| `is_buy_near` | `bool` | Yes | True if buying foreign currency on the near leg. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |

**Notes**:

- Combination of a spot exchange and an offsetting forward exchange.
- Used for short-term funding, hedging, or rolling FX positions.
- Swap points = `forward_rate - spot_rate`, driven by interest rate differential.

<!-- Rates Derivatives -->

## `Caplet`

```python
from valax.instruments import Caplet

class Caplet(eqx.Module):
    fixing_date: Int[Array, ""]                         # rate observation date (ordinal)
    start_date: Int[Array, ""]                          # accrual period start (ordinal)
    end_date: Int[Array, ""]                            # accrual period end = payment date (ordinal)
    strike: Float[Array, ""]                            # cap/floor rate K
    notional: Float[Array, ""]                          # notional principal
    is_cap: bool = eqx.field(static=True, default=True)        # True=caplet, False=floorlet
    day_count: str = eqx.field(static=True, default="act_360") # day count convention
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `fixing_date` | `Int[Array, ""]` | No | Date when the reference rate is observed (ordinal). |
| `start_date` | `Int[Array, ""]` | No | Accrual period start date (ordinal). |
| `end_date` | `Int[Array, ""]` | No | Accrual period end date = payment date (ordinal). |
| `strike` | `Float[Array, ""]` | No | Cap/floor rate K. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `is_cap` | `bool` | Yes | True for caplet, False for floorlet. Not differentiable. |
| `day_count` | `str` | Yes | Day count convention for accrual fractions. |

**Notes**:

- Caplet pays `max(F - K, 0) × τ × notional` at `end_date`; floorlet pays `max(K - F, 0) × τ × notional`.
- `F` is the simply-compounded forward rate over `[start_date, end_date]`.
- Building block for caps and floors.

## `Cap`

```python
from valax.instruments import Cap

class Cap(eqx.Module):
    fixing_dates: Int[Array, "n"]                       # rate fixing dates (ordinals)
    start_dates: Int[Array, "n"]                        # accrual period start dates (ordinals)
    end_dates: Int[Array, "n"]                          # accrual period end dates = payment dates (ordinals)
    strike: Float[Array, ""]                            # uniform cap/floor rate K
    notional: Float[Array, ""]                          # notional principal
    is_cap: bool = eqx.field(static=True, default=True)        # True=cap, False=floor
    day_count: str = eqx.field(static=True, default="act_360") # day count convention
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `fixing_dates` | `Int[Array, "n"]` | No | Rate fixing date for each period (ordinals). |
| `start_dates` | `Int[Array, "n"]` | No | Accrual period start dates (ordinals). |
| `end_dates` | `Int[Array, "n"]` | No | Accrual period end dates = payment dates (ordinals). |
| `strike` | `Float[Array, ""]` | No | Uniform cap/floor rate K applied to all periods. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `is_cap` | `bool` | Yes | True for cap, False for floor. Not differentiable. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- A strip of caplets (or floorlets) over a payment schedule.
- Each period independently pays `max(F_i - K, 0) × τ_i × notional` (cap) or `max(K - F_i, 0) × τ_i × notional` (floor).
- Cap price equals the sum of the individual caplet prices.

## `InterestRateSwap`

```python
from valax.instruments import InterestRateSwap

class InterestRateSwap(eqx.Module):
    start_date: Int[Array, ""]                          # effective date = first floating reset (ordinal)
    fixed_dates: Int[Array, "n"]                        # fixed leg payment dates including maturity (ordinals)
    fixed_rate: Float[Array, ""]                        # annual fixed coupon rate
    notional: Float[Array, ""]                          # notional principal
    pay_fixed: bool = eqx.field(static=True, default=True)     # True=pay fixed / receive float
    day_count: str = eqx.field(static=True, default="act_360") # day count convention
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `start_date` | `Int[Array, ""]` | No | Swap effective date = first floating reset date (ordinal). |
| `fixed_dates` | `Int[Array, "n"]` | No | Fixed leg payment dates including maturity (ordinals). |
| `fixed_rate` | `Float[Array, ""]` | No | Annual fixed coupon rate (e.g., 0.05 for 5%). Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `pay_fixed` | `bool` | Yes | True = pay fixed / receive float (payer swap). Not differentiable. |
| `day_count` | `str` | Yes | Day count convention for fixed leg accrual. |

**Notes**:

- Vanilla fixed-for-floating interest rate swap.
- Floating leg PV uses the replication identity: `PV(float) = notional × (DF(start_date) − DF(maturity))`.
- `pay_fixed=True` is a payer swap; `pay_fixed=False` is a receiver swap.

## `Swaption`

```python
from valax.instruments import Swaption

class Swaption(eqx.Module):
    expiry_date: Int[Array, ""]                         # option expiry = underlying swap start (ordinal)
    fixed_dates: Int[Array, "n"]                        # underlying swap fixed leg payment dates (ordinals)
    strike: Float[Array, ""]                            # fixed rate in the underlying swap
    notional: Float[Array, ""]                          # notional principal
    is_payer: bool = eqx.field(static=True, default=True)      # True=payer swaption
    day_count: str = eqx.field(static=True, default="act_360") # day count convention
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `expiry_date` | `Int[Array, ""]` | No | Option expiry = underlying swap start date (ordinal). |
| `fixed_dates` | `Int[Array, "n"]` | No | Underlying swap fixed leg payment dates (ordinals). |
| `strike` | `Float[Array, ""]` | No | Fixed rate in the underlying swap. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `is_payer` | `bool` | Yes | True = payer swaption (right to pay fixed, receive float). Not differentiable. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- European option to enter a vanilla fixed-for-float interest rate swap at expiry.
- Payer swaption profits when rates rise; receiver swaption profits when rates fall.
- The underlying swap starts on `expiry_date` with fixed leg payments on `fixed_dates`.

## `BermudanSwaption`

```python
from valax.instruments import BermudanSwaption

class BermudanSwaption(eqx.Module):
    exercise_dates: Int[Array, "n_exercise"]            # dates when exercise is allowed (ordinals)
    fixed_dates: Int[Array, "n_periods"]                # full set of fixed leg payment dates (ordinals)
    strike: Float[Array, ""]                            # fixed rate K
    notional: Float[Array, ""]                          # notional principal
    is_payer: bool = eqx.field(static=True, default=True)      # True=payer Bermudan
    day_count: str = eqx.field(static=True, default="act_360") # day count convention
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `exercise_dates` | `Int[Array, "n_exercise"]` | No | Dates at which exercise is allowed (ordinals). |
| `fixed_dates` | `Int[Array, "n_periods"]` | No | Full set of fixed leg payment dates; `fixed_dates[-1]` is swap maturity (ordinals). |
| `strike` | `Float[Array, ""]` | No | Fixed rate K. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `is_payer` | `bool` | Yes | True = right to pay fixed / receive float (payer Bermudan). Not differentiable. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Bermudan option: exercisable at any date in `exercise_dates` (multiple discrete opportunities).
- If exercised at `exercise_dates[e]`, the holder enters a tail swap with fixed payments on `fixed_dates[e:]`.
- Exercise dates must align with the LMM tenor structure (typically the start of each swap period).

## `OISSwap`

```python
from valax.instruments import OISSwap

class OISSwap(eqx.Module):
    start_date: Int[Array, ""]         # effective date (ordinal)
    fixed_dates: Int[Array, "n"]       # fixed leg dates (ordinals)
    float_dates: Int[Array, "m"]       # floating leg dates (ordinals)
    fixed_rate: Float[Array, ""]       # annual fixed rate
    notional: Float[Array, ""]         # notional principal
    pay_fixed: bool = eqx.field(static=True, default=True)
    compounding: str = eqx.field(static=True, default="compounded")  # "compounded" or "averaged"
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `start_date` | `Int[Array, ""]` | No | Effective date as ordinal. |
| `fixed_dates` | `Int[Array, "n"]` | No | Fixed leg payment dates as ordinals. |
| `float_dates` | `Int[Array, "m"]` | No | Floating leg payment dates as ordinals. |
| `fixed_rate` | `Float[Array, ""]` | No | Annual fixed rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `pay_fixed` | `bool` | Yes | True if paying fixed, receiving floating. |
| `compounding` | `str` | Yes | Floating leg compounding method: `"compounded"` or `"averaged"`. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Fixed vs. compounded overnight rate swap (e.g., SOFR OIS).
- The floating leg compounds daily overnight rates over each accrual period.
- `compounding="averaged"` uses arithmetic average instead of geometric compounding.

## `CrossCurrencySwap`

```python
from valax.instruments import CrossCurrencySwap

class CrossCurrencySwap(eqx.Module):
    start_date: Int[Array, ""]           # effective date (ordinal)
    payment_dates: Int[Array, "n"]       # payment dates (ordinals)
    maturity_date: Int[Array, ""]        # maturity date (ordinal)
    domestic_notional: Float[Array, ""]  # domestic currency notional
    foreign_notional: Float[Array, ""]   # foreign currency notional
    basis_spread: Float[Array, ""]       # spread on domestic floating leg
    exchange_notional: bool = eqx.field(static=True, default=True)  # True if notional exchange occurs
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `start_date` | `Int[Array, ""]` | No | Effective date as ordinal. |
| `payment_dates` | `Int[Array, "n"]` | No | Payment dates as ordinals. |
| `maturity_date` | `Int[Array, ""]` | No | Maturity date as ordinal. |
| `domestic_notional` | `Float[Array, ""]` | No | Domestic currency notional. Differentiable. |
| `foreign_notional` | `Float[Array, ""]` | No | Foreign currency notional. Differentiable. |
| `basis_spread` | `Float[Array, ""]` | No | Spread on the domestic floating leg. Differentiable. |
| `exchange_notional` | `bool` | Yes | True if notional amounts are exchanged at start and maturity. |
| `currency_pair` | `str` | Yes | Currency pair identifier (e.g., `"EUR/USD"`). |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Two-currency basis swap: each leg pays floating rate in its own currency, plus a basis spread on one leg.
- Notional exchange at inception and maturity exposes the swap to FX risk.
- The basis spread reflects the cross-currency funding premium.

## `TotalReturnSwap`

```python
from valax.instruments import TotalReturnSwap

class TotalReturnSwap(eqx.Module):
    start_date: Int[Array, ""]         # effective date (ordinal)
    payment_dates: Int[Array, "n"]     # reset dates (ordinals)
    notional: Float[Array, ""]         # notional principal
    funding_spread: Float[Array, ""]   # spread over floating rate
    is_total_return_receiver: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `start_date` | `Int[Array, ""]` | No | Effective date as ordinal. |
| `payment_dates` | `Int[Array, "n"]` | No | Reset/payment dates as ordinals. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `funding_spread` | `Float[Array, ""]` | No | Spread over the floating reference rate. Differentiable. |
| `is_total_return_receiver` | `bool` | Yes | True if receiving total return, paying funding. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Total return receiver gets asset appreciation + income; pays floating rate + spread.
- Provides synthetic exposure to the reference asset without ownership.
- Resets periodically; gains and losses are settled at each payment date.

## `CMSSwap`

```python
from valax.instruments import CMSSwap

class CMSSwap(eqx.Module):
    start_date: Int[Array, ""]         # effective date (ordinal)
    payment_dates: Int[Array, "n"]     # payment dates (ordinals)
    fixed_rate: Float[Array, ""]       # fixed leg rate
    notional: Float[Array, ""]         # notional principal
    cms_tenor: int = eqx.field(static=True, default=10)       # reference swap rate tenor in years
    pay_fixed: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `start_date` | `Int[Array, ""]` | No | Effective date as ordinal. |
| `payment_dates` | `Int[Array, "n"]` | No | Payment dates as ordinals. |
| `fixed_rate` | `Float[Array, ""]` | No | Fixed leg rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `cms_tenor` | `int` | Yes | Tenor of the reference constant maturity swap rate in years (e.g., 10). |
| `pay_fixed` | `bool` | Yes | True if paying fixed, receiving CMS floating. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Floating leg is linked to a constant maturity swap rate (e.g., 10Y swap rate).
- Requires a CMS convexity adjustment to account for the timing/payment mismatch.
- `cms_tenor` is static as it determines the reference rate used for each fixing.

## `CMSCapFloor`

```python
from valax.instruments import CMSCapFloor

class CMSCapFloor(eqx.Module):
    payment_dates: Int[Array, "n"]     # payment dates (ordinals)
    strike: Float[Array, ""]           # cap/floor strike rate
    notional: Float[Array, ""]         # notional principal
    cms_tenor: int = eqx.field(static=True, default=10)       # CMS rate tenor in years
    is_cap: bool = eqx.field(static=True, default=True)       # True for cap, False for floor
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Payment dates as ordinals. |
| `strike` | `Float[Array, ""]` | No | Cap or floor strike rate. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `cms_tenor` | `int` | Yes | CMS rate tenor in years (e.g., 10). |
| `is_cap` | `bool` | Yes | True for cap, False for floor. Not differentiable. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Cap/floor on CMS rates: each caplet pays `max(CMS_rate - strike, 0) × notional × dcf`.
- Requires a CMS convexity adjustment and replication-based pricing.
- Used to hedge or express views on the level of long-term swap rates.

## `RangeAccrual`

```python
from valax.instruments import RangeAccrual

class RangeAccrual(eqx.Module):
    payment_dates: Int[Array, "n"]     # payment dates (ordinals)
    coupon_rate: Float[Array, ""]      # maximum coupon rate
    lower_barrier: Float[Array, ""]    # lower range bound
    upper_barrier: Float[Array, ""]    # upper range bound
    notional: Float[Array, ""]         # notional principal
    reference_index: str = eqx.field(static=True, default="rate")  # "rate", "cms", or "fx"
    day_count: str = eqx.field(static=True, default="act_360")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Payment dates as ordinals. |
| `coupon_rate` | `Float[Array, ""]` | No | Maximum coupon rate (accrued when index is in range). Differentiable. |
| `lower_barrier` | `Float[Array, ""]` | No | Lower bound of the accrual range. Differentiable. |
| `upper_barrier` | `Float[Array, ""]` | No | Upper bound of the accrual range. Differentiable. |
| `notional` | `Float[Array, ""]` | No | Notional principal. Differentiable. |
| `reference_index` | `str` | Yes | Reference index type: `"rate"`, `"cms"`, or `"fx"`. |
| `day_count` | `str` | Yes | Day count convention. |

**Notes**:

- Coupon accrues only on days when the reference index fixes within `[lower_barrier, upper_barrier]`.
- Effective coupon = `coupon_rate × (days_in_range / total_days)` per period.
- Can be decomposed into a portfolio of daily digital options for pricing.
- `reference_index` is static as it determines which market data and model to use.
