# Equity Exotics

VALAX supports a comprehensive set of exotic equity options beyond the standard
vanillas, barriers, Asians, and lookbacks documented in the core
[Analytical Pricing](analytical.md) and [Monte Carlo](monte-carlo.md) guides.
Each exotic below has unique payoff features that require specialized pricing
methods.

!!! info "Implementation status"
    - **Implemented**: Spread option (Margrabe / Kirk), barrier, Asian, lookback,
      variance swap (see [Analytical Pricing](analytical.md) and
      [Monte Carlo](monte-carlo.md)).
    - **Planned**: Digital, Compound, Chooser, Autocallable, Worst-of-Basket, Cliquet.
      See the [Roadmap (Tier 3.6)](../roadmap.md#36-equity-exotics) for tracking.

## Digital Option

!!! warning "Planned"

**Market context.** A digital (binary) option pays a **fixed cash amount** if the
underlying finishes in-the-money, regardless of how far. Digitals are used in
structured products, sports betting analogues in finance, and as building blocks for
more complex payoffs (e.g., range accruals decompose into strips of digitals).

### Payoff

Cash-or-nothing digital call:

$$C_{\text{digital}} = e^{-rT} \cdot \text{payout} \cdot \Phi(d_2)$$

Cash-or-nothing digital put:

$$P_{\text{digital}} = e^{-rT} \cdot \text{payout} \cdot \Phi(-d_2)$$

where $d_2 = \frac{\ln(S/K) + (r - q - \sigma^2/2)T}{\sigma\sqrt{T}}$ is the
standard Black-Scholes-Merton $d_2$.

### Greeks and Hedging

The digital payoff is **discontinuous** at the strike, which creates singular
Greeks:

- **Delta** has a spike near $K$ (approaches a Dirac delta as $T \to 0$).
- **Gamma** changes sign at the strike.
- **Vega** is also discontinuous — a small change in vol dramatically affects the
  probability mass at the boundary.

In practice, digitals are hedged as tight **call spreads** (or put spreads):

$$\text{Digital call} \approx \frac{1}{\epsilon}\bigl[C(K - \epsilon/2) - C(K + \epsilon/2)\bigr]$$

where $\epsilon$ is a small notional-adjusted spread width. This converts the
discontinuous payoff into a smooth one for hedging purposes.

### Code Example

```python
from valax.instruments import DigitalOption
import jax.numpy as jnp

digital_call = DigitalOption(
    strike=jnp.array(100.0),
    expiry=jnp.array(0.5),       # 6-month expiry
    payout=jnp.array(1_000.0),   # pays $1,000 if ITM
    is_call=True,
)
```

---

## Compound Option

!!! warning "Planned"

**Market context.** A compound option is an **option on an option**. There are four
types: call-on-call (CoC), call-on-put (CoP), put-on-call (PoC), and put-on-put
(PoP). Compound options appear in real options analysis (staged investment decisions),
FX markets (options on currency option premiums), and corporate finance (equity as
a call on firm value, making equity options compound options on assets).

### Pricing (Geske Formula)

The Geske (1979) closed-form solution for a call on a call uses the **bivariate
normal distribution** $N_2$:

$$V = S\, e^{-qT_2}\, N_2\!\bigl(a_1,\, b_1;\; \sqrt{T_1/T_2}\,\bigr) - K_2\, e^{-rT_2}\, N_2\!\bigl(a_2,\, b_2;\; \sqrt{T_1/T_2}\,\bigr) - K_1\, e^{-rT_1}\, N(a_2)$$

where:

- $T_1$ = outer expiry (compound option maturity)
- $T_2$ = inner expiry (underlying option maturity), $T_2 > T_1$
- $K_1$ = outer strike (premium to acquire the underlying option)
- $K_2$ = inner strike (strike of the underlying option)
- $a_{1,2}$ and $b_{1,2}$ are adjusted $d_1, d_2$ terms
- $N_2(\cdot, \cdot; \rho)$ = bivariate normal CDF with correlation
  $\rho = \sqrt{T_1 / T_2}$

The formula requires solving for $S^*$, the critical stock price at $T_1$ at which
the underlying option value equals $K_1$.

### Code Example

```python
from valax.instruments import CompoundOption
import jax.numpy as jnp

# Call on a call: right to buy a 1-year call in 3 months for $5
coc = CompoundOption(
    outer_expiry=jnp.array(0.25),   # 3 months
    outer_strike=jnp.array(5.0),    # premium to acquire underlying option
    inner_expiry=jnp.array(1.0),    # underlying option expires in 1 year
    inner_strike=jnp.array(100.0),  # underlying option strike
    outer_is_call=True,
    inner_is_call=True,
)
```

---

## Chooser Option

!!! warning "Planned"

**Market context.** A chooser option gives the holder the right to **choose**
whether the option becomes a call or a put at a future "choose date." This is
valuable when the direction of the underlying is uncertain but a large move is
expected (e.g., ahead of an earnings announcement, regulatory decision, or
election).

### Simple Chooser Decomposition

For a **simple chooser** (same strike $K$ and same expiry $T$ for both the call
and the put), put-call parity yields an elegant decomposition:

$$V_{\text{chooser}} = C(S, K, T) + e^{-q(T - T_c)}\, P\!\left(S,\; K\, e^{-(r-q)(T - T_c)},\; T_c\right)$$

where $T_c$ is the choose date. The chooser equals a call plus a discounted put
with an adjusted strike and shorter maturity. This means a simple chooser can be
priced in closed form using two Black-Scholes evaluations.

For **complex choosers** (different strikes or expiries for the call and put arms),
no decomposition exists and numerical methods (PDE or MC) are required.

### Code Example

```python
from valax.instruments import ChooserOption
import jax.numpy as jnp

chooser = ChooserOption(
    choose_date=jnp.array(0.25),   # choose in 3 months
    expiry=jnp.array(1.0),         # option expires in 1 year
    strike=jnp.array(100.0),
)
```

---

## Spread Option

!!! success "Implemented (analytic + Monte Carlo)"
    - Analytic: `valax.pricing.analytic.spread.margrabe_price` (exact at $K = 0$), `kirk_price` (approximation for $K \neq 0$), `spread_option_price`.
    - Monte Carlo: via `mc_price_dispatch(SpreadOption, MultiAssetGBMModel, ...)`. See [Monte Carlo guide](monte-carlo.md) and [theory §2.9](../theory.md#29-two-asset-correlated-bsm-and-spread-options).

**Market context.** Spread options pay the difference between two asset prices minus
a fixed strike. They are ubiquitous in commodity markets (**crack spreads**: crude
oil vs. refined products; **spark spreads**: natural gas vs. electricity), equity
relative-value trading, and interest rate markets (CMS spread options).

### Kirk's Approximation

For $K \neq 0$, no exact closed form exists. **Kirk's approximation** treats
$S_2 + K$ as a single asset and applies Black-76:

$$V \approx \text{Black76}(S_1,\; S_2 + K,\; \sigma_{\text{spread}},\; T)$$

where the effective spread volatility is:

$$\sigma_{\text{spread}} = \sqrt{\sigma_1^2 - 2\rho\,\sigma_1\,\sigma_2\,\frac{S_2}{S_2 + K} + \left(\sigma_2\,\frac{S_2}{S_2 + K}\right)^2}$$

- $\sigma_1, \sigma_2$ — volatilities of $S_1$ and $S_2$
- $\rho$ — correlation between $S_1$ and $S_2$

### Margrabe Special Case

When $K = 0$ (pure exchange option), the **Margrabe formula** gives an exact
closed-form solution:

$$V = S_1\, e^{-q_1 T}\, \Phi(d_1) - S_2\, e^{-q_2 T}\, \Phi(d_2)$$

where $d_{1,2}$ use $\sigma_{\text{spread}} = \sqrt{\sigma_1^2 - 2\rho\sigma_1\sigma_2 + \sigma_2^2}$.

### Code Example

```python
from valax.instruments import SpreadOption
from valax.pricing.analytic.spread import margrabe_price, kirk_price
import jax.numpy as jnp

# Crack spread call: profit if gasoline - crude > $5
crack_spread = SpreadOption(
    expiry=jnp.array(0.25),
    strike=jnp.array(5.0),         # spread strike
    notional=jnp.array(1_000.0),
    is_call=True,
)

# Kirk's approximation (handles K > 0 and K = 0 gracefully — prefer this for JIT)
price = kirk_price(
    crack_spread,
    s1=jnp.array(85.0), s2=jnp.array(75.0),
    vol1=jnp.array(0.35), vol2=jnp.array(0.30),
    rho=jnp.array(0.7),
    rate=jnp.array(0.05),
)
```

### Monte Carlo validation

The same spread option can be priced via Monte Carlo under correlated
GBM — this is the standard cross-check for Kirk's approximation and
the only viable approach for path-dependent spread exotics (Asian
spreads, spread barriers, Bermudan spreads).

```python
from valax.models import MultiAssetGBMModel
from valax.pricing.mc import mc_price_dispatch, MCConfig

multi = MultiAssetGBMModel(
    vols=jnp.array([0.35, 0.30]),
    rate=jnp.array(0.05),
    dividends=jnp.zeros(2),
    correlation=jnp.array([[1.0, 0.7], [0.7, 1.0]]),
)

result = mc_price_dispatch(
    crack_spread, multi,
    config=MCConfig(n_paths=50_000, n_steps=50),
    key=jax.random.PRNGKey(42),
    spots=jnp.array([85.0, 75.0]),
)

print(f"Kirk:  {float(kirk_px):.4f}")
print(f"MC:    {float(result.price):.4f} ± {float(result.stderr):.4f}")
```

For $K = 0$ Margrabe is exact and MC should agree to within 3 SE; for
$K \neq 0$ Kirk has a small approximation bias that MC reveals.

---

## Autocallable

!!! warning "Planned"

**Market context.** Autocallable structured notes are among the **most traded
structured products globally**, particularly in Asia (Korea, Japan, Hong Kong)
and Europe. A typical autocallable offers an attractive coupon in exchange for
conditional downside exposure. The product is path-dependent with multiple
observation dates.

### Payoff Mechanics (Step by Step)

At each **observation date** $t_k$ ($k = 1, \ldots, n$):

1. **Autocall check**: If $S(t_k) \geq B_{\text{autocall}} \cdot S_0$, the note
   redeems early at par plus accrued coupons. The investor receives
   $N \cdot (1 + k \cdot c)$ where $c$ is the periodic coupon rate.

2. **Coupon check**: If $S(t_k) \geq B_{\text{coupon}} \cdot S_0$ (but below the
   autocall barrier), a coupon $N \cdot c$ is paid for this period.
   - If `has_memory=True` (phoenix/memory feature): any previously missed coupons
     are also paid.
   - If `has_memory=False`: missed coupons are forfeited.

3. **Knock-in check** (continuous or discrete): If $S(t) < B_{\text{ki}} \cdot S_0$
   at *any* monitoring point, the knock-in put is activated.

4. **At maturity** (if not autocalled):
   - If knock-in was **not** triggered: investor receives par ($N$).
   - If knock-in **was** triggered: investor receives
     $N \cdot \min\!\left(1,\; \frac{S(T)}{K \cdot S_0}\right)$, i.e., exposed
     to downside below the strike.

### Pricing

No closed-form solution exists. Autocallables require **Monte Carlo simulation**
with path monitoring at all observation dates (and continuous monitoring for the
knock-in barrier if specified). Stochastic local volatility (SLV) models are
preferred for accurate barrier pricing.

See [Monte Carlo](monte-carlo.md) for the simulation engine.

### Code Example

```python
from valax.instruments import Autocallable
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

autocallable = Autocallable(
    observation_dates=jnp.array([
        ymd_to_ordinal(2025, 9, 15),
        ymd_to_ordinal(2026, 3, 15),
        ymd_to_ordinal(2026, 9, 15),
        ymd_to_ordinal(2027, 3, 15),
    ]),
    autocall_barrier=jnp.array(1.0),    # 100% of initial spot
    coupon_barrier=jnp.array(0.70),     # 70% of initial spot
    coupon_rate=jnp.array(0.08),        # 8% per period
    ki_barrier=jnp.array(0.60),         # 60% knock-in put barrier
    strike=jnp.array(1.0),             # at-the-money put strike
    notional=jnp.array(100_000.0),
    has_memory=False,
)
```

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `autocall_barrier` | 0.95–1.05 | Early redemption trigger (fraction of $S_0$) |
| `coupon_barrier` | 0.60–0.80 | Coupon payment trigger |
| `ki_barrier` | 0.50–0.70 | Knock-in put barrier |
| `coupon_rate` | 0.05–0.15 | Per-period coupon rate |
| `has_memory` | True/False | Phoenix (memory) coupon feature |

!!! warning
    Autocallables have **significant vega and correlation exposure** that is hidden
    from simple scenario analysis. A 5-point increase in implied volatility can
    reduce the note value substantially due to increased knock-in probability.
    Always run full Greeks analysis via `jax.grad` through the MC pricer.

---

## Worst-of Basket Option

!!! success "Monte Carlo implemented"
    Via `mc_price_dispatch(WorstOfBasketOption, MultiAssetGBMModel, ..., spots=...)`.
    Analytic pricing remains on the roadmap (no tractable closed form for
    $\geq 2$ correlated lognormals).

**Market context.** Worst-of options are common building blocks in structured
products (particularly autocallables referenced to a basket). The payoff depends
on the **worst-performing** asset in a multi-asset basket.

### Payoff

$$\text{Payoff} = N \cdot \max\!\left(\min_i\!\left(\frac{S_i(T)}{S_i(0)}\right) - K,\; 0\right) \quad \text{(call)}$$

where $S_i(0)$ and $S_i(T)$ are the initial and terminal prices of asset $i$,
and $K$ is the strike expressed as a return level.

### Correlation Sensitivity

Worst-of options are **highly correlation-sensitive**:

| Correlation | Call Price | Put Price | Intuition |
|-------------|-----------|-----------|-----------|
| High ($\rho \to 1$) | Higher | Lower | Assets move together — worst performer is close to average |
| Low ($\rho \to 0$) | Lower | Higher | Higher chance at least one asset underperforms badly |

Pricing requires **correlated multi-asset Monte Carlo**. The standard approach is
Cholesky decomposition of the correlation matrix $\Sigma = LL^T$ to generate
correlated Brownian increments: $\mathbf{Z}_{\text{corr}} = L \cdot \mathbf{Z}_{\text{indep}}$.

### Code Example

```python
from valax.instruments import WorstOfBasketOption
import jax.numpy as jnp

# Worst-of put on 3 stocks, strike at 100% (ATM)
wof_put = WorstOfBasketOption(
    expiry=jnp.array(1.0),
    strike=jnp.array(1.0),          # 100% — at-the-money
    notional=jnp.array(1_000_000.0),
    n_assets=3,
    is_call=False,
)

# Correlation matrix and spots passed to the MC pricer, not the instrument
# See Monte Carlo guide for multi-asset simulation
```

### Monte Carlo pricing

```python
from valax.models import MultiAssetGBMModel
from valax.pricing.mc import mc_price_dispatch, MCConfig

# 3-asset correlation matrix
C = jnp.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.4],
    [0.3, 0.4, 1.0],
])
multi = MultiAssetGBMModel(
    vols=jnp.array([0.25, 0.30, 0.35]),
    rate=jnp.array(0.05),
    dividends=jnp.zeros(3),
    correlation=C,
)

wof_put_3 = WorstOfBasketOption(
    expiry=jnp.array(1.0),
    strike=jnp.array(1.0),        # 100% ATM in return space
    notional=jnp.array(1_000_000.0),
    n_assets=3,
    is_call=False,
)

result = mc_price_dispatch(
    wof_put_3, multi,
    config=MCConfig(n_paths=50_000, n_steps=50),
    key=jax.random.PRNGKey(0),
    spots=jnp.array([100.0, 100.0, 100.0]),
)
print(f"Worst-of put: {float(result.price):,.2f} ± {float(result.stderr):,.2f}")

# Correlation-vega — valuable because the put is highly correlation-sensitive.
def price_fn(rho):
    model_r = MultiAssetGBMModel(
        vols=multi.vols, rate=multi.rate, dividends=multi.dividends,
        correlation=C.at[0, 1].set(rho).at[1, 0].set(rho),
    )
    return mc_price_dispatch(
        wof_put_3, model_r,
        config=MCConfig(n_paths=20_000, n_steps=20),
        key=jax.random.PRNGKey(0),
        spots=jnp.array([100.0, 100.0, 100.0]),
    ).price

dprice_drho = jax.grad(price_fn)(jnp.array(0.6))
```

For a worst-of **put**, expect `dprice/drho < 0` — more diversification
(lower correlation) means a higher probability that at least one asset
performs badly, making downside protection more valuable.

---

## Cliquet / Ratchet

!!! warning "Planned"

**Market context.** Cliquet (ratchet) options are popular in structured notes and
insurance products (equity-indexed annuities). The payoff is the sum of
**capped and floored periodic returns** — the investor participates in each period's
return within bounds, making the product sensitive to the **forward volatility smile**.

### Payoff

$$\text{Payoff} = N \cdot \max\!\left(\sum_{i=1}^{n} \min\!\left(\max\!\left(\frac{S(t_i)}{S(t_{i-1})} - 1,\; f\right),\; c\right),\; g\right)$$

where:

- $c$ — local cap (maximum credited return per period)
- $f$ — local floor (minimum credited return per period, often 0 %)
- $g$ — global floor (minimum total accumulated return)
- $S(t_i)/S(t_{i-1})$ — periodic return

### Forward Vol Sensitivity

Cliquets are **forward-starting** options: each period is an option that starts at
the beginning of the period. The value depends on the implied volatility for each
**future** period — not just the current smile. This makes cliquets sensitive to:

- Forward volatility skew
- Term structure of volatility
- Local volatility surface curvature

Standard Black-Scholes with flat vol significantly misprices cliquets. **Local
volatility** or **stochastic local volatility (SLV)** models are required.

### Code Example

```python
from valax.instruments import Cliquet
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

cliquet = Cliquet(
    observation_dates=jnp.array([
        ymd_to_ordinal(2025, 6, 15),
        ymd_to_ordinal(2025, 9, 15),
        ymd_to_ordinal(2025, 12, 15),
        ymd_to_ordinal(2026, 3, 15),
        ymd_to_ordinal(2026, 6, 15),
        ymd_to_ordinal(2026, 9, 15),
        ymd_to_ordinal(2026, 12, 15),
        ymd_to_ordinal(2027, 3, 15),
    ]),
    local_cap=jnp.array(0.04),      # 4% cap per quarter
    local_floor=jnp.array(0.0),     # 0% floor (no negative contribution)
    global_floor=jnp.array(0.0),    # 0% global floor
    notional=jnp.array(1_000_000.0),
)
```

!!! tip
    To check model sensitivity, price the same cliquet under Black-Scholes (flat
    vol), local volatility, and Heston. The difference reveals the model risk
    inherent in the product. Use `jax.vmap` to batch across models efficiently.
