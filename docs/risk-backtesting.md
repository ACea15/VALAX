# Backtesting & FRTB PLA

*The two regulator-mandated validations that decide whether a desk keeps its internal-models capital treatment. Prerequisite: the P&L vectors built in [VaR & P&L](risk-var.md#pl-vectors-predict-vs-actual). Maths: [Models & Theory § 7.6–7.7](theory/risk-measures.md#76-var-backtesting).*

## VaR Backtesting

A VaR forecast is only as good as its track record. The Basel framework requires 99% one-day VaR to be backtested on a rolling 250-day window, with capital multipliers driven by the breach count. VALAX provides the standard Kupiec, Christoffersen, and traffic-light tools in `valax/risk/backtesting.py`.

For the underlying statistics, see [Models & Theory § 7.6](theory/risk-measures.md#76-var-backtesting).

### Counting breaches

A breach occurs on a day where the realised loss exceeds that day's VaR forecast:

```python
import jax.numpy as jnp
from valax.risk import var_breaches

# Both vectors have shape (n_days,):
#   actual_pnl[t] is realised one-day P&L (negative = loss)
#   var_forecast[t] is that morning's VaR forecast (positive number = loss threshold)
breaches = var_breaches(actual_pnl, var_forecast)
# breaches.shape == (n_days,), boolean
n_breaches = int(jnp.sum(breaches))
```

### Kupiec proportion-of-failures (POF)

Tests whether the *count* of breaches is consistent with the VaR confidence level:

```python
from valax.risk import kupiec_pof

result = kupiec_pof(breaches, confidence=0.99)
result["n"]          # number of observation days
result["x"]          # number of breaches
result["lr_uc"]      # likelihood-ratio statistic, χ²₁ under H₀
result["p_value"]    # right-tail p-value
```

Reject the model at the 5% level if `lr_uc > 3.84` (or equivalently `p_value < 0.05`).

### Christoffersen independence and conditional coverage

Detects *clustering* of breaches (a model that breaches twice in a week then never again):

```python
from valax.risk import christoffersen_independence, christoffersen_conditional_coverage

ind = christoffersen_independence(breaches)
ind["lr_ind"]        # χ²₁ statistic, tests pi_01 = pi_11
ind["p_value"]

cc = christoffersen_conditional_coverage(breaches, confidence=0.99)
cc["lr_cc"]          # χ²₂ — joint test of correct rate AND independence
cc["p_value"]
```

### Basel traffic light

The regulatory zoning is just a count-based lookup on the cumulative binomial:

```python
from valax.risk import basel_traffic_light

zone = basel_traffic_light(n_breaches=n_breaches, n_obs=250, confidence=0.99)
# zone in {"green", "yellow", "red"}
```

For a 250-day window at 99% VaR: 0–4 breaches green, 5–9 yellow, ≥10 red — the multiplier on capital scales accordingly.

---

## FRTB P&L Attribution Test

The FRTB Internal Models Approach requires a second daily validation on top of the VaR backtest: each desk must demonstrate that its **risk-theoretical P&L (RTPL)** tracks its **hypothetical P&L (HPL)** in both rank order (Spearman) and distribution (Kolmogorov–Smirnov). Failing the PLA test forces the desk off internal models and onto the (usually more punitive) standardized approach.

For the regulatory background and threshold derivation, see [Models & Theory § 7.7](theory/risk-measures.md#77-frtb-pl-attribution-test).

### Spearman rank correlation

```python
from valax.risk import pla_spearman

rho = pla_spearman(rtpl_series, hpl_series)
# rho is a scalar in [-1, 1]
```

Tests monotonic agreement: do RTPL and HPL produce the same ordering of best-to-worst days?

### Kolmogorov–Smirnov statistic

```python
from valax.risk import pla_ks, ks_statistic

D = pla_ks(rtpl_series, hpl_series)
# D is the max distance between empirical CDFs in [0, 1]
```

`ks_statistic(x, y)` is the underlying two-sample KS computation, exposed for general use.

### Traffic-light zone

```python
from valax.risk import pla_traffic_light

zone = pla_traffic_light(spearman=rho, ks_stat=D, n_obs=len(rtpl_series))
# zone in {"green", "amber", "red"}
```

The zone applies the BCBS d558 thresholds: green if Spearman ≥ 0.80 *and* KS $p$-value ≥ 0.264; red if either Spearman < 0.70 or KS $p$-value < 0.055; amber otherwise.

### End-to-end PLA workflow

Putting the pieces together: compute the ladder once, build both vectors over the same scenario set (the 250-day historical window), and read the test outputs:

```python
import jax.numpy as jnp
from valax.risk import (
    compute_ladder,
    risk_theoretical_pnl_vector,
    hypothetical_pnl_vector,
    pla_spearman,
    pla_ks,
    pla_traffic_light,
)

# scenarios: 250-day historical ScenarioSet
ladder = compute_ladder(pricing_fn, instruments, base_market)
rtpl   = risk_theoretical_pnl_vector(ladder, scenarios, base_market)
hpl    = hypothetical_pnl_vector(pricing_fn, instruments, base_market, scenarios)

rho = pla_spearman(rtpl, hpl)
D   = pla_ks(rtpl, hpl)
zone = pla_traffic_light(rho, D, n_obs=rtpl.shape[0])
print(f"Spearman={float(rho):.3f}, KS={float(D):.3f}, zone={zone}")
```

If the zone is red, inspect the unexplained vector first — large systematic residuals point to missing risk factors or third-order convexity that the ladder doesn't capture.
