# Credit Derivatives

!!! warning "Planned — not yet implemented"
    Credit derivatives (CDS, CDO) are on the VALAX roadmap (Priority P2.4, Tier 3.4).
    Survival curves, hazard rate bootstrapping, and CDS pricing are prerequisites for
    the full XVA suite (CVA in particular). The content below documents the intended
    interface and pricing approach. See the [Roadmap](../roadmap.md) — Priority 2 (Production Pricing Capabilities) and Tier 3.4 (Credit Derivatives).

Credit derivatives transfer **credit risk** — the risk that a borrower defaults —
between counterparties without transferring the underlying debt instrument.
The fundamental building block is the **survival curve**: the probability
$P(\tau > t)$ that the reference entity has not defaulted by time $t$. The curve
is characterized by a **hazard rate** $h(t)$ such that

$$P(\tau > t) = \exp\!\left(-\int_0^t h(s)\,ds\right)$$

Survival curves are bootstrapped from observed CDS spreads. The **recovery rate**
$R$ (typically 40 % for senior unsecured corporates) determines the loss-given-default.

## Credit Default Swap (CDS)

**Market context.** The Credit Default Swap is the most liquid credit derivative,
with roughly \$3.8 trillion notional outstanding globally. A CDS is economically
equivalent to credit insurance: the **protection buyer** pays a periodic premium
(the *spread*) in exchange for a contingent payment if the reference entity defaults.
CDS spreads are the primary observable for single-name credit risk and feed into
index products (CDX, iTraxx), structured credit, and CVA calculations.

### Cashflows

A CDS has two legs:

| Leg | Cashflow | Timing |
|-----|----------|--------|
| **Protection leg** | $N \cdot (1 - R)$ | At default (if $\tau < T$) |
| **Premium leg** | $s \cdot N \cdot \tau_i \cdot P(\tau > t_i)$ per period | Quarterly to maturity |
| **Accrued premium** | Fractional spread from last coupon date to default date | At default |

where $N$ is the notional, $R$ the recovery rate, $s$ the annual spread, and
$\tau_i$ the day count fraction for period $i$.

### Fair Spread

The **par CDS spread** is the value of $s$ that makes the NPV zero at inception.
Setting the present value of the protection leg equal to the premium leg:

$$s = \frac{\displaystyle\sum_{i=1}^{n} \mathrm{DF}(t_i) \cdot q(t_i) \cdot \Delta t_i \cdot (1 - R)}{\displaystyle\sum_{i=1}^{n} \mathrm{DF}(t_i) \cdot P(\tau > t_i) \cdot \Delta\tau_i}$$

where:

- $\mathrm{DF}(t_i)$ — risk-free discount factor to time $t_i$
- $q(t_i)$ — marginal default probability density in period $i$, i.e.,
  $q(t_i) \approx P(\tau > t_{i-1}) - P(\tau > t_i)$
- $P(\tau > t_i)$ — survival probability to $t_i$
- $R$ — recovery rate
- $\Delta\tau_i$ — day count fraction for premium accrual in period $i$

### Pricing Approaches

1. **Deterministic rates + bootstrapped survival curve.** The standard market
   approach. Hazard rates are piece-wise constant, bootstrapped from quoted CDS
   spreads. Interest rate and credit risk are assumed independent.
2. **Reduced-form model with stochastic hazard rates.** Required for wrong-way
   risk, CVA, and portfolio credit models where default correlation matters.

See [Analytical Pricing](analytical.md) for the deterministic-rates implementation
and [Monte Carlo](monte-carlo.md) for stochastic hazard rate simulation.

### Code Example

```python
from valax.instruments import CDS
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

cds = CDS(
    effective_date=ymd_to_ordinal(2025, 3, 20),
    maturity_date=ymd_to_ordinal(2030, 3, 20),
    premium_dates=generate_schedule(2025, 6, 20, 2030, 3, 20, frequency=4),
    spread=jnp.array(0.01),       # 100 bps annual spread
    notional=jnp.array(10_000_000.0),
    recovery_rate=jnp.array(0.4), # 40% — market standard for senior unsecured
    is_protection_buyer=True,
    day_count="act_360",
)
```

### Key Conventions

| Convention | Detail |
|------------|--------|
| Standard maturities | Mar 20, Jun 20, Sep 20, Dec 20 (IMM dates) |
| Premium frequency | Quarterly (4× per year) |
| Day count | ACT/360 |
| Recovery rate | 40 % (senior unsecured), 25 % (subordinated), 35 % (senior secured) |
| Quoting | Index CDS: upfront points + 100 bps or 500 bps running. Single-name: running spread |
| Business day | Modified Following |

!!! note
    The ISDA standard model uses piece-wise flat hazard rates and deterministic
    discounting. VALAX's planned CDS pricer will be consistent with the ISDA CDS
    Standard Model (also known as the ISDA Calculator).

---

## CDO Tranche

**Market context.** A Collateralized Debt Obligation (CDO) pools credit risk from
a portfolio of reference entities (typically 125 names in standard index tranches)
and distributes losses to **tranches** ordered by seniority. Each tranche absorbs
portfolio losses within an **attachment** $a$ and **detachment** $d$ range. The
equity tranche (e.g., 0–3 %) takes the first losses and receives the highest
spread; the super-senior tranche (e.g., 15–100 %) is exposed only to catastrophic
losses.

### Tranche Loss

Given a portfolio loss fraction $L$, the loss absorbed by the tranche $[a, d]$ is:

$$\text{Tranche loss}(L) = \min\!\bigl(\max(L - a,\; 0),\; d - a\bigr)$$

The tranche notional is $N \cdot (d - a)$ where $N$ is the total portfolio notional.
The tranche coupon is paid on the *outstanding* tranche notional (reduced by losses).

### Gaussian Copula Pricing

The standard market model is the **one-factor Gaussian copula**. The joint default
probability of $n$ names is:

$$P(\text{joint default}) = \Phi_n\!\bigl(\Phi^{-1}(q_1),\, \ldots,\, \Phi^{-1}(q_n);\; \Sigma\bigr)$$

where $q_i$ is the marginal default probability of name $i$, $\Phi$ the standard
normal CDF, and $\Sigma$ the correlation matrix. Under the one-factor assumption
with uniform pairwise correlation $\rho$:

$$P(\text{default}_i \mid M) = \Phi\!\left(\frac{\Phi^{-1}(q_i) - \sqrt{\rho}\, M}{\sqrt{1 - \rho}}\right)$$

where $M \sim \mathcal{N}(0,1)$ is the common factor. The conditional independence
simplifies the portfolio loss distribution to a mixture of independent Bernoulli trials.

The market convention is to quote **base correlation**: for each detachment point,
the implied correlation that reprices the $[0, d]$ tranche. Base correlation is
monotonically increasing in $d$ (unlike compound correlation, which can be
non-monotonic).

### Code Example

```python
from valax.instruments import CDOTranche
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 3%-7% mezzanine tranche on a 125-name portfolio
tranche = CDOTranche(
    effective_date=ymd_to_ordinal(2025, 3, 20),
    maturity_date=ymd_to_ordinal(2030, 3, 20),
    premium_dates=generate_schedule(2025, 6, 20, 2030, 3, 20, frequency=4),
    attachment=jnp.array(0.03),     # 3%
    detachment=jnp.array(0.07),     # 7%
    spread=jnp.array(0.05),         # 500 bps running
    notional=jnp.array(500_000_000.0),
    recovery_rate=jnp.array(0.4),
    n_names=125,
    day_count="act_360",
)
```

!!! warning
    The Gaussian copula model is known for its limitations (static correlation,
    inability to capture correlation skew). It remains the market standard for
    quoting and relative value, but should not be used for absolute risk measurement
    without supplementary stress testing.
