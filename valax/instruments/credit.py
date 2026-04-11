"""Credit derivative instrument definitions (data-only pytrees).

Credit derivatives transfer credit risk between counterparties.  The
fundamental building block is the **Credit Default Swap (CDS)**, which
provides insurance against default of a reference entity.

Key concepts:

- **Survival curve**: probability that the reference entity has *not*
  defaulted by a given date.  Bootstrapped from CDS spreads.
- **Hazard rate**: instantaneous default intensity.  The survival
  probability is ``P(τ > t) = exp(-∫₀ᵗ h(s) ds)``.
- **Recovery rate**: fraction of notional recovered in the event of
  default (typically 40% for senior unsecured corporates).
"""

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class CDS(eqx.Module):
    """Credit Default Swap — protection against default of a reference entity.

    **Protection leg:** The protection buyer receives
    ``notional * (1 - recovery)`` if the reference entity defaults
    before maturity.

    **Premium leg:** The protection buyer pays a periodic spread
    (premium) on the notional.  Payments accrue and are made at
    ``premium_dates`` until default or maturity, whichever comes first.

    The **fair spread** (par CDS spread) is the rate that makes the
    NPV of the contract zero at inception:

    .. math::

        \\text{PV(protection leg)} = \\text{PV(premium leg)}

    Attributes:
        effective_date: Contract start date as ordinal.
        maturity_date: Contract maturity date as ordinal.
        premium_dates: Premium payment dates as ordinals (shape n).
        spread: Annual CDS spread (e.g., 0.01 = 100 bps).
        notional: Notional principal amount.
        recovery_rate: Expected recovery rate (e.g., 0.4 for 40%).
        is_protection_buyer: True if this position is buying protection
            (short credit risk).
        premium_frequency: Premiums per year (1, 2, or 4). Default is
            quarterly (market standard).
        day_count: Day count convention for premium accrual.
    """

    effective_date: Int[Array, ""]
    maturity_date: Int[Array, ""]
    premium_dates: Int[Array, " n"]
    spread: Float[Array, ""]
    notional: Float[Array, ""]
    recovery_rate: Float[Array, ""]
    is_protection_buyer: bool = eqx.field(static=True, default=True)
    premium_frequency: int = eqx.field(static=True, default=4)
    day_count: str = eqx.field(static=True, default="act_360")


class CDOTranche(eqx.Module):
    """Collateralized Debt Obligation tranche — credit portfolio exposure.

    A CDO pools credit risk from a portfolio of reference entities and
    distributes losses across tranches ordered by seniority.  Each
    tranche absorbs losses within a ``[attachment, detachment]`` range.

    For example, a 3%-7% tranche absorbs portfolio losses between 3%
    and 7% of the total notional.  The **equity tranche** (0%-3%)
    takes the first losses; the **senior tranche** takes the last.

    Pricing typically uses the **Gaussian copula** model with a single
    correlation parameter (**base correlation**).

    Attributes:
        effective_date: Trade date as ordinal.
        maturity_date: Maturity date as ordinal.
        premium_dates: Premium payment dates as ordinals (shape n).
        attachment: Lower attachment point (e.g., 0.03 for 3%).
        detachment: Upper detachment point (e.g., 0.07 for 7%).
        spread: Running premium spread on tranche notional.
        notional: Total portfolio notional.
        n_names: Number of reference entities in the portfolio.
        recovery_rate: Assumed uniform recovery rate.
        day_count: Day count convention for premium accrual.
    """

    effective_date: Int[Array, ""]
    maturity_date: Int[Array, ""]
    premium_dates: Int[Array, " n"]
    attachment: Float[Array, ""]
    detachment: Float[Array, ""]
    spread: Float[Array, ""]
    notional: Float[Array, ""]
    recovery_rate: Float[Array, ""]
    n_names: int = eqx.field(static=True, default=125)
    day_count: str = eqx.field(static=True, default="act_360")
