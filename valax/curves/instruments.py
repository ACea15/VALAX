"""Bootstrap instrument definitions (data-only pytrees).

These represent market quotes used as inputs to curve construction,
not tradeable instruments for pricing. They live in ``curves/``
rather than ``instruments/`` because they are curve-building inputs,
keeping the dependency graph clean.
"""

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class DepositRate(eqx.Module):
    """Money market deposit rate quote.

    The implied discount factor is:

        DF(end) = DF(start) / (1 + rate * tau(start, end))

    When start_date == reference_date, DF(start) = 1.

    Attributes:
        start_date: Deposit effective date (ordinal).
        end_date: Deposit maturity date (ordinal).
        rate: Simply-compounded deposit rate.
        day_count: Day count convention.
    """

    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")


class FRA(eqx.Module):
    """Forward Rate Agreement quote.

    The implied discount factor relationship is:

        DF(end) = DF(start) / (1 + rate * tau(start, end))

    Requires DF(start) to be known from a prior instrument.

    Attributes:
        start_date: FRA effective date (ordinal).
        end_date: FRA maturity date (ordinal).
        rate: Simply-compounded forward rate.
        day_count: Day count convention.
    """

    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")


class SwapRate(eqx.Module):
    """Par swap rate quote for curve bootstrap.

    The par condition is:

        rate * A = DF(start) - DF(maturity)

    where A = sum_i(tau_i * DF(T_i)) is the fixed-leg annuity.

    Distinct from ``InterestRateSwap`` in ``valax.instruments.rates``,
    which represents an actual swap contract with notional and direction.

    Attributes:
        start_date: Swap effective date (ordinal).
        fixed_dates: Fixed leg payment dates including maturity (ordinal, shape n).
        rate: Par swap rate (annualized).
        day_count: Day count convention for fixed leg accrual.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
