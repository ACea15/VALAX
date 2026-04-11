"""Inflation derivative instrument definitions (data-only pytrees).

Inflation derivatives are linked to a **Consumer Price Index (CPI)** or
similar inflation index.  The two fundamental swap types are:

- **Zero-coupon inflation swap (ZCIS):** single exchange at maturity of
  the cumulative inflation return vs. a fixed rate.
- **Year-on-year inflation swap (YYIS):** periodic exchanges of annual
  inflation vs. a fixed rate.

Inflation caps and floors are option overlays on these swap structures.

Key conventions:

- Inflation indices are published with a lag (typically 2-3 months).
- CPI ratios are used (``CPI(T) / CPI(0)``), not absolute index levels.
- Seasonality adjustments may be required for monthly indices.
"""

from typing import Optional

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class ZeroCouponInflationSwap(eqx.Module):
    """Zero-coupon inflation swap (ZCIS).

    At maturity, the inflation receiver gets:

    .. math::

        N \\cdot \\left(\\frac{\\text{CPI}(T)}{\\text{CPI}(0)} - 1\\right)

    and pays the fixed leg:

    .. math::

        N \\cdot \\left((1 + K)^T - 1\\right)

    where :math:`K` is the fixed rate and :math:`T` is the swap tenor
    in years.  NPV = 0 at inception when :math:`K` equals the
    break-even inflation rate.

    Attributes:
        effective_date: Swap start date as ordinal.
        maturity_date: Maturity / payment date as ordinal.
        fixed_rate: Annual fixed (break-even) rate.
        notional: Notional principal.
        base_cpi: CPI index level at inception (CPI(0)).
        is_inflation_receiver: True = receive inflation / pay fixed.
        index_lag: Publication lag of the CPI index in months.
        day_count: Day count convention.
    """

    effective_date: Int[Array, ""]
    maturity_date: Int[Array, ""]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    base_cpi: Float[Array, ""]
    is_inflation_receiver: bool = eqx.field(static=True, default=True)
    index_lag: int = eqx.field(static=True, default=3)
    day_count: str = eqx.field(static=True, default="act_act")


class YearOnYearInflationSwap(eqx.Module):
    """Year-on-year inflation swap (YYIS).

    Periodic exchange of annual inflation returns vs. a fixed rate.
    At each payment date :math:`t_i`:

    - **Inflation leg** pays:
      :math:`N \\cdot \\left(\\frac{\\text{CPI}(t_i)}{\\text{CPI}(t_{i-1})} - 1\\right)`
    - **Fixed leg** pays: :math:`N \\cdot K \\cdot \\tau_i`

    where :math:`\\tau_i` is the day count fraction.

    YoY swaps have a **convexity adjustment** relative to zero-coupon
    swaps because the periodic CPI ratio expectations differ from
    the ratio of expected CPIs.

    Attributes:
        effective_date: Swap start date as ordinal.
        payment_dates: Payment dates as ordinals (shape n).
        fixed_rate: Annual fixed rate.
        notional: Notional principal.
        base_cpi: CPI level at inception.
        is_inflation_receiver: True = receive inflation / pay fixed.
        index_lag: Publication lag in months.
        day_count: Day count convention.
    """

    effective_date: Int[Array, ""]
    payment_dates: Int[Array, " n"]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    base_cpi: Float[Array, ""]
    is_inflation_receiver: bool = eqx.field(static=True, default=True)
    index_lag: int = eqx.field(static=True, default=3)
    day_count: str = eqx.field(static=True, default="act_act")


class InflationCapFloor(eqx.Module):
    """Inflation cap or floor — option on year-on-year CPI returns.

    An inflation cap is a strip of **inflation caplets**, each paying:

    .. math::

        N \\cdot \\max\\left(\\frac{\\text{CPI}(t_i)}{\\text{CPI}(t_{i-1})} - 1 - K,\\; 0\\right)

    An inflation floor pays the put-side equivalent.  Priced via
    Black-76 on the forward year-on-year inflation rate.

    Attributes:
        effective_date: Start date as ordinal.
        payment_dates: Caplet/floorlet payment dates (ordinals, shape n).
        strike: Strike inflation rate (e.g., 0.02 for 2%).
        notional: Notional principal.
        base_cpi: CPI level at inception.
        is_cap: True for inflation cap, False for inflation floor.
        index_lag: Publication lag in months.
        day_count: Day count convention.
    """

    effective_date: Int[Array, ""]
    payment_dates: Int[Array, " n"]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    base_cpi: Float[Array, ""]
    is_cap: bool = eqx.field(static=True, default=True)
    index_lag: int = eqx.field(static=True, default=3)
    day_count: str = eqx.field(static=True, default="act_act")
