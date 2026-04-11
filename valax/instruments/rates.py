"""Interest rate derivative instrument definitions (data-only pytrees)."""

from typing import Optional

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array


class Caplet(eqx.Module):
    """Single caplet or floorlet.

    Pays max(F - K, 0) * tau * notional at end_date (caplet) or
    max(K - F, 0) * tau * notional (floorlet), where F is the
    simply-compounded forward rate over [start_date, end_date].

    Attributes:
        fixing_date: Date when the reference rate is observed (ordinal).
        start_date: Accrual period start date (ordinal).
        end_date: Accrual period end date = payment date (ordinal).
        strike: Cap/floor rate K.
        notional: Notional principal.
        is_cap: True for caplet, False for floorlet.
        day_count: Day count convention for accrual fractions.
    """

    fixing_date: Int[Array, ""]
    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    is_cap: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class Cap(eqx.Module):
    """Cap or floor: a strip of caplets/floorlets over a payment schedule.

    Attributes:
        fixing_dates: Rate fixing date for each period (ordinal array, shape n).
        start_dates: Accrual period start dates (ordinal array, shape n).
        end_dates: Accrual period end dates = payment dates (ordinal array, shape n).
        strike: Uniform cap/floor rate K applied to all periods.
        notional: Notional principal.
        is_cap: True for cap, False for floor.
        day_count: Day count convention.
    """

    fixing_dates: Int[Array, " n"]
    start_dates: Int[Array, " n"]
    end_dates: Int[Array, " n"]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    is_cap: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class InterestRateSwap(eqx.Module):
    """Vanilla fixed-for-floating interest rate swap.

    Uses the standard replication identity for the floating leg:
        PV(float leg) = notional * (DF(start_date) - DF(maturity))

    Attributes:
        start_date: Swap effective date = first floating reset date (ordinal).
        fixed_dates: Fixed leg payment dates including maturity (ordinal, shape n).
        fixed_rate: Annual fixed coupon rate (e.g., 0.05 for 5%).
        notional: Notional principal.
        pay_fixed: True = pay fixed / receive float (payer swap).
        day_count: Day count convention for fixed leg accrual.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    pay_fixed: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class Swaption(eqx.Module):
    """European option to enter into a vanilla fixed-for-float interest rate swap.

    The underlying swap starts on expiry_date. On that date, the holder
    can choose to enter as payer (pay fixed / receive float) or receiver.

    Attributes:
        expiry_date: Option expiry = underlying swap start date (ordinal).
        fixed_dates: Underlying swap fixed leg payment dates (ordinal, shape n).
        strike: Fixed rate in the underlying swap.
        notional: Notional principal.
        is_payer: True = payer swaption (right to pay fixed, receive float).
        day_count: Day count convention.
    """

    expiry_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    is_payer: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class CMSSwap(eqx.Module):
    """Constant Maturity Swap (CMS) — floating leg linked to a swap rate.

    One leg pays a **CMS rate** (e.g., the 10Y swap rate observed at
    each fixing date), while the other pays a fixed rate (or another
    floating index).

    CMS rates require a **convexity adjustment** because the expectation
    of a swap rate under the payment measure differs from the forward
    swap rate.  This adjustment depends on the volatility smile and is
    typically computed via static replication (Hagan) or SABR integration.

    Attributes:
        start_date: Effective date (ordinal).
        payment_dates: Payment dates for both legs (ordinals, shape n).
        fixed_rate: Fixed leg rate.
        cms_tenor: Tenor of the reference swap rate in years
            (e.g., 10 for the 10Y rate).
        notional: Notional principal.
        pay_fixed: True = pay fixed / receive CMS.
        day_count: Day count convention.
    """

    start_date: Int[Array, ""]
    payment_dates: Int[Array, " n"]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    cms_tenor: int = eqx.field(static=True, default=10)
    pay_fixed: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class CMSCapFloor(eqx.Module):
    """CMS cap or floor — option on CMS rates.

    A strip of caplets/floorlets on a CMS rate:

    - CMS caplet: ``max(CMS_rate(t_i) - K, 0) * tau * notional``
    - CMS floorlet: ``max(K - CMS_rate(t_i), 0) * tau * notional``

    Like CMS swaps, pricing requires a convexity/timing adjustment
    beyond standard Black-76.

    Attributes:
        payment_dates: Payment dates (ordinals, shape n).
        strike: Cap/floor strike rate.
        notional: Notional principal.
        cms_tenor: Tenor of the CMS rate in years.
        is_cap: True for cap, False for floor.
        day_count: Day count convention.
    """

    payment_dates: Int[Array, " n"]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    cms_tenor: int = eqx.field(static=True, default=10)
    is_cap: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class RangeAccrual(eqx.Module):
    """Range accrual note — coupon accrues while index stays in range.

    The coupon for each period is proportional to the **fraction of
    business days** during which the reference index (e.g., SOFR,
    CMS rate, FX rate) is within the range ``[lower_barrier, upper_barrier]``:

    .. math::

        C_i = N \\cdot R \\cdot \\tau_i \\cdot \\frac{n_{\\text{in range}}}{n_{\\text{total}}}

    Range accruals are popular in structured rates and FX products.
    Pricing requires daily path monitoring via MC or semi-analytic
    methods with copula adjustments.

    Attributes:
        payment_dates: Coupon payment dates (ordinals, shape n).
        coupon_rate: Maximum coupon rate (if always in range).
        lower_barrier: Lower bound of the accrual range.
        upper_barrier: Upper bound of the accrual range.
        notional: Notional principal.
        reference_index: Type of reference index (static): ``"rate"``,
            ``"cms"``, or ``"fx"``.
        day_count: Day count convention.
    """

    payment_dates: Int[Array, " n"]
    coupon_rate: Float[Array, ""]
    lower_barrier: Float[Array, ""]
    upper_barrier: Float[Array, ""]
    notional: Float[Array, ""]
    reference_index: str = eqx.field(static=True, default="rate")
    day_count: str = eqx.field(static=True, default="act_360")


class OISSwap(eqx.Module):
    """Overnight Index Swap (OIS) — fixed vs. compounded overnight rate.

    The dominant post-LIBOR swap type.  The floating leg pays the
    **compounded** daily overnight rate (SOFR, €STR, SONIA) over each
    accrual period, while the fixed leg pays a fixed rate.

    Daily compounding means the floating leg value depends on the
    *product* of daily forward rates, not a simple average.  For the
    current accrual period, realized fixings are used for past dates
    and forward rates for future dates.

    Attributes:
        start_date: Swap effective date (ordinal).
        fixed_dates: Fixed leg payment dates (ordinals, shape n).
        float_dates: Floating leg payment dates (ordinals, shape m).
            May differ from fixed_dates if payment frequencies differ.
        fixed_rate: Annual fixed rate.
        notional: Notional principal.
        pay_fixed: True = pay fixed / receive overnight compounded.
        compounding: Compounding convention: ``"compounded"`` (standard)
            or ``"averaged"`` (simple average, used in some markets).
        day_count: Day count convention.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    float_dates: Int[Array, " m"]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    pay_fixed: bool = eqx.field(static=True, default=True)
    compounding: str = eqx.field(static=True, default="compounded")
    day_count: str = eqx.field(static=True, default="act_360")


class CrossCurrencySwap(eqx.Module):
    """Cross-currency basis swap — exchange of floating rates in two currencies.

    Counterparties exchange:

    1. **Initial notional exchange**: at inception, each side pays
       notional in its respective currency at the prevailing spot rate.
    2. **Periodic coupons**: domestic floating rate + basis spread vs.
       foreign floating rate.
    3. **Final notional re-exchange**: at maturity, notionals are
       returned.

    The **basis spread** reflects supply/demand for funding in each
    currency and deviations from covered interest rate parity.

    Attributes:
        start_date: Effective date (ordinal).
        payment_dates: Common payment dates for both legs (ordinals, shape n).
        maturity_date: Final exchange / maturity date (ordinal).
        domestic_notional: Notional in domestic currency.
        foreign_notional: Notional in foreign currency.
        basis_spread: Spread added to the domestic floating leg (e.g., -0.002).
        exchange_notional: True if initial and final notional exchange occurs.
        currency_pair: E.g., ``"EUR/USD"`` (static).
        day_count: Day count convention.
    """

    start_date: Int[Array, ""]
    payment_dates: Int[Array, " n"]
    maturity_date: Int[Array, ""]
    domestic_notional: Float[Array, ""]
    foreign_notional: Float[Array, ""]
    basis_spread: Float[Array, ""]
    exchange_notional: bool = eqx.field(static=True, default=True)
    currency_pair: str = eqx.field(static=True, default="FOR/DOM")
    day_count: str = eqx.field(static=True, default="act_360")


class TotalReturnSwap(eqx.Module):
    """Total Return Swap (TRS) — exchange of asset total return vs. funding rate.

    The total return payer passes all economic exposure (price
    appreciation + income) of a reference asset to the receiver in
    exchange for a funding rate (floating + spread).

    - **Total return leg**: ``(P_end - P_start) / P_start + coupons/dividends``
    - **Funding leg**: ``floating_rate + spread``

    TRS are used for synthetic exposure, funding optimization, and
    prime brokerage.

    Attributes:
        start_date: Effective date (ordinal).
        payment_dates: Reset/payment dates (ordinals, shape n).
        notional: Notional principal.
        funding_spread: Spread over the floating rate on the funding leg.
        is_total_return_receiver: True = receive total return / pay funding.
        day_count: Day count convention for funding leg.
    """

    start_date: Int[Array, ""]
    payment_dates: Int[Array, " n"]
    notional: Float[Array, ""]
    funding_spread: Float[Array, ""]
    is_total_return_receiver: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")


class BermudanSwaption(eqx.Module):
    """Bermudan option to enter a vanilla fixed-for-float interest rate swap.

    The holder may exercise at any date in exercise_dates. If exercised at
    exercise_dates[e], the holder enters a tail swap with fixed leg payments
    on fixed_dates[e:] (the remaining coupons from that point onward).

    Exercise dates must align with the LMM tenor structure. Typically
    exercise_dates[e] = fixed_dates[e] shifted back by one period, or
    equivalently, the exercise dates are the start of each swap period.

    Attributes:
        exercise_dates: Ordinal dates at which exercise is allowed, shape (n_exercise,).
        fixed_dates: Full set of fixed leg payment dates, shape (n_periods,).
                     fixed_dates[-1] is the swap maturity.
        strike: Fixed rate K.
        notional: Notional principal.
        is_payer: True = right to pay fixed / receive float (payer Bermudan).
        day_count: Day count convention.
    """

    exercise_dates: Int[Array, " n_exercise"]
    fixed_dates: Int[Array, " n_periods"]
    strike: Float[Array, ""]
    notional: Float[Array, ""]
    is_payer: bool = eqx.field(static=True, default=True)
    day_count: str = eqx.field(static=True, default="act_360")
