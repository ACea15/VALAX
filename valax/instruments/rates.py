"""Interest rate derivative instrument definitions (data-only pytrees)."""

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
