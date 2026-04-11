"""Fixed income instrument definitions (data-only pytrees)."""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array


class ZeroCouponBond(eqx.Module):
    """Zero-coupon bond — single cash flow at maturity.

    Attributes:
        maturity: Maturity date as ordinal (days since epoch).
        face_value: Par/face value paid at maturity.
    """

    maturity: Int[Array, ""]
    face_value: Float[Array, ""]


class FixedRateBond(eqx.Module):
    """Fixed-rate coupon bond.

    Data-only pytree describing the bond contract. Payment dates
    and the settlement date are stored as integer ordinals.

    Attributes:
        payment_dates: Ordinal dates of coupon payments (includes maturity).
        settlement_date: Valuation/settlement date as ordinal.
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%).
        face_value: Par/face value.
        frequency: Coupons per year (1, 2, or 4).
        day_count: Day count convention for accrual.
    """

    payment_dates: Int[Array, " n_payments"]
    settlement_date: Int[Array, ""]
    coupon_rate: Float[Array, ""]
    face_value: Float[Array, ""]
    frequency: int = eqx.field(static=True, default=2)
    day_count: str = eqx.field(static=True, default="act_365")


class FloatingRateBond(eqx.Module):
    """Floating-rate note (FRN) — coupon resets periodically off a reference rate.

    Each coupon payment is:

    .. math::

        C_i = N \\cdot (F_i + s) \\cdot \\tau_i

    where :math:`F_i` is the reference rate (e.g., SOFR, EURIBOR) observed
    at ``fixing_dates[i]``, :math:`s` is the fixed spread, and
    :math:`\\tau_i` is the accrual day count fraction for the period.

    For **seasoned** FRNs, past fixings are known and stored in
    ``fixing_rates``.  For future periods, fixings are projected from
    the forward curve.

    Attributes:
        payment_dates: Coupon payment dates as ordinals (shape n).
        fixing_dates: Rate fixing/observation dates as ordinals (shape n).
        settlement_date: Valuation/settlement date as ordinal.
        spread: Fixed spread over the reference rate (e.g., 0.005 for 50 bps).
        face_value: Par/face value.
        fixing_rates: Known past fixing rates (shape n), NaN for future fixings.
            If None, all fixings are projected from the forward curve.
        frequency: Coupon resets per year (1, 2, or 4). Default is quarterly.
        day_count: Day count convention for accrual fractions.
    """

    payment_dates: Int[Array, " n"]
    fixing_dates: Int[Array, " n"]
    settlement_date: Int[Array, ""]
    spread: Float[Array, ""]
    face_value: Float[Array, ""]
    fixing_rates: Optional[Float[Array, " n"]] = None
    frequency: int = eqx.field(static=True, default=4)
    day_count: str = eqx.field(static=True, default="act_360")


class CallableBond(eqx.Module):
    """Callable fixed-rate bond — issuer has the right to redeem early.

    A standard fixed-rate bond with embedded call option(s).  At each
    call date the issuer may redeem the bond at the call price (often
    par or a small premium).

    Pricing requires a model that handles the embedded optionality,
    typically a short-rate model (Hull-White) with backward induction
    on a tree or PDE.  The **option-adjusted spread (OAS)** is the
    constant spread added to the model discount curve that equates the
    model price to the market price.

    Attributes:
        payment_dates: Coupon payment dates as ordinals (shape n).
        settlement_date: Valuation/settlement date as ordinal.
        coupon_rate: Annual fixed coupon rate.
        face_value: Par/face value.
        call_dates: Dates on which the issuer may call (ordinals, shape m).
        call_prices: Redemption price at each call date (shape m).
            Typically par (1.0) or a small premium above par.
        frequency: Coupons per year (1, 2, or 4).
        day_count: Day count convention for accrual fractions.
    """

    payment_dates: Int[Array, " n"]
    settlement_date: Int[Array, ""]
    coupon_rate: Float[Array, ""]
    face_value: Float[Array, ""]
    call_dates: Int[Array, " m"]
    call_prices: Float[Array, " m"]
    frequency: int = eqx.field(static=True, default=2)
    day_count: str = eqx.field(static=True, default="act_365")


class PuttableBond(eqx.Module):
    """Puttable fixed-rate bond — holder has the right to sell back early.

    A standard fixed-rate bond with embedded put option(s).  At each
    put date the bondholder may redeem the bond at the put price,
    forcing the issuer to repay early.

    Put options protect the investor against rising rates or credit
    deterioration.  The embedded put increases the bond value relative
    to an otherwise identical non-puttable bond.

    Pricing mechanics are analogous to callable bonds: short-rate model
    with backward induction, but the exercise decision belongs to the
    bondholder (maximizing bond value) rather than the issuer.

    Attributes:
        payment_dates: Coupon payment dates as ordinals (shape n).
        settlement_date: Valuation/settlement date as ordinal.
        coupon_rate: Annual fixed coupon rate.
        face_value: Par/face value.
        put_dates: Dates on which the holder may put (ordinals, shape m).
        put_prices: Redemption price at each put date (shape m).
        frequency: Coupons per year (1, 2, or 4).
        day_count: Day count convention for accrual fractions.
    """

    payment_dates: Int[Array, " n"]
    settlement_date: Int[Array, ""]
    coupon_rate: Float[Array, ""]
    face_value: Float[Array, ""]
    put_dates: Int[Array, " m"]
    put_prices: Float[Array, " m"]
    frequency: int = eqx.field(static=True, default=2)
    day_count: str = eqx.field(static=True, default="act_365")


class ConvertibleBond(eqx.Module):
    """Convertible bond — fixed-rate bond with equity conversion option.

    The bondholder may convert the bond into a fixed number of shares
    of the issuer's equity at any time (American-style) or at specific
    dates.  The conversion ratio determines how many shares are received
    per unit of face value.

    Convertible bonds are equity-credit hybrids: their value depends on
    the stock price (equity component), credit quality (credit spread),
    and interest rates.  Pricing typically uses a PDE or tree model with
    both equity and credit dimensions.

    Attributes:
        payment_dates: Coupon payment dates as ordinals (shape n).
        settlement_date: Valuation/settlement date as ordinal.
        coupon_rate: Annual fixed coupon rate.
        face_value: Par/face value.
        conversion_ratio: Number of shares received per unit of face value.
        call_dates: Issuer call dates (ordinals, shape m). Empty if non-callable.
        call_prices: Issuer call prices (shape m).
        frequency: Coupons per year (1, 2, or 4).
        day_count: Day count convention for accrual fractions.
    """

    payment_dates: Int[Array, " n"]
    settlement_date: Int[Array, ""]
    coupon_rate: Float[Array, ""]
    face_value: Float[Array, ""]
    conversion_ratio: Float[Array, ""]
    call_dates: Optional[Int[Array, " m"]] = None
    call_prices: Optional[Float[Array, " m"]] = None
    frequency: int = eqx.field(static=True, default=2)
    day_count: str = eqx.field(static=True, default="act_365")
