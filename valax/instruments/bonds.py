"""Fixed income instrument definitions (data-only pytrees)."""

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
