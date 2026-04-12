"""Hull-White one-factor short-rate model.

The Hull-White (extended Vasicek) model is the workhorse of rates desks
for pricing instruments with embedded optionality — callable bonds,
puttable bonds, Bermudan swaptions, and IR exotics.  It specifies the
risk-neutral dynamics of the instantaneous short rate:

.. math::

    dr(t) = [\\theta(t) - a\\,r(t)]\\,dt + \\sigma\\,dW(t)

where :math:`a` is the mean-reversion speed, :math:`\\sigma` is the
short-rate volatility, and :math:`\\theta(t)` is a time-dependent drift
calibrated to **exactly fit** the initial discount curve.

Affine structure gives closed-form zero-coupon bond prices conditional
on the short rate at any future time:

.. math::

    P(t, T \\mid r) = A(t, T)\\,e^{-B(t, T)\\,r}

The model is parameterised as an ``equinox.Module`` so it is a valid JAX
pytree — ``jax.grad``, ``jax.jit``, and ``jax.vmap`` all work out of
the box.

References:
    Hull & White (1990), "Pricing Interest-Rate-Derivative Securities".
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 3.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


class HullWhiteModel(eqx.Module):
    """Hull-White one-factor model.

    Attributes:
        mean_reversion: Mean-reversion speed :math:`a` (positive scalar).
        volatility: Short-rate volatility :math:`\\sigma` (positive scalar).
        initial_curve: Initial discount curve :math:`P^M(0, t)` used for
            exact-fit :math:`\\theta(t)` calibration.
    """

    mean_reversion: Float[Array, ""]
    volatility: Float[Array, ""]
    initial_curve: DiscountCurve


# ── Helpers ───────────────────────────────────────────────────────────

def hw_B(
    a: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Mean-reversion decay factor :math:`B(\\tau) = (1 - e^{-a\\tau})/a`.

    For :math:`a \\to 0` this reduces to :math:`\\tau` (Vasicek/Ho-Lee
    limit).  We use a safe formulation for small :math:`a`.
    """
    return (1.0 - jnp.exp(-a * tau)) / a


def _pillar_times(curve: DiscountCurve) -> Float[Array, " n"]:
    """Year fractions from the reference date to each pillar."""
    return year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )


def _log_df_at_time(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Log discount factor :math:`\\ln P^M(0, t)` at year-fraction *t*.

    Interpolates in log-DF space exactly as the ``DiscountCurve`` does
    internally, but accepts a continuous year-fraction argument rather
    than an integer ordinal date.
    """
    pillar_t = _pillar_times(model.initial_curve)
    log_dfs = jnp.log(model.initial_curve.discount_factors)
    return jnp.interp(t, pillar_t, log_dfs)


def _market_df(
    model: HullWhiteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    pillar_t = _pillar_times(model.initial_curve)
    log_dfs = jnp.log(model.initial_curve.discount_factors)
    return jnp.exp(jnp.interp(t, pillar_t, log_dfs))


def _instantaneous_forward(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Instantaneous forward rate :math:`f^M(0, t) = -d\\ln P^M(0,t)/dt`.

    Computed via ``jax.grad`` of the log-DF interpolation, giving exact
    piecewise-constant forwards for a log-linear curve.
    """
    return -jax.grad(lambda s: _log_df_at_time(model, s))(t)


# ── Analytic zero-coupon bond price ───────────────────────────────────

def hw_bond_price(
    model: HullWhiteModel,
    r: Float[Array, ""],
    t: Float[Array, ""],
    T: Float[Array, ""],
) -> Float[Array, ""]:
    """Zero-coupon bond price under Hull-White given short rate *r* at time *t*.

    .. math::

        P(t, T \\mid r) = A(t, T)\\,e^{-B(t, T)\\,r}

    where

    .. math::

        B(t, T) = \\frac{1 - e^{-a(T - t)}}{a}

    .. math::

        \\ln A(t, T) = \\ln\\frac{P^M(0, T)}{P^M(0, t)}
                       + B(t, T)\\,f^M(0, t)
                       - \\frac{\\sigma^2}{4a}(1 - e^{-2at})\\,B(t, T)^2

    (**Exact-fit property**: when :math:`r = f^M(0, 0)` and :math:`t = 0`,
    this recovers the initial curve discount factor :math:`P^M(0, T)`.)

    Args:
        model: Hull-White model (carries initial curve and parameters).
        r: Current short rate (scalar).
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.
    """
    a = model.mean_reversion
    sigma = model.volatility

    B = hw_B(a, T - t)
    f_t = _instantaneous_forward(model, t)

    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)

    ln_A = (
        ln_PM_T - ln_PM_t
        + B * f_t
        - (sigma**2 / (4.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t)) * B**2
    )
    return jnp.exp(ln_A - B * r)


# ── Short-rate distribution ──────────────────────────────────────────

def hw_short_rate_variance(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Variance of the short rate at time *t* (unconditional).

    .. math::

        \\text{Var}[r(t)] = \\frac{\\sigma^2}{2a}(1 - e^{-2at})
    """
    a = model.mean_reversion
    sigma = model.volatility
    return (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t))
