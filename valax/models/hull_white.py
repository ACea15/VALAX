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


def _log_df_grid(
    curve: DiscountCurve,
) -> tuple[Float[Array, " n_nodes"], Float[Array, " n_nodes"]]:
    """Log-DF interpolation nodes, anchored at the reference date.

    ``DiscountCurve`` interpolates log-linearly with *flat* extrapolation.  A
    curve whose first pillar sits strictly after the reference date therefore
    has zero log-DF slope on :math:`[0, t_1)`, which silently drives the
    instantaneous forward :func:`_instantaneous_forward` to zero at the short
    end.  Prepending the no-arbitrage anchor :math:`\\ln P^M(0, 0) = 0` removes
    that artefact.

    When the curve already carries a ``t = 0`` pillar the anchor is a duplicate
    node holding the same value, so the interpolant is unchanged.

    Args:
        curve: Initial market discount curve.

    Returns:
        Pair of ``(times, log_discount_factors)`` interpolation nodes.
    """
    pillar_t = _pillar_times(curve)
    log_dfs = jnp.log(curve.discount_factors)
    return (
        jnp.concatenate([jnp.zeros((1,), dtype=pillar_t.dtype), pillar_t]),
        jnp.concatenate([jnp.zeros((1,), dtype=log_dfs.dtype), log_dfs]),
    )


def _log_df_at_time(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Log discount factor :math:`\\ln P^M(0, t)` at year-fraction *t*.

    Interpolates in log-DF space exactly as the ``DiscountCurve`` does
    internally, but accepts a continuous year-fraction argument rather
    than an integer ordinal date, and grounds the curve at ``t = 0``.
    """
    times, log_dfs = _log_df_grid(model.initial_curve)
    return jnp.interp(t, times, log_dfs)


def _market_df(
    model: HullWhiteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(model.initial_curve)
    return jnp.exp(jnp.interp(t, times, log_dfs))


def _instantaneous_forward(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Instantaneous forward rate :math:`f^M(0, t) = -d\\ln P^M(0,t)/dt`.

    Computed via ``jax.grad`` of the log-DF interpolation, giving exact
    piecewise-constant forwards for a log-linear curve.
    """
    return -jax.grad(lambda s: _log_df_at_time(model, s))(t)


# ── Public curve accessors (continuous time) ──────────────────────────

def hw_market_df(
    model: HullWhiteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at a year fraction.

    ``DiscountCurve.__call__`` takes integer ordinal dates; this accessor takes
    a continuous year fraction, which is the natural coordinate for the
    model's analytics.  Both interpolate log-linearly on the same pillars and
    agree wherever the two coordinates coincide.

    Args:
        model: Hull-White model carrying the initial curve.
        t: Year fraction(s) from the curve reference date.

    Returns:
        Market discount factor at each ``t``.
    """
    return _market_df(model, t)


def hw_instantaneous_forward(
    model: HullWhiteModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Market instantaneous forward rate :math:`f^M(0, t)`.

    Args:
        model: Hull-White model carrying the initial curve.
        t: Year fraction from the curve reference date.

    Returns:
        Instantaneous forward rate at ``t``.
    """
    return _instantaneous_forward(model, t)


def hw_alpha(
    model: HullWhiteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Deterministic exact-fit shift :math:`\\alpha(t)` in :math:`r = x + \\alpha`.

    Hull-White admits a clean split of the short rate into a *centred*
    Ornstein-Uhlenbeck state and a deterministic shift that carries the whole
    dependence on the initial curve:

    .. math::

        r(t) = x(t) + \\alpha(t), \\qquad
        dx(t) = -a\\,x(t)\\,dt + \\sigma\\,dW(t), \\quad x(0) = 0

    .. math::

        \\alpha(t) = f^M(0, t)
            + \\frac{\\sigma^2}{2a^2}\\bigl(1 - e^{-at}\\bigr)^2

    This is the parameterisation every numerical scheme wants: the state
    variable :math:`x` is a zero-mean OU process with *time-independent*
    drift and diffusion, so a lattice, a PDE mesh, or an exact Monte-Carlo
    step can be built once on :math:`x` and shifted by :math:`\\alpha` to
    recover the discount rate. It is the continuous-time analogue of the
    trinomial tree's :math:`\\alpha_i` calibration shifts, and it makes the
    exact fit to the initial curve a closed-form translation rather than a
    numerical forward induction.

    At :math:`t = 0` the second term vanishes and
    :math:`\\alpha(0) = f^M(0, 0) = r(0)`.

    Args:
        model: Hull-White model carrying the initial curve and parameters.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The shift :math:`\\alpha(t)` at each ``t``.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §3.3 (eq. 3.30).
    """
    a = model.mean_reversion
    sigma = model.volatility
    t_arr = jnp.asarray(t)
    forward = jnp.vectorize(lambda s: _instantaneous_forward(model, s))(t_arr)
    convexity = (sigma**2 / (2.0 * a**2)) * (1.0 - jnp.exp(-a * t_arr)) ** 2
    return forward + convexity


def _convexity_integral(
    a: Float[Array, ""],
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """:math:`\\int_0^t (1 - e^{-as})^2\\,ds`, the antiderivative behind
    :func:`hw_alpha_average`'s convexity term."""
    return (
        t
        - 2.0 * (1.0 - jnp.exp(-a * t)) / a
        + (1.0 - jnp.exp(-2.0 * a * t)) / (2.0 * a)
    )


def hw_alpha_average(
    model: HullWhiteModel,
    t0: Float[Array, "..."],
    t1: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Exact time-average of :math:`\\alpha` over ``[t0, t1]``.

    .. math::

        \\bar\\alpha(t_0, t_1)
            = \\frac{1}{t_1 - t_0}\\int_{t_0}^{t_1} \\alpha(s)\\,ds

    Both halves are integrated in closed form. The market forward term
    telescopes into a ratio of discount factors,

    .. math::

        \\int_{t_0}^{t_1} f^M(0, s)\\,ds
            = \\ln\\frac{P^M(0, t_0)}{P^M(0, t_1)},

    and the convexity term integrates analytically via
    :math:`\\int_0^t (1 - e^{-as})^2 ds`.

    **Why this matters for discretised schemes.** Sampling :math:`\\alpha` at a
    step midpoint is only second-order accurate when :math:`\\alpha` is smooth.
    It is not: a log-linear discount curve has a *piecewise-constant*
    instantaneous forward that jumps at every pillar, so midpoint sampling
    leaves an :math:`O(\\Delta t)` error on each step straddling a pillar and a
    scheme built on it **stalls** — empirically a Hull-White PDE plateaus at a
    ~4e-6 zero-coupon-bond repricing error no matter how finely time is
    refined, while the same scheme on a flat curve converges cleanly at
    second order. Averaging exactly restores the model's defining exact-fit
    property for an arbitrary curve shape, because each step then discounts by
    precisely the market forward discount factor across it.

    Args:
        model: Hull-White model carrying the initial curve and parameters.
        t0: Start of the averaging interval, in year fractions.
        t1: End of the averaging interval, in year fractions.

    Returns:
        The average of :math:`\\alpha` over each ``[t0, t1]`` interval.
    """
    a = model.mean_reversion
    sigma = model.volatility
    dt = t1 - t0

    forward_avg = (_log_df_at_time(model, t0) - _log_df_at_time(model, t1)) / dt
    convexity_avg = (sigma**2 / (2.0 * a**2)) * (
        _convexity_integral(a, t1) - _convexity_integral(a, t0)
    ) / dt
    return forward_avg + convexity_avg


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
