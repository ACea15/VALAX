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

from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.models.short_rate import (
    as_buckets,
    check_buckets,
    decayed_integral,
    default_bucket_times,
    is_constant,
    piecewise_integral,
    require_initial_curve,
    volatility_bucket_fields,
)


class HullWhiteModel(eqx.Module):
    """Hull-White one-factor model with piecewise-constant volatility.

    The short-rate volatility :math:`\\sigma(t)` is piecewise constant on
    buckets (see :mod:`valax.models.short_rate`): ``volatility[j]`` applies on
    :math:`[\\tau_j, \\tau_{j+1})` with :math:`\\tau_0 = 0`, the interior
    breakpoints are ``volatility_times`` and the last value extends flat. A
    constant volatility is one bucket; passing a scalar ``volatility``
    promotes it to one, and every closed form then runs its original
    constant-parameter formula.

    Attributes:
        mean_reversion: Mean-reversion speed :math:`a` (positive scalar).
        volatility: Short-rate volatility per bucket (positive).
        initial_curve: Initial discount curve :math:`P^M(0, t)` used for
            exact-fit :math:`\\theta(t)` calibration.
            ``None`` (the default) leaves the model unbound:
            :func:`valax.core.pricing.price` binds it from the market (see
            :mod:`valax.models.short_rate`).
        curve_id: Market id of the curve to bind, or ``None`` to use the
            instrument's discount curve. Static.
        volatility_times: Interior bucket breakpoints in year fractions from
            the curve's reference date, increasing; empty for one bucket.
            Data, not a calibrated parameter (it is set from the trade's
            exercise dates by the co-terminal bootstrap).
    """

    mean_reversion: Float[Array, ""]
    volatility: Float[Array, " n_buckets"]
    initial_curve: DiscountCurve | None
    curve_id: str | None = eqx.field(static=True)
    volatility_times: Float[Array, " n_buckets_minus_1"]

    def __init__(
        self,
        mean_reversion: Float[Array, ""],
        volatility: Float[Array, " n_buckets"] | Float[Array, ""],
        initial_curve: DiscountCurve | None = None,
        curve_id: str | None = None,
        *,
        volatility_times: Float[Array, " n_buckets_minus_1"] | None = None,
    ):
        """Build a model; a scalar ``volatility`` becomes one bucket.

        Args:
            mean_reversion: Mean-reversion speed :math:`a`.
            volatility: Volatility per bucket, or a scalar for a constant one.
            initial_curve: Bound initial curve, or ``None`` (bound by
                ``price``).
            curve_id: Market id of the curve to bind (static).
            volatility_times: Interior breakpoints (``n_buckets - 1``);
                required for more than one bucket.
        """
        self.mean_reversion = mean_reversion
        self.volatility = as_buckets(volatility)
        self.initial_curve = initial_curve
        self.curve_id = curve_id
        self.volatility_times = (
            default_bucket_times(self.volatility)
            if volatility_times is None else volatility_times
        )

    def __check_init__(self) -> None:
        """Validate the bucket structure (shapes only)."""
        check_buckets("HullWhiteModel", self.volatility, self.volatility_times)


@volatility_bucket_fields.register(HullWhiteModel)
def _hw_bucket_fields(model: HullWhiteModel) -> tuple[str, ...]:
    return ("volatility",)


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
    times, log_dfs = _log_df_grid(require_initial_curve(model))
    return jnp.interp(t, times, log_dfs)


def _market_df(
    model: HullWhiteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(require_initial_curve(model))
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

    With a piecewise-constant :math:`\\sigma(t)` the convexity term is the
    general

    .. math::

        \\int_0^t \\sigma(u)^2 e^{-a(t-u)} B(u, t)\\,du
            = \\frac{I_a(t) - I_{2a}(t)}{a},
        \\qquad I_k(t) = \\int_0^t \\sigma(u)^2 e^{-k(t-u)}\\,du,

    which reduces to the expression above for a constant :math:`\\sigma`.

    Args:
        model: Hull-White model carrying the initial curve and parameters.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The shift :math:`\\alpha(t)` at each ``t``.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §3.3 (eq. 3.30).
    """
    a = model.mean_reversion
    t_arr = jnp.asarray(t)
    forward = jnp.vectorize(lambda s: _instantaneous_forward(model, s))(t_arr)
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        convexity = (sigma**2 / (2.0 * a**2)) * (1.0 - jnp.exp(-a * t_arr)) ** 2
        return forward + convexity
    return forward + _convexity(model, t_arr)


def _variance_integrals(
    model: HullWhiteModel,
    rate: Float[Array, ""],
    t: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """``(I_k(t), ∫_0^t I_k)`` for :math:`v = \\sigma^2` at decay rate ``k``."""
    return decayed_integral(
        model.volatility**2, model.volatility_times, rate, t
    )


def _convexity(model: HullWhiteModel, t: Float[Array, "..."]) -> Float[Array, "..."]:
    """Exact-fit convexity :math:`(I_a(t) - I_{2a}(t))/a` (piecewise σ)."""
    a = model.mean_reversion
    return (
        _variance_integrals(model, a, t)[0]
        - _variance_integrals(model, 2.0 * a, t)[0]
    ) / a


def _convexity_antiderivative(
    model: HullWhiteModel, t: Float[Array, "..."]
) -> Float[Array, "..."]:
    """:math:`\\int_0^t` of the exact-fit convexity (piecewise σ)."""
    a = model.mean_reversion
    return (
        _variance_integrals(model, a, t)[1]
        - _variance_integrals(model, 2.0 * a, t)[1]
    ) / a


def hw_state_variance(
    model: HullWhiteModel,
    s: Float[Array, "..."],
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Conditional variance of the centred state :math:`x(t)` given :math:`x(s)`.

    .. math::

        V(s, t) = \\int_s^t \\sigma(u)^2 e^{-2a(t-u)}\\,du
                = I_{2a}(t) - e^{-2a(t-s)}\\,I_{2a}(s),

    which is :math:`\\frac{\\sigma^2}{2a}(1 - e^{-2a(t-s)})` for a constant
    :math:`\\sigma`. It is the exact Monte Carlo step variance and the
    variance a trinomial-tree level must match.

    Args:
        model: Hull-White model.
        s: Start time(s) in year fractions.
        t: End time(s) in year fractions (``t >= s``).

    Returns:
        The conditional variance.
    """
    a = model.mean_reversion
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        return (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * (t - s)))
    i_t = _variance_integrals(model, 2.0 * a, t)[0]
    i_s = _variance_integrals(model, 2.0 * a, s)[0]
    return i_t - jnp.exp(-2.0 * a * (t - s)) * i_s


def hw_mean_variance_rate(
    model: HullWhiteModel,
    t0: Float[Array, "..."],
    t1: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Time average of :math:`\\sigma(t)^2` over ``[t0, t1]``.

    The diffusion coefficient a discretised scheme should use for a step that
    may straddle a bucket breakpoint (exact for the step's total variance
    rate).

    Args:
        model: Hull-White model.
        t0: Start of each interval in year fractions.
        t1: End of each interval in year fractions (``t1 > t0``).

    Returns:
        :math:`\\frac{1}{t_1 - t_0}\\int_{t_0}^{t_1}\\sigma(u)^2\\,du`.
    """
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        return jnp.broadcast_to(sigma**2, jnp.broadcast_shapes(
            jnp.shape(t0), jnp.shape(t1)
        ))
    sq, times = model.volatility**2, model.volatility_times
    return (
        piecewise_integral(sq, times, t1) - piecewise_integral(sq, times, t0)
    ) / (t1 - t0)


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
    dt = t1 - t0

    forward_avg = (_log_df_at_time(model, t0) - _log_df_at_time(model, t1)) / dt
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        convexity_avg = (sigma**2 / (2.0 * a**2)) * (
            _convexity_integral(a, t1) - _convexity_integral(a, t0)
        ) / dt
    else:
        convexity_avg = (
            _convexity_antiderivative(model, t1)
            - _convexity_antiderivative(model, t0)
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

    B = hw_B(a, T - t)
    f_t = _instantaneous_forward(model, t)

    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)

    if is_constant(model.volatility):
        # Kept in the original evaluation order: reordering the terms leaves
        # the value unchanged but reorders reverse-mode accumulation, which
        # moved a gradient by one ulp.
        sigma = model.volatility[0]
        ln_A = (
            ln_PM_T - ln_PM_t
            + B * f_t
            - (sigma**2 / (4.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t)) * B**2
        )
    else:
        # General form: (1/2) B^2 Var[x(t)] with Var[x(t)] = I_{2a}(t).
        ln_A = (
            ln_PM_T - ln_PM_t
            + B * f_t
            - 0.5 * B**2 * hw_short_rate_variance(model, t)
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

    for a constant :math:`\\sigma`, and
    :math:`I_{2a}(t) = \\int_0^t \\sigma(u)^2 e^{-2a(t-u)}\\,du` in general.

    Args:
        model: Hull-White model.
        t: Time(s) in year fractions.

    Returns:
        The variance of :math:`r(t)` (equivalently of the state :math:`x(t)`).
    """
    a = model.mean_reversion
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        return (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t))
    return _variance_integrals(model, 2.0 * a, t)[0]
