"""G2++ two-additive-factor Gaussian short-rate model.

G2++ is the two-factor generalisation of Hull-White (equivalent to the
two-factor Hull-White / HW-2F model).  Adding a second stochastic factor
lets rates of different maturities **decorrelate** — something the
one-factor model structurally cannot do, since there every rate is driven
by a single shock and all tenor rates move in lock-step.  This unlocks
decorrelation-sensitive instruments: CMS-spread options, steepeners and
flatteners, spread range-accruals, and Bermudan swaptions whose value
depends on the joint dynamics of short and long rates.

The short rate is the sum of two mean-reverting factors plus a
deterministic shift that exact-fits the initial curve:

.. math::

    r(t) = x(t) + y(t) + \\varphi(t)

.. math::

    dx(t) = -a\\,x(t)\\,dt + \\sigma\\,dW_1(t), \\quad x(0) = 0

    dy(t) = -b\\,y(t)\\,dt + \\eta\\,dW_2(t), \\quad y(0) = 0

    dW_1(t)\\,dW_2(t) = \\rho\\,dt

Here :math:`a, b` are the mean-reversion speeds, :math:`\\sigma, \\eta`
the factor volatilities, and :math:`\\rho \\in (-1, 1)` the instantaneous
correlation between the two driving Brownian motions.  **The tenor-rate
decorrelation the model delivers is not the same as** :math:`\\rho`:
:math:`\\rho` is an input, whereas the correlation between rates of
different maturities is an output that also depends on the split between
:math:`a` and :math:`b`.  Genuine decorrelation requires :math:`|\\rho| <
1` — at :math:`\\rho = \\pm 1` the two Brownian motions collapse to a
single shock and the model degenerates to a one-factor world.

The model stays affine/Gaussian, so zero-coupon bond prices are
closed-form conditional on the two factors:

.. math::

    P(t, T \\mid x, y)
        = \\frac{P^M(0, T)}{P^M(0, t)}
          \\exp\\!\\Bigl[\\tfrac{1}{2}\\bigl(V(t, T) - V(0, T) + V(0, t)\\bigr)
                        - B(a, t, T)\\,x - B(b, t, T)\\,y\\Bigr]

The model is an ``equinox.Module`` and hence a valid JAX pytree —
``jax.grad``, ``jax.jit``, and ``jax.vmap`` all work out of the box.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 4 (§4.2).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


class G2PPModel(eqx.Module):
    """G2++ two-factor Gaussian short-rate model.

    Attributes:
        mean_reversion_x: Mean-reversion speed :math:`a` of the first
            factor (positive scalar).
        mean_reversion_y: Mean-reversion speed :math:`b` of the second
            factor (positive scalar).
        volatility_x: Volatility :math:`\\sigma` of the first factor
            (positive scalar).
        volatility_y: Volatility :math:`\\eta` of the second factor
            (positive scalar).
        correlation: Instantaneous correlation :math:`\\rho \\in (-1, 1)`
            between the two driving Brownian motions.
        initial_curve: Initial discount curve :math:`P^M(0, t)` used for
            exact-fit :math:`\\varphi(t)` calibration.
    """

    mean_reversion_x: Float[Array, ""]
    mean_reversion_y: Float[Array, ""]
    volatility_x: Float[Array, ""]
    volatility_y: Float[Array, ""]
    correlation: Float[Array, ""]
    initial_curve: DiscountCurve


# ── Helpers ───────────────────────────────────────────────────────────

def g2pp_B(
    z: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Mean-reversion decay factor :math:`B(z, \\tau) = (1 - e^{-z\\tau})/z`.

    The same shape function :func:`~valax.models.hull_white.hw_B` carries
    per factor; G2++ evaluates it once with :math:`z = a` and once with
    :math:`z = b`.  For :math:`z \\to 0` it reduces to :math:`\\tau`.

    Args:
        z: Mean-reversion speed of the factor.
        tau: Time to maturity :math:`T - t` (year fractions).

    Returns:
        The decay factor :math:`B(z, \\tau)`.
    """
    return (1.0 - jnp.exp(-z * tau)) / z


def _pillar_times(curve: DiscountCurve) -> Float[Array, " n"]:
    """Year fractions from the reference date to each pillar."""
    return year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )


def _log_df_grid(
    curve: DiscountCurve,
) -> tuple[Float[Array, " n_nodes"], Float[Array, " n_nodes"]]:
    """Log-DF interpolation nodes, anchored at the reference date.

    Mirrors the Hull-White accessor: prepends the no-arbitrage anchor
    :math:`\\ln P^M(0, 0) = 0` so a curve whose first pillar sits strictly
    after the reference date does not silently drive the short-end
    instantaneous forward to zero.

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
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Log discount factor :math:`\\ln P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(model.initial_curve)
    return jnp.interp(t, times, log_dfs)


def _market_df(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(model.initial_curve)
    return jnp.exp(jnp.interp(t, times, log_dfs))


def _instantaneous_forward(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Instantaneous forward rate :math:`f^M(0, t) = -d\\ln P^M(0,t)/dt`."""
    return -jax.grad(lambda s: _log_df_at_time(model, s))(t)


# ── Public curve accessors (continuous time) ──────────────────────────

def g2pp_market_df(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at a year fraction.

    Args:
        model: G2++ model carrying the initial curve.
        t: Year fraction(s) from the curve reference date.

    Returns:
        Market discount factor at each ``t``.
    """
    return _market_df(model, t)


def g2pp_instantaneous_forward(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Market instantaneous forward rate :math:`f^M(0, t)`.

    Args:
        model: G2++ model carrying the initial curve.
        t: Year fraction from the curve reference date.

    Returns:
        Instantaneous forward rate at ``t``.
    """
    return _instantaneous_forward(model, t)


# ── Gaussian variance term ────────────────────────────────────────────

def _factor_variance_term(
    z: Float[Array, ""],
    vol: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Single-factor contribution to :math:`V(t, T)` with :math:`\\tau = T - t`.

    .. math::

        \\frac{\\text{vol}^2}{z^2}\\Bigl[\\tau + \\frac{2}{z}e^{-z\\tau}
            - \\frac{1}{2z}e^{-2z\\tau} - \\frac{3}{2z}\\Bigr]
    """
    return (vol**2 / z**2) * (
        tau
        + (2.0 / z) * jnp.exp(-z * tau)
        - (1.0 / (2.0 * z)) * jnp.exp(-2.0 * z * tau)
        - 3.0 / (2.0 * z)
    )


def _cross_variance_term(
    a: Float[Array, ""],
    b: Float[Array, ""],
    sigma: Float[Array, ""],
    eta: Float[Array, ""],
    rho: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Cross-factor contribution to :math:`V(t, T)` with :math:`\\tau = T - t`.

    .. math::

        2\\rho\\frac{\\sigma\\eta}{ab}\\Bigl[\\tau
            + \\frac{e^{-a\\tau} - 1}{a}
            + \\frac{e^{-b\\tau} - 1}{b}
            - \\frac{e^{-(a+b)\\tau} - 1}{a + b}\\Bigr]
    """
    return 2.0 * rho * sigma * eta / (a * b) * (
        tau
        + (jnp.exp(-a * tau) - 1.0) / a
        + (jnp.exp(-b * tau) - 1.0) / b
        - (jnp.exp(-(a + b) * tau) - 1.0) / (a + b)
    )


def g2pp_V(
    model: G2PPModel,
    t: Float[Array, "..."],
    T: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Gaussian variance term :math:`V(t, T)` of the integrated short rate.

    This is the variance of :math:`\\int_t^T (x(u) + y(u))\\,du` conditional
    on the factors at time :math:`t`.  It depends on :math:`t` and
    :math:`T` only through :math:`\\tau = T - t` and drives both the
    exact-fit ZCB price and the Gaussian swaption formula.

    .. math::

        V(t, T)
        = \\frac{\\sigma^2}{a^2}\\Bigl[(T-t) + \\tfrac{2}{a}e^{-a(T-t)}
            - \\tfrac{1}{2a}e^{-2a(T-t)} - \\tfrac{3}{2a}\\Bigr]
        + \\frac{\\eta^2}{b^2}\\Bigl[(T-t) + \\tfrac{2}{b}e^{-b(T-t)}
            - \\tfrac{1}{2b}e^{-2b(T-t)} - \\tfrac{3}{2b}\\Bigr]
        + 2\\rho\\frac{\\sigma\\eta}{ab}\\Bigl[(T-t)
            + \\tfrac{e^{-a(T-t)} - 1}{a}
            + \\tfrac{e^{-b(T-t)} - 1}{b}
            - \\tfrac{e^{-(a+b)(T-t)} - 1}{a+b}\\Bigr]

    Args:
        model: G2++ model carrying the factor parameters.
        t: Start time in year fractions.
        T: End time in year fractions.

    Returns:
        The variance term :math:`V(t, T)`.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.10).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation
    tau = T - t
    return (
        _factor_variance_term(a, sigma, tau)
        + _factor_variance_term(b, eta, tau)
        + _cross_variance_term(a, b, sigma, eta, rho, tau)
    )


# ── Deterministic exact-fit shift ─────────────────────────────────────

def g2pp_phi(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Deterministic shift :math:`\\varphi(t)` in :math:`r = x + y + \\varphi`.

    The shift carries the whole dependence on the initial curve, so the
    factors :math:`x, y` are zero-mean OU processes with time-independent
    coefficients — the parameterisation every numerical scheme wants.

    .. math::

        \\varphi(t) = f^M(0, t)
            + \\frac{\\sigma^2}{2a^2}\\bigl(1 - e^{-at}\\bigr)^2
            + \\frac{\\eta^2}{2b^2}\\bigl(1 - e^{-bt}\\bigr)^2
            + \\rho\\frac{\\sigma\\eta}{ab}\\bigl(1 - e^{-at}\\bigr)
              \\bigl(1 - e^{-bt}\\bigr)

    At :math:`t = 0` every convexity term vanishes and
    :math:`\\varphi(0) = f^M(0, 0) = r(0)`.

    Args:
        model: G2++ model carrying the initial curve and parameters.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The shift :math:`\\varphi(t)` at each ``t``.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.5).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation
    t_arr = jnp.asarray(t)
    forward = jnp.vectorize(lambda s: _instantaneous_forward(model, s))(t_arr)
    ea = 1.0 - jnp.exp(-a * t_arr)
    eb = 1.0 - jnp.exp(-b * t_arr)
    convexity = (
        (sigma**2 / (2.0 * a**2)) * ea**2
        + (eta**2 / (2.0 * b**2)) * eb**2
        + rho * sigma * eta / (a * b) * ea * eb
    )
    return forward + convexity


# ── Analytic zero-coupon bond price ───────────────────────────────────

def g2pp_bond_price(
    model: G2PPModel,
    x: Float[Array, ""],
    y: Float[Array, ""],
    t: Float[Array, ""],
    T: Float[Array, ""],
) -> Float[Array, ""]:
    """Zero-coupon bond price under G2++ given factors *x*, *y* at time *t*.

    .. math::

        P(t, T \\mid x, y)
            = \\frac{P^M(0, T)}{P^M(0, t)}
              \\exp\\!\\Bigl[\\tfrac{1}{2}\\bigl(V(t, T) - V(0, T) + V(0, t)\\bigr)
                            - B(a, t, T)\\,x - B(b, t, T)\\,y\\Bigr]

    (**Exact-fit property**: at :math:`t = 0` with :math:`x = y = 0` the
    bracket vanishes and this recovers the initial curve discount factor
    :math:`P^M(0, T)`.)

    Args:
        model: G2++ model (carries initial curve and parameters).
        x: Current value of the first factor (scalar).
        y: Current value of the second factor (scalar).
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.14).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y

    zero = jnp.zeros_like(t)
    v_tT = g2pp_V(model, t, T)
    v_0T = g2pp_V(model, zero, T)
    v_0t = g2pp_V(model, zero, t)

    B_a = g2pp_B(a, T - t)
    B_b = g2pp_B(b, T - t)

    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)

    ln_A = (
        ln_PM_T - ln_PM_t
        + 0.5 * (v_tT - v_0T + v_0t)
        - B_a * x - B_b * y
    )
    return jnp.exp(ln_A)


# ── Factor distribution ──────────────────────────────────────────────

def g2pp_factor_covariance(
    model: G2PPModel,
    dt: Float[Array, ""],
) -> Float[Array, "2 2"]:
    """Conditional covariance of :math:`(x, y)` over a step of length *dt*.

    Given :math:`(x(s), y(s))`, the increment to :math:`(x(s+dt), y(s+dt))`
    is Gaussian with mean :math:`(x(s)e^{-a\\,dt}, y(s)e^{-b\\,dt})` and the
    covariance returned here (time-homogeneous, independent of :math:`s`):

    .. math::

        \\text{Var}[x] = \\frac{\\sigma^2}{2a}\\bigl(1 - e^{-2a\\,dt}\\bigr),
        \\quad
        \\text{Var}[y] = \\frac{\\eta^2}{2b}\\bigl(1 - e^{-2b\\,dt}\\bigr)

    .. math::

        \\text{Cov}[x, y]
            = \\rho\\frac{\\sigma\\eta}{a + b}\\bigl(1 - e^{-(a+b)\\,dt}\\bigr)

    Used by the exact Monte-Carlo scheme (Cholesky of this 2x2 matrix).

    Args:
        model: G2++ model carrying the factor parameters.
        dt: Step length in year fractions.

    Returns:
        The ``2x2`` conditional covariance matrix of :math:`(x, y)`.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation

    var_x = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * dt))
    var_y = (eta**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * dt))
    cov_xy = rho * sigma * eta / (a + b) * (1.0 - jnp.exp(-(a + b) * dt))

    return jnp.array([[var_x, cov_xy], [cov_xy, var_y]])


def g2pp_short_rate_variance(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Variance of the short rate at time *t* (unconditional).

    .. math::

        \\text{Var}[r(t)]
            = \\frac{\\sigma^2}{2a}(1 - e^{-2at})
            + \\frac{\\eta^2}{2b}(1 - e^{-2bt})
            + 2\\rho\\frac{\\sigma\\eta}{a + b}(1 - e^{-(a+b)t})

    Args:
        model: G2++ model carrying the factor parameters.
        t: Year fraction from the curve reference date.

    Returns:
        The variance of :math:`r(t) = x(t) + y(t) + \\varphi(t)`.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation

    var_x = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t))
    var_y = (eta**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * t))
    cov_xy = rho * sigma * eta / (a + b) * (1.0 - jnp.exp(-(a + b) * t))
    return var_x + var_y + 2.0 * cov_xy
