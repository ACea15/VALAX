r"""Cheyette (quasi-Gaussian) Markovian HJM short-rate model.

The Cheyette model is the low-dimensional Markovian representation of the
Heath-Jarrow-Morton framework under a **separable** forward-rate volatility
structure.  Where Hull-White and G2++ are *Gaussian* (constant/deterministic
volatility, so rates are normally distributed and the smile is flat), Cheyette
lets the short-rate volatility depend on the **state** — a local-volatility
function :math:`\sigma_r(t, x)` — and so reproduces a volatility **skew** on
rates while staying a finite-dimensional Markov process that a lattice, PDE, or
Monte-Carlo scheme can price.

One factor
----------

The short rate is the initial forward plus a single mean-reverting state
variable, augmented by an auxiliary accumulator :math:`y` that makes the pair
Markovian:

.. math::

    r(t) = f^M(0, t) + x(t)

.. math::

    dx(t) = \bigl(y(t) - \kappa\,x(t)\bigr)\,dt
            + \sigma_r(t, x(t))\,dW(t), \quad x(0) = 0

    dy(t) = \bigl(\sigma_r(t, x(t))^2 - 2\kappa\,y(t)\bigr)\,dt, \quad y(0) = 0

Here :math:`\kappa` is the mean-reversion speed and :math:`\sigma_r(t, x)` is
the instantaneous short-rate volatility.  VALAX parameterises the local
volatility with a **displaced-diffusion / linear** form,

.. math::

    \sigma_r(t, x) = \sigma\,\bigl(1 + \beta\,x\bigr),

where :math:`\sigma` sets the overall level and the skew parameter
:math:`\beta` tilts the volatility with the state.  At :math:`\beta = 0` the
volatility is constant, the state is Gaussian, and the model **reduces exactly
to Hull-White** (see :func:`cheyette_y_gaussian`), which is the primary
validation oracle.

Exact fit and the affine bond
-----------------------------

Because :math:`r = f^M(0, t) + x` with :math:`x(0) = 0`, and the convexity is
generated dynamically by the :math:`y`-drift rather than a static shift, the
model fits the initial curve *by construction*.  The whole quasi-Gaussian class
shares one remarkable property: the zero-coupon bond is **affine in the state**
regardless of the local-volatility form (only the dynamics of :math:`(x, y)`
see :math:`\sigma_r`):

.. math::

    P(t, T \mid x, y)
        = \frac{P^M(0, T)}{P^M(0, t)}
          \exp\!\Bigl[-x\,G(t, T) - \tfrac{1}{2}\,y\,G(t, T)^2\Bigr],
    \qquad
    G(t, T) = \frac{1 - e^{-\kappa (T - t)}}{\kappa}.

At :math:`t = 0`, :math:`x = y = 0` the exponent vanishes and
:math:`P = P^M(0, T)`.

The model is an ``equinox.Module`` and hence a valid JAX pytree — ``jax.grad``,
``jax.jit``, and ``jax.vmap`` all work out of the box.

References:
    Cheyette, O. (1992), "Markov Representation of the Heath-Jarrow-Morton
        Model".
    Ritchken, P. & Sankarasubramanian, L. (1995), "Volatility Structures of
        Forward Rates and the Dynamics of the Term Structure".
    Andersen, L. & Piterbarg, V. (2010), *Interest Rate Modeling*, ch. 13
        (quasi-Gaussian models).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.models.short_rate import require_initial_curve


class CheyetteModel(eqx.Module):
    r"""One-factor Cheyette (quasi-Gaussian) short-rate model.

    Attributes:
        mean_reversion: Mean-reversion speed :math:`\kappa` (positive
            scalar).
        volatility: Local-volatility level :math:`\sigma` (positive
            scalar).
        skew: Local-volatility skew :math:`\beta` in
            :math:`\sigma_r(t, x) = \sigma (1 + \beta x)`.  ``0`` recovers
            the Gaussian (Hull-White) model.
        initial_curve: Initial discount curve :math:`P^M(0, t)`, fitted
            exactly by construction. ``None`` (the default) leaves the model
            unbound: :func:`valax.core.pricing.price` binds it from the
            market (see :mod:`valax.models.short_rate`).
        curve_id: Market id of the curve to bind, or ``None`` to use the
            instrument's discount curve. Static.
    """

    mean_reversion: Float[Array, ""]
    volatility: Float[Array, ""]
    skew: Float[Array, ""]
    initial_curve: DiscountCurve | None = None
    curve_id: str | None = eqx.field(static=True, default=None)


class CheyetteModel2F(eqx.Module):
    r"""Two-factor Cheyette (quasi-Gaussian) short-rate model.

    Adds a second mean-reverting factor so rates of different maturities can
    **decorrelate** — the low-dimensional, skew-capable analogue of G2++.  The
    short rate is the sum of two state variables on top of the initial forward,

    .. math::

        r(t) = f^M(0, t) + x_1(t) + x_2(t),

    with per-factor local volatilities
    :math:`\sigma_{r,i}(t, x_i) = \sigma_i (1 + \beta_i x_i)` and instantaneous
    correlation :math:`\rho` between the two driving Brownian motions.  When
    both skews vanish the factors are Gaussian and the model **reduces to
    G2++**.

    Attributes:
        mean_reversion_1: Mean-reversion speed :math:`\kappa_1` of the first
            factor (positive scalar).
        mean_reversion_2: Mean-reversion speed :math:`\kappa_2` of the second
            factor (positive scalar).
        volatility_1: Local-volatility level :math:`\sigma_1` of the first
            factor (positive scalar).
        volatility_2: Local-volatility level :math:`\sigma_2` of the second
            factor (positive scalar).
        skew_1: Local-volatility skew :math:`\beta_1` of the first factor.
        skew_2: Local-volatility skew :math:`\beta_2` of the second factor.
        correlation: Instantaneous correlation :math:`\rho \in (-1, 1)`
            between the two driving Brownian motions.
        initial_curve: Initial discount curve :math:`P^M(0, t)`, fitted
            exactly by construction. ``None`` (the default) leaves the model
            unbound: :func:`valax.core.pricing.price` binds it from the
            market (see :mod:`valax.models.short_rate`).
        curve_id: Market id of the curve to bind, or ``None`` to use the
            instrument's discount curve. Static.
    """

    mean_reversion_1: Float[Array, ""]
    mean_reversion_2: Float[Array, ""]
    volatility_1: Float[Array, ""]
    volatility_2: Float[Array, ""]
    skew_1: Float[Array, ""]
    skew_2: Float[Array, ""]
    correlation: Float[Array, ""]
    initial_curve: DiscountCurve | None = None
    curve_id: str | None = eqx.field(static=True, default=None)


# ── Helpers ───────────────────────────────────────────────────────────

def cheyette_G(
    kappa: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Mean-reversion decay factor :math:`G(\tau) = (1 - e^{-\kappa\tau})/\kappa`.

    This is the quasi-Gaussian analogue of :func:`~valax.models.hull_white.hw_B`
    — the sensitivity of the log bond price to the state factor.  For
    :math:`\kappa \to 0` it reduces to :math:`\tau`.

    Args:
        kappa: Mean-reversion speed of the factor.
        tau: Time to maturity :math:`T - t` (year fractions).

    Returns:
        The decay factor :math:`G(\tau)`.
    """
    return (1.0 - jnp.exp(-kappa * tau)) / kappa


def _pillar_times(curve: DiscountCurve) -> Float[Array, " n"]:
    """Year fractions from the reference date to each pillar."""
    return year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )


def _log_df_grid(
    curve: DiscountCurve,
) -> tuple[Float[Array, " n_nodes"], Float[Array, " n_nodes"]]:
    r"""Log-DF interpolation nodes, anchored at the reference date.

    Mirrors the Hull-White / G2++ accessor: prepends the no-arbitrage anchor
    :math:`\ln P^M(0, 0) = 0` so a curve whose first pillar sits strictly after
    the reference date does not silently drive the short-end instantaneous
    forward to zero.

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
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Log discount factor :math:`\ln P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(require_initial_curve(model))
    return jnp.interp(t, times, log_dfs)


def _market_df(
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(require_initial_curve(model))
    return jnp.exp(jnp.interp(t, times, log_dfs))


def _instantaneous_forward(
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Instantaneous forward rate :math:`f^M(0, t) = -d\ln P^M(0,t)/dt`."""
    return -jax.grad(lambda s: _log_df_at_time(model, s))(t)


# ── Public curve accessors (continuous time) ──────────────────────────

def cheyette_market_df(
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Market discount factor :math:`P^M(0, t)` at a year fraction.

    ``DiscountCurve.__call__`` takes integer ordinal dates; this accessor takes
    a continuous year fraction, the natural coordinate for the model analytics.

    Args:
        model: Cheyette model (1F or 2F) carrying the initial curve.
        t: Year fraction(s) from the curve reference date.

    Returns:
        Market discount factor at each ``t``.
    """
    return _market_df(model, t)


def cheyette_instantaneous_forward(
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Market instantaneous forward rate :math:`f^M(0, t)`.

    Args:
        model: Cheyette model (1F or 2F) carrying the initial curve.
        t: Year fraction from the curve reference date.

    Returns:
        Instantaneous forward rate at ``t``.
    """
    return _instantaneous_forward(model, t)


# ── Local volatility ──────────────────────────────────────────────────

def cheyette_local_vol(
    model: CheyetteModel,
    x: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""One-factor local volatility :math:`\sigma_r(t, x) = \sigma\,(1 + \beta x)`.

    The state-dependence carried by :math:`\beta` is what produces the rate
    skew; :math:`\beta = 0` gives the constant Gaussian volatility.  The value
    is floored at zero for numerical robustness at large negative states.

    Args:
        model: One-factor Cheyette model.
        x: State factor value(s).

    Returns:
        Instantaneous short-rate volatility at each state.
    """
    return jnp.maximum(model.volatility * (1.0 + model.skew * x), 0.0)


def cheyette_local_vol_2f(
    model: CheyetteModel2F,
    x1: Float[Array, "..."],
    x2: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""Two-factor per-factor local volatilities.

    .. math::

        \sigma_{r,1}(t, x_1) = \sigma_1 (1 + \beta_1 x_1), \qquad
        \sigma_{r,2}(t, x_2) = \sigma_2 (1 + \beta_2 x_2),

    each floored at zero for robustness.

    Args:
        model: Two-factor Cheyette model.
        x1: First-factor value(s).
        x2: Second-factor value(s).

    Returns:
        Pair ``(sigma_r1, sigma_r2)`` of per-factor volatilities.
    """
    sig1 = jnp.maximum(model.volatility_1 * (1.0 + model.skew_1 * x1), 0.0)
    sig2 = jnp.maximum(model.volatility_2 * (1.0 + model.skew_2 * x2), 0.0)
    return sig1, sig2


# ── Gaussian-limit accumulators ───────────────────────────────────────

def cheyette_y_gaussian(
    model: CheyetteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Deterministic accumulator :math:`y(t)` in the Gaussian limit.

    When the local volatility is constant (:math:`\beta = 0`) the ODE
    :math:`\dot y = \sigma^2 - 2\kappa y`, :math:`y(0) = 0` solves in closed
    form,

    .. math::

        y(t) = \frac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa t}\bigr),

    which also equals :math:`\mathrm{Var}[x(t)]` in that limit.  This is the
    quantity that reproduces the Hull-White convexity, and it is the reference
    used by the reduction-limit tests.

    Args:
        model: One-factor Cheyette model.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The Gaussian-limit accumulator :math:`y(t)`.
    """
    kappa = model.mean_reversion
    sigma = model.volatility
    return (sigma**2 / (2.0 * kappa)) * (1.0 - jnp.exp(-2.0 * kappa * t))


def cheyette_y_gaussian_2f(
    model: CheyetteModel2F,
    t: Float[Array, ""],
) -> Float[Array, "2 2"]:
    r"""Deterministic accumulator matrix :math:`y_{ij}(t)` in the Gaussian limit.

    With constant local volatilities the symmetric auxiliary state solves

    .. math::

        y_{ij}(t) = \rho_{ij}\,\frac{\sigma_i \sigma_j}{\kappa_i + \kappa_j}
                    \bigl(1 - e^{-(\kappa_i + \kappa_j) t}\bigr),

    with :math:`\rho_{11} = \rho_{22} = 1` and :math:`\rho_{12} = \rho`.  The
    diagonal entries match the per-factor Gaussian variances and the
    off-diagonal matches the G2++ cross-covariance, which is what drives the
    G2++ reduction limit.

    Args:
        model: Two-factor Cheyette model.
        t: Year fraction from the curve reference date.

    Returns:
        The ``2x2`` symmetric accumulator matrix :math:`y_{ij}(t)`.
    """
    k1 = model.mean_reversion_1
    k2 = model.mean_reversion_2
    s1 = model.volatility_1
    s2 = model.volatility_2
    rho = model.correlation

    y11 = (s1**2 / (2.0 * k1)) * (1.0 - jnp.exp(-2.0 * k1 * t))
    y22 = (s2**2 / (2.0 * k2)) * (1.0 - jnp.exp(-2.0 * k2 * t))
    y12 = rho * s1 * s2 / (k1 + k2) * (1.0 - jnp.exp(-(k1 + k2) * t))
    return jnp.array([[y11, y12], [y12, y22]])


def cheyette_gaussian_mean(
    model: CheyetteModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Gaussian-limit mean of the state factor, :math:`\mathbb{E}[x(t)]`.

    Unlike the *centred* factors of Hull-White / G2++, the Cheyette state
    carries a non-zero drift :math:`y(t)` and so has a non-zero mean.  In the
    constant-volatility (:math:`\beta = 0`) limit the mean solves
    :math:`\dot m = y(t) - \kappa m`, :math:`m(0) = 0` in closed form,

    .. math::

        m(t) = \frac{\sigma^2}{2\kappa^2}\bigl(1 - e^{-\kappa t}\bigr)^2,

    which is exactly the Hull-White convexity shift :math:`\alpha(t) - f^M(0,t)`.
    A centred Hull-White state :math:`x_{\mathrm{HW}}` therefore corresponds to
    the Cheyette factor :math:`x = x_{\mathrm{HW}} + m(t)`; this is the mapping
    the reduction-limit tests use.

    Args:
        model: One-factor Cheyette model.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The Gaussian-limit factor mean :math:`m(t)`.
    """
    kappa = model.mean_reversion
    sigma = model.volatility
    return (sigma**2 / (2.0 * kappa**2)) * (1.0 - jnp.exp(-kappa * t)) ** 2


def cheyette_gaussian_mean_2f(
    model: CheyetteModel2F,
    t: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""Gaussian-limit means :math:`(\mathbb{E}[x_1(t)], \mathbb{E}[x_2(t)])`.

    In the constant-volatility limit each factor mean solves
    :math:`\dot m_i = \sum_j y_{ij}(t) - \kappa_i m_i`, giving

    .. math::

        m_i(t) = \frac{\sigma_i^2}{2\kappa_i^2}\bigl(1 - e^{-\kappa_i t}\bigr)^2
            + \rho\,\frac{\sigma_1\sigma_2}{\kappa_1 + \kappa_2}
              \Bigl[\frac{1 - e^{-\kappa_i t}}{\kappa_i}
                    - e^{-\kappa_i t}\frac{1 - e^{-\kappa_{j} t}}{\kappa_{j}}\Bigr],

    with :math:`j` the other factor.  Their sum equals the G2++ shift convexity
    :math:`\varphi(t) - f^M(0, t)`, so a centred G2++ state :math:`(x, y)` maps
    to the Cheyette factors :math:`(x + m_1(t), y + m_2(t))` — the mapping used
    by the G2++ reduction-limit tests.

    Args:
        model: Two-factor Cheyette model.
        t: Year fraction(s) from the curve reference date.

    Returns:
        Pair ``(m1, m2)`` of Gaussian-limit factor means.
    """
    k1 = model.mean_reversion_1
    k2 = model.mean_reversion_2
    s1 = model.volatility_1
    s2 = model.volatility_2
    rho = model.correlation

    e1 = 1.0 - jnp.exp(-k1 * t)
    e2 = 1.0 - jnp.exp(-k2 * t)
    cross = rho * s1 * s2 / (k1 + k2)

    m1 = (s1**2 / (2.0 * k1**2)) * e1**2 + cross * (
        e1 / k1 - jnp.exp(-k1 * t) * e2 / k2
    )
    m2 = (s2**2 / (2.0 * k2**2)) * e2**2 + cross * (
        e2 / k2 - jnp.exp(-k2 * t) * e1 / k1
    )
    return m1, m2


# ── Short rate ────────────────────────────────────────────────────────

def cheyette_short_rate(
    model: CheyetteModel,
    x: Float[Array, "..."],
    t: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Short rate :math:`r(t) = f^M(0, t) + x(t)` from the one-factor state.

    Args:
        model: One-factor Cheyette model.
        x: State factor value(s) at time ``t``.
        t: Year fraction from the curve reference date.

    Returns:
        Short rate at each state.
    """
    return _instantaneous_forward(model, t) + x


def cheyette_short_rate_2f(
    model: CheyetteModel2F,
    x1: Float[Array, "..."],
    x2: Float[Array, "..."],
    t: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Short rate :math:`r(t) = f^M(0, t) + x_1(t) + x_2(t)`.

    Args:
        model: Two-factor Cheyette model.
        x1: First-factor value(s) at time ``t``.
        x2: Second-factor value(s) at time ``t``.
        t: Year fraction from the curve reference date.

    Returns:
        Short rate at each state.
    """
    return _instantaneous_forward(model, t) + x1 + x2


# ── Analytic zero-coupon bond price ───────────────────────────────────

def cheyette_bond_price(
    model: CheyetteModel,
    x: Float[Array, ""],
    y: Float[Array, ""],
    t: Float[Array, ""],
    T: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Zero-coupon bond price under one-factor Cheyette given state *(x, y)*.

    The whole quasi-Gaussian class shares the affine reconstruction — it holds
    for **any** local-volatility form, since :math:`\sigma_r` enters only the
    dynamics of :math:`(x, y)`, not the bond formula:

    .. math::

        P(t, T \mid x, y)
            = \frac{P^M(0, T)}{P^M(0, t)}
              \exp\!\Bigl[-x\,G(t, T) - \tfrac{1}{2}\,y\,G(t, T)^2\Bigr],
        \quad
        G(t, T) = \frac{1 - e^{-\kappa (T - t)}}{\kappa}.

    (**Exact-fit property**: at :math:`t = 0` with :math:`x = y = 0` the
    exponent vanishes and this recovers :math:`P^M(0, T)`.)

    Args:
        model: One-factor Cheyette model (carries curve and parameters).
        x: Current state factor (scalar).
        y: Current auxiliary accumulator (scalar).
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.
    """
    G = cheyette_G(model.mean_reversion, T - t)
    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)
    ln_A = ln_PM_T - ln_PM_t - x * G - 0.5 * y * G**2
    return jnp.exp(ln_A)


def cheyette_bond_price_2f(
    model: CheyetteModel2F,
    x1: Float[Array, ""],
    x2: Float[Array, ""],
    y11: Float[Array, ""],
    y22: Float[Array, ""],
    y12: Float[Array, ""],
    t: Float[Array, ""],
    T: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Zero-coupon bond price under two-factor Cheyette given the state.

    .. math::

        P(t, T \mid x, y)
            = \frac{P^M(0, T)}{P^M(0, t)}
              \exp\!\Bigl[-\sum_i x_i G_i
                          - \tfrac{1}{2}\sum_{i,j} y_{ij} G_i G_j\Bigr],
        \quad
        G_i = \frac{1 - e^{-\kappa_i (T - t)}}{\kappa_i},

    where the quadratic form expands to
    :math:`y_{11} G_1^2 + 2 y_{12} G_1 G_2 + y_{22} G_2^2`.

    (**Exact-fit property**: at :math:`t = 0` with zero state the exponent
    vanishes and this recovers :math:`P^M(0, T)`.)

    Args:
        model: Two-factor Cheyette model (carries curve and parameters).
        x1: First state factor (scalar).
        x2: Second state factor (scalar).
        y11: Auxiliary accumulator :math:`y_{11}` (scalar).
        y22: Auxiliary accumulator :math:`y_{22}` (scalar).
        y12: Auxiliary cross accumulator :math:`y_{12}` (scalar).
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.
    """
    G1 = cheyette_G(model.mean_reversion_1, T - t)
    G2 = cheyette_G(model.mean_reversion_2, T - t)
    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)
    quad = y11 * G1**2 + 2.0 * y12 * G1 * G2 + y22 * G2**2
    ln_A = ln_PM_T - ln_PM_t - x1 * G1 - x2 * G2 - 0.5 * quad
    return jnp.exp(ln_A)
