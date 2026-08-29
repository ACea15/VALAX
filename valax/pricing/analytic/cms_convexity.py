"""CMS convexity adjustment via Hagan's "convexity conundrums" model.

A CMS coupon pays a swap rate :math:`S` at a date under whose forward
(discount) measure :math:`S` is *not* a martingale — the swap rate is a
martingale only under its own annuity measure.  The expectation therefore
differs from the forward par swap rate by a **convexity adjustment** that
depends on the swap-rate volatility smile.

Two model routes are provided, both built on Hagan (2003):

- :func:`_hagan_analytic` — the closed-form quadratic-payoff approximation.
  Expands the convexity payoff to second order and uses a single implied
  variance read at the forward rate.  This is the analogue of QuantLib's
  ``AnalyticHaganPricer``.
- :func:`_hagan_replication` — static replication of the convexity payoff as a
  strike-continuum of vanilla swaptions, integrating the whole smile.  This is
  the analogue of QuantLib's ``NumericHaganPricer`` and is the more accurate
  ("replication") route.

Both use the *street-standard* model for the numeraire ratio: the annuity is
approximated as a function of the swap rate alone via a flat-yield bond,

.. math::

    A(S) = \\frac{1 - (1 + S/q)^{-nq}}{S}, \\qquad g(S) = \\frac{S}{A(S)},

with ``n`` the underlying-swap tenor (years) and ``q`` the fixed-leg frequency
(payments per year).  The convexity payoff is :math:`h(S) = S\\,A(S_0)/A(S)`,
whose curvature ``h''`` drives the adjustment.  ``g''`` is obtained by
automatic differentiation rather than by hand, so the standard model can be
swapped for another closed form without re-deriving derivatives.

The two routes agree exactly in the limit where ``g`` is quadratic (constant
``g''``): there the replication integral collapses to
:math:`\\tfrac12 A(S_0) g''(S_0)\\,\\mathrm{Var}^A[S]`, which is precisely the
analytic formula.  This gives a clean internal cross-check in the low-vol /
flat-smile regime.

References:
    Hagan (2003), "Convexity Conundrums: Pricing CMS Swaps, Caps, and Floors",
        Wilmott Magazine.
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 13.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


_EPS = 1e-10

# A *vol source* is a callable ``(strike, expiry, tenor) -> implied vol`` that
# also carries an ``is_normal`` bool attribute (e.g. ``SwaptionCube`` /
# ``ConstantVol``).  Public entry points also accept a bare scalar (flat
# lognormal vol), captured by the ``| Float[Array, ""] | float`` union below.
VolSource = Callable[..., Float[Array, ""]]


def _resolve_vol_source(
    vol: VolSource | Float[Array, ""] | float,
) -> tuple[VolSource, bool]:
    """Normalize a scalar-or-callable into a ``(source, is_normal)`` pair.

    A *vol source* is any callable exposing an ``is_normal`` attribute — e.g.
    :class:`~valax.surfaces.swaption_cube.SwaptionCube` or
    :class:`~valax.surfaces.constant.ConstantVol`.  A bare scalar is wrapped in
    a coordinate-ignoring lognormal closure.  This mirrors
    :func:`valax.pricing.analytic.rates_smile._as_vol_source` but avoids
    importing the surfaces package (keeping the pricing→surfaces dependency
    one-directional and the import order clean).

    Args:
        vol: A vol source or a bare scalar volatility.

    Returns:
        Tuple ``(source, is_normal)`` where ``source(strike, expiry, tenor)``
        returns an implied vol and ``is_normal`` is the quoting convention.
    """
    if callable(vol) and hasattr(vol, "is_normal"):
        return vol, vol.is_normal
    v = jnp.asarray(vol)
    return (lambda *coords: v), False


# ── Standard-model annuity / numeraire-ratio functions ────────────────

def _standard_annuity(
    swap_rate: Float[Array, ""],
    tenor: int,
    freq: int,
) -> Float[Array, ""]:
    """Flat-yield ("street-standard") annuity per unit notional.

    Args:
        swap_rate: Par swap rate ``S`` (positive).
        tenor: Underlying-swap tenor in years.
        freq: Fixed-leg payments per year.

    Returns:
        Annuity :math:`A(S) = (1 - (1 + S/q)^{-nq}) / S`.
    """
    s = jnp.maximum(swap_rate, _EPS)
    m = tenor * freq
    return (1.0 - (1.0 + s / freq) ** (-m)) / s


def _standard_g(
    swap_rate: Float[Array, ""],
    tenor: int,
    freq: int,
) -> Float[Array, ""]:
    """Standard-model numeraire ratio ``g(S) = S / A(S)``."""
    return swap_rate / _standard_annuity(swap_rate, tenor, freq)


# ── Black / Bachelier option values (per unit annuity, undiscounted) ───

def _black_call_put(
    F: Float[Array, ""],
    K: Float[Array, " n"],
    sigma: Float[Array, " n"],
    T: Float[Array, ""],
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Undiscounted Black-76 call and put on a forward ``F`` (per annuity)."""
    sqrt_T = jnp.sqrt(jnp.maximum(T, _EPS))
    v = jnp.maximum(sigma * sqrt_T, _EPS)
    d1 = (jnp.log(jnp.maximum(F, _EPS) / jnp.maximum(K, _EPS)) + 0.5 * v * v) / v
    d2 = d1 - v
    Phi = jax.scipy.stats.norm.cdf
    call = F * Phi(d1) - K * Phi(d2)
    put = K * Phi(-d2) - F * Phi(-d1)
    return call, put


def _bachelier_call_put(
    F: Float[Array, ""],
    K: Float[Array, " n"],
    sigma: Float[Array, " n"],
    T: Float[Array, ""],
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Undiscounted Bachelier call and put on a forward ``F`` (per annuity)."""
    v = jnp.maximum(sigma * jnp.sqrt(jnp.maximum(T, _EPS)), _EPS)
    d = (F - K) / v
    Phi = jax.scipy.stats.norm.cdf
    phi = jax.scipy.stats.norm.pdf
    call = (F - K) * Phi(d) + v * phi(d)
    put = call - (F - K)
    return call, put


# ── The two convexity routes ──────────────────────────────────────────

def _hagan_analytic(
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    tenor: int,
    freq: int,
    sigma_atm: Float[Array, ""],
    is_normal: bool,
) -> Float[Array, ""]:
    """Closed-form Hagan convexity adjustment (quadratic approximation).

    Expands the convexity payoff :math:`h(S) = S\\,A(S_0)/A(S)` to second order
    about the forward ``S_0`` and applies

    .. math::

        \\text{CA} = \\tfrac12\\, A(S_0)\\, g''(S_0)\\, \\mathrm{Var}^A[S],

    with ``Var`` the annuity-measure variance of the swap rate — lognormal
    (:math:`S_0^2(e^{\\sigma^2 T} - 1)`) or normal (:math:`\\sigma^2 T`) per the
    quoting convention.  ``g''`` is evaluated by automatic differentiation.

    Args:
        forward: Forward par swap rate ``S_0``.
        expiry: Time to fixing (year fraction).
        tenor: Underlying-swap tenor (years).
        freq: Fixed-leg frequency (payments per year).
        sigma_atm: Implied vol at the forward (annuity-measure smile at ATM).
        is_normal: Normal (Bachelier) variance if True, else lognormal.

    Returns:
        Convexity adjustment ``E^{pay}[S] - S_0`` (a rate).
    """
    g2 = jax.grad(jax.grad(lambda s: _standard_g(s, tenor, freq)))(forward)
    a0 = _standard_annuity(forward, tenor, freq)
    if is_normal:
        var = sigma_atm**2 * expiry
    else:
        var = forward**2 * jnp.expm1(sigma_atm**2 * expiry)
    return 0.5 * a0 * g2 * var


def _hagan_replication(
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    tenor: int,
    freq: int,
    vol_at: Callable[[Float[Array, ""]], Float[Array, ""]],
    is_normal: bool,
    n_strikes: int,
    n_std: float,
) -> Float[Array, ""]:
    """Static-replication Hagan convexity adjustment over the full smile.

    Replicates the convexity payoff with a strike-continuum of vanilla
    swaptions (Carr–Madan):

    .. math::

        \\text{CA} = A(S_0)\\left[
            \\int_0^{S_0} g''(K)\\, P(K)\\, dK
            + \\int_{S_0}^{\\infty} g''(K)\\, C(K)\\, dK \\right],

    where ``C(K)`` / ``P(K)`` are undiscounted Black/Bachelier call/put values
    (per unit annuity) evaluated at the smile vol ``σ(K)``.  The integrals are
    truncated at ``n_std`` implied standard deviations and discretised on
    ``n_strikes`` points via the trapezoidal rule (fixed sizes → JIT-safe).

    Args:
        forward: Forward par swap rate ``S_0``.
        expiry: Time to fixing (year fraction).
        tenor: Underlying-swap tenor (years).
        freq: Fixed-leg frequency (payments per year).
        vol_at: Callable ``K -> σ(K)`` giving the smile vol at strike ``K``.
        is_normal: Bachelier options/vols if True, else Black-76.
        n_strikes: Quadrature points per wing (static).
        n_std: Truncation width in implied standard deviations.

    Returns:
        Convexity adjustment ``E^{pay}[S] - S_0`` (a rate).
    """
    a0 = _standard_annuity(forward, tenor, freq)
    sigma_atm = vol_at(forward)
    sqrt_T = jnp.sqrt(jnp.maximum(expiry, _EPS))

    if is_normal:
        width = n_std * sigma_atm * sqrt_T
        k_min = jnp.maximum(forward - width, _EPS)
        k_max = forward + width
        opt = _bachelier_call_put
    else:
        width = n_std * jnp.maximum(sigma_atm, _EPS) * sqrt_T
        k_min = jnp.maximum(forward * jnp.exp(-width), _EPS)
        k_max = forward * jnp.exp(width)
        opt = _black_call_put

    g2_fn = jax.grad(jax.grad(lambda s: _standard_g(s, tenor, freq)))

    def wing(k_lo, k_hi, use_put):
        strikes = jnp.linspace(k_lo, k_hi, n_strikes)
        sig = jax.vmap(vol_at)(strikes)
        g2 = jax.vmap(g2_fn)(strikes)
        call, put = opt(forward, strikes, sig, expiry)
        payoff = jnp.where(use_put, put, call)
        integrand = g2 * payoff
        return jnp.trapezoid(integrand, strikes)

    put_wing = wing(k_min, forward, True)
    call_wing = wing(forward, k_max, False)
    return a0 * (put_wing + call_wing)


# ── Public API ────────────────────────────────────────────────────────

def cms_convexity_adjustment(
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    tenor: int,
    vol: VolSource | Float[Array, ""] | float,
    method: str = "replication",
    freq: int = 1,
    n_strikes: int = 129,
    n_std: float = 6.0,
) -> Float[Array, ""]:
    """CMS convexity adjustment ``E^{pay}[S] - S_0`` under Hagan's model.

    Dispatches to the closed-form (``"analytic"``) or static-replication
    (``"replication"``) route.  The vol argument is a *vol source* — a callable
    exposing an ``is_normal`` flag (e.g. a
    :class:`~valax.surfaces.swaption_cube.SwaptionCube`) queried at
    ``(strike, expiry, tenor)`` — or a bare scalar (flat lognormal vol).

    Args:
        forward: Forward par swap rate ``S_0``.
        expiry: Time to fixing (year fraction).
        tenor: Underlying-swap tenor in years.
        vol: Vol source or scalar volatility.
        method: ``"replication"`` (smile-integrating, accurate) or
            ``"analytic"`` (single-vol quadratic approximation).
        freq: Fixed-leg frequency of the underlying swap (payments per year).
            Defaults to 1 (annual), matching the synthetic annual swap used by
            the CMS pricers.
        n_strikes: Replication quadrature points per wing (ignored by the
            analytic route).
        n_std: Replication truncation width in implied standard deviations.

    Returns:
        Convexity adjustment (a rate) to add to ``forward``.

    Raises:
        ValueError: If ``method`` is not ``"analytic"`` or ``"replication"``.
    """
    src, is_normal = _resolve_vol_source(vol)
    tenor_f = jnp.asarray(float(tenor))

    def vol_at(strike):
        return src(strike, expiry, tenor_f)

    if method == "analytic":
        return _hagan_analytic(
            forward, expiry, tenor, freq, vol_at(forward), is_normal
        )
    if method == "replication":
        return _hagan_replication(
            forward, expiry, tenor, freq, vol_at, is_normal, n_strikes, n_std
        )
    raise ValueError(
        f"Unknown convexity method {method!r}; "
        "expected 'analytic' or 'replication'."
    )


def cms_convexity_adjusted_rates(
    forwards: Float[Array, " n"],
    expiries: Float[Array, " n"],
    tenor: int,
    vol: VolSource | Float[Array, ""] | float,
    method: str = "replication",
    freq: int = 1,
    n_strikes: int = 129,
    n_std: float = 6.0,
) -> Float[Array, " n"]:
    """Vectorised convexity-adjusted CMS rates ``S_0 + CA`` per fixing.

    Applies :func:`cms_convexity_adjustment` across a schedule of forward swap
    rates and fixing times sharing one ``tenor`` and vol source.

    Args:
        forwards: Forward par swap rate at each fixing.
        expiries: Time to each fixing (year fractions).
        tenor: Underlying-swap tenor in years.
        vol: Vol source or scalar volatility.
        method: ``"replication"`` or ``"analytic"``.
        freq: Fixed-leg frequency of the underlying swap.
        n_strikes: Replication quadrature points per wing.
        n_std: Replication truncation width in implied standard deviations.

    Returns:
        Convexity-adjusted CMS rates, one per fixing.
    """
    src, is_normal = _resolve_vol_source(vol)
    tenor_f = jnp.asarray(float(tenor))
    g2_fn = jax.grad(jax.grad(lambda s: _standard_g(s, tenor, freq)))

    def one(forward, expiry):
        def vol_at(strike):
            return src(strike, expiry, tenor_f)

        if method == "analytic":
            a0 = _standard_annuity(forward, tenor, freq)
            sig = vol_at(forward)
            if is_normal:
                var = sig**2 * expiry
            else:
                var = forward**2 * jnp.expm1(sig**2 * expiry)
            return forward + 0.5 * a0 * g2_fn(forward) * var
        if method == "replication":
            ca = _hagan_replication(
                forward, expiry, tenor, freq, vol_at, is_normal,
                n_strikes, n_std,
            )
            return forward + ca
        raise ValueError(
            f"Unknown convexity method {method!r}; "
            "expected 'analytic' or 'replication'."
        )

    return jax.vmap(one)(forwards, expiries)
