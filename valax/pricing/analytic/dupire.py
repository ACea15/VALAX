"""Dupire local volatility extraction from an implied vol surface.

The Dupire (1994) formula gives the unique deterministic ``sigma(S, t)``
function that, under

    dS_t = (r - q) S_t dt + sigma(S_t, t) S_t dW_t,

reproduces a given vanilla call surface by construction. The cleanest
numerical form for autodiff-friendly libraries is Gatheral's
*implied-vol-space* representation in total variance
``w(k, T) = sigma_IV(k, T)^2 * T`` with log-moneyness
``k = log(K / F(T))``:

.. math::

    \\sigma_{loc}^2(k, T) =
        \\frac{\\partial_T w}
             {1 - \\frac{k}{w}\\partial_k w
              + \\frac{1}{4}\\left(-\\frac{1}{4} - \\frac{1}{w} + \\frac{k^2}{w^2}\\right)(\\partial_k w)^2
              + \\frac{1}{2}\\partial_{kk} w}.

This is the equation we implement.

Why IV-space rather than price-space Dupire?
    - Price-space Dupire needs ``∂²C/∂K²`` which on any non-parametric
      discrete grid is numerically extremely noisy.
    - With a calibrated SVI surface, all of ``∂_k w``, ``∂_{kk} w``,
      ``∂_T w`` are clean ``jax.grad`` calls on the closed-form
      ``svi_total_variance``.
    - Differentiability w.r.t. surface parameters (vega bucketing
      sensitivities of local vol) follows for free.

Numerical safeguards:
    - ``w`` is clamped below by ``1e-10`` (matches the surface's own
      ``__call__`` clamp), so divisions by ``w`` cannot blow up at the
      ATM forward where ``k = 0``.
    - The numerator ``∂_T w`` should be ≥ 0 (calendar-spread arbitrage).
      We clamp at zero before ``sqrt`` so a tiny floating-point negative
      from interpolation noise does not poison the gradient.
    - The denominator is **not** clamped: a non-positive denominator
      indicates butterfly arbitrage in the input surface and the
      resulting NaN is the correct diagnostic — arbitrage in,
      no-arbitrage out is impossible.

x64 enforcement:
    ``jax_enable_x64`` MUST be on. Dupire involves second derivatives of
    total variance, which f32 can mangle to ~3 significant digits in
    near-ATM regions. We raise ``RuntimeError`` at call time rather than
    silently degrade.

References:
    - Dupire, B. (1994). Pricing with a Smile. Risk.
    - Gatheral, J. (2006). The Volatility Surface, ch. 1.
    - Gatheral & Jacquier (2014). Arbitrage-free SVI volatility surfaces.
"""

from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


class _TotalVarianceSurface(Protocol):
    """Duck-typed surface protocol expected by Dupire.

    Any surface object exposing ``total_variance(log_moneyness, expiry)``
    returning a scalar ``Float[Array, ""]`` works. ``SVIVolSurface``,
    ``SABRVolSurface``, and ``GridVolSurface`` all comply.
    """

    def total_variance(
        self,
        log_moneyness: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]: ...


def _check_x64() -> None:
    """Raise if x64 is not enabled.

    Dupire's accuracy collapses under f32 in any region where ``w`` is
    small (i.e. short-dated ATM). Cheap and loud is the right trade-off.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "dupire_local_vol requires jax_enable_x64=True. "
            "Add `jax.config.update('jax_enable_x64', True)` at the top "
            "of your script/notebook before importing valax."
        )


def dupire_local_vol(
    surface: _TotalVarianceSurface,
    log_moneyness: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Local volatility from an implied vol surface via Gatheral's formula.

    Args:
        surface: Object exposing ``total_variance(k, T) -> Float[""]``.
            Typically a ``SVIVolSurface``; ``SABRVolSurface`` and
            ``GridVolSurface`` also comply.
        log_moneyness: ``k = log(K / F(T))`` for the query.
        expiry: Year fraction, must be > 0.

    Returns:
        Scalar local volatility ``sigma_loc(k, T)``.

    Notes:
        - Scalar-in / scalar-out. Use ``jax.vmap`` to batch.
        - ``jax.grad`` w.r.t. surface parameters works directly.
        - At a butterfly-arbitrage violation, the denominator becomes
          non-positive and ``sqrt`` produces NaN — this is intentional
          and signals a malformed input surface.
    """
    _check_x64()
    numerator, g, _ = _dupire_terms(surface, log_moneyness, expiry)
    # sigma_loc^2 = numerator / denominator. NaN if denominator <= 0
    # (butterfly arbitrage diagnostic).
    sigma2 = numerator / g
    return jnp.sqrt(sigma2)


def _dupire_terms(
    surface: _TotalVarianceSurface,
    log_moneyness: Float[Array, ""],
    expiry: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Numerator, denominator and total variance of Gatheral's formula.

    Args:
        surface: Object exposing ``total_variance(k, T)``.
        log_moneyness: ``k = log(K / F(T))``.
        expiry: Year fraction, > 0.

    Returns:
        ``(∂_T w clamped at 0, denominator g, w)`` at ``(k, T)``; the local
        variance is their ratio ``num / g``.
    """

    def w_fn(k: Float[Array, ""], T: Float[Array, ""]) -> Float[Array, ""]:
        return surface.total_variance(k, T)

    # Evaluate w, ∂_k w, ∂_{kk} w, ∂_T w via autodiff on the surface's
    # own total_variance method.
    w = w_fn(log_moneyness, expiry)
    dw_dk = jax.grad(w_fn, argnums=0)(log_moneyness, expiry)
    d2w_dk2 = jax.grad(jax.grad(w_fn, argnums=0), argnums=0)(
        log_moneyness, expiry
    )
    dw_dT = jax.grad(w_fn, argnums=1)(log_moneyness, expiry)

    # Numerator: ∂_T w (clamp at zero for calendar-arb fp noise).
    numerator = jnp.maximum(dw_dT, 0.0)

    # w must be strictly positive for the 1/w terms; the surface
    # protocol's clamp at 1e-10 already guarantees this. Belt-and-braces:
    w_safe = jnp.maximum(w, 1e-10)

    # Denominator: Gatheral eq. (1.10).
    g = (
        1.0
        - (log_moneyness / w_safe) * dw_dk
        + 0.25
        * (
            -0.25
            - 1.0 / w_safe
            + (log_moneyness * log_moneyness) / (w_safe * w_safe)
        )
        * (dw_dk * dw_dk)
        + 0.5 * d2w_dk2
    )
    return numerator, g, w


class LocalVolGuard(eqx.Module):
    """Domain and range guard for Dupire local volatility in path simulation.

    A Monte Carlo path can wander where the implied-vol surface is no longer
    a usable model — e.g. ``k = log(K/F) ≈ -14`` on a Hagan SABR smile,
    whose expansion is only valid near the money and returns NaN there — or
    where the surface has a (numerical) butterfly arbitrage, so Gatheral's
    denominator is not positive. Raw :func:`dupire_local_vol` then returns
    NaN and one path poisons the price. The guard makes the evaluation total:

    1. **Domain.** The query is clamped to
       :math:`|k| \\le n_{\\mathrm{sd}}\\sqrt{w(0, T)}` — ``n_sd`` ATM
       total-implied standard deviations. Beyond it the local vol is
       extrapolated **flat in k**, the usual practitioner choice: no quoted
       strike constrains the surface there, so its wing parametrisation
       carries no information worth trusting.
    2. **Fallback.** If Gatheral's denominator is not positive, or any term
       is non-finite, at the clamped point, the local variance falls back to
       the surface's *implied* variance ``w(k, T) / T`` there (and to the ATM
       implied variance if even that is non-finite) — the natural
       zeroth-order proxy of local variance.
    3. **Range.** The local variance is clipped to
       ``[variance_floor, variance_cap_multiple · w(0, T) / T]``.

    Inside the domain, where the surface is arbitrage-free and the local
    variance is within range, the guarded value equals raw Dupire exactly.
    Every branch is selected by ``jnp.where`` on sanitised inputs (the
    clamped ``k`` and a dummy denominator), so gradients stay finite when the
    unguarded branch would be NaN.

    Attributes:
        n_sd: Half-width of the trusted ``k`` domain in ATM total-implied
            standard deviations ``sqrt(w(0, T))``. Default 8.
        variance_floor: Lower bound on the local variance. Default ``1e-8``.
        variance_cap_multiple: Upper bound on the local variance as a
            multiple of the ATM implied variance ``w(0, T) / T``. Default 25
            (local vol at most 5× the ATM implied vol).
    """

    n_sd: float = eqx.field(static=True, default=8.0)
    variance_floor: float = eqx.field(static=True, default=1e-8)
    variance_cap_multiple: float = eqx.field(static=True, default=25.0)


#: The guard used by the Monte Carlo local-vol path generator and the SLV
#: leverage calibration unless the caller passes another one.
DEFAULT_LOCAL_VOL_GUARD = LocalVolGuard()


def guarded_local_variance(
    surface: _TotalVarianceSurface,
    log_moneyness: Float[Array, ""],
    expiry: Float[Array, ""],
    guard: LocalVolGuard = DEFAULT_LOCAL_VOL_GUARD,
) -> Float[Array, ""]:
    """Dupire local variance, made total by a :class:`LocalVolGuard`.

    Args:
        surface: Object exposing ``total_variance(k, T)``.
        log_moneyness: ``k = log(K / F(T))``; clamped to the guard's domain.
        expiry: Year fraction, must be > 0.
        guard: Domain / fallback / range configuration.

    Returns:
        Finite local variance ``σ²_loc(k, T)`` in
        ``[guard.variance_floor, guard.variance_cap_multiple · w(0, T)/T]``.
    """
    _check_x64()
    w_atm = surface.total_variance(jnp.zeros_like(log_moneyness), expiry)
    w_atm = jnp.where(jnp.isfinite(w_atm), jnp.maximum(w_atm, 1e-12), 1e-12)
    var_atm = w_atm / expiry
    half_width = guard.n_sd * jnp.sqrt(w_atm)
    k = jnp.clip(log_moneyness, -half_width, half_width)

    numerator, g, w = _dupire_terms(surface, k, expiry)
    ok = jnp.isfinite(numerator) & jnp.isfinite(g) & (g > 1e-12)
    # Double-where: the division only ever sees a positive denominator.
    raw = jnp.where(ok, numerator, 0.0) / jnp.where(ok, g, 1.0)
    implied = jnp.where(jnp.isfinite(w), w, w_atm) / expiry
    var = jnp.where(ok, raw, implied)
    var = jnp.where(jnp.isfinite(var), var, var_atm)
    return jnp.clip(
        var, guard.variance_floor, guard.variance_cap_multiple * var_atm
    )


def guarded_local_vol(
    surface: _TotalVarianceSurface,
    log_moneyness: Float[Array, ""],
    expiry: Float[Array, ""],
    guard: LocalVolGuard = DEFAULT_LOCAL_VOL_GUARD,
) -> Float[Array, ""]:
    """Dupire local volatility, made total by a :class:`LocalVolGuard`.

    Equal to :func:`dupire_local_vol` wherever the latter is finite, inside
    the guard's domain and range; see :class:`LocalVolGuard` for what happens
    elsewhere.

    Args:
        surface: Object exposing ``total_variance(k, T)``.
        log_moneyness: ``k = log(K / F(T))``.
        expiry: Year fraction, must be > 0.
        guard: Domain / fallback / range configuration.

    Returns:
        Finite, strictly positive local volatility.
    """
    return jnp.sqrt(guarded_local_variance(surface, log_moneyness, expiry, guard))


def dupire_local_vol_from_strike(
    surface: _TotalVarianceSurface,
    strike: Float[Array, ""],
    expiry: Float[Array, ""],
    forward: Float[Array, ""],
) -> Float[Array, ""]:
    """Dupire local vol given absolute strike + forward.

    Thin ergonomic wrapper around :func:`dupire_local_vol` that
    converts ``K`` to ``k = log(K / F(T))`` internally.

    Args:
        surface: Object exposing ``total_variance(k, T)``.
        strike: Absolute strike ``K``.
        expiry: Year fraction.
        forward: Forward at ``expiry``, i.e. ``F(T)``. For a
            deterministic-rate equity this is
            ``spot * exp((r - q) * T)``.

    Returns:
        Scalar local volatility ``sigma_loc(K, T)``.
    """
    k = jnp.log(strike / forward)
    return dupire_local_vol(surface, k, expiry)
