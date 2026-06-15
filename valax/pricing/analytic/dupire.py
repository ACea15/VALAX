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

    # sigma_loc^2 = numerator / denominator. NaN if denominator <= 0
    # (butterfly arbitrage diagnostic).
    sigma2 = numerator / g
    return jnp.sqrt(sigma2)


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
