"""Synthetic observation layer: clean truth → noisy quotes.

Given a ground-truth model or curve, produce the noisy inputs that
each calibration routine in ``valax.calibration`` expects.  Noise is
zero-mean Gaussian with a tunable basis-point (or price-unit) standard
deviation; setting the noise level to zero returns the clean values.

Naming convention
-----------------
- ``synthesize_*`` returns the kind of data a desk receives, with
  noise applied.  Pass ``noise=0`` for the noiseless version.
- The clean (noise-free) computation is kept as a private helper
  ``_clean_*`` so tests can compare against it directly.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol

from valax.market.synthetic.seeds import SeedRegistry


# ── Vol smile ─────────────────────────────────────────────────────


def _clean_sabr_smile(
    model: SABRModel,
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    strikes: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Noise-free SABR implied-vol smile at ``strikes``."""
    return jax.vmap(lambda K: sabr_implied_vol(model, forward, K, expiry))(
        strikes
    )


def synthesize_sabr_smile(
    registry: SeedRegistry,
    model: SABRModel,
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    strikes: Float[Array, " n"],
    *,
    vol_bp_noise: float = 5.0,
    stream_name: str = "synthetic.obs.sabr_smile",
) -> Float[Array, " n"]:
    """Synthesise a noisy SABR implied-vol smile.

    Args:
        registry: Seed registry.
        model: Ground-truth :class:`SABRModel`.
        forward: Forward of the underlying.
        expiry: Time to expiry (year fraction).
        strikes: Strike grid.
        vol_bp_noise: 1-sigma of the additive vol noise, in basis
            points (e.g., ``5`` for 5 bp = 0.0005 vol).
        stream_name: Registry stream name.

    Returns:
        Noisy vol vector of shape ``(n,)``.  With ``vol_bp_noise == 0``
        equals :func:`_clean_sabr_smile`.
    """
    clean = _clean_sabr_smile(model, forward, expiry, strikes)
    if vol_bp_noise == 0.0:
        return clean
    key = registry.key(stream_name)
    eps = jax.random.normal(
        key, shape=strikes.shape, dtype=jnp.float64
    )
    return clean + (vol_bp_noise * 1e-4) * eps


# ── Price strip ───────────────────────────────────────────────────


def synthesize_price_strip(
    registry: SeedRegistry,
    pricer: Callable[..., Float[Array, ""]],
    pricer_args_per_strike: Callable[[Float[Array, ""]], tuple],
    strikes: Float[Array, " n"],
    *,
    price_rel_noise: float = 1e-3,
    stream_name: str = "synthetic.obs.price_strip",
) -> Float[Array, " n"]:
    """Synthesise a noisy strip of option prices.

    The pricer is kept abstract so callers can use the analytic
    Black-Scholes pricer, a Heston semi-analytic pricer, or any Monte
    Carlo wrapper.

    Args:
        registry: Seed registry.
        pricer: Scalar pricer ``pricer(*args) -> price``.
        pricer_args_per_strike: Callable mapping a strike to the tuple
            of arguments to splat into ``pricer``.  Example::

                lambda K: (EuropeanOption(K, T, is_call=True),
                           spot, vol, rate, dividend)

        strikes: Strike grid; each strike is mapped through
            ``pricer_args_per_strike`` and the pricer.
        price_rel_noise: 1-sigma of multiplicative Gaussian noise
            applied to each price (i.e., ``noisy = clean * (1 + eps)``).
        stream_name: Registry stream name.

    Returns:
        Noisy price vector of shape ``(n,)``.
    """
    clean_prices = jnp.stack(
        [pricer(*pricer_args_per_strike(K)) for K in strikes]
    )
    if price_rel_noise == 0.0:
        return clean_prices
    key = registry.key(stream_name)
    eps = jax.random.normal(
        key, shape=clean_prices.shape, dtype=jnp.float64
    )
    return clean_prices * (1.0 + price_rel_noise * eps)


# ── Curve quotes ──────────────────────────────────────────────────


def synthesize_curve_quotes(
    registry: SeedRegistry,
    par_rates: Float[Array, " n"],
    *,
    bp_noise: float = 1.0,
    stream_name: str = "synthetic.obs.curve_quotes",
) -> Float[Array, " n"]:
    """Add basis-point Gaussian noise to a vector of par rates.

    The actual par-rate computation depends on the instrument family
    (deposit, FRA, swap, OIS).  Pre-compute the clean par rates from
    a truth curve via the standard bootstrap helpers, then call this
    function once to add a uniform observation noise floor.

    Args:
        registry: Seed registry.
        par_rates: Clean par rates implied by the truth curve.
        bp_noise: 1-sigma additive noise, in basis points.
        stream_name: Registry stream name.

    Returns:
        Noisy par-rate vector.
    """
    if bp_noise == 0.0:
        return par_rates
    key = registry.key(stream_name)
    eps = jax.random.normal(
        key, shape=par_rates.shape, dtype=jnp.float64
    )
    return par_rates + (bp_noise * 1e-4) * eps


__all__ = [
    "synthesize_sabr_smile",
    "synthesize_price_strip",
    "synthesize_curve_quotes",
    "_clean_sabr_smile",
]
