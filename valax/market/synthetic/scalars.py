"""Scalar-market generator (single-asset, single-strike, single-expiry).

Returns the six JAX scalars consumed by the analytic Black-Scholes
pricer in ``examples/comparisons/01_european_options.py``.  This is the
narrowest possible synthetic surface and is intended for fuzz-testing
the closed-form pricers and their Greeks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.seeds import SeedRegistry


# Default expiry range for the scalar generator (years).
_DEFAULT_EXPIRY_RANGE: tuple[float, float] = (0.05, 2.0)

# Default moneyness range for the scalar generator's strike draw.
# Strike = spot * moneyness.
_DEFAULT_MONEYNESS_RANGE: tuple[float, float] = (0.7, 1.3)


def _u(key: Array, low: float, high: float) -> Float[Array, ""]:
    """Uniform scalar draw in ``[low, high]``."""
    return jax.random.uniform(
        key, shape=(), dtype=jnp.float64, minval=low, maxval=high
    )


def sample_scalar_market(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    expiry_range: tuple[float, float] = _DEFAULT_EXPIRY_RANGE,
    moneyness_range: tuple[float, float] = _DEFAULT_MONEYNESS_RANGE,
) -> dict[str, Float[Array, ""]]:
    """Draw a single-asset / single-option scalar market snapshot.

    Args:
        registry: Seed registry.
        cfg: Synthetic market config; ``cfg.spot_range``, ``cfg.vol_range``,
            ``cfg.rate_range``, ``cfg.div_range`` are used.
        expiry_range: Uniform sampling range for the option expiry, in
            year fractions.
        moneyness_range: Uniform sampling range for the strike, expressed
            as ``strike / spot``.

    Returns:
        Dict with keys ``"spot"``, ``"vol"``, ``"rate"``, ``"dividend"``,
        ``"expiry"``, ``"strike"`` — all ``Float[Array, ""]`` float64.

    Stream names:
        ``synthetic.scalar.{spot,vol,rate,dividend,expiry,moneyness}``.
    """
    spot = _u(registry.key("synthetic.scalar.spot"), *cfg.spot_range)
    vol = _u(registry.key("synthetic.scalar.vol"), *cfg.vol_range)
    rate = _u(registry.key("synthetic.scalar.rate"), *cfg.rate_range)
    dividend = _u(registry.key("synthetic.scalar.dividend"), *cfg.div_range)
    expiry = _u(registry.key("synthetic.scalar.expiry"), *expiry_range)
    moneyness = _u(
        registry.key("synthetic.scalar.moneyness"), *moneyness_range
    )
    strike = spot * moneyness
    return {
        "spot": spot,
        "vol": vol,
        "rate": rate,
        "dividend": dividend,
        "expiry": expiry,
        "strike": strike,
    }


__all__ = ["sample_scalar_market"]
