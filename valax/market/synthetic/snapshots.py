"""Full :class:`~valax.market.MarketData` snapshot generators.

Builds a canonical :class:`MarketData` (spots, vols, dividends, curve)
plus optionally a correlation matrix usable to construct a
:class:`~valax.models.MultiAssetGBMModel`.

Stream names are documented per call so a downstream test can audit
exactly which keys were consumed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.market.data import MarketData

from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.correlations import sample_correlation_from_config
from valax.market.synthetic.curves import sample_discount_curve
from valax.market.synthetic.seeds import SeedRegistry


def _u_vec(
    key: Array, n: int, low: float, high: float
) -> Float[Array, " n"]:
    return jax.random.uniform(
        key, shape=(n,), dtype=jnp.float64, minval=low, maxval=high
    )


def sample_market_data(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
) -> MarketData:
    """Draw a complete :class:`MarketData` snapshot.

    Args:
        registry: Seed registry.  Stream names used:
            ``synthetic.snapshot.{spots,vols,dividends}`` and whatever
            :func:`sample_discount_curve` consumes for the configured
            ``cfg.curve_kind``.
        cfg: Configuration.

    Returns:
        :class:`MarketData` with vectors of length ``cfg.n_assets`` and
        a discount curve produced by the configured curve generator.

    Notes:
        - Spots are strictly positive by construction (uniform on a
          positive range).
        - Vols are strictly positive.
        - Dividends are non-negative (the default ``div_range`` starts
          at zero).
        - All arrays are ``float64``.
    """
    n = cfg.n_assets
    spots = _u_vec(
        registry.key("synthetic.snapshot.spots"), n, *cfg.spot_range
    )
    vols = _u_vec(
        registry.key("synthetic.snapshot.vols"), n, *cfg.vol_range
    )
    divs = _u_vec(
        registry.key("synthetic.snapshot.dividends"), n, *cfg.div_range
    )
    curve = sample_discount_curve(registry, cfg)
    return MarketData(
        spots=spots,
        vols=vols,
        dividends=divs,
        discount_curve=curve,
    )


def sample_market_with_correlation(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    block_sizes: tuple[int, ...] | None = None,
) -> tuple[MarketData, Float[Array, "n_assets n_assets"]]:
    """Sample a :class:`MarketData` and a matching correlation matrix.

    Args:
        registry: Seed registry.
        cfg: Configuration.
        block_sizes: Required only when ``cfg.correlation_kind == 'block'``.

    Returns:
        Tuple ``(market_data, correlation)``.  The correlation matrix
        can be passed straight to
        :class:`~valax.models.MultiAssetGBMModel` together with
        ``market_data.vols`` and a flat ``market_data.dividends``.
    """
    md = sample_market_data(registry, cfg)
    corr = sample_correlation_from_config(registry, cfg, block_sizes)
    return md, corr


__all__ = ["sample_market_data", "sample_market_with_correlation"]
