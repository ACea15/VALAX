"""Synthetic market evolution: a tape of :class:`MarketData` snapshots.

Given a starting :class:`MarketData` and a date grid, simulate one
realisation of the market through those dates.  Spots follow a
correlated GBM (via the existing
:func:`~valax.pricing.mc.multi_asset_paths.generate_correlated_gbm_paths`);
vols, dividends, and the discount curve are held constant in this
first iteration.  Returns a stacked :class:`MarketData` whose array
leaves carry a leading ``n_dates`` axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.market.data import MarketData
from valax.models.multi_asset import MultiAssetGBMModel
from valax.pricing.mc.multi_asset_paths import generate_correlated_gbm_paths

from valax.market.synthetic.seeds import SeedRegistry


def evolve_market(
    registry: SeedRegistry,
    md0: MarketData,
    dates: Int[Array, " n_dates"],
    correlation: Float[Array, "n_assets n_assets"],
    *,
    n_paths: int = 1,
    stream_name: str = "synthetic.paths.spots",
) -> MarketData:
    """Evolve ``md0`` over ``dates`` via correlated GBM on spots.

    The returned :class:`MarketData` is a *stacked pytree*: each array
    leaf has a leading ``n_dates`` axis.  Vols, dividends, and the
    discount curve are broadcast unchanged across time, so the result
    is identical to ``md0`` at every date except for ``spots``.

    Args:
        registry: Seed registry.
        md0: Initial market state at ``dates[0]``.
        dates: Strictly increasing ordinal dates.  ``dates[0]`` must
            equal ``md0.discount_curve.reference_date``.
        correlation: Correlation matrix for the spot Brownians.
        n_paths: Number of paths.  Set to 1 for a single tape;
            increase to vmap a batch of tapes.
        stream_name: Registry stream name override.

    Returns:
        :class:`MarketData` whose ``spots`` has shape
        ``(n_paths, n_dates, n_assets)`` when ``n_paths > 1`` else
        ``(n_dates, n_assets)``.  Other fields broadcast accordingly.

    Notes:
        - This is *one* synthetic future, not a calibration tool.  It
          is intended for end-to-end "tape" tests that verify the rest
          of the library can be driven through a sequence of snapshots.
        - The GBM drift uses a single flat risk-free rate read from the
          curve (zero rate between ``dates[0]`` and ``dates[-1]``); a
          full stochastic-rate path is out of scope for v1.
    """
    if dates.shape[0] < 2:
        raise ValueError(f"dates must have length >= 2, got {dates.shape[0]}")

    # Year fractions from dates[0] to each subsequent date.
    times = year_fraction(dates[0], dates, md0.discount_curve.day_count)
    # We need n_steps == n_dates - 1 to land on each requested date.
    n_steps = int(dates.shape[0]) - 1
    T = float(times[-1])
    if T <= 0.0:
        raise ValueError(
            "dates must span a strictly positive interval; got T=0."
        )

    # Approximate flat rate for the whole window from the discount curve.
    df_end = md0.discount_curve(dates[-1])
    flat_rate = -jnp.log(df_end) / T

    model = MultiAssetGBMModel(
        vols=md0.vols,
        rate=flat_rate,
        dividends=md0.dividends,
        correlation=correlation,
    )

    key = registry.key(stream_name)
    # Shape: (n_paths, n_steps + 1, n_assets) == (n_paths, n_dates, n_assets).
    paths = generate_correlated_gbm_paths(
        model=model,
        spots=md0.spots,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        key=key,
    )

    if n_paths == 1:
        spots_path = paths[0]  # (n_dates, n_assets)
        leading_shape: tuple[int, ...] = (int(dates.shape[0]),)
    else:
        spots_path = paths  # (n_paths, n_dates, n_assets)
        leading_shape = (n_paths, int(dates.shape[0]))

    # Broadcast vols/dividends across the leading shape; curve is kept
    # as the same DiscountCurve object (held constant through time).
    vols_b = jnp.broadcast_to(
        md0.vols, leading_shape + md0.vols.shape
    )
    divs_b = jnp.broadcast_to(
        md0.dividends, leading_shape + md0.dividends.shape
    )
    return MarketData(
        spots=spots_path,
        vols=vols_b,
        dividends=divs_b,
        discount_curve=md0.discount_curve,
    )


__all__ = ["evolve_market"]
