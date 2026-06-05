"""Configuration object for synthetic-market generators.

A single ``SyntheticMarketConfig`` carries every shape, range, and
distribution choice consumed by the Stage-1 generators.  All fields
are static (compile-time) so the config can travel inside an
``eqx.Module`` without breaking JIT.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Int
from jax import Array

from valax.dates.daycounts import ymd_to_ordinal


def _default_reference_date() -> Array:
    """Default reference date factory: 2026-01-01 ordinal.

    Used as ``default_factory`` because :class:`eqx.Module` (and
    :mod:`dataclasses`) forbid mutable / non-hashable defaults like
    JAX arrays.
    """
    return ymd_to_ordinal(2026, 1, 1)

# Default NSS-style coefficient ranges:
#   beta0  level         in [-0.01, 0.06]
#   beta1  slope         in [-0.03, 0.03]
#   beta2  curvature     in [-0.04, 0.04]
#   beta3  second hump   in [-0.04, 0.04]
#   tau1   short decay   in [ 0.5,   3.0]  (years)
#   tau2   long  decay   in [ 3.0,  15.0]  (years)
_DEFAULT_NSS_RANGES: tuple[tuple[float, float], ...] = (
    (-0.01, 0.06),
    (-0.03, 0.03),
    (-0.04, 0.04),
    (-0.04, 0.04),
    (0.5, 3.0),
    (3.0, 15.0),
)


class SyntheticMarketConfig(eqx.Module):
    """Static configuration for synthetic-market sampling.

    Attributes:
        n_assets: Number of underlyings to sample.
        reference_date: Valuation date as an ordinal (days since
            1970-01-01).  Defaults to 2026-01-01.
        spot_range: Uniform sampling range for spot prices.
        vol_range: Uniform sampling range for implied volatilities.
        rate_range: Uniform sampling range for the flat short-rate
            level (only used when ``curve_kind == "flat"``).
        div_range: Uniform sampling range for continuous dividend yields.
        curve_kind: ``"flat"`` produces a 2-pillar discount curve;
            ``"nss"`` produces a Nelson-Siegel-Svensson curve over a
            fixed pillar grid.
        nss_pillars_years: Maturity grid (in years) for the NSS curve's
            pillar dates.  Defaults to standard money-market + swap grid.
        nss_param_ranges: Six (low, high) tuples for NSS parameters
            ``(beta0, beta1, beta2, beta3, tau1, tau2)``.
        correlation_kind: ``"identity"`` for a no-correlation matrix,
            ``"random"`` for a random valid correlation matrix,
            ``"block"`` for a block-structured matrix.
        min_corr: Lower clip for sampled correlations.
        max_corr: Upper clip for sampled off-diagonal correlations.
        day_count: Day count convention for the synthetic curve.

    Notes:
        Every numeric range field is a plain Python tuple, not a JAX
        array, so the config remains a fully static pytree leaf.  The
        generators do the array construction internally.
    """

    n_assets: int = eqx.field(static=True, default=3)
    reference_date: Int[Array, ""] = eqx.field(
        default_factory=_default_reference_date
    )
    spot_range: tuple[float, float] = eqx.field(
        static=True, default=(50.0, 200.0)
    )
    vol_range: tuple[float, float] = eqx.field(
        static=True, default=(0.10, 0.45)
    )
    rate_range: tuple[float, float] = eqx.field(
        static=True, default=(-0.005, 0.06)
    )
    div_range: tuple[float, float] = eqx.field(
        static=True, default=(0.0, 0.04)
    )
    curve_kind: Literal["flat", "nss"] = eqx.field(
        static=True, default="nss"
    )
    nss_pillars_years: tuple[float, ...] = eqx.field(
        static=True,
        default=(
            1 / 12, 3 / 12, 6 / 12,            # money market
            1.0, 2.0, 3.0, 5.0, 7.0, 10.0,     # swap belly
            15.0, 20.0, 30.0,                  # long end
        ),
    )
    nss_param_ranges: tuple[tuple[float, float], ...] = eqx.field(
        static=True, default=_DEFAULT_NSS_RANGES
    )
    correlation_kind: Literal["identity", "random", "block"] = eqx.field(
        static=True, default="random"
    )
    min_corr: float = eqx.field(static=True, default=-0.3)
    max_corr: float = eqx.field(static=True, default=0.85)
    day_count: str = eqx.field(static=True, default="act_365")


def default_config(n_assets: int = 3) -> SyntheticMarketConfig:
    """Convenience constructor for a config with the package defaults."""
    return SyntheticMarketConfig(n_assets=n_assets)


__all__ = ["SyntheticMarketConfig", "default_config"]
