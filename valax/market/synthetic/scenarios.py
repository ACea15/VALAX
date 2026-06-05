"""Random :class:`~valax.market.ScenarioSet` generator.

Produces shock vectors matching the existing scenario container in
``valax/market/scenario.py``, suitable for feeding into
``valax.risk`` engines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from valax.market.scenario import ScenarioSet

from valax.market.synthetic.seeds import SeedRegistry


# Default shock scales (1-sigma) used when not overridden.
_DEFAULT_SPOT_BPS = 200.0       # 200 bps = 2% spot shock std
_DEFAULT_VOL_BPS = 100.0        # 100 bps = 1 vol-point std
_DEFAULT_RATE_BPS = 25.0        # 25 bps rate parallel-shock std
_DEFAULT_DIV_BPS = 25.0


def sample_scenario_set(
    registry: SeedRegistry,
    n_scenarios: int,
    n_assets: int,
    n_pillars: int,
    *,
    spot_sigma_bps: float = _DEFAULT_SPOT_BPS,
    vol_sigma_bps: float = _DEFAULT_VOL_BPS,
    rate_sigma_bps: float = _DEFAULT_RATE_BPS,
    dividend_sigma_bps: float = _DEFAULT_DIV_BPS,
    multiplicative: bool = True,
) -> ScenarioSet:
    """Draw ``n_scenarios`` iid Gaussian shocks.

    Args:
        registry: Seed registry.
        n_scenarios: Number of scenarios in the batch.
        n_assets: Number of underlyings (controls ``spot_shocks``,
            ``vol_shocks``, ``dividend_shocks`` shapes).
        n_pillars: Number of curve pillars (controls ``rate_shocks``
            shape).
        spot_sigma_bps: 1-sigma for spot shocks, in basis points.
            Interpreted as a return if ``multiplicative=True``,
            else as an absolute spot change.
        vol_sigma_bps: 1-sigma for vol shocks, in basis points.
        rate_sigma_bps: 1-sigma for rate shocks, in basis points.
        dividend_sigma_bps: 1-sigma for dividend shocks, in basis points.
        multiplicative: Forwarded to
            :class:`~valax.market.ScenarioSet.multiplicative`.

    Stream prefix: ``synthetic.scenarios.*``.
    """
    bp = 1e-4
    spot_shocks = (
        spot_sigma_bps
        * bp
        * jax.random.normal(
            registry.key("synthetic.scenarios.spot"),
            shape=(n_scenarios, n_assets),
            dtype=jnp.float64,
        )
    )
    vol_shocks = (
        vol_sigma_bps
        * bp
        * jax.random.normal(
            registry.key("synthetic.scenarios.vol"),
            shape=(n_scenarios, n_assets),
            dtype=jnp.float64,
        )
    )
    rate_shocks = (
        rate_sigma_bps
        * bp
        * jax.random.normal(
            registry.key("synthetic.scenarios.rate"),
            shape=(n_scenarios, n_pillars),
            dtype=jnp.float64,
        )
    )
    dividend_shocks = (
        dividend_sigma_bps
        * bp
        * jax.random.normal(
            registry.key("synthetic.scenarios.dividend"),
            shape=(n_scenarios, n_assets),
            dtype=jnp.float64,
        )
    )
    return ScenarioSet(
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        rate_shocks=rate_shocks,
        dividend_shocks=dividend_shocks,
        multiplicative=multiplicative,
    )


__all__ = ["sample_scenario_set"]
