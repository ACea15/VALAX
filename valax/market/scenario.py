"""Scenario data structures for risk factor shocks."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class MarketScenario(eqx.Module):
    """Additive risk factor shocks representing a single scenario.

    Each field is a delta (change), not an absolute level.
    Apply to a MarketData via ``apply_scenario(base, scenario)``.

    When ``multiplicative`` is True, ``spot_shocks`` are interpreted as
    returns: ``new_spot = old_spot * (1 + spot_shocks)``. All other
    shocks are always additive.

    Attributes:
        spot_shocks: Spot price changes per asset.
        vol_shocks: Volatility bumps per asset.
        rate_shocks: Zero-rate bumps at each curve pillar.
        dividend_shocks: Dividend yield bumps per asset.
        multiplicative: If True, spot_shocks are returns.
    """

    spot_shocks: Float[Array, " n_assets"]
    vol_shocks: Float[Array, " n_assets"]
    rate_shocks: Float[Array, " n_pillars"]
    dividend_shocks: Float[Array, " n_assets"]
    multiplicative: bool = eqx.field(static=True, default=False)


class ScenarioSet(eqx.Module):
    """Batch of scenarios with a leading ``n_scenarios`` axis.

    Every leaf array has shape ``(n_scenarios, ...)``, matching the
    corresponding ``MarketScenario`` field shape with a prepended batch
    dimension.  Designed for ``jax.vmap`` over the scenario axis.

    Attributes:
        spot_shocks: Shape ``(n_scenarios, n_assets)``.
        vol_shocks: Shape ``(n_scenarios, n_assets)``.
        rate_shocks: Shape ``(n_scenarios, n_pillars)``.
        dividend_shocks: Shape ``(n_scenarios, n_assets)``.
        multiplicative: If True, spot_shocks are returns.
    """

    spot_shocks: Float[Array, "n_scenarios n_assets"]
    vol_shocks: Float[Array, "n_scenarios n_assets"]
    rate_shocks: Float[Array, "n_scenarios n_pillars"]
    dividend_shocks: Float[Array, "n_scenarios n_assets"]
    multiplicative: bool = eqx.field(static=True, default=False)


def zero_scenario(
    n_assets: int,
    n_pillars: int,
) -> MarketScenario:
    """Create a no-op scenario (all shocks are zero)."""
    return MarketScenario(
        spot_shocks=jnp.zeros(n_assets),
        vol_shocks=jnp.zeros(n_assets),
        rate_shocks=jnp.zeros(n_pillars),
        dividend_shocks=jnp.zeros(n_assets),
    )


def stack_scenarios(scenarios: list[MarketScenario]) -> ScenarioSet:
    """Stack a list of MarketScenario into a batched ScenarioSet."""
    return ScenarioSet(
        spot_shocks=jnp.stack([s.spot_shocks for s in scenarios]),
        vol_shocks=jnp.stack([s.vol_shocks for s in scenarios]),
        rate_shocks=jnp.stack([s.rate_shocks for s in scenarios]),
        dividend_shocks=jnp.stack([s.dividend_shocks for s in scenarios]),
        multiplicative=scenarios[0].multiplicative,
    )
