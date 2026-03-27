"""Scenario generation for VaR and stress testing.

Provides three flavours:

1. **Parametric** — correlated samples from multivariate normal or t.
2. **Historical** — observed risk factor changes sliced into scenarios.
3. **Stress** — deterministic named shocks (parallel, steepener, etc.).

All generators return ``MarketScenario`` or ``ScenarioSet`` pytrees
ready for ``apply_scenario`` and ``jax.vmap``.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.market.scenario import MarketScenario, ScenarioSet, stack_scenarios


# ── Parametric (Monte Carlo) scenarios ───────────────────────────────


def parametric_scenarios(
    key: jax.Array,
    cov: Float[Array, "n_factors n_factors"],
    n_scenarios: int,
    n_assets: int,
    n_pillars: int,
    distribution: str = "normal",
    df: float = 5.0,
) -> ScenarioSet:
    """Sample correlated risk factor scenarios from a parametric distribution.

    Column ordering in the covariance matrix must be::

        [spot_0 .. spot_{n-1}, vol_0 .. vol_{n-1},
         rate_0 .. rate_{p-1}, div_0 .. div_{n-1}]

    where ``n = n_assets`` and ``p = n_pillars``.

    Args:
        key: JAX PRNG key.
        cov: Covariance matrix of all risk factors.
        n_scenarios: Number of scenarios to generate.
        n_assets: Number of spot / vol / dividend factors.
        n_pillars: Number of rate curve pillars.
        distribution: ``"normal"`` or ``"t"``.
        df: Degrees of freedom for the t-distribution.

    Returns:
        A ``ScenarioSet`` with ``n_scenarios`` rows.
    """
    n_factors = cov.shape[0]
    L = jnp.linalg.cholesky(cov)

    key1, key2 = jax.random.split(key)
    z = jax.random.normal(key1, shape=(n_scenarios, n_factors))

    if distribution == "t":
        # t = normal / sqrt(chi2 / df)
        # chi2(df) = 2 * Gamma(df/2, 1)
        chi2 = 2.0 * jax.random.gamma(key2, df / 2.0, shape=(n_scenarios, 1))
        z = z * jnp.sqrt(df / chi2)

    samples = z @ L.T  # (n_scenarios, n_factors)

    return _split_factors(samples, n_assets, n_pillars)


# ── Historical simulation scenarios ──────────────────────────────────


def historical_scenarios(
    returns: Float[Array, "n_obs n_factors"],
    n_assets: int,
    n_pillars: int,
) -> ScenarioSet:
    """Build scenarios from historical risk factor changes.

    Args:
        returns: Observed daily changes with columns ordered as
            ``[spots, vols, rates, dividends]``.
        n_assets: Number of spot / vol / dividend factors.
        n_pillars: Number of rate curve pillars.

    Returns:
        A ``ScenarioSet`` with ``n_scenarios = n_obs``.
    """
    return _split_factors(returns, n_assets, n_pillars)


# ── Stress / deterministic scenarios ─────────────────────────────────


def stress_scenario(
    n_assets: int,
    n_pillars: int,
    spot_shock: float = 0.0,
    vol_shock: float = 0.0,
    parallel_rate_shift: float = 0.0,
    rate_shocks: Float[Array, " n_pillars"] | None = None,
    dividend_shock: float = 0.0,
) -> MarketScenario:
    """Construct a single deterministic stress scenario.

    If ``rate_shocks`` is provided it is used directly; otherwise
    a uniform ``parallel_rate_shift`` is applied to all pillars.
    """
    if rate_shocks is None:
        rate_shocks = jnp.full(n_pillars, parallel_rate_shift)
    return MarketScenario(
        spot_shocks=jnp.full(n_assets, spot_shock),
        vol_shocks=jnp.full(n_assets, vol_shock),
        rate_shocks=rate_shocks,
        dividend_shocks=jnp.full(n_assets, dividend_shock),
    )


def steepener(
    n_assets: int,
    n_pillars: int,
    short_bump: float,
    long_bump: float,
) -> MarketScenario:
    """Steepener: linearly interpolated rate shock from short to long end."""
    rate_shocks = jnp.linspace(short_bump, long_bump, n_pillars)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


def flattener(
    n_assets: int,
    n_pillars: int,
    short_bump: float,
    long_bump: float,
) -> MarketScenario:
    """Flattener: linearly interpolated rate shock (typically short up, long down)."""
    rate_shocks = jnp.linspace(short_bump, long_bump, n_pillars)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


def butterfly(
    n_assets: int,
    n_pillars: int,
    wing_bump: float,
    belly_bump: float,
) -> MarketScenario:
    """Butterfly: quadratic profile — wings at ``wing_bump``, belly at ``belly_bump``.

    The bump at each pillar follows a parabola::

        bump(x) = wing + (belly - wing) * 4 * x * (1 - x)

    where ``x`` runs from 0 to 1 across the pillars.
    """
    x = jnp.linspace(0.0, 1.0, n_pillars)
    rate_shocks = wing_bump + (belly_bump - wing_bump) * 4.0 * x * (1.0 - x)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


# ── Internal helpers ─────────────────────────────────────────────────


def _split_factors(
    samples: Float[Array, "n_scenarios n_factors"],
    n_assets: int,
    n_pillars: int,
) -> ScenarioSet:
    """Slice a factor matrix into a ScenarioSet.

    Column ordering: [spots, vols, rates, dividends].
    """
    i = 0
    spot_shocks = samples[:, i : i + n_assets]; i += n_assets
    vol_shocks = samples[:, i : i + n_assets]; i += n_assets
    rate_shocks = samples[:, i : i + n_pillars]; i += n_pillars
    dividend_shocks = samples[:, i : i + n_assets]

    return ScenarioSet(
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        rate_shocks=rate_shocks,
        dividend_shocks=dividend_shocks,
    )
