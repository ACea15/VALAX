"""Correlated multi-asset GBM path generation.

Generates paths for :math:`n` correlated underlyings under the
risk-neutral measure using the **exact log-Euler** (log-Milstein-equivalent
for pure GBM) scheme:

.. math::

    \\log S_i(t + \\Delta t)
    = \\log S_i(t) + (r - q_i - \\tfrac{1}{2}\\sigma_i^2)\\Delta t
      + \\sigma_i \\sqrt{\\Delta t}\\,(L\\,Z)_i

where :math:`Z \\sim \\mathcal{N}(0, I_n)` is a standard normal vector
and :math:`L` is the Cholesky factor of the correlation matrix
(:math:`L L^T = \\rho`). Because pure GBM has a log-linear SDE, the
log-Euler step is **exact** for any :math:`\\Delta t` — there is no
discretization bias. The ``n_steps`` parameter only matters for
path-dependent payoffs (barriers, Asians, autocallables) that need
intermediate observations.

The implementation stays purely in log-space for numerical stability
and returns spots (not log-spots) for compatibility with the existing
payoff functions in :mod:`valax.pricing.mc.payoffs`.

References:
    Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
    §3.2.3 and §4.4.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.models.multi_asset import MultiAssetGBMModel


def generate_correlated_gbm_paths(
    model: MultiAssetGBMModel,
    spots: Float[Array, " n_assets"],
    T: Float[Array, ""] | float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> Float[Array, "n_paths n_steps_plus1 n_assets"]:
    """Simulate correlated GBM paths for :math:`n` underlyings.

    Args:
        model: :class:`MultiAssetGBMModel` (vols, rate, dividends, correlation).
        spots: Initial spot prices, shape ``(n_assets,)``.
        T: Time horizon in year fractions.
        n_steps: Number of time steps (static). Controls the observation
            grid resolution for path-dependent payoffs; the log-Euler
            step is exact for pure GBM regardless of step size, so set
            this based on payoff monitoring needs, not accuracy.
        n_paths: Number of Monte Carlo paths (static).
        key: JAX PRNG key.

    Returns:
        Paths array of shape ``(n_paths, n_steps + 1, n_assets)``,
        with ``paths[:, 0, :] == spots`` and per-asset log-returns
        following the prescribed correlation structure.

    Notes:
        - The Cholesky factor is computed inside the function each call.
          For a fixed correlation matrix across many repricings, consider
          caching via ``jax.jit`` (JAX will reuse the compiled graph).
        - Independent paths use independent Brownian increments; the
          correlation is *between assets within a path*, not between paths.
    """
    n_assets = spots.shape[0]
    dt = jnp.asarray(T, dtype=spots.dtype) / n_steps
    sqrt_dt = jnp.sqrt(dt)

    # Cholesky factor: L @ L.T == correlation (up to numerical noise).
    # For a valid correlation matrix this is stable.
    L = jnp.linalg.cholesky(model.correlation)

    # Risk-neutral drift per asset (log-space).
    # mu_i = (r - q_i - 0.5 * sigma_i^2) * dt   applied per log-step.
    log_drift = (model.rate - model.dividends - 0.5 * model.vols**2) * dt  # (n_assets,)

    # Diffusion coefficient for the log-step: sigma_i * sqrt(dt) * (L Z)_i.
    # Pre-scale L's rows by sigma_i * sqrt(dt) so a single matmul suffices.
    sigma_sqrt_dt = model.vols * sqrt_dt  # (n_assets,)
    # scaled_L[i, :] = sigma_i * sqrt(dt) * L[i, :]
    scaled_L = sigma_sqrt_dt[:, None] * L  # (n_assets, n_assets)

    log_spots0 = jnp.log(spots)  # (n_assets,)

    def single_path(subkey: jax.Array) -> Float[Array, "n_steps_plus1 n_assets"]:
        """Generate one correlated path."""
        # All independent standard normals for this path in one draw.
        # Shape (n_steps, n_assets) — one vector of n_assets increments per step.
        Z = jax.random.normal(subkey, shape=(n_steps, n_assets), dtype=spots.dtype)

        # Correlated log increments per step: (n_steps, n_assets).
        #   log_step[k] = log_drift + Z[k] @ scaled_L.T
        # because (scaled_L @ Z[k])_i = sum_j scaled_L[i,j] * Z[k,j].
        log_steps = log_drift[None, :] + Z @ scaled_L.T

        # Cumulative log-spot trajectory including t=0.
        cum_log = jnp.cumsum(log_steps, axis=0)  # (n_steps, n_assets)
        full_log = jnp.concatenate([log_spots0[None, :], log_spots0 + cum_log], axis=0)

        return jnp.exp(full_log)  # (n_steps + 1, n_assets)

    keys = jax.random.split(key, n_paths)
    return jax.vmap(single_path)(keys)
