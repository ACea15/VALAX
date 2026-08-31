r"""G2++ two-factor short-rate Monte Carlo path generator.

Uses the **exact** conditional distribution of the two correlated
Ornstein-Uhlenbeck factors at each time step, eliminating the
discretisation bias of a naive Euler-Maruyama scheme.  Conditional on the
factors at time :math:`t`, the pair :math:`(x(t+\Delta t), y(t+\Delta t))`
is jointly Gaussian with

.. math::

    \mathbb{E}[x(t+\Delta t) \mid x(t)] = x(t)\,e^{-a\Delta t}, \qquad
    \mathbb{E}[y(t+\Delta t) \mid y(t)] = y(t)\,e^{-b\Delta t}

and time-homogeneous conditional covariance ``g2pp_factor_covariance``.
Increments are drawn by Cholesky-factoring that :math:`2\times2` matrix
once, then applying it to standard normals — the two-factor analogue of the
one-factor exact scheme in :mod:`valax.pricing.mc.hull_white_paths`.

The short rate is recovered as :math:`r(t) = x(t) + y(t) + \varphi(t)` and
the money-market log-discount factor is accumulated with the trapezoidal
rule within each interval.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, §4.2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.g2pp import (
    G2PPModel,
    g2pp_phi,
    g2pp_factor_covariance,
    g2pp_instantaneous_forward,
    g2pp_market_df,
)


class G2PPPathResult(eqx.Module):
    r"""Output of :func:`generate_g2pp_paths`.

    Attributes:
        factor_x: Simulated first factor :math:`x(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        factor_y: Simulated second factor :math:`y(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        short_rates: Simulated short rates
            :math:`r(t_i) = x(t_i) + y(t_i) + \varphi(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        log_discount_factors: Cumulative log money-market discount factor
            :math:`-\int_0^{t_i} r(s)\,ds` per path and grid point, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
    """

    factor_x: Float[Array, "n_paths n_steps_plus1"]
    factor_y: Float[Array, "n_paths n_steps_plus1"]
    short_rates: Float[Array, "n_paths n_steps_plus1"]
    log_discount_factors: Float[Array, "n_paths n_steps_plus1"]


def generate_g2pp_paths(
    model: G2PPModel,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> G2PPPathResult:
    r"""Simulate G2++ short-rate paths via exact two-factor conditional sampling.

    At each step the factor increment is drawn from its exact conditional
    Gaussian distribution (Cholesky of the :math:`2\times2` conditional
    covariance), so the factor marginals are exact for any step size.  The
    money-market log-discount factor accumulates the path integral of the
    short rate.  The **deterministic forward part** of the drift is integrated
    exactly -- over each step it telescopes to the market discount-factor ratio
    :math:`\ln P^M(0, t_i)/P^M(0, t_{i+1})` -- so a curved initial forward
    introduces no repricing bias (the two-factor analogue of Hull-White's exact
    step-average ``hw_alpha_average``).  Only the small, smooth convexity and
    stochastic ``x + y`` parts use the trapezoidal rule.

    Args:
        model: G2++ model carrying the initial curve and factor parameters.
        T: Horizon in year fractions.
        n_steps: Number of equally-spaced time steps.
        n_paths: Number of independent Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        :class:`G2PPPathResult` with per-path factor, short-rate and
        log-discount-factor trajectories of shape ``(n_paths, n_steps + 1)``.
    """
    dt = jnp.asarray(T / n_steps, dtype=jnp.float64)
    times = jnp.linspace(0.0, T, n_steps + 1)

    e_adt = jnp.exp(-model.mean_reversion_x * dt)
    e_bdt = jnp.exp(-model.mean_reversion_y * dt)

    # Cholesky of the 2x2 conditional covariance: increment = z @ L.T draws a
    # correlated (x, y) shock with exactly this covariance for z ~ N(0, I).
    cov = g2pp_factor_covariance(model, dt)
    chol = jnp.linalg.cholesky(cov)

    phi_grid = g2pp_phi(model, times)  # (n_steps + 1,)

    # Deterministic drift split: phi = f^M(0, .) + convexity.  The forward part
    # is integrated exactly (it telescopes to a market-DF ratio), the small
    # convexity part is trapezoided.
    ln_pm_grid = jnp.log(g2pp_market_df(model, times))                   # (n+1,)
    forward_grid = jax.vmap(lambda s: g2pp_instantaneous_forward(model, s))(times)
    convexity_grid = phi_grid - forward_grid                            # (n+1,)

    # Standard normals for both factors, all steps and paths: (n_steps, n_paths, 2).
    normals = jax.random.normal(
        key, shape=(n_steps, n_paths, 2), dtype=jnp.float64
    )

    def step_fn(carry, xs):
        x_i, y_i, log_df_i = carry
        z, i = xs  # z: (n_paths, 2)

        shock = z @ chol.T  # (n_paths, 2)
        x_next = x_i * e_adt + shock[:, 0]
        y_next = y_i * e_bdt + shock[:, 1]

        r_i = x_i + y_i + phi_grid[i]
        r_next = x_next + y_next + phi_grid[i + 1]

        # -∫ r ds over the step: exact forward part + trapezoidal remainder.
        det_forward = ln_pm_grid[i + 1] - ln_pm_grid[i]
        stochastic = 0.5 * dt * ((x_i + y_i) + (x_next + y_next))
        convexity = 0.5 * dt * (convexity_grid[i] + convexity_grid[i + 1])
        log_df_next = log_df_i + det_forward - convexity - stochastic

        return (x_next, y_next, log_df_next), (x_next, y_next, r_next, log_df_next)

    zeros = jnp.zeros(n_paths, dtype=jnp.float64)
    r0 = zeros + phi_grid[0]

    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    _, (x_seq, y_seq, r_seq, ldf_seq) = jax.lax.scan(
        step_fn, (zeros, zeros, zeros), (normals, step_indices),
    )

    def _prepend(col0, seq):
        return jnp.concatenate([col0[None, :], seq], axis=0).T

    return G2PPPathResult(
        factor_x=_prepend(zeros, x_seq),
        factor_y=_prepend(zeros, y_seq),
        short_rates=_prepend(r0, r_seq),
        log_discount_factors=_prepend(zeros, ldf_seq),
    )
