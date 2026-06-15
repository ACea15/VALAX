"""Local volatility Monte Carlo path generation.

Log-Euler scheme with Itô correction:

    log S_{t+dt} = log S_t + (r - q - 0.5 * sigma_loc^2) * dt
                 + sigma_loc * sqrt(dt) * Z,

where ``sigma_loc = dupire_local_vol(surface, k_t, t)`` and
``k_t = log(S_t / F(t))`` with ``F(t) = S_0 * exp((r - q) * t)``.

Design choices (mirroring the Andersen-QE Heston template at
``valax/pricing/mc/paths.py``):

    * **``jax.lax.scan`` over time**, with state ``log_S`` shaped
      ``(n_paths,)``. Inside each step the Dupire local-vol evaluation
      is ``jax.vmap``-ed across paths.
    * **Midpoint-in-time σ**: at step ``n`` (from ``t_n = n·dt`` to
      ``t_{n+1} = (n+1)·dt``) we evaluate
      ``sigma_loc(S_n, t_n + 0.5·dt)``. This (a) avoids querying the
      Dupire formula at the singular ``T = 0`` boundary, where total
      variance vanishes and the ``1/w`` terms in the denominator blow
      up, and (b) gives a half-order improvement in the weak-error
      constant compared to plain left-endpoint Euler.
    * **Per-step PRNG split**: ``keys = random.split(key, n_steps)``,
      then each step draws one Gaussian vector of shape ``(n_paths,)``.
    * **Output shape**: ``(n_paths, n_steps + 1)`` matching
      ``generate_gbm_paths`` and ``generate_heston_paths``.

Autodiff: the Dupire evaluation involves second derivatives of the
surface's total variance, so ``jax_enable_x64`` must be on (enforced at
the Dupire layer). Gradients through ``generate_local_vol_paths`` w.r.t.
surface parameters are well-defined as long as no path triggers a
butterfly-arbitrage NaN at the queried ``(k_t, t_n)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.dupire import dupire_local_vol


def generate_local_vol_paths(
    model: LocalVolModel,
    spot: Float[Array, ""],
    T: Float[Array, ""] | float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> Float[Array, "n_paths n_steps_plus1"]:
    """Generate spot paths under a local volatility SDE.

    Args:
        model: A ``LocalVolModel`` carrying the surface and rate / div.
        spot: Initial spot price ``S_0``.
        T: Terminal time (year fraction).
        n_steps: Number of time steps (``dt = T / n_steps``).
        n_paths: Number of Monte Carlo paths.
        key: Top-level PRNG key.

    Returns:
        Spot paths, shape ``(n_paths, n_steps + 1)``. Column 0 is
        ``spot`` broadcast across paths; column ``n_steps`` is ``S_T``.
    """
    T = jnp.asarray(T)
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    mu = model.rate - model.dividend

    log_spot = jnp.log(spot)
    surface = model.surface

    # σ_loc evaluation at (S_n, t_n), vmapped over the n_paths log-spots.
    # F(t_n) = spot * exp(mu * t_n) ⇒ k_t = log(S_n / F(t_n))
    #                              = log_S_n - log_spot - mu * t_n.
    def sigma_at_paths(
        log_S: Float[Array, " n_paths"],
        t: Float[Array, ""],
    ) -> Float[Array, " n_paths"]:
        log_F = log_spot + mu * t
        k = log_S - log_F  # log-moneyness, shape (n_paths,)
        # Per-path Dupire evaluation. The surface itself is treated as a
        # closed-over pytree; the scalar arguments are vmapped.
        return jax.vmap(lambda kk: dupire_local_vol(surface, kk, t))(k)

    def step(carry, scan_input):
        log_S = carry
        t_n, key_t = scan_input
        z = jax.random.normal(key_t, shape=(n_paths,))
        sigma = sigma_at_paths(log_S, t_n)
        log_S_next = (
            log_S
            + (mu - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * z
        )
        return log_S_next, log_S_next

    log_S0 = jnp.broadcast_to(log_spot, (n_paths,))

    # Midpoint times: t_n + 0.5·dt for n = 0 .. n_steps - 1, i.e.
    # 0.5·dt, 1.5·dt, ..., (n_steps - 0.5)·dt. Avoids the T = 0
    # singularity of Dupire's 1/w terms.
    times = (jnp.arange(n_steps).astype(T.dtype) + 0.5) * dt
    keys = jax.random.split(key, n_steps)

    _, log_S_seq = jax.lax.scan(step, log_S0, (times, keys))
    # log_S_seq has shape (n_steps, n_paths); prepend the initial
    # log-spot and transpose to (n_paths, n_steps + 1).
    log_S = jnp.concatenate([log_S0[None, :], log_S_seq], axis=0)
    return jnp.exp(log_S).T
