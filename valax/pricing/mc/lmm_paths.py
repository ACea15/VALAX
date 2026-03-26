"""LMM forward rate path generation via diffrax.

Simulates N forward rates under the spot LIBOR measure using log-Euler
discretization (Euler-Maruyama on log-forwards via diffrax). The time
grid starts at t=0 (reference date), includes all tenor dates, and uses
configurable subdivisions between tenors.

The output provides forward rates at their fixing dates and path-wise
discount factors DF(0, T_k) for payoff discounting.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from jaxtyping import Float, Int
from jax import Array

from valax.models.lmm import LMMModel, LMMDrift, LMMDiffusion


class LMMPathResult(eqx.Module):
    """Result of LMM forward rate path simulation.

    Attributes:
        forwards_at_fixing: Realized forward rates F_i(T_i) at each fixing,
                            shape (n_paths, N). Used for caplet pricing.
        forwards_at_tenors: Full forward rate panel at each tenor date,
                            shape (n_paths, N, N). Entry [p, i, j] = F_j(T_i).
                            Used for swaption pricing (all forwards at expiry).
        discount_factors: Path-wise discount factors DF(0, T_k) for k=0..N,
                          shape (n_paths, N+1). Includes DF(0, T_0) from the
                          initial curve; subsequent DFs built from realized forwards.
    """

    forwards_at_fixing: Float[Array, "n_paths n_forwards"]
    forwards_at_tenors: Float[Array, "n_paths n_forwards n_forwards"]
    discount_factors: Float[Array, "n_paths n_tenors"]


def generate_lmm_paths(
    model: LMMModel,
    n_steps_per_period: int,
    n_paths: int,
    key: jax.Array,
) -> LMMPathResult:
    """Generate forward rate paths under the LMM using diffrax.

    Simulates N log-forward rates from t=0 (reference date) to the last
    fixing date T_{N-1} under the spot LIBOR measure. Uses Euler-Maruyama
    via diffrax.Euler() with a time grid aligned to tenor dates.

    The simulation from t=0 to T_0 captures the stochastic evolution of
    forward rates before any fixings. At each T_i, forward i is fixed
    and its realized value is recorded.

    Args:
        model: LMM model (initial forwards, vol, correlation, tenors).
        n_steps_per_period: Euler steps between consecutive time grid nodes.
        n_paths: Number of Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        LMMPathResult with forwards at fixings and discount factors.
    """
    N = model.initial_forwards.shape[0]
    n_factors = model.loading_matrix.shape[1]
    tenor_times = model.tenor_times  # (N+1,) year fracs from ref to T_i
    tenor_times_fwd = tenor_times[:-1]  # (N,) year fracs to forward start dates

    # Build time grid: [0, T_0], [T_0, T_1], ..., [T_{N-1}, T_N]
    # with n_steps_per_period subdivisions per segment.
    # Python loop at trace time is fine (N is static from array shape).
    segments = []
    tenor_step_indices = []  # index of each T_i in the full ts array
    idx = 0

    # Segment [0, T_0]: evolution before first fixing
    seg0 = jnp.linspace(jnp.array(0.0), tenor_times[0], n_steps_per_period + 1)
    segments.append(seg0[:-1])
    idx += n_steps_per_period

    # Segments [T_i, T_{i+1}] for i = 0..N-1
    for i in range(N):
        tenor_step_indices.append(idx)  # T_i is at this index
        seg = jnp.linspace(tenor_times[i], tenor_times[i + 1], n_steps_per_period + 1)
        segments.append(seg[:-1])
        idx += n_steps_per_period

    # Final point T_N
    segments.append(tenor_times[-1:])
    ts = jnp.concatenate(segments)

    tenor_step_indices_arr = jnp.array(tenor_step_indices, dtype=jnp.int32)

    total_steps = ts.shape[0]
    t1 = tenor_times[-1]
    dt = t1 / jnp.array(total_steps - 1, dtype=jnp.float64)

    # Build drift and diffusion
    drift = LMMDrift(
        accrual_fractions=model.accrual_fractions,
        loading_matrix=model.loading_matrix,
        tenor_times_fwd=tenor_times_fwd,
        vol_structure=model.vol_structure,
    )
    diffusion = LMMDiffusion(
        loading_matrix=model.loading_matrix,
        tenor_times_fwd=tenor_times_fwd,
        vol_structure=model.vol_structure,
    )

    y0 = jnp.log(model.initial_forwards)
    initial_df = model.initial_df  # DF(0, T_0) from curve

    def single_path(subkey):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=t1, tol=dt / 2, shape=(n_factors,), key=subkey
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, bm),
        )
        sol = diffrax.diffeqsolve(
            terms,
            diffrax.Euler(),
            t0=0.0,
            t1=t1,
            dt0=dt,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        # sol.ys shape: (total_steps, N)
        log_fwd_all = sol.ys

        # Extract F_i(T_i): forward i at its fixing date
        forwards_at_fix = jnp.exp(log_fwd_all[tenor_step_indices_arr, jnp.arange(N)])

        # Discount factors from realized forwards:
        # DF(0, T_0) = initial_df (from curve)
        # DF(0, T_{k+1}) = DF(0, T_0) / prod_{j=0}^{k} (1 + tau_j * F_j(T_j))
        accrual = 1.0 + model.accrual_fractions * forwards_at_fix
        cum_accrual = jnp.cumprod(accrual)
        dfs = jnp.concatenate([
            initial_df[None],                    # DF(0, T_0)
            initial_df / cum_accrual,            # DF(0, T_1), ..., DF(0, T_N)
        ])

        return forwards_at_fix, dfs

    keys = jax.random.split(key, n_paths)
    forwards_at_fixing, discount_factors = jax.vmap(single_path)(keys)

    return LMMPathResult(
        forwards_at_fixing=forwards_at_fixing,
        discount_factors=discount_factors,
    )
