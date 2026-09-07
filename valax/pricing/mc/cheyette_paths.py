r"""Cheyette (quasi-Gaussian) short-rate Monte Carlo path generators.

Unlike Hull-White and G2++, the Cheyette state is **not** Gaussian once the
local volatility :math:`\sigma_r(t, x)` depends on the state, so the exact
conditional (Cholesky-of-covariance) sampling used by those models does not
transfer.  Instead each factor is advanced with an **integrating-factor**
(exponential-Euler) step that treats the linear mean-reversion drift exactly
and freezes the state-dependent local volatility over the step:

.. math::

    x(t+\Delta t) = x(t)\,e^{-\kappa\Delta t}
        + \frac{y(t)}{\kappa}\bigl(1 - e^{-\kappa\Delta t}\bigr)
        + \sigma_r(t, x(t))\,\sqrt{\tfrac{1 - e^{-2\kappa\Delta t}}{2\kappa}}\;z,

and the auxiliary accumulator with the exact solution of its linear ODE for a
frozen forcing:

.. math::

    y(t+\Delta t) = y(t)\,e^{-2\kappa\Delta t}
        + \frac{\sigma_r(t, x(t))^2}{2\kappa}\bigl(1 - e^{-2\kappa\Delta t}\bigr).

In the constant-volatility (:math:`\beta = 0`) limit the accumulator step is
**exact** and the factor step is exact given the frozen (deterministic) drift,
so the scheme collapses onto the exact Hull-White / G2++ simulation up to the
smooth :math:`O(\Delta t^2)` drift-freezing error.

The money-market log-discount factor accumulates :math:`-\int_0^{t} r\,ds` with
the **deterministic forward part integrated exactly** — it telescopes to the
market discount-factor ratio :math:`\ln P^M(0, t_i)/P^M(0, t_{i+1})`, so a
curved initial forward introduces no repricing bias (the same discipline as
:mod:`valax.pricing.mc.g2pp_paths`).  Because the Cheyette short rate is simply
:math:`r = f^M(0, t) + \sum_i x_i` with the convexity carried dynamically by the
:math:`y`-drift, only the smooth stochastic :math:`\sum_i x_i` part uses the
trapezoidal rule; there is no separate convexity shift to integrate.

References:
    Andersen, L. & Piterbarg, V. (2010), *Interest Rate Modeling*, ch. 13.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.cheyette import (
    CheyetteModel,
    CheyetteModel2F,
    cheyette_local_vol,
    cheyette_local_vol_2f,
    cheyette_instantaneous_forward,
    cheyette_market_df,
)


class CheyettePathResult(eqx.Module):
    r"""Output of :func:`generate_cheyette_paths` (one factor).

    Attributes:
        factor_x: Simulated state factor :math:`x(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        aux_y: Simulated auxiliary accumulator :math:`y(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        short_rates: Simulated short rates
            :math:`r(t_i) = f^M(0, t_i) + x(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        log_discount_factors: Cumulative log money-market discount factor
            :math:`-\int_0^{t_i} r(s)\,ds` per path and grid point, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
    """

    factor_x: Float[Array, "n_paths n_steps_plus1"]
    aux_y: Float[Array, "n_paths n_steps_plus1"]
    short_rates: Float[Array, "n_paths n_steps_plus1"]
    log_discount_factors: Float[Array, "n_paths n_steps_plus1"]


class Cheyette2FPathResult(eqx.Module):
    r"""Output of :func:`generate_cheyette_paths_2f` (two factors).

    Attributes:
        factor_x1: Simulated first factor :math:`x_1(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        factor_x2: Simulated second factor :math:`x_2(t_i)`, shape
            ``(n_paths, n_steps + 1)``.  Column 0 is identically zero.
        aux_y11: Auxiliary accumulator :math:`y_{11}(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        aux_y22: Auxiliary accumulator :math:`y_{22}(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        aux_y12: Auxiliary cross accumulator :math:`y_{12}(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        short_rates: Simulated short rates
            :math:`r(t_i) = f^M(0, t_i) + x_1(t_i) + x_2(t_i)`, shape
            ``(n_paths, n_steps + 1)``.
        log_discount_factors: Cumulative log money-market discount factor
            :math:`-\int_0^{t_i} r(s)\,ds`, shape ``(n_paths, n_steps + 1)``.
    """

    factor_x1: Float[Array, "n_paths n_steps_plus1"]
    factor_x2: Float[Array, "n_paths n_steps_plus1"]
    aux_y11: Float[Array, "n_paths n_steps_plus1"]
    aux_y22: Float[Array, "n_paths n_steps_plus1"]
    aux_y12: Float[Array, "n_paths n_steps_plus1"]
    short_rates: Float[Array, "n_paths n_steps_plus1"]
    log_discount_factors: Float[Array, "n_paths n_steps_plus1"]


def generate_cheyette_paths(
    model: CheyetteModel,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> CheyettePathResult:
    r"""Simulate one-factor Cheyette paths via an integrating-factor scheme.

    Each step advances the state factor with the exact linear-drift solution and
    a frozen local-volatility diffusion, and the auxiliary accumulator with the
    exact solution of its linear ODE.  The money-market log-discount factor is
    accumulated with the deterministic forward part integrated exactly (a
    telescoping market discount-factor ratio) and the stochastic remainder
    trapezoided, so a curved initial forward introduces no repricing bias and
    the martingale identity :math:`\mathbb{E}[D(0, T)] = P^M(0, T)` holds within
    Monte-Carlo error.

    Args:
        model: One-factor Cheyette model carrying the initial curve and
            parameters :math:`\kappa` (mean reversion), :math:`\sigma`
            (volatility level), and :math:`\beta` (skew).
        T: Horizon in year fractions.
        n_steps: Number of equally-spaced time steps.
        n_paths: Number of independent Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        :class:`CheyettePathResult` with per-path factor, accumulator,
        short-rate and log-discount-factor trajectories of shape
        ``(n_paths, n_steps + 1)``.
    """
    kappa = model.mean_reversion
    dt = jnp.asarray(T / n_steps, dtype=jnp.float64)
    times = jnp.linspace(0.0, T, n_steps + 1)

    e_kdt = jnp.exp(-kappa * dt)
    e_2kdt = jnp.exp(-2.0 * kappa * dt)
    # Exact standard deviation of the diffusion integral for unit local vol.
    diff_sd = jnp.sqrt((1.0 - e_2kdt) / (2.0 * kappa))

    # Exact forward-part discounting: telescopes to the market-DF ratio.
    ln_pm_grid = jnp.log(cheyette_market_df(model, times))               # (n+1,)
    forward_grid = jax.vmap(
        lambda s: cheyette_instantaneous_forward(model, s)
    )(times)                                                             # (n+1,)

    normals = jax.random.normal(key, shape=(n_steps, n_paths), dtype=jnp.float64)

    def step_fn(carry, xs):
        x_i, y_i, log_df_i = carry
        z, i = xs

        sigma_r = cheyette_local_vol(model, x_i)                        # (n_paths,)
        x_next = (
            x_i * e_kdt
            + (y_i / kappa) * (1.0 - e_kdt)
            + sigma_r * diff_sd * z
        )
        y_next = y_i * e_2kdt + (sigma_r**2 / (2.0 * kappa)) * (1.0 - e_2kdt)

        # -∫ r ds over the step: exact forward part + trapezoidal x remainder.
        det_forward = ln_pm_grid[i + 1] - ln_pm_grid[i]
        stochastic = 0.5 * dt * (x_i + x_next)
        log_df_next = log_df_i + det_forward - stochastic

        r_next = forward_grid[i + 1] + x_next
        return (x_next, y_next, log_df_next), (x_next, y_next, r_next, log_df_next)

    zeros = jnp.zeros(n_paths, dtype=jnp.float64)
    r0 = zeros + forward_grid[0]
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    _, (x_seq, y_seq, r_seq, ldf_seq) = jax.lax.scan(
        step_fn, (zeros, zeros, zeros), (normals, step_indices),
    )

    def _prepend(col0, seq):
        return jnp.concatenate([col0[None, :], seq], axis=0).T

    return CheyettePathResult(
        factor_x=_prepend(zeros, x_seq),
        aux_y=_prepend(zeros, y_seq),
        short_rates=_prepend(r0, r_seq),
        log_discount_factors=_prepend(zeros, ldf_seq),
    )


def generate_cheyette_paths_2f(
    model: CheyetteModel2F,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> Cheyette2FPathResult:
    r"""Simulate two-factor Cheyette paths via an integrating-factor scheme.

    Both factors are advanced with the exact linear-drift solution and frozen
    per-factor local-volatility diffusions driven by **correlated** shocks
    (correlation :math:`\rho`, obtained by a :math:`2\times2` Cholesky applied to
    independent normals).  The symmetric auxiliary accumulator :math:`y_{ij}` is
    advanced with the exact solution of its linear ODE.  Discounting uses the
    exact forward part plus a trapezoidal :math:`x_1 + x_2` remainder, so the
    martingale identity holds within Monte-Carlo error.

    Note:
        The frozen-vol diffusion cross-covariance
        :math:`\sigma_{r,1}\sigma_{r,2}\rho\,\mathrm{sd}_1\mathrm{sd}_2` matches
        the exact :math:`\sigma_{r,1}\sigma_{r,2}\rho\,(1-e^{-(\kappa_1+\kappa_2)
        \Delta t})/(\kappa_1+\kappa_2)` to :math:`O(\Delta t^2)`, consistent with
        the scheme's overall order.

    Args:
        model: Two-factor Cheyette model carrying the initial curve and
            per-factor parameters plus the correlation :math:`\rho`.
        T: Horizon in year fractions.
        n_steps: Number of equally-spaced time steps.
        n_paths: Number of independent Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        :class:`Cheyette2FPathResult` with per-path factor, accumulator,
        short-rate and log-discount-factor trajectories of shape
        ``(n_paths, n_steps + 1)``.
    """
    k1 = model.mean_reversion_1
    k2 = model.mean_reversion_2
    rho = model.correlation

    dt = jnp.asarray(T / n_steps, dtype=jnp.float64)
    times = jnp.linspace(0.0, T, n_steps + 1)

    e_k1dt = jnp.exp(-k1 * dt)
    e_k2dt = jnp.exp(-k2 * dt)
    e_2k1dt = jnp.exp(-2.0 * k1 * dt)
    e_2k2dt = jnp.exp(-2.0 * k2 * dt)
    e_k12dt = jnp.exp(-(k1 + k2) * dt)
    diff_sd1 = jnp.sqrt((1.0 - e_2k1dt) / (2.0 * k1))
    diff_sd2 = jnp.sqrt((1.0 - e_2k2dt) / (2.0 * k2))

    # Cholesky of the [[1, rho], [rho, 1]] shock correlation.
    corr = jnp.array([[1.0, rho], [rho, 1.0]])
    chol = jnp.linalg.cholesky(corr)

    ln_pm_grid = jnp.log(cheyette_market_df(model, times))
    forward_grid = jax.vmap(
        lambda s: cheyette_instantaneous_forward(model, s)
    )(times)

    normals = jax.random.normal(
        key, shape=(n_steps, n_paths, 2), dtype=jnp.float64
    )

    def step_fn(carry, xs):
        x1_i, x2_i, y11_i, y22_i, y12_i, log_df_i = carry
        z, i = xs  # z: (n_paths, 2)

        zc = z @ chol.T  # correlated normals, (n_paths, 2)
        z1 = zc[:, 0]
        z2 = zc[:, 1]

        sig1, sig2 = cheyette_local_vol_2f(model, x1_i, x2_i)

        drift1 = (y11_i + y12_i) / k1
        drift2 = (y12_i + y22_i) / k2
        x1_next = x1_i * e_k1dt + drift1 * (1.0 - e_k1dt) + sig1 * diff_sd1 * z1
        x2_next = x2_i * e_k2dt + drift2 * (1.0 - e_k2dt) + sig2 * diff_sd2 * z2

        y11_next = y11_i * e_2k1dt + (sig1**2 / (2.0 * k1)) * (1.0 - e_2k1dt)
        y22_next = y22_i * e_2k2dt + (sig2**2 / (2.0 * k2)) * (1.0 - e_2k2dt)
        y12_next = y12_i * e_k12dt + (
            rho * sig1 * sig2 / (k1 + k2)
        ) * (1.0 - e_k12dt)

        det_forward = ln_pm_grid[i + 1] - ln_pm_grid[i]
        stochastic = 0.5 * dt * ((x1_i + x2_i) + (x1_next + x2_next))
        log_df_next = log_df_i + det_forward - stochastic

        r_next = forward_grid[i + 1] + x1_next + x2_next
        carry_next = (x1_next, x2_next, y11_next, y22_next, y12_next, log_df_next)
        out = (x1_next, x2_next, y11_next, y22_next, y12_next, r_next, log_df_next)
        return carry_next, out

    zeros = jnp.zeros(n_paths, dtype=jnp.float64)
    r0 = zeros + forward_grid[0]
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    init = (zeros, zeros, zeros, zeros, zeros, zeros)
    _, (x1_seq, x2_seq, y11_seq, y22_seq, y12_seq, r_seq, ldf_seq) = jax.lax.scan(
        step_fn, init, (normals, step_indices),
    )

    def _prepend(col0, seq):
        return jnp.concatenate([col0[None, :], seq], axis=0).T

    return Cheyette2FPathResult(
        factor_x1=_prepend(zeros, x1_seq),
        factor_x2=_prepend(zeros, x2_seq),
        aux_y11=_prepend(zeros, y11_seq),
        aux_y22=_prepend(zeros, y22_seq),
        aux_y12=_prepend(zeros, y12_seq),
        short_rates=_prepend(r0, r_seq),
        log_discount_factors=_prepend(zeros, ldf_seq),
    )
