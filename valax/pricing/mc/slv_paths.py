"""Stochastic-Local Volatility Monte Carlo path generation.

Generates joint ``(S_t, V_t)`` paths under the Heston × leverage SDE:

.. math::

    dS_t / S_t &= (r - q) dt + L(k_t, t)\\sqrt{V_t}\\,dW_1 \\\\
    dV_t &= \\kappa(\\theta - V_t) dt + \\xi \\sqrt{V_t}\\,dW_2 \\\\
    \\langle dW_1, dW_2 \\rangle &= \\rho\\,dt,

with :math:`k_t = \\log(S_t / F(t))`, :math:`F(t) = S_0 \\exp((r-q)t)`,
and :math:`L` the calibrated ``LeverageGrid`` carried on the
:class:`SLVModel`.

Discretisation
--------------

* **Variance leg: Andersen QE.** Identical scheme to
  ``generate_heston_paths``, factored into ``_qe_variance_step_factory``.
  Exact in distribution at each ``dt`` regardless of Feller's condition.
  The log-spot update is intentionally *decoupled* from Andersen's
  central K-constants — those K-constants bake in a constant Heston
  diffusion, which is no longer valid once the leverage multiplies
  ``sqrt(V)``. We use the explicit log-Euler / Milstein update for the
  spot leg instead, with the ``Z_1`` component correlated to the
  variance via ``Z_v``.
* **Log-spot leg: ``scheme=`` switch.**

  * ``"midpoint_euler"`` (default): log-Euler with Itô correction,

    .. math::

       \\Delta\\log S_n = (\\mu - \\tfrac{1}{2}\\sigma_n^2)\\Delta t
                       + \\sigma_n \\sqrt{\\Delta t}\\,Z_{1,n},

    with :math:`\\sigma_n = L(k_n, t_n + \\tfrac{1}{2}\\Delta t)
    \\sqrt{V_n}`. Midpoint-in-time on ``L`` mirrors the LV-1
    convention.
  * ``"milstein"``: adds the strong-order correction in ``k`` only
    (the variance contribution to the strong-order term is absorbed
    by Andersen-QE on the variance leg),

    .. math::

       + \\tfrac{1}{2}\\sigma_n (\\partial_k L)_n \\sqrt{V_n}
         \\Delta t (Z_{1,n}^2 - 1).

* **Correlation.** Andersen-QE samples ``V_{n+1}`` from a
  non-Gaussian conditional, so the usual Cholesky decomposition of
  ``(Z_1, Z_2)`` doesn't apply directly. We follow the standard
  practice for QE-coupled spot legs: derive a residual Gaussian
  ``Z_⊥`` independent of the variance draw, and use

  .. math::

     Z_1 = \\rho\\,Z_V + \\sqrt{1 - \\rho^2}\\,Z_\\perp,

  where :math:`Z_V` is the standard Gaussian fed into the QE quadratic
  branch. This is exact on the quadratic branch in the ``L \\equiv 1``
  limit and is the standard approximation on the exponential branch,
  matching QuantLib's SLV engine.

Output
------

``(spot_paths, var_paths)`` each shaped ``(n_paths, n_steps + 1)``.
Column 0 is the initial condition broadcast across paths.

Autodiff
--------

Gradients flow through ``model.leverage.values`` (and through Heston
params and surface params, the latter via the calibration routine).
The Milstein scheme additionally takes ``jax.grad`` of ``L`` w.r.t.
``k`` per step.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.models.slv import SLVModel


SchemeName = Literal["midpoint_euler", "milstein"]


def _qe_variance_step_factory(
    kappa: Float[Array, ""],
    theta: Float[Array, ""],
    xi: Float[Array, ""],
    dt: Float[Array, ""],
):
    """Build a closure that advances ``V`` one Andersen-QE step.

    Returns ``step(v, z_v, u) -> v_next``, all rank-1 over ``n_paths``.
    The closure captures the step-invariant Andersen constants ``E``,
    ``(1-E)``, ``(1-E)/kappa`` so they are only computed once per call
    to ``generate_slv_paths``.

    ``z_v`` and ``u`` are pre-sampled by the caller so that the same
    ``z_v`` can be re-used to correlate the spot leg (see module
    docstring on correlation).
    """
    psi_c = 1.5
    tiny = 1e-300

    kdt = kappa * dt
    E = jnp.exp(-kdt)
    one_minus_E = -jnp.expm1(-kdt)
    kdt_safe = jnp.where(jnp.abs(kdt) > 1e-8, kdt, 1.0)
    taylor = 1.0 - 0.5 * kdt + kdt * kdt / 6.0
    one_minus_E_over_kappa = dt * jnp.where(
        jnp.abs(kdt) > 1e-8, one_minus_E / kdt_safe, taylor,
    )

    def step(v, z_v, u):
        m = theta + (v - theta) * E
        s2 = (
            v * xi * xi * E * one_minus_E_over_kappa
            + 0.5 * theta * xi * xi * one_minus_E * one_minus_E_over_kappa
        )
        m_safe = jnp.where(m > tiny, m, tiny)
        psi = s2 / (m_safe * m_safe)

        # Quadratic branch.
        inv_psi = 1.0 / jnp.where(psi > tiny, psi, 1.0)
        two_inv = 2.0 * inv_psi
        rad = jnp.maximum(two_inv * (two_inv - 1.0), 0.0)
        b2 = jnp.maximum(two_inv - 1.0 + jnp.sqrt(rad), 0.0)
        a = m_safe / (1.0 + b2)
        v_quad = a * (jnp.sqrt(b2) + z_v) ** 2

        # Exponential branch.
        p = jnp.clip((psi - 1.0) / (psi + 1.0), min=0.0, max=1.0 - 1e-12)
        beta = (1.0 - p) / m_safe
        denom = jnp.maximum(1.0 - u, tiny)
        log_arg = jnp.maximum((1.0 - p) / denom, 1.0)
        v_exp = jnp.where(
            u <= p, 0.0, jnp.log(log_arg) / jnp.maximum(beta, tiny),
        )

        v_next = jnp.where(psi <= psi_c, v_quad, v_exp)
        v_next = jnp.maximum(v_next, 0.0)
        return v_next

    return step


def generate_slv_paths(
    model: SLVModel,
    spot: Float[Array, ""],
    T: Float[Array, ""] | float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
    *,
    scheme: SchemeName = "midpoint_euler",
) -> tuple[
    Float[Array, "n_paths n_steps_plus1"],
    Float[Array, "n_paths n_steps_plus1"],
]:
    """Generate joint ``(S_t, V_t)`` paths under SLV.

    Args:
        model: A calibrated :class:`SLVModel`.
        spot: Initial spot ``S_0``.
        T: Terminal time (year fraction).
        n_steps: Number of time steps (``dt = T / n_steps``).
        n_paths: Number of MC paths.
        key: Top-level PRNG key.
        scheme: Log-spot scheme — ``"midpoint_euler"`` (default) or
            ``"milstein"``. See module docstring for the trade-off.

    Returns:
        ``(spot_paths, var_paths)`` of shape ``(n_paths, n_steps + 1)``
        each. Column 0 is the initial state broadcast across paths.

    Raises:
        ValueError: if ``scheme`` is not one of the accepted literals.
    """
    if scheme not in ("midpoint_euler", "milstein"):
        raise ValueError(
            f"generate_slv_paths: scheme must be 'midpoint_euler' or "
            f"'milstein', got {scheme!r}"
        )

    T = jnp.asarray(T)
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    mu = model.rate - model.dividend
    rho = model.rho
    sqrt_1_minus_rho2 = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))

    log_spot = jnp.log(spot)
    leverage = model.leverage

    qe_step = _qe_variance_step_factory(
        model.kappa, model.theta, model.xi, dt,
    )

    def _k_of(log_S, t):
        return log_S - (log_spot + mu * t)

    if scheme == "midpoint_euler":
        def _L_at_paths(log_S, t_mid):
            k = _k_of(log_S, t_mid)
            return jax.vmap(lambda kk: leverage(kk, t_mid))(k)

        def step(carry, scan_input):
            log_S, v = carry
            t_n, t_mid, key_t = scan_input
            k_v, k_perp, k_u = jax.random.split(key_t, 3)
            z_v = jax.random.normal(k_v, shape=(n_paths,))
            z_perp = jax.random.normal(k_perp, shape=(n_paths,))
            u = jax.random.uniform(k_u, shape=(n_paths,))

            z_s = rho * z_v + sqrt_1_minus_rho2 * z_perp

            v_next = qe_step(v, z_v, u)

            L_n = _L_at_paths(log_S, t_mid)
            sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
            sigma = L_n * sqrt_v

            log_S_next = (
                log_S
                + (mu - 0.5 * sigma * sigma) * dt
                + sigma * sqrt_dt * z_s
            )
            return (log_S_next, v_next), (log_S_next, v_next)

    else:  # scheme == "milstein"
        _L_and_grad_scalar = jax.value_and_grad(
            lambda kk, tt: leverage(kk, tt), argnums=0,
        )

        def _L_and_grad_at_paths(log_S, t_mid):
            k = _k_of(log_S, t_mid)
            return jax.vmap(lambda kk: _L_and_grad_scalar(kk, t_mid))(k)

        def step(carry, scan_input):
            log_S, v = carry
            t_n, t_mid, key_t = scan_input
            k_v, k_perp, k_u = jax.random.split(key_t, 3)
            z_v = jax.random.normal(k_v, shape=(n_paths,))
            z_perp = jax.random.normal(k_perp, shape=(n_paths,))
            u = jax.random.uniform(k_u, shape=(n_paths,))
            z_s = rho * z_v + sqrt_1_minus_rho2 * z_perp

            v_next = qe_step(v, z_v, u)

            L_n, dL_dk = _L_and_grad_at_paths(log_S, t_mid)
            sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
            sigma = L_n * sqrt_v
            # ∂σ/∂k = (∂L/∂k) · √V (V is the state, not a function of k).
            dsigma_dk = dL_dk * sqrt_v

            log_S_next = (
                log_S
                + (mu - 0.5 * sigma * sigma) * dt
                + sigma * sqrt_dt * z_s
                + 0.5 * sigma * dsigma_dk * dt * (z_s * z_s - 1.0)
            )
            return (log_S_next, v_next), (log_S_next, v_next)

    log_S0 = jnp.broadcast_to(log_spot, (n_paths,))
    v0 = jnp.broadcast_to(model.v0, (n_paths,))

    t_n = jnp.arange(n_steps).astype(T.dtype) * dt
    t_mid = t_n + 0.5 * dt
    keys = jax.random.split(key, n_steps)

    _, (log_s_seq, v_seq) = jax.lax.scan(
        step, (log_S0, v0), (t_n, t_mid, keys),
    )
    log_s = jnp.concatenate([log_S0[None, :], log_s_seq], axis=0)
    v = jnp.concatenate([v0[None, :], v_seq], axis=0)
    return jnp.exp(log_s).T, v.T
