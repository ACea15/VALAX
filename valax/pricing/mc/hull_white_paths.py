"""Hull-White one-factor short-rate Monte Carlo path generator.

Uses the **exact** conditional distribution of the Ornstein-Uhlenbeck
short-rate process at each time step, eliminating the
:math:`O(\\sqrt{\\Delta t})` strong-order discretisation bias of the naive
Euler-Maruyama scheme.  See :doc:`/theory/hull-white-mc` for the full
mathematical derivation.

The key idea is that the short rate :math:`r(t)` driven by the H-W SDE
is Gaussian at every future time, with analytically known conditional
mean and variance:

.. math::

    r(t + \\Delta t) \\mid r(t) \\sim
    \\mathcal{N}\\!\\left(
        \\mu(t, r(t), \\Delta t),\\;
        v(\\Delta t)
    \\right)

Drawing directly from this Gaussian distribution — rather than
approximating the SDE with a finite difference — makes the short-rate
marginal distribution exact at every grid point regardless of step size.
The money-market numeraire accumulates the path integral of :math:`r`
analytically within each interval.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, §3.3.
    Hull & White (1994), "Numerical Procedures for Implementing Term
        Structure Models: Single-Factor Models".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.hull_white import HullWhiteModel, hw_alpha


# ─────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────


class HullWhitePathResult(eqx.Module):
    """Output of :func:`generate_hull_white_paths`.

    Attributes:
        short_rates: Simulated short rates at each time grid point,
            shape ``(n_paths, n_steps + 1)``.  Column 0 is the initial
            short rate :math:`r(0) = f^M(0, 0)`.
        log_discount_factors: Cumulative log money-market discount
            factor :math:`-\\int_0^{t_i} r(s)\\,ds` for each path and
            grid point, shape ``(n_paths, n_steps + 1)``.  Column 0 is
            identically zero (no time has elapsed).  The stochastic
            discount factor to time :math:`t_i` on path :math:`p` is
            ``jnp.exp(log_discount_factors[p, i])``.
    """

    short_rates: Float[Array, "n_paths n_steps_plus1"]
    log_discount_factors: Float[Array, "n_paths n_steps_plus1"]


# ─────────────────────────────────────────────────────────────────────
# Path generator
# ─────────────────────────────────────────────────────────────────────


def generate_hull_white_paths(
    model: HullWhiteModel,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> HullWhitePathResult:
    """Simulate Hull-White short-rate paths via exact conditional sampling.

    At each step the short rate is drawn from its *exact* conditional
    Gaussian distribution rather than being approximated by an
    Euler-Maruyama increment.  The conditional mean and variance are
    available in closed form for the Ornstein-Uhlenbeck dynamics of
    the H-W model (see :doc:`/theory/hull-white-mc`).

    The money-market log-discount factor is accumulated analytically
    within each interval using the trapezoidal approximation to
    :math:`\\int_{t_i}^{t_{i+1}} r(s)\\,ds`:

    .. math::

        \\int_{t_i}^{t_{i+1}} r(s)\\,ds \\approx
        \\frac{\\Delta t}{2}\\bigl(r(t_i) + r(t_{i+1})\\bigr)

    This is consistent with the exact mean trajectory of the OU bridge
    and is unbiased to :math:`O(\\Delta t^2)`.

    Args:
        model: Hull-White model carrying the initial curve and
            parameters :math:`a` (mean reversion) and
            :math:`\\sigma` (short-rate volatility).
        T: Horizon in year fractions.
        n_steps: Number of equally-spaced time steps.  Each step has
            width :math:`\\Delta t = T / n_{\\text{steps}}`.
        n_paths: Number of independent Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        :class:`HullWhitePathResult` with fields ``short_rates`` and
        ``log_discount_factors``, each of shape
        ``(n_paths, n_steps + 1)``.
    """
    a = model.mean_reversion
    sigma = model.volatility

    dt = jnp.asarray(T / n_steps, dtype=jnp.float64)
    times = jnp.linspace(0.0, T, n_steps + 1)  # (n_steps + 1,)

    # ── Conditional moments for r(t + dt) | r(t) ─────────────────────
    # Write r(t) = x(t) + alpha(t), where x satisfies the centred OU
    #
    #   dx = -a x dt + sigma dW
    #
    # and alpha is the time-dependent exact-fit shift (Brigo-Mercurio §3.3):
    #
    #   alpha(t) = f^M(0, t) + sigma^2/(2a^2) * (1 - exp(-a*t))^2
    #
    # The x-process has the exact conditional distribution
    #
    #   x(t+dt) | x(t)  ~  N( x(t)*exp(-a*dt),  sigma^2/(2a)*(1-exp(-2a*dt)) )
    #
    # Translating back to r:
    #
    #   r(t+dt) | r(t)  ~  N( alpha(t+dt) + (r(t)-alpha(t))*exp(-a*dt),  v_dt )
    #
    # with v_dt = sigma^2/(2a)*(1-exp(-2a*dt)).
    #
    # This is the unique unbiased exact-sampling formula for any dt.

    e_adt = jnp.exp(-a * dt)                                               # scalar
    v_dt = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * dt))       # conditional variance

    # Precompute the exact-fit shift on the full time grid (n_steps+1,).
    # alpha(t) = f^M(0,t) + sigma^2/(2a^2) * (1-exp(-a*t))^2
    # (Brigo-Mercurio (2006), §3.3, eq. 3.30 — note denominator 2a², not 2a.)
    alpha_grid = hw_alpha(model, times)                               # alpha(t_i)

    # Initial short rate: r(0) = alpha(0) = f^M(0,0)  [since (1-e^0)^2 = 0].
    r0 = alpha_grid[0]  # scalar

    # ── Generate standard normals for all paths and all steps ─────────
    # Shape: (n_steps, n_paths)
    normals = jax.random.normal(key, shape=(n_steps, n_paths), dtype=jnp.float64)

    # ── Scan over time steps ──────────────────────────────────────────
    # carry = (r_i [n_paths], log_df_i [n_paths])

    def step_fn(carry, xs):
        r_i, log_df_i = carry
        z, i = xs

        alpha_i    = alpha_grid[i]       # alpha(t_i)
        alpha_next = alpha_grid[i + 1]   # alpha(t_{i+1})

        # Exact conditional mean: alpha(t+dt) + (r_i - alpha(t)) * e^{-a dt}
        mu = alpha_next + (r_i - alpha_i) * e_adt

        # Draw r_{i+1} from N(mu, v_dt).
        r_next = mu + jnp.sqrt(v_dt) * z  # (n_paths,)

        # Accumulate log-discount factor: trapezoidal rule.
        log_df_next = log_df_i - 0.5 * dt * (r_i + r_next)

        return (r_next, log_df_next), (r_next, log_df_next)

    r0_paths = jnp.broadcast_to(r0, (n_paths,))          # (n_paths,)
    log_df0 = jnp.zeros(n_paths, dtype=jnp.float64)      # (n_paths,)

    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    _, (r_seq, ldf_seq) = jax.lax.scan(
        step_fn, (r0_paths, log_df0), (normals, step_indices),
    )
    # r_seq, ldf_seq: (n_steps, n_paths); prepend initial state.
    r_all = jnp.concatenate([r0_paths[None, :], r_seq], axis=0).T     # (n_paths, n_steps+1)
    ldf_all = jnp.concatenate([log_df0[None, :], ldf_seq], axis=0).T  # (n_paths, n_steps+1)

    return HullWhitePathResult(
        short_rates=r_all,
        log_discount_factors=ldf_all,
    )
