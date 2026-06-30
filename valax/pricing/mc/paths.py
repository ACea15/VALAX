"""SDE path generation.

GBM paths are generated via diffrax (Euler-Maruyama is unbiased for GBM).

Heston paths are generated via Andersen's (2008) Quadratic-Exponential
(QE) scheme implemented with ``jax.lax.scan``. The QE scheme samples
the variance process by exact two-moment matching against either a
shifted-squared-normal (quadratic branch, low variance-of-variance)
or a Bernoulli–exponential mixture (exponential branch, high
variance-of-variance). It has none of the ``O(1/sqrt(n_steps))`` bias
that the naive Euler-with-reflection scheme exhibits when the variance
process spends time near the absorbing boundary — i.e. when Feller's
condition ``2·kappa·theta > xi²`` is violated, which is the common
case for single-expiry SABR-style calibrations of Heston.

References:
    Andersen, L. (2008). "Simple and Efficient Simulation of the Heston
    Stochastic Volatility Model." Journal of Computational Finance,
    11(3), pp. 1–42.
"""

import jax
import jax.numpy as jnp
import diffrax
from jaxtyping import Float
from jax import Array

from valax.models.black_scholes import BlackScholesModel, GBMDrift, GBMDiffusion
from valax.models.heston import HestonModel


def generate_gbm_paths(
    model: BlackScholesModel,
    spot: Float[Array, ""],
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> Float[Array, "n_paths n_steps+1"]:
    """Generate GBM paths using diffrax Euler-Maruyama.

    Returns array of shape (n_paths, n_steps+1) including initial spot.
    """
    mu = model.rate - model.dividend
    drift = GBMDrift(mu=mu)
    diffusion = GBMDiffusion(sigma=model.vol)
    dt = T / n_steps
    ts = jnp.linspace(0.0, T, n_steps + 1)

    def single_path(subkey):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=T, tol=dt / 2, shape=(), key=subkey
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, bm),
        )
        sol = diffrax.diffeqsolve(
            terms,
            diffrax.Euler(),
            t0=0.0,
            t1=T,
            dt0=dt,
            y0=spot,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # shape (n_steps+1,)

    keys = jax.random.split(key, n_paths)
    return jax.vmap(single_path)(keys)


def generate_heston_paths(
    model: HestonModel,
    spot: Float[Array, ""],
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> tuple[Float[Array, "n_paths n_steps+1"], Float[Array, "n_paths n_steps+1"]]:
    """Generate Heston paths via the Andersen (2008) QE scheme.

    The variance process is sampled by Andersen's quadratic / exponential
    switch at ``psi_c = 1.5``; the log-spot is propagated by the matching
    "central" discretisation with trapezoidal weights
    ``gamma1 = gamma2 = 0.5``. The scheme is bias-free in distribution at
    each ``dt`` step for the variance — there is no penalty when Feller's
    condition is violated, which is the regime the previous Euler-with-
    reflection scheme had ``O(1/sqrt(n_steps))`` bias in.

    Args:
        model: Heston model carrying ``v0, kappa, theta, xi, rho, rate,
            dividend``.
        spot: Initial spot ``S_0``.
        T: Horizon in years (the same time unit as ``model.rate``).
        n_steps: Number of discretisation steps. The QE scheme is exact
            in distribution for the variance at each step, but the
            log-spot update is only conditionally Gaussian, so a
            moderate ``n_steps`` (e.g. 50–100 per year) is still
            recommended.
        n_paths: Number of independent paths.
        key: PRNGKey driving all three Gaussian / uniform draws per
            step.

    Returns:
        Tuple ``(spot_paths, var_paths)``, each of shape
        ``(n_paths, n_steps+1)`` and including the initial state at
        column 0.

    References:
        Andersen, L. (2008). "Simple and Efficient Simulation of the
        Heston Stochastic Volatility Model." Journal of Computational
        Finance, 11(3), pp. 1–42.
    """
    # ── Tunable scheme constants (set to Andersen's recommended defaults).
    psi_c = 1.5
    gamma1 = 0.5
    gamma2 = 0.5
    # Numerical guards. All are below any meaningful physical value yet
    # safely above subnormal float64.
    tiny = 1e-300

    dt = jnp.asarray(T / n_steps)
    kappa = model.kappa
    theta = model.theta
    xi = model.xi
    rho = model.rho
    mu = model.rate - model.dividend

    # ── Step-invariant quantities (computed once per call).
    kdt = kappa * dt
    E = jnp.exp(-kdt)
    one_minus_E = -jnp.expm1(-kdt)          # = 1 - E, accurate for small kdt
    # (1 - E) / kappa, numerically stable as kappa → 0.
    # We compute it as dt · ((1 - exp(-kdt)) / kdt), where the inner
    # ratio is replaced by its 3-term Taylor series when kdt is tiny.
    kdt_safe = jnp.where(jnp.abs(kdt) > 1e-8, kdt, 1.0)
    taylor = 1.0 - 0.5 * kdt + kdt * kdt / 6.0
    one_minus_E_over_kappa = dt * jnp.where(
        jnp.abs(kdt) > 1e-8,
        one_minus_E / kdt_safe,
        taylor,
    )

    # The K-constants of Andersen's central discretisation. ``xi`` must
    # be strictly positive; we keep a tiny guard so a degenerate
    # ``xi = 0`` collapses to a constant-vol BSM-like log-Euler rather
    # than producing NaNs.
    xi_safe = jnp.where(jnp.abs(xi) > tiny, xi, tiny)
    K0 = -rho * kappa * theta / xi_safe * dt
    K1 = gamma1 * dt * (kappa * rho / xi_safe - 0.5) - rho / xi_safe
    K2 = gamma2 * dt * (kappa * rho / xi_safe - 0.5) + rho / xi_safe
    K3 = gamma1 * dt * (1.0 - rho * rho)
    K4 = gamma2 * dt * (1.0 - rho * rho)
    drift_dt = mu * dt

    def step(carry, key_t):
        log_s, v = carry  # each shape (n_paths,)
        k_v, k_s, k_u = jax.random.split(key_t, 3)
        z_v = jax.random.normal(k_v, shape=(n_paths,))
        z_s = jax.random.normal(k_s, shape=(n_paths,))
        u = jax.random.uniform(k_u, shape=(n_paths,))

        # ── Conditional moments of V(t+dt) given V(t) = v.
        m = theta + (v - theta) * E
        s2 = (
            v * xi * xi * E * one_minus_E_over_kappa
            + 0.5 * theta * xi * xi * one_minus_E * one_minus_E_over_kappa
        )
        m_safe = jnp.where(m > tiny, m, tiny)
        psi = s2 / (m_safe * m_safe)

        # ── Quadratic branch (psi ≤ psi_c).
        # V_next = a · (b + Z_V)², with b² = 2/ψ − 1 + √((2/ψ)(2/ψ − 1)),
        # a = m / (1 + b²). Both quantities are non-negative for ψ ≤ 2.
        inv_psi = 1.0 / jnp.where(psi > tiny, psi, 1.0)
        two_inv = 2.0 * inv_psi
        rad = jnp.maximum(two_inv * (two_inv - 1.0), 0.0)
        b2 = jnp.maximum(two_inv - 1.0 + jnp.sqrt(rad), 0.0)
        a = m_safe / (1.0 + b2)
        v_quad = a * (jnp.sqrt(b2) + z_v) ** 2

        # ── Exponential branch (psi > psi_c, equivalently psi > 1 is
        # the well-defined region; psi_c is the empirical switch point
        # for efficiency).  V_next is a Bernoulli–exponential mixture
        # with atom p at 0 and density β·exp(−β·v) for v > 0.
        p = jnp.clip((psi - 1.0) / (psi + 1.0), min=0.0, max=1.0 - 1e-12)
        beta = (1.0 - p) / m_safe
        # Inverse CDF using the *complement* of U (numerically stable):
        # F⁻¹(u) = 0 if u ≤ p else log((1−p)/(1−u))/β.
        denom = jnp.maximum(1.0 - u, tiny)
        log_arg = jnp.maximum((1.0 - p) / denom, 1.0)   # ≥ 1 ⇒ log ≥ 0
        v_exp = jnp.where(
            u <= p,
            0.0,
            jnp.log(log_arg) / jnp.maximum(beta, tiny),
        )

        v_next = jnp.where(psi <= psi_c, v_quad, v_exp)
        v_next = jnp.maximum(v_next, 0.0)               # belt-and-braces

        # ── Log-spot update (Andersen central discretisation).
        var_term = jnp.maximum(K3 * v + K4 * v_next, 0.0)
        log_s_next = (
            log_s
            + drift_dt
            + K0
            + K1 * v
            + K2 * v_next
            + jnp.sqrt(var_term) * z_s
        )
        return (log_s_next, v_next), (log_s_next, v_next)

    log_s0 = jnp.broadcast_to(jnp.log(spot), (n_paths,))
    v0 = jnp.broadcast_to(model.v0, (n_paths,))

    keys = jax.random.split(key, n_steps)
    _, (log_s_seq, v_seq) = jax.lax.scan(step, (log_s0, v0), keys)
    # log_s_seq / v_seq have shape (n_steps, n_paths); prepend the
    # initial state, then transpose to (n_paths, n_steps+1).
    log_s = jnp.concatenate([log_s0[None, :], log_s_seq], axis=0)
    v = jnp.concatenate([v0[None, :], v_seq], axis=0)
    return jnp.exp(log_s).T, v.T
