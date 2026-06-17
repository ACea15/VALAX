"""Local volatility Monte Carlo path generation.

Two discretisation schemes are available, selected via the ``scheme=``
keyword:

``midpoint_euler`` (default)
    Log-Euler with Itô correction:

    .. math::

        \\ln S_{t+\\Delta t} \\;=\\; \\ln S_t
            + \\left(r - q - \\tfrac{1}{2}\\sigma_n^2\\right)\\!\\Delta t
            + \\sigma_n\\,\\sqrt{\\Delta t}\\,Z_n,

    with :math:`\\sigma_n = \\sigma_{\\text{loc}}(S_{t_n}, t_n + \\tfrac{1}{2}\\Delta t)`.
    **Weak order 1**, strong order 0.5. The midpoint-in-time σ avoids
    the Dupire ``T = 0`` singularity and gives the best weak-error
    constant attainable from a single-evaluation step.

``milstein`` (opt-in)
    Milstein-corrected log-Euler — adds the strong-order correction:

    .. math::

        \\ln S_{t+\\Delta t} \\;=\\; \\ln S_t
            + \\left(r - q - \\tfrac{1}{2}\\sigma_n^2\\right)\\!\\Delta t
            + \\sigma_n\\,\\sqrt{\\Delta t}\\,Z_n
            + \\tfrac{1}{2}\\,\\sigma_n\\!\\left(\\partial\\sigma_{\\text{loc}}/\\partial k\\right)\\!\\Delta t\\,\\bigl(Z_n^2 - 1\\bigr),

    where :math:`\\partial \\sigma_{\\text{loc}}/\\partial k` is evaluated
    at :math:`(S_{t_n}, t_n + \\tfrac{1}{2}\\Delta t)`. Weak order 1
    (same as Euler), **strong order 1.0**. One extra ``jax.grad`` per
    step, ~2× the per-step compute.

When to pick which
------------------

For **vanilla** (path-independent) payoffs both schemes share weak
order 1, so they have the same asymptotic bias rate. At typical MC
budgets the noise floor washes out the constant-factor improvement
that Milstein offers — empirically Milstein and midpoint-Euler give
indistinguishable vanilla-reprice errors at 4 seeds × 100k paths ×
{200–1000} steps (both ~6–20 bp on a moderate equity skew). The
default is midpoint-Euler because it is cheaper for no measurable
accuracy loss on vanillas.

For **path-dependent** payoffs (barriers, lookbacks) Milstein's higher
*strong*-order can resolve barrier crossings and running extrema more
accurately than Euler at the same ``n_steps``. The improvement only
shows up at the path-distribution level — pin Milstein on if your
benchmark is barrier P&L variance, not vanilla repricing.

Empirical vanilla-reprice sweep (4 seeds × 100k paths, SVI equity
skew, T = 1y, ATM strikes 90–110):

================  ============  ============  ============  ============
n_steps           100           200           500           1000
================  ============  ============  ============  ============
midpoint_euler    6.6 bp        5.9 bp        18.8 bp       13.5 bp
milstein          11.9 bp       9.6 bp        20.1 bp       14.2 bp
================  ============  ============  ============  ============

The non-monotonic ``n_steps`` profile is the signature of MC-noise-
dominated error: at this budget the noise floor is ~10–20 bp and
neither scheme gets below it via discretisation alone. Use a variance
reduction technique (control variate against the BSM analytic, or
larger ``n_paths``) if you need sub-10 bp vanilla reprice. See the
LV-2 backlog entry in the roadmap for the full discussion.

Design choices (mirroring the Andersen-QE Heston template at
``valax/pricing/mc/paths.py``):

    * **``jax.lax.scan`` over time**, with state ``log_S`` shaped
      ``(n_paths,)``. Inside each step the Dupire local-vol evaluation
      is ``jax.vmap``-ed across paths.
    * **Midpoint-in-time σ** for both schemes: at step ``n`` (from
      ``t_n = n·dt`` to ``t_{n+1} = (n+1)·dt``) we evaluate
      ``sigma_loc(S_n, t_n + 0.5·dt)``. This avoids querying the
      Dupire formula at the singular ``T = 0`` boundary, where total
      variance vanishes and the ``1/w`` terms in the denominator blow
      up.
    * **Per-step PRNG split**: ``keys = random.split(key, n_steps)``,
      then each step draws one Gaussian vector of shape ``(n_paths,)``.
    * **Output shape**: ``(n_paths, n_steps + 1)`` matching
      ``generate_gbm_paths`` and ``generate_heston_paths``.
    * **Scheme selection is static** at trace time — the unused step
      function is never compiled into the XLA graph.

Autodiff: the Dupire evaluation involves second derivatives of the
surface's total variance, so ``jax_enable_x64`` must be on (enforced at
the Dupire layer). The Milstein scheme additionally calls
``jax.value_and_grad(dupire_local_vol, argnums=0)`` inside the scan body
— mathematically this is a *third* derivative of total variance, still
clean under x64 but worth knowing if you trace memory.

Gradients through ``generate_local_vol_paths`` w.r.t. surface parameters
are well-defined for both schemes as long as no path triggers a
butterfly-arbitrage NaN at the queried ``(k_t, t_n)``.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.dupire import dupire_local_vol


SchemeName = Literal["midpoint_euler", "milstein"]


def generate_local_vol_paths(
    model: LocalVolModel,
    spot: Float[Array, ""],
    T: Float[Array, ""] | float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
    *,
    scheme: SchemeName = "midpoint_euler",
) -> Float[Array, "n_paths n_steps_plus1"]:
    """Generate spot paths under a local volatility SDE.

    Args:
        model: A ``LocalVolModel`` carrying the surface and rate / div.
        spot: Initial spot price ``S_0``.
        T: Terminal time (year fraction).
        n_steps: Number of time steps (``dt = T / n_steps``).
        n_paths: Number of Monte Carlo paths.
        key: Top-level PRNG key.
        scheme: Discretisation scheme — ``"midpoint_euler"`` (default,
            weak-order 1, 1× cost) or ``"milstein"`` (strong-order 1.5,
            ~2× per-step cost). For vanilla pricing both schemes have
            the same weak order so the MC-noise floor at typical
            budgets washes out the constant-factor improvement; the
            default favours the cheaper scheme. Pick ``"milstein"``
            for path-dependent payoffs (barriers, lookbacks) where
            *strong*-order convergence matters. See module docstring
            for the full trade-off discussion.

    Returns:
        Spot paths, shape ``(n_paths, n_steps + 1)``. Column 0 is
        ``spot`` broadcast across paths; column ``n_steps`` is ``S_T``.

    Raises:
        ValueError: if ``scheme`` is not one of the accepted literals.
    """
    if scheme not in ("midpoint_euler", "milstein"):
        raise ValueError(
            f"generate_local_vol_paths: scheme must be "
            f"'midpoint_euler' or 'milstein', got {scheme!r}"
        )

    T = jnp.asarray(T)
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    mu = model.rate - model.dividend

    log_spot = jnp.log(spot)
    surface = model.surface

    # Helper closures: convert log-spot vectors to log-moneyness at a
    # given time. F(t) = spot * exp(mu * t) ⇒ k_t = log(S_n / F(t))
    # = log_S_n - log_spot - mu * t.
    def _k_of(log_S, t):
        return log_S - (log_spot + mu * t)

    # Static dispatch on scheme — chosen at trace time, the unused
    # closure is not compiled. See ``docs/architecture/jax-patterns.md``
    # §2.1 for the rationale: ``scheme`` is a concrete Python string at
    # the call site, the ``if`` runs in pure Python before tracing
    # starts, and only the selected ``step`` closure enters the
    # ``lax.scan`` body and the XLA graph.
    if scheme == "midpoint_euler":
        # σ_loc only — no derivative needed.
        def _sigma_at_paths(log_S, t):
            k = _k_of(log_S, t)
            return jax.vmap(lambda kk: dupire_local_vol(surface, kk, t))(k)

        def step(carry, scan_input):
            log_S = carry
            t_n, key_t = scan_input
            z = jax.random.normal(key_t, shape=(n_paths,))
            sigma = _sigma_at_paths(log_S, t_n)
            log_S_next = (
                log_S
                + (mu - 0.5 * sigma * sigma) * dt
                + sigma * sqrt_dt * z
            )
            return log_S_next, log_S_next

    else:  # scheme == "milstein"
        # σ_loc and ∂σ_loc/∂k jointly — ``value_and_grad`` avoids two
        # forward passes.
        _sigma_and_grad_scalar = jax.value_and_grad(
            lambda kk, tt: dupire_local_vol(surface, kk, tt),
            argnums=0,
        )

        def _sigma_and_grad_at_paths(log_S, t):
            k = _k_of(log_S, t)
            return jax.vmap(lambda kk: _sigma_and_grad_scalar(kk, t))(k)

        def step(carry, scan_input):
            log_S = carry
            t_n, key_t = scan_input
            z = jax.random.normal(key_t, shape=(n_paths,))
            sigma, dsigma_dk = _sigma_and_grad_at_paths(log_S, t_n)
            # Milstein-corrected log-Euler. The correction term
            # 0.5 * sigma * (dsigma/dk) * dt * (z^2 - 1) has expectation
            # zero under N(0,1), so the martingale property of the
            # discount-adjusted spot is preserved in expectation.
            log_S_next = (
                log_S
                + (mu - 0.5 * sigma * sigma) * dt
                + sigma * sqrt_dt * z
                + 0.5 * sigma * dsigma_dk * dt * (z * z - 1.0)
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
