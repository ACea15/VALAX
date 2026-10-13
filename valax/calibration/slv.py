"""SLV leverage-function calibration.

Two-pass calibration of a Stochastic-Local Volatility model.

* **Pass 1** (Heston → vanillas) is delegated to
  :func:`~valax.calibration.heston.calibrate_heston` (optionally with
  parameters held fixed and several expiries), or supplied pre-fitted.
* **Pass 2** (this module): build ``L(k, t)`` so that the Markovian
  projection of the SLV model matches the input local-vol surface,

  .. math::

      L^2(k, t) = \\sigma_{\\mathrm{Dupire}}^2(k, t) \\,/\\,
                  \\widehat{\\mathbb{E}}[V_t \\mid k_t = k],

  with the conditional expectation estimated from a simulated particle
  swarm.

Algorithm
---------

**One pass is the Guyon-Henry-Labordère (2012) particle method.** The swarm
is marched forward through ``time_grid`` (a ``lax.scan``). At each row time
``t_i`` the conditional expectation ``E[V | k]`` is estimated from the
*current* particles, the leverage row
``L(·, t_i) = σ_Dupire(·, t_i) / sqrt(Ê[V | ·])`` is set (clipped to
``[L_min, L_max]``), and the particles are stepped onward with the rows
built so far. A single pass is therefore the self-consistent one-shot
calibration: the swarm that estimates row ``i`` was simulated under rows
``0 … i-1`` of the same leverage.

**Time convention (shared with the pricer).** The particle step from
``t_{i-1}`` to ``t_i`` uses exactly the query of
:func:`~valax.pricing.mc.slv_paths.generate_slv_paths`: ``L`` interpolated
bilinearly (via :class:`~valax.surfaces.leverage.LeverageGrid`) at the
midpoint time ``t_mid = (t_{i-1} + t_i)/2`` and at
``k = log S - log F(t_mid)``. That interpolation blends row ``i-1``
(already built) with row ``i`` (not yet built — it needs the particles at
``t_i``), and before ``time_grid[0]`` it is flat at row 0. The missing
row is *predicted*: on the first pass by
``σ_Dupire(·, t_i) / sqrt(Ê_{i-1}[V | ·])`` (the conditional expectation
frozen over one step; ``Ê_0 = v_0`` exactly, since every particle starts
at ``(S_0, v_0)``), on later passes by the previous pass's row ``i``.

**Outer passes.** ``n_iterations`` such passes are run, each with fresh
randomness (``fold_in(key, pass)``). Because each pass is already
self-consistent, pass ``K ≥ 2`` differs from pass ``K-1`` only by the
re-simulation noise of the estimator and the (``O(Δt)``) predictor
correction: the reported increment ``L_K - L_{K-1}`` is a **Monte Carlo
stability check** of the estimator, not the contraction of a Picard
iteration. See :func:`calibrate_slv_leverage` for how it is reported.

Two conditional-expectation estimators are exposed via ``method=``:

* ``"particle"`` (default): Nadaraya-Watson Gaussian kernel directly
  on the particles, no regularisation.
* ``"kernel"``: the same Nadaraya-Watson estimator shrunk toward the
  unconditional particle mean ``V̄`` by ``ridge`` *pseudo-particles*:

  .. math::

      \\widehat{\\mathbb E}[V \\mid k] =
      \\frac{\\sum_j w_j(k)\\,V_j + \\lambda\\,\\bar V}
           {\\sum_j w_j(k) + \\lambda},
      \\qquad w_j(k) = e^{-\\frac12 ((k - k_j)/h)^2} \\in (0, 1].

  ``Σ_j w_j(k)`` is the effective number of particles near ``k``; where it
  is much larger than ``λ`` the estimate is the plain kernel regression,
  where it is comparable or smaller (the wings) the estimate is pulled to
  ``V̄``. ``λ`` is dimensionless and independent of ``n_paths``, so the
  bias it introduces vanishes as the swarm grows.

Bandwidth
---------

By default we use Silverman's rule of thumb on the particle swarm's
empirical ``k`` standard deviation at each time slice,

.. math::

    h_t = 1.06 \\, \\widehat{\\sigma}_{k_t} \\, N^{-1/5}.

Pass an explicit ``bandwidth=`` scalar (or callable
``bandwidth(t_i, k_particles) -> h``) to override.

References:
    - Guyon, J., & Henry-Labordère, P. (2012). "Being Particular About
      Calibration." Risk, January.
    - Henry-Labordère, P. (2009). Analysis, Geometry, and Modeling in
      Finance. CRC. (Ch. 12 on SLV.)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult, fixed_point_diagnostics
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import explicit_level
from valax.models.heston import HestonModel
from valax.pricing.analytic.dupire import (
    DEFAULT_LOCAL_VOL_GUARD,
    LocalVolGuard,
    dupire_local_vol,
    guarded_local_variance,
)
from valax.surfaces.leverage import LeverageGrid

# ``SLVModel`` is imported lazily inside ``calibrate_slv`` to avoid an
# import cycle: ``valax.surfaces.__init__`` imports ``sabr_surface``
# which imports ``valax.calibration.sabr`` which triggers this module.
# If ``SLVModel`` were imported at top level, that chain would re-enter
# ``valax.models.slv`` before ``LeverageGrid`` is fully loaded.


EstimatorName = Literal["particle", "kernel"]

#: Default ``ridge`` of ``method="kernel"``: the number of pseudo-particles
#: at the unconditional mean ``V̄``. Chosen by measurement on example 17's
#: SABR market at 50 000 particles: versus no ridge it cuts the seed-to-seed
#: max|ΔL| 1.7–5× (wings), moves the ATM leverage by ≤ 0.2 % and leaves the
#: vanilla repricing equal or better; 100 starts to bias ATM (0.7 %). See
#: ``docs/guide/slv.md#choosing-particle-vs-kernel``.
DEFAULT_RIDGE = 30.0


# ─────────────────────────────────────────────────────────────────────
# Conditional-expectation estimators
# ─────────────────────────────────────────────────────────────────────


def _silverman_bandwidth(
    k_particles: Float[Array, " n_paths"],
) -> Float[Array, ""]:
    """Silverman's rule of thumb: ``h = 1.06 · σ̂ · N^{-1/5}``."""
    n = jnp.array(k_particles.shape[0], dtype=k_particles.dtype)
    sigma_hat = jnp.std(k_particles)
    # Tiny floor so a degenerate (collapsed) swarm doesn't divide by zero.
    return jnp.maximum(1.06 * sigma_hat * n ** (-0.2), 1e-6)


def _nadaraya_watson(
    k_query: Float[Array, " n_k"],
    k_particles: Float[Array, " n_paths"],
    v_particles: Float[Array, " n_paths"],
    bandwidth: Float[Array, ""],
    ridge: Float[Array, ""],
) -> Float[Array, " n_k"]:
    """Gaussian-kernel Nadaraya-Watson estimator of ``E[V | k]`` with a ridge.

    .. math::

        \\widehat{\\mathbb E}[V \\mid k] =
        \\frac{\\sum_j w_j V_j + \\lambda \\bar V}{\\sum_j w_j + \\lambda},
        \\qquad w_j = e^{-\\frac12 ((k - k_j)/h)^2}.

    The weights are the *unnormalised* Gaussian kernel, each in ``(0, 1]``,
    so ``Σ_j w_j`` is the effective particle count near ``k`` and ``λ`` is a
    number of pseudo-particles at the unconditional mean ``V̄``. Evaluation
    is stabilised in log space with the prior treated as one more weight
    ``log λ``, so far from every particle the estimate tends to ``V̄`` (for
    ``λ > 0``) or to the nearest particles' values (for ``λ = 0``) instead of
    ``0/0``.

    Args:
        k_query: Evaluation points in ``k`` space.
        k_particles: Particle ``k`` values.
        v_particles: Particle ``V`` values.
        bandwidth: Gaussian kernel bandwidth ``h``.
        ridge: Number of pseudo-particles ``λ ≥ 0`` at the unconditional
            mean; ``0`` is the plain Nadaraya-Watson estimator.

    Returns:
        Estimated conditional means at ``k_query``.
    """
    diff = (k_query[:, None] - k_particles[None, :]) / bandwidth
    log_w = -0.5 * diff * diff
    ridge = jnp.asarray(ridge, dtype=log_w.dtype)
    log_prior = jnp.where(ridge > 0.0, jnp.log(jnp.where(ridge > 0.0, ridge, 1.0)), -jnp.inf)
    # Common shift: the largest log-weight among particles and the prior.
    shift = jnp.maximum(jnp.max(log_w, axis=1), log_prior)
    w = jnp.exp(log_w - shift[:, None])
    w_prior = jnp.exp(log_prior - shift)

    prior = jnp.mean(v_particles)
    num = jnp.sum(w * v_particles[None, :], axis=1) + w_prior * prior
    den = jnp.sum(w, axis=1) + w_prior
    # den >= 1 by construction of the shift.
    return num / den


# ─────────────────────────────────────────────────────────────────────
# One Guyon-Henry-Labordère pass
# ─────────────────────────────────────────────────────────────────────


#: Default settings of the SLV leverage fixed point. Only ``atol`` is read: the
#: convergence tolerance on ``max|L_K - L_{K-1}|`` between the last two outer
#: passes. Each pass re-simulates the swarm with fresh randomness, so the
#: increment never falls below the Monte Carlo noise of the leverage estimate
#: (roughly ``1/sqrt(n_paths)`` relative); set ``atol`` above that floor.
SLV_SETTINGS = SolverSettings(atol=5e-2, rtol=0.0)


def _leverage_fixed_point(
    heston: HestonModel,
    surface: Any,
    spot: Float[Array, ""],
    log_moneyness_grid: Float[Array, " n_k"],
    time_grid: Float[Array, " n_t"],
    n_paths: int,
    key: jax.Array,
    *,
    method: str,
    n_iterations: int,
    bandwidth: Float[Array, ""] | Callable | None,
    ridge: float,
    L_max: float,
    L_min: float,
    guard: LocalVolGuard | None,
    n_substeps: int = 1,
) -> tuple[Float[Array, "n_t n_k"], Float[Array, "n_t n_k"], Float[Array, ""]]:
    """Run ``n_iterations`` GHL passes; return ``(L_K, L_{K-1}, clip_fraction)``.

    ``L_0 ≡ 1`` (the Heston limit) is the reference of a single pass.
    """
    # Import locally to avoid an import cycle (slv_paths imports SLVModel,
    # which conceptually sits "below" calibration).
    from valax.pricing.mc.slv_paths import _qe_variance_step_factory, _safe_sqrt

    dtype = jnp.result_type(spot, float)
    log_moneyness_grid = jnp.asarray(log_moneyness_grid, dtype=dtype)
    time_grid = jnp.asarray(time_grid, dtype=dtype)
    ridge_eff = jnp.asarray(ridge if method == "kernel" else 0.0, dtype=dtype)
    n_t = time_grid.shape[0]
    n_k = log_moneyness_grid.shape[0]

    mu = heston.rate - heston.dividend
    rho = heston.rho
    sqrt_1_minus_rho2 = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))
    log_spot0 = jnp.log(spot)

    if bandwidth is None:
        bw_fn = lambda t, kp: _silverman_bandwidth(kp)
    elif callable(bandwidth):
        bw_fn = bandwidth
    else:
        bw_const = jnp.asarray(bandwidth)
        bw_fn = lambda t, kp: bw_const

    # Dupire targets σ²_loc(k_j, t_i) on the output grid, once.
    if guard is None:
        def _local_var(kk, t):
            return dupire_local_vol(surface, kk, t) ** 2
    else:
        def _local_var(kk, t):
            return guarded_local_variance(surface, kk, t, guard)
    sigma2_target = jax.vmap(
        lambda t: jax.vmap(lambda kk: _local_var(kk, t))(log_moneyness_grid)
    )(time_grid)  # (n_t, n_k)
    # Target of the row after i (the predictor of the first pass); the last
    # row has no successor and repeats itself.
    sigma2_next = jnp.concatenate([sigma2_target[1:], sigma2_target[-1:]], axis=0)
    t_prev_grid = jnp.concatenate([jnp.zeros((1,), dtype=dtype), time_grid[:-1]])

    def _leverage_row(sigma2_row, ev):
        raw = jnp.sqrt(jnp.maximum(sigma2_row / jnp.maximum(ev, 1e-12), 0.0))
        return jnp.clip(raw, L_min, L_max), (raw > L_max) | (raw < L_min)

    # Ê_0[V | k] = v0 exactly: every particle starts at (S_0, v_0).
    row0_pred, _ = _leverage_row(
        sigma2_target[0], jnp.broadcast_to(heston.v0, (n_k,))
    )

    def ghl_pass(pass_key, L_prev, use_prev: bool):
        """One self-consistent forward sweep; ``use_prev`` is static."""
        step_keys = jax.vmap(lambda i: jax.random.fold_in(pass_key, i))(
            jnp.arange(n_t)
        )

        def body(carry, inp):
            log_S, v, values, pred_row = carry
            i, t_prev, t_i, step_key, s2_row, s2_next = inp
            predictor = L_prev[i] if use_prev else pred_row
            values = values.at[i].set(predictor)
            grid = LeverageGrid(
                log_moneyness_grid=log_moneyness_grid,
                time_grid=time_grid,
                values=values,
            )

            dt = (t_i - t_prev) / n_substeps
            qe_step = _qe_variance_step_factory(
                heston.kappa, heston.theta, heston.xi, dt,
            )

            def substep(state, j):
                log_S, v = state
                t_mid = t_prev + (j + 0.5) * dt
                sub_key = (
                    step_key if n_substeps == 1
                    else jax.random.fold_in(step_key, j)
                )
                k_v, k_perp, k_u = jax.random.split(sub_key, 3)
                z_v = jax.random.normal(k_v, shape=(n_paths,), dtype=dtype)
                z_perp = jax.random.normal(k_perp, shape=(n_paths,), dtype=dtype)
                u = jax.random.uniform(k_u, shape=(n_paths,), dtype=dtype)
                z_s = rho * z_v + sqrt_1_minus_rho2 * z_perp
                v_next = qe_step(v, z_v, u)
                # Same query as generate_slv_paths: L(k(t_mid), t_mid).
                k_mid = log_S - (log_spot0 + mu * t_mid)
                L_n = jax.vmap(lambda kk: grid(kk, t_mid))(k_mid)
                sigma = L_n * _safe_sqrt(v)
                log_S_next = (
                    log_S + (mu - 0.5 * sigma * sigma) * dt
                    + sigma * jnp.sqrt(dt) * z_s
                )
                return (log_S_next, v_next), None

            (log_S_next, v_next), _ = jax.lax.scan(
                substep, (log_S, v), jnp.arange(n_substeps),
            )

            # Particles at t_i → Ê[V | k] → row i.
            k_i = log_S_next - (log_spot0 + mu * t_i)
            ev = _nadaraya_watson(
                log_moneyness_grid, k_i, v_next,
                bandwidth=bw_fn(t_i, k_i), ridge=ridge_eff,
            )
            row, clipped = _leverage_row(s2_row, ev)
            values = values.at[i].set(row)
            next_pred, _ = _leverage_row(s2_next, ev)
            return (log_S_next, v_next, values, next_pred), (row, clipped)

        carry0 = (
            jnp.broadcast_to(log_spot0, (n_paths,)).astype(dtype),
            jnp.broadcast_to(heston.v0, (n_paths,)).astype(dtype),
            jnp.ones((n_t, n_k), dtype=dtype),
            row0_pred,
        )
        xs = (
            jnp.arange(n_t), t_prev_grid, time_grid, step_keys,
            sigma2_target, sigma2_next,
        )
        _, (rows, clipped) = jax.lax.scan(body, carry0, xs)
        return rows, clipped

    L_previous = jnp.ones((n_t, n_k), dtype=dtype)
    L_values, clipped = ghl_pass(jax.random.fold_in(key, 0), L_previous, False)
    if n_iterations >= 2:
        def outer(carry, pass_key):
            L_last, _, _ = carry
            L_new, clip_new = ghl_pass(pass_key, L_last, True)
            return (L_new, L_last, clip_new), None

        pass_keys = jax.vmap(lambda p: jax.random.fold_in(key, p))(
            jnp.arange(1, n_iterations)
        )
        (L_values, L_previous, clipped), _ = jax.lax.scan(
            outer, (L_values, L_previous, clipped), pass_keys,
        )
    return L_values, L_previous, jnp.mean(clipped.astype(dtype))


# ─────────────────────────────────────────────────────────────────────
# Public calibration API
# ─────────────────────────────────────────────────────────────────────


def calibrate_slv_leverage(
    heston: HestonModel,
    surface: Any,
    spot: Float[Array, ""],
    log_moneyness_grid: Float[Array, " n_k"],
    time_grid: Float[Array, " n_t"],
    n_paths: int,
    key: jax.Array,
    *,
    method: EstimatorName = "particle",
    n_iterations: int = 2,
    bandwidth: Float[Array, ""] | Callable | None = None,
    ridge: float = DEFAULT_RIDGE,
    L_max: float = 5.0,
    L_min: float = 0.05,
    n_substeps: int = 1,
    local_vol_guard: LocalVolGuard | None = DEFAULT_LOCAL_VOL_GUARD,
    settings: SolverSettings | None = None,
    name: str = "slv_leverage",
) -> CalibrationResult:
    """Calibrate the SLV leverage function ``L(k, t)``.

    Every pass is a full Guyon-Henry-Labordère forward sweep (see the module
    docstring): rows are built in time order from the particles simulated
    under the rows already built, so ``n_iterations=1`` is the
    self-consistent one-shot calibration.

    **How the fixed point is reported.** There is no residual to drive to
    zero — the leverage is the output of ``n_iterations`` Monte Carlo passes
    ``L_{k+1} = G(L_k)`` (``L_k`` enters pass ``k+1`` only as the predictor
    of the not-yet-built row in the midpoint-in-time interpolation) — so the
    level is *explicit* (``theta=None``): its value is an unrolled,
    differentiable map of the inputs (Heston model, target surface, spot),
    and its risk is taken by plain reverse-mode AD through the passes, never
    by the implicit function theorem (``problem="fixed_point"``). Its
    diagnostics are those of the fixed point
    (:func:`~valax.calibration.result.fixed_point_diagnostics`):

    * ``residuals`` — the last increment ``L_K - L_{K-1}`` on the grid. With
      self-consistent passes this is the difference of two re-simulated
      estimates: a **Monte Carlo stability check** of the estimator;
    * ``converged`` — ``max|L_K - L_{K-1}| <= settings.atol``, and always
      ``False`` for ``n_iterations=1`` (a single pass has nothing to compare
      with but the ``L ≡ 1`` warm start);
    * ``extras["clip_fraction"]`` — the share of grid cells whose raw
      leverage hit ``[L_min, L_max]`` in the last pass (a high value means
      the estimator is degenerate somewhere, whatever the increment says).

    The increment is bounded below by Monte Carlo noise; ``settings.atol``
    must sit above that floor (see :data:`SLV_SETTINGS`).

    Args:
        heston: Pass-1 calibrated ``HestonModel`` (its ``rate`` / ``dividend``
            set the flat particle drift).
        surface: Implied-vol surface with ``total_variance(k, T)`` —
            the Dupire target.
        spot: Spot at calibration date ``S_0``.
        log_moneyness_grid: Output grid in ``k = log(K/F(T))``. Must
            be sorted ascending. The grid is the leverage's eventual
            ``log_moneyness_grid``.
        time_grid: Output grid in time (year fractions), sorted ascending,
            ``time_grid[0] > 0`` (Dupire is singular at 0). The swarm takes
            one step per row, so the spacing is the calibration time step.
        n_paths: Particle count.
        key: PRNG key; pass ``p`` uses ``fold_in(key, p)``.
        method: ``"particle"`` (plain Nadaraya-Watson, ``ridge`` ignored)
            or ``"kernel"`` (shrinkage by ``ridge`` pseudo-particles).
        n_iterations: GHL passes (default 2, the minimum that measures the
            Monte Carlo stability of the estimate). ``1`` is the classical
            one-shot particle method — usable, but always reported not
            converged.
        bandwidth: Gaussian-kernel bandwidth. ``None`` (default) →
            Silverman's rule per slice. Scalar or callable
            ``(t, k_particles) -> h`` to override.
        ridge: For ``method="kernel"``, the number ``λ`` of pseudo-particles
            at the unconditional mean ``V̄`` (dimensionless, independent of
            ``n_paths``; default :data:`DEFAULT_RIDGE`). Cells with far more
            than ``λ`` effective particles are unaffected; sparse wing cells
            are shrunk toward ``V̄``. Ignored when ``method="particle"``.
        L_max: Upper clip on ``L`` to prevent runaway values when the
            kernel density is degenerate in a region with strong skew.
        L_min: Lower clip on ``L`` (same rationale as ``L_max``).
        n_substeps: Particle steps per leverage row (default 1). The swarm's
            time step is ``Δt_row / n_substeps``; match it to the pricer's
            ``T / n_steps`` — the calibration and the pricer then discretise
            the same SDE the same way, which matters at high vol-of-vol.
            Every sub-step queries ``L`` at its own midpoint time.
        local_vol_guard: Guard applied to the Dupire target (see
            :class:`~valax.pricing.analytic.dupire.LocalVolGuard`); ``None``
            uses raw :func:`~valax.pricing.analytic.dupire.dupire_local_vol`.
        settings: Fixed-point tolerance (``atol``); defaults to
            :data:`SLV_SETTINGS`.
        name: Level name used in diagnostics.

    Returns:
        The fixed-point :class:`~valax.calibration.result.CalibrationResult`;
        the :class:`LeverageGrid` of shape
        ``(len(time_grid), len(log_moneyness_grid))`` is ``result.value``.

    Raises:
        ValueError: If ``method`` is unknown, ``n_iterations < 1`` or
            ``ridge < 0``.
    """
    if method not in ("particle", "kernel"):
        raise ValueError(
            f"calibrate_slv_leverage: method must be 'particle' or "
            f"'kernel', got {method!r}"
        )
    if n_iterations < 1:
        raise ValueError(
            f"n_iterations must be >= 1, got {n_iterations}"
        )
    if isinstance(ridge, (int, float)) and ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
    if heston.rate is None or heston.dividend is None:
        raise ValueError(
            "calibrate_slv_leverage: the Heston backbone must carry rate and "
            "dividend (they set the particle drift)."
        )
    settings = SLV_SETTINGS if settings is None else settings
    inputs = LevelInputs(
        quotes=None,
        upstream={
            "heston": heston,
            "surface": surface,
            "spot": jnp.asarray(spot),
        },
        fixed={
            "log_moneyness_grid": jnp.asarray(log_moneyness_grid),
            "time_grid": jnp.asarray(time_grid),
        },
    )

    def run(inp: LevelInputs):
        up, fx = inp.upstream, inp.fixed
        return _leverage_fixed_point(
            up["heston"], up["surface"], up["spot"],
            fx["log_moneyness_grid"], fx["time_grid"], n_paths, key,
            method=method, n_iterations=n_iterations, bandwidth=bandwidth,
            ridge=ridge, L_max=L_max, L_min=L_min, guard=local_vol_guard,
            n_substeps=n_substeps,
        )

    def assemble(_theta: None, inp: LevelInputs) -> LeverageGrid:
        values, _, _ = run(inp)
        return LeverageGrid(
            log_moneyness_grid=inp.fixed["log_moneyness_grid"],
            time_grid=inp.fixed["time_grid"],
            values=values,
        )

    values, previous, clip_fraction = run(inputs)
    diagnostics = fixed_point_diagnostics(
        values - previous,
        n_iterations=n_iterations,
        tol=settings.atol,
        extras={"clip_fraction": clip_fraction},
    )
    return explicit_level(
        name, inputs, assemble,
        diagnostics=diagnostics,
        problem="fixed_point",
        settings=settings,
        value=LeverageGrid(
            log_moneyness_grid=inputs.fixed["log_moneyness_grid"],
            time_grid=inputs.fixed["time_grid"],
            values=values,
        ),
    )


def _supplied_heston_level(model: HestonModel, name: str) -> CalibrationResult:
    """Wrap a caller-supplied Heston backbone as an (input) explicit level."""
    inputs = LevelInputs(quotes=None, upstream=None, fixed={"heston": model})
    return explicit_level(
        name, inputs, lambda _theta, inp: inp.fixed["heston"], value=model,
    )


def calibrate_slv(
    strikes: Float[Array, " n"] | None = None,
    market_prices: Float[Array, " n"] | None = None,
    spot: Float[Array, ""] | None = None,
    rate: Float[Array, ""] | None = None,
    dividend: Float[Array, ""] | None = None,
    expiry: Float[Array, ""] | Float[Array, " n"] | None = None,
    pricing_fn: Callable | None = None,
    surface: Any = None,
    log_moneyness_grid: Float[Array, " n_k"] | None = None,
    time_grid: Float[Array, " n_t"] | None = None,
    n_paths: int | None = None,
    key: jax.Array | None = None,
    *,
    heston: HestonModel | CalibrationResult | None = None,
    fixed_params: Mapping[str, float | Array] | None = None,
    heston_weights: Float[Array, " n"] | None = None,
    method: EstimatorName = "particle",
    n_iterations: int = 2,
    bandwidth: Float[Array, ""] | Callable | None = None,
    ridge: float = DEFAULT_RIDGE,
    L_max: float = 5.0,
    L_min: float = 0.05,
    n_substeps: int = 1,
    local_vol_guard: LocalVolGuard | None = DEFAULT_LOCAL_VOL_GUARD,
    heston_initial_guess: HestonModel | None = None,
    heston_settings: SolverSettings | None = None,
    settings: SolverSettings | None = None,
    name: str = "slv",
) -> CalibrationResult:
    """End-to-end two-pass SLV calibration.

    Pass 1 (the Heston backbone) is either

    * **fitted** — :func:`~valax.calibration.heston.calibrate_heston` on the
      vanilla quotes (``strikes`` … ``pricing_fn``), with ``fixed_params``
      held fixed and ``expiry`` either a scalar or one expiry per strike
      (a multi-expiry fit is far better conditioned than a single smile); or
    * **supplied** — ``heston=`` a fitted
      :class:`~valax.models.heston.HestonModel` or a
      :class:`~valax.calibration.result.CalibrationResult` whose value is one
      (e.g. a :func:`~valax.calibration.nodes.price_fit_node` result). A
      result is recorded as the child as is; a bare model is recorded as an
      explicit input level. ``rate`` / ``dividend``, if given, override the
      backbone's carry (required when the backbone leaves them ``None``).

    Pass 2: :func:`calibrate_slv_leverage` against the Dupire surface (a
    fixed-point level). The returned result is pass 2's, with its value
    replaced by the assembled :class:`SLVModel` and pass 1's result attached
    as its child — so ``result.check()`` covers both passes and
    ``result.converged`` is ``False`` if either failed.

    Args:
        strikes: Vanilla strikes of the Heston fit (fitted backbone only).
        market_prices: Vanilla prices of the Heston fit (fitted only).
        spot: Spot ``S_0`` (both passes; always required).
        rate: Risk-free rate (fitted: forwarded; supplied: optional carry
            override).
        dividend: Dividend yield (as ``rate``).
        expiry: Scalar expiry or one per strike (fitted only).
        pricing_fn: ``(model, strike, spot, rate, dividend, expiry) -> price``
            (fitted only).
        surface: Dupire target surface (required).
        log_moneyness_grid: Leverage ``k`` grid (required).
        time_grid: Leverage time grid (required).
        n_paths: Particle count (required).
        key: PRNG key of the particle passes (required).
        heston: Pre-fitted backbone; skips the Heston fit.
        fixed_params: Heston fields held fixed in the fit (fitted only).
        heston_weights: Per-quote weights of the Heston fit (fitted only).
        method: Forwarded to ``calibrate_slv_leverage``.
        n_iterations: Forwarded to ``calibrate_slv_leverage``.
        bandwidth: Forwarded to ``calibrate_slv_leverage``.
        ridge: Forwarded to ``calibrate_slv_leverage``.
        L_max: Forwarded to ``calibrate_slv_leverage``.
        L_min: Forwarded to ``calibrate_slv_leverage``.
        n_substeps: Forwarded to ``calibrate_slv_leverage``.
        local_vol_guard: Forwarded to ``calibrate_slv_leverage``.
        heston_initial_guess: Starting point of the Heston fit (fitted only).
        heston_settings: Solver settings of the Heston fit (fitted only).
        settings: Fixed-point tolerance of pass 2.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the
        calibrated :class:`SLVModel` is ``result.value`` and the Heston level
        is ``result.children[0]``.

    Raises:
        ValueError: On a missing required argument, on Heston-fit arguments
            combined with ``heston=``, or on a supplied backbone with no
            rate / dividend and none given.
    """
    from valax.calibration.heston import calibrate_heston
    from valax.models.slv import SLVModel  # lazy import — see note at top of file

    missing = [
        n for n, v in (
            ("spot", spot), ("surface", surface),
            ("log_moneyness_grid", log_moneyness_grid),
            ("time_grid", time_grid), ("n_paths", n_paths), ("key", key),
        ) if v is None
    ]
    if missing:
        raise ValueError(f"calibrate_slv: missing required argument(s) {missing}")

    if heston is not None:
        fit_args = {
            "strikes": strikes, "market_prices": market_prices,
            "expiry": expiry, "pricing_fn": pricing_fn,
            "fixed_params": fixed_params, "heston_weights": heston_weights,
            "heston_initial_guess": heston_initial_guess,
            "heston_settings": heston_settings,
        }
        given = [n for n, v in fit_args.items() if v is not None]
        if given:
            raise ValueError(
                f"calibrate_slv: heston= supplies the backbone, so the Heston "
                f"fit arguments {given} must not be passed."
            )
        if isinstance(heston, CalibrationResult):
            heston_result = heston
        elif isinstance(heston, HestonModel):
            heston_result = _supplied_heston_level(heston, f"{name}.heston")
        else:
            raise ValueError(
                f"calibrate_slv: heston= must be a HestonModel or a "
                f"CalibrationResult, got {type(heston).__name__}"
            )
        backbone = heston_result.value
        if not isinstance(backbone, HestonModel):
            raise ValueError(
                f"calibrate_slv: heston= result value is a "
                f"{type(backbone).__name__}, not a HestonModel"
            )
        if rate is not None:
            backbone = eqx.tree_at(
                lambda m: m.rate, backbone, jnp.asarray(rate),
                is_leaf=lambda x: x is None,
            )
        if dividend is not None:
            backbone = eqx.tree_at(
                lambda m: m.dividend, backbone, jnp.asarray(dividend),
                is_leaf=lambda x: x is None,
            )
        if backbone.rate is None or backbone.dividend is None:
            raise ValueError(
                "calibrate_slv: the supplied Heston backbone has no rate / "
                "dividend; pass rate= and dividend=."
            )
    else:
        fit_missing = [
            n for n, v in (
                ("strikes", strikes), ("market_prices", market_prices),
                ("rate", rate), ("dividend", dividend), ("expiry", expiry),
                ("pricing_fn", pricing_fn),
            ) if v is None
        ]
        if fit_missing:
            raise ValueError(
                f"calibrate_slv: without heston=, the Heston fit needs "
                f"{fit_missing}"
            )
        heston_result = calibrate_heston(
            strikes=strikes,
            market_prices=market_prices,
            spot=spot,
            rate=rate,
            dividend=dividend,
            expiry=expiry,
            pricing_fn=pricing_fn,
            initial_guess=heston_initial_guess,
            weights=heston_weights,
            fixed_params=fixed_params,
            settings=heston_settings,
            name=f"{name}.heston",
        )
        backbone = heston_result.value

    leverage = calibrate_slv_leverage(
        backbone,
        surface,
        spot,
        log_moneyness_grid,
        time_grid,
        n_paths,
        key,
        method=method,
        n_iterations=n_iterations,
        bandwidth=bandwidth,
        ridge=ridge,
        L_max=L_max,
        L_min=L_min,
        n_substeps=n_substeps,
        local_vol_guard=local_vol_guard,
        settings=settings,
        name=f"{name}.leverage",
    )
    model = SLVModel.from_heston_and_leverage(backbone, surface, leverage.value)
    return dataclasses.replace(leverage, value=model, children=(heston_result,))
