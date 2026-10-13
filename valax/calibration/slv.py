"""SLV leverage-function calibration.

Two-pass calibration of a Stochastic-Local Volatility model.

* **Pass 1** (Heston → vanillas) is delegated to the existing
  ``calibrate_heston``; this module assumes the caller has already
  produced a fitted ``HestonModel``.
* **Pass 2** (this module): build ``L(k, t)`` so that the Markovian
  projection of the SLV model matches the input local-vol surface,

  .. math::

      L^2(k, t) = \\sigma_{\\mathrm{Dupire}}^2(k, t) \\,/\\,
                  \\widehat{\\mathbb{E}}[V_t \\mid k_t = k],

  with the conditional expectation estimated from a simulated particle
  swarm.

Algorithm
---------

The classical Guyon-Henry-Labordère (2012) particle method advances
the particle swarm time-slice by time-slice, fitting one row of the
leverage grid per step. This module wraps that inner forward sweep in
an **outer fixed-point iteration** (``n_iterations``): each outer
iteration re-simulates the swarm from ``t=0`` using the leverage built
on the previous pass, then rebuilds the grid. ``n_iterations=1``
recovers the standard one-shot particle method; ``n_iterations >= 2``
(the default is 2) trades MC budget for improved self-consistency and is
what makes convergence *measurable*: the result reports the last increment
``L_K - L_{K-1}`` as a fixed-point level (see :func:`calibrate_slv_leverage`).

Two conditional-expectation estimators are exposed via ``method=``:

* ``"particle"`` (default): Nadaraya-Watson Gaussian kernel directly
  on the particles. Cheap, unbiased in the asymptotic limit, can be
  noisy in the tails where particle density is low.
* ``"kernel"``: same NW core but adds a small ridge term
  ``ridge`` to the kernel-weight denominator (Tikhonov-style
  regularisation). This biases the estimator toward the prior
  (empirical particle mean of ``V``) in low-density regions,
  stabilising tail behaviour at the cost of a small bias in
  well-populated regions. Recommended when the leverage grid extends
  beyond ``±3σ`` of the simulated swarm.

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
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult, fixed_point_diagnostics
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import explicit_level
from valax.models.heston import HestonModel
from valax.pricing.analytic.dupire import dupire_local_vol
from valax.surfaces.leverage import LeverageGrid

# ``SLVModel`` is imported lazily inside ``calibrate_slv`` to avoid an
# import cycle: ``valax.surfaces.__init__`` imports ``sabr_surface``
# which imports ``valax.calibration.sabr`` which triggers this module.
# If ``SLVModel`` were imported at top level, that chain would re-enter
# ``valax.models.slv`` before ``LeverageGrid`` is fully loaded.


EstimatorName = Literal["particle", "kernel"]


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
    """Gaussian-kernel Nadaraya-Watson estimator of ``E[V | k]``.

    Args:
        k_query: Evaluation points in ``k`` space.
        k_particles: Particle ``k`` values, shape ``(n_paths,)``.
        v_particles: Particle ``V`` values, shape ``(n_paths,)``.
        bandwidth: Gaussian kernel bandwidth ``h``.
        ridge: Non-negative regulariser added to the denominator. For
            ``method="kernel"`` this is typically ``ridge ≈ 1e-3``;
            for ``method="particle"`` pass ``0.0``. The prior mean
            used by the regulariser is the empirical particle mean of
            ``V`` — i.e. the unconditional MC estimate, which the
            estimator falls back to in zero-density regions.

    Returns:
        Estimated conditional means at ``k_query``, shape ``(n_k,)``.
    """
    # Pairwise (k_query, k_particle) distances.
    diff = (k_query[:, None] - k_particles[None, :]) / bandwidth
    log_w = -0.5 * diff * diff
    # Row-wise log-sum-exp stabilisation: subtract per-row max so
    # weights do not underflow at large bandwidths or wide queries.
    log_w_max = jnp.max(log_w, axis=1, keepdims=True)
    w = jnp.exp(log_w - log_w_max)

    num = jnp.sum(w * v_particles[None, :], axis=1)
    den = jnp.sum(w, axis=1)

    # Tikhonov ridge toward the empirical mean (constant prior). The
    # ridge has the same kernel-mass units as ``den`` after the
    # log-sum-exp stabilisation, so a value on the order of ``n_paths
    # · exp(-log_w_max)`` is comparable to the data weight. We expose
    # ``ridge`` as a raw scalar (caller picks scale) and add a
    # ``ridge * prior`` contribution to the numerator. With
    # ``ridge = 0`` this reduces to pure Nadaraya-Watson.
    prior = jnp.mean(v_particles)
    num = num + ridge * prior
    den = den + ridge

    return num / jnp.maximum(den, 1e-300)


# ─────────────────────────────────────────────────────────────────────
# Single-time-slice particle propagation
# ─────────────────────────────────────────────────────────────────────


def _propagate_one_step(
    log_S: Float[Array, " n_paths"],
    v: Float[Array, " n_paths"],
    log_spot0: Float[Array, ""],
    mu: Float[Array, ""],
    rho: Float[Array, ""],
    qe_step,
    L_at_k: Callable[[Float[Array, " n_paths"]], Float[Array, " n_paths"]],
    t_mid: Float[Array, ""],
    dt: Float[Array, ""],
    sqrt_dt: Float[Array, ""],
    key: jax.Array,
    n_paths: int,
):
    """One SLV step using the midpoint-Euler scheme — shared by the
    calibration inner loop only.

    Kept local to this module rather than imported from
    ``slv_paths`` because the calibration inner loop carries an
    *evolving* leverage that does not yet form a complete
    ``LeverageGrid``; we accept any ``k -> L`` callable instead.
    """
    sqrt_1_minus_rho2 = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))
    k_v, k_perp, k_u = jax.random.split(key, 3)
    z_v = jax.random.normal(k_v, shape=(n_paths,))
    z_perp = jax.random.normal(k_perp, shape=(n_paths,))
    u = jax.random.uniform(k_u, shape=(n_paths,))
    z_s = rho * z_v + sqrt_1_minus_rho2 * z_perp

    v_next = qe_step(v, z_v, u)

    # k at the *start* of the step (state); the midpoint-time argument
    # only enters via the leverage query.
    k_now = log_S - (log_spot0 + mu * (t_mid - 0.5 * dt))
    L_n = L_at_k(k_now)
    sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
    sigma = L_n * sqrt_v

    log_S_next = (
        log_S
        + (mu - 0.5 * sigma * sigma) * dt
        + sigma * sqrt_dt * z_s
    )
    return log_S_next, v_next


# ─────────────────────────────────────────────────────────────────────
# Public calibration API
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
) -> tuple[Float[Array, "n_t n_k"], Float[Array, "n_t n_k"], Float[Array, ""]]:
    """Run the outer fixed point; return ``(L_K, L_{K-1}, clip_fraction)``."""
    # Import locally to avoid an import cycle (slv_paths imports SLVModel,
    # which conceptually sits "below" calibration).
    from valax.pricing.mc.slv_paths import _qe_variance_step_factory

    ridge_eff = jnp.asarray(ridge if method == "kernel" else 0.0)
    n_t = time_grid.shape[0]
    n_k = log_moneyness_grid.shape[0]

    mu = heston.rate - heston.dividend
    rho = heston.rho
    log_spot0 = jnp.log(spot)

    # Bandwidth resolver.
    if bandwidth is None:
        bw_fn = lambda t, kp: _silverman_bandwidth(kp)
    elif callable(bandwidth):
        bw_fn = bandwidth
    else:
        bw_const = jnp.asarray(bandwidth)
        bw_fn = lambda t, kp: bw_const

    # Pre-compute Dupire targets on the output grid: σ²_loc(k, t_i)
    # for every (i, j). Done once, outside the outer iteration.
    def _dupire_row(t):
        return jax.vmap(lambda kk: dupire_local_vol(surface, kk, t))(
            log_moneyness_grid
        )
    sigma_loc_target = jax.vmap(_dupire_row)(time_grid)  # (n_t, n_k)
    sigma2_loc_target = sigma_loc_target * sigma_loc_target

    # Warm-start at L ≡ 1 (Heston limit). Subsequent outer iterations
    # re-simulate with the previous iteration's leverage grid.
    L_values = jnp.ones((n_t, n_k), dtype=spot.dtype)
    L_previous = L_values
    clipped = jnp.zeros((n_t, n_k), dtype=bool)

    for outer in range(n_iterations):
        outer_key = jax.random.fold_in(key, outer)

        # Initialise particles at t = 0.
        log_S = jnp.broadcast_to(log_spot0, (n_paths,))
        v = jnp.broadcast_to(heston.v0, (n_paths,))
        t_prev = jnp.array(0.0, dtype=spot.dtype)

        # We rebuild ``L_values`` row by row in time. The leverage
        # callable consumed inside each step uses the *previous outer
        # iteration's* values (snapshot frozen at the top of this loop).
        L_grid_prev = LeverageGrid(
            log_moneyness_grid=log_moneyness_grid,
            time_grid=time_grid,
            values=L_values,
        )

        new_rows, clip_rows = [], []
        for i in range(n_t):
            t_i = time_grid[i]
            dt_i = t_i - t_prev
            sqrt_dt_i = jnp.sqrt(dt_i)
            t_mid = t_prev + 0.5 * dt_i

            qe_step = _qe_variance_step_factory(
                heston.kappa, heston.theta, heston.xi, dt_i,
            )

            def L_at_k(k_arr, _t=t_mid, _grid=L_grid_prev):
                return jax.vmap(lambda kk: _grid(kk, _t))(k_arr)

            step_key = jax.random.fold_in(outer_key, i)
            log_S, v = _propagate_one_step(
                log_S, v, log_spot0, mu, rho, qe_step,
                L_at_k, t_mid, dt_i, sqrt_dt_i, step_key, n_paths,
            )

            # Particles at t_i: estimate E[V | k] on the output k-grid.
            k_particles = log_S - (log_spot0 + mu * t_i)
            h = bw_fn(t_i, k_particles)
            ev_given_k = _nadaraya_watson(
                log_moneyness_grid,
                k_particles,
                v,
                bandwidth=h,
                ridge=ridge_eff,
            )

            # L²(k, t_i) = σ²_Dupire(k, t_i) / Ê[V | k].
            ev_safe = jnp.maximum(ev_given_k, 1e-12)
            L_raw = jnp.sqrt(
                jnp.maximum(sigma2_loc_target[i] / ev_safe, 0.0)
            )
            clip_rows.append((L_raw > L_max) | (L_raw < L_min))
            new_rows.append(jnp.clip(L_raw, L_min, L_max))

            t_prev = t_i

        L_previous = L_values
        L_values = jnp.stack(new_rows, axis=0)
        clipped = jnp.stack(clip_rows, axis=0)

    return L_values, L_previous, jnp.mean(clipped.astype(spot.dtype))


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
    ridge: float = 1e-3,
    L_max: float = 5.0,
    L_min: float = 0.05,
    settings: SolverSettings | None = None,
    name: str = "slv_leverage",
) -> CalibrationResult:
    """Calibrate the SLV leverage function ``L(k, t)``.

    **How the fixed point is reported.** There is no residual to drive to
    zero — the leverage is the output of ``n_iterations`` passes of a Monte
    Carlo fixed point ``L_{k+1} = G(L_k)`` — so the level is *explicit*
    (``theta=None``): its value is an unrolled, differentiable map of the
    inputs (Heston model, target surface, spot), and its risk is taken by
    plain reverse-mode AD through the passes, never by the implicit function
    theorem (``problem="fixed_point"``). Its diagnostics are those of the
    fixed point (:func:`~valax.calibration.result.fixed_point_diagnostics`):

    * ``residuals`` — the last increment ``L_K - L_{K-1}`` on the grid;
    * ``converged`` — ``max|L_K - L_{K-1}| <= settings.atol``, and always
      ``False`` for ``n_iterations=1`` (a single pass measures only the
      distance from the ``L ≡ 1`` warm start);
    * ``extras["clip_fraction"]`` — the share of grid cells whose raw
      leverage hit ``[L_min, L_max]`` in the last pass (a high value means
      the estimator is degenerate somewhere, whatever the increment says).

    Each pass re-simulates the swarm with fresh randomness, so the increment
    is bounded below by Monte Carlo noise; ``settings.atol`` must sit above
    that floor (see :data:`SLV_SETTINGS`).

    Args:
        heston: Pass-1 calibrated ``HestonModel``.
        surface: Implied-vol surface with ``total_variance(k, T)`` —
            the Dupire target.
        spot: Spot at calibration date ``S_0``.
        log_moneyness_grid: Output grid in ``k = log(K/F(T))``. Must
            be sorted ascending. The grid is the leverage's eventual
            ``log_moneyness_grid``.
        time_grid: Output grid in time (year fractions). Must be sorted
            ascending; ``time_grid[0]`` should be > 0 (Dupire is
            singular at 0).
        n_paths: Particle count.
        key: PRNG key.
        method: ``"particle"`` (no regularisation, ``ridge`` ignored)
            or ``"kernel"`` (Tikhonov ridge of size ``ridge``).
        n_iterations: Outer fixed-point passes (default 2, the minimum that
            measures convergence). ``1`` is the classical one-shot particle
            method — usable, but always reported not converged.
        bandwidth: Gaussian-kernel bandwidth. ``None`` (default) →
            Silverman's rule per slice. Scalar or callable
            ``(t, k_particles) -> h`` to override.
        ridge: Ridge regulariser for ``method="kernel"``. Ignored when
            ``method="particle"``.
        L_max: Upper clip on ``L`` to prevent runaway values when the
            kernel density is degenerate in a region with strong skew.
        L_min: Lower clip on ``L`` (same rationale as ``L_max``).
        settings: Fixed-point tolerance (``atol``); defaults to
            :data:`SLV_SETTINGS`.
        name: Level name used in diagnostics.

    Returns:
        The fixed-point :class:`~valax.calibration.result.CalibrationResult`;
        the :class:`LeverageGrid` of shape
        ``(len(time_grid), len(log_moneyness_grid))`` is ``result.value``.

    Raises:
        ValueError: If ``method`` is unknown or ``n_iterations < 1``.
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
            ridge=ridge, L_max=L_max, L_min=L_min,
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


def calibrate_slv(
    strikes: Float[Array, " n"],
    market_prices: Float[Array, " n"],
    spot: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    expiry: Float[Array, ""],
    pricing_fn: Callable,
    surface: Any,
    log_moneyness_grid: Float[Array, " n_k"],
    time_grid: Float[Array, " n_t"],
    n_paths: int,
    key: jax.Array,
    *,
    method: EstimatorName = "particle",
    n_iterations: int = 2,
    bandwidth: Float[Array, ""] | Callable | None = None,
    ridge: float = 1e-3,
    heston_initial_guess: HestonModel | None = None,
    heston_settings: SolverSettings | None = None,
    settings: SolverSettings | None = None,
    name: str = "slv",
) -> CalibrationResult:
    """End-to-end two-pass SLV calibration.

    Pass 1: :func:`~valax.calibration.heston.calibrate_heston` against the
    vanilla quotes (a least-squares level).
    Pass 2: :func:`calibrate_slv_leverage` against the Dupire surface (a
    fixed-point level).

    The returned result is pass 2's, with its value replaced by the
    assembled :class:`SLVModel` and pass 1's result attached as its child —
    so ``result.check()`` covers both passes and ``result.converged`` is
    ``False`` if either failed.

    Args:
        strikes: Forwarded to ``calibrate_heston``.
        market_prices: Forwarded to ``calibrate_heston``.
        spot: Forwarded to both passes.
        rate: Forwarded to ``calibrate_heston``.
        dividend: Forwarded to ``calibrate_heston``.
        expiry: Forwarded to ``calibrate_heston``.
        pricing_fn: Forwarded to ``calibrate_heston``.
        surface: Forwarded to ``calibrate_slv_leverage``.
        log_moneyness_grid: Forwarded to ``calibrate_slv_leverage``.
        time_grid: Forwarded to ``calibrate_slv_leverage``.
        n_paths: Forwarded to ``calibrate_slv_leverage``.
        key: Forwarded to ``calibrate_slv_leverage``.
        method: Forwarded to ``calibrate_slv_leverage``.
        n_iterations: Forwarded to ``calibrate_slv_leverage``.
        bandwidth: Forwarded to ``calibrate_slv_leverage``.
        ridge: Forwarded to ``calibrate_slv_leverage``.
        heston_initial_guess: Forwarded to ``calibrate_heston``.
        heston_settings: Solver settings of pass 1.
        settings: Fixed-point tolerance of pass 2.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the
        calibrated :class:`SLVModel` is ``result.value`` and the Heston fit
        is ``result.children[0]``.
    """
    from valax.calibration.heston import calibrate_heston
    from valax.models.slv import SLVModel  # lazy import — see note at top of file

    heston = calibrate_heston(
        strikes=strikes,
        market_prices=market_prices,
        spot=spot,
        rate=rate,
        dividend=dividend,
        expiry=expiry,
        pricing_fn=pricing_fn,
        initial_guess=heston_initial_guess,
        settings=heston_settings,
        name=f"{name}.heston",
    )
    leverage = calibrate_slv_leverage(
        heston.value,
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
        settings=settings,
        name=f"{name}.leverage",
    )
    model = SLVModel.from_heston_and_leverage(heston.value, surface, leverage.value)
    return dataclasses.replace(leverage, value=model, children=(heston,))
