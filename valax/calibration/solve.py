r"""Level builders: the single implementation of "run a calibration".

Every calibration in VALAX — chain node or standalone calibrator
(``calibrate_sabr``, ``calibrate_hull_white``, ``strip_caplet_vols``, …) —
packages its solve through one of four builders, so every level reports
convergence the same way and carries a residual that is, by construction, the
one the solver used:

* :func:`solve_level` — one solve (``"root"`` or ``"least_squares"``);
* :func:`solve_batched_level` — one independent solve per node, ``vmap``-ed,
  reported per node (slice / cube / strip loops);
* :func:`triangular_root_level` — a square lower-triangular system (a
  co-terminal bootstrap), solved bucket by bucket, reported per bucket;
* :func:`explicit_level` — no implicit solve (explicit maps, fixed points).

This module depends only on the solver settings, the level and the result
types — not on ``valax.core`` — so the per-model calibrators, which
``valax.core`` imports transitively, can use it without an import cycle.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import (
    CalibrationDiagnostics,
    CalibrationProvenance,
    CalibrationResult,
    diagnostics_from_solution,
    explicit_diagnostics,
)
from valax.calibration.settings import SolverSettings, run_solver


# ── Level builders: the contract every node factory uses ─────────────


def solve_level(
    name: str,
    problem: str,
    residual_fn: Callable[[Any, LevelInputs], Float[Array, " n"]],
    theta0: Any,
    inputs: LevelInputs,
    assemble_fn: Callable[[Any, LevelInputs], Any],
    settings: SolverSettings | None = None,
    *,
    labels: Sequence[str] = (),
    provenance: CalibrationProvenance | None = None,
) -> CalibrationResult:
    r"""Solve one calibration level and package it as a result.

    The single implementation of "run a calibration": every solved node
    routes through here, so every level reports convergence the same way and
    carries a residual that is, by construction, the one the solver used.

    * ``problem="root"`` — ``residual_fn`` is a square system ``R(theta) = 0``;
      the level residual is ``R`` itself.
    * ``problem="least_squares"`` — ``residual_fn`` returns the fit residual
      vector ``r(theta)`` (weights included); the level residual is the
      first-order condition :math:`\nabla_\theta \tfrac12\|r\|^2`, which is
      square, so the implicit function theorem applies with the full Hessian.

    Args:
        name: Level name for diagnostics.
        problem: ``"root"`` or ``"least_squares"``.
        residual_fn: ``(theta, inputs) -> residual vector``.
        theta0: Initial guess (any pytree of float arrays).
        inputs: The level's quotes / upstream market / fixed inputs.
        assemble_fn: ``(theta, inputs) -> calibrated object``.
        settings: Solver settings. Defaults to :class:`SolverSettings`.
        labels: Optional node names indexing equal-sized blocks of the fit
            residual (e.g. one per strip bucket), so a failure is reported by
            node.
        provenance: Pricing engine the level was calibrated on, if any.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`.

    Raises:
        ValueError: If ``problem`` is not ``"root"`` or ``"least_squares"``.
    """
    _check_problem("solve_level", problem)
    settings = SolverSettings() if settings is None else settings
    solution = run_solver(problem, residual_fn, theta0, inputs, settings)
    theta = solution.value
    fit_residuals = residual_fn(theta, inputs)
    diagnostics = diagnostics_from_solution(solution, fit_residuals)
    if problem == "root" and labels:
        # The residual at the starting point: when a joint solve diverges to
        # NaN its final residuals carry no information, but each quote's
        # distance from the initial guess still singles out an off-market
        # one (see CalibrationResult.failed_labels). One residual evaluation.
        diagnostics = dataclasses.replace(diagnostics, extras={
            "initial_residuals": jax.lax.stop_gradient(
                jnp.ravel(residual_fn(theta0, inputs))
            ),
        })

    return CalibrationResult(
        value=assemble_fn(theta, inputs),
        theta=theta,
        inputs=inputs,
        diagnostics=diagnostics,
        residual_fn=_level_residual(problem, residual_fn),
        assemble_fn=assemble_fn,
        settings=settings,
        name=name,
        labels=tuple(labels),
        problem=problem,
        provenance=provenance,
    )


def _check_problem(caller: str, problem: str) -> None:
    """Reject problem kinds a calibration level cannot be."""
    if problem not in ("root", "least_squares"):
        raise ValueError(
            f"{caller}: problem must be 'root' or 'least_squares', got "
            f"{problem!r}."
        )


def _level_residual(
    problem: str,
    residual_fn: Callable[[Any, Any], Float[Array, " n"]],
) -> Callable[[Any, Any], Float[Array, " n"]]:
    """The square level residual of a solved problem.

    ``"root"``: the residual itself. ``"least_squares"``: the first-order
    condition of the weighted fit, flattened to a vector so every level
    residual has the same shape contract whatever pytree ``theta`` is.
    """
    if problem == "root":
        return residual_fn

    def level_residual(th: Any, inp: Any) -> Float[Array, " n"]:
        def half_sse(t: Any) -> Float[Array, ""]:
            r = residual_fn(t, inp)
            return 0.5 * jnp.sum(r * r)

        return ravel_pytree(jax.grad(half_sse)(th))[0]

    return level_residual


def solve_batched_level(
    name: str,
    problem: str,
    node_residual_fn: Callable[[Any, Any], Float[Array, " n"]],
    theta0: Any,
    inputs: LevelInputs,
    node_inputs_fn: Callable[[LevelInputs], Any],
    assemble_fn: Callable[[Any, LevelInputs], Any],
    settings: SolverSettings | None = None,
    *,
    labels: Sequence[str],
    provenance: CalibrationProvenance | None = None,
    node_solve_fn: Callable[[Any, Any], tuple[Any, CalibrationDiagnostics]]
    | None = None,
    sub_labels: Sequence[str] = (),
) -> CalibrationResult:
    r"""Solve one independent problem per node and package them as one level.

    The shape of every slice / cube / strip loop: smile per expiry, SABR per
    cube cell, SVI per slice. Each node is solved separately — so a node's
    failure is *its own* ``converged`` flag and is reported by name — but the
    solves are ``vmap``-ed (one trace, one compile, whatever the node count)
    and the nodes form one level for the adjoint: the level residual stacks
    the per-node residuals (first-order conditions for fits), whose Jacobian
    in ``theta`` is block-diagonal.

    Per-node inputs are a *function* of the level inputs
    (``node_inputs_fn``), not a precomputed constant: forwards read from the
    upstream curves stay differentiable, so the adjoint carries the curve
    risk of every node.

    Args:
        name: Level name for diagnostics.
        problem: ``"root"`` or ``"least_squares"`` (per node).
        node_residual_fn: ``(theta_i, node_inputs_i) -> residual vector`` of
            one node.
        theta0: Initial guesses, every leaf with a leading node axis.
        inputs: The level's quotes / upstream market / fixed inputs.
        node_inputs_fn: ``inputs -> pytree`` of per-node inputs, every leaf
            with a leading node axis.
        assemble_fn: ``(theta, inputs) -> calibrated object`` (``theta``
            batched).
        settings: Solver settings (shared by all nodes).
        labels: One name per node (e.g. ``"1Yx5Y"``), required so failures
            are reported by name.
        provenance: Pricing engine the level was calibrated on, if any.
        node_solve_fn: Optional ``(theta0_i, node_inputs_i) -> (theta_i,
            diagnostics_i)`` replacing the default single ``run_solver``
            call per node — for a node that is itself solved in stages (a
            sequential cap strip). It must drive ``node_residual_fn`` to
            zero (or to its least-squares optimum).
        sub_labels: Names of the sub-nodes of every node, for a
            ``node_solve_fn`` that reports per-stage diagnostics (one entry
            per cap-strip bucket): its diagnostics then carry a trailing
            sub-node axis of this length, and the result's labels are the
            row-major product ``"<label> <sub_label>"`` (``"K=3.00% cap
            2Y"``), aligned with the flattened ``(node, sub_node)``
            diagnostics. A failure is then named down to the sub-node.

    Returns:
        A :class:`~valax.calibration.result.CalibrationResult` whose
        diagnostics carry a leading node axis (and a sub-node axis when
        ``sub_labels`` is given) aligned with its labels.

    Raises:
        ValueError: If ``problem`` is unknown, the number of labels does not
            match the node axis, or ``sub_labels`` does not match the
            sub-node axis of the diagnostics.
    """
    _check_problem("solve_batched_level", problem)
    settings = SolverSettings() if settings is None else settings
    labels = tuple(labels)
    n_nodes = jax.tree.leaves(theta0)[0].shape[0]
    if len(labels) != n_nodes:
        raise ValueError(
            f"solve_batched_level({name!r}): {len(labels)} labels for "
            f"{n_nodes} nodes."
        )
    node_inputs = node_inputs_fn(inputs)

    def solve_one(th0: Any, node_in: Any) -> tuple:
        sol = run_solver(problem, node_residual_fn, th0, node_in, settings)
        r = node_residual_fn(sol.value, node_in)
        return sol.value, diagnostics_from_solution(sol, r)

    solve = solve_one if node_solve_fn is None else node_solve_fn
    theta, diagnostics = jax.vmap(solve)(theta0, node_inputs)
    sub_labels = tuple(sub_labels)
    if sub_labels:
        sub_shape = jnp.shape(diagnostics.converged)[1:]
        if sub_shape != (len(sub_labels),):
            raise ValueError(
                f"solve_batched_level({name!r}): {len(sub_labels)} sub-labels "
                f"for per-node diagnostics of shape {sub_shape}."
            )
        labels = tuple(f"{l} {s}" for l in labels for s in sub_labels)
    node_level = _level_residual(problem, node_residual_fn)

    def level_residual(th: Any, inp: LevelInputs) -> Float[Array, " n"]:
        return jax.vmap(node_level)(th, node_inputs_fn(inp)).reshape(-1)

    return CalibrationResult(
        value=assemble_fn(theta, inputs),
        theta=theta,
        inputs=inputs,
        diagnostics=diagnostics,
        residual_fn=level_residual,
        assemble_fn=assemble_fn,
        settings=settings,
        name=name,
        labels=labels,
        problem=problem,
        provenance=provenance,
    )


def explicit_level(
    name: str,
    inputs: LevelInputs,
    assemble_fn: Callable[[Any, LevelInputs], Any],
    *,
    diagnostics: CalibrationDiagnostics | None = None,
    problem: str = "explicit",
    settings: SolverSettings | None = None,
    provenance: CalibrationProvenance | None = None,
    children: Sequence[CalibrationResult] = (),
    value: Any = None,
) -> CalibrationResult:
    """Package a level with no implicit solve (an explicit map of its inputs).

    Installing a quoted spot, building a flat curve from a quoted rate, or a
    Dupire transform of an upstream surface are levels whose value is a
    closed-form function of the inputs. They join the chain like any other
    level; the adjoint differentiates them by plain reverse-mode AD.

    A fixed-point iteration run for a fixed number of passes (the SLV
    leverage calibration) is also explicit in this sense — the value is an
    unrolled, differentiable map of the inputs — but it reports its own
    convergence: pass ``problem="fixed_point"`` and the
    :func:`~valax.calibration.result.fixed_point_diagnostics`.

    Args:
        name: Level name for diagnostics.
        inputs: The level's quotes / upstream market / fixed inputs.
        assemble_fn: ``(None, inputs) -> calibrated object``.
        diagnostics: Convergence report; defaults to trivially converged.
        problem: ``"explicit"`` or ``"fixed_point"``.
        settings: Settings recorded on the result (tolerances of a fixed
            point); defaults to :class:`SolverSettings`.
        provenance: Pricing engine the level was calibrated on, if any.
        children: Sub-results checked together with this level.
        value: Precomputed ``assemble_fn(None, inputs)``, to avoid running an
            expensive map twice; computed when ``None``.

    Returns:
        A :class:`~valax.calibration.result.CalibrationResult` with
        ``theta=None`` and no residual.
    """
    return CalibrationResult(
        value=assemble_fn(None, inputs) if value is None else value,
        theta=None,
        inputs=inputs,
        diagnostics=explicit_diagnostics() if diagnostics is None else diagnostics,
        residual_fn=None,
        assemble_fn=assemble_fn,
        settings=SolverSettings() if settings is None else settings,
        name=name,
        problem=problem,
        provenance=provenance,
        children=tuple(children),
    )


# ── Helpers shared by the slice / cube / strip loops ─────────────────


def stack_slices(
    strikes_per_node: Sequence[Float[Array, " n_k"]],
    values_per_node: Sequence[Float[Array, " n_k"]],
    weights_per_node: Sequence[Float[Array, " n_k"]] | None = None,
) -> tuple[
    Float[Array, "n_nodes n_max"],
    Float[Array, "n_nodes n_max"],
    Float[Array, "n_nodes n_max"],
]:
    """Stack ragged per-node smiles into rectangular arrays for a batched level.

    Short rows are padded by repeating their last strike and quote with
    **zero weight**, so a padded entry contributes exactly zero to its node's
    residual and the fit is unchanged. Eager (Python-level shapes).

    Args:
        strikes_per_node: One strike array per node.
        values_per_node: One quote array per node, same lengths.
        weights_per_node: Optional per-node weights; default ones.

    Returns:
        ``(strikes, values, weights)``, each ``(n_nodes, n_max)``.

    Raises:
        ValueError: If node counts or row lengths are inconsistent, or a row
            is empty.
    """
    n = len(strikes_per_node)
    if len(values_per_node) != n or (
        weights_per_node is not None and len(weights_per_node) != n
    ):
        raise ValueError("stack_slices: per-node sequences differ in length.")
    rows_k, rows_v, rows_w = [], [], []
    lengths = [int(jnp.shape(k)[0]) for k in strikes_per_node]
    if n == 0 or min(lengths) == 0:
        raise ValueError("stack_slices: every node needs at least one quote.")
    n_max = max(lengths)
    for i in range(n):
        k = jnp.asarray(strikes_per_node[i], dtype=jnp.float64)
        v = jnp.asarray(values_per_node[i], dtype=jnp.float64)
        if v.shape != k.shape:
            raise ValueError(
                f"stack_slices: node {i} has {k.shape[0]} strikes but "
                f"{v.shape[0]} quotes."
            )
        w = (
            jnp.ones_like(k)
            if weights_per_node is None
            else jnp.asarray(weights_per_node[i], dtype=jnp.float64)
        )
        pad = n_max - k.shape[0]
        rows_k.append(jnp.concatenate([k, jnp.full((pad,), k[-1])]))
        rows_v.append(jnp.concatenate([v, jnp.full((pad,), v[-1])]))
        rows_w.append(jnp.concatenate([w, jnp.zeros((pad,))]))
    return jnp.stack(rows_k), jnp.stack(rows_v), jnp.stack(rows_w)


def period_label(years: float) -> str:
    """Human-readable label for a year fraction: ``"3M"``, ``"5Y"``, ``"1.37Y"``.

    Used to name the nodes of a batched level (smile expiries, cube cells,
    strip buckets) so a failure is reported as ``"5Yx10Y"`` rather than as an
    array index. Year fractions are snapped to the nearest market tenor, since
    day counts make a "10Y" swap 10.14 years under Act/360: within half a
    month below one year, within 3% above. Eager.

    Args:
        years: Year fraction (a Python or concrete array scalar).

    Returns:
        The label.
    """
    with jax.ensure_compile_time_eval():
        t = float(years)
    months = t * 12.0
    if round(months) < 12 and abs(months - round(months)) <= 0.5 and t > 0:
        return f"{max(int(round(months)), 1)}M"
    if round(t) >= 1 and abs(t - round(t)) <= 0.03 * round(t):
        return f"{int(round(t))}Y"
    return f"{t:.3g}Y"


def period_labels(years: Any, prefix: str = "") -> tuple[str, ...]:
    """:func:`period_label` of every entry of an axis, never failing a trace.

    When the axis is traced (a caller ``jit``-ed over it), the labels fall
    back to positional ``"<prefix>#i"`` names: labels are diagnostics and
    must not make a calibration untraceable.

    Args:
        years: Year-fraction axis.
        prefix: Prepended to every label (e.g. ``"cap "``).

    Returns:
        One label per entry.
    """
    n = int(jnp.shape(years)[0])
    try:
        with jax.ensure_compile_time_eval():
            return tuple(prefix + period_label(t) for t in jnp.asarray(years))
    except jax.errors.ConcretizationTypeError:
        return tuple(f"{prefix}#{i}" for i in range(n))


# ── Triangular root levels ───────────────────────────────────────────


#: Default settings of a triangular level: Newton to 1e-12.
_TRIANGULAR_SETTINGS = SolverSettings(
    method="newton", rtol=1e-12, atol=1e-12, max_steps=100
)


def triangular_root_level(
    name: str,
    bucket_residual_fn: Callable[[int, Float[Array, " n"], LevelInputs], Float[Array, ""]],
    theta0: Float[Array, " n"],
    inputs: LevelInputs,
    assemble_fn: Callable[[Float[Array, " n"], LevelInputs], Any],
    settings: SolverSettings | None = None,
    *,
    labels: Sequence[str],
    provenance: CalibrationProvenance | None = None,
) -> CalibrationResult:
    r"""A square, lower-triangular root level, solved bucket by bucket.

    The level has one unknown and one equation per bucket, and equation
    :math:`k` reads only the unknowns :math:`\theta_{1..k}` — a co-terminal
    bootstrap (swaption :math:`k` sees the volatility buckets up to its
    expiry), a cap strip. Its Jacobian is therefore lower triangular, and the
    level is solved as the classical bootstrap: one scalar Newton root-find
    per bucket with the earlier buckets held fixed (passed in the solver's
    ``args``, so autodiff differentiates through every stage) —
    :math:`O(n)` bucket evaluations per sweep. Bucket :math:`k` starts from
    its own seed ``theta0[k]`` (detached).

    The result's level residual is the full triangular :math:`R(\theta)`, so
    the IFT adjoint is one triangular solve, and the diagnostics are per
    bucket, aligned with ``labels``: a bucket converged when its solve
    succeeded with a finite residual and every bucket before it converged; a
    failed bucket installs NaN — never a best-effort value — and
    ``extras["root_cause"]`` marks the first one, so
    :meth:`~valax.calibration.result.CalibrationResult.root_cause_labels`
    names it while the later buckets inherit the failure. Downstream of a
    failure every bucket is solved against finite stand-ins, so the healthy
    buckets before it keep their exact roots.

    Each bucket's *reported* residual is its own — evaluated with the buckets
    after it at finite stand-ins — because a later failure's NaN enters an
    earlier equation with weight exactly zero, and ``0 * NaN`` is NaN.

    A joint solver (one Newton solve of the whole system, Jacobian tagged
    lower triangular for ``lineax`` substitution) was built and measured
    against this one and removed: it costs :math:`O(n^2)` bucket
    evaluations per step (770 vs 108 pricings at :math:`n = 10`), compiles
    slower, never ran faster, and on an infeasible bucket it marks every
    bucket failed with the wrong root cause (see the D14 architecture note).

    Args:
        name: Level name for diagnostics.
        bucket_residual_fn: ``(k, theta, inputs) -> residual of bucket k``
            with ``k`` static; must depend on ``theta[:k + 1]`` only.
        theta0: Initial guess, one seed per bucket; bucket ``k`` starts from
            ``theta0[k]``, and the buckets after ``k`` sit at their seeds
            while ``k`` is solved (they do not enter its equation).
        inputs: The level inputs.
        assemble_fn: ``(theta, inputs) -> calibrated object``.
        settings: Solver settings (Newton); defaults to Newton at 1e-12.
        labels: One label per bucket.
        provenance: Pricing engine the level was calibrated on, if any.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`
        (``problem="root"``).

    Raises:
        ValueError: If the labels do not match the bucket count.
    """
    settings = _TRIANGULAR_SETTINGS if settings is None else settings
    theta0 = jnp.asarray(theta0)
    n = theta0.shape[0]
    labels = tuple(labels)
    if len(labels) != n:
        raise ValueError(
            f"triangular_root_level({name!r}): {len(labels)} labels for {n} "
            f"buckets."
        )

    def residual_fn(theta: Float[Array, " n"], inp: LevelInputs) -> Float[Array, " n"]:
        return jnp.stack([bucket_residual_fn(k, theta, inp) for k in range(n)])

    theta, diagnostics = _solve_sequential(
        bucket_residual_fn, theta0, inputs, settings
    )

    return CalibrationResult(
        value=assemble_fn(theta, inputs),
        theta=theta,
        inputs=inputs,
        diagnostics=diagnostics,
        residual_fn=residual_fn,
        assemble_fn=assemble_fn,
        settings=settings,
        name=name,
        labels=labels,
        problem="root",
        provenance=provenance,
    )


def _own_residuals(
    bucket_residual_fn: Callable,
    theta: Float[Array, " n"],
    theta0: Float[Array, " n"],
    inputs: LevelInputs,
) -> Float[Array, " n"]:
    """Each bucket's residual with the buckets *after* it at finite stand-ins.

    Equal to the level residual for a finite ``theta`` (bucket ``k`` reads
    ``theta[:k + 1]`` only). After a failure, the later buckets hold NaN, and
    although they enter an earlier bucket's equation with weight exactly zero,
    ``0 * NaN`` is NaN — so a healthy bucket would report a NaN residual.
    Replacing what bucket ``k`` must not read makes the report per bucket.
    """
    n = theta.shape[0]
    stand_in = jax.lax.stop_gradient(theta0)
    return jnp.stack([
        bucket_residual_fn(k, jnp.concatenate([theta[: k + 1], stand_in[k + 1:]]), inputs)
        for k in range(n)
    ])


def _bucket_diagnostics(
    final: Float[Array, " n"],
    converged: Any,
    root_cause: Any,
    n_steps: Any,
) -> CalibrationDiagnostics:
    """Per-bucket diagnostics of a triangular level."""
    return CalibrationDiagnostics(
        residuals=final[:, None],
        max_abs_residual=jnp.abs(final),
        rmse=jnp.abs(final),
        n_steps=jnp.asarray(n_steps, dtype=jnp.int32),
        converged=converged,
        extras={"root_cause": root_cause},
    )


def _solve_sequential(
    bucket_residual_fn: Callable,
    theta0: Float[Array, " n"],
    inputs: LevelInputs,
    settings: SolverSettings,
) -> tuple[Float[Array, " n"], CalibrationDiagnostics]:
    """The bootstrap: one scalar root per bucket, earlier buckets held fixed."""
    import optimistix as optx

    n = theta0.shape[0]
    known = theta0[:0]
    seeds = jax.lax.stop_gradient(theta0)
    converged, steps, root_cause = [], [], []
    upstream_ok = jnp.asarray(True)
    for k in range(n):

        def one_bucket(x, args, _k=k):
            prev, inp = args
            theta = jnp.concatenate([
                prev, jnp.reshape(x, (1,)),
                jax.lax.stop_gradient(theta0[_k + 1:]),
            ])
            return bucket_residual_fn(_k, theta, inp)

        # Downstream of a failed bucket the known values hold NaN, and a NaN
        # residual makes lineax raise at runtime even with throw=False. Solve
        # against a finite stand-in instead (detached; an identity on the
        # healthy path, so gradients are untouched) — the bucket is reported
        # failed and installs NaN regardless.
        prev = jnp.where(upstream_ok, known, jax.lax.stop_gradient(theta0[:k]))
        sol = run_solver("root", one_bucket, seeds[k], (prev, inputs), settings)
        ok = (
            upstream_ok
            & (sol.result == optx.RESULTS.successful)
            & jnp.isfinite(one_bucket(sol.value, (prev, inputs)))
        )
        # A failed bucket installs NaN, never a plausible-looking value; the
        # ``where`` is against a constant, so a converged bucket's gradient is
        # untouched.
        known = jnp.concatenate([known, jnp.reshape(jnp.where(ok, sol.value, jnp.nan), (1,))])
        converged.append(ok)
        root_cause.append(~ok & upstream_ok)
        upstream_ok = upstream_ok & ok
        steps.append(sol.stats.get("num_steps", jnp.asarray(-1)))

    return known, _bucket_diagnostics(
        _own_residuals(bucket_residual_fn, known, theta0, inputs),
        jnp.stack(converged), jnp.stack(root_cause), jnp.stack(steps),
    )


__all__ = [
    "explicit_level",
    "period_label",
    "period_labels",
    "solve_batched_level",
    "solve_level",
    "stack_slices",
    "triangular_root_level",
]
