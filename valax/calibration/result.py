"""The one return type of every calibration: value, level and diagnostics.

A :class:`CalibrationResult` carries everything the rest of the stack needs
from a calibration, and nothing it would otherwise have to rebuild:

* ``value`` — the calibrated object (a ``CurveGraph``, a vol surface, a model);
* ``theta``, ``inputs``, ``residual_fn`` — the level ``R(theta; inputs) = 0``
  exactly as the solver drove it to zero, so the implicit-function-theorem
  adjoint differentiates *the solved system* and never a re-implementation;
* ``assemble_fn`` — the map ``(theta, inputs) -> value``, which is how a
  cotangent on the calibrated object is pulled back to ``theta`` (and, for
  objects that also read their inputs directly, to the inputs);
* ``diagnostics`` — residuals, iteration count and a **traceable**
  ``converged`` flag, so failure is visible inside ``jit`` / ``vmap``;
* ``labels`` — names of the level's nodes (smile expiries, cube cells, strip
  buckets), so a failed slice / cube / strip is reported *by name*;
* ``provenance`` — the pricing engine the level was calibrated on, which is
  load-bearing when that engine carries a modelling bias (see
  :class:`CalibrationProvenance` and :func:`check_engine`);
* ``children`` — sub-results of a composite calibration (e.g. the Heston fit
  inside an SLV calibration), checked together with the parent.

Levels with no solve (an explicit map such as installing a quoted spot, a flat
rate, or a Dupire transform) are the degenerate case ``theta = None``,
``residual_fn = None``: the whole value is ``assemble_fn(None, inputs)`` and
plain reverse-mode AD differentiates it.

Batched levels — one independent solve per node, ``vmap``-ed — report
per-node diagnostics: ``converged``, ``max_abs_residual``, ``rmse`` and
``n_steps`` carry a leading node axis aligned with ``labels``.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Bool, Float, Int
from jax import Array

from valax.calibration.level import (
    CalibrationLevel,
    LevelInputs,
    condition_alert,
    level_condition_number,
)
from valax.calibration.settings import SolverSettings


#: Level kinds recorded in :attr:`CalibrationResult.problem`.
LEVEL_KINDS = ("root", "least_squares", "explicit", "fixed_point")

#: Engine kinds recorded in :attr:`CalibrationProvenance.engine_kind`.
ENGINE_KINDS = ("exact", "discretised", "modelling_bias")


class CalibrationError(RuntimeError):
    """Raised by :meth:`CalibrationResult.check` for a failed calibration."""


class CalibrationProvenance(eqx.Module):
    """The pricing engine a calibration level was fitted on.

    Model parameters are properties of the model, not of a method — *when*
    the calibration engine is exact (closed forms such as Jamshidian), or
    approximate only through a discretisation that converges away. For an
    engine with a **modelling** bias (frozen-``y`` Cheyette, which replaces
    the stochastic accumulator by its deterministic limit), the fitted
    parameters are "the parameters that make *that engine* reprice the
    basket", valid on that engine and nowhere else. The provenance is then
    load-bearing, and :func:`check_engine` refuses a mismatched engine.

    Provenance is metadata on the result, not a field of the model: models
    fitted on different engines have the same type and stack under ``vmap``.

    Attributes:
        method: Pricing method the basket was priced with (``"analytic"``,
            ``"pde"``, ``"mc"``, ``"lattice"``, or a descriptive engine name
            such as ``"cheyette_frozen_y"``).
        engine_kind: ``"exact"``, ``"discretised"`` or ``"modelling_bias"``.
        config: The engine configuration (must be hashable; every VALAX
            config is an all-static ``eqx.Module``), or ``None``.
    """

    method: str = eqx.field(static=True)
    engine_kind: str = eqx.field(static=True, default="exact")
    config: Any = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Reject unknown engine kinds and unhashable configs."""
        if self.engine_kind not in ENGINE_KINDS:
            raise ValueError(
                f"CalibrationProvenance: engine_kind must be one of "
                f"{ENGINE_KINDS}, got {self.engine_kind!r}."
            )
        try:
            hash(self.config)
        except TypeError as exc:
            raise TypeError(
                "CalibrationProvenance: config must be hashable (an "
                "all-static eqx.Module); got "
                f"{type(self.config).__name__}."
            ) from exc


class CalibrationDiagnostics(eqx.Module):
    """Convergence report of one calibration level.

    All members are arrays so the report survives ``jit`` and ``vmap``: a
    batch of per-trade calibrations returns a batch of ``converged`` flags
    rather than failing the whole trace. For a *batched* level (one solve per
    node) every member carries a leading node axis.

    Attributes:
        residuals: Final residual vector (the solved residual for exact
            levels, the fit residual ``r`` for least-squares levels, the last
            fixed-point increment for fixed-point levels). Empty for explicit
            levels.
        max_abs_residual: ``max|residuals|`` (zero for explicit levels).
        rmse: Root-mean-square of ``residuals``.
        n_steps: Solver iterations consumed (``-1`` if not reported, ``0``
            for explicit levels; fixed-point passes for fixed-point levels).
        converged: Whether the solver reported success.
        extras: Level-specific auxiliary diagnostics (arrays keyed by name),
            e.g. the fraction of a leverage grid that hit its clip bounds.
            Reported, never used to decide convergence.
    """

    residuals: Float[Array, "*nodes n_residuals"]
    max_abs_residual: Float[Array, "*nodes"]
    rmse: Float[Array, "*nodes"]
    n_steps: Int[Array, "*nodes"]
    converged: Bool[Array, "*nodes"]
    extras: dict = eqx.field(default_factory=dict)


def diagnostics_from_solution(
    solution: optx.Solution,
    residuals: Float[Array, " n_residuals"],
) -> CalibrationDiagnostics:
    """Summarise an ``optimistix`` solution and its final residuals.

    ``vmap``-safe: called per node inside a batched solve it yields the
    batched diagnostics of :func:`~valax.calibration.chain.solve_batched_level`.

    Args:
        solution: The ``optimistix.Solution`` returned by the solve.
        residuals: The residual vector evaluated at ``solution.value``.

    Returns:
        The traceable :class:`CalibrationDiagnostics`.
    """
    residuals = jnp.atleast_1d(jnp.asarray(residuals))
    n_steps = solution.stats.get("num_steps", jnp.asarray(-1))
    return CalibrationDiagnostics(
        residuals=residuals,
        max_abs_residual=jnp.max(jnp.abs(residuals)),
        rmse=jnp.sqrt(jnp.mean(residuals ** 2)),
        n_steps=jnp.asarray(n_steps, dtype=jnp.int32),
        converged=solution.result == optx.RESULTS.successful,
    )


def explicit_diagnostics() -> CalibrationDiagnostics:
    """Diagnostics of a level with no solve: trivially converged.

    Returns:
        A :class:`CalibrationDiagnostics` with empty residuals.
    """
    return CalibrationDiagnostics(
        residuals=jnp.zeros((0,)),
        max_abs_residual=jnp.asarray(0.0),
        rmse=jnp.asarray(0.0),
        n_steps=jnp.asarray(0, dtype=jnp.int32),
        converged=jnp.asarray(True),
    )


def fixed_point_diagnostics(
    increment: Float[Array, "..."],
    *,
    n_iterations: int,
    tol: float,
    extras: dict | None = None,
) -> CalibrationDiagnostics:
    """Diagnostics of a fixed-point iteration ``x_{k+1} = G(x_k)``.

    The residual is the last increment ``x_K - x_{K-1}``; the level is
    reported converged when ``max|increment| <= tol``. A single iteration has
    no meaningful increment (it measures the distance from the warm start),
    so ``n_iterations < 2`` is always reported *not* converged.

    Args:
        increment: ``x_K - x_{K-1}`` (any shape; flattened).
        n_iterations: Number of fixed-point passes performed (static).
        tol: Convergence tolerance on ``max|increment|``.
        extras: Auxiliary diagnostics to attach.

    Returns:
        The traceable :class:`CalibrationDiagnostics`.
    """
    residuals = jnp.ravel(jnp.asarray(increment))
    max_abs = jnp.max(jnp.abs(residuals))
    converged = (max_abs <= tol) & jnp.asarray(n_iterations >= 2)
    return CalibrationDiagnostics(
        residuals=residuals,
        max_abs_residual=max_abs,
        rmse=jnp.sqrt(jnp.mean(residuals ** 2)),
        n_steps=jnp.asarray(n_iterations, dtype=jnp.int32),
        converged=converged,
        extras=dict(extras or {}),
    )


class CalibrationResult(eqx.Module):
    """Calibrated object plus the level and diagnostics that produced it.

    Build with :func:`valax.calibration.chain.solve_level` (one solve),
    :func:`valax.calibration.chain.solve_batched_level` (one solve per node),
    or :func:`valax.calibration.chain.explicit_level` (no solve) rather than
    directly.

    Attributes:
        value: The calibrated object.
        theta: Solved unknowns at the solution, or ``None`` for an explicit
            level.
        inputs: The :class:`~valax.calibration.level.LevelInputs` the level
            was calibrated against.
        diagnostics: Convergence report.
        residual_fn: ``R(theta, inputs)`` driven to zero, or ``None`` for an
            explicit level. For least-squares fits this is the first-order
            condition (gradient of the half sum of squares). Static.
        assemble_fn: ``(theta, inputs) -> value``. Static.
        settings: The solver settings used. Static.
        name: Human-readable level name, used in alerts and errors. Static.
        labels: Node names (smile expiries, cube cells, strip buckets). For
            a batched level they index the leading axis of the diagnostics;
            for a joint level they index equal-sized blocks of the residual
            vector. Empty when the level has no node structure. Static.
        problem: Level kind, one of :data:`LEVEL_KINDS`. Static.
        provenance: Engine the level was calibrated on, or ``None`` when
            the level is engine-free (curves, closed-form vol fits). Static.
        children: Sub-results of a composite calibration, checked together
            with this one.
    """

    value: Any
    theta: Any
    inputs: LevelInputs
    diagnostics: CalibrationDiagnostics
    residual_fn: Callable[[Any, Any], Float[Array, " n"]] | None = eqx.field(
        static=True
    )
    assemble_fn: Callable[[Any, Any], Any] = eqx.field(static=True)
    settings: SolverSettings = eqx.field(static=True, default=SolverSettings())
    name: str = eqx.field(static=True, default="calibration")
    labels: tuple[str, ...] = eqx.field(static=True, default=())
    problem: str = eqx.field(static=True, default="root")
    provenance: CalibrationProvenance | None = eqx.field(
        static=True, default=None
    )
    children: tuple = ()

    @property
    def is_explicit(self) -> bool:
        """``True`` for a level with no solve (``theta is None``)."""
        return self.residual_fn is None

    @property
    def converged(self) -> Bool[Array, ""]:
        """Traceable ``all(converged)`` over nodes and children."""
        flags = [jnp.all(self.diagnostics.converged)]
        flags += [c.converged for c in self.children]
        return jnp.all(jnp.stack(flags))

    @property
    def level(self) -> CalibrationLevel | None:
        """The level as a :class:`~valax.calibration.level.CalibrationLevel`.

        ``None`` for an explicit level, which has nothing to invert.
        """
        if self.residual_fn is None:
            return None
        return CalibrationLevel(
            residual_fn=self.residual_fn, theta=self.theta, inputs=self.inputs
        )

    def condition_number(self) -> Float[Array, ""]:
        """Condition number of ``dR/dtheta`` at the solution (eager).

        For a batched level ``dR/dtheta`` is block-diagonal and this is the
        ratio of its extreme singular values over *all* blocks.

        Returns:
            The 2-norm condition number; ``1.0`` for an explicit level.
        """
        level = self.level
        if level is None:
            return jnp.asarray(1.0)
        return level_condition_number(level)

    def node_max_abs_residuals(self) -> np.ndarray:
        """Per-node ``max|residual|`` aligned with :attr:`labels` (eager).

        Returns:
            One value per label, or an empty array when the level has no
            labels or its residuals do not split evenly across them.
        """
        n = len(self.labels)
        if n == 0:
            return np.zeros((0,))
        res = np.abs(np.asarray(self.diagnostics.residuals))
        if res.ndim >= 2 and res.shape[0] == n:
            return res.reshape(n, -1).max(axis=1)
        flat = res.ravel()
        if flat.size == 0 or flat.size % n:
            return np.zeros((0,))
        return flat.reshape(n, -1).max(axis=1)

    def failed_labels(self, max_residual: float | None = None) -> tuple[str, ...]:
        """Names of the nodes that failed (eager).

        A node fails when its own solve did not converge (batched levels) or
        when its residual exceeds ``max_residual``. For a joint level whose
        single solve failed, nodes are attributed by residual: those above
        ``max_residual`` (default ``1e3 * settings.atol`` for exact levels),
        or else the worst node.

        Args:
            max_residual: Optional bound on each node's ``max|residual|``.

        Returns:
            Failed node labels, in label order.
        """
        if not self.labels:
            return ()
        conv = np.asarray(self.diagnostics.converged)
        node_res = self.node_max_abs_residuals()
        failed = np.zeros(len(self.labels), dtype=bool)
        batched = conv.ndim >= 1 and conv.shape[0] == len(self.labels)
        if batched:
            failed |= ~conv.astype(bool).reshape(len(self.labels), -1).all(1)
        if node_res.size:
            failed |= ~np.isfinite(node_res)
            bound = max_residual
            if bound is None and not batched and not bool(conv.all()):
                if self.problem == "root":
                    bound = 1e3 * max(self.settings.atol, 1e-14)
            if bound is not None:
                failed |= node_res > bound
            if not batched and not bool(conv.all()) and not failed.any():
                failed[int(np.nanargmax(node_res))] = True
        return tuple(l for l, f in zip(self.labels, failed) if f)

    def check(
        self,
        *,
        max_residual: float | None = None,
        max_condition_number: float | None = None,
    ) -> "CalibrationResult":
        """Raise if the calibration failed; return ``self`` otherwise.

        Eager: call outside ``jit``. Inside a trace, read
        :attr:`converged` instead. Children are checked first; a failed
        node of a labelled level is named in the error.

        Args:
            max_residual: Optional bound on ``max|residuals|``. Useful for
                exact levels, whose residual must vanish; leave ``None`` for
                least-squares fits, whose residual is a fit error.
            max_condition_number: Optional bound on the condition number of
                ``dR/dtheta``.

        Returns:
            ``self``, so ``result = calibrate(...).check()`` chains.

        Raises:
            CalibrationError: On solver failure, residual above
                ``max_residual``, or an ill-conditioned level.
        """
        for child in self.children:
            child.check(max_residual=max_residual)
        diag = self.diagnostics
        if not bool(jnp.all(diag.converged)):
            nodes = self.failed_labels(max_residual)
            where = f" at node(s) {list(nodes)}" if nodes else ""
            raise CalibrationError(
                f"{self.name}: {self.problem} level did not converge{where} "
                f"(max|residual| = {float(jnp.max(diag.max_abs_residual)):.3e}, "
                f"steps = {int(jnp.max(diag.n_steps))}, "
                f"settings = {self.settings})."
            )
        if max_residual is not None:
            worst = float(jnp.max(diag.max_abs_residual))
            if worst > max_residual:
                nodes = self.failed_labels(max_residual)
                where = f" at node(s) {list(nodes)}" if nodes else ""
                raise CalibrationError(
                    f"{self.name}: max|residual| = {worst:.3e} exceeds "
                    f"{max_residual:.3e}{where}."
                )
        if max_condition_number is not None:
            alert = condition_alert(
                self.condition_number(),
                level_name=self.name,
                threshold=max_condition_number,
            )
            if alert is not None:
                raise CalibrationError(alert)
        return self


def check_engine(
    result: CalibrationResult,
    method: str,
    config: Any = None,
) -> CalibrationResult:
    """Refuse to use a modelling-bias calibration on a different engine.

    Implements the provenance policy: an engine with a **modelling** bias
    (``engine_kind == "modelling_bias"``, e.g. frozen-``y`` Cheyette) makes
    the fitted parameters engine-bound, so pricing with another method or
    config is an error. For exact and discretised engines the provenance is
    audit metadata and any engine is accepted.

    Eager (call before pricing, outside ``jit``); ``price()`` itself does not
    consult provenance.

    Args:
        result: A calibration result (its children are checked too).
        method: The pricing method about to be used.
        config: The pricing config about to be used.

    Returns:
        ``result``, unchanged.

    Raises:
        CalibrationError: On an engine mismatch for a modelling-bias level.
    """
    for child in result.children:
        check_engine(child, method, config)
    prov = result.provenance
    if prov is None or prov.engine_kind != "modelling_bias":
        return result
    if method != prov.method or config != prov.config:
        raise CalibrationError(
            f"{result.name}: calibrated on engine {prov.method!r} "
            f"(config {prov.config!r}), which carries a modelling bias, so its "
            f"parameters are only valid on that engine; refusing to price "
            f"with {method!r} (config {config!r})."
        )
    return result


__all__ = [
    "CalibrationDiagnostics",
    "CalibrationError",
    "CalibrationProvenance",
    "CalibrationResult",
    "ENGINE_KINDS",
    "LEVEL_KINDS",
    "check_engine",
    "diagnostics_from_solution",
    "explicit_diagnostics",
    "fixed_point_diagnostics",
]
