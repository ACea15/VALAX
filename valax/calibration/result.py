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

import functools
from typing import Any, Callable

import equinox as eqx
import jax.errors as jax_errors
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

#: Failure attribution of a non-converged joint root level: the nodes whose
#: ``max|residual|`` is at least this fraction of the largest node residual
#: are named (:meth:`CalibrationResult.failed_labels`) — those within a factor
#: two of the worst. Relative rather than absolute because a failed joint
#: solve spreads residuals over every node; the cluster around the worst one
#: is where the bad quote sits.
FAILURE_CLUSTER_FRACTION = 0.5


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


#: Provenance of a fit priced by a closed form (Jamshidian for Hull-White,
#: the Brigo-Mercurio integral for G2++, Heston COS, Black-Scholes): exact, so
#: the parameters are portable to any engine.
CLOSED_FORM_PROVENANCE = CalibrationProvenance(
    method="analytic", engine_kind="exact"
)


@functools.singledispatch
def engine_provenance(
    model: Any, method: str, config: Any = None
) -> CalibrationProvenance:
    """The provenance of a fit whose basket is priced by ``(model, method, config)``.

    Dispatches on the model type, so a model whose engine carries a modelling
    bias registers its own rule (Cheyette's frozen-``y`` PDE does, in
    :mod:`valax.calibration.cheyette`). The default: ``"analytic"`` is a
    closed form (:data:`CLOSED_FORM_PROVENANCE`); every other method
    (``"pde"``, ``"mc"``, ``"lattice"``) is ``"discretised"`` with its config
    recorded — the bias converges away, so the provenance is audit metadata.

    Args:
        model: The model (or template) being calibrated.
        method: A concrete pricing method (resolve ``"auto"`` first).
        config: The pricing config of the basket.

    Returns:
        The :class:`CalibrationProvenance` to record on the result.
    """
    del model
    if method == "analytic":
        return CLOSED_FORM_PROVENANCE
    return CalibrationProvenance(
        method=method, engine_kind="discretised", config=config
    )


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
            Reported, never used to decide convergence. The key
            ``"root_cause"`` (bool, shaped like ``converged``) is understood
            by :meth:`CalibrationResult.root_cause_labels`: a sequential solve
            (a cap strip) marks the node whose *own* solve failed first, as
            opposed to the later nodes that inherit its failure.
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
    max_abs = jnp.max(jnp.abs(residuals))
    return CalibrationDiagnostics(
        residuals=residuals,
        max_abs_residual=max_abs,
        rmse=jnp.sqrt(jnp.mean(residuals ** 2)),
        n_steps=jnp.asarray(n_steps, dtype=jnp.int32),
        # A non-finite residual is never a solution, whatever the solver says.
        converged=(solution.result == optx.RESULTS.successful)
        & jnp.isfinite(max_abs),
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
            a batched level they index the node axes of the diagnostics —
            the leading axis, or several leading axes flattened row-major
            (a cap matrix strip is labelled per ``(strike, bucket)``, e.g.
            ``"K=3.00% cap 2Y"``, against ``(n_strikes, n_buckets)``
            diagnostics); for a joint level they index equal-sized blocks
            of the residual vector. Empty when the level has no node
            structure. Static.
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

    def _per_node(self, arr: Any) -> np.ndarray | None:
        """``arr`` as ``(n_labels, -1)`` when its leading axes are the node axes.

        The node axes are the leading axes whose sizes multiply to the number
        of labels — one axis for a slice / cube level, two (strike, bucket)
        for a cap matrix strip. ``None`` when no such prefix exists.
        """
        n = len(self.labels)
        a = np.asarray(arr)
        size = 1
        for k, d in enumerate(a.shape):
            size *= d
            if size == n:
                return a.reshape((n, -1))
            if size > n:
                break
        return None

    def _nonfinite_quote_mask(self) -> np.ndarray | None:
        """Per-label ``~isfinite(quote)`` when the labels name the quotes.

        Recognised when ``inputs.quotes`` is a ``QuoteSet`` holding every
        label as a key (a :func:`~valax.calibration.nodes.curve_graph_node`)
        or a flat vector with one entry per label (a
        :func:`~valax.curves.bootstrap_graph.bootstrap_curve_graph`, labelled
        ``"<i>:<Type>"`` in quote order); ``None`` otherwise.
        """
        n = len(self.labels)
        quotes = getattr(self.inputs, "quotes", None)
        values = getattr(quotes, "values", None)
        try:
            if isinstance(values, dict):
                if not all(label in values for label in self.labels):
                    return None
                arr = np.array(
                    [np.asarray(values[label], dtype=float).reshape(-1)
                     for label in self.labels]
                )
                if arr.shape != (n, 1):
                    return None
                arr = arr[:, 0]
            else:
                arr = np.asarray(quotes, dtype=float)
                if arr.ndim != 1 or arr.size != n:
                    return None
        except (TypeError, ValueError, jax_errors.TracerArrayConversionError):
            return None
        return ~np.isfinite(arr)

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
        per = self._per_node(res) if res.ndim >= 2 else None
        if per is not None and per.shape[1] > 0:
            return per.max(axis=1)
        flat = res.ravel()
        if flat.size == 0 or flat.size % n:
            return np.zeros((0,))
        return flat.reshape(n, -1).max(axis=1)

    def failed_labels(self, max_residual: float | None = None) -> tuple[str, ...]:
        """Names of the nodes that failed (eager).

        **Batched levels** (one solve per node): a node fails when its own
        solve did not converge, its residual is non-finite, or it exceeds
        ``max_residual``.

        **Joint levels** (one solve for every node, e.g. a curve build): a
        node is attributed in this order —

        1. if any node's *own quote* is non-finite (labels that name quotes:
           ``QuoteSet`` keys, or a flat quote vector aligned with the
           labels), exactly those nodes — a NaN quote poisons every residual
           of the joint solve, so residuals cannot localise it;
        2. with an explicit ``max_residual``, the nodes above it (and any
           non-finite node);
        3. otherwise, for a failed solve: for a root level the **worst
           cluster** — nodes whose residual is at least
           :data:`FAILURE_CLUSTER_FRACTION` of the largest, so one bad quote
           is not drowned in the residuals its failure spreads to the rest
           of the system; for a least-squares level the worst node. Partly
           non-finite residuals name the non-finite nodes. When *every*
           residual is non-finite (the solve diverged to NaN, e.g. an
           infeasible quote), the cluster is taken on the residuals at the
           starting point (``diagnostics.extras["initial_residuals"]``,
           recorded by :func:`~valax.calibration.solve.solve_level` for
           labelled root levels), where an off-market quote stands out — a
           heuristic, so treat it as "start here"; with no such record every
           node is named.

        Args:
            max_residual: Optional bound on each node's ``max|residual|``.

        Returns:
            Failed node labels, in label order. For a sequential solve, see
            :meth:`root_cause_labels` for the first failure.
        """
        labels = self.labels
        if not labels:
            return ()
        n = len(labels)
        conv = np.asarray(self.diagnostics.converged).astype(bool)
        node_res = self.node_max_abs_residuals()
        has_res = node_res.size == n
        nonfinite = ~np.isfinite(node_res) if has_res else np.zeros(n, bool)
        failed = np.zeros(n, dtype=bool)
        per_conv = self._per_node(conv) if conv.ndim >= 1 else None

        if per_conv is not None:  # batched: every node reports its own solve
            failed |= ~per_conv.all(axis=1) | nonfinite
            if max_residual is not None and has_res:
                failed |= node_res > max_residual
            return tuple(l for l, f in zip(labels, failed) if f)

        solve_failed = not bool(conv.all())
        if max_residual is not None and has_res:
            failed |= nonfinite | (node_res > max_residual)
        if solve_failed or failed.any():
            bad_quotes = self._nonfinite_quote_mask()
            if bad_quotes is not None and bad_quotes.any():
                return tuple(l for l, f in zip(labels, bad_quotes) if f)
        if solve_failed and has_res and not failed.any():
            evidence = node_res
            if nonfinite.all():
                # Diverged to NaN: the final residuals say nothing. Fall back
                # on the residuals at the starting point, where an
                # off-market quote stands out from the rest.
                initial = self._split_max(
                    self.diagnostics.extras.get("initial_residuals")
                    if isinstance(self.diagnostics.extras, dict) else None
                )
                if initial is None or not np.isfinite(initial).all():
                    return labels
                evidence = initial
            elif nonfinite.any():
                return tuple(l for l, f in zip(labels, nonfinite) if f)
            top = float(evidence.max())
            if self.problem == "root" and top > 0.0:
                failed |= evidence >= FAILURE_CLUSTER_FRACTION * top
            else:
                failed[int(np.argmax(evidence))] = True
        return tuple(l for l, f in zip(labels, failed) if f)

    def _split_max(self, residuals: Any) -> np.ndarray | None:
        """Per-label ``max|r|`` of a flat residual vector, or ``None``."""
        if residuals is None:
            return None
        flat = np.abs(np.asarray(residuals)).ravel()
        n = len(self.labels)
        if flat.size == 0 or flat.size % n:
            return None
        return flat.reshape(n, -1).max(axis=1)

    def root_cause_labels(
        self, max_residual: float | None = None
    ) -> tuple[str, ...]:
        """The failed nodes that *caused* the failure (eager).

        For a sequential solve — a cap strip, where bucket ``i`` reads the
        vols of buckets ``< i`` — a failed bucket installs NaN and every later
        bucket of its column inherits the failure. The level then records
        ``diagnostics.extras["root_cause"]``, and this returns only the nodes
        whose own solve failed first (one per strike column). For any other
        level it equals :meth:`failed_labels`.

        Args:
            max_residual: Forwarded to :meth:`failed_labels`.

        Returns:
            Root-cause labels, in label order.
        """
        failed = self.failed_labels(max_residual)
        extras = self.diagnostics.extras
        cause = extras.get("root_cause") if isinstance(extras, dict) else None
        if cause is None or not failed:
            return failed
        per = self._per_node(np.asarray(cause).astype(bool))
        if per is None:
            return failed
        roots = tuple(
            l for l, c in zip(self.labels, per.any(axis=1)) if c and l in failed
        )
        return roots or failed

    def _where(self, max_residual: float | None) -> str:
        """`` at node(s) [...]`` (plus the root cause) for an error message."""
        nodes = self.failed_labels(max_residual)
        if not nodes:
            return ""
        where = f" at node(s) {list(nodes)}"
        roots = self.root_cause_labels(max_residual)
        if roots != nodes:
            where += (
                f"; first failure(s) {list(roots)} — the later nodes of a "
                f"sequential solve inherit it"
            )
        return where

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
            where = self._where(max_residual)
            raise CalibrationError(
                f"{self.name}: {self.problem} level did not converge{where} "
                f"(max|residual| = {float(jnp.max(diag.max_abs_residual)):.3e}, "
                f"steps = {int(jnp.max(diag.n_steps))}, "
                f"settings = {self.settings})."
            )
        if max_residual is not None:
            worst = float(jnp.max(diag.max_abs_residual))
            if not worst <= max_residual:  # NaN-safe
                where = self._where(max_residual)
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
    "CLOSED_FORM_PROVENANCE",
    "CalibrationDiagnostics",
    "CalibrationError",
    "CalibrationProvenance",
    "CalibrationResult",
    "ENGINE_KINDS",
    "FAILURE_CLUSTER_FRACTION",
    "LEVEL_KINDS",
    "check_engine",
    "diagnostics_from_solution",
    "engine_provenance",
    "explicit_diagnostics",
    "fixed_point_diagnostics",
]
