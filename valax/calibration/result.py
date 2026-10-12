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
  ``converged`` flag, so failure is visible inside ``jit`` / ``vmap``.

Levels with no solve (an explicit map such as installing a quoted spot, a flat
rate, or a Dupire transform) are the degenerate case ``theta = None``,
``residual_fn = None``: the whole value is ``assemble_fn(None, inputs)`` and
plain reverse-mode AD differentiates it.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
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


class CalibrationError(RuntimeError):
    """Raised by :meth:`CalibrationResult.check` for a failed calibration."""


class CalibrationDiagnostics(eqx.Module):
    """Convergence report of one calibration level.

    All members are arrays so the report survives ``jit`` and ``vmap``: a
    batch of per-trade calibrations returns a batch of ``converged`` flags
    rather than failing the whole trace.

    Attributes:
        residuals: Final residual vector (the solved residual for exact
            levels, the fit residual ``r`` for least-squares levels). Empty
            for explicit levels.
        max_abs_residual: ``max|residuals|`` (zero for explicit levels).
        rmse: Root-mean-square of ``residuals``.
        n_steps: Solver iterations consumed (``-1`` if not reported, ``0``
            for explicit levels).
        converged: Whether the solver reported success.
    """

    residuals: Float[Array, " n_residuals"]
    max_abs_residual: Float[Array, ""]
    rmse: Float[Array, ""]
    n_steps: Int[Array, ""]
    converged: Bool[Array, ""]


def diagnostics_from_solution(
    solution: optx.Solution,
    residuals: Float[Array, " n_residuals"],
) -> CalibrationDiagnostics:
    """Summarise an ``optimistix`` solution and its final residuals.

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


class CalibrationResult(eqx.Module):
    """Calibrated object plus the level and diagnostics that produced it.

    Build with :func:`valax.calibration.chain.solve_level` (solved levels) or
    :func:`valax.calibration.chain.explicit_level` (no solve) rather than
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

    @property
    def is_explicit(self) -> bool:
        """``True`` for a level with no solve (``theta is None``)."""
        return self.residual_fn is None

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

        Returns:
            The 2-norm condition number; ``1.0`` for an explicit level.
        """
        level = self.level
        if level is None:
            return jnp.asarray(1.0)
        return level_condition_number(level)

    def check(
        self,
        *,
        max_residual: float | None = None,
        max_condition_number: float | None = None,
    ) -> "CalibrationResult":
        """Raise if the calibration failed; return ``self`` otherwise.

        Eager: call outside ``jit``. Inside a trace, read
        ``diagnostics.converged`` instead.

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
        diag = self.diagnostics
        if not bool(jnp.all(diag.converged)):
            raise CalibrationError(
                f"{self.name}: solver did not converge "
                f"(max|residual| = {float(jnp.max(diag.max_abs_residual)):.3e}, "
                f"steps = {int(jnp.max(diag.n_steps))}, "
                f"settings = {self.settings})."
            )
        if max_residual is not None:
            worst = float(jnp.max(diag.max_abs_residual))
            if worst > max_residual:
                raise CalibrationError(
                    f"{self.name}: max|residual| = {worst:.3e} exceeds "
                    f"{max_residual:.3e}."
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


__all__ = [
    "CalibrationDiagnostics",
    "CalibrationError",
    "CalibrationResult",
    "diagnostics_from_solution",
    "explicit_diagnostics",
]
