"""Solver settings and the one solver factory every calibration uses.

Every VALAX calibration is one of three numerical problems:

* ``"root"`` — a square system ``R(theta; inputs) = 0`` (curve bootstraps,
  quote conversions, exact model bootstraps);
* ``"least_squares"`` — an over-determined fit ``min ||r(theta; inputs)||^2``
  (SABR / SVI smiles, Heston, model baskets);
* ``"minimise"`` — a general scalar objective.

:class:`SolverSettings` names the algorithm and its tolerances, and
:func:`run_solver` is the single place that turns those settings into an
``optimistix`` call. Calibrators never build solvers themselves, so changing a
tolerance, an iteration cap or the algorithm is a settings change rather than a
code change, and every calibration reports convergence the same way (see
:class:`valax.calibration.result.CalibrationDiagnostics`).

All fields are static: settings vary once per run, not per trade, and a solver
choice changes the traced program anyway.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import optax
import optimistix as optx


#: Problem kinds understood by :func:`run_solver`.
PROBLEMS = ("root", "least_squares", "minimise")

#: Algorithms admitted per problem kind. ``"auto"`` resolves to the first.
_METHODS: dict[str, tuple[str, ...]] = {
    "root": ("newton", "chord", "levenberg_marquardt", "dogleg"),
    "least_squares": (
        "levenberg_marquardt",
        "gauss_newton",
        "dogleg",
        "bfgs",
        "optax_adam",
    ),
    "minimise": ("bfgs", "nonlinear_cg", "optax_adam"),
}


class SolverSettings(eqx.Module):
    """Algorithm, tolerances and iteration budget of one calibration solve.

    Attributes:
        method: Algorithm name, or ``"auto"`` for the default of the problem
            kind (Newton for ``"root"``, Levenberg-Marquardt for
            ``"least_squares"``, BFGS for ``"minimise"``).
        rtol: Relative tolerance passed to the ``optimistix`` solver.
        atol: Absolute tolerance passed to the ``optimistix`` solver.
        max_steps: Iteration cap.
        throw: Whether ``optimistix`` raises on failure. Defaults to
            ``False`` so non-convergence is *reported* through
            :attr:`~valax.calibration.result.CalibrationDiagnostics.converged`
            and stays traceable; use
            :meth:`~valax.calibration.result.CalibrationResult.check` for an
            eager hard failure.
        learning_rate: Step size for ``"optax_adam"``; ignored otherwise.
    """

    method: str = eqx.field(static=True, default="auto")
    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-10)
    max_steps: int = eqx.field(static=True, default=256)
    throw: bool = eqx.field(static=True, default=False)
    learning_rate: float = eqx.field(static=True, default=1e-3)


def resolve_method(settings: SolverSettings, problem: str) -> str:
    """Resolve ``settings.method`` against a problem kind.

    Args:
        settings: The solver settings.
        problem: One of :data:`PROBLEMS`.

    Returns:
        The concrete algorithm name.

    Raises:
        ValueError: If ``problem`` is unknown or the method is not admitted
            for it.
    """
    if problem not in _METHODS:
        raise ValueError(
            f"Unknown problem kind {problem!r}; expected one of {PROBLEMS}."
        )
    admitted = _METHODS[problem]
    if settings.method == "auto":
        return admitted[0]
    if settings.method not in admitted:
        raise ValueError(
            f"Solver method {settings.method!r} is not available for a "
            f"{problem!r} problem; choose one of {admitted} or 'auto'."
        )
    return settings.method


def make_solver(settings: SolverSettings, problem: str) -> Any:
    """Build the ``optimistix`` solver named by ``settings``.

    Args:
        settings: The solver settings.
        problem: One of :data:`PROBLEMS`.

    Returns:
        An ``optimistix`` root finder, least-squares solver or minimiser.

    Raises:
        ValueError: If the method is not admitted for ``problem``.
    """
    method = resolve_method(settings, problem)
    rtol, atol = settings.rtol, settings.atol
    if method == "newton":
        return optx.Newton(rtol=rtol, atol=atol)
    if method == "chord":
        return optx.Chord(rtol=rtol, atol=atol)
    if method == "levenberg_marquardt":
        return optx.LevenbergMarquardt(rtol=rtol, atol=atol)
    if method == "gauss_newton":
        return optx.GaussNewton(rtol=rtol, atol=atol)
    if method == "dogleg":
        return optx.Dogleg(rtol=rtol, atol=atol)
    if method == "bfgs":
        return optx.BFGS(rtol=rtol, atol=atol)
    if method == "nonlinear_cg":
        return optx.NonlinearCG(rtol=rtol, atol=atol)
    # method == "optax_adam"
    return optx.OptaxMinimiser(
        optax.adam(settings.learning_rate), rtol=rtol, atol=atol
    )


def run_solver(
    problem: str,
    fn: Callable[[Any, Any], Any],
    y0: Any,
    args: Any,
    settings: SolverSettings,
) -> optx.Solution:
    """Run one calibration solve.

    The single entry point every VALAX calibration routes through. ``fn``
    follows the ``optimistix`` convention ``fn(y, args)``: a residual vector
    for ``"root"`` / ``"least_squares"``, a scalar for ``"minimise"``. The
    inputs that sensitivities are wanted for must travel in ``args`` (not a
    Python closure) so the implicit-function-theorem adjoint ``optimistix``
    attaches to the solve sees them as differentiable.

    Args:
        problem: One of :data:`PROBLEMS`.
        fn: The residual or objective.
        y0: Initial guess.
        args: Differentiable inputs forwarded to ``fn``.
        settings: Algorithm and tolerances.

    Returns:
        The ``optimistix.Solution``.

    Raises:
        ValueError: If ``problem`` or the method is not recognised.
    """
    solver = make_solver(settings, problem)
    kwargs = dict(
        args=args, max_steps=settings.max_steps, throw=settings.throw
    )
    if problem == "root":
        return optx.root_find(fn, solver, y0, **kwargs)
    if problem == "least_squares":
        return optx.least_squares(fn, solver, y0, **kwargs)
    return optx.minimise(fn, solver, y0, **kwargs)


__all__ = [
    "PROBLEMS",
    "SolverSettings",
    "make_solver",
    "resolve_method",
    "run_solver",
]
