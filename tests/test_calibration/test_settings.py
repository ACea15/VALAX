"""Solver settings, the shared solver factory, and calibration results."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from valax.calibration import (
    CalibrationError,
    LevelInputs,
    SolverSettings,
    make_solver,
    run_solver,
    solve_level,
)
from valax.market import QuoteSet


class TestSolverFactory:
    @pytest.mark.parametrize(
        "problem, method, cls",
        [
            ("root", "auto", optx.Newton),
            ("root", "chord", optx.Chord),
            ("least_squares", "auto", optx.LevenbergMarquardt),
            ("least_squares", "gauss_newton", optx.GaussNewton),
            ("least_squares", "bfgs", optx.BFGS),
            ("minimise", "auto", optx.BFGS),
            ("minimise", "optax_adam", optx.OptaxMinimiser),
        ],
    )
    def test_method_resolution(self, problem, method, cls):
        assert isinstance(make_solver(SolverSettings(method=method), problem), cls)

    def test_tolerances_are_forwarded(self):
        solver = make_solver(SolverSettings(rtol=1e-4, atol=1e-5), "root")
        assert solver.rtol == 1e-4 and solver.atol == 1e-5

    def test_inadmissible_method_rejected(self):
        with pytest.raises(ValueError, match="not available"):
            make_solver(SolverSettings(method="newton"), "least_squares")

    def test_unknown_problem_rejected(self):
        with pytest.raises(ValueError, match="Unknown problem"):
            make_solver(SolverSettings(), "integrate")

    def test_run_solver_root(self):
        sol = run_solver(
            "root", lambda y, a: y ** 2 - a, jnp.array(1.0), jnp.array(2.0),
            SolverSettings(),
        )
        assert float(sol.value) == pytest.approx(2.0 ** 0.5, abs=1e-10)


def _sqrt_level(target, settings=None):
    """Exact level ``theta^2 = q``."""
    inputs = LevelInputs(quotes=QuoteSet.from_dict({"Q": target}))
    return solve_level(
        "sqrt",
        "root",
        lambda th, inp: jnp.atleast_1d(th ** 2 - inp.quotes["Q"]),
        jnp.array(1.0),
        inputs,
        lambda th, inp: th,
        settings,
    )


class TestCalibrationResult:
    def test_converged_level(self):
        res = _sqrt_level(4.0).check(max_residual=1e-9)
        assert float(res.value) == pytest.approx(2.0)
        assert bool(res.diagnostics.converged)
        assert float(res.condition_number()) == pytest.approx(1.0)

    def test_unconvergeable_fit_is_reported(self):
        """Anti-vacuity: a fit that cannot converge must say so."""
        res = _sqrt_level(-1.0, SolverSettings(max_steps=20))
        assert not bool(res.diagnostics.converged)
        with pytest.raises(CalibrationError, match="did not converge"):
            res.check()

    def test_step_cap_is_honoured(self):
        res = _sqrt_level(4.0, SolverSettings(max_steps=1))
        assert not bool(res.diagnostics.converged)

    def test_converged_flag_is_traceable(self):
        """Under vmap a failed member flags itself without failing the batch."""
        flags = jax.vmap(lambda t: _sqrt_level(t).diagnostics.converged)(
            jnp.array([4.0, -1.0, 9.0])
        )
        assert flags.tolist() == [True, False, True]

    def test_jit_smoke(self):
        out = eqx.filter_jit(lambda t: _sqrt_level(t).value)(jnp.array(9.0))
        assert float(out) == pytest.approx(3.0)

    def test_least_squares_level_is_first_order_condition(self):
        """The level of a fit is the gradient of the half SSE: zero at the optimum."""
        inputs = LevelInputs(quotes=QuoteSet.from_dict({"a": 1.0, "b": 3.0}))
        res = solve_level(
            "mean",
            "least_squares",
            lambda th, inp: th - inp.quotes.vector(("a", "b")),
            jnp.array(0.0),
            inputs,
            lambda th, inp: th,
        )
        assert float(res.value) == pytest.approx(2.0)
        # Fit residuals are the data misfit, not zero...
        assert float(res.diagnostics.max_abs_residual) == pytest.approx(1.0)
        # ...but the level residual (stationarity) vanishes.
        assert float(jnp.max(jnp.abs(res.level.residual_fn(res.theta, inputs)))) < 1e-10
