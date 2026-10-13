"""Level builders: batched / labelled levels, provenance, fixed points.

Covers :mod:`valax.calibration.solve` and the reporting side of
:mod:`valax.calibration.result`: a batched level is one adjoint-able level
whose IFT sensitivities match autodiff through the solves and full
recalibration; a failed node is named; provenance is enforced only for
modelling-bias engines.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import (
    CalibrationError,
    CalibrationProvenance,
    LevelInputs,
    SolverSettings,
    check_engine,
    explicit_level,
    fixed_point_diagnostics,
    level_adjoint,
    solve_batched_level,
    solve_level,
)
from valax.pricing.pde.config import PDEConfig

jax.config.update("jax_enable_x64", True)

_X = jnp.array([0.0, 0.5, 1.0, 1.5])
_LS = SolverSettings(method="levenberg_marquardt", rtol=1e-12, atol=1e-12)


def _node_residual(theta, node_in):
    """Fit ``u * a * exp(b x)`` to four quotes (over-determined)."""
    q, u = node_in
    return u * theta[0] * jnp.exp(theta[1] * _X) - q


def _node_inputs(inputs):
    q = inputs.quotes
    n = q.shape[0]
    return q, jnp.broadcast_to(inputs.upstream, (n,))


def _assemble(theta, _inputs):
    return theta


def _quotes():
    a = jnp.array([1.0, 2.0, 0.5])
    b = jnp.array([0.1, -0.2, 0.3])
    clean = a[:, None] * jnp.exp(b[:, None] * _X[None, :])
    # Inconsistent noise so the fit residual (and Hessian curvature term)
    # does not vanish.
    noise = 1e-2 * jnp.array([[1, -1, 1, -1], [0.5, 0.2, -0.3, 0.1], [-1, 0, 1, 0]])
    return clean + noise


def _batched(q, u):
    return solve_batched_level(
        "toy", "least_squares", _node_residual,
        jnp.tile(jnp.array([1.0, 0.0]), (3, 1)),
        LevelInputs(quotes=q, upstream=u),
        _node_inputs, _assemble, _LS,
        labels=("n0", "n1", "n2"),
    )


def _price(theta):
    return jnp.sum(theta[:, 0] + theta[:, 1] ** 2)


class TestBatchedLevel:
    def test_converges_per_node(self):
        res = _batched(_quotes(), jnp.asarray(1.0))
        assert res.diagnostics.converged.shape == (3,)
        assert bool(jnp.all(res.diagnostics.converged))
        assert res.labels == ("n0", "n1", "n2")
        assert res.problem == "least_squares"
        res.check()

    def test_matches_unbatched_solves(self):
        q = _quotes()
        res = _batched(q, jnp.asarray(1.0))
        for i in range(3):
            one = solve_level(
                "one", "least_squares",
                lambda th, inp: _node_residual(th, (inp.quotes, inp.upstream)),
                jnp.array([1.0, 0.0]), LevelInputs(quotes=q[i], upstream=1.0),
                _assemble, _LS,
            )
            np.testing.assert_allclose(res.theta[i], one.theta, atol=1e-10)

    def test_adjoint_matches_autodiff_and_fd(self):
        q, u = _quotes(), jnp.asarray(1.3)
        res = _batched(q, u)
        theta_bar = jax.grad(_price)(res.theta)
        adj = level_adjoint(res.level, theta_bar)

        def pv(q_, u_):
            return _price(_batched(q_, u_).value)

        g_q, g_u = jax.grad(pv, argnums=(0, 1))(q, u)
        np.testing.assert_allclose(adj.inputs_bar.quotes, g_q, atol=1e-8)
        np.testing.assert_allclose(adj.inputs_bar.upstream, g_u, atol=1e-8)

        # Full-recalibration finite differences (h large enough to sit above
        # the solver's termination noise; the FD truncation error is ~1e-6).
        h = 1e-3
        e = jnp.zeros_like(q).at[1, 2].set(h)
        fd_q = (pv(q + e, u) - pv(q - e, u)) / (2 * h)
        fd_u = (pv(q, u + h) - pv(q, u - h)) / (2 * h)
        np.testing.assert_allclose(adj.inputs_bar.quotes[1, 2], fd_q, rtol=1e-5)
        np.testing.assert_allclose(adj.inputs_bar.upstream, fd_u, rtol=1e-5)

    def test_traceable_under_jit_and_vmap(self):
        q = _quotes()
        jitted = eqx.filter_jit(lambda q_: _batched(q_, jnp.asarray(1.0)).theta)
        np.testing.assert_allclose(
            jitted(q), _batched(q, jnp.asarray(1.0)).theta, atol=1e-12
        )
        batch = jnp.stack([q, q * 1.01])
        out = jax.vmap(lambda q_: _batched(q_, jnp.asarray(1.0)).theta)(batch)
        assert out.shape == (2, 3, 2)

    def test_label_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels"):
            solve_batched_level(
                "toy", "least_squares", _node_residual,
                jnp.zeros((3, 2)), LevelInputs(quotes=_quotes(), upstream=1.0),
                _node_inputs, _assemble, _LS, labels=("a", "b"),
            )


def _sqrt_residual(theta, q):
    return theta ** 2 - q


def _joint_residual(theta, inputs):
    return _sqrt_residual(theta, inputs.quotes)


class TestFailureNaming:
    """Anti-vacuity: an unsolvable node is reported, by name."""

    def test_batched_names_failed_node(self):
        q = jnp.array([[4.0], [-1.0], [9.0]])  # x^2 = -1 has no real root
        res = solve_batched_level(
            "sqrt", "root", _sqrt_residual, jnp.ones((3, 1)),
            LevelInputs(quotes=q), lambda inp: inp.quotes, _assemble,
            SolverSettings(method="newton", max_steps=40),
            labels=("2Y", "5Y", "10Y"),
        )
        np.testing.assert_array_equal(
            np.asarray(res.diagnostics.converged), [True, False, True]
        )
        assert not bool(res.converged)
        assert res.failed_labels() == ("5Y",)
        with pytest.raises(CalibrationError, match=r"5Y"):
            res.check()

    def test_joint_level_attributes_failure_by_residual(self):
        q = jnp.array([4.0, -1.0, 9.0])
        # Start the impossible component off 1.0 so Newton oscillates
        # (finite) instead of hitting x = 0 and poisoning the joint step.
        res = solve_level(
            "joint", "root", _joint_residual, jnp.array([1.0, 0.7, 1.0]),
            LevelInputs(quotes=q), _assemble,
            SolverSettings(method="newton", max_steps=40),
            labels=("b1", "b2", "b3"),
        )
        assert not bool(res.converged)
        assert res.failed_labels() == ("b2",)
        with pytest.raises(CalibrationError, match=r"b2"):
            res.check()

    def test_max_residual_names_node(self):
        res = _batched(_quotes(), jnp.asarray(1.0))
        worst = int(np.argmax(res.node_max_abs_residuals()))
        bound = float(np.sort(res.node_max_abs_residuals())[-2]) * 1.0001
        with pytest.raises(CalibrationError, match=res.labels[worst]):
            res.check(max_residual=bound)


class TestProvenance:
    def test_modelling_bias_is_enforced(self):
        cfg = PDEConfig(n_spot=101, n_time=50)
        prov = CalibrationProvenance(
            method="pde", engine_kind="modelling_bias", config=cfg
        )
        res = explicit_level("m", LevelInputs(quotes=jnp.zeros(1)),
                             lambda _t, _i: 1.0, provenance=prov)
        assert res.provenance == prov
        assert check_engine(res, "pde", PDEConfig(n_spot=101, n_time=50)) is res
        with pytest.raises(CalibrationError, match="modelling bias"):
            check_engine(res, "pde", PDEConfig(n_spot=201, n_time=50))
        with pytest.raises(CalibrationError):
            check_engine(res, "mc", cfg)

    def test_exact_engine_is_portable(self):
        prov = CalibrationProvenance(method="analytic", engine_kind="exact")
        res = explicit_level("m", LevelInputs(quotes=jnp.zeros(1)),
                             lambda _t, _i: 1.0, provenance=prov)
        assert check_engine(res, "pde", PDEConfig()) is res

    def test_validation(self):
        with pytest.raises(ValueError, match="engine_kind"):
            CalibrationProvenance(method="pde", engine_kind="approximate")
        with pytest.raises(TypeError, match="hashable"):
            CalibrationProvenance(method="pde", config=[1, 2])


class TestFixedPointAndChildren:
    def test_fixed_point_diagnostics(self):
        small = jnp.full((2, 3), 1e-6)
        assert bool(fixed_point_diagnostics(small, n_iterations=3, tol=1e-4).converged)
        assert not bool(
            fixed_point_diagnostics(small, n_iterations=1, tol=1e-4).converged
        )
        assert not bool(
            fixed_point_diagnostics(10 * small + 1e-3, n_iterations=3, tol=1e-4)
            .converged
        )

    def test_failed_child_fails_parent(self):
        bad = solve_level(
            "inner", "root", _joint_residual, jnp.ones(1),
            LevelInputs(quotes=jnp.array([-1.0])), _assemble,
            SolverSettings(method="newton", max_steps=20),
        )
        parent = explicit_level(
            "outer", LevelInputs(quotes=jnp.zeros(1)), lambda _t, _i: 0.0,
            children=(bad,),
        )
        assert not bool(parent.converged)
        with pytest.raises(CalibrationError, match="inner"):
            parent.check()
