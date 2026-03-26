"""Tests for parameter reparametrization transforms."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.sabr import SABRModel
from valax.calibration.transforms import (
    positive,
    bounded,
    unit_interval,
    correlation,
    model_to_unconstrained,
    unconstrained_to_model,
    SABR_TRANSFORMS,
)


class TestPositive:
    @pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 5.0, 25.0])
    def test_roundtrip(self, x):
        spec = positive()
        x = jnp.array(x)
        assert jnp.allclose(spec.from_unconstrained(spec.to_unconstrained(x)), x, atol=1e-6)

    def test_output_positive(self):
        spec = positive()
        for y in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert float(spec.from_unconstrained(jnp.array(y))) > 0.0

    def test_differentiable(self):
        spec = positive()
        grad = jax.grad(lambda y: spec.from_unconstrained(y))(jnp.array(1.0))
        assert jnp.isfinite(grad)


class TestBounded:
    @pytest.mark.parametrize("x", [0.1, 0.5, 2.0, 4.9])
    def test_roundtrip(self, x):
        spec = bounded(0.0, 5.0)
        x = jnp.array(x)
        assert jnp.allclose(spec.from_unconstrained(spec.to_unconstrained(x)), x, atol=1e-5)

    def test_output_in_bounds(self):
        spec = bounded(-2.0, 3.0)
        for y in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            val = float(spec.from_unconstrained(jnp.array(y)))
            assert -2.0 <= val <= 3.0


class TestCorrelation:
    @pytest.mark.parametrize("x", [-0.9, -0.5, 0.0, 0.5, 0.9])
    def test_roundtrip(self, x):
        spec = correlation()
        x = jnp.array(x)
        assert jnp.allclose(spec.from_unconstrained(spec.to_unconstrained(x)), x, atol=1e-6)

    def test_output_in_range(self):
        spec = correlation()
        for y in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            val = float(spec.from_unconstrained(jnp.array(y)))
            assert -1.0 <= val <= 1.0

    def test_differentiable(self):
        spec = correlation()
        grad = jax.grad(lambda y: spec.from_unconstrained(y))(jnp.array(0.0))
        assert jnp.isfinite(grad)


class TestModelRoundtrip:
    def test_sabr_roundtrip(self):
        model = SABRModel(
            alpha=jnp.array(0.3),
            beta=jnp.array(0.5),
            rho=jnp.array(-0.3),
            nu=jnp.array(0.4),
        )
        raw = model_to_unconstrained(model, SABR_TRANSFORMS)
        recovered = unconstrained_to_model(raw, SABR_TRANSFORMS, model)
        assert jnp.allclose(recovered.alpha, model.alpha, atol=1e-6)
        assert jnp.allclose(recovered.beta, model.beta, atol=1e-5)
        assert jnp.allclose(recovered.rho, model.rho, atol=1e-6)
        assert jnp.allclose(recovered.nu, model.nu, atol=1e-6)

    def test_partial_transforms(self):
        """When beta is fixed, only alpha/rho/nu are transformed."""
        model = SABRModel(
            alpha=jnp.array(0.3),
            beta=jnp.array(0.5),
            rho=jnp.array(-0.3),
            nu=jnp.array(0.4),
        )
        partial = {k: v for k, v in SABR_TRANSFORMS.items() if k != "beta"}
        raw = model_to_unconstrained(model, partial)
        assert "beta" not in raw
        recovered = unconstrained_to_model(raw, partial, model)
        # beta comes from template
        assert float(recovered.beta) == float(model.beta)
        assert jnp.allclose(recovered.alpha, model.alpha, atol=1e-6)
