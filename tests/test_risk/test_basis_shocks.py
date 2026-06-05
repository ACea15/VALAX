"""Tests for multi-curve basis shock primitives."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.curves.multi_curve import MultiCurveSet
from valax.dates.daycounts import ymd_to_ordinal
from valax.risk.shocks import (
    bump_discount_curve,
    bump_forward_curve,
    parallel_basis_shift,
)


def _flat_curve(ref, rate=0.04):
    pillars = jnp.array([
        ref,
        ymd_to_ordinal(2027, 1, 1),
        ymd_to_ordinal(2028, 1, 1),
        ymd_to_ordinal(2031, 1, 1),
    ])
    times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
    )


@pytest.fixture
def multi_curve():
    ref = ymd_to_ordinal(2026, 1, 1)
    return MultiCurveSet(
        discount_curve=_flat_curve(ref, 0.03),  # OIS
        forward_curves={
            "3M": _flat_curve(ref, 0.04),
            "6M": _flat_curve(ref, 0.045),
        },
    )


class TestBumpDiscountCurve:
    def test_forward_curves_untouched(self, multi_curve):
        bumps = jnp.array([0.0, 0.01, 0.01, 0.01])  # +100 bp on OIS
        bumped = bump_discount_curve(multi_curve, bumps)
        for tenor in ("3M", "6M"):
            assert jnp.allclose(
                bumped.forward_curves[tenor].discount_factors,
                multi_curve.forward_curves[tenor].discount_factors,
                atol=1e-12,
            )

    def test_discount_curve_actually_bumped(self, multi_curve):
        bumps = jnp.array([0.0, 0.01, 0.01, 0.01])
        bumped = bump_discount_curve(multi_curve, bumps)
        # Discount factors at later pillars must shrink under positive rate bump
        diff = (
            bumped.discount_curve.discount_factors[1:]
            - multi_curve.discount_curve.discount_factors[1:]
        )
        assert jnp.all(diff < 0.0)


class TestBumpForwardCurve:
    def test_targeted_tenor_changed(self, multi_curve):
        bumps = jnp.array([0.0, 0.005, 0.005, 0.005])
        bumped = bump_forward_curve(multi_curve, "3M", bumps)
        assert not jnp.allclose(
            bumped.forward_curves["3M"].discount_factors,
            multi_curve.forward_curves["3M"].discount_factors,
            atol=1e-12,
        )

    def test_other_tenor_untouched(self, multi_curve):
        bumps = jnp.array([0.0, 0.005, 0.005, 0.005])
        bumped = bump_forward_curve(multi_curve, "3M", bumps)
        assert jnp.allclose(
            bumped.forward_curves["6M"].discount_factors,
            multi_curve.forward_curves["6M"].discount_factors,
            atol=1e-12,
        )

    def test_discount_curve_untouched(self, multi_curve):
        bumps = jnp.array([0.0, 0.005, 0.005, 0.005])
        bumped = bump_forward_curve(multi_curve, "3M", bumps)
        assert jnp.allclose(
            bumped.discount_curve.discount_factors,
            multi_curve.discount_curve.discount_factors,
            atol=1e-12,
        )

    def test_unknown_tenor_raises(self, multi_curve):
        with pytest.raises(KeyError):
            bump_forward_curve(multi_curve, "12M", jnp.zeros(4))


class TestParallelBasisShift:
    def test_matches_manual_bump(self, multi_curve):
        manual = bump_forward_curve(multi_curve, "3M", jnp.full(4, 0.0025))
        shifted = parallel_basis_shift(multi_curve, "3M", jnp.array(0.0025))
        assert jnp.allclose(
            shifted.forward_curves["3M"].discount_factors,
            manual.forward_curves["3M"].discount_factors,
            atol=1e-12,
        )

    def test_basis_alone_no_discount_move(self, multi_curve):
        """Pure basis shift should keep discount curve identical."""
        shifted = parallel_basis_shift(multi_curve, "3M", jnp.array(0.0025))
        assert jnp.allclose(
            shifted.discount_curve.discount_factors,
            multi_curve.discount_curve.discount_factors,
            atol=1e-12,
        )

    def test_grad_through_basis_shift(self, multi_curve):
        """Differentiability: PV of a far DF on the 3M curve depends on basis."""

        def pv(b):
            shifted = parallel_basis_shift(multi_curve, "3M", b)
            return shifted.forward_curves["3M"].discount_factors[-1]

        g = jax.grad(pv)(jnp.array(0.0))
        # Bumping the 3M curve up reduces its far DF
        assert g < 0.0
        assert jnp.isfinite(g)
