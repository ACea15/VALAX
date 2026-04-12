"""Tests for Hull-White one-factor model: analytics and ZCB pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.hull_white import (
    HullWhiteModel,
    hw_B,
    hw_bond_price,
    hw_short_rate_variance,
    _instantaneous_forward,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


def _flat_curve(ref_date, rate, n_years=16):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=ref_date,
    )


@pytest.fixture
def flat_curve(ref_date):
    return _flat_curve(ref_date, 0.05)


@pytest.fixture
def model(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


# ── B function ───────────────────────────────────────────────────────

class TestHWB:
    def test_b_zero_at_tau_zero(self):
        assert float(hw_B(jnp.array(0.1), jnp.array(0.0))) == pytest.approx(0.0, abs=1e-15)

    def test_b_positive(self):
        assert float(hw_B(jnp.array(0.1), jnp.array(5.0))) > 0.0

    def test_b_limit_large_tau(self):
        """B(a, ∞) → 1/a."""
        a = jnp.array(0.10)
        b = hw_B(a, jnp.array(100.0))
        assert float(b) == pytest.approx(1.0 / float(a), rel=1e-4)

    def test_b_known_value(self):
        """B(0.1, 5) = (1 - exp(-0.5)) / 0.1."""
        expected = (1.0 - float(jnp.exp(-0.5))) / 0.1
        assert float(hw_B(jnp.array(0.1), jnp.array(5.0))) == pytest.approx(expected, rel=1e-10)


# ── Instantaneous forward ───────────────────────────────────────────

class TestInstantaneousForward:
    def test_flat_curve_forward(self, model):
        """On a flat 5% CC curve, f(0,t) = 0.05 for all t."""
        for t in [0.0, 1.0, 5.0, 10.0]:
            f = _instantaneous_forward(model, jnp.array(t))
            assert float(f) == pytest.approx(0.05, abs=1e-4)


# ── Exact-fit property ───────────────────────────────────────────────

class TestHWBondPrice:
    def test_exact_fit_flat_curve(self, model):
        """P_HW(0, T | r=f(0,0)) recovers the initial curve DF for flat curve."""
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        for T in [0.5, 1.0, 3.0, 5.0, 10.0]:
            p_hw = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(T))
            p_mkt = float(jnp.exp(-0.05 * T))
            assert float(p_hw) == pytest.approx(p_mkt, abs=1e-6)

    def test_exact_fit_steep_curve(self, ref_date):
        """Exact-fit holds on a non-flat curve (within tree tolerance)."""
        pillars = jnp.array(
            [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(16)],
            dtype=jnp.int32,
        )
        times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
        rates = 0.03 + 0.005 * times
        curve = DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.exp(-rates * times),
            reference_date=ref_date,
        )
        model = HullWhiteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.01),
            initial_curve=curve,
        )
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        for T in [1.0, 5.0, 10.0]:
            p_hw = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(T))
            rate = 0.03 + 0.005 * T
            p_mkt = float(jnp.exp(-rate * T))
            assert float(p_hw) == pytest.approx(p_mkt, abs=1e-4)

    def test_bond_price_positive(self, model):
        """Bond prices are always positive."""
        for r in [-0.01, 0.0, 0.05, 0.10, 0.15]:
            p = hw_bond_price(model, jnp.array(r), jnp.array(0.0), jnp.array(5.0))
            assert float(p) > 0.0

    def test_bond_price_decreasing_in_r(self, model):
        """Higher short rate → lower bond price."""
        prices = [
            float(hw_bond_price(model, jnp.array(r), jnp.array(0.0), jnp.array(5.0)))
            for r in [0.02, 0.04, 0.06, 0.08]
        ]
        assert all(p1 > p2 for p1, p2 in zip(prices, prices[1:]))

    def test_jit_compatible(self, model):
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        eager = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(5.0))
        jitted = jax.jit(hw_bond_price)(model, f0, jnp.array(0.0), jnp.array(5.0))
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_r(self, model):
        """dP/dr = -B * P (standard HW sensitivity)."""
        r = jnp.array(0.05)
        t = jnp.array(0.0)
        T = jnp.array(5.0)
        P = hw_bond_price(model, r, t, T)
        dPdr = jax.grad(hw_bond_price, argnums=1)(model, r, t, T)
        B = hw_B(model.mean_reversion, T - t)
        assert float(dPdr) == pytest.approx(-float(B) * float(P), rel=1e-8)


# ── Variance ─────────────────────────────────────────────────────────

class TestHWVariance:
    def test_variance_zero_at_zero(self, model):
        v = hw_short_rate_variance(model, jnp.array(0.0))
        assert float(v) == pytest.approx(0.0, abs=1e-15)

    def test_variance_known_value(self, model):
        """sigma^2/(2a) * (1 - exp(-2at))."""
        t = 5.0
        expected = 0.01**2 / (2 * 0.10) * (1.0 - float(jnp.exp(-2 * 0.10 * t)))
        assert float(hw_short_rate_variance(model, jnp.array(t))) == pytest.approx(expected, rel=1e-10)

    def test_variance_increasing(self, model):
        """Variance increases monotonically with time."""
        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        vars_ = [float(hw_short_rate_variance(model, jnp.array(t))) for t in times]
        assert all(v1 < v2 for v1, v2 in zip(vars_, vars_[1:]))
