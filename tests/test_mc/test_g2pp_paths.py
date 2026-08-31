"""Tests for the G2++ two-factor exact Monte-Carlo path generator.

The load-bearing check is the martingale/no-arbitrage identity
``E[D(0, T)] == P^M(0, T)`` (the exact scheme must reprice every zero-coupon
bond within Monte-Carlo error), plus the exact-fit initial state, the factor
conditional moments, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.g2pp import G2PPModel, g2pp_market_df, g2pp_factor_covariance
from valax.pricing.mc.g2pp_paths import generate_g2pp_paths, G2PPPathResult
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal


_ACT365 = "act_365"
_N_PATHS = 200_000
_N_STEPS = 60


@pytest.fixture(scope="module")
def ref_date():
    return int(ymd_to_ordinal(2025, 1, 1))


def _flat_curve(ref, rate=0.03, n_years=21):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(ref),
        day_count=_ACT365,
    )


def _steep_curve(ref, n_years=21):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    zero = 0.02 + 0.03 * (1.0 - jnp.exp(-0.3 * times))
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=jnp.int32(ref),
        day_count=_ACT365,
    )


@pytest.fixture
def model(ref_date):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(-0.70),
        initial_curve=_flat_curve(ref_date),
    )


class TestShape:
    def test_result_shapes(self, model):
        res = generate_g2pp_paths(model, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        assert res.factor_x.shape == (1000, _N_STEPS + 1)
        assert res.factor_y.shape == (1000, _N_STEPS + 1)
        assert res.short_rates.shape == (1000, _N_STEPS + 1)
        assert res.log_discount_factors.shape == (1000, _N_STEPS + 1)

    def test_initial_state(self, model):
        res = generate_g2pp_paths(model, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        # Factors start at zero; log-DF starts at zero; r(0) = phi(0) = f(0,0).
        assert jnp.allclose(res.factor_x[:, 0], 0.0)
        assert jnp.allclose(res.factor_y[:, 0], 0.0)
        assert jnp.allclose(res.log_discount_factors[:, 0], 0.0)
        assert float(res.short_rates[:, 0].std()) == pytest.approx(0.0, abs=1e-12)
        assert float(res.short_rates[0, 0]) == pytest.approx(0.03, abs=1e-4)


class TestZCBRepricing:
    """E[D(0, T)] must equal P^M(0, T) within Monte-Carlo error."""

    @pytest.mark.parametrize("T", [2.0, 5.0, 10.0])
    def test_flat_curve(self, model, T):
        res = generate_g2pp_paths(model, T=T, n_steps=_N_STEPS, n_paths=_N_PATHS, key=jax.random.PRNGKey(1))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(g2pp_market_df(model, jnp.array(T)))
        assert abs(mc - analytic) < 3.0 * stderr + 1e-4

    def test_steep_curve(self, ref_date):
        model = G2PPModel(
            mean_reversion_x=jnp.array(0.50), mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.010), volatility_y=jnp.array(0.008),
            correlation=jnp.array(-0.70), initial_curve=_steep_curve(ref_date),
        )
        res = generate_g2pp_paths(model, T=8.0, n_steps=96, n_paths=_N_PATHS, key=jax.random.PRNGKey(2))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(g2pp_market_df(model, jnp.array(8.0)))
        assert abs(mc - analytic) < 3.0 * stderr + 2e-4


class TestFactorMoments:
    def test_terminal_covariance(self, model):
        """Terminal (x, y) covariance matches the closed-form over [0, T]."""
        T = 3.0
        res = generate_g2pp_paths(model, T=T, n_steps=_N_STEPS, n_paths=_N_PATHS, key=jax.random.PRNGKey(3))
        x = res.factor_x[:, -1]
        y = res.factor_y[:, -1]
        # Factors start at zero so the [0, T] conditional covariance is the
        # unconditional terminal covariance.
        cov_closed = g2pp_factor_covariance(model, jnp.array(T))
        assert float(x.mean()) == pytest.approx(0.0, abs=5e-4)
        assert float(y.mean()) == pytest.approx(0.0, abs=5e-4)
        assert float(jnp.var(x)) == pytest.approx(float(cov_closed[0, 0]), rel=2e-2)
        assert float(jnp.var(y)) == pytest.approx(float(cov_closed[1, 1]), rel=2e-2)
        cov_xy = float(jnp.mean(x * y) - x.mean() * y.mean())
        assert cov_xy == pytest.approx(float(cov_closed[0, 1]), rel=3e-2)


class TestJIT:
    def test_jit(self, model):
        f = eqx.filter_jit(lambda m, k: generate_g2pp_paths(m, T=5.0, n_steps=_N_STEPS, n_paths=500, key=k))
        res = f(model, jax.random.PRNGKey(4))
        assert jnp.all(jnp.isfinite(res.log_discount_factors))
