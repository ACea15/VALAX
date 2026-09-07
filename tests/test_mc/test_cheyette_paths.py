"""Tests for the Cheyette (quasi-Gaussian) Monte-Carlo path generators.

The load-bearing check is the martingale/no-arbitrage identity
``E[D(0, T)] == P^M(0, T)`` — the integrating-factor scheme with exact-forward
discounting must reprice every zero-coupon bond within Monte-Carlo error on both
flat and steep curves, with and without a volatility skew — plus the exact-fit
initial state, convergence under time refinement, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.cheyette import (
    CheyetteModel,
    CheyetteModel2F,
    cheyette_market_df,
    cheyette_y_gaussian,
)
from valax.pricing.mc.cheyette_paths import (
    generate_cheyette_paths,
    generate_cheyette_paths_2f,
    CheyettePathResult,
    Cheyette2FPathResult,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal


_ACT365 = "act_365"
_N_PATHS = 200_000
_N_STEPS = 80


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
    return CheyetteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.010),
        skew=jnp.array(-3.0),
        initial_curve=_flat_curve(ref_date),
    )


@pytest.fixture
def model_2f(ref_date):
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.50),
        mean_reversion_2=jnp.array(0.10),
        volatility_1=jnp.array(0.010),
        volatility_2=jnp.array(0.008),
        skew_1=jnp.array(-2.0),
        skew_2=jnp.array(1.5),
        correlation=jnp.array(-0.70),
        initial_curve=_flat_curve(ref_date),
    )


class TestShape:
    def test_result_shapes_1f(self, model):
        res = generate_cheyette_paths(model, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        assert res.factor_x.shape == (1000, _N_STEPS + 1)
        assert res.aux_y.shape == (1000, _N_STEPS + 1)
        assert res.short_rates.shape == (1000, _N_STEPS + 1)
        assert res.log_discount_factors.shape == (1000, _N_STEPS + 1)

    def test_result_shapes_2f(self, model_2f):
        res = generate_cheyette_paths_2f(model_2f, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        for arr in (res.factor_x1, res.factor_x2, res.aux_y11, res.aux_y22, res.aux_y12,
                    res.short_rates, res.log_discount_factors):
            assert arr.shape == (1000, _N_STEPS + 1)

    def test_initial_state_1f(self, model):
        res = generate_cheyette_paths(model, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        assert jnp.allclose(res.factor_x[:, 0], 0.0)
        assert jnp.allclose(res.aux_y[:, 0], 0.0)
        assert jnp.allclose(res.log_discount_factors[:, 0], 0.0)
        assert float(res.short_rates[:, 0].std()) == pytest.approx(0.0, abs=1e-12)
        assert float(res.short_rates[0, 0]) == pytest.approx(0.03, abs=1e-4)

    def test_initial_state_2f(self, model_2f):
        res = generate_cheyette_paths_2f(model_2f, T=5.0, n_steps=_N_STEPS, n_paths=1000, key=jax.random.PRNGKey(0))
        assert jnp.allclose(res.factor_x1[:, 0], 0.0)
        assert jnp.allclose(res.factor_x2[:, 0], 0.0)
        assert jnp.allclose(res.log_discount_factors[:, 0], 0.0)
        assert float(res.short_rates[0, 0]) == pytest.approx(0.03, abs=1e-4)


class TestZCBRepricing:
    """E[D(0, T)] must equal P^M(0, T) within Monte-Carlo error."""

    @pytest.mark.parametrize("T", [2.0, 5.0, 10.0])
    def test_flat_curve_1f(self, model, T):
        res = generate_cheyette_paths(model, T=T, n_steps=_N_STEPS, n_paths=_N_PATHS, key=jax.random.PRNGKey(1))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(cheyette_market_df(model, jnp.array(T)))
        assert abs(mc - analytic) < 3.0 * stderr + 1e-4

    def test_steep_curve_1f(self, ref_date):
        model = CheyetteModel(
            mean_reversion=jnp.array(0.10), volatility=jnp.array(0.010),
            skew=jnp.array(-4.0), initial_curve=_steep_curve(ref_date),
        )
        res = generate_cheyette_paths(model, T=8.0, n_steps=96, n_paths=_N_PATHS, key=jax.random.PRNGKey(2))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(cheyette_market_df(model, jnp.array(8.0)))
        assert abs(mc - analytic) < 3.0 * stderr + 2e-4

    @pytest.mark.parametrize("T", [2.0, 5.0, 10.0])
    def test_flat_curve_2f(self, model_2f, T):
        res = generate_cheyette_paths_2f(model_2f, T=T, n_steps=_N_STEPS, n_paths=_N_PATHS, key=jax.random.PRNGKey(3))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(cheyette_market_df(model_2f, jnp.array(T)))
        assert abs(mc - analytic) < 3.0 * stderr + 1e-4

    def test_steep_curve_2f(self, ref_date):
        model = CheyetteModel2F(
            mean_reversion_1=jnp.array(0.50), mean_reversion_2=jnp.array(0.10),
            volatility_1=jnp.array(0.010), volatility_2=jnp.array(0.008),
            skew_1=jnp.array(-3.0), skew_2=jnp.array(2.0),
            correlation=jnp.array(-0.70), initial_curve=_steep_curve(ref_date),
        )
        res = generate_cheyette_paths_2f(model, T=8.0, n_steps=96, n_paths=_N_PATHS, key=jax.random.PRNGKey(4))
        df = jnp.exp(res.log_discount_factors[:, -1])
        mc = float(df.mean())
        stderr = float(df.std() / jnp.sqrt(_N_PATHS))
        analytic = float(cheyette_market_df(model, jnp.array(8.0)))
        assert abs(mc - analytic) < 3.0 * stderr + 2e-4


class TestGaussianLimitMoments:
    """With skew = 0 the accumulator y and factor variance are the closed-form
    Gaussian quantities."""

    def test_accumulator_deterministic_and_exact(self, ref_date):
        model = CheyetteModel(
            mean_reversion=jnp.array(0.20), volatility=jnp.array(0.011),
            skew=jnp.array(0.0), initial_curve=_flat_curve(ref_date),
        )
        T = 5.0
        res = generate_cheyette_paths(model, T=T, n_steps=_N_STEPS, n_paths=2000, key=jax.random.PRNGKey(5))
        # In the Gaussian limit the accumulator path is deterministic.
        assert float(res.aux_y[:, -1].std()) == pytest.approx(0.0, abs=1e-14)
        y_closed = float(cheyette_y_gaussian(model, jnp.array(T)))
        assert float(res.aux_y[0, -1]) == pytest.approx(y_closed, rel=1e-6)

    def test_factor_variance_matches_accumulator(self, ref_date):
        model = CheyetteModel(
            mean_reversion=jnp.array(0.20), volatility=jnp.array(0.011),
            skew=jnp.array(0.0), initial_curve=_flat_curve(ref_date),
        )
        T = 5.0
        res = generate_cheyette_paths(model, T=T, n_steps=_N_STEPS, n_paths=_N_PATHS, key=jax.random.PRNGKey(6))
        # Var[x(T)] == y(T) in the Gaussian limit.
        var_x = float(jnp.var(res.factor_x[:, -1]))
        y_closed = float(cheyette_y_gaussian(model, jnp.array(T)))
        assert var_x == pytest.approx(y_closed, rel=2e-2)


class TestConvergence:
    """Refining the time grid drives the martingale error below MC noise."""

    def test_zcb_error_shrinks_with_steps(self, model):
        T = 10.0
        errs = []
        for ns in [20, 160]:
            res = generate_cheyette_paths(model, T=T, n_steps=ns, n_paths=_N_PATHS, key=jax.random.PRNGKey(7))
            df = jnp.exp(res.log_discount_factors[:, -1])
            analytic = float(cheyette_market_df(model, jnp.array(T)))
            errs.append(abs(float(df.mean()) - analytic))
        # The fine grid is at least as accurate as the coarse one.
        assert errs[1] <= errs[0] + 5e-5


class TestJIT:
    def test_jit_1f(self, model):
        f = eqx.filter_jit(lambda m, k: generate_cheyette_paths(m, T=5.0, n_steps=_N_STEPS, n_paths=500, key=k))
        res = f(model, jax.random.PRNGKey(8))
        assert jnp.all(jnp.isfinite(res.log_discount_factors))

    def test_jit_2f(self, model_2f):
        f = eqx.filter_jit(lambda m, k: generate_cheyette_paths_2f(m, T=5.0, n_steps=_N_STEPS, n_paths=500, key=k))
        res = f(model_2f, jax.random.PRNGKey(9))
        assert jnp.all(jnp.isfinite(res.log_discount_factors))
