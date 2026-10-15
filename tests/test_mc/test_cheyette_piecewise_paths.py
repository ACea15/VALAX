"""One-factor Cheyette Monte Carlo with a piecewise-constant level σ(t).

Each step freezes :math:`\\sigma_r = \\sigma_{\\mathrm{step}}(1 + \\beta x)` with
:math:`\\sigma_{\\mathrm{step}}^2` the exact time average of σ(t)² over the
step. At β = 0 the accumulator recursion is then exact on steps inside one
bucket, so the simulated ``y`` equals the Gaussian-limit ``y(t)`` on the grid,
and swaption prices match Jamshidian with the same σ(t). Measured on four
200k-path seeds (pooled 800k): z = +1.00 (3Yx4Y receiver) and -1.26 (18Mx5Y
payer); per-seed z on the 3Yx4Y were (+3.31, -1.69, +0.07, +0.27), the same
pattern as for a constant σ = 1.05% (+3.11, -1.52, -0.08, +0.17), i.e. noise.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.cheyette import (
    CheyetteModel,
    cheyette_market_df,
    cheyette_y_gaussian,
)
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.mc import MCConfig
from valax.pricing.mc.cheyette_paths import generate_cheyette_paths

REF = int(ymd_to_ordinal(2026, 1, 1))
RATE = 0.035
KAPPA = 0.05
STEP_YEARS = (1.0, 2.0, 3.0, 4.0)
VOLS = (0.012, 0.008, 0.015, 0.006, 0.010)


def _curve() -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, 20)]
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-RATE * t),
        reference_date=jnp.int32(REF),
        day_count="act_365",
    )


CURVE = _curve()
MARKET = MarketState.single_curve(CURVE)


def _cheyette(skew, vols=VOLS, times=STEP_YEARS):
    return CheyetteModel(
        mean_reversion=jnp.asarray(KAPPA),
        volatility=jnp.asarray(vols),
        skew=jnp.asarray(skew),
        volatility_times=jnp.asarray(times),
    )


def _swaption(exp_days, n, payer):
    expiry = REF + exp_days
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array([expiry + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(RATE),
        notional=jnp.asarray(1.0),
        is_payer=payer,
        day_count="act_365",
    )


class TestPaths:
    def test_beta_zero_y_is_the_gaussian_accumulator(self):
        """Steps inside one bucket advance y exactly (breakpoints on the grid)."""
        model = bind_curves(_cheyette(0.0), CURVE)
        res = generate_cheyette_paths(model, T=5.0, n_steps=50, n_paths=8,
                                      key=jax.random.PRNGKey(0))
        times = jnp.linspace(0.0, 5.0, 51)
        expected = np.asarray(cheyette_y_gaussian(model, times))
        np.testing.assert_allclose(np.asarray(res.aux_y[0]), expected, rtol=1e-12, atol=1e-20)
        # y is deterministic at beta = 0: identical across paths.
        np.testing.assert_array_equal(np.asarray(res.aux_y[0]), np.asarray(res.aux_y[-1]))

    def test_straddling_steps_carry_the_step_variance(self):
        """With breakpoints mid-step, the step uses the averaged σ²."""
        model = bind_curves(_cheyette(0.0), CURVE)
        # dt = 5/7: every breakpoint falls strictly inside a step.
        res = generate_cheyette_paths(model, T=5.0, n_steps=7, n_paths=8,
                                      key=jax.random.PRNGKey(0))
        times = np.linspace(0.0, 5.0, 8)
        y_exact = np.asarray(cheyette_y_gaussian(model, jnp.asarray(times)))
        # Only the within-step weighting of σ² is approximated: O(κ dt).
        # Measured max 3.0e-3 relative; a left-point bucket lookup (σ at the
        # step start) misses by up to 30%.
        np.testing.assert_allclose(np.asarray(res.aux_y[0, 1:]), y_exact[1:], rtol=5e-3)

    def test_equal_buckets_equal_constant(self):
        const = bind_curves(_cheyette(0.5, vols=(0.011,), times=()), CURVE)
        split = bind_curves(_cheyette(0.5, vols=(0.011, 0.011), times=(1.5,)), CURVE)
        key = jax.random.PRNGKey(3)
        a = generate_cheyette_paths(const, T=3.0, n_steps=30, n_paths=64, key=key)
        b = generate_cheyette_paths(split, T=3.0, n_steps=30, n_paths=64, key=key)
        for fa, fb in [(a.factor_x, b.factor_x), (a.aux_y, b.aux_y),
                       (a.log_discount_factors, b.log_discount_factors)]:
            np.testing.assert_allclose(np.asarray(fb), np.asarray(fa), rtol=1e-12, atol=1e-18)

    def test_martingale_with_skew(self):
        """E[D(0, T)] = P^M(0, T) within 2 stderr at β = 0.5."""
        model = bind_curves(_cheyette(0.5), CURVE)
        res = generate_cheyette_paths(model, T=5.0, n_steps=50, n_paths=100_000,
                                      key=jax.random.PRNGKey(11))
        d = np.exp(np.asarray(res.log_discount_factors[:, -1]))
        stderr = d.std(ddof=1) / np.sqrt(d.size)
        target = float(cheyette_market_df(model, jnp.asarray(5.0)))
        assert abs(d.mean() - target) < 2.0 * stderr

    def test_filter_jit_smoke(self):
        model = bind_curves(_cheyette(0.5), CURVE)
        f = eqx.filter_jit(generate_cheyette_paths)
        res = f(model, 3.0, 12, 16, jax.random.PRNGKey(0))
        assert np.all(np.isfinite(np.asarray(res.log_discount_factors)))


class TestSwaptionVsJamshidian:
    @pytest.mark.parametrize(
        "exp_days,n,payer", [(1095, 4, False), (548, 5, True)],
        ids=["3Yx4Y_receiver", "18Mx5Y_payer"],
    )
    def test_beta_zero_within_two_stderr(self, exp_days, n, payer):
        sw = _swaption(exp_days, n, payer)
        hw = HullWhiteModel(
            mean_reversion=jnp.asarray(KAPPA), volatility=jnp.asarray(VOLS),
            volatility_times=jnp.asarray(STEP_YEARS),
        )
        jam = float(hw_swaption_price(sw, bind_curves(hw, CURVE)))
        prices, errs = [], []
        for seed in range(4):
            r = price(sw, _cheyette(0.0), MARKET, method="mc",
                      config=MCConfig(n_paths=200_000, n_steps=60),
                      key=jax.random.PRNGKey(seed))
            prices.append(float(r.price))
            errs.append(float(r.stderr))
        pooled = np.mean(prices)
        stderr = np.sqrt(np.sum(np.square(errs))) / len(errs)
        assert abs(pooled - jam) < 2.0 * stderr
