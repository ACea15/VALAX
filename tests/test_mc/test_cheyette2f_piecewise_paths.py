"""Two-factor Cheyette Monte Carlo with piecewise σ₁(t) and ratio r(t).

Each step freezes σ_{r,i} = σ_{i,step}(1 + β_i x_i) with σ_{i,step}² the exact
step average of σ_i(t)², and correlates the shocks so the step covariance is
the exact average of ρσ₁σ₂. Oracles:

* β = 0, steps aligned with the breakpoints: the y accumulators are exact
  (they equal the Gaussian closed form to round-off);
* β = 0: a 3Y×5Y ATM payer swaption against the G2++ closed form with the same
  σ(t), r(t), κ₁ = a, κ₂ = b, ρ, within 2·stderr (measured 0.3 stderr);
* the martingale identity E[D(0, T)] = P^M(0, T) within 2·stderr, at β = 0
  and β ≠ 0;
* equal buckets against the constant model.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.cheyette import (
    CheyetteModel2F,
    cheyette_bond_price_2f,
    cheyette_market_df,
    cheyette_y_gaussian_2f,
)
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc.cheyette_paths import generate_cheyette_paths_2f

_ACT365 = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))
VOLS = (0.012, 0.008, 0.015)
RATIOS = (0.8, 0.6, 0.9)
TIMES = (1.0, 2.5)
T_EXPIRY = 3.0
N_STEPS = 150  # dt = 0.02: the breakpoints 1.0, 2.5 fall on the time grid
N_PATHS = 50_000


def _curve():
    years = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0)
    zeros = (0.020, 0.021, 0.022, 0.025, 0.027, 0.031, 0.034, 0.038, 0.040, 0.041)
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(zeros) * times),
        reference_date=jnp.int32(REF),
        day_count=_ACT365,
    )


CURVE = _curve()


def _cheyette(skew_1=0.0, skew_2=0.0, vols=VOLS, ratios=RATIOS, times=TIMES):
    return CheyetteModel2F(
        mean_reversion_1=jnp.asarray(0.5),
        mean_reversion_2=jnp.asarray(0.1),
        volatility_1=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        skew_1=jnp.asarray(skew_1),
        skew_2=jnp.asarray(skew_2),
        correlation=jnp.asarray(-0.7),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


def _g2pp():
    return G2PPModel(
        mean_reversion_x=jnp.asarray(0.5),
        mean_reversion_y=jnp.asarray(0.1),
        volatility_x=jnp.asarray(VOLS),
        volatility_ratio=jnp.asarray(RATIOS),
        correlation=jnp.asarray(-0.7),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(TIMES),
    )


def _atm_payer():
    expiry = REF + 365 * 3
    fixed = jnp.array([expiry + 365 * k for k in range(1, 6)], dtype=jnp.int32)
    ann = _annuity(jnp.int32(expiry), fixed, CURVE, _ACT365)
    strike = (CURVE(jnp.int32(expiry)) - CURVE(fixed[-1])) / ann
    return Swaption(
        expiry_date=jnp.int32(expiry), fixed_dates=fixed, strike=strike,
        notional=jnp.asarray(1.0), is_payer=True, day_count=_ACT365,
    )


@pytest.fixture(scope="module")
def gaussian_paths():
    return generate_cheyette_paths_2f(
        _cheyette(), T_EXPIRY, N_STEPS, N_PATHS, jax.random.PRNGKey(0)
    )


def _mean_and_stderr(x):
    return float(jnp.mean(x)), float(jnp.std(x) / jnp.sqrt(x.size))


class TestGaussianLimit:
    def test_accumulators_exact_on_aligned_grid(self, gaussian_paths):
        model = _cheyette()
        for i in (25, 50, 100, 125, 150):
            y = cheyette_y_gaussian_2f(model, jnp.asarray(i * T_EXPIRY / N_STEPS))
            np.testing.assert_allclose(np.asarray(gaussian_paths.aux_y11[:, i]), float(y[0, 0]), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(gaussian_paths.aux_y22[:, i]), float(y[1, 1]), rtol=1e-12)
            np.testing.assert_allclose(np.asarray(gaussian_paths.aux_y12[:, i]), float(y[0, 1]), rtol=1e-12)

    def test_swaption_matches_g2pp_closed_form(self, gaussian_paths):
        s = _atm_payer()
        model = _cheyette()
        pay_t = jnp.arange(4.0, 9.0)
        p = gaussian_paths

        def swap_value(x1, x2, y11, y22, y12):
            bonds = jax.vmap(
                lambda T: cheyette_bond_price_2f(
                    model, x1, x2, y11, y22, y12, jnp.asarray(T_EXPIRY), T
                )
            )(pay_t)
            return 1.0 - s.strike * jnp.sum(bonds) - bonds[-1]

        swap = jax.vmap(swap_value)(
            p.factor_x1[:, -1], p.factor_x2[:, -1],
            p.aux_y11[:, -1], p.aux_y22[:, -1], p.aux_y12[:, -1],
        )
        payoff = jnp.exp(p.log_discount_factors[:, -1]) * jnp.maximum(swap, 0.0)
        mean, stderr = _mean_and_stderr(payoff)
        analytic = float(g2pp_swaption_price(s, _g2pp()))
        assert abs(mean - analytic) < 2.0 * stderr

    def test_martingale(self, gaussian_paths):
        mean, stderr = _mean_and_stderr(jnp.exp(gaussian_paths.log_discount_factors[:, -1]))
        assert abs(mean - float(cheyette_market_df(_cheyette(), jnp.asarray(T_EXPIRY)))) < 2.0 * stderr


class TestSkewed:
    def test_martingale_with_skew(self):
        model = _cheyette(0.5, -0.3)
        p = generate_cheyette_paths_2f(model, 4.0, 100, N_PATHS, jax.random.PRNGKey(3))
        mean, stderr = _mean_and_stderr(jnp.exp(p.log_discount_factors[:, -1]))
        assert abs(mean - float(cheyette_market_df(model, jnp.asarray(4.0)))) < 2.0 * stderr

    def test_equal_buckets_match_constant(self):
        """Same normals: the per-step path reproduces the constant path."""
        const = _cheyette(0.5, -0.3, vols=(0.011,), ratios=(0.7,), times=())
        split = _cheyette(0.5, -0.3, vols=(0.011,) * 3, ratios=(0.7,) * 3)
        key = jax.random.PRNGKey(7)
        a = generate_cheyette_paths_2f(const, 3.0, 40, 512, key)
        b = generate_cheyette_paths_2f(split, 3.0, 40, 512, key)
        for name in ("factor_x1", "factor_x2", "aux_y11", "aux_y12", "log_discount_factors"):
            np.testing.assert_allclose(
                np.asarray(getattr(b, name)), np.asarray(getattr(a, name)),
                rtol=1e-11, atol=1e-16, err_msg=name,
            )

    def test_filter_jit_and_grad(self):
        """JIT smoke, and a finite bucket vega of a discount-factor mean."""
        key = jax.random.PRNGKey(1)

        @eqx.filter_jit
        def mean_df(model):
            p = generate_cheyette_paths_2f(model, 3.0, 30, 256, key)
            return jnp.mean(jnp.exp(p.log_discount_factors[:, -1]))

        model = _cheyette(0.5, -0.3)
        assert jnp.isfinite(mean_df(model))
        grad = eqx.filter_grad(mean_df)(model)
        assert np.all(np.isfinite(np.asarray(grad.volatility_1)))
        assert np.all(np.isfinite(np.asarray(grad.volatility_ratio)))
