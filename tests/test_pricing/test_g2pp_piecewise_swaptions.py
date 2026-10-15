"""G2++ European swaptions with piecewise-constant σ(t) and ratio r(t).

The Brigo-Mercurio integral holds unchanged for deterministic time-dependent
volatilities; only the T-forward moments and the bond coefficients change.
Checked against independent routes:

* the exact Monte-Carlo engine (per-step exact covariance) within 2·stderr;
* payer-receiver parity against the curve (volatility-free);
* two equal buckets == the constant closed form;
* autodiff vega per bucket vs central finite differences, and exactly zero
  sensitivity to buckets that start after expiry.

QuantLib has no time-dependent two-factor G2 model, so there is no external
oracle; the PDE cross-check lives in ``tests/test_pde/test_g2pp_piecewise.py``.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc import MCConfig

_ACT365 = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))
VOLS = (0.012, 0.008, 0.015)
RATIOS = (0.8, 0.6, 0.9)
TIMES = (1.0, 2.5)


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


def _model(vols=VOLS, ratios=RATIOS, times=TIMES, curve=CURVE):
    return G2PPModel(
        mean_reversion_x=jnp.asarray(0.5),
        mean_reversion_y=jnp.asarray(0.1),
        volatility_x=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        correlation=jnp.asarray(-0.7),
        initial_curve=curve,
        volatility_times=jnp.asarray(times),
    )


def _swaption(expiry_y, tenor_y, moneyness=0.0, is_payer=True):
    expiry = REF + 365 * expiry_y
    fixed = jnp.array([expiry + 365 * k for k in range(1, tenor_y + 1)], dtype=jnp.int32)
    ann = _annuity(jnp.int32(expiry), fixed, CURVE, _ACT365)
    atm = float((CURVE(jnp.int32(expiry)) - CURVE(fixed[-1])) / ann)
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=fixed,
        strike=jnp.asarray(atm + moneyness),
        notional=jnp.asarray(1.0),
        is_payer=is_payer,
        day_count=_ACT365,
    )


class TestAgainstMonteCarlo:
    """Analytic vs the exact MC engine (independent per-step covariance)."""

    @pytest.mark.parametrize(
        "expiry_y,tenor_y,moneyness,is_payer,seed",
        [(3, 5, 0.0, True, 11), (3, 5, 0.0, False, 12), (2, 4, 0.01, True, 13), (4, 3, -0.01, False, 14)],
    )
    def test_within_two_stderr(self, expiry_y, tenor_y, moneyness, is_payer, seed):
        s = _swaption(expiry_y, tenor_y, moneyness, is_payer)
        model = _model()
        analytic = float(g2pp_swaption_price(s, model))
        res = price(
            s, dataclasses.replace(model, initial_curve=None), MarketState.single_curve(CURVE),
            method="mc", config=MCConfig(n_paths=50_000, n_steps=60), key=jax.random.PRNGKey(seed),
        )
        z = (float(res.price) - analytic) / float(res.stderr)
        assert abs(z) < 2.0, (analytic, float(res.price), float(res.stderr))


class TestAnalyticProperties:
    def test_payer_receiver_parity(self):
        """Payer - receiver = forward payer swap, a curve-only quantity."""
        model = _model()
        for moneyness in (-0.01, 0.0, 0.015):
            pay = _swaption(3, 5, moneyness, True)
            rec = dataclasses.replace(pay, is_payer=False)
            starts = jnp.concatenate([pay.expiry_date[None], pay.fixed_dates[:-1]])
            taus = year_fraction(starts, pay.fixed_dates, _ACT365)
            swap = float(
                CURVE(pay.expiry_date) - CURVE(pay.fixed_dates[-1])
                - pay.strike * jnp.sum(taus * CURVE(pay.fixed_dates))
            )
            diff = float(g2pp_swaption_price(pay, model) - g2pp_swaption_price(rec, model))
            assert diff == pytest.approx(swap, rel=1e-8, abs=1e-12)

    def test_equal_buckets_equal_constant(self):
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011, 0.011, 0.011), ratios=(0.7, 0.7, 0.7), times=TIMES)
        for s in (_swaption(3, 5), _swaption(2, 4, 0.01, False), _swaption(1, 7, -0.005)):
            assert float(g2pp_swaption_price(s, split)) == pytest.approx(
                float(g2pp_swaption_price(s, const)), rel=1e-12
            )

    def test_time_dependence_matters(self):
        """sigma(t) prices differ from every constant bucket's model."""
        s = _swaption(3, 5)
        p = float(g2pp_swaption_price(s, _model()))
        for v, r in zip(VOLS, RATIOS):
            flat = float(g2pp_swaption_price(s, _model(vols=(v,), ratios=(r,), times=())))
            assert abs(p / flat - 1.0) > 1e-2

    def test_through_price_binds_curve(self):
        s = _swaption(3, 5)
        model = _model()
        via_price = price(
            s, dataclasses.replace(model, initial_curve=None),
            MarketState.single_curve(CURVE), method="analytic",
        ).price
        assert float(via_price) == float(g2pp_swaption_price(s, model))

    def test_filter_jit_smoke(self):
        s = _swaption(3, 5)
        model = _model()
        jitted = eqx.filter_jit(g2pp_swaption_price)(s, model)
        assert float(jitted) == pytest.approx(float(g2pp_swaption_price(s, model)), rel=1e-12)

    def test_vmap_over_different_breakpoints(self):
        s = _swaption(3, 5)
        m1, m2 = _model(times=(1.0, 2.5)), _model(times=(0.5, 2.0))
        stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), m1, m2)
        batched = eqx.filter_vmap(lambda m: g2pp_swaption_price(s, m))(stacked)
        np.testing.assert_allclose(
            np.asarray(batched),
            [float(g2pp_swaption_price(s, m)) for m in (m1, m2)],
            rtol=1e-12,
        )


class TestBucketVega:
    @pytest.mark.parametrize("field", ["volatility_x", "volatility_ratio"])
    def test_gradient_matches_fd(self, field):
        s = _swaption(3, 5)
        model = _model()

        def f(v):
            return g2pp_swaption_price(s, eqx.tree_at(lambda m: getattr(m, field), model, v))

        v0 = getattr(model, field)
        grad = np.asarray(jax.grad(f)(v0))
        assert np.all(np.isfinite(grad))
        h = 1e-5 * float(jnp.max(v0))
        for i in range(v0.shape[0]):
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4)

    def test_buckets_after_expiry_have_no_effect(self):
        """A 2Y expiry reads buckets [0, 1) and [1, 2.5) only up to 2Y."""
        s = _swaption(2, 5)
        model = _model(times=(1.0, 2.0 + 1e-9))  # bucket 2 starts after expiry

        def f(vols, ratios):
            return g2pp_swaption_price(
                s, eqx.tree_at(lambda m: (m.volatility_x, m.volatility_ratio), model, (vols, ratios))
            )

        g_v, g_r = jax.grad(f, argnums=(0, 1))(model.volatility_x, model.volatility_ratio)
        g_v, g_r = np.asarray(g_v), np.asarray(g_r)
        assert np.all(np.abs(g_v[:2]) > 0.0) and np.all(np.abs(g_r[:2]) > 0.0)
        assert abs(g_v[2]) <= 1e-12 * np.max(np.abs(g_v))
        assert abs(g_r[2]) <= 1e-12 * np.max(np.abs(g_r))
