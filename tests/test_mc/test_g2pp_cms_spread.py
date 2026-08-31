"""G2++ CMS-spread swap (steepener / flattener) Monte-Carlo recipe.

This is the decorrelation-sensitive product that motivates the second factor,
so the load-bearing tests are:

- **Steepener + flattener = 0** (the two are exact negatives on shared paths),
  which pins the sign/discounting conventions.
- **Monotone dependence on rho**, evaluated with common random numbers so the
  comparison is essentially noise-free -- a one-factor model could not produce
  this sensitivity at all.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import CMSSpreadSwap
from valax.models.g2pp import G2PPModel
from valax.pricing.mc.dispatch import MCConfig, mc_price_dispatch

_ACT365 = "act_365"
_N_PATHS = 120_000
_N_STEPS = 60


@pytest.fixture(scope="module")
def ref_date():
    return int(ymd_to_ordinal(2025, 1, 1))


@pytest.fixture(scope="module")
def curve(ref_date):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(26)], dtype=jnp.int32
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    # Upward-sloping curve so the 10Y rate exceeds the 2Y rate.
    zero = 0.02 + 0.02 * (1.0 - jnp.exp(-0.2 * times))
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=jnp.int32(ref_date),
        day_count=_ACT365,
    )


def _model(curve, rho):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.012),
        volatility_y=jnp.array(0.009),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


def _swap(ref_date, pay_fixed, fixed_rate=0.0):
    pay = jnp.array(
        [int(ymd_to_ordinal(2026 + i, 1, 1)) for i in range(5)], dtype=jnp.int32
    )
    return CMSSpreadSwap(
        start_date=jnp.int32(ref_date),
        payment_dates=pay,
        fixed_rate=jnp.array(fixed_rate),
        notional=jnp.array(1e6),
        cms_tenor_long=10,
        cms_tenor_short=2,
        pay_fixed=pay_fixed,
        day_count=_ACT365,
    )


class TestConventions:
    def test_steepener_flattener_sum_to_zero(self, ref_date, curve):
        """On shared paths a steepener and flattener are exact negatives."""
        model = _model(curve, -0.5)
        cfg = MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS)
        key = jax.random.PRNGKey(3)
        steep = mc_price_dispatch(_swap(ref_date, True), model, cfg, key)
        flat = mc_price_dispatch(_swap(ref_date, False), model, cfg, key)
        assert float(steep.price) + float(flat.price) == pytest.approx(0.0, abs=1e-6)

    def test_steepener_positive_on_upward_curve(self, ref_date, curve):
        model = _model(curve, -0.5)
        res = mc_price_dispatch(
            _swap(ref_date, True), model, MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS),
            jax.random.PRNGKey(0),
        )
        assert float(res.price) > 3.0 * float(res.stderr)


class TestDecorrelation:
    def test_monotone_in_rho(self, ref_date, curve):
        """Steepener value is monotone increasing in rho (common random numbers)."""
        cfg = MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS)
        key = jax.random.PRNGKey(7)
        swap = _swap(ref_date, True)

        def price_at(rho):
            return float(mc_price_dispatch(swap, _model(curve, rho), cfg, key).price)

        p_lo, p_mid, p_hi = price_at(-0.9), price_at(0.0), price_at(0.9)
        assert p_lo < p_mid < p_hi


class TestJIT:
    def test_jit_contract(self, ref_date, curve):
        model = _model(curve, -0.5)
        cfg = MCConfig(n_paths=4000, n_steps=_N_STEPS)

        @eqx.filter_jit
        def run(instrument, m, k):
            return mc_price_dispatch(instrument, m, cfg, k)

        res = run(_swap(ref_date, True), model, jax.random.PRNGKey(0))
        assert jnp.isfinite(res.price)
