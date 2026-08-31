"""G2++ Monte-Carlo dispatcher recipes.

Covers the three ``(instrument, G2PPModel)`` recipes registered in
``valax.pricing.mc.recipes``:

- ``FixedRateBond``    — validated against the analytic curve PV.
- ``FloatingRateBond`` — validated against the par-at-reset identity.
- ``Swaption``         — triangulated against the semi-analytic
  ``g2pp_swaption_price``.  MC and the Brigo-Mercurio integral are independent
  numerical methods sharing only the analytic ZCB, so agreement pins down the
  scheme.

Also asserts the ``eqx.filter_jit`` contract for the registered recipes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import FixedRateBond, FloatingRateBond
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc.dispatch import MCConfig, mc_price_dispatch

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


@pytest.fixture
def curve(ref_date):
    return _flat_curve(ref_date)


@pytest.fixture
def model(curve):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(-0.70),
        initial_curve=curve,
    )


class TestFixedBond:
    def test_matches_curve_pv(self, ref_date, curve, model):
        pay = jnp.array(
            [int(ymd_to_ordinal(2026 + i, 1, 1)) for i in range(5)], dtype=jnp.int32
        )
        bond = FixedRateBond(
            payment_dates=pay, settlement_date=jnp.int32(ref_date),
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=1, day_count=_ACT365,
        )
        res = mc_price_dispatch(
            bond, model, MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS), jax.random.PRNGKey(0)
        )
        amounts = jnp.full((5,), 4.0).at[-1].add(100.0)
        curve_pv = float(jnp.sum(amounts * curve(pay)))
        assert abs(float(res.price) - curve_pv) < 3.0 * float(res.stderr) + 1e-2


class TestFloatingBond:
    def test_par_at_reset(self, ref_date, model):
        pay = jnp.array(
            [int(ymd_to_ordinal(2026 + i, 1, 1)) for i in range(5)], dtype=jnp.int32
        )
        frn = FloatingRateBond(
            payment_dates=pay, fixing_dates=pay, settlement_date=jnp.int32(ref_date),
            spread=jnp.array(0.0), face_value=jnp.array(100.0),
            fixing_rates=None, day_count=_ACT365,
        )
        res = mc_price_dispatch(
            frn, model, MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS), jax.random.PRNGKey(1)
        )
        # Zero-spread FRN prices to par (face) at settlement.
        assert abs(float(res.price) - 100.0) < 3.0 * float(res.stderr) + 1e-2


class TestSwaption:
    @pytest.mark.parametrize("is_payer", [True, False])
    def test_mc_matches_analytic(self, ref_date, curve, model, is_payer):
        expiry = jnp.int32(int(ymd_to_ordinal(2030, 1, 1)))
        fixed = jnp.array(
            [int(ymd_to_ordinal(2031 + k, 1, 1)) for k in range(5)], dtype=jnp.int32
        )
        ann = _annuity(expiry, fixed, curve, _ACT365)
        fwd = float((curve(expiry) - curve(fixed[-1])) / ann)
        sw = Swaption(
            expiry_date=expiry, fixed_dates=fixed, strike=jnp.array(fwd),
            notional=jnp.array(1.0), is_payer=is_payer, day_count=_ACT365,
        )
        analytic = float(g2pp_swaption_price(sw, model))
        res = mc_price_dispatch(
            sw, model, MCConfig(n_paths=_N_PATHS, n_steps=_N_STEPS), jax.random.PRNGKey(7)
        )
        z = (float(res.price) - analytic) / float(res.stderr)
        assert abs(z) < 3.5, f"is_payer={is_payer}: MC={res.price:.6f} analytic={analytic:.6f} z={z:.2f}"


class TestJITContract:
    def test_fixed_bond_jit(self, ref_date, model):
        pay = jnp.array(
            [int(ymd_to_ordinal(2026 + i, 1, 1)) for i in range(5)], dtype=jnp.int32
        )
        bond = FixedRateBond(
            payment_dates=pay, settlement_date=jnp.int32(ref_date),
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=1, day_count=_ACT365,
        )
        cfg = MCConfig(n_paths=2000, n_steps=_N_STEPS)

        @eqx.filter_jit
        def run(instrument, m, k):
            return mc_price_dispatch(instrument, m, cfg, k)

        res = run(bond, model, jax.random.PRNGKey(0))
        assert jnp.isfinite(res.price)
