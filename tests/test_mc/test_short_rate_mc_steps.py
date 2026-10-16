"""Short-rate Monte Carlo reads its time grid from ``MCConfig.n_steps``.

Before D15 the Hull-White, G2++ and Cheyette 1F/2F recipes took the step
count as a recipe keyword (default 100) that ``price()`` could not pass, so
``MCConfig(n_steps=...)`` was silently ignored: 4 and 400 steps gave
bit-identical prices. Anti-vacuity: they must now differ; and the default
config must reproduce the former 100-step price exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond, FixedRateBond
from valax.instruments.rates import Swaption
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.mc import MCConfig
from valax.pricing.mc.dispatch import mc_price_dispatch

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
N_PATHS = 2048


def _market():
    pillars = jnp.array([int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(12)],
                        dtype=jnp.int32)
    t = (pillars - REF).astype(jnp.float64) / 365.0
    curve = DiscountCurve(pillar_dates=pillars,
                          discount_factors=jnp.exp(-(0.02 + 0.002 * t) * t),
                          reference_date=jnp.int32(REF), day_count="act_365")
    return MarketState.single_curve(curve)


def _dates(years):
    return jnp.array([int(ymd_to_ordinal(2025 + y, 1, 1)) for y in years], dtype=jnp.int32)


def _swaption():
    return Swaption(expiry_date=_dates([2])[0], fixed_dates=_dates([3, 4, 5, 6]),
                    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                    day_count="act_365")


def _bond():
    return FixedRateBond(payment_dates=_dates([1, 2, 3, 4, 5]),
                         settlement_date=jnp.int32(REF),
                         coupon_rate=jnp.asarray(0.03), face_value=jnp.asarray(100.0),
                         frequency=1)


def _callable():
    bond = _bond()
    return CallableBond(payment_dates=bond.payment_dates,
                        settlement_date=bond.settlement_date,
                        coupon_rate=bond.coupon_rate, face_value=bond.face_value,
                        call_dates=_dates([2, 3, 4]),
                        call_prices=jnp.full(3, 1.0), frequency=1)


CASES = {
    "hull_white callable": (
        _callable, HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                                  volatility=jnp.asarray(0.01))),
    "g2pp swaption": (
        _swaption, G2PPModel(mean_reversion_x=jnp.asarray(0.5),
                             mean_reversion_y=jnp.asarray(0.1),
                             volatility_x=jnp.asarray(0.01),
                             volatility_ratio=jnp.asarray(0.8),
                             correlation=jnp.asarray(-0.7))),
    "cheyette swaption": (
        _swaption, CheyetteModel(mean_reversion=jnp.asarray(0.03),
                                 volatility=jnp.asarray(0.008),
                                 skew=jnp.asarray(0.2))),
    "cheyette2f bond": (
        _bond, CheyetteModel2F(mean_reversion_1=jnp.asarray(0.5),
                               mean_reversion_2=jnp.asarray(0.05),
                               volatility_1=jnp.asarray(0.008),
                               volatility_ratio=jnp.asarray(0.8),
                               skew_1=jnp.asarray(0.2), skew_2=jnp.asarray(0.1),
                               correlation=jnp.asarray(-0.6))),
}


@pytest.mark.parametrize("case", list(CASES))
def test_mc_config_n_steps_is_used(case):
    make, model = CASES[case]
    inst, market, key = make(), _market(), jax.random.PRNGKey(7)

    def mc(n_steps):
        return price(inst, model, market, method="mc",
                     config=MCConfig(n_paths=N_PATHS, n_steps=n_steps), key=key)

    coarse, fine = mc(4), mc(400)
    assert coarse.config.n_steps == 4 and fine.config.n_steps == 400
    assert float(coarse.price) != float(fine.price), case


@pytest.mark.parametrize("case", list(CASES))
def test_hundred_steps_reproduce_the_former_default(case):
    """The recipe keyword still wins, and 100 config steps equal it exactly."""
    from valax.core.binding import bind_curves

    make, model = CASES[case]
    inst, market, key = make(), _market(), jax.random.PRNGKey(7)
    via_config = price(inst, model, market, method="mc",
                       config=MCConfig(n_paths=N_PATHS, n_steps=100), key=key)
    bound = bind_curves(model, market.curve("_default_"))
    direct = mc_price_dispatch(inst, bound, MCConfig(n_paths=N_PATHS, n_steps=1),
                               key, n_steps=100)
    np.testing.assert_array_equal(np.asarray(via_config.price), np.asarray(direct.price))


def test_default_mc_config_is_recorded():
    make, model = CASES["g2pp swaption"]
    result = price(make(), model, _market(), method="mc", key=jax.random.PRNGKey(0))
    assert result.config == MCConfig(n_paths=100_000, n_steps=100)
