"""The Hull-White tree routes of ``price(..., method="lattice")`` (D15).

Callable / puttable bonds and European / Bermudan swaptions under Hull-White
price on the **general** tree with a config-sized half-width (``TreeConfig``:
J = ``ceil(width_factor * n_steps)``, factor 2, unless ``j_max`` is set), so
the routes run on a traced model
under ``jit`` / ``grad``. ``auto``
is unchanged (PDE for the bonds and the Bermudan, Jamshidian for the
swaption); the tree is the independent check.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves
from valax.core.market_state import MarketState
from valax.core.pricing import auto_method, price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond, PuttableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.lattice import (
    TreeConfig,
    bermudan_swaption_tree_price,
    callable_bond_price,
    hw_tree_width,
    puttable_bond_price,
    swaption_tree_price,
)

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
N = 60  # tree steps in these tests (keeps J = N cheap)


def _dates(years):
    return jnp.array([int(ymd_to_ordinal(2025 + y, 1, 1)) for y in years], dtype=jnp.int32)


def _curve():
    pillars = _dates(range(12))
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(pillar_dates=pillars,
                         discount_factors=jnp.exp(-(0.02 + 0.002 * t) * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


MARKET = MarketState.single_curve(_curve())


def _model(piecewise: bool) -> HullWhiteModel:
    if piecewise:
        return HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                              volatility=jnp.asarray([0.012, 0.009, 0.011]),
                              volatility_times=jnp.asarray([1.0, 3.0]))
    return HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01))


def _bond_kwargs():
    return dict(payment_dates=_dates([1, 2, 3, 4, 5]), settlement_date=jnp.int32(REF),
                coupon_rate=jnp.asarray(0.04), face_value=jnp.asarray(1.0), frequency=1,
                day_count="act_365")


def _callable():
    return CallableBond(**_bond_kwargs(), call_dates=_dates([2, 3, 4]),
                        call_prices=jnp.full(3, 1.0))


def _puttable():
    return PuttableBond(**_bond_kwargs(), put_dates=_dates([2, 3, 4]),
                        put_prices=jnp.full(3, 1.0))


def _swaption():
    return Swaption(expiry_date=_dates([2])[0], fixed_dates=_dates([3, 4, 5, 6]),
                    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                    day_count="act_365")


def _bermudan():
    return BermudanSwaption(exercise_dates=_dates([2, 3, 4]),
                            fixed_dates=_dates([3, 4, 5, 6]),
                            strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                            day_count="act_365")


def _direct(name, inst, model, cfg):
    """The same price by the direct function on the general tree."""
    bound = bind_curves(model, MARKET.curve("_default_"))
    kw = dict(n_steps=cfg.n_steps, j_max=cfg.half_width, general=True)
    if name == "callable":
        return callable_bond_price(inst, bound, **kw)
    if name == "puttable":
        return puttable_bond_price(inst, bound, **kw)
    kw["smooth_payoff"] = cfg.smooth_payoff
    if name == "swaption":
        return swaption_tree_price(inst, bound, **kw)
    return bermudan_swaption_tree_price(inst, bound, **kw)


CASES = {"callable": _callable, "puttable": _puttable,
         "swaption": _swaption, "bermudan": _bermudan}
CFG = TreeConfig(n_steps=N)


@pytest.mark.parametrize("piecewise", [False, True], ids=["constant", "sigma_t"])
@pytest.mark.parametrize("name", list(CASES))
def test_route_equals_the_direct_general_tree(name, piecewise):
    inst, model = CASES[name](), _model(piecewise)
    result = price(inst, model, MARKET, method="lattice", config=CFG)
    assert result.method == "lattice" and result.config == CFG
    np.testing.assert_array_equal(np.asarray(result.price),
                                  np.asarray(_direct(name, inst, model, CFG)))


def test_auto_is_unchanged():
    model = _model(False)
    assert auto_method(_callable(), model) == "pde"
    assert auto_method(_puttable(), model) == "pde"
    assert auto_method(_bermudan(), model) == "pde"
    assert auto_method(_swaption(), model) == "analytic"


@pytest.mark.parametrize("piecewise", [False, True], ids=["constant", "sigma_t"])
def test_config_sized_width_equals_the_tight_width(piecewise):
    """The default width changes nothing: the extra nodes carry no mass."""
    inst, model = _swaption(), _model(piecewise)
    bound = bind_curves(model, MARKET.curve("_default_"))
    tight = hw_tree_width(bound, 2.0, N)
    assert CFG.half_width == 2 * N and tight <= CFG.half_width
    wide = price(inst, model, MARKET, method="lattice", config=CFG).price
    narrow = price(inst, model, MARKET, method="lattice",
                   config=TreeConfig(n_steps=N, j_max=tight)).price
    assert np.isfinite(float(narrow))
    np.testing.assert_allclose(float(wide), float(narrow), rtol=1e-12)


def test_a_too_narrow_tree_prices_nan():
    """Where sigma(t) drops the tree needs J > n_steps; short of it, NaN.

    The 0.012 -> 0.009 drop at 1Y needs 70 half-width nodes at 60 steps, so
    ``width_factor=1`` (J = 60) is flagged invalid — the reason the default
    factor is 2.
    """
    inst, model = _swaption(), _model(True)
    bound = bind_curves(model, MARKET.curve("_default_"))
    assert hw_tree_width(bound, 2.0, N) > N
    narrow = price(inst, model, MARKET, method="lattice",
                   config=TreeConfig(n_steps=N, width_factor=1.0)).price
    assert np.isnan(float(narrow))
    assert np.isfinite(float(price(inst, model, MARKET, method="lattice",
                                   config=CFG).price))


def test_width_factor_below_one_is_rejected():
    with pytest.raises(ValueError, match="width_factor"):
        TreeConfig(width_factor=0.5)


def test_tree_route_is_close_to_jamshidian():
    inst, model = _swaption(), _model(True)
    tree = float(price(inst, model, MARKET, method="lattice",
                       config=TreeConfig(n_steps=400)).price)
    exact = float(hw_swaption_price(inst, bind_curves(model, MARKET.curve("_default_"))))
    assert abs(tree / exact - 1.0) < 5e-3  # first order in n_steps


@pytest.mark.parametrize("name", list(CASES))
def test_default_config_works_under_jit_and_grad(name):
    """No j_max needed on a traced model: J comes from the config."""
    inst, model = CASES[name](), _model(True)
    cfg = TreeConfig(n_steps=N)

    def value(m):
        return price(inst, m, MARKET, method="lattice", config=cfg).price

    assert np.isfinite(float(eqx.filter_jit(value)(model)))
    vega = eqx.filter_grad(value)(model).volatility
    assert np.all(np.isfinite(np.asarray(vega)))
    assert np.any(np.asarray(vega) != 0.0)


def test_default_tree_config_is_recorded():
    result = price(_swaption(), _model(False), MARKET, method="lattice")
    assert result.config == TreeConfig()
    assert result.config.half_width == 200
