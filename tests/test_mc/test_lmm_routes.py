"""The four LMM pairs through ``price()`` (D18).

The recipes map the trade's periods onto the model's tenor grid from the
dates and bind the curve from the market; each route equals the recipe called
with explicit indices on a curve-bound model, bit for bit.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core import MarketState, price
from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Cap, Caplet, Swaption
from valax.models.lmm import ExponentialCorrelation, LMMModel, PiecewiseConstantVol
from valax.pricing.analytic.caplets import caplet_price_black76
from valax.pricing.mc import MCConfig, lmm_period_indices
from valax.pricing.mc.dispatch import mc_price_dispatch

REF = int(ymd_to_ordinal(2025, 1, 1))
TENOR = jnp.array(
    [int(ymd_to_ordinal(y, 1, 1)) for y in range(2026, 2031)], dtype=jnp.int32
)
CFG = MCConfig(n_paths=4_000, n_steps=50)  # 10 Euler steps per grid segment
KEY = jax.random.PRNGKey(0)


def _curve(rate: float = 0.04) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + y, 1, 1)) for y in range(13)], dtype=jnp.int32
    )
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=jnp.exp(-rate * t - 0.001 * t * t),
        reference_date=jnp.int32(REF),
    )


CURVE = _curve()
DC = CURVE.day_count
MARKET = MarketState.single_curve(CURVE)
MODEL = LMMModel(
    tenor_dates=TENOR,
    vol_structure=PiecewiseConstantVol(vols=0.2 * jnp.ones((4, 4))),
    corr_structure=ExponentialCorrelation(beta=jnp.array(0.1)),
)
K = jnp.array(0.045)
NOTIONAL = jnp.array(1e6)


def _caplet(i: int = 1, **kw) -> Caplet:
    return Caplet(
        fixing_date=TENOR[i], start_date=TENOR[i], end_date=TENOR[i + 1],
        strike=K, notional=NOTIONAL, day_count=kw.pop("day_count", DC), **kw,
    )


def _cap() -> Cap:
    return Cap(
        fixing_dates=TENOR[:-1], start_dates=TENOR[:-1], end_dates=TENOR[1:],
        strike=K, notional=NOTIONAL, day_count=DC,
    )


def _swaption(**kw) -> Swaption:
    return Swaption(
        expiry_date=TENOR[1], fixed_dates=TENOR[2:], strike=K,
        notional=NOTIONAL, day_count=DC, **kw,
    )


def _bermudan(exercise=TENOR[:-1], fixed=TENOR[1:]) -> BermudanSwaption:
    return BermudanSwaption(
        exercise_dates=exercise, fixed_dates=fixed, strike=K,
        notional=NOTIONAL, day_count=DC,
    )


BOUND = bind_curves(MODEL, CURVE)
TAUS = BOUND.accrual_fractions

#: Each route with the pre-D18 explicit market args that must reproduce it.
ROUTES = {
    "caplet": (_caplet(), dict(forward_index=1, tau=TAUS[1])),
    "cap": (_cap(), dict(forward_indices=jnp.arange(4), taus=TAUS)),
    "swaption": (_swaption(), dict(forward_indices=jnp.arange(1, 4), taus=TAUS[1:])),
    "bermudan": (_bermudan(), dict(exercise_indices=jnp.arange(4), taus=TAUS)),
}


@pytest.mark.parametrize("name", ROUTES)
def test_route_equals_recipe_with_explicit_indices(name):
    instrument, explicit = ROUTES[name]
    routed = price(instrument, MODEL, MARKET, config=CFG, key=KEY)
    direct = mc_price_dispatch(
        instrument, BOUND, CFG, KEY, n_steps_per_period=10, **explicit
    )
    assert routed.method == "mc"
    assert float(routed.price) == float(direct.price)
    assert float(routed.stderr) == float(direct.stderr)


@pytest.mark.parametrize("name", ROUTES)
def test_market_delta_equals_bound_model_delta(name):
    """Curve risk lands on the market, equal to differentiating the curve
    inside a bound model priced directly."""
    instrument, explicit = ROUTES[name]

    def via_market(curve):
        return price(
            instrument, MODEL, MarketState.single_curve(curve), config=CFG, key=KEY,
        ).price

    def via_model(curve):
        return mc_price_dispatch(
            instrument, bind_curves(MODEL, curve), CFG, KEY,
            n_steps_per_period=10, **explicit,
        ).price

    g_market = eqx.filter_grad(via_market)(CURVE).discount_factors
    g_model = eqx.filter_grad(via_model)(CURVE).discount_factors
    assert jnp.any(g_market != 0.0)
    np.testing.assert_allclose(g_market, g_model, rtol=1e-12, atol=1e-9)


def test_model_gradient_has_no_curve_leaf():
    grad = eqx.filter_grad(
        lambda m: price(_caplet(), m, MARKET, config=CFG, key=KEY).price
    )(MODEL)
    assert grad.initial_curve is None
    assert float(jnp.sum(grad.vol_structure.vols)) != 0.0


def test_caplet_matches_black76():
    """Caplets are exact under the LMM: Black-76 within 2 pooled stderr
    (measured |z| <= 1.04 at the default config on three forwards)."""
    caplet = _caplet(2)
    cfg = MCConfig(n_paths=50_000, n_steps=50)
    results = [
        price(caplet, MODEL, MARKET, config=cfg, key=jax.random.PRNGKey(k))
        for k in range(4)
    ]
    mc = np.mean([float(r.price) for r in results])
    stderr = np.sqrt(np.sum([float(r.stderr) ** 2 for r in results])) / 4
    black = float(caplet_price_black76(caplet, CURVE, jnp.array(0.2)))
    assert abs(mc - black) < 2.0 * stderr, (mc, black, stderr)


def test_single_exercise_bermudan_is_the_european():
    """Same paths, same exercise: the LSM collapses to the European payoff."""
    european = price(_swaption(), MODEL, MARKET, config=CFG, key=KEY)
    bermudan = price(
        _bermudan(exercise=TENOR[1:2], fixed=TENOR[2:]), MODEL, MARKET,
        config=CFG, key=KEY,
    )
    np.testing.assert_allclose(bermudan.price, european.price, rtol=1e-12)


def test_n_steps_comes_from_the_config():
    coarse = price(_swaption(), MODEL, MARKET, config=MCConfig(4_000, 5), key=KEY)
    fine = price(_swaption(), MODEL, MARKET, config=MCConfig(4_000, 100), key=KEY)
    direct = mc_price_dispatch(
        _swaption(), BOUND, MCConfig(4_000, 5), KEY, n_steps_per_period=1,
        **ROUTES["swaption"][1],
    )
    assert float(coarse.price) == float(direct.price)
    assert float(coarse.price) != float(fine.price)


def test_period_indices_from_dates():
    idx, ok = lmm_period_indices(TENOR, TENOR[1:3], TENOR[2:4])
    np.testing.assert_array_equal(idx, [1, 2])
    assert bool(ok)
    _, ok = lmm_period_indices(TENOR, TENOR[1] + 1, TENOR[2])
    assert not bool(ok)


class TestRefusals:
    def test_pre_bound_model(self):
        with pytest.raises(ValueError, match="pre-bound"):
            price(_caplet(), BOUND, MARKET, config=CFG, key=KEY)

    def test_dates_off_the_grid(self):
        off = Caplet(
            fixing_date=TENOR[1], start_date=TENOR[1] + 3, end_date=TENOR[2],
            strike=K, notional=NOTIONAL, day_count=DC,
        )
        with pytest.raises(ValueError, match="not on the LMMModel tenor grid"):
            price(off, MODEL, MARKET, config=CFG, key=KEY)
        # Under jit the dates are traced: the guard is a NaN price.
        jitted = eqx.filter_jit(
            lambda i: price(i, MODEL, MARKET, config=CFG, key=KEY).price
        )
        assert jnp.isnan(jitted(off))
        assert jnp.isfinite(jitted(_caplet()))

    def test_bermudan_swap_must_end_on_the_grid(self):
        with pytest.raises(ValueError, match="not on the LMMModel tenor grid"):
            price(
                _bermudan(exercise=TENOR[:2], fixed=TENOR[1:4]), MODEL, MARKET,
                config=CFG, key=KEY,
            )

    def test_day_count_mismatch(self):
        other = "act_360" if DC != "act_360" else "act_365"
        with pytest.raises(ValueError, match="forward accrual convention"):
            price(_caplet(day_count=other), MODEL, MARKET, config=CFG, key=KEY)

    def test_projection_curve(self):
        graph = MarketState(curves=CurveGraph(curves={
            "_default_": CURVE, "USD.SOFR.3M": _curve(0.045),
        }))
        with pytest.raises(ValueError, match="projection curve"):
            price(_swaption(forward_id="USD.SOFR.3M"), MODEL, graph, config=CFG, key=KEY)

    def test_unbound_kernel_call(self):
        with pytest.raises(ValueError, match="not bound to a curve"):
            mc_price_dispatch(_caplet(), MODEL, CFG, KEY)
