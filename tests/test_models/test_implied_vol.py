"""``ImpliedVolModel``: caplets, caps and swaptions off a market vol source.

``price(inst, ImpliedVolModel(surface_id), market)`` must be exactly the
curve-aware kernel of :mod:`valax.pricing.analytic.rates_smile` called on
the resolved curves and the surface filed under ``surface_id`` — for a SABR
swaption cube, a SABR optionlet surface and a flat ``ConstantVol`` quoted
normal or lognormal, single- and dual-curve.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import auto_method, price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.rates import Cap, Caplet, Swaption
from valax.models import ImpliedVolModel
from valax.pricing.analytic.rates_smile import cap_price, caplet_price, swaption_price
from valax.surfaces import ConstantVol, OptionletVolSurface, SwaptionCube

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
OIS, FWD = "USD.SOFR.OIS", "USD.SOFR.3M"
CUBE, OPTIONLET = "USD.SOFR.CUBE", "USD.SOFR.OPTIONLET"
FLAT_LN, FLAT_N = "USD.SOFR.VOL.LN", "USD.SOFR.VOL.N"


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array([int(ymd_to_ordinal(2025 + y, 1, 1)) for y in
                         (0, 1, 2, 3, 4, 5, 7, 10)], dtype=jnp.int32)
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
                         reference_date=jnp.int32(REF))


def _cube() -> SwaptionCube:
    shape = (4, 3)
    return SwaptionCube(
        expiries=jnp.array([0.5, 1.0, 2.0, 5.0]), tenors=jnp.array([1.0, 5.0, 10.0]),
        forwards=jnp.full(shape, 0.03), alphas=jnp.full(shape, 0.22),
        betas=jnp.full(shape, 0.5), rhos=jnp.full(shape, -0.25),
        nus=jnp.full(shape, 0.4),
    )


def _optionlet_surface() -> OptionletVolSurface:
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0])
    n = expiries.shape[0]
    return OptionletVolSurface(
        expiries=expiries, forwards=jnp.full(n, 0.03),
        alphas=jnp.linspace(0.20, 0.28, n), betas=jnp.full(n, 0.5),
        rhos=jnp.full(n, -0.25), nus=jnp.full(n, 0.4),
    )


MARKET = MarketState(
    curves=CurveGraph(curves={OIS: _curve(0.03), FWD: _curve(0.034)}),
    surfaces={
        CUBE: _cube(), OPTIONLET: _optionlet_surface(),
        FLAT_LN: ConstantVol(jnp.asarray(0.25)),
        FLAT_N: ConstantVol(jnp.asarray(0.009), is_normal=True),
    },
)


def _swaption(forward_id: str | None) -> Swaption:
    return Swaption(
        expiry_date=jnp.int32(ymd_to_ordinal(2026, 1, 1)),
        fixed_dates=generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=1).payment_dates,
        strike=jnp.asarray(0.032), notional=jnp.asarray(1e6), is_payer=True,
        discount_id=OIS, forward_id=forward_id,
    )


def _caplet(forward_id: str | None) -> Caplet:
    d = jnp.int32(ymd_to_ordinal(2025, 10, 1))
    return Caplet(fixing_date=d, start_date=d, end_date=jnp.int32(ymd_to_ordinal(2026, 1, 1)),
                  strike=jnp.asarray(0.031), notional=jnp.asarray(1e6), is_cap=True,
                  discount_id=OIS, forward_id=forward_id)


def _cap(forward_id: str | None) -> Cap:
    months = [(2025 + (m // 12), m % 12 + 1) for m in range(3, 27, 3)]
    starts = [int(ymd_to_ordinal(y, m, 1)) for y, m in months[:-1]]
    ends = [int(ymd_to_ordinal(y, m, 1)) for y, m in months[1:]]
    arr = lambda xs: jnp.array(xs, dtype=jnp.int32)  # noqa: E731
    return Cap(fixing_dates=arr(starts), start_dates=arr(starts), end_dates=arr(ends),
               strike=jnp.asarray(0.031), notional=jnp.asarray(1e6), is_cap=False,
               discount_id=OIS, forward_id=forward_id)


KERNELS = {"swaption": (_swaption, swaption_price),
           "caplet": (_caplet, caplet_price),
           "cap": (_cap, cap_price)}
SURFACES = {"swaption": (CUBE, FLAT_LN, FLAT_N),
            "caplet": (OPTIONLET, FLAT_LN, FLAT_N),
            "cap": (OPTIONLET, FLAT_LN, FLAT_N)}
CASES = [(name, sid) for name, sids in SURFACES.items() for sid in sids]


@pytest.mark.parametrize("forward_id", [None, FWD], ids=["single", "dual"])
@pytest.mark.parametrize(("name", "surface_id"), CASES)
def test_route_is_the_kernel(name, surface_id, forward_id):
    build, kernel = KERNELS[name]
    inst = build(forward_id)
    got = price(inst, ImpliedVolModel(surface_id), MARKET)
    assert got.method == "analytic" and got.config is None
    want = kernel(inst, MARKET.curve(OIS), MARKET.surface(surface_id),
                  None if forward_id is None else MARKET.curve(FWD))
    np.testing.assert_array_equal(np.asarray(got.price), np.asarray(want))
    assert float(got.price) > 0.0


def test_the_surface_id_selects_the_vol():
    """Normal and lognormal flat vols give different prices (not a constant)."""
    inst = _swaption(FWD)
    prices = {sid: float(price(inst, ImpliedVolModel(sid), MARKET).price)
              for sid in (CUBE, FLAT_LN, FLAT_N)}
    assert len(set(prices.values())) == 3


def test_auto_is_analytic():
    for build, _ in KERNELS.values():
        assert auto_method(build(None), ImpliedVolModel(CUBE)) == "analytic"


def test_unknown_surface_id_raises():
    with pytest.raises(KeyError):
        price(_swaption(None), ImpliedVolModel("EUR.ESTR.CUBE"), MARKET)


def test_non_vol_source_is_rejected():
    market = MarketState(curves=MARKET.curves, surfaces={CUBE: jnp.asarray(0.2)})
    with pytest.raises(ValueError, match="not a rates vol source"):
        price(_swaption(None), ImpliedVolModel(CUBE), market)


@pytest.mark.parametrize("name", list(KERNELS))
def test_jit_and_market_gradients(name):
    """JIT-safe; the sensitivities land on the curves and the surface."""
    build, _ = KERNELS[name]
    inst = build(FWD)
    sid = CUBE if name == "swaption" else OPTIONLET

    @eqx.filter_jit
    def value(market):
        return price(inst, ImpliedVolModel(sid), market).price

    assert np.isfinite(float(value(MARKET)))
    grads = eqx.filter_grad(value)(MARKET)
    assert np.any(np.asarray(grads.surfaces[sid].alphas) != 0.0)
    assert np.any(np.asarray(grads.curves.curves[FWD].discount_factors) != 0.0)
