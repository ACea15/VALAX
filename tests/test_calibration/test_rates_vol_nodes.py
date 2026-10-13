"""Rates vol levels as chain nodes: swaption cube, cap strip, optionlet SABR.

Acceptance for the rates vol leg of the chain (curves → vol → price):

* a noise-free synthetic snapshot recovers the true curves, cube and caplet
  strip;
* for a swaption on the cube and a cap on the optionlet surface, the
  explicit adjoint (:func:`chain_adjoint`) and autodiff through
  :func:`calibrate_market` agree on every quote and fixed input, and both
  match full-recalibration finite differences;
* every level's condition number is finite and reported.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import (
    ILL_CONDITIONED_THRESHOLD,
    calibrate_market,
    swaption_cube_node,
)
from valax.greeks.quote_risk import chain_adjoint
from valax.market.synthetic import (
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    rates_vol_snapshot,
)
from valax.pricing.analytic.rates_smile import cap_price, swaption_price
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity

jax.config.update("jax_enable_x64", True)

RATES = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7))
VOLS = RatesVolSpec(expiry_years=(1, 2), tenor_years=(2, 5), cap_years=(1, 2, 3))


@pytest.fixture(scope="module")
def clean_world():
    return rates_vol_snapshot(SeedRegistry(11, "test"), SyntheticMarketConfig(), RATES, VOLS)


@pytest.fixture(scope="module")
def noisy_world():
    return rates_vol_snapshot(
        SeedRegistry(11, "test"), SyntheticMarketConfig(), RATES, VOLS,
        bp_noise=0.5, vol_bp_noise=0.5,
    )


@pytest.fixture(scope="module")
def noisy_calibrated(noisy_world):
    snap, _ = noisy_world
    return calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)


def _swaption_pv(world):
    sw0 = world.swaptions[1][1]  # 2Y x 5Y cell
    disc, fwd = world.discount_id, world.forward_id
    cube_id = world.spec.cube_id
    # A fixed contract: strike 25bp above the true forward.
    s0, _ = swaption_forward_and_annuity(
        sw0, world.rates.truth[disc], world.rates.truth[fwd]
    )
    sw = eqx.tree_at(lambda s: s.strike, sw0, s0 + 0.0025)

    def pv(market):
        return swaption_price(
            sw, market.curve(disc), market.surfaces[cube_id], market.curve(fwd)
        )

    return pv


def _cap_pv(world):
    cap0 = world.caps[-1]
    cap = eqx.tree_at(lambda c: c.strike, cap0, world.cap_strikes[1] + 0.0010)
    disc, fwd, sid = world.discount_id, world.forward_id, world.spec.optionlet_id

    def pv(market):
        return cap_price(cap, market.curve(disc), market.surfaces[sid], market.curve(fwd))

    return pv


def _autodiff(chain, quotes, fixed, pv):
    def value(q, p):
        return pv(calibrate_market(chain, q, p).market)

    return eqx.filter_jit(jax.grad(value, argnums=(0, 1)))(quotes, fixed)


def _fd(chain, quotes, fixed, pv, key, h, fixed_key=False):
    f = eqx.filter_jit(lambda q, p: pv(calibrate_market(chain, q, p).market))
    if fixed_key:
        up = f(quotes, fixed.with_values({key: fixed[key] + h}))
        dn = f(quotes, fixed.with_values({key: fixed[key] - h}))
    else:
        up = f(quotes.with_values({key: quotes[key] + h}), fixed)
        dn = f(quotes.with_values({key: quotes[key] - h}), fixed)
    return float((up - dn) / (2 * h))


class TestTruthRecovery:
    def test_noise_free_quotes_recover_the_truth(self, clean_world):
        snap, world = clean_world
        cal = calibrate_market(snap.chain, snap.clean, snap.fixed, strict=True)
        for cid in world.rates.curve_ids:
            np.testing.assert_allclose(
                cal.market.curve(cid).discount_factors,
                world.rates.truth[cid].discount_factors, atol=1e-12,
            )
        cube = cal.market.surfaces[world.spec.cube_id]
        np.testing.assert_allclose(cube.forwards, world.cube.forwards, atol=1e-12)
        np.testing.assert_allclose(cube.alphas, world.cube.alphas, atol=1e-10)
        np.testing.assert_allclose(cube.rhos, world.cube.rhos, atol=1e-8)
        np.testing.assert_allclose(cube.nus, world.cube.nus, atol=1e-8)
        strip = cal.market.surfaces[world.spec.strip_id]
        np.testing.assert_allclose(strip.vols, world.strip.vols, atol=1e-12)


class TestConditioning:
    def test_every_level_reports_a_finite_condition_number(self, noisy_calibrated):
        for res in noisy_calibrated.results:
            kappa = float(res.condition_number())
            assert np.isfinite(kappa) and kappa >= 1.0, res.name
            assert kappa < ILL_CONDITIONED_THRESHOLD, (res.name, kappa)


@pytest.mark.parametrize("product", ["swaption", "cap"])
class TestQuoteRisk:
    """Adjoint vs autodiff vs full-recalibration FD, curve → vol → price."""

    def _pv(self, product, world):
        return _swaption_pv(world) if product == "swaption" else _cap_pv(world)

    def test_adjoint_matches_autodiff(self, product, noisy_world, noisy_calibrated):
        snap, world = noisy_world
        pv = self._pv(product, world)
        market_bar = eqx.filter_grad(pv)(noisy_calibrated.market)
        adj = chain_adjoint(noisy_calibrated, market_bar)
        g_q, g_p = _autodiff(snap.chain, snap.quotes, snap.fixed, pv)
        scale = max(abs(float(v)) for v in g_q.values.values())
        for k, v in g_q.values.items():
            np.testing.assert_allclose(
                float(adj.quotes[k]), float(v), rtol=1e-6, atol=1e-8 * scale,
                err_msg=k,
            )
        for k, v in g_p.values.items():
            np.testing.assert_allclose(
                float(adj.fixed[k]), float(v), rtol=1e-6, atol=1e-8 * scale,
                err_msg=k,
            )
        # Both the curve and the vol legs carry risk: the chain is coupled.
        curve_keys = [k for k in g_q.values if ".OIS." in k or ".IRS" in k]
        vol_keys = [k for k in g_q.values if ".SWPN." in k or ".CAP." in k]
        assert max(abs(float(adj.quotes[k])) for k in curve_keys) > 0.0
        assert max(abs(float(adj.quotes[k])) for k in vol_keys) > 0.0

    def test_adjoint_matches_full_recalibration_fd(
        self, product, noisy_world, noisy_calibrated
    ):
        snap, world = noisy_world
        pv = self._pv(product, world)
        market_bar = eqx.filter_grad(pv)(noisy_calibrated.market)
        adj = chain_adjoint(noisy_calibrated, market_bar)
        vols = snap.parts["vols"]
        if product == "swaption":
            vol_key = vols.swaption_keys[1][1][3]
            curve_keys = ("USD.OIS.5Y", "USD.IRS3M.3Y")
        else:
            vol_key = vols.cap_keys[1][-1]
            curve_keys = ("USD.OIS.2Y", "USD.IRS3M.3Y")
        for key, h in ((curve_keys[0], 1e-5), (curve_keys[1], 1e-5), (vol_key, 1e-6)):
            fd = _fd(snap.chain, snap.quotes, snap.fixed, pv, key, h)
            assert abs(fd) > 0.0, key  # a live sensitivity, not a vacuous match
            np.testing.assert_allclose(float(adj.quotes[key]), fd, rtol=2e-4, err_msg=key)
        if product == "swaption":
            beta_key = vols.beta_key
            fd = _fd(snap.chain, snap.quotes, snap.fixed, pv, beta_key, 1e-5, fixed_key=True)
            np.testing.assert_allclose(float(adj.fixed[beta_key]), fd, rtol=2e-3)


class TestNodeContract:
    def test_bad_surface_id_fails_at_construction(self, clean_world):
        _, world = clean_world
        with pytest.raises(ValueError, match="surface identifier"):
            swaption_cube_node(
                "not a surface id",
                swaptions=world.swaptions,
                strikes_per_node=[[jnp.zeros(1) for _ in r] for r in world.swaptions],
                quote_keys=[[("a",) for _ in r] for r in world.swaptions],
                discount_id=world.discount_id,
                reference_date=world.rates.reference_date,
            )

    def test_bad_cell_is_named(self, clean_world):
        snap, world = clean_world
        key = snap.parts["vols"].swaption_keys[0][1][2]  # a 1Yx5Y quote
        bad = snap.clean.with_values({key: jnp.nan})
        cal = calibrate_market(snap.chain, bad, snap.fixed)
        cube = cal.result(snap.parts["vols"].cube_node.name)
        assert cube.failed_labels() == ("1Yx5Y",)
