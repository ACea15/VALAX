"""Synthetic quote universes: truth → quotes → calibration recovers truth."""

import jax.numpy as jnp
import numpy as np
import pytest

import valax
from valax.calibration import calibrate_market, calibrate_model, price_fit_node
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.core.pricing import price
from valax.instruments.options import EuropeanOption
from valax.market import (
    EquityWorldSpec,
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    equity_snapshot,
    rates_snapshot,
    sample_equity_trades,
    sample_quote_scenarios,
    sample_swap_book,
    synthesize_model_quotes,
)
from valax.models.heston import HestonModel

CFG = default_config()


def _reg(seed=7):
    return SeedRegistry(seed, valax.__version__)


class TestRatesUniverse:
    def test_noise_free_quotes_recalibrate_to_truth(self):
        snap, world = rates_snapshot(_reg(), CFG)
        market = calibrate_market(snap.chain, snap.quotes, strict=True).market
        for cid in world.curve_ids:
            np.testing.assert_allclose(
                market.curves[cid].discount_factors,
                world.truth[cid].discount_factors,
                atol=1e-12,
            )

    def test_projection_curve_carries_basis(self):
        _, world = rates_snapshot(_reg(), CFG)
        ois = world.truth["USD.SOFR.OIS"].discount_factors
        proj = world.truth["USD.TERM.3M"].discount_factors
        assert float(jnp.max(jnp.abs(ois[1:] - proj[1:]))) > 1e-5

    def test_zero_basis_is_the_single_curve_limit(self):
        snap, world = rates_snapshot(_reg(), CFG, RatesWorldSpec(zero_basis=True))
        np.testing.assert_allclose(
            world.truth["USD.SOFR.OIS"].discount_factors,
            world.truth["USD.TERM.3M"].discount_factors,
        )
        # IBOR swap par rates then equal OIS par rates at the same maturity
        # up to the float-leg frequency (telescoping), within a few bp.
        q = snap.clean
        assert abs(float(q["USD.IRS3M.5Y"] - q["USD.OIS.5Y"])) < 5e-4

    def test_noise_is_reproducible_and_sized(self):
        a, _ = rates_snapshot(_reg(3), CFG, bp_noise=1.0)
        b, _ = rates_snapshot(_reg(3), CFG, bp_noise=1.0)
        diff = a.quotes.vector() - a.clean.vector()
        np.testing.assert_array_equal(a.quotes.vector(), b.quotes.vector())
        assert 1e-6 < float(jnp.std(diff)) < 5e-4

    def test_swap_book_at_par_prices_to_zero(self):
        snap, world = rates_snapshot(_reg(), CFG)
        swaps = sample_swap_book(_reg(), world, off_market_bp=0.0)
        for s in swaps:
            pv = price(s, None, snap.truth_market).price
            assert abs(float(pv)) < 1e-6 * float(s.notional)


class TestEquityUniverse:
    def test_noise_free_smile_recalibrates_to_truth(self):
        snap, world = equity_snapshot(_reg(11), CFG)
        market = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True).market
        fitted = market.surfaces["_default_"]
        for field in ("alphas", "rhos", "nus", "forwards"):
            np.testing.assert_allclose(
                getattr(fitted, field), getattr(world.smile, field), atol=1e-9
            )

    def test_quote_keys_and_coords(self):
        snap, world = equity_snapshot(_reg(11), CFG, EquityWorldSpec(expiries=(0.5, 1.0)))
        smile = snap.parts["smile"]
        assert smile.vol_keys[0][2] == "SPX.VOL.6M.100"
        assert smile.coords["SPX.VOL.1Y.80"] == (1.0, 0.8)
        assert set(snap.chain.quote_keys) == set(snap.quotes.values)

    def test_trades_price_on_truth(self):
        snap, world = equity_snapshot(_reg(11), CFG)
        trades = sample_equity_trades(_reg(), world, kinds=("european",))
        from valax.models.black_scholes import BlackScholesModel

        for t in trades:
            assert float(price(t, BlackScholesModel(), snap.truth_market).price) > 0.0

    def test_model_quotes_recover_true_parameters(self):
        """Premiums under a true Heston model → price_fit_node recovers it."""
        snap, world = equity_snapshot(_reg(11), CFG, EquityWorldSpec(expiries=(1.0,)))
        truth = HestonModel(
            v0=jnp.asarray(0.05), kappa=jnp.asarray(1.5), theta=jnp.asarray(0.05),
            xi=jnp.asarray(0.5), rho=jnp.asarray(-0.6),
        )
        basket = [EuropeanOption(strike=k, expiry=jnp.asarray(1.0)) for k in world.strikes[0]]
        keys = [f"SPX.PREM.1Y.{i}" for i in range(len(basket))]
        quotes = synthesize_model_quotes(_reg(), truth, basket, keys, snap.truth_market)
        node = price_fit_node(
            "heston",
            template=HestonModel(
                v0=jnp.asarray(0.04), kappa=truth.kappa, theta=truth.theta,
                xi=jnp.asarray(0.4), rho=jnp.asarray(-0.4),
            ),
            transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "xi", "rho")},
            basket=basket,
            quote_keys=keys,
        )
        fit = calibrate_model(node, quotes, snap.truth_market, strict=True).value
        for name in ("v0", "xi", "rho"):
            assert float(getattr(fit, name)) == pytest.approx(
                float(getattr(truth, name)), rel=1e-4
            )


class TestScenarios:
    def test_shapes_and_untouched_keys(self):
        snap, _ = equity_snapshot(_reg(11), CFG)
        parts = snap.parts
        sc = sample_quote_scenarios(
            _reg(), snap.quotes, 8,
            rate_tenors=parts["curves"].tenors,
            vol_coords=parts["smile"].coords,
            spot_keys=(parts["smile"].spot_key,),
        )
        assert all(v.shape == (8,) for v in sc.values.values())
        np.testing.assert_allclose(sc["SPX.DIV"], snap.quotes["SPX.DIV"])
        assert float(jnp.std(sc["SPX.SPOT"])) > 0.0

    def test_spot_and_vol_move_against_each_other(self):
        snap, _ = equity_snapshot(_reg(11), CFG)
        parts = snap.parts
        sc = sample_quote_scenarios(
            _reg(), snap.quotes, 2000,
            vol_coords=parts["smile"].coords,
            spot_keys=(parts["smile"].spot_key,),
            vol_skew_bp=0.0,
        )
        atm = parts["smile"].vol_keys[0][2]
        corr = np.corrcoef(np.log(sc["SPX.SPOT"]), sc[atm])[0, 1]
        assert corr < -0.5
