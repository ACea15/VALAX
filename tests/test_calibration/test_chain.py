"""Calibration chains: construction, calibration, traceability.

Risk through the chain is tested in ``tests/test_greeks/test_quote_risk.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import (
    CalibrationChain,
    CalibrationError,
    SolverSettings,
    calibrate_market,
    calibrate_model,
    curve_graph_node,
    market_inputs_node,
    price_fit_node,
    positive,
)
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves import bootstrap_curve_graph
from valax.instruments.options import EuropeanOption
from valax.market import QuoteSet
from valax.models.black_scholes import BlackScholesModel
from valax.surfaces.sabr_surface import SABRVolSurface

from tests.test_calibration._chain_fixtures import (
    DC_KEYS,
    DEFAULT,
    EXPIRIES,
    OIS,
    P3M,
    REF,
    STRIKES,
    VOL_KEYS,
    VOLS,
    dc_instruments,
    dc_specs,
    dual_curve_chain,
    equity_chain,
    flat_chain,
)


class TestChainConstruction:
    def test_duplicate_names_rejected(self):
        node = market_inputs_node("x", spots={DEFAULT: "S"})
        with pytest.raises(ValueError, match="duplicate"):
            CalibrationChain(nodes=(node, node))

    def test_model_scope_node_rejected(self):
        node = price_fit_node(
            "m", template=BlackScholesModel(vol=jnp.asarray(0.2)),
            transforms={"vol": positive()}, basket=[], quote_keys=[],
        )
        assert node.scope == "model"
        with pytest.raises(ValueError, match="model-scope"):
            CalibrationChain(nodes=(node,))

    def test_curve_node_validates_eagerly(self):
        with pytest.raises(ValueError, match="one instrument per pillar"):
            curve_graph_node("bad", REF, dc_specs(), dc_instruments()[:3], DC_KEYS[:3])

    def test_quote_keys_union(self):
        chain, _, _ = equity_chain()
        assert chain.quote_keys[:3] == ("USD.OIS.1Y", "USD.OIS.2Y", "USD.OIS.5Y")
        assert chain.fixed_keys == ("SPX.SABR.BETA",)

    def test_missing_quote_raises(self):
        chain, quotes = flat_chain()
        with pytest.raises(KeyError):
            calibrate_market(chain, QuoteSet.from_dict({"SPX.SPOT": 1.0}))


class TestTierZero:
    def test_installs_flat_market(self):
        chain, quotes = flat_chain()
        market = calibrate_market(chain, quotes).market
        assert float(market.spot()) == 100.0
        assert float(market.dividend()) == pytest.approx(0.01)
        assert float(market.surface().vol) == pytest.approx(0.2)
        # market_inputs_node's flat curves default to reference date 0.
        df = market.discount_curve(jnp.asarray(365, jnp.int32))
        assert float(df) == pytest.approx(np.exp(-0.05), rel=1e-12)

    def test_empty_chain_returns_base(self):
        base = MarketState.flat(0.03, spot=50.0)
        out = calibrate_market(CalibrationChain(), QuoteSet(), base=base)
        assert out.market is base and bool(out.converged)


class TestCurveLevel:
    def test_matches_eager_bootstrap(self):
        """The traceable node solves the same system as bootstrap_curve_graph."""
        chain, quotes = dual_curve_chain()
        cm = calibrate_market(chain, quotes, strict=True)
        built = bootstrap_curve_graph(REF, dc_specs(), dc_instruments())
        assert bool(built.converged)
        graph = built.value
        for cid in (OIS, P3M):
            np.testing.assert_allclose(
                cm.market.curves[cid].discount_factors,
                graph[cid].discount_factors,
                rtol=1e-12,
            )
        res = cm.result("usd_curves")
        assert float(res.diagnostics.max_abs_residual) < 1e-12
        assert float(res.condition_number()) < 1e6

    def test_jit_and_vmap_over_quote_scenarios(self):
        chain, quotes = dual_curve_chain()
        shifts = jnp.array([-1e-4, 0.0, 1e-4])

        @eqx.filter_jit
        def far_df(shift):
            q = jax.tree.map(lambda v: v + shift, quotes)
            return calibrate_market(chain, q).market.curves[OIS].discount_factors[-1]

        batch = jax.vmap(far_df)(shifts)
        assert batch.shape == (3,)
        assert float(batch[0]) > float(batch[1]) > float(batch[2])


class TestSmileLevel:
    def test_fits_and_reads_upstream_forwards(self):
        chain, quotes, fixed = equity_chain()
        cm = calibrate_market(chain, quotes, fixed, strict=True)
        surface = cm.market.surfaces[DEFAULT]
        assert isinstance(surface, SABRVolSurface)
        # Forwards come from the upstream spot / curve / dividend.
        spot, div = 100.0, 0.01
        curve = cm.market.curves[DEFAULT]
        for i, t in enumerate(EXPIRIES.tolist()):
            from valax.core.binding import project_rate

            r = float(project_rate(curve, jnp.asarray(t)))
            assert float(surface.forwards[i]) == pytest.approx(
                spot * np.exp((r - div) * t), rel=1e-12
            )
        # The smile reprices the quotes to within a fit error.
        for i, t in enumerate(EXPIRIES.tolist()):
            for j, k in enumerate(STRIKES[i].tolist()):
                assert float(surface(jnp.asarray(k), jnp.asarray(t))) == pytest.approx(
                    VOLS[i][j], abs=2e-3
                )
        assert float(jnp.max(surface.betas)) == pytest.approx(0.5)

    def test_beta_comes_from_fixed_inputs(self):
        chain, quotes, fixed = equity_chain()
        cm = calibrate_market(chain, quotes, fixed.with_values({"SPX.SABR.BETA": 0.7}))
        assert float(cm.market.surfaces[DEFAULT].betas[0]) == pytest.approx(0.7)

    def test_missing_fixed_input_raises(self):
        chain, quotes, _ = equity_chain()
        with pytest.raises(KeyError):
            calibrate_market(chain, quotes)


class TestModelLevel:
    def test_implied_vol_by_price_fit(self):
        """A one-parameter price fit recovers the vol that generated the price."""
        chain, quotes = flat_chain()
        market = calibrate_market(chain, quotes).market
        opt = EuropeanOption(strike=jnp.asarray(110.0), expiry=jnp.asarray(1.0))
        target = price(opt, BlackScholesModel(vol=jnp.asarray(0.27)), market).price
        node = price_fit_node(
            "bs_vol",
            template=BlackScholesModel(vol=jnp.asarray(0.2)),
            transforms={"vol": positive()},
            basket=[opt],
            quote_keys=["OPT.PREMIUM"],
        )
        res = calibrate_model(
            node, QuoteSet.from_dict({"OPT.PREMIUM": target}), market, strict=True
        )
        assert float(res.value.vol) == pytest.approx(0.27, rel=1e-8)
        # Rate and dividend stay bound from the market at pricing time.
        assert res.value.rate is None

    def test_strict_raises_on_failure(self):
        chain, quotes = flat_chain()
        market = calibrate_market(chain, quotes).market
        opt = EuropeanOption(strike=jnp.asarray(110.0), expiry=jnp.asarray(1.0))
        node = price_fit_node(
            "bs_vol",
            template=BlackScholesModel(vol=jnp.asarray(0.2)),
            transforms={"vol": positive()},
            basket=[opt],
            quote_keys=["OPT.PREMIUM"],
            settings=SolverSettings(max_steps=1),
        )
        with pytest.raises(CalibrationError):
            calibrate_model(
                node, QuoteSet.from_dict({"OPT.PREMIUM": 5.0}), market, strict=True
            )
