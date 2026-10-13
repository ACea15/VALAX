"""Per-trade Hull-White calibration to a co-terminal basket (rates level 2).

Chain: curves → swaption cube (market scope) → Hull-White fitted per trade to
its co-terminal basket off the cube (model scope) → Bermudan priced through
``price`` (PDE). Acceptance:

* quote risk by the adjoint path (``chain_adjoint`` with the model level,
  i.e. :math:`\\lambda_m`) equals the autodiff path on every quote and fixed
  input, and both match full-recalibration finite differences;
* the basket target shifts report the vega profile across the basket, and
  it matches finite differences;
* same-shaped per-trade baskets batch under ``eqx.filter_vmap`` and
  reproduce the trade-by-trade fits.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import (
    CalibrationChain,
    calibrate_market,
    calibrate_model,
    coterminal_swaptions,
    positive,
    swaption_basket_node,
)
from valax.core.pricing import price
from valax.greeks.quote_risk import (
    quote_risk_adjoint,
    quote_risk_autodiff,
    reconcile_quote_risk,
)
from valax.instruments.rates import BermudanSwaption
from valax.market.quotes import QuoteSet
from valax.market.synthetic import (
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    rates_vol_snapshot,
)
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde.config import PDEConfig

# G2++ / Cheyette per-trade validation (appended below).
from valax.calibration.cheyette import CHEYETTE_SETTINGS
from valax.calibration.transforms import free
from valax.core.binding import bind_curves
from valax.models.cheyette import CheyetteModel
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.pde.config import PDEConfig2D

jax.config.update("jax_enable_x64", True)

RATES = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10))
VOLS = RatesVolSpec(expiry_years=(1, 2, 5), tenor_years=(2, 5), cap_years=(1,))
CFG = PDEConfig(n_spot=120, n_time=80)
A_KEY = "HW.MEAN_REVERSION"


def _bermudan(world, strike_shift=0.0, first=2):
    ref = int(world.rates.reference_date)
    disc = world.discount_id
    fwd = world.forward_id
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (first, first + 1, first + 2)],
                                 dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(first + 1, first + 6)],
                              dtype=jnp.int32),
        strike=jnp.asarray(0.01 + strike_shift),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=disc,
        forward_id=fwd,
    )


def _node(world, basket, name="hw_trade"):
    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                                volatility=jnp.asarray(0.01)),
        basket=basket,
        cube_id=world.spec.cube_id,
        transforms={"volatility": positive()},
        fixed_fields={"mean_reversion": A_KEY},
        target_keys=keys,
    )
    return node, keys


@pytest.fixture(scope="module")
def setup():
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "test"), SyntheticMarketConfig(), RATES, VOLS,
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])  # curves, cube
    trade = _bermudan(world)
    basket = coterminal_swaptions(trade)
    node, target_keys = _node(world, basket)
    fixed = snap.fixed.with_values({A_KEY: jnp.asarray(0.05)}).with_values(
        {k: jnp.asarray(0.0) for k in target_keys}
    )
    quotes = snap.quotes.select(chain.quote_keys)
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    model = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=target_keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, model=model)


def _full_reval(s):
    def pv(q, p):
        cal = calibrate_market(s["chain"], q, p)
        model = s["node"].calibrate(q, cal.market, p).value
        return price(s["trade"], model, cal.market, method="pde", config=CFG).price

    return eqx.filter_jit(pv)


class TestBasket:
    def test_coterminal_basket(self, setup):
        trade, basket = setup["trade"], setup["basket"]
        assert len(basket) == 3
        for e, sw in zip(trade.exercise_dates, basket):
            assert int(sw.expiry_date) == int(e)
            assert int(sw.fixed_dates[-1]) == int(trade.fixed_dates[-1])
            assert int(sw.fixed_dates[0]) > int(e)

    def test_jamshidian_is_reachable_and_matches_pde(self, setup):
        market = setup["calibrated"].market
        model = setup["model"].value
        sw = setup["basket"][0]
        a = price(sw, model, market).price  # auto -> analytic (Jamshidian)
        b = price(sw, model, market, method="pde", config=PDEConfig(n_spot=300, n_time=300)).price
        np.testing.assert_allclose(a, b, rtol=2e-4)

    def test_model_is_curve_free_and_converged(self, setup):
        res = setup["model"]
        assert res.value.initial_curve is None
        assert bool(res.converged)
        assert res.labels == setup["target_keys"]
        assert np.isfinite(float(res.condition_number()))


class TestModelLevelRisk:
    """λ_m through chain_adjoint vs autodiff vs full-recalibration FD."""

    @pytest.fixture(scope="class")
    def risks(self, setup):
        adj = quote_risk_adjoint(
            setup["trade"], None, setup["calibrated"], model_result=setup["model"],
            method="pde", config=CFG,
        )
        auto = quote_risk_autodiff(
            setup["trade"], None, setup["chain"], setup["quotes"], setup["fixed"],
            model_node=setup["node"], method="pde", config=CFG,
        )
        return adj, auto

    def test_adjoint_matches_autodiff(self, risks):
        adj, auto = risks
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

    def test_vega_flows_to_the_cube_quotes(self, setup, risks):
        adj, _ = risks
        vol_keys = [k for k in adj.quotes.values if ".SWPN." in k]
        assert max(abs(float(adj.quotes[k])) for k in vol_keys) > 0.0
        # The vega profile across the basket: a long Bermudan payer is long
        # every co-terminal European it can exercise into.
        profile = [float(adj.fixed[k]) for k in setup["target_keys"]]
        assert all(p > 0.0 for p in profile), profile

    def test_matches_full_recalibration_fd(self, setup, risks):
        adj, _ = risks
        pv = _full_reval(setup)
        q, p = setup["quotes"], setup["fixed"]
        vol_key = "USD.SWPN.2Yx5Y.+0"
        cases = [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.IRS3M.7Y", 1e-5),
                 ("q", vol_key, 2e-6), ("p", A_KEY, 1e-4),
                 ("p", setup["target_keys"][1], 1e-6)]
        for kind, key, h in cases:
            if kind == "q":
                up = pv(q.with_values({key: q[key] + h}), p)
                dn = pv(q.with_values({key: q[key] - h}), p)
                got = float(adj.quotes[key])
            else:
                up = pv(q, p.with_values({key: p[key] + h}))
                dn = pv(q, p.with_values({key: p[key] - h}))
                got = float(adj.fixed[key])
            fd = float((up - dn) / (2 * h))
            assert abs(fd) > 0.0, key
            np.testing.assert_allclose(got, fd, rtol=5e-4, err_msg=key)


class TestBatching:
    def test_filter_vmap_over_same_shaped_baskets(self, setup):
        world = setup["world"]
        market = setup["calibrated"].market
        trades = [_bermudan(world, 0.0), _bermudan(world, 0.004)]
        baskets = [coterminal_swaptions(t) for t in trades]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
        fixed = setup["fixed"]

        def fit(basket):
            node, _ = _node(world, basket)
            return node.calibrate(QuoteSet(), market, fixed)

        batched = eqx.filter_vmap(fit)(stacked)
        assert batched.value.volatility.shape == (2,)
        assert batched.value.initial_curve is None
        assert bool(jnp.all(batched.diagnostics.converged))
        for i, b in enumerate(baskets):
            one = fit(b).value.volatility
            np.testing.assert_allclose(batched.value.volatility[i], one, rtol=1e-10)


# ── G2++ per trade (two factors, basket through the registered 2-D PDE) ──
#
# ``price`` has no analytic G2++ swaption route, so the basket is priced by
# ``price(..., method="pde")`` (the 2-D ADI recipe), as is the Bermudan. The
# fit is on-engine: basket and Bermudan share the grid. (σ, η) are fitted;
# a, b and ρ are fixed inputs. The co-terminal options alone leave the
# fast/slow factor split unidentified, so the basket adds the 1Y-tenor
# swaption at the first exercise.
#
# The discrete PDE price is only piecewise smooth in its inputs (the payoff
# ``max(swap, 0)`` is sampled at grid nodes, so a node crossing the exercise
# boundary is a kink), so the finite differences below use steps small
# enough to stay inside one smooth piece; adjoint and autodiff are the exact
# derivative of the discrete map.

G2_CFG = PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0)
G2_FIXED = {"G2.A": 0.8, "G2.B": 0.03, "G2.RHO": -0.7}


def _g2_basket(trade):
    co = coterminal_swaptions(trade)
    short = eqx.tree_at(lambda s: s.fixed_dates, co[0], co[0].fixed_dates[:1])
    return co + (short,)


def _g2_node(world, basket, name="g2_trade"):
    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=G2PPModel(
            mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
            volatility_x=jnp.asarray(0.006), volatility_y=jnp.asarray(0.006),
            correlation=jnp.asarray(-0.7),
        ),
        basket=basket,
        cube_id=world.spec.cube_id,
        transforms={"volatility_x": positive(), "volatility_y": positive()},
        fixed_fields={"mean_reversion_x": "G2.A", "mean_reversion_y": "G2.B",
                      "correlation": "G2.RHO"},
        target_keys=keys,
        method="pde",
        config=G2_CFG,
    )
    return node, keys


@pytest.fixture(scope="module")
def g2_setup(setup):
    world, chain, quotes = setup["world"], setup["chain"], setup["quotes"]
    trade = setup["trade"]
    basket = _g2_basket(trade)
    node, target_keys = _g2_node(world, basket)
    fixed = setup["fixed"].with_values(
        {k: jnp.asarray(v) for k, v in G2_FIXED.items()}
    ).with_values({k: jnp.asarray(0.0) for k in target_keys})
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    model = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=target_keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, model=model)


def _full_reval_cfg(s, cfg):
    def pv(q, p):
        cal = calibrate_market(s["chain"], q, p)
        model = s["node"].calibrate(q, cal.market, p).value
        return price(s["trade"], model, cal.market, method="pde", config=cfg).price

    return eqx.filter_jit(pv)


def _check_fd(s, adj, pv, cases, rtol):
    q, p = s["quotes"], s["fixed"]
    for kind, key, h in cases:
        if kind == "q":
            up = pv(q.with_values({key: q[key] + h}), p)
            dn = pv(q.with_values({key: q[key] - h}), p)
            got = float(adj.quotes[key])
        else:
            up = pv(q, p.with_values({key: p[key] + h}))
            dn = pv(q, p.with_values({key: p[key] - h}))
            got = float(adj.fixed[key])
        fd = float((up - dn) / (2 * h))
        assert abs(fd) > 0.0, key
        np.testing.assert_allclose(got, fd, rtol=rtol, err_msg=key)


class TestG2PPPerTrade:
    def test_short_tenor_option_makes_the_fit_well_posed(self, g2_setup):
        s = g2_setup
        market = s["calibrated"].market
        co_node, co_keys = _g2_node(s["world"], s["basket"][:3], name="g2_co")
        fixed = s["fixed"].with_values({k: jnp.asarray(0.0) for k in co_keys})
        co = co_node.calibrate(QuoteSet(), market, fixed)
        assert float(co.condition_number()) > 1e5
        assert float(s["model"].condition_number()) < 1e4

    def test_model_is_curve_free_and_converged(self, g2_setup):
        res = g2_setup["model"]
        assert res.value.initial_curve is None
        assert bool(res.converged)
        assert res.labels == g2_setup["target_keys"]

    def test_pde_basket_matches_brigo_mercurio(self, g2_setup):
        s = g2_setup
        market = s["calibrated"].market
        model = s["model"].value
        disc = market.curve(s["world"].discount_id)
        fwd = market.curve(s["world"].forward_id)
        for sw in s["basket"]:
            pde = price(sw, model, market, method="pde", config=G2_CFG).price
            exact = g2pp_swaption_price(sw, bind_curves(model, disc), fwd)
            np.testing.assert_allclose(pde, exact, rtol=1e-2)

    @pytest.fixture(scope="class")
    def g2_risks(self, g2_setup):
        s = g2_setup
        adj = quote_risk_adjoint(
            s["trade"], None, s["calibrated"], model_result=s["model"],
            method="pde", config=G2_CFG,
        )
        auto = quote_risk_autodiff(
            s["trade"], None, s["chain"], s["quotes"], s["fixed"],
            model_node=s["node"], method="pde", config=G2_CFG,
        )
        return adj, auto

    def test_adjoint_matches_autodiff(self, g2_risks):
        adj, auto = g2_risks
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)
        assert float(adj.level_residuals["g2_trade"]) < 1e-8

    def test_matches_full_recalibration_fd(self, g2_setup, g2_risks):
        adj, _ = g2_risks
        # Steps sit between two floors: large enough that the recalibration
        # noise (BFGS stops at stationarity ~1e-9, i.e. ~1e-11 in price) is
        # negligible, small enough to stay inside one smooth piece of the
        # discrete PDE map. The fixed input checked is b (the slow factor's
        # mean reversion): ∂V/∂a is ~3e-5, too small to resolve above the
        # recalibration noise.
        cases = [("q", "USD.OIS.5Y", 1e-6), ("q", "USD.SWPN.2Yx5Y.+0", 1e-6),
                 ("p", "G2.B", 1e-5), ("p", g2_setup["target_keys"][1], 1e-6)]
        _check_fd(g2_setup, adj, _full_reval_cfg(g2_setup, G2_CFG), cases, rtol=5e-4)

    def test_filter_vmap_over_same_shaped_baskets(self, g2_setup):
        s = g2_setup
        world, market, fixed = s["world"], s["calibrated"].market, s["fixed"]
        trades = [_bermudan(world, 0.0), _bermudan(world, 0.004)]
        baskets = [_g2_basket(t) for t in trades]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)

        def fit(basket):
            node, _ = _g2_node(world, basket)
            return node.calibrate(QuoteSet(), market, fixed)

        batched = eqx.filter_vmap(fit)(stacked)
        assert batched.value.volatility_x.shape == (2,)
        assert batched.value.initial_curve is None
        assert bool(jnp.all(batched.diagnostics.converged))
        one_fit = eqx.filter_jit(fit)
        for i, b in enumerate(baskets):
            one = one_fit(b).value
            # Same BFGS on PDE prices whose reduction order differs under
            # vmap: the two stop ~1e-9..3e-8 apart (relative) in the level's
            # flattest direction, inside the solver tolerance — not
            # bit-identical like the closed-form HW fit.
            for f in ("volatility_x", "volatility_y"):
                np.testing.assert_allclose(
                    getattr(batched.value, f)[i], getattr(one, f), rtol=1e-7
                )


# ── Cheyette per trade (frozen-y 1-D PDE, single-curve market) ──────────
#
# The Cheyette PDE recipes take no projection curve (``basis_overlay`` is not
# registered for Cheyette), so ``price`` refuses a trade naming a projection
# curve distinct from its discount curve. The Cheyette level therefore runs
# on a single-curve market: no projection tenors, the OIS curve discounts and
# projects, and the trades carry ``forward_id=None``. (σ, β) are fitted —
# β needs a strike slice, so the co-terminal basket adds its first option at
# K + 100bp — with κ a fixed input.

CH_CFG = PDEConfig(n_spot=101, n_time=60, spot_range=5.0)
CH_KAPPA = "CH.KAPPA"


def _single_curve_bermudan(world, strike=0.01):
    ref = int(world.rates.reference_date)
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id,
        forward_id=None,
    )


@pytest.fixture(scope="module")
def ch_setup():
    rates = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10), projection_tenors=())
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "test"), SyntheticMarketConfig(), rates, VOLS,
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])
    trade = _single_curve_bermudan(world)
    basket = coterminal_swaptions(trade) + coterminal_swaptions(
        trade, strike=trade.strike + 0.01)[:1]
    keys = tuple(f"ch_trade.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        "ch_trade",
        template=CheyetteModel(mean_reversion=jnp.asarray(0.03),
                               volatility=jnp.asarray(0.008), skew=jnp.asarray(0.0)),
        basket=basket,
        cube_id=VOLS.cube_id,
        transforms={"volatility": positive(), "skew": free()},
        fixed_fields={"mean_reversion": CH_KAPPA},
        target_keys=keys,
        method="pde",
        config=CH_CFG,
        settings=CHEYETTE_SETTINGS,
    )
    fixed = snap.fixed.with_values({CH_KAPPA: jnp.asarray(0.03)}).with_values(
        {k: jnp.asarray(0.0) for k in keys}
    )
    quotes = snap.quotes.select(chain.quote_keys)
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    model = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, model=model)


class TestCheyettePerTrade:
    def test_refuses_a_distinct_projection_curve(self, setup):
        model = CheyetteModel(mean_reversion=jnp.asarray(0.03),
                              volatility=jnp.asarray(0.008), skew=jnp.asarray(0.0))
        with pytest.raises(ValueError, match="projection curve"):
            price(setup["trade"], model, setup["calibrated"].market,
                  method="pde", config=CH_CFG)

    def test_model_is_curve_free_and_converged(self, ch_setup):
        res = ch_setup["model"]
        assert ch_setup["world"].forward_id == ch_setup["world"].discount_id
        assert res.value.initial_curve is None
        assert bool(res.converged)
        pv = price(ch_setup["trade"], res.value, ch_setup["calibrated"].market,
                   method="pde", config=CH_CFG).price
        euro = max(
            float(price(sw, res.value, ch_setup["calibrated"].market,
                        method="pde", config=CH_CFG).price)
            for sw in ch_setup["basket"][:3]
        )
        assert float(pv) >= euro

    @pytest.fixture(scope="class")
    def ch_risks(self, ch_setup):
        s = ch_setup
        adj = quote_risk_adjoint(
            s["trade"], None, s["calibrated"], model_result=s["model"],
            method="pde", config=CH_CFG,
        )
        auto = quote_risk_autodiff(
            s["trade"], None, s["chain"], s["quotes"], s["fixed"],
            model_node=s["node"], method="pde", config=CH_CFG,
        )
        return adj, auto

    def test_adjoint_matches_autodiff(self, ch_risks):
        adj, auto = ch_risks
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

    def test_matches_full_recalibration_fd(self, ch_setup, ch_risks):
        adj, _ = ch_risks
        cases = [("q", "USD.OIS.5Y", 1e-6), ("q", "USD.SWPN.2Yx5Y.+0", 1e-6),
                 ("p", CH_KAPPA, 1e-4), ("p", ch_setup["target_keys"][1], 1e-6)]
        _check_fd(ch_setup, adj, _full_reval_cfg(ch_setup, CH_CFG), cases, rtol=5e-4)
