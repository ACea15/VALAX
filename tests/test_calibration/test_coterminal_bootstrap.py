"""Per-trade co-terminal bootstrap: piecewise σ(t), one bucket per exercise.

``swaption_basket_node(mode="coterminal_bootstrap")`` gives Hull-White one
volatility bucket per co-terminal (breakpoints at the basket expiries), so the
level is square and lower triangular and reprices every co-terminal exactly.
Solved bucket by bucket by :func:`~valax.calibration.solve.triangular_root_level`:

* every co-terminal repriced to ≤ 1e-10 (relative);
* :math:`\\partial R/\\partial\\theta` strictly lower triangular;
* adjoint (λ_m, one per exercise date) = autodiff = full-recalibration FD;
* ``filter_vmap`` over trades with different exercise dates;
* an infeasible co-terminal is named as the root cause, never priced, and the
  healthy bucket before it stays exact.
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
from valax.dates.daycounts import year_fraction
from valax.greeks.quote_risk import (
    quote_risk_adjoint,
    quote_risk_autodiff,
    reconcile_quote_risk,
)
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.market.quotes import QuoteSet
from valax.market.synthetic import (
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    rates_vol_snapshot,
)
from valax.models.sabr import SABRModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde.config import PDEConfig, PDEConfig2D

jax.config.update("jax_enable_x64", True)

RATES = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10))
VOLS = RatesVolSpec(expiry_years=(1, 2, 5), tenor_years=(2, 5), cap_years=(1,))
CFG = PDEConfig(n_spot=120, n_time=80)
A_KEY = "HW.MEAN_REVERSION"


def _bermudan(world, strike_shift=0.0, first=2, n_ex=3):
    ref = int(world.rates.reference_date)
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in range(first, first + n_ex)],
                                 dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(first + 1, first + n_ex + 3)],
                              dtype=jnp.int32),
        strike=jnp.asarray(0.01 + strike_shift),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id,
        forward_id=world.forward_id,
    )


def _node(world, basket, name="hw_boot", **kwargs):
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
        mode="coterminal_bootstrap",
        **kwargs,
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
    node, keys = _node(world, basket)
    fixed = snap.fixed.with_values({A_KEY: jnp.asarray(0.05)}).with_values(
        {k: jnp.asarray(0.0) for k in keys}
    )
    quotes = snap.quotes.select(chain.quote_keys)
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, result=result)


def _basket_errors(setup, model):
    from valax.pricing.analytic.rates_smile import swaption_price
    from valax.core.resolve import discount_id_of, forward_id_of

    market = setup["calibrated"].market
    cube = market.surfaces[setup["world"].spec.cube_id]
    errs = []
    for sw in setup["basket"]:
        fwd = forward_id_of(sw)
        target = swaption_price(sw, market.curve(discount_id_of(sw)), cube,
                                None if fwd is None else market.curve(fwd))
        errs.append(float(price(sw, model, market).price / target - 1.0))
    return np.asarray(errs)


class TestExactBootstrap:
    def test_reprices_every_coterminal(self, setup):
        res = setup["result"]
        assert bool(res.converged)
        assert res.problem == "root"
        assert np.max(np.abs(_basket_errors(setup, res.value))) < 1e-10

    def test_one_bucket_per_exercise(self, setup):
        res = setup["result"]
        model, trade = res.value, setup["trade"]
        n = len(setup["basket"])
        assert model.volatility.shape == (n,)
        ref = setup["calibrated"].market.curve(setup["world"].discount_id).reference_date
        expected = year_fraction(ref, trade.exercise_dates[:-1], trade.day_count)
        np.testing.assert_allclose(model.volatility_times, expected, rtol=0, atol=0)
        # λ_m is indexed like theta and labelled by exercise (target keys).
        assert res.theta.shape == (n,)
        assert res.labels == setup["target_keys"]
        assert model.initial_curve is None

    def test_jacobian_lower_triangular(self, setup):
        res = setup["result"]
        jac = np.asarray(jax.jacfwd(res.residual_fn)(res.theta, res.inputs))
        assert np.all(np.triu(jac, 1) == 0.0)
        assert np.all(np.abs(np.diag(jac)) > 0.0)
        assert np.isfinite(float(res.condition_number()))

    def test_every_bucket_reports_its_own_steps(self, setup):
        steps = np.asarray(setup["result"].diagnostics.n_steps)
        assert steps.shape == (len(setup["basket"]),)
        assert np.all(steps > 0)


def test_least_squares_misses_what_bootstrap_hits(setup):
    """The motivation: constant σ cannot match the strip (~1% here)."""
    lsq = swaption_basket_node(
        "hw_lsq",
        template=HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01)),
        basket=setup["basket"], cube_id=setup["world"].spec.cube_id,
        transforms={"volatility": positive()}, fixed_fields={"mean_reversion": A_KEY},
    )
    fit = calibrate_model(lsq, setup["quotes"], setup["calibrated"].market, setup["fixed"])
    assert np.max(np.abs(_basket_errors(setup, fit.value))) > 1e-4


def _two_trade_baskets(world):
    """Two same-shaped trades with different exercise dates."""
    trades = [_bermudan(world, 0.0, first=2, n_ex=2),
              _bermudan(world, 0.003, first=3, n_ex=2)]
    return [coterminal_swaptions(t) for t in trades]


def test_breakpoints_are_traced_data_fast(setup):
    """Fast sibling of the vmap test: one compiled fit serves both trades.

    The breakpoints come from the (traced) basket, so the same jitted
    bootstrap reprices trades with different exercise dates, each on its own
    buckets.
    """
    world, market, fixed = setup["world"], setup["calibrated"].market, setup["fixed"]

    @eqx.filter_jit
    def fit(basket):
        node, _ = _node(world, basket)
        return node.calibrate(QuoteSet(), market, fixed)

    results = [fit(b) for b in _two_trade_baskets(world)]
    for r in results:
        assert bool(r.converged)
        assert float(jnp.max(jnp.abs(r.diagnostics.residuals))) < 1e-10
    assert not np.allclose(results[0].value.volatility_times,
                           results[1].value.volatility_times)


@pytest.mark.slow
def test_filter_vmap_over_trades_with_different_dates(setup):
    """Same shape, different exercise grids: the breakpoints are traced data.

    Two-exercise trades keep the compile small; still ~17 s (the vmapped
    bootstrap's compile), so slow, with ``test_breakpoints_are_traced_data_fast``
    as its fast sibling. The 3-exercise G2++ variant is in ``TestG2PPBootstrap``.
    """
    world, market, fixed = setup["world"], setup["calibrated"].market, setup["fixed"]
    baskets = _two_trade_baskets(world)
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)

    def fit(basket):
        node, _ = _node(world, basket)
        return node.calibrate(QuoteSet(), market, fixed)

    batched = eqx.filter_vmap(fit)(stacked)
    assert bool(jnp.all(batched.diagnostics.converged))
    fit_one = eqx.filter_jit(fit)  # compiled once, reused for both baskets
    for i, b in enumerate(baskets):
        one = fit_one(b).value
        np.testing.assert_allclose(batched.value.volatility[i], one.volatility, rtol=1e-10)
        np.testing.assert_allclose(batched.value.volatility_times[i], one.volatility_times,
                                   rtol=0, atol=0)
    assert not np.allclose(batched.value.volatility_times[0], batched.value.volatility_times[1])


def test_infeasible_coterminal_is_named(setup):
    """Anti-vacuity: a target below the σ→0 price cannot be met.

    Bucket 1's target is shifted to a tenth of its value; the variance
    carried over from bucket 0 alone already prices above it. The level must
    fail, name bucket 1 as the root cause, and install NaN from it on — while
    the healthy bucket before it stays exact, unflagged, and reports its own
    finite residual (not the 0·NaN leak of the later buckets' NaN through the
    triangular primitive).
    """
    node = setup["node"]
    keys = setup["target_keys"]
    market = setup["calibrated"].market
    good = setup["result"]
    target_1 = float(price(setup["basket"][1], good.value, market).price)
    bad_fixed = setup["fixed"].with_values({keys[1]: jnp.asarray(-0.9 * target_1)})
    res = calibrate_model(node, setup["quotes"], market, bad_fixed)
    assert not bool(res.converged)
    assert res.root_cause_labels() == (keys[1],)
    assert set(res.failed_labels()) == {keys[1], keys[2]}
    theta = np.asarray(res.theta)
    assert np.all(np.isnan(theta[1:]))
    assert theta[0] == pytest.approx(float(good.theta[0]), abs=1e-14)
    assert abs(float(np.ravel(res.diagnostics.residuals)[0])) < 1e-12


class TestValidation:
    def _kw(self, setup):
        return dict(template=HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                                            volatility=jnp.asarray(0.01)),
                    basket=setup["basket"], cube_id=setup["world"].spec.cube_id,
                    transforms={"volatility": positive()}, mode="coterminal_bootstrap")

    def test_weights_rejected(self, setup):
        with pytest.raises(ValueError, match="weights"):
            swaption_basket_node("x", weights=jnp.ones(3), **self._kw(setup))

    def test_one_field_only(self, setup):
        kw = self._kw(setup)
        kw["transforms"] = {"volatility": positive(), "mean_reversion": positive()}
        with pytest.raises(ValueError, match="one bucketed volatility field"):
            swaption_basket_node("x", **kw)

    def test_increasing_expiries(self, setup):
        kw = self._kw(setup)
        kw["basket"] = tuple(reversed(setup["basket"]))
        with pytest.raises(ValueError, match="strictly increasing"):
            swaption_basket_node("x", **kw)

    def test_unknown_mode(self, setup):
        kw = self._kw(setup)
        kw["mode"] = "bootstrap"
        with pytest.raises(ValueError, match="mode"):
            swaption_basket_node("x", **kw)

    def test_model_without_buckets(self, setup):
        # Every short-rate model now carries bucketed volatilities; SABR has
        # no volatility term structure to bootstrap.
        kw = self._kw(setup)
        kw["template"] = SABRModel(
            alpha=jnp.asarray(0.02), beta=jnp.asarray(0.5),
            rho=jnp.asarray(-0.2), nu=jnp.asarray(0.3),
        )
        kw["transforms"] = {"alpha": positive()}
        with pytest.raises(TypeError, match="volatility_times"):
            swaption_basket_node("x", **kw)

    def test_default_labels_are_exercise_dates(self, setup):
        kw = self._kw(setup)
        node = swaption_basket_node("x", **kw)
        res = node.calibrate(setup["quotes"], setup["calibrated"].market, QuoteSet())
        assert all(label.startswith("exercise 20") for label in res.labels)


def _european_trade(setup):
    """A European expiring inside bucket 1 (Jamshidian: cheap quote risk).

    Buckets are ``[0, T_1)``, ``[T_1, T_2)``, ``[T_2, ∞)`` with ``T_k`` the
    exercise dates; expiring at ``T_1 + 180d`` it reads buckets 0 and 1 only.
    """
    trade = setup["trade"]
    return Swaption(
        expiry_date=trade.exercise_dates[0] + 180,
        fixed_dates=trade.fixed_dates[1:],
        strike=trade.strike, notional=trade.notional, day_count=trade.day_count,
        discount_id=trade.discount_id, forward_id=trade.forward_id,
    )


@pytest.mark.slow
def test_adjoint_matches_autodiff_european(setup):
    """Quote risk through the bootstrap on a cheap trade (slow: ~25 s).

    Also the triangular structure of λ_m: the trade cannot see the last
    bucket, so its exposure to the last co-terminal's target is exactly 0.
    """
    trade = _european_trade(setup)
    adj = quote_risk_adjoint(trade, None, setup["calibrated"],
                             model_result=setup["result"])
    auto = quote_risk_autodiff(trade, None, setup["chain"], setup["quotes"], setup["fixed"],
                               model_node=setup["node"])
    rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
    assert rec.ok, (rec.mismatched, rec.max_abs_diff)
    scale = abs(float(adj.fixed[setup["target_keys"][0]]))
    assert scale > 0.0
    assert abs(float(adj.fixed[setup["target_keys"][-1]])) < 1e-12 * scale


def test_vega_profile_is_triangular_fast(setup):
    """Fast sibling: the adjoint path alone (no autodiff compile).

    The European reads buckets 0 and 1, so it is exposed to the first two
    co-terminal targets and not at all to the last.
    """
    adj = quote_risk_adjoint(_european_trade(setup), None, setup["calibrated"],
                             model_result=setup["result"])
    profile = np.array([float(adj.fixed[k]) for k in setup["target_keys"]])
    scale = np.max(np.abs(profile))
    assert abs(profile[0]) > 1e-3 * scale and abs(profile[1]) > 1e-3 * scale
    # Exactly zero in exact arithmetic; chain_adjoint's dense LU leaves
    # roundoff (measured 2e-17).
    assert abs(profile[2]) < 1e-12 * scale


@pytest.mark.slow
class TestBermudanRisk:
    """λ_m (adjoint) = autodiff = full-recalibration FD (slow)."""

    @pytest.fixture(scope="class")
    def risks(self, setup):
        adj = quote_risk_adjoint(
            setup["trade"], None, setup["calibrated"],
            model_result=setup["result"], method="pde", config=CFG,
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

    def test_vega_profile_one_per_exercise(self, setup, risks):
        adj, _ = risks
        profile = [float(adj.fixed[k]) for k in setup["target_keys"]]
        assert len(profile) == len(setup["trade"].exercise_dates)
        assert all(p > 0.0 for p in profile), profile

    def test_matches_full_recalibration_fd(self, setup, risks):
        adj, _ = risks
        node = setup["node"]

        def pv(q, p):
            cal = calibrate_market(setup["chain"], q, p)
            model = node.calibrate(q, cal.market, p).value
            return price(setup["trade"], model, cal.market, method="pde", config=CFG).price

        pv = eqx.filter_jit(pv)
        q, p = setup["quotes"], setup["fixed"]
        cases = [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.SWPN.2Yx5Y.+0", 1e-5),
                 ("p", A_KEY, 1e-4), ("p", setup["target_keys"][1], 1e-5)]
        for kind, key, h in cases:
            if kind == "q":
                up, dn = pv(q.with_values({key: q[key] + h}), p), pv(q.with_values({key: q[key] - h}), p)
                got = float(adj.quotes[key])
            else:
                up, dn = pv(q, p.with_values({key: p[key] + h})), pv(q, p.with_values({key: p[key] - h}))
                got = float(adj.fixed[key])
            fd = float((up - dn) / (2 * h))
            assert abs(fd) > 0.0, key
            np.testing.assert_allclose(got, fd, rtol=5e-4, err_msg=key)


# ── G2++: σ(t) bootstrapped at a fixed volatility ratio ────────────────
#
# With a, b, ρ and the ratio r = η/σ held fixed, one σ bucket per
# co-terminal makes the G2++ level square and triangular too — no extra
# short-tenor option is needed to identify the two factors (contrast the
# least-squares set-up in test_per_trade_basket.py). The ratio is a fixed
# input, so its sensitivity is reported on both risk paths.

G2_FIXED = {"G2.A": 0.8, "G2.B": 0.03, "G2.RHO": -0.7, "G2.RATIO": 2.0}


def _g2_node(world, basket, name="g2_boot"):
    from valax.models.g2pp import G2PPModel

    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=G2PPModel(
            mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
            volatility_x=jnp.asarray(0.006), volatility_ratio=jnp.asarray(2.0),
            correlation=jnp.asarray(-0.7),
        ),
        basket=basket,
        cube_id=world.spec.cube_id,
        transforms={"volatility_x": positive()},
        fixed_fields={"mean_reversion_x": "G2.A", "mean_reversion_y": "G2.B",
                      "correlation": "G2.RHO", "volatility_ratio": "G2.RATIO"},
        target_keys=keys,
        mode="coterminal_bootstrap",
    )
    return node, keys


def _g2_state(setup, basket):
    world, market = setup["world"], setup["calibrated"].market
    node, keys = _g2_node(world, basket)
    fixed = setup["fixed"].with_values(
        {k: jnp.asarray(v) for k, v in G2_FIXED.items()}
    ).with_values({k: jnp.asarray(0.0) for k in keys})
    result = calibrate_model(node, setup["quotes"], market, fixed, strict=True)
    return dict(setup, basket=basket, node=node, target_keys=keys, fixed=fixed,
                result=result)


@pytest.fixture(scope="module")
def g2_small(setup):
    """Fast sibling: a 2-exercise trade."""
    basket = coterminal_swaptions(_bermudan(setup["world"], n_ex=2))
    return _g2_state(setup, basket)


@pytest.fixture(scope="module")
def g2_setup(setup):
    """The 3-exercise trade (slow to compile)."""
    return _g2_state(setup, setup["basket"])


class TestG2PPBootstrapFast:
    def test_reprices_every_coterminal(self, g2_small):
        res = g2_small["result"]
        assert bool(res.converged)
        assert np.max(np.abs(_basket_errors(g2_small, res.value))) < 1e-10

    def test_ratio_is_held_and_broadcast(self, g2_small):
        model = g2_small["result"].value
        n = len(g2_small["basket"])
        assert model.volatility_x.shape == (n,)
        assert model.volatility_times.shape == (n - 1,)
        np.testing.assert_array_equal(model.volatility_ratio, np.full(n, 2.0))
        # The fitted σ(t) is genuinely time-dependent (the point of the bucket).
        assert np.ptp(np.asarray(model.volatility_x)) > 0.0

    def test_jacobian_lower_triangular(self, g2_small):
        res = g2_small["result"]
        jac = np.asarray(jax.jacfwd(res.residual_fn)(res.theta, res.inputs))
        assert np.all(np.triu(jac, 1) == 0.0)
        assert np.all(np.abs(np.diag(jac)) > 0.0)


@pytest.mark.slow
class TestG2PPBootstrap:
    def test_reprices_every_coterminal(self, g2_setup):
        res = g2_setup["result"]
        assert bool(res.converged)
        assert np.max(np.abs(_basket_errors(g2_setup, res.value))) < 1e-10

    def test_risk_adjoint_autodiff_and_fd(self, g2_setup):
        """λ_m (adjoint) = autodiff = full-recalibration FD, incl. the ratio (slow)."""
        s = g2_setup
        trade = _european_trade(s)
        node, res = s["node"], s["result"]
        adj = quote_risk_adjoint(trade, None, s["calibrated"], model_result=res)
        auto = quote_risk_autodiff(trade, None, s["chain"], s["quotes"], s["fixed"],
                                   model_node=node)
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

        def pv(q, p):
            cal = calibrate_market(s["chain"], q, p)
            model = node.calibrate(q, cal.market, p).value
            return price(trade, model, cal.market).price

        pv = eqx.filter_jit(pv)
        q, p = s["quotes"], s["fixed"]
        for kind, key, h in [("q", "USD.SWPN.2Yx5Y.+0", 1e-5), ("p", "G2.RATIO", 1e-4),
                             ("p", "G2.RHO", 1e-4), ("p", s["target_keys"][0], 1e-5)]:
            if kind == "q":
                up, dn = pv(q.with_values({key: q[key] + h}), p), pv(q.with_values({key: q[key] - h}), p)
                got = float(adj.quotes[key])
            else:
                up, dn = pv(q, p.with_values({key: p[key] + h})), pv(q, p.with_values({key: p[key] - h}))
                got = float(adj.fixed[key])
            fd = float((up - dn) / (2 * h))
            assert abs(fd) > 0.0, key
            np.testing.assert_allclose(got, fd, rtol=5e-4, err_msg=key)
        # The held ratio is a risk input of the trade.
        assert float(adj.fixed["G2.RATIO"]) != 0.0

    def test_filter_vmap_over_trades_with_different_dates(self, g2_setup):
        s = g2_setup
        trades = [_bermudan(s["world"], 0.0, first=2), _bermudan(s["world"], 0.003, first=3)]
        baskets = [coterminal_swaptions(t) for t in trades]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)

        def fit(basket):
            node, _ = _g2_node(s["world"], basket)
            return node.calibrate(QuoteSet(), s["calibrated"].market, s["fixed"])

        batched = eqx.filter_vmap(fit)(stacked)
        assert bool(jnp.all(batched.diagnostics.converged))
        for i, b in enumerate(baskets):
            np.testing.assert_allclose(batched.value.volatility_x[i], fit(b).value.volatility_x,
                                       rtol=1e-10)


# ── Cheyette (1F): σ(t) bootstrapped on the frozen-y PDE, β held fixed ──
#
# Cheyette prices single-curve only, so this runs on a market without
# projection tenors. The basket is priced on the frozen-y PDE (the Bermudan's
# engine), so the fit is engine-bound (modelling-bias provenance) and every
# co-terminal is repriced exactly *on that grid*.

CH_CFG = PDEConfig(n_spot=101, n_time=60, spot_range=5.0)
CH_FIXED = {"CH.KAPPA": 0.03, "CH.BETA": 0.3}


def _ch_node(world, basket, name="ch_boot"):
    from valax.models.cheyette import CheyetteModel

    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=CheyetteModel(mean_reversion=jnp.asarray(0.03),
                               volatility=jnp.asarray(0.008), skew=jnp.asarray(0.3)),
        basket=basket,
        cube_id=VOLS.cube_id,
        transforms={"volatility": positive()},
        fixed_fields={"mean_reversion": "CH.KAPPA", "skew": "CH.BETA"},
        target_keys=keys,
        mode="coterminal_bootstrap",
        method="pde",
        config=CH_CFG,
    )
    return node, keys


@pytest.fixture(scope="module")
def ch_setup():
    rates = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10), projection_tenors=())
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "test"), SyntheticMarketConfig(), rates, VOLS,
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])
    ref = int(world.rates.reference_date)
    trade = BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(0.01), notional=jnp.asarray(1.0),
        day_count=world.rates.day_count, discount_id=world.discount_id, forward_id=None,
    )
    basket = coterminal_swaptions(trade)
    node, keys = _ch_node(world, basket)
    fixed = snap.fixed.with_values(
        {k: jnp.asarray(v) for k, v in CH_FIXED.items()}
    ).with_values({k: jnp.asarray(0.0) for k in keys})
    quotes = snap.quotes.select(chain.quote_keys)
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, result=result)


def _engine_basket_errors(s, model):
    """Basket repricing error on the calibration engine (the frozen-y PDE)."""
    from valax.pricing.analytic.rates_smile import swaption_price

    market = s["calibrated"].market
    disc = market.curve(s["world"].discount_id)
    cube = market.surfaces[VOLS.cube_id]
    return np.asarray([
        float(price(sw, model, market, method="pde", config=CH_CFG).price
              / swaption_price(sw, disc, cube) - 1.0)
        for sw in s["basket"]
    ])


class TestCheyetteBootstrap:
    def test_reprices_every_coterminal_on_its_engine(self, ch_setup):
        res = ch_setup["result"]
        assert bool(res.converged)
        assert np.max(np.abs(_engine_basket_errors(ch_setup, res.value))) < 1e-10
        model = res.value
        assert model.volatility.shape == (3,)
        assert float(model.skew) == pytest.approx(0.3)

    def test_fit_is_engine_bound(self, ch_setup):
        from valax.calibration.cheyette import cheyette_provenance
        from valax.calibration.result import CalibrationError, check_engine

        res = ch_setup["result"]
        assert res.provenance == cheyette_provenance("frozen", CH_CFG)
        assert check_engine(res, "pde", CH_CFG) is res
        with pytest.raises(CalibrationError, match="modelling bias"):
            check_engine(res, "pde", PDEConfig(n_spot=201, n_time=120, spot_range=5.0))

    def test_jacobian_lower_triangular(self, ch_setup):
        res = ch_setup["result"]
        jac = np.asarray(jax.jacfwd(res.residual_fn)(res.theta, res.inputs))
        assert np.all(np.triu(jac, 1) == 0.0)
        assert np.all(np.abs(np.diag(jac)) > 0.0)

    @pytest.mark.slow
    def test_bermudan_risk_adjoint_autodiff_and_fd(self, ch_setup):
        """λ_m on the PDE: adjoint = autodiff = full-recalibration FD (slow)."""
        s = ch_setup
        node, res = s["node"], s["result"]
        adj = quote_risk_adjoint(s["trade"], None, s["calibrated"], model_result=res,
                                 method="pde", config=CH_CFG)
        auto = quote_risk_autodiff(s["trade"], None, s["chain"], s["quotes"], s["fixed"],
                                   model_node=node, method="pde", config=CH_CFG)
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

        def pv(q, p):
            cal = calibrate_market(s["chain"], q, p)
            model = node.calibrate(q, cal.market, p).value
            return price(s["trade"], model, cal.market, method="pde", config=CH_CFG).price

        pv = eqx.filter_jit(pv)
        q, p = s["quotes"], s["fixed"]
        for kind, key, h in [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.SWPN.2Yx5Y.+0", 1e-5),
                             ("p", "CH.BETA", 1e-4), ("p", s["target_keys"][1], 1e-5)]:
            if kind == "q":
                up, dn = pv(q.with_values({key: q[key] + h}), p), pv(q.with_values({key: q[key] - h}), p)
                got = float(adj.quotes[key])
            else:
                up, dn = pv(q, p.with_values({key: p[key] + h})), pv(q, p.with_values({key: p[key] - h}))
                got = float(adj.fixed[key])
            fd = float((up - dn) / (2 * h))
            assert abs(fd) > 0.0, key
            np.testing.assert_allclose(got, fd, rtol=5e-4, err_msg=key)


# ── Cheyette 2F: σ₁(t) bootstrapped on the frozen-y 2-D PDE ─────────────
#
# κ₁, κ₂, β₁, β₂, ρ and the ratio σ₂/σ₁ held fixed (the two-factor analogue
# of the G2++ set-up); single-curve market, engine-bound fit.

CH2_CFG = PDEConfig2D(n_x=41, n_y=41, n_time=30, x_range=5.0)
CH2_FIXED = {"CH2.K1": 0.5, "CH2.K2": 0.05, "CH2.B1": 0.2, "CH2.B2": 0.1,
             "CH2.RHO": -0.6, "CH2.RATIO": 0.8}


def _ch2_state(n_ex):
    from valax.models.cheyette import CheyetteModel2F

    rates = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10), projection_tenors=())
    snap, world = rates_vol_snapshot(
        SeedRegistry(3, "test"), SyntheticMarketConfig(), rates, VOLS,
        bp_noise=0.5, vol_bp_noise=0.5,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])
    ref = int(world.rates.reference_date)
    trade = BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in range(2, 2 + n_ex)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 5 + n_ex)], dtype=jnp.int32),
        strike=jnp.asarray(0.01), notional=jnp.asarray(1.0),
        day_count=world.rates.day_count, discount_id=world.discount_id, forward_id=None,
    )
    basket = coterminal_swaptions(trade)
    keys = tuple(f"ch2_boot.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        "ch2_boot",
        template=CheyetteModel2F(
            mean_reversion_1=jnp.asarray(0.5), mean_reversion_2=jnp.asarray(0.05),
            volatility_1=jnp.asarray(0.008), volatility_ratio=jnp.asarray(0.8),
            skew_1=jnp.asarray(0.2), skew_2=jnp.asarray(0.1), correlation=jnp.asarray(-0.6),
        ),
        basket=basket, cube_id=VOLS.cube_id,
        transforms={"volatility_1": positive()},
        fixed_fields={"mean_reversion_1": "CH2.K1", "mean_reversion_2": "CH2.K2",
                      "skew_1": "CH2.B1", "skew_2": "CH2.B2", "correlation": "CH2.RHO",
                      "volatility_ratio": "CH2.RATIO"},
        target_keys=keys, mode="coterminal_bootstrap", method="pde", config=CH2_CFG,
    )
    fixed = snap.fixed.with_values(
        {k: jnp.asarray(v) for k, v in CH2_FIXED.items()}
    ).with_values({k: jnp.asarray(0.0) for k in keys})
    quotes = snap.quotes.select(chain.quote_keys)
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, result=result)


def _ch2_engine_errors(s):
    from valax.pricing.analytic.rates_smile import swaption_price

    market = s["calibrated"].market
    disc = market.curve(s["world"].discount_id)
    cube = market.surfaces[VOLS.cube_id]
    return np.asarray([
        float(price(sw, s["result"].value, market, method="pde", config=CH2_CFG).price
              / swaption_price(sw, disc, cube) - 1.0)
        for sw in s["basket"]
    ])


@pytest.fixture(scope="module")
def ch2_small():
    """Fast sibling: a 2-exercise trade on a small 2-D grid."""
    return _ch2_state(2)


class TestCheyette2FBootstrap:
    def test_reprices_every_coterminal_on_its_engine(self, ch2_small):
        res = ch2_small["result"]
        assert bool(res.converged)
        assert np.max(np.abs(_ch2_engine_errors(ch2_small))) < 1e-10
        model = res.value
        n = len(ch2_small["basket"])
        assert model.volatility_1.shape == (n,)
        np.testing.assert_array_equal(model.volatility_ratio, np.full(n, 0.8))

    def test_fit_is_engine_bound(self, ch2_small):
        res = ch2_small["result"]
        assert res.provenance.engine_kind == "modelling_bias"

    @pytest.mark.slow
    def test_bermudan_risk_adjoint_autodiff_and_fd(self):
        """3 exercises: adjoint = autodiff = full-recalibration FD (slow)."""
        s = _ch2_state(3)
        assert np.max(np.abs(_ch2_engine_errors(s))) < 1e-10
        adj = quote_risk_adjoint(s["trade"], None, s["calibrated"], model_result=s["result"],
                                 method="pde", config=CH2_CFG)
        auto = quote_risk_autodiff(s["trade"], None, s["chain"], s["quotes"], s["fixed"],
                                   model_node=s["node"], method="pde", config=CH2_CFG)
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

        def pv(q, p):
            cal = calibrate_market(s["chain"], q, p)
            model = s["node"].calibrate(q, cal.market, p).value
            return price(s["trade"], model, cal.market, method="pde", config=CH2_CFG).price

        pv = eqx.filter_jit(pv)
        q, p = s["quotes"], s["fixed"]
        for kind, key, h in [("q", "USD.SWPN.2Yx5Y.+0", 1e-5), ("p", "CH2.RATIO", 1e-4),
                             ("p", s["target_keys"][1], 1e-5)]:
            if kind == "q":
                up, dn = pv(q.with_values({key: q[key] + h}), p), pv(q.with_values({key: q[key] - h}), p)
                got = float(adj.quotes[key])
            else:
                up, dn = pv(q, p.with_values({key: p[key] + h})), pv(q, p.with_values({key: p[key] - h}))
                got = float(adj.fixed[key])
            fd = float((up - dn) / (2 * h))
            assert abs(fd) > 0.0, key
            np.testing.assert_allclose(got, fd, rtol=5e-4, err_msg=key)

    @pytest.mark.slow
    def test_filter_vmap_over_trades_with_different_dates(self, ch2_small):
        s = ch2_small
        ref = int(s["world"].rates.reference_date)
        trades = [
            eqx.tree_at(lambda t: t.strike, s["trade"], jnp.asarray(0.012)),
            eqx.tree_at(
                lambda t: (t.exercise_dates, t.fixed_dates), s["trade"],
                (s["trade"].exercise_dates + 90, s["trade"].fixed_dates + 90),
            ),
        ]
        baskets = [coterminal_swaptions(t) for t in trades]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
        assert not np.array_equal(baskets[0][0].expiry_date, baskets[1][0].expiry_date)
        assert ref > 0

        def fit(basket):
            return swaption_basket_node(
                "ch2_boot", template=s["result"].inputs.data["template"], basket=basket,
                cube_id=VOLS.cube_id, transforms={"volatility_1": positive()},
                fixed_fields={"mean_reversion_1": "CH2.K1", "mean_reversion_2": "CH2.K2",
                              "skew_1": "CH2.B1", "skew_2": "CH2.B2",
                              "correlation": "CH2.RHO", "volatility_ratio": "CH2.RATIO"},
                target_keys=s["target_keys"], mode="coterminal_bootstrap",
                method="pde", config=CH2_CFG,
            ).calibrate(QuoteSet(), s["calibrated"].market, s["fixed"])

        batched = eqx.filter_vmap(fit)(stacked)
        assert bool(jnp.all(batched.diagnostics.converged))
        for i, b in enumerate(baskets):
            np.testing.assert_allclose(batched.value.volatility_1[i], fit(b).value.volatility_1,
                                       rtol=1e-10)


def test_g2pp_unbucketed_field_rejected(setup):
    from valax.models.g2pp import G2PPModel

    with pytest.raises(ValueError, match="bucketed volatility field"):
        swaption_basket_node(
            "x",
            template=G2PPModel(
                mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
                volatility_x=jnp.asarray(0.006), volatility_ratio=jnp.asarray(2.0),
                correlation=jnp.asarray(-0.7),
            ),
            basket=setup["basket"], cube_id=setup["world"].spec.cube_id,
            transforms={"correlation": positive()}, mode="coterminal_bootstrap",
        )
