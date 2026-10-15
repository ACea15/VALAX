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

# G2++ / Cheyette per-trade validation and provenance (below).
from valax.calibration.cheyette import CHEYETTE_SETTINGS, cheyette_provenance
from valax.calibration.result import (
    CLOSED_FORM_PROVENANCE,
    CalibrationError,
    check_engine,
)
from valax.calibration.transforms import free
from valax.core.binding import bind_curves
from valax.models.cheyette import CheyetteModel
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.rates_smile import swaption_price
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


@pytest.mark.slow
class TestModelLevelRisk:
    """λ_m through chain_adjoint vs autodiff vs full-recalibration FD (slow: ~35 s)."""

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
        # One step size per input class: the PDE payoff is smoothed
        # (``smooth_payoff``), so the price is C^1 and the step need not stay
        # inside a smooth piece of the grid map.
        cases = [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.IRS3M.7Y", 1e-5),
                 ("q", vol_key, 1e-5), ("p", A_KEY, 1e-4),
                 ("p", setup["target_keys"][1], 1e-5)]
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
            # Observed ≤ 5.3e-6 (the vol quote's O(h^2) truncation).
            np.testing.assert_allclose(got, fd, rtol=5e-5, err_msg=key)


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
        assert batched.value.volatility.shape == (2, 1)
        assert batched.value.initial_curve is None
        assert bool(jnp.all(batched.diagnostics.converged))
        for i, b in enumerate(baskets):
            one = fit(b).value.volatility
            np.testing.assert_allclose(batched.value.volatility[i], one, rtol=1e-10)


class TestJittedResultDiagnostics:
    """A result returned from jit / vmap holds no stale tracer.

    The basket (and template, weights) travel as ``inputs.data`` leaves
    rather than being closed over by the static residual, so the eager
    helpers of a jitted per-trade fit still work.
    """

    @staticmethod
    def _fit_fn(setup):
        world, market, fixed = setup["world"], setup["calibrated"].market, setup["fixed"]

        def fit(basket):
            # Built inside the trace: basket and template are traced here.
            node = swaption_basket_node(
                "hw_two", template=HullWhiteModel(
                    mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01)),
                basket=basket, cube_id=world.spec.cube_id,
                transforms={"volatility": positive(), "mean_reversion": positive()},
            )
            return node.calibrate(QuoteSet(), market, fixed)

        return fit

    def test_condition_number_after_filter_jit(self, setup):
        """The jitted result's eager helper returns the right κ at its own θ.

        The oracle is an independent SVD of a forward-mode ``dR/dθ`` at the
        jitted θ (the helper uses reverse mode).
        """
        from jax.flatten_util import ravel_pytree

        jitted = eqx.filter_jit(self._fit_fn(setup))(setup["basket"])
        assert bool(jitted.converged)
        assert jitted.inputs.data is not None
        k_jit = float(jitted.condition_number())  # used to raise UnexpectedTracerError
        flat, unravel = ravel_pytree(jitted.theta)
        jac = np.asarray(jax.jacfwd(
            lambda x: jnp.ravel(jitted.residual_fn(unravel(x), jitted.inputs))
        )(flat))
        sv = np.linalg.svd(jac, compute_uv=False)
        k_oracle = sv[0] / sv[-1]
        assert np.isfinite(k_oracle) and k_oracle > 1.0
        # Measured 3.7e-11 (κ ≈ 2e6: the SVD of a near-singular J amplifies
        # roundoff).
        np.testing.assert_allclose(k_jit, k_oracle, rtol=1e-9)
        assert jitted.failed_labels() == ()

    def test_condition_number_inside_filter_vmap(self, setup):
        world = setup["world"]
        fit = self._fit_fn(setup)
        baskets = [coterminal_swaptions(_bermudan(world, s)) for s in (0.0, 0.004)]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
        kappas = eqx.filter_vmap(lambda b: fit(b).condition_number())(stacked)
        assert kappas.shape == (2,)
        for i, b in enumerate(baskets):
            # (vol, mean reversion) from one co-terminal basket is poorly
            # identified (kappa ~ 1e6), and BFGS under vmap stops at a
            # slightly different iterate: agreement to 1e-3, not 1e-6.
            np.testing.assert_allclose(
                kappas[i], float(fit(b).condition_number()), rtol=1e-3
            )

    def test_price_fit_node_after_filter_jit(self, setup):
        from valax.calibration import price_fit_node
        from valax.calibration.basket import BASKET_SETTINGS

        market, basket = setup["calibrated"].market, setup["basket"]
        targets = jnp.stack([
            price(sw, HullWhiteModel(mean_reversion=jnp.asarray(0.03),
                                     volatility=jnp.asarray(0.008)), market).price
            for sw in basket
        ])
        keys = tuple(f"T{i}" for i in range(len(basket)))
        quotes = QuoteSet.from_vector(keys, targets)

        def fit(b):
            node = price_fit_node(
                "hw_price_fit",
                template=HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                                        volatility=jnp.asarray(0.01)),
                transforms={"volatility": positive(), "mean_reversion": positive()},
                basket=b, quote_keys=keys, weights=1.0 / targets,
                settings=BASKET_SETTINGS,  # BFGS: Jamshidian needs a line search
            )
            return node.calibrate(quotes, market)

        eager = fit(basket)
        jitted = eqx.filter_jit(fit)(basket)
        np.testing.assert_allclose(
            float(jitted.condition_number()), float(eager.condition_number()),
            rtol=1e-6,
        )


class TestProvenance:
    def test_closed_form_basket_is_exact(self, setup):
        assert setup["model"].provenance == CLOSED_FORM_PROVENANCE
        assert check_engine(setup["model"], "pde", CFG) is setup["model"]

    def test_auto_resolves_per_instrument(self, setup):
        world, basket = setup["world"], setup["basket"]
        node = swaption_basket_node(
            "hw_auto",
            template=HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                                    volatility=jnp.asarray(0.01)),
            basket=basket, cube_id=world.spec.cube_id,
            transforms={"volatility": positive()},
            fixed_fields={"mean_reversion": A_KEY}, method="auto",
        )
        res = node.calibrate(QuoteSet(), setup["calibrated"].market, setup["fixed"])
        assert res.provenance == CLOSED_FORM_PROVENANCE


# ── G2++ per trade (two factors) ────────────────────────────────────────
#
# The basket is priced by the registered closed form (Brigo-Mercurio,
# ``price``'s default for Swaption × G2PPModel); the Bermudan by the 2-D ADI
# PDE. (σ, η) are fitted; a, b and ρ are fixed inputs. The co-terminal
# options alone leave the fast/slow factor split unidentified, so the basket
# adds the 1Y-tenor swaption at the first exercise.
#
# The Bermudan's PDE payoff and exercise projection are smoothed over each
# node's cell (``smooth_payoff``, the default), so the discrete price is C^1
# in its inputs and full-recalibration finite differences need no hand-tuned
# steps; adjoint and autodiff are the exact derivative of the discrete map.
#
# ``TestG2PPOnEngine`` (slow) fits the same basket on the Bermudan's own PDE
# grid instead — the engine-consistent alternative — and checks it against
# the closed-form fit and on the discretised differentiation path.

G2_CFG = PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0)
G2_FIXED = {"G2.A": 0.8, "G2.B": 0.03, "G2.RHO": -0.7}


def _g2_basket(trade):
    co = coterminal_swaptions(trade)
    short = eqx.tree_at(lambda s: s.fixed_dates, co[0], co[0].fixed_dates[:1])
    return co + (short,)


def _g2_node(world, basket, name="g2_trade", method="analytic", config=None):
    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=G2PPModel(
            mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
            volatility_x=jnp.asarray(0.006), volatility_ratio=jnp.asarray(1.0),
            correlation=jnp.asarray(-0.7),
        ),
        basket=basket,
        cube_id=world.spec.cube_id,
        transforms={"volatility_x": positive(), "volatility_ratio": positive()},
        fixed_fields={"mean_reversion_x": "G2.A", "mean_reversion_y": "G2.B",
                      "correlation": "G2.RHO"},
        target_keys=keys,
        method=method,
        config=config,
    )
    return node, keys


def _g2_state(setup, method="analytic", config=None, name="g2_trade"):
    world, chain, quotes = setup["world"], setup["chain"], setup["quotes"]
    trade = setup["trade"]
    basket = _g2_basket(trade)
    node, target_keys = _g2_node(world, basket, name, method, config)
    fixed = setup["fixed"].with_values(
        {k: jnp.asarray(v) for k, v in G2_FIXED.items()}
    ).with_values({k: jnp.asarray(0.0) for k in target_keys})
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    model = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
    return dict(world=world, chain=chain, trade=trade, basket=basket, node=node,
                target_keys=target_keys, fixed=fixed, quotes=quotes,
                calibrated=calibrated, model=model)


@pytest.fixture(scope="module")
def g2_setup(setup):
    return _g2_state(setup)


def _full_reval_cfg(s, cfg):
    def pv(q, p):
        cal = calibrate_market(s["chain"], q, p)
        model = s["node"].calibrate(q, cal.market, p).value
        return price(s["trade"], model, cal.market, method="pde", config=cfg).price

    return eqx.filter_jit(pv)


def _check_fd(s, adj, pv, cases, rtol):
    """Central FD with full recalibration; a case may add an absolute floor."""
    q, p = s["quotes"], s["fixed"]
    for kind, key, h, *atol in cases:
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
        np.testing.assert_allclose(got, fd, rtol=rtol, atol=(atol or [0.0])[0],
                                   err_msg=key)


def _condition_in_sigma_eta(res):
    """Condition number of a G2++ (σ, r) fit, measured in (σ, η) coordinates.

    The fit's unknowns are θ = (u(σ), u(r)), η = r·σ, with u the inverse
    softplus of :func:`positive`. A condition number depends on coordinates:
    this one fit measures 11,014 in θ but 4,685 in φ = (u(σ), u(η)), the
    coordinates this test's bound was set in. At the solution the fit's
    gradient vanishes, so the level Hessian maps exactly as
    :math:`H_\\varphi = A^\\top H_\\theta A`, :math:`A = \\partial\\theta/\\partial\\varphi`.
    """
    from jax.flatten_util import ravel_pytree

    spec = positive()
    flat, unravel = ravel_pytree(res.theta)
    h_theta = jax.jacfwd(lambda f: res.residual_fn(unravel(f), res.inputs))(flat)

    def theta_of(phi):
        sigma = spec.from_unconstrained(phi[0:1])
        eta = spec.from_unconstrained(phi[1:2])
        return ravel_pytree({
            "volatility_x": phi[0:1],
            "volatility_ratio": spec.to_unconstrained(eta / sigma),
        })[0]

    sigma = spec.from_unconstrained(res.theta["volatility_x"])
    ratio = spec.from_unconstrained(res.theta["volatility_ratio"])
    phi0 = jnp.concatenate([res.theta["volatility_x"], spec.to_unconstrained(ratio * sigma)])
    np.testing.assert_allclose(theta_of(phi0), flat, rtol=1e-12)
    a = jax.jacfwd(theta_of)(phi0)
    return float(np.linalg.cond(np.asarray(a.T @ h_theta @ a)))


class TestG2PPPerTrade:
    @pytest.mark.slow  # two eager condition numbers (~19 s)
    def test_short_tenor_option_makes_the_fit_well_posed(self, g2_setup):
        s = g2_setup
        market = s["calibrated"].market
        co_node, co_keys = _g2_node(s["world"], s["basket"][:3], name="g2_co")
        fixed = s["fixed"].with_values({k: jnp.asarray(0.0) for k in co_keys})
        co = co_node.calibrate(QuoteSet(), market, fixed)
        # Rank deficiency is coordinate-free: the co-terminals alone stay
        # singular in any parametrisation.
        assert float(co.condition_number()) > 1e5
        # The bound was set in (σ, η); evaluate it there (see helper).
        assert _condition_in_sigma_eta(s["model"]) < 1e4

    def test_model_is_curve_free_and_converged(self, g2_setup):
        res = g2_setup["model"]
        assert res.value.initial_curve is None
        assert bool(res.converged)
        assert res.labels == g2_setup["target_keys"]
        assert res.provenance == CLOSED_FORM_PROVENANCE

    def test_basket_is_the_closed_form_and_the_pde_agrees(self, g2_setup):
        s = g2_setup
        market = s["calibrated"].market
        model = s["model"].value
        disc = market.curve(s["world"].discount_id)
        fwd = market.curve(s["world"].forward_id)
        for sw in s["basket"]:
            got = price(sw, model, market).price  # auto -> analytic
            exact = g2pp_swaption_price(sw, bind_curves(model, disc), fwd)
            assert float(got) == float(exact)
            pde = price(sw, model, market, method="pde", config=G2_CFG).price
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

    @pytest.mark.slow  # the g2_risks fixture: autodiff through the 2-D PDE (~50 s)
    def test_adjoint_matches_autodiff(self, g2_risks):
        adj, auto = g2_risks
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)
        assert float(adj.level_residuals["g2_trade"]) < 1e-8

    @pytest.mark.slow
    def test_matches_full_recalibration_fd(self, g2_setup, g2_risks):
        adj, _ = g2_risks
        # The Bermudan's PDE payoff and exercise are smoothed (C^1 in the
        # inputs), so ordinary steps work; larger steps also shrink the
        # recalibration's stopping noise (~1e-11 in price) divided by 2h.
        # Observed agreement ≤ 5e-6 relative. ∂V/∂a is tiny (~1.6e-5: the
        # fast factor's mean reversion is nearly offset by the re-fitted σ);
        # its FD is resolved to ~7e-10 absolute and gets a floor of 1e-8.
        cases = [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.SWPN.2Yx5Y.+0", 1e-5),
                 ("p", "G2.A", 1e-4, 1e-8), ("p", "G2.B", 1e-4),
                 ("p", g2_setup["target_keys"][1], 1e-5)]
        _check_fd(g2_setup, adj, _full_reval_cfg(g2_setup, G2_CFG), cases, rtol=5e-5)

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
        assert batched.value.volatility_x.shape == (2, 1)
        assert batched.value.initial_curve is None
        assert bool(jnp.all(batched.diagnostics.converged))
        one_fit = eqx.filter_jit(fit)
        for i, b in enumerate(baskets):
            one = one_fit(b).value
            # Closed-form basket, but a flatter level than Hull-White's (two
            # factors, condition number ~1e3): BFGS under vmap reduces in a
            # different order and stops ~7e-9 apart (relative) inside its
            # 1e-12 loss tolerance, so not to HW's 1e-10.
            for f in ("volatility_x", "volatility_y"):
                np.testing.assert_allclose(
                    getattr(batched.value, f)[i], getattr(one, f), rtol=5e-8
                )


@pytest.mark.slow
class TestG2PPOnEngine:
    """The same basket fitted on the Bermudan's own PDE grid (engine-consistent).

    The closed-form fit has portable parameters; the PDE fit reprices its
    basket exactly *on the grid*, so the Bermudan's discretisation error
    largely cancels against its basket. The two parameter sets differ by the
    grid's swaption error (~3e-3 relative here), and the PDE fit is recorded
    as ``discretised`` with its config.
    """

    @pytest.fixture(scope="class")
    def g2_pde(self, setup):
        return _g2_state(setup, method="pde", config=G2_CFG, name="g2_pde")

    def test_provenance_and_closeness_to_the_exact_fit(self, g2_pde, g2_setup):
        res = g2_pde["model"]
        assert bool(res.converged)
        assert res.provenance.engine_kind == "discretised"
        assert res.provenance.method == "pde" and res.provenance.config == G2_CFG
        exact = g2_setup["model"].value
        for f in ("volatility_x", "volatility_y"):
            np.testing.assert_allclose(
                getattr(res.value, f), getattr(exact, f), rtol=5e-2, err_msg=f
            )
        # The PDE fit minimises the basket RMSE *as its grid prices it*, so
        # on that grid it can do no worse than the closed-form fit (a max-miss
        # comparison would not be guaranteed by least squares).
        market = g2_pde["calibrated"].market
        world = g2_pde["world"]
        cube = market.surfaces[world.spec.cube_id]
        disc, fwd = market.curve(world.discount_id), market.curve(world.forward_id)

        def rmse_on_grid(model):
            misses = np.array([
                float(price(sw, model, market, method="pde", config=G2_CFG).price
                      / swaption_price(sw, disc, cube, fwd)) - 1.0
                for sw in g2_pde["basket"]
            ])
            return float(np.sqrt(np.mean(misses ** 2)))

        np.testing.assert_allclose(rmse_on_grid(res.value),
                                   float(res.diagnostics.rmse), rtol=1e-6)
        assert rmse_on_grid(res.value) <= rmse_on_grid(exact)

    def test_adjoint_matches_autodiff_on_the_discretised_path(self, g2_pde):
        s = g2_pde
        adj = quote_risk_adjoint(
            s["trade"], None, s["calibrated"], model_result=s["model"],
            method="pde", config=G2_CFG,
        )
        auto = quote_risk_autodiff(
            s["trade"], None, s["chain"], s["quotes"], s["fixed"],
            model_node=s["node"], method="pde", config=G2_CFG,
        )
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)


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

    def test_frozen_y_fit_is_engine_bound(self, ch_setup):
        res = ch_setup["model"]
        assert res.provenance == cheyette_provenance("frozen", CH_CFG)
        assert res.provenance.engine_kind == "modelling_bias"
        assert check_engine(res, "pde", CH_CFG) is res
        finer = PDEConfig(n_spot=201, n_time=120, spot_range=5.0)
        for method, cfg in (("pde", finer), ("mc", None)):
            with pytest.raises(CalibrationError, match="modelling bias"):
                check_engine(res, method, cfg)

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

    @pytest.mark.slow  # the ch_risks fixture (~38 s)
    def test_adjoint_matches_autodiff(self, ch_risks):
        adj, auto = ch_risks
        rec = reconcile_quote_risk(adj, auto, rtol=1e-6, atol=1e-10)
        assert rec.ok, (rec.mismatched, rec.max_abs_diff)

    @pytest.mark.slow
    def test_matches_full_recalibration_fd(self, ch_setup, ch_risks):
        adj, _ = ch_risks
        cases = [("q", "USD.OIS.5Y", 1e-5), ("q", "USD.SWPN.2Yx5Y.+0", 1e-5),
                 ("p", CH_KAPPA, 1e-4), ("p", ch_setup["target_keys"][1], 1e-5)]
        # C^1 PDE price (smoothed payoff): observed agreement ≤ 5e-6.
        _check_fd(ch_setup, adj, _full_reval_cfg(ch_setup, CH_CFG), cases, rtol=5e-5)
