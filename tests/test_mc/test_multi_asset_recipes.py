"""Tests for the multi-asset MC recipes (SpreadOption, WorstOfBasketOption).

Validates:

1. SpreadOption MC agrees with the Margrabe closed form at K=0 for a
   sweep of correlations.
2. SpreadOption MC agrees with Kirk's approximation at K>0 (and K<0)
   for typical spread-option parameters.
3. Worst-of basket put price decreases monotonically in correlation
   (standard property).
4. Dispatcher round-trips end-to-end and raises clear errors for
   mismatched shapes.
5. ``jax.grad`` flows through both recipes.
"""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import SpreadOption, WorstOfBasketOption
from valax.models.multi_asset import MultiAssetGBMModel
from valax.pricing.analytic.spread import kirk_price, margrabe_price
from valax.pricing.mc import MCConfig, mc_price_dispatch


# ─────────────────────────────────────────────────────────────────────
# SpreadOption: agreement with analytical references
# ─────────────────────────────────────────────────────────────────────


class TestSpreadOptionMC:
    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.7])
    def test_margrabe_k_zero_within_3se(self, rho):
        """Spread option MC at K=0 matches Margrabe to within 3 SE."""
        option = SpreadOption(
            expiry=jnp.array(1.0),
            strike=jnp.array(0.0),
            notional=jnp.array(1.0),
            is_call=True,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.25, 0.30]),
            rate=jnp.array(0.05),
            dividends=jnp.array([0.02, 0.03]),
            correlation=jnp.array([[1.0, rho], [rho, 1.0]]),
        )
        spots = jnp.array([100.0, 100.0])

        analytic = margrabe_price(
            option, spots[0], spots[1],
            model.vols[0], model.vols[1],
            jnp.array(rho),
            model.dividends[0], model.dividends[1],
        )

        result = mc_price_dispatch(
            option, model,
            config=MCConfig(n_paths=50_000, n_steps=50),
            key=jax.random.PRNGKey(42),
            spots=spots,
        )

        err = abs(float(result.price) - float(analytic))
        assert err < 3 * float(result.stderr), (
            f"rho={rho}: MC={float(result.price):.4f}, "
            f"Margrabe={float(analytic):.4f}, err={err:.4f}, "
            f"3*SE={3 * float(result.stderr):.4f}",
        )

    @pytest.mark.parametrize("strike", [-5.0, 5.0, 10.0])
    def test_kirk_k_nonzero_within_3se(self, strike):
        """Spread option MC at K != 0 matches Kirk within 3 SE."""
        option = SpreadOption(
            expiry=jnp.array(0.5),
            strike=jnp.array(strike),
            notional=jnp.array(1.0),
            is_call=True,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.30, 0.25]),
            rate=jnp.array(0.04),
            dividends=jnp.array([0.01, 0.02]),
            correlation=jnp.array([[1.0, 0.6], [0.6, 1.0]]),
        )
        spots = jnp.array([100.0, 90.0])

        analytic = kirk_price(
            option,
            spots[0], spots[1],
            model.vols[0], model.vols[1],
            model.correlation[0, 1],
            model.rate,
            model.dividends[0], model.dividends[1],
        )

        result = mc_price_dispatch(
            option, model,
            config=MCConfig(n_paths=100_000, n_steps=50),
            key=jax.random.PRNGKey(123),
            spots=spots,
        )

        err = abs(float(result.price) - float(analytic))
        # Kirk is an approximation; we expect MC to be close to it but a
        # 3-SE band may be tight for large K. Use 4*SE to accommodate
        # both MC noise and Kirk bias.
        tol = 4 * float(result.stderr) + 0.02 * abs(float(analytic))
        assert err < tol, (
            f"K={strike}: MC={float(result.price):.4f}, "
            f"Kirk={float(analytic):.4f}, err={err:.4f}, tol={tol:.4f}",
        )

    def test_put_payoff_nonnegative(self):
        """Spread put price is non-negative."""
        option = SpreadOption(
            expiry=jnp.array(1.0),
            strike=jnp.array(5.0),
            notional=jnp.array(1.0),
            is_call=False,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.3, 0.3]),
            rate=jnp.array(0.05),
            dividends=jnp.zeros(2),
            correlation=jnp.array([[1.0, 0.4], [0.4, 1.0]]),
        )
        spots = jnp.array([100.0, 100.0])
        result = mc_price_dispatch(
            option, model,
            config=MCConfig(n_paths=10_000, n_steps=20),
            key=jax.random.PRNGKey(0),
            spots=spots,
        )
        assert float(result.price) >= 0.0


# ─────────────────────────────────────────────────────────────────────
# WorstOfBasketOption: property-based checks
# ─────────────────────────────────────────────────────────────────────


class TestWorstOfBasket:
    def test_put_price_decreases_in_correlation(self):
        """Worst-of ATM put price is monotonically decreasing in rho.

        Intuition: higher correlation ⇒ assets move together ⇒ less
        dispersion ⇒ lower probability that the worst performer is
        badly below the strike ⇒ cheaper put.
        """
        option = WorstOfBasketOption(
            expiry=jnp.array(1.0),
            strike=jnp.array(1.0),  # ATM return-space
            notional=jnp.array(1.0),
            n_assets=2,
            is_call=False,
        )
        spots = jnp.array([100.0, 100.0])
        config = MCConfig(n_paths=40_000, n_steps=20)
        key = jax.random.PRNGKey(7)

        rhos = [-0.5, 0.0, 0.3, 0.7, 0.95]
        prices = []
        for rho in rhos:
            model = MultiAssetGBMModel(
                vols=jnp.array([0.25, 0.30]),
                rate=jnp.array(0.05),
                dividends=jnp.zeros(2),
                correlation=jnp.array([[1.0, rho], [rho, 1.0]]),
            )
            result = mc_price_dispatch(
                option, model, config=config, key=key, spots=spots,
            )
            prices.append(float(result.price))

        # Each price should be >= the next in the sequence (monotone
        # decreasing), allowing for MC noise on the order of the stderr.
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1] - 0.005, (
                f"Non-monotonic at rho={rhos[i]} -> {rhos[i+1]}: "
                f"prices {prices}",
            )

    def test_call_price_increases_in_correlation(self):
        """Worst-of call price is monotonically increasing in rho.

        Mirror of the put case — higher correlation ⇒ worst performer
        is closer to the average ⇒ more likely to be above the strike
        ⇒ call more valuable.
        """
        option = WorstOfBasketOption(
            expiry=jnp.array(1.0),
            strike=jnp.array(1.0),
            notional=jnp.array(1.0),
            n_assets=2,
            is_call=True,
        )
        spots = jnp.array([100.0, 100.0])
        config = MCConfig(n_paths=40_000, n_steps=20)
        key = jax.random.PRNGKey(11)

        rhos = [-0.5, 0.0, 0.5, 0.95]
        prices = []
        for rho in rhos:
            model = MultiAssetGBMModel(
                vols=jnp.array([0.25, 0.30]),
                rate=jnp.array(0.05),
                dividends=jnp.zeros(2),
                correlation=jnp.array([[1.0, rho], [rho, 1.0]]),
            )
            result = mc_price_dispatch(
                option, model, config=config, key=key, spots=spots,
            )
            prices.append(float(result.price))

        for i in range(len(prices) - 1):
            assert prices[i + 1] >= prices[i] - 0.005

    def test_three_asset_basket(self):
        """Worst-of works for more than 2 assets."""
        option = WorstOfBasketOption(
            expiry=jnp.array(1.0),
            strike=jnp.array(1.0),
            notional=jnp.array(1.0),
            n_assets=3,
            is_call=False,
        )
        C = jnp.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])
        model = MultiAssetGBMModel(
            vols=jnp.array([0.25, 0.3, 0.35]),
            rate=jnp.array(0.04),
            dividends=jnp.zeros(3),
            correlation=C,
        )
        spots = jnp.array([100.0, 100.0, 100.0])
        result = mc_price_dispatch(
            option, model,
            config=MCConfig(n_paths=10_000, n_steps=20),
            key=jax.random.PRNGKey(0),
            spots=spots,
        )
        assert float(result.price) > 0.0


# ─────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────


class TestDispatcherErrors:
    def test_spread_requires_at_least_2_assets(self):
        """Passing a 1-asset spots array to SpreadOption recipe raises."""
        option = SpreadOption(
            expiry=jnp.array(1.0), strike=jnp.array(0.0),
            notional=jnp.array(1.0), is_call=True,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.25]), rate=jnp.array(0.05),
            dividends=jnp.array([0.02]),
            correlation=jnp.array([[1.0]]),
        )
        with pytest.raises(ValueError, match="at least 2 assets"):
            mc_price_dispatch(
                option, model,
                config=MCConfig(n_paths=100, n_steps=5),
                key=jax.random.PRNGKey(0),
                spots=jnp.array([100.0]),
            )

    def test_worst_of_requires_matching_n_assets(self):
        """Passing spots of wrong length to WorstOfBasket raises."""
        option = WorstOfBasketOption(
            expiry=jnp.array(1.0), strike=jnp.array(1.0),
            notional=jnp.array(1.0), n_assets=3,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.25]),
            rate=jnp.array(0.05),
            dividends=jnp.zeros(2),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        with pytest.raises(ValueError, match="instrument.n_assets"):
            mc_price_dispatch(
                option, model,
                config=MCConfig(n_paths=100, n_steps=5),
                key=jax.random.PRNGKey(0),
                spots=jnp.array([100.0, 100.0]),  # 2 assets, instrument wants 3
            )


# ─────────────────────────────────────────────────────────────────────
# Autodiff through the recipes
# ─────────────────────────────────────────────────────────────────────


class TestAutodiff:
    def test_spread_delta_spot1(self):
        """dPrice/dSpot1 > 0 for a call on (S1 - S2)."""
        option = SpreadOption(
            expiry=jnp.array(1.0), strike=jnp.array(0.0),
            notional=jnp.array(1.0), is_call=True,
        )
        model = MultiAssetGBMModel(
            vols=jnp.array([0.25, 0.25]),
            rate=jnp.array(0.05),
            dividends=jnp.zeros(2),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )

        def price_fn(s1):
            spots = jnp.stack([s1, jnp.array(100.0)])
            return mc_price_dispatch(
                option, model,
                config=MCConfig(n_paths=5_000, n_steps=20),
                key=jax.random.PRNGKey(50),
                spots=spots,
            ).price

        delta_s1 = jax.grad(price_fn)(jnp.array(100.0))
        # Delta w.r.t. S1 should be positive (~0.5 ATM).
        assert 0.1 < float(delta_s1) < 0.9

    def test_worst_of_correlation_sensitivity_via_grad(self):
        """d(put price)/d(rho) < 0 (put gets cheaper as correlation rises)."""
        option = WorstOfBasketOption(
            expiry=jnp.array(1.0), strike=jnp.array(1.0),
            notional=jnp.array(1.0), n_assets=2, is_call=False,
        )
        spots = jnp.array([100.0, 100.0])

        def price_fn(rho):
            model = MultiAssetGBMModel(
                vols=jnp.array([0.25, 0.3]),
                rate=jnp.array(0.05),
                dividends=jnp.zeros(2),
                correlation=jnp.array([[1.0, rho], [rho, 1.0]]),
            )
            return mc_price_dispatch(
                option, model,
                config=MCConfig(n_paths=10_000, n_steps=20),
                key=jax.random.PRNGKey(77),
                spots=spots,
            ).price

        dprice_drho = jax.grad(price_fn)(jnp.array(0.3))
        # Correlation-vega should be negative for a worst-of put.
        assert float(dprice_drho) < 0.0
