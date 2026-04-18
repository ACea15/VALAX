"""Tests for the unified MC dispatcher.

Validates that:

1. Every registered recipe runs end-to-end without error.
2. Dispatcher results agree with the underlying low-level pricers.
3. Unregistered (instrument, model) combos raise a clear ValueError.
4. Custom recipes registered by users work end-to-end.
5. The dispatcher is JAX-transformable (``jax.grad`` flows through).
"""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import (
    AsianOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    VarianceSwap,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.mc import (
    MCConfig,
    MCResult,
    mc_price_dispatch,
    register,
    registered_recipes,
)
from valax.pricing.mc.engine import mc_price_with_stderr


# ─────────────────────────────────────────────────────────────────────
# Registration / introspection
# ─────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_expected_recipes_registered(self):
        """All equity + LMM recipes we ship should be in the registry."""
        names = set(registered_recipes())
        expected = {
            ("EuropeanOption", "BlackScholesModel"),
            ("EuropeanOption", "HestonModel"),
            ("AsianOption", "BlackScholesModel"),
            ("AsianOption", "HestonModel"),
            ("EquityBarrierOption", "BlackScholesModel"),
            ("EquityBarrierOption", "HestonModel"),
            ("LookbackOption", "BlackScholesModel"),
            ("LookbackOption", "HestonModel"),
            ("VarianceSwap", "BlackScholesModel"),
            ("VarianceSwap", "HestonModel"),
            ("Caplet", "LMMModel"),
            ("Cap", "LMMModel"),
            ("Swaption", "LMMModel"),
            ("BermudanSwaption", "LMMModel"),
        }
        missing = expected - names
        assert not missing, f"Missing recipes: {missing}"

    def test_unregistered_combo_raises(self):
        """Unknown (instrument, model) pair raises ValueError with listing."""
        class FakeModel:
            pass

        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        with pytest.raises(ValueError) as exc:
            mc_price_dispatch(
                option, FakeModel(),
                MCConfig(n_paths=100, n_steps=10),
                jax.random.PRNGKey(0),
            )
        msg = str(exc.value)
        assert "No MC recipe registered" in msg
        assert "FakeModel" in msg
        assert "Available recipes" in msg

    def test_register_duplicate_raises_without_overwrite(self):
        """Re-registering without overwrite=True should raise."""
        class _Instr:
            pass

        class _Model:
            pass

        @register(_Instr, _Model)
        def _r1(**_):
            return MCResult(
                price=jnp.array(0.0), stderr=jnp.array(0.0), n_paths=1,
            )

        with pytest.raises(ValueError, match="already registered"):
            @register(_Instr, _Model)
            def _r2(**_):  # pragma: no cover
                return MCResult(
                    price=jnp.array(0.0), stderr=jnp.array(0.0), n_paths=1,
                )

    def test_register_overwrite_replaces(self):
        """overwrite=True replaces the existing recipe."""
        class _Instr2:
            pass

        class _Model2:
            pass

        @register(_Instr2, _Model2)
        def _first(**_):
            return MCResult(
                price=jnp.array(1.0), stderr=jnp.array(0.0), n_paths=1,
            )

        @register(_Instr2, _Model2, overwrite=True)
        def _second(**_):
            return MCResult(
                price=jnp.array(2.0), stderr=jnp.array(0.0), n_paths=1,
            )

        result = mc_price_dispatch(
            _Instr2(), _Model2(),
            MCConfig(n_paths=1, n_steps=1),
            jax.random.PRNGKey(0),
        )
        assert float(result.price) == 2.0


# ─────────────────────────────────────────────────────────────────────
# Equity recipes: agreement with underlying pricers
# ─────────────────────────────────────────────────────────────────────


class TestEquityRecipes:
    @pytest.fixture(scope="class")
    def bsm_model(self):
        return BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.02),
        )

    @pytest.fixture(scope="class")
    def heston_model(self):
        return HestonModel(
            v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
            xi=jnp.array(0.3), rho=jnp.array(-0.7),
            rate=jnp.array(0.05), dividend=jnp.array(0.02),
        )

    def test_european_bsm_matches_analytic(self, bsm_model):
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        spot = jnp.array(100.0)
        analytical = black_scholes_price(
            option, spot, bsm_model.vol, bsm_model.rate, bsm_model.dividend,
        )

        result = mc_price_dispatch(
            option, bsm_model,
            config=MCConfig(n_paths=50_000, n_steps=100),
            key=jax.random.PRNGKey(42),
            spot=spot,
        )

        assert abs(float(result.price) - float(analytical)) < 3 * float(result.stderr)
        assert result.n_paths == 50_000

    def test_european_bsm_agrees_with_legacy_mc_price(self, bsm_model):
        """Dispatcher result must match the legacy mc_price_with_stderr for the
        same instrument, model, config, and key."""
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        spot = jnp.array(100.0)
        config = MCConfig(n_paths=10_000, n_steps=50)
        key = jax.random.PRNGKey(7)

        legacy_price, legacy_se = mc_price_with_stderr(
            option, spot, bsm_model, config, key,
        )
        result = mc_price_dispatch(option, bsm_model, config, key, spot=spot)

        # Should be bitwise-equal because the same path generator and payoff
        # are used.
        assert jnp.allclose(result.price, legacy_price)
        assert jnp.allclose(result.stderr, legacy_se)

    def test_european_heston_within_3se_of_bsm(self, heston_model):
        """Heston MC with small vol-of-vol should agree with BSM within 3 SE."""
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        spot = jnp.array(100.0)
        bsm_ref = black_scholes_price(
            option, spot,
            jnp.sqrt(heston_model.v0), heston_model.rate, heston_model.dividend,
        )
        result = mc_price_dispatch(
            option, heston_model,
            config=MCConfig(n_paths=20_000, n_steps=100),
            key=jax.random.PRNGKey(123),
            spot=spot,
        )
        assert abs(float(result.price) - float(bsm_ref)) < 3 * float(result.stderr)

    def test_asian_less_than_european(self, bsm_model):
        """Asian (arithmetic) call <= European call for same strike/expiry."""
        strike = jnp.array(100.0)
        expiry = jnp.array(1.0)
        spot = jnp.array(100.0)
        key = jax.random.PRNGKey(3)
        config = MCConfig(n_paths=20_000, n_steps=100)

        euro = EuropeanOption(strike=strike, expiry=expiry, is_call=True)
        asian = AsianOption(
            strike=strike, expiry=expiry, is_call=True, averaging="arithmetic",
        )

        r_euro = mc_price_dispatch(euro, bsm_model, config, key, spot=spot)
        r_asian = mc_price_dispatch(asian, bsm_model, config, key, spot=spot)

        # Asian with arithmetic averaging is strictly cheaper than European
        # for an ATM call under GBM with positive vol.
        assert float(r_asian.price) < float(r_euro.price)

    def test_knock_out_barrier_less_than_european(self, bsm_model):
        """Up-and-out barrier < European (vanilla bound)."""
        strike = jnp.array(100.0)
        expiry = jnp.array(1.0)
        spot = jnp.array(100.0)
        key = jax.random.PRNGKey(5)
        config = MCConfig(n_paths=20_000, n_steps=100)

        euro = EuropeanOption(strike=strike, expiry=expiry, is_call=True)
        barrier = EquityBarrierOption(
            strike=strike, expiry=expiry, is_call=True,
            barrier=jnp.array(130.0),
            is_up=True, is_knock_in=False,
            smoothing=0.5,
        )

        r_euro = mc_price_dispatch(euro, bsm_model, config, key, spot=spot)
        r_barrier = mc_price_dispatch(barrier, bsm_model, config, key, spot=spot)

        assert float(r_barrier.price) < float(r_euro.price)

    def test_lookback_dispatches(self, bsm_model):
        """Fixed-strike lookback runs through the dispatcher."""
        lb = LookbackOption(
            strike=jnp.array(100.0),
            expiry=jnp.array(1.0),
            is_call=True,
            is_fixed_strike=True,
        )
        result = mc_price_dispatch(
            lb, bsm_model,
            config=MCConfig(n_paths=5_000, n_steps=50),
            key=jax.random.PRNGKey(11),
            spot=jnp.array(100.0),
        )
        # Lookback max-payoff call is >= European call; at a minimum > 0.
        assert float(result.price) > 0.0

    def test_variance_swap_dispatches(self, bsm_model):
        """Variance swap runs through the dispatcher under GBM."""
        vs = VarianceSwap(
            expiry=jnp.array(1.0),
            strike_var=jnp.array(0.04),
            notional_var=jnp.array(1_000_000.0),
        )
        result = mc_price_dispatch(
            vs, bsm_model,
            config=MCConfig(n_paths=5_000, n_steps=252),
            key=jax.random.PRNGKey(17),
            spot=jnp.array(100.0),
        )
        # Under GBM with vol=20%, realized variance ≈ 0.04 ⇒ payoff ≈ 0,
        # within MC noise.
        assert abs(float(result.price)) < 1e5  # very loose upper bound


# ─────────────────────────────────────────────────────────────────────
# Autodiff through the dispatcher
# ─────────────────────────────────────────────────────────────────────


class TestAutodiffThroughDispatcher:
    def test_delta_via_grad(self):
        """jax.grad of the dispatched price w.r.t. spot gives sensible delta."""
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.02),
        )
        config = MCConfig(n_paths=10_000, n_steps=50)
        key = jax.random.PRNGKey(999)

        def price_fn(spot):
            return mc_price_dispatch(
                option, model, config, key, spot=spot,
            ).price

        delta = jax.grad(price_fn)(jnp.array(100.0))
        # ATM call delta is ~0.6 under BSM; MC delta should be close.
        assert 0.3 < float(delta) < 0.9

    def test_vega_via_grad(self):
        """jax.grad through the model parameter gives non-trivial vega."""
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        config = MCConfig(n_paths=10_000, n_steps=50)
        key = jax.random.PRNGKey(321)
        spot = jnp.array(100.0)

        def price_fn(vol):
            model = BlackScholesModel(
                vol=vol, rate=jnp.array(0.05), dividend=jnp.array(0.02),
            )
            return mc_price_dispatch(
                option, model, config, key, spot=spot,
            ).price

        vega = jax.grad(price_fn)(jnp.array(0.20))
        # ATM vega is ~37 per unit vol; MC noise makes this wide.
        assert float(vega) > 0.0


# ─────────────────────────────────────────────────────────────────────
# Custom recipe round-trip
# ─────────────────────────────────────────────────────────────────────


class TestCustomRecipe:
    def test_user_recipe_roundtrip(self):
        """User can register a custom recipe and price through the dispatcher."""
        import equinox as eqx
        from jaxtyping import Float
        from jax import Array

        class MyOption(eqx.Module):
            notional: Float[Array, ""]

        class MyModel(eqx.Module):
            rate: Float[Array, ""]

        @register(MyOption, MyModel)
        def _my_recipe(*, instrument, model, config, key, **kwargs):
            # Constant payoff of notional, discounted at the rate.
            discounted = jnp.exp(-model.rate) * instrument.notional
            return MCResult(
                price=discounted,
                stderr=jnp.array(0.0, dtype=discounted.dtype),
                n_paths=config.n_paths,
            )

        result = mc_price_dispatch(
            MyOption(notional=jnp.array(1_000.0)),
            MyModel(rate=jnp.array(0.05)),
            MCConfig(n_paths=1, n_steps=1),
            jax.random.PRNGKey(0),
        )
        assert jnp.isclose(result.price, jnp.exp(-0.05) * 1_000.0)
