"""Tests for the unified price() signature (ontology P2).

The hard gates from the migration report §10: single-curve parity with
the legacy call conventions, identical numbers through the bridged
MC/PDE registries, and a dispatch that covers what the old registries
covered.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import (
    METHODS,
    PricingResult,
    auto_method,
    available_methods,
    price,
    register_pricer,
    registered_pricers,
)
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.instruments.bonds import FixedRateBond
from valax.instruments.options import AmericanOption, DigitalOption, EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.analytic.digital import digital_option_price
from valax.pricing.analytic.swaptions import swap_price
from valax.pricing.lattice.binomial import BinomialConfig, binomial_price
from valax.pricing.mc import MCConfig
from valax.pricing.mc.dispatch import mc_price_dispatch
from valax.pricing.pde import PDEConfig
from valax.pricing.pde.dispatch import pde_price_dispatch


REF = jnp.array(739252, dtype=jnp.int32)
SPOT = jnp.array(100.0)


def _curve(level=0.05):
    times = jnp.arange(0, 11 * 365, 365, dtype=jnp.int32)
    return DiscountCurve(
        pillar_dates=REF + times,
        discount_factors=jnp.exp(-level * times / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _equity_market():
    return MarketState.single_curve(_curve(), spot=SPOT, vol=jnp.array(0.2))


def _bs_model():
    return BlackScholesModel(
        vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0)
    )


def _option(is_call=True):
    return EuropeanOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=is_call
    )


# ── Gate 1: single-curve parity with the legacy conventions ──────────

class TestSingleCurveParity:
    def test_black_scholes_matches_loose_scalar_call(self):
        model, option = _bs_model(), _option()
        unified = price(option, model, _equity_market())
        legacy = black_scholes_price(
            option, SPOT, model.vol, model.rate, model.dividend
        )
        assert unified.method == "analytic"
        assert abs(float(unified.price) - float(legacy)) < 1e-10

    def test_digital_matches_loose_scalar_call(self):
        model = _bs_model()
        digital = DigitalOption(
            strike=jnp.array(100.0),
            expiry=jnp.array(1.0),
            payout=jnp.array(1.0),
        )
        unified = price(digital, model, _equity_market())
        legacy = digital_option_price(
            digital, SPOT, model.vol, model.rate, model.dividend
        )
        assert abs(float(unified.price) - float(legacy)) < 1e-10

    def test_american_lattice_matches_legacy_call(self):
        model = _bs_model()
        american = AmericanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False
        )
        unified = price(american, model, _equity_market())
        legacy = binomial_price(
            american, SPOT, model.vol, model.rate, model.dividend,
            BinomialConfig(american=True),
        )
        assert unified.method == "lattice"
        assert abs(float(unified.price) - float(legacy)) < 1e-10

    def test_zero_selector_configuration(self):
        """A bare instrument needs no ids to find the default curve."""
        bond = FixedRateBond(
            payment_dates=REF + jnp.array([365, 730], jnp.int32),
            settlement_date=REF,
            coupon_rate=jnp.array(0.04),
            face_value=jnp.array(100.0),
            frequency=1,
        )
        market = MarketState.single_curve(_curve())
        unified = price(bond, None, market)
        legacy = fixed_rate_bond_price(bond, _curve())
        assert abs(float(unified.price) - float(legacy)) < 1e-10


# ── Gate 2: multi-curve equivalence ──────────────────────────────────

class TestMultiCurve:
    def _swap(self):
        return InterestRateSwap(
            start_date=REF,
            fixed_dates=REF + jnp.array([365, 730], jnp.int32),
            fixed_rate=jnp.array(0.04),
            notional=jnp.array(1_000_000.0),
        )

    def test_selectors_pick_the_named_curves(self):
        """price() on a graph equals the explicit dual-curve call."""
        ois, proj = _curve(0.04), _curve(0.045)
        market = MarketState(
            curves=CurveGraph(
                curves={"USD.SOFR.OIS": ois, "USD.SOFR.3M": proj}
            ),
        )
        swap = self._swap()
        # Attach selectors directly on the instrument (P3): the base
        # class now carries discount_id / forward_id static fields.
        selected = InterestRateSwap(
            start_date=swap.start_date,
            fixed_dates=swap.fixed_dates,
            fixed_rate=swap.fixed_rate,
            notional=swap.notional,
            discount_id="USD.SOFR.OIS",
            forward_id="USD.SOFR.3M",
        )
        unified = price(selected, None, market)
        legacy = swap_price(swap, ois, proj)
        assert abs(float(unified.price) - float(legacy)) < 1e-10

    def test_selector_subclass_reaches_the_legacy_mc_recipe(self):
        """A user subclass inherits its base's bridged MC recipe (MRO)."""
        class _TaggedOption(EuropeanOption):
            tag: str = eqx.field(static=True, default="x")

        market = _equity_market()
        model = _bs_model()
        option = _TaggedOption(strike=jnp.array(100.0), expiry=jnp.array(1.0))
        config = MCConfig(n_paths=2_000, n_steps=2)
        key = jax.random.PRNGKey(3)
        unified = price(option, model, market, method="mc", config=config, key=key)
        legacy = mc_price_dispatch(
            _option(), model, config, key, spot=SPOT,
        )
        assert float(unified.price) == float(legacy.price)

    def test_single_curve_degenerates_bit_for_bit(self):
        curve = _curve()
        swap = self._swap()
        market = MarketState.single_curve(curve)
        unified = price(swap, None, market)
        legacy = swap_price(swap, curve, None)
        assert float(unified.price) == float(legacy)

    def test_named_underlying_selects_the_right_spot(self):
        """A real underlying_id field routes an equity trade to its spot."""
        market = MarketState(
            curves=CurveGraph(curves={"_default_": _curve()}),
            spots={"AAPL": jnp.array(50.0), "MSFT": jnp.array(150.0)},
        )
        model = _bs_model()
        option = EuropeanOption(
            strike=jnp.array(50.0), expiry=jnp.array(1.0),
            underlying_id="AAPL",
        )
        unified = price(option, model, market)
        legacy = black_scholes_price(
            option, jnp.array(50.0), model.vol, model.rate, model.dividend
        )
        assert abs(float(unified.price) - float(legacy)) < 1e-10


# ── Gate 3: bridged MC/PDE registries give identical numbers ─────────

class TestLegacyBridge:
    def test_mc_matches_the_dispatcher_exactly(self):
        model, option, market = _bs_model(), _option(), _equity_market()
        config = MCConfig(n_paths=5_000, n_steps=4)
        key = jax.random.PRNGKey(7)
        unified = price(option, model, market, method="mc", config=config, key=key)
        legacy = mc_price_dispatch(option, model, config, key, spot=SPOT)
        assert unified.method == "mc"
        assert float(unified.price) == float(legacy.price)
        assert float(unified.stderr) == float(legacy.stderr)

    def test_pde_matches_the_dispatcher_exactly(self):
        model, option, market = _bs_model(), _option(), _equity_market()
        config = PDEConfig()
        unified = price(option, model, market, method="pde", config=config)
        legacy = pde_price_dispatch(option, model, config, spot=SPOT)
        assert unified.method == "pde"
        assert float(unified.price) == float(legacy.price)

    def test_mc_has_no_stderr_on_analytic_path(self):
        result = price(_option(), _bs_model(), _equity_market())
        assert result.stderr is None

    def test_registry_covers_the_legacy_pairs(self):
        """Every (instrument, model) in the old registries is reachable."""
        from valax.pricing.mc.dispatch import _REGISTRY as mc_registry
        from valax.pricing.pde.dispatch import _REGISTRY as pde_registry

        reachable = set(registered_pricers())
        for (i, m) in mc_registry:
            assert (i.__name__, m.__name__, "mc") in reachable
        for (i, m) in pde_registry:
            assert (i.__name__, m.__name__, "pde") in reachable


# ── Method selection (Q2) ────────────────────────────────────────────

class TestMethodSelection:
    def test_auto_prefers_the_policy_entry(self):
        assert auto_method(_option(), _bs_model()) == "analytic"
        american = AmericanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False
        )
        assert auto_method(american, _bs_model()) == "lattice"

    def test_available_methods_are_priority_ordered(self):
        methods = available_methods(_option(), _bs_model())
        assert methods == tuple(m for m in METHODS if m in methods)
        assert set(methods) >= {"analytic", "mc", "pde"}

    def test_explicit_method_overrides_the_policy(self):
        result = price(
            _option(), _bs_model(), _equity_market(),
            method="lattice", config=BinomialConfig(n_steps=300),
        )
        assert result.method == "lattice"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown pricing method"):
            price(_option(), _bs_model(), _equity_market(), method="quadrature")

    def test_unregistered_pair_lists_what_exists(self):
        class _Exotic(eqx.Module):
            notional: jnp.ndarray

        with pytest.raises(ValueError, match="No pricer registered"):
            price(_Exotic(notional=jnp.array(1.0)), None, _equity_market())

    def test_unregistered_method_for_a_known_pair(self):
        bond = FixedRateBond(
            payment_dates=REF + jnp.array([365], jnp.int32),
            settlement_date=REF,
            coupon_rate=jnp.array(0.04),
            face_value=jnp.array(100.0),
            frequency=1,
        )
        with pytest.raises(ValueError, match="No mc recipe registered"):
            price(bond, None, MarketState.single_curve(_curve()), method="mc",
                  key=jax.random.PRNGKey(0))

    def test_mc_without_key_raises(self):
        with pytest.raises(ValueError, match="needs a PRNG key"):
            price(
                _option(), _bs_model(), _equity_market(),
                method="mc", config=MCConfig(n_paths=100, n_steps=1),
            )


# ── Result container and JAX behaviour ───────────────────────────────

class TestPricingResult:
    def test_float_conversion(self):
        result = price(_option(), _bs_model(), _equity_market())
        assert float(result) == float(result.price)

    def test_jit_smoke(self):
        market, model, option = _equity_market(), _bs_model(), _option()

        @eqx.filter_jit
        def priced(m, mod):
            return price(option, mod, m).price

        assert jnp.isfinite(priced(market, model))

    def test_grad_flows_to_market_and_model(self):
        market, model, option = _equity_market(), _bs_model(), _option()

        def value(pair):
            m, mod = pair
            return price(option, mod, m).price

        d_market, d_model = eqx.filter_grad(value)((market, model))
        # Delta w.r.t. the market spot, vega w.r.t. the model vol.
        assert 0.0 < float(d_market.spots["_default_"]) < 1.0
        assert float(d_model.vol) > 0.0


# ── Extensibility ────────────────────────────────────────────────────

class TestRegistration:
    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_pricer(EuropeanOption, BlackScholesModel, "analytic")(
                lambda **_: PricingResult(price=jnp.array(0.0))
            )

    def test_custom_pricer_is_reachable(self):
        class _MyInstrument(eqx.Module):
            notional: jnp.ndarray

        @register_pricer(_MyInstrument, type(None), "analytic")
        def _my_pricer(*, instrument, model, market, resolved, config, key):
            return PricingResult(
                price=instrument.notional * resolved.discount_curve(REF + 365),
                method="analytic",
            )

        market = MarketState.single_curve(_curve())
        result = price(_MyInstrument(notional=jnp.array(100.0)), None, market)
        assert float(result.price) == pytest.approx(
            100.0 * float(_curve()(REF + 365))
        )
