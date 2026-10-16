"""The pricing policy: every reachable pair has an explicit ``auto`` method.

``method="auto"`` used to fall back to a global priority order (analytic →
lattice → pde → mc) for pairs without a policy entry, so adding a pricer could
silently change what ``auto`` meant for an existing pair. Every pair the
library registers now has an explicit entry (:mod:`valax.core.policy`); these
tests pin the whole map against the Phase 4 golden file — captured from the
fallback before it was removed, so no ``auto`` price changed — and make a
registration without a policy a CI failure instead of a guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.core import pricing as P
from valax.core.pricing import (
    PricingResult,
    auto_method,
    register_pricer,
    set_auto_method,
)

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "phase4_golden.json"


def _library_pairs() -> set[tuple[type, type]]:
    """Every registered (instrument, model) pair of the library itself.

    Restricted to ``valax`` classes: tests elsewhere register throwaway
    pairs in the same process.
    """
    P._ensure_backends()
    pairs = {(i, m) for (i, m, _) in P._REGISTRY}
    for method in ("pde", "mc"):
        pairs |= set(P._legacy_registry(method))

    def library(cls: type) -> bool:
        return cls is type(None) or cls.__module__.startswith("valax.")

    return {(i, m) for (i, m) in pairs if library(i) and library(m)}


def _key(instrument_cls: type, model_cls: type) -> str:
    return f"{instrument_cls.__qualname__}|{model_cls.__qualname__}"


def test_every_library_pair_has_an_explicit_policy():
    missing = sorted(
        _key(i, m) for (i, m) in _library_pairs()
        if not any((cls, m) in P._AUTO_POLICY for cls in P._instrument_bases(i))
    )
    assert not missing, f"registered pairs without an auto policy: {missing}"


#: Pairs registered after the golden capture, with their stated ``auto``.
ADDED_SINCE_GOLDEN = {
    "Caplet|ImpliedVolModel": "analytic",
    "Cap|ImpliedVolModel": "analytic",
    "Swaption|ImpliedVolModel": "analytic",
    "MultiLegSwap|NoneType": "analytic",
    "CrossCurrencySwap|NoneType": "analytic",
    "ZeroCouponInflationSwap|NoneType": "analytic",
    "YearOnYearInflationSwap|NoneType": "analytic",
    "InflationCapFloor|ImpliedVolModel": "analytic",
    "FXForward|NoneType": "analytic",
    "FXVanillaOption|ImpliedVolModel": "analytic",
}


def test_auto_map_is_the_frozen_one():
    """The map equals the golden capture taken before the change, plus the
    pairs registered since (each named, with its ``auto``)."""
    golden = json.loads(GOLDEN.read_text())["auto_methods"]
    expected = {k: v["auto"] for k, v in golden.items()} | ADDED_SINCE_GOLDEN
    current = {_key(i, m): auto_method(i, m) for (i, m) in _library_pairs()}
    assert set(current) == set(expected), (
        sorted(set(current) ^ set(expected))
    )
    changed = {k: (expected[k], v) for k, v in current.items()
               if expected[k] != v}
    assert not changed, changed
    assert len(golden) == 66 and len(current) == 66 + len(ADDED_SINCE_GOLDEN)


def test_policy_names_a_registered_method():
    for (i, m) in _library_pairs():
        assert auto_method(i, m) in P.available_methods(i, m), _key(i, m)


class _Exotic(eqx.Module):
    notional: jnp.ndarray


@pytest.fixture
def unpoliced_pair():
    """A pair with a pricer and no policy, removed again afterwards."""
    key = (_Exotic, type(None), "analytic")

    @register_pricer(_Exotic, type(None), "analytic")
    def _adapter(*, instrument, model, market, resolved, config, key):
        return PricingResult(price=instrument.notional, method="analytic")

    yield
    P._REGISTRY.pop(key, None)
    P._AUTO_POLICY.pop((_Exotic, type(None)), None)


def test_auto_without_policy_raises(unpoliced_pair):
    """Anti-vacuity: no fallback order rescues a pair without a policy."""
    with pytest.raises(ValueError, match="no auto policy"):
        auto_method(_Exotic, type(None))
    # Declaring it fixes it.
    set_auto_method(_Exotic, type(None), "analytic")
    assert auto_method(_Exotic, type(None)) == "analytic"


def test_policy_naming_an_unregistered_method_raises(unpoliced_pair):
    set_auto_method(_Exotic, type(None), "pde")
    with pytest.raises(ValueError, match="names 'pde'"):
        auto_method(_Exotic, type(None))


# ── Default configs ──────────────────────────────────────────────────


def test_every_numerical_method_of_every_pair_has_a_default():
    missing = sorted(
        f"{_key(i, m)}|{method}"
        for (i, m) in _library_pairs()
        for method in P.available_methods(i, m)
        if method not in P.CONFIGLESS_METHODS
        and P.default_config(i, m, method) is None
    )
    assert not missing, f"methods without a default config: {missing}"


def _equity():
    from valax.core.market_state import MarketState
    from valax.instruments.options import EuropeanOption
    from valax.models.black_scholes import BlackScholesModel

    market = MarketState.flat(rate=0.03, spot=100.0, vol=0.2)
    model = BlackScholesModel(vol=None, rate=None, dividend=None)
    option = EuropeanOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0),
                            is_call=True)
    return option, model, market


@pytest.mark.parametrize("method", ["analytic", "lattice", "pde", "mc"])
def test_every_method_prices_without_a_config(method):
    """Before D15, pde and mc raised AttributeError on config=None."""
    import jax

    from valax.core.pricing import effective_config, price

    option, model, market = _equity()
    result = price(option, model, market, method=method,
                   key=jax.random.PRNGKey(0))
    assert jnp.isfinite(result.price)
    assert result.method == method
    # The result records the config it used.
    assert result.config == effective_config(option, model, method)
    if method == "analytic":
        assert result.config is None
    else:
        assert result.config is not None


def test_explicit_config_wins_and_is_recorded():
    from valax.core.pricing import price
    from valax.pricing.pde.config import PDEConfig

    option, model, market = _equity()
    cfg = PDEConfig(n_spot=101, n_time=50)
    result = price(option, model, market, method="pde", config=cfg)
    assert result.config == cfg
    default = price(option, model, market, method="pde")
    assert default.config == PDEConfig()
    assert float(result.price) != float(default.price)  # the grid was used


def test_auto_result_records_method_and_config():
    from valax.core.pricing import price

    option, model, market = _equity()
    result = price(option, model, market)
    assert result.method == "analytic" and result.config is None


@pytest.fixture
def exotic_lattice_pair():
    """A pair with a lattice pricer and policy but no default config."""
    key = (_Exotic, type(None), "lattice")

    @register_pricer(_Exotic, type(None), "lattice")
    def _adapter(*, instrument, model, market, resolved, config, key):
        return PricingResult(price=instrument.notional, method="lattice")

    set_auto_method(_Exotic, type(None), "lattice")
    yield
    P._REGISTRY.pop(key, None)
    P._AUTO_POLICY.pop((_Exotic, type(None)), None)
    P._DEFAULT_CONFIGS.pop(key, None)


def test_numerical_method_without_default_raises(exotic_lattice_pair):
    """Anti-vacuity: no config and no default is an error, not a None."""
    from valax.core.market_state import MarketState
    from valax.core.pricing import price, set_default_config
    from valax.pricing.lattice.binomial import BinomialConfig

    market = MarketState.flat(rate=0.03)
    with pytest.raises(ValueError, match="No default lattice config"):
        price(_Exotic(notional=jnp.asarray(1.0)), None, market)
    set_default_config(_Exotic, type(None), "lattice", BinomialConfig())
    result = price(_Exotic(notional=jnp.asarray(1.0)), None, market)
    assert result.config == BinomialConfig()


def test_set_default_config_validates():
    from valax.core.pricing import set_default_config

    with pytest.raises(ValueError, match="Unknown pricing method"):
        set_default_config(_Exotic, type(None), "tree", object())
    with pytest.raises(ValueError, match="cannot be None"):
        set_default_config(_Exotic, type(None), "pde", None)


# ── Provenance records the grid actually used ────────────────────────


def _swaption():
    from valax.dates.daycounts import ymd_to_ordinal
    from valax.instruments.rates import Swaption

    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(2027, 1, 1))),
        fixed_dates=jnp.array([int(ymd_to_ordinal(2027 + k, 1, 1)) for k in (1, 2, 3)],
                              dtype=jnp.int32),
        strike=jnp.asarray(0.03), notional=jnp.asarray(1.0), day_count="act_365",
    )


def test_basket_provenance_records_the_default_grid():
    """A PDE basket fit with config=None is bound to the grid it used."""
    from valax.calibration.nodes import _basket_provenance
    from valax.models.hull_white import HullWhiteModel
    from valax.pricing.pde.config import PDEConfig

    hw = HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                        volatility=jnp.asarray(0.01))
    prov = _basket_provenance(hw, (_swaption(),), "pde", None)
    assert prov.engine_kind == "discretised"
    assert prov.config == PDEConfig()
    # The closed form records no grid.
    assert _basket_provenance(hw, (_swaption(),), "auto", None).config is None


def test_cheyette_fit_and_price_share_the_default_grid():
    """Engine-bound frozen-y: the calibrator's default grid is price()'s."""
    from valax.calibration.cheyette import (
        _DEFAULT_CONFIG_FROZEN_1F,
        cheyette_provenance,
    )
    from valax.calibration.nodes import _basket_provenance
    from valax.calibration.result import check_engine
    from valax.core.pricing import effective_config
    from valax.models.cheyette import CheyetteModel

    ch = CheyetteModel(mean_reversion=jnp.asarray(0.03),
                       volatility=jnp.asarray(0.008), skew=jnp.asarray(0.2))
    grid = effective_config(_swaption(), ch, "pde")
    # What calibrate_cheyette records with its default grid.
    fit = cheyette_provenance("frozen", _DEFAULT_CONFIG_FROZEN_1F)
    assert fit.config == grid
    assert _basket_provenance(ch, (_swaption(),), "pde", None) == fit

    class _Result:  # check_engine reads .provenance, .name and .children
        provenance = fit
        name = "ch"
        children = ()

    assert check_engine(_Result(), "pde", grid) is not None
