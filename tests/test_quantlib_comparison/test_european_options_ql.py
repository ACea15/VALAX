"""Cross-validation: VALAX vs QuantLib for European option pricing.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md). Each test is
parametrized over a range of seeds; each seed produces an independent
sampled market via `valax.market.sample_scalar_market`. The same
market is fed to both the VALAX pricer and QuantLib's analytic engine
through the shared adapter in `_ql_adapters.py`, and the prices /
Greeks / implied vols are required to agree at the tolerance
documented per test.

A failure at any seed prints the originating market, which is enough
to reproduce locally.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.greeks.autodiff import greeks
from valax.instruments.options import EuropeanOption
from valax.market import SeedRegistry, default_config, sample_scalar_market
from valax.pricing.analytic.black_scholes import (
    black_scholes_implied_vol,
    black_scholes_price,
)

from tests.test_quantlib_comparison._ql_adapters import market_to_ql_bsm


# Seed range for parametric sweep. 20 seeds = 20 sampled markets per
# test method. Override via VALAX_QL_SWEEP_SEEDS if a triage run needs
# a smaller or larger range.
SEEDS = tuple(range(20260101, 20260121))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def call_setup(request):
    """One sampled market plus matching VALAX + QL call objects.

    Returns a dict with keys:
        market    — effective (snapped-expiry) sampled market.
        valax_opt — VALAX EuropeanOption for a call.
        ql_opt    — QL VanillaOption + analytic BS engine.
        ql_proc   — QL BlackScholesMertonProcess (for IV solver).
    """
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=3)
    raw_market = sample_scalar_market(registry, cfg)
    ql_opt, ql_proc, eff = market_to_ql_bsm(raw_market, is_call=True)
    valax_opt = EuropeanOption(
        strike=eff["strike"], expiry=eff["expiry"], is_call=True,
    )
    return {
        "market": eff, "valax_opt": valax_opt,
        "ql_opt": ql_opt, "ql_proc": ql_proc,
    }


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def put_setup(request):
    """Same as :func:`call_setup` but for puts."""
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=3)
    raw_market = sample_scalar_market(registry, cfg)
    ql_opt, ql_proc, eff = market_to_ql_bsm(raw_market, is_call=False)
    valax_opt = EuropeanOption(
        strike=eff["strike"], expiry=eff["expiry"], is_call=False,
    )
    return {
        "market": eff, "valax_opt": valax_opt,
        "ql_opt": ql_opt, "ql_proc": ql_proc,
    }


# ---------------------------------------------------------------------------
# Price tests
# ---------------------------------------------------------------------------

class TestEuropeanPrices:
    """VALAX and QuantLib must agree on BS call/put prices to ~1e-10."""

    def test_call_price_matches(self, call_setup):
        m = call_setup["market"]
        v = float(black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        ))
        q = call_setup["ql_opt"].NPV()
        assert v == pytest.approx(q, abs=1e-10), (
            f"market={ {k: float(x) for k, x in m.items()} } "
            f"VALAX={v:.12f}  QL={q:.12f}"
        )

    def test_put_price_matches(self, put_setup):
        m = put_setup["market"]
        v = float(black_scholes_price(
            put_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        ))
        q = put_setup["ql_opt"].NPV()
        assert v == pytest.approx(q, abs=1e-10), (
            f"market={ {k: float(x) for k, x in m.items()} } "
            f"VALAX={v:.12f}  QL={q:.12f}"
        )


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestEuropeanGreeks:
    """VALAX autodiff Greeks must match QuantLib's analytic Greeks."""

    def test_delta_matches(self, call_setup):
        m = call_setup["market"]
        g = greeks(
            black_scholes_price, call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["delta"]) == pytest.approx(
            call_setup["ql_opt"].delta(), abs=1e-10
        )

    def test_gamma_matches(self, call_setup):
        m = call_setup["market"]
        g = greeks(
            black_scholes_price, call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["gamma"]) == pytest.approx(
            call_setup["ql_opt"].gamma(), abs=1e-10
        )

    def test_vega_matches(self, call_setup):
        m = call_setup["market"]
        g = greeks(
            black_scholes_price, call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["vega"]) == pytest.approx(
            call_setup["ql_opt"].vega(), abs=1e-8
        )


# ---------------------------------------------------------------------------
# Implied vol tests
# ---------------------------------------------------------------------------

class TestImpliedVol:
    """Round-trip: price → implied vol → must recover input vol.

    The Newton solver in VALAX uses 20 iterations by default. For
    sampled markets with extreme moneyness or short expiries, that's
    not always enough; the round-trip tolerance is the dominant
    failure surface here, not the QL comparison.
    """

    def test_implied_vol_round_trip(self, call_setup):
        m = call_setup["market"]
        price = black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        recovered = float(black_scholes_implied_vol(
            call_setup["valax_opt"],
            m["spot"], m["rate"], m["dividend"], price,
            n_iterations=40,
        ))
        assert recovered == pytest.approx(float(m["vol"]), abs=1e-8)

    def test_implied_vol_matches_quantlib(self, call_setup):
        m = call_setup["market"]
        price = float(black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        ))
        v_iv = float(black_scholes_implied_vol(
            call_setup["valax_opt"],
            m["spot"], m["rate"], m["dividend"], jnp.array(price),
            n_iterations=40,
        ))
        q_iv = call_setup["ql_opt"].impliedVolatility(
            price, call_setup["ql_proc"], 1e-10, 1000, 1e-4, 4.0,
        )
        assert v_iv == pytest.approx(q_iv, abs=1e-6)
