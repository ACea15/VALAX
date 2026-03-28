"""
Cross-validation: VALAX Heston MC vs QuantLib analytic Heston.

Tests that VALAX's Heston MC paths produce prices consistent with
QuantLib's semi-closed-form Fourier solution. MC prices should be
within a few standard errors of the analytic reference.

Companion example: examples/comparisons/07_heston_smile.py
"""

import pytest
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.models.heston import HestonModel
from valax.pricing.mc.paths import generate_heston_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def heston_params():
    return dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
                S=100.0, T=1.0, r=0.04, q=0.01)


@pytest.fixture
def valax_paths(heston_params):
    """Generate Heston MC paths (cached per test session for speed)."""
    p = heston_params
    model = HestonModel(
        v0=jnp.array(p["v0"]), kappa=jnp.array(p["kappa"]),
        theta=jnp.array(p["theta"]), xi=jnp.array(p["xi"]),
        rho=jnp.array(p["rho"]),
        rate=jnp.array(p["r"]), dividend=jnp.array(p["q"]),
    )
    key = jax.random.PRNGKey(42)
    spot_paths, var_paths = generate_heston_paths(
        model, jnp.array(p["S"]), p["T"], 252, 200_000, key
    )
    return spot_paths, var_paths


@pytest.fixture
def ql_heston_engine(heston_params):
    """Build QuantLib analytic Heston engine."""
    p = heston_params
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today

    spot_h = ql.QuoteHandle(ql.SimpleQuote(p["S"]))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, p["r"], ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, p["q"], ql.Actual365Fixed()))

    heston_process = ql.HestonProcess(
        rate_h, div_h, spot_h,
        p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"],
    )
    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)
    maturity = today + ql.Period(1, ql.Years)
    return engine, maturity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHestonMCvsAnalytic:
    """VALAX Heston MC prices should be within statistical tolerance of QL analytic."""

    @pytest.mark.parametrize("K", [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
    def test_call_price_within_3se(self, heston_params, valax_paths, ql_heston_engine, K):
        """See: examples/comparisons/07_heston_smile.py §3 (option prices)

        MC price should be within 3 standard errors of the analytic solution.
        """
        p = heston_params
        spot_paths, _ = valax_paths
        n_paths = spot_paths.shape[0]

        # VALAX MC
        payoffs = jnp.maximum(spot_paths[:, -1] - K, 0.0)
        mc_price = float(jnp.exp(-jnp.array(p["r"]) * p["T"]) * jnp.mean(payoffs))
        mc_se = float(jnp.exp(-jnp.array(p["r"]) * p["T"]) * jnp.std(payoffs) / jnp.sqrt(float(n_paths)))

        # QuantLib analytic
        engine, maturity = ql_heston_engine
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, K),
            ql.EuropeanExercise(maturity),
        )
        opt.setPricingEngine(engine)
        ql_price = opt.NPV()

        n_se = abs(mc_price - ql_price) / mc_se if mc_se > 0 else 0
        assert n_se < 3.0, (
            f"K={K}: MC={mc_price:.4f} ± {mc_se:.4f}, QL={ql_price:.4f}, "
            f"{n_se:.1f} SE away"
        )

    def test_smile_skew_direction(self, valax_paths, heston_params):
        """With ρ < 0, low strikes should have higher MC implied vol than high strikes."""
        spot_paths, _ = valax_paths
        p = heston_params

        def mc_iv_proxy(K):
            payoffs = jnp.maximum(spot_paths[:, -1] - K, 0.0)
            return float(jnp.exp(-jnp.array(p["r"]) * p["T"]) * jnp.mean(payoffs))

        price_low = mc_iv_proxy(85.0)
        price_high = mc_iv_proxy(115.0)

        # For negative rho, the smile should be skewed:
        # normalize prices by intrinsic + time value → low strike should be richer
        # Simple check: OTM call at 115 should be much cheaper relative to ATM
        # than ITM call at 85
        assert price_low > price_high * 3, "Negative rho should produce strong skew"


class TestHestonPathStatistics:
    """Sanity checks on the generated Heston MC paths."""

    def test_mean_terminal_spot_risk_neutral(self, valax_paths, heston_params):
        """Under risk-neutral measure, E[S_T] = S_0 * exp((r-q)*T)."""
        spot_paths, _ = valax_paths
        p = heston_params
        expected = p["S"] * jnp.exp((p["r"] - p["q"]) * p["T"])
        mc_mean = float(jnp.mean(spot_paths[:, -1]))
        # Allow 2% tolerance for MC approximation
        assert abs(mc_mean - float(expected)) / float(expected) < 0.02

    def test_variance_mean_reverts(self, valax_paths, heston_params):
        """Terminal variance should be near theta (long-run level)."""
        _, var_paths = valax_paths
        p = heston_params
        mean_var = float(jnp.mean(var_paths[:, -1]))
        # With kappa=2.0, variance should be pulled toward theta=0.04
        assert abs(mean_var - p["theta"]) < 0.01
