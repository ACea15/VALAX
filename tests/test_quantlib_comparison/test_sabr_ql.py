"""
Cross-validation: VALAX vs QuantLib for SABR volatility model.

Tests that VALAX's Hagan SABR formula produces identical implied vols
to QuantLib's sabrVolatility() across a range of strikes and parameters.

Companion example: examples/comparisons/03_sabr_smile.py
"""

import pytest
import jax.numpy as jnp
import QuantLib as ql
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol


# ---------------------------------------------------------------------------
# Parameterized SABR configurations
# ---------------------------------------------------------------------------

SABR_CASES = [
    # (alpha, beta, rho, nu, forward, expiry, description)
    (0.04, 0.5, -0.25, 0.4, 0.03, 2.0, "rates_typical"),
    (0.04, 0.5, -0.25, 0.4, 0.03, 0.5, "rates_short_expiry"),
    (0.04, 0.5, -0.25, 0.4, 0.03, 5.0, "rates_long_expiry"),
    (0.20, 1.0, -0.3, 0.5, 100.0, 1.0, "equity_lognormal"),
    (0.04, 0.0, -0.25, 0.4, 0.03, 2.0, "normal_backbone"),
]

STRIKE_OFFSETS = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSABRImpliedVol:
    """VALAX and QuantLib SABR implied vols must match exactly (same formula)."""

    @pytest.mark.parametrize("alpha,beta,rho,nu,forward,expiry,desc", SABR_CASES,
                             ids=[c[-1] for c in SABR_CASES])
    def test_sabr_smile_matches(self, alpha, beta, rho, nu, forward, expiry, desc):
        """See: examples/comparisons/03_sabr_smile.py §4 (smile comparison)

        Validates across the entire strike range for each parameter set.
        """
        model = SABRModel(
            alpha=jnp.array(alpha), beta=jnp.array(beta),
            rho=jnp.array(rho), nu=jnp.array(nu),
        )

        for offset in STRIKE_OFFSETS:
            K = forward * offset
            if K <= 0:
                continue

            v_iv = float(sabr_implied_vol(
                model, jnp.array(forward), jnp.array(K), jnp.array(expiry)
            ))
            try:
                q_iv = ql.sabrVolatility(K, forward, expiry, alpha, beta, nu, rho)
            except RuntimeError:
                continue  # QuantLib may reject extreme strikes

            assert abs(v_iv - q_iv) < 1e-10, (
                f"{desc} K/F={offset}: VALAX={v_iv:.10f}, QL={q_iv:.10f}"
            )

    def test_atm_vol_matches(self):
        """ATM vol (K=F) is the most numerically sensitive case."""
        alpha, beta, rho, nu = 0.04, 0.5, -0.25, 0.4
        forward, expiry = 0.03, 2.0

        model = SABRModel(
            alpha=jnp.array(alpha), beta=jnp.array(beta),
            rho=jnp.array(rho), nu=jnp.array(nu),
        )

        v_iv = float(sabr_implied_vol(
            model, jnp.array(forward), jnp.array(forward), jnp.array(expiry)
        ))
        q_iv = ql.sabrVolatility(forward, forward, expiry, alpha, beta, nu, rho)

        assert abs(v_iv - q_iv) < 1e-12, f"ATM: VALAX={v_iv}, QL={q_iv}"
