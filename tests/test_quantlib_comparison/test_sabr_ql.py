"""Cross-validation: VALAX vs QuantLib for the Hagan-SABR implied vol formula.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Two test classes:
  * :class:`TestSABRImpliedVolFixedCases` keeps the original
    hand-picked parameter scenarios (rates, equity, normal backbone)
    that exercise specific corners of Hagan's expansion.
  * :class:`TestSABRImpliedVolSweep` adds a synthetic-data sweep:
    each seed samples a SABR ground truth via
    :func:`valax.market.sample_sabr_params`, then compares VALAX and
    QL across a strike grid around the forward.

Both classes assert at ``abs < 1e-10`` because both libraries
implement Hagan's 2002 formula — any disagreement is a transcription
bug, not a method choice.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.market import (
    SeedRegistry,
    default_config,
    sample_sabr_params,
)
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol


SEEDS = tuple(range(20260101, 20260121))


# ---------------------------------------------------------------------------
# Fixed parameter scenarios (the original SABR_CASES) — kept because
# they exercise specific corners (equity lognormal, normal backbone)
# that uniform sampling under cfg.vol_range might miss.
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


class TestSABRImpliedVolFixedCases:

    @pytest.mark.parametrize(
        "alpha,beta,rho,nu,forward,expiry,desc",
        SABR_CASES,
        ids=[c[-1] for c in SABR_CASES],
    )
    def test_sabr_smile_matches(
        self, alpha, beta, rho, nu, forward, expiry, desc,
    ):
        model = SABRModel(
            alpha=jnp.array(alpha), beta=jnp.array(beta),
            rho=jnp.array(rho), nu=jnp.array(nu),
        )
        for offset in STRIKE_OFFSETS:
            K = forward * offset
            if K <= 0:
                continue
            v_iv = float(sabr_implied_vol(
                model, jnp.array(forward), jnp.array(K), jnp.array(expiry),
            ))
            try:
                q_iv = ql.sabrVolatility(K, forward, expiry, alpha, beta, nu, rho)
            except RuntimeError:
                continue   # QL rejects extreme strikes
            assert v_iv == pytest.approx(q_iv, abs=1e-10), (
                f"{desc} K/F={offset}  VALAX={v_iv:.12f}  QL={q_iv:.12f}"
            )

    def test_atm_matches(self):
        alpha, beta, rho, nu = 0.04, 0.5, -0.25, 0.4
        forward, expiry = 0.03, 2.0
        model = SABRModel(
            alpha=jnp.array(alpha), beta=jnp.array(beta),
            rho=jnp.array(rho), nu=jnp.array(nu),
        )
        v_iv = float(sabr_implied_vol(
            model, jnp.array(forward), jnp.array(forward), jnp.array(expiry),
        ))
        q_iv = ql.sabrVolatility(forward, forward, expiry, alpha, beta, nu, rho)
        assert v_iv == pytest.approx(q_iv, abs=1e-12)


# ---------------------------------------------------------------------------
# Synthetic sweep — random SABR truths via the synthetic generator.
# ---------------------------------------------------------------------------


class TestSABRImpliedVolSweep:

    EXPIRY = 1.0
    FORWARD = 100.0
    # Strike grid is moneyness-based to keep the test independent of
    # the absolute forward level chosen by the synthetic generator.
    MONEYNESS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)

    @pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
    def sabr_truth(self, request):
        registry = SeedRegistry(
            master_seed=request.param, library_version=valax.__version__,
        )
        cfg = default_config(n_assets=1)
        return sample_sabr_params(registry, cfg, fixed_beta=0.5)

    def test_sweep_matches_quantlib(self, sabr_truth):
        alpha = float(sabr_truth.alpha)
        beta = float(sabr_truth.beta)
        rho = float(sabr_truth.rho)
        nu = float(sabr_truth.nu)

        for moneyness in self.MONEYNESS:
            K = self.FORWARD * moneyness
            v_iv = float(sabr_implied_vol(
                sabr_truth, jnp.array(self.FORWARD), jnp.array(K),
                jnp.array(self.EXPIRY),
            ))
            try:
                q_iv = ql.sabrVolatility(
                    K, self.FORWARD, self.EXPIRY, alpha, beta, nu, rho,
                )
            except RuntimeError:
                continue
            assert v_iv == pytest.approx(q_iv, abs=1e-10), (
                f"alpha={alpha:.4f} beta={beta} rho={rho:+.4f} nu={nu:.4f} "
                f"K/F={moneyness}  VALAX={v_iv:.12f}  QL={q_iv:.12f}"
            )
