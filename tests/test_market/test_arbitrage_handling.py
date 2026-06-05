"""Detect-or-regularise tests for arbitrageable synthetic data.

Each test feeds a deliberately-broken object (from
``valax.market.synthetic.arbitrage``) into a real library consumer
and asserts one of three outcomes:

1. **Detect**: the consumer raises one of the reserved exceptions in
   :mod:`valax.core.diagnostics`.  Preferred.
2. **Graceful failure**: the consumer returns a sentinel (NaN, inf)
   that the caller can check.  Acceptable.
3. **Silent mishandle**: the consumer accepts the bad data and
   returns a finite, wrong-looking number.  Marked
   ``@pytest.mark.xfail(strict=True)`` with a roadmap reason — green
   once detection lands in the library.

The xfail set is the *public, machine-readable backlog* of missing
safety checks.  Removing an xfail is a real engineering deliverable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from valax.core.diagnostics import NonPSDCorrelationError
from valax.market.synthetic import (
    inject_basket_variance_violation,
    inject_calendar_arb,
    inject_non_psd_correlation,
    inject_pcp_violation,
    sample_correlation,
    sample_market_with_correlation,
    sample_multi_asset_gbm_params,
)
from valax.models.multi_asset import MultiAssetGBMModel, validate_correlation
from valax.pricing.mc.multi_asset_paths import generate_correlated_gbm_paths


pytestmark = pytest.mark.arbitrage


# ── Non-PSD correlation ────────────────────────────────────────────


class TestNonPSDCorrelation:
    """Non-PSD correlation matrix in the multi-asset MC pipeline."""

    def test_validate_correlation_returns_negative_min_eig(
        self, seed_registry
    ):
        """`validate_correlation` is the existing soft check.

        It returns the minimum eigenvalue; the user is expected to
        compare against a tolerance.  This test documents the current
        behaviour — *it is the baseline*, not the desired endpoint.
        """
        c = sample_correlation(seed_registry, n=5)
        bad, _ = inject_non_psd_correlation(c, eps=0.5)
        min_eig = float(validate_correlation(bad))
        assert min_eig < 0

    def test_cholesky_returns_nan_for_non_psd(self, seed_registry):
        """Cholesky on a non-PSD matrix returns NaN (the current
        ``jax.numpy.linalg.cholesky`` behaviour on CPU)."""
        c = sample_correlation(seed_registry, n=5)
        bad, _ = inject_non_psd_correlation(c, eps=0.5)
        L = jnp.linalg.cholesky(bad)
        assert bool(jnp.any(jnp.isnan(L)))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Roadmap: MultiAssetGBMModel should raise "
            "NonPSDCorrelationError on construction when the supplied "
            "correlation has min(eigvalsh) < -tol.  Currently silent."
        ),
    )
    def test_model_constructor_should_raise(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        bad, _ = inject_non_psd_correlation(corr, eps=0.5)
        with pytest.raises(NonPSDCorrelationError):
            MultiAssetGBMModel(
                vols=md.vols,
                rate=jnp.array(0.03),
                dividends=md.dividends,
                correlation=bad,
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Roadmap: generate_correlated_gbm_paths should detect a "
            "non-PSD correlation matrix before invoking Cholesky and "
            "raise NonPSDCorrelationError instead of returning NaN paths."
        ),
    )
    def test_paths_should_raise_on_non_psd(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        bad, _ = inject_non_psd_correlation(corr, eps=0.5)
        model = MultiAssetGBMModel(
            vols=md.vols,
            rate=jnp.array(0.03),
            dividends=md.dividends,
            correlation=bad,
        )
        with pytest.raises(NonPSDCorrelationError):
            generate_correlated_gbm_paths(
                model=model,
                spots=md.spots,
                T=1.0,
                n_steps=10,
                n_paths=8,
                key=jax.random.PRNGKey(0),
            )


# ── Basket-variance violation ──────────────────────────────────────


class TestBasketVarianceViolation:
    def test_validate_correlation_detects_overshoot(self, seed_registry):
        """``validate_correlation`` *does* catch this today by
        returning a negative minimum eigenvalue.  Documenting the
        current passing behaviour."""
        c = sample_correlation(seed_registry, n=3)
        bad, _ = inject_basket_variance_violation(c, 0, 1, new_value=1.5)
        min_eig = float(validate_correlation(bad))
        assert min_eig < -1e-6, (
            f"validate_correlation should flag overshoot; got {min_eig}"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Roadmap: MultiAssetGBMModel.__init__ should call "
            "validate_correlation internally and raise "
            "NonPSDCorrelationError instead of leaving the check to "
            "the caller."
        ),
    )
    def test_constructor_should_raise_automatically(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        bad, _ = inject_basket_variance_violation(corr, 0, 1, new_value=1.5)
        with pytest.raises(NonPSDCorrelationError):
            MultiAssetGBMModel(
                vols=md.vols,
                rate=jnp.array(0.03),
                dividends=md.dividends,
                correlation=bad,
            )


# ── Calendar arbitrage in total variance ───────────────────────────


class TestCalendarArbitrage:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Roadmap: no surface-level calendar-arb checker exists yet. "
            "Once SVIVolSurface gains a no_arbitrage assert, swap two "
            "slices and expect a CalendarArbError."
        ),
    )
    def test_surface_constructor_should_reject(self):
        w = jnp.array([0.04, 0.08, 0.12, 0.16])
        bad, _ = inject_calendar_arb(w, i=1, j=3)
        # Placeholder consumer check; will be replaced once a real
        # detector exists.
        is_monotone = bool(jnp.all(bad[1:] >= bad[:-1]))
        assert is_monotone, "calendar arb detector not implemented"


# ── Put-call parity violation ──────────────────────────────────────


class TestPutCallParity:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Roadmap: implied-vol inversion + parity checker should "
            "flag C - P that doesn't match S - K*DF and raise "
            "PutCallParityError on the offending strip."
        ),
    )
    def test_quote_validator_should_reject(self):
        calls = jnp.array([10.0, 5.0, 1.0])
        puts = jnp.array([1.0, 5.0, 10.0])
        (bad_calls, _), _ = inject_pcp_violation(calls, puts, bp=200.0)
        # Placeholder check: today nothing in the library validates
        # this.  When a checker lands, replace with the actual call.
        forward = 100.0
        df = 0.99
        strikes = jnp.array([90.0, 100.0, 110.0])
        parity_lhs = bad_calls - puts
        parity_rhs = df * (forward - strikes)
        max_err = float(jnp.max(jnp.abs(parity_lhs - parity_rhs)))
        assert max_err < 1e-2, "parity violation went undetected"
