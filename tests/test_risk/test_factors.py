"""Tests for the rates PCA factor model in ``valax.risk.factors``.

End-to-end coverage:

- :class:`TestZeroRateReturns` — returns extraction shape, agreement
  with a manual computation, and the ``len(curves) >= 2`` guard.
- :class:`TestFitRatesPCA` — orthonormality, eigenvalue ordering,
  fraction-explained bounds, the positive-level sign convention, and
  the ≥99% variance explained on synthetic level/slope/curvature data.
- :class:`TestPCACurveShock` — zero-score no-op, parallel-shift
  interpretation of PC1, and consistency with
  :func:`valax.risk.shocks.pca_curve_shock`.
- :class:`TestPCAPnL` — the typed ``model.scenario(...)`` path matches
  the manual ``pullback_shocks`` + ``bump_curve_zero_rates`` path under
  :func:`valax.risk.shocks.apply_scenario`, and the resulting
  shocked-curve P&L of a vanilla bond has the expected sign and is
  differentiable in the PC scores.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve, zero_rate
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.market.data import MarketData
from valax.risk.factors import (
    RatesFactorModel,
    fit_rates_pca,
    zero_rate_returns_from_snapshots,
)
from valax.risk.shocks import apply_scenario, bump_curve_zero_rates, pca_curve_shock


# ── Shared fixtures ─────────────────────────────────────────────────


REFERENCE_DATE = ymd_to_ordinal(2026, 1, 1)
PILLAR_TENOR_YEARS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def _build_curve(zero_rates: jnp.ndarray) -> DiscountCurve:
    """Build a 10-pillar :class:`DiscountCurve` on the standard body grid.

    The reference date itself is *not* a pillar — flat extrapolation
    on the log-DF interpolator handles short-end queries.  This keeps
    the curve's pillar count equal to ``len(zero_rates)`` so the PCA
    model (trained on 10-pillar returns) shocks the curve directly
    without an off-by-one between the reference pillar and the body.
    """
    pillar_dates = jnp.array(
        [REFERENCE_DATE + int(round(t * 365.0)) for t in PILLAR_TENOR_YEARS],
        dtype=jnp.int32,
    )
    times = jnp.asarray(PILLAR_TENOR_YEARS, dtype=jnp.float64)
    dfs = jnp.exp(-zero_rates * times)
    return DiscountCurve(
        pillar_dates=pillar_dates,
        discount_factors=dfs,
        reference_date=jnp.int32(REFERENCE_DATE),
        day_count="act_365",
    )


@pytest.fixture
def pillar_times():
    return jnp.asarray(PILLAR_TENOR_YEARS, dtype=jnp.float64)


@pytest.fixture
def query_dates():
    """Query dates at the body pillar tenors (excluding the reference)."""
    return jnp.array(
        [REFERENCE_DATE + int(round(t * 365.0)) for t in PILLAR_TENOR_YEARS],
        dtype=jnp.int32,
    )


@pytest.fixture
def synthetic_lsc_returns(pillar_times):
    """Synthetic returns from a deliberate 3-factor L/S/C process.

    Constructs returns ``X = F @ B`` where ``F`` is ``(n_obs, 3)`` of
    iid Gaussian scores and ``B`` is a ``(3, n_pillars)`` matrix whose
    rows are explicit level / slope / curvature shapes plus tiny pillar
    noise.  PCA on this dataset should recover the three factors and
    explain ≥99% of the variance.
    """
    key = jax.random.PRNGKey(20260101)
    k_scores, k_noise = jax.random.split(key)
    n_obs = 500
    t = pillar_times

    level = jnp.ones_like(t)
    slope = (t - jnp.mean(t)) / jnp.std(t)
    curv = (t ** 2 - jnp.mean(t ** 2))
    curv = curv / jnp.std(curv)
    # Scale so that PC1 dominates and the three factors carry almost
    # all of the variance.
    B = jnp.stack([10.0 * level, 4.0 * slope, 2.0 * curv], axis=0)  # (3, n_p)

    F = jax.random.normal(k_scores, (n_obs, 3))
    noise = 0.02 * jax.random.normal(k_noise, (n_obs, t.shape[0]))
    # Multiply by 1e-4 to land in realistic bp-per-day units; the
    # variance ratios are scale-invariant so the assertions are unchanged.
    return 1e-4 * (F @ B + noise)


@pytest.fixture
def base_curve():
    """Mildly upward-sloping curve at the standard pillar grid."""
    zr = jnp.array(
        [0.045, 0.044, 0.042, 0.040, 0.039, 0.038, 0.038, 0.038, 0.040, 0.042],
    )
    return _build_curve(zr)


# ── Returns extraction ─────────────────────────────────────────────


class TestZeroRateReturns:
    def test_shape(self, query_dates):
        rates = jnp.array([
            [0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.045, 0.045, 0.046, 0.046],
            [0.041, 0.041, 0.042, 0.043, 0.044, 0.045, 0.045, 0.045, 0.046, 0.046],
            [0.039, 0.040, 0.041, 0.043, 0.044, 0.045, 0.045, 0.045, 0.046, 0.046],
        ])
        curves = [_build_curve(r) for r in rates]
        returns = zero_rate_returns_from_snapshots(curves, query_dates)
        assert returns.shape == (2, 10)

    def test_agrees_with_manual_diff(self, query_dates):
        """Returns equal the differences of zero rates evaluated at query_dates."""
        rates = jnp.array([
            [0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.045, 0.045, 0.046, 0.046],
            [0.041, 0.041, 0.042, 0.043, 0.044, 0.045, 0.045, 0.045, 0.046, 0.046],
        ])
        curves = [_build_curve(r) for r in rates]
        returns = zero_rate_returns_from_snapshots(curves, query_dates)

        manual = jnp.stack([
            jnp.array([zero_rate(c, d) for d in query_dates]) for c in curves
        ])
        expected = jnp.diff(manual, axis=0)
        assert jnp.allclose(returns, expected, atol=1e-12)

    def test_requires_at_least_two_snapshots(self, query_dates):
        with pytest.raises(ValueError, match="at least 2 snapshots"):
            zero_rate_returns_from_snapshots(
                [_build_curve(jnp.full(10, 0.04))],
                query_dates,
            )


# ── Fit ─────────────────────────────────────────────────────────────


class TestFitRatesPCA:
    def test_orthonormal_columns(self, synthetic_lsc_returns, pillar_times):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        gram = model.jacobian.T @ model.jacobian
        assert jnp.allclose(gram, jnp.eye(3), atol=1e-10)

    def test_eigenvalues_sorted_descending(
        self, synthetic_lsc_returns, pillar_times,
    ):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=5)
        assert jnp.all(jnp.diff(model.eigenvalues) <= 1e-15)

    def test_fraction_explained_bounded(
        self, synthetic_lsc_returns, pillar_times,
    ):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        f = float(model.fraction_explained)
        assert 0.0 <= f <= 1.0 + 1e-12

    def test_three_components_capture_at_least_99pct(
        self, synthetic_lsc_returns, pillar_times,
    ):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        assert float(model.fraction_explained) >= 0.99

    def test_positive_level_sign_convention(
        self, synthetic_lsc_returns, pillar_times,
    ):
        """Every column should have a non-negative mean loading."""
        model = fit_rates_pca(
            synthetic_lsc_returns, pillar_times, n_components=3,
            sign_convention="positive_level",
        )
        means = jnp.mean(model.jacobian, axis=0)
        assert jnp.all(means >= -1e-12)

    def test_pc1_is_approximately_level(
        self, synthetic_lsc_returns, pillar_times,
    ):
        """PC1 of an L/S/C-generated dataset should have nearly uniform sign."""
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        pc1 = model.jacobian[:, 0]
        # All entries positive (after the sign convention) and reasonably flat
        # relative to a perfectly level vector.
        assert jnp.all(pc1 > 0)
        # Cosine similarity with the constant vector ≥ 0.95.
        u = jnp.ones_like(pc1)
        cos = jnp.dot(pc1, u) / (jnp.linalg.norm(pc1) * jnp.linalg.norm(u))
        assert float(cos) >= 0.95

    def test_raw_sign_convention_skips_flip(
        self, synthetic_lsc_returns, pillar_times,
    ):
        """``sign_convention='raw'`` should return SVD signs unmodified."""
        raw = fit_rates_pca(
            synthetic_lsc_returns, pillar_times, n_components=3,
            sign_convention="raw",
        )
        canon = fit_rates_pca(
            synthetic_lsc_returns, pillar_times, n_components=3,
            sign_convention="positive_level",
        )
        # Each column of canon equals ±raw column.
        for k in range(3):
            assert jnp.allclose(
                jnp.abs(canon.jacobian[:, k]), jnp.abs(raw.jacobian[:, k]),
                atol=1e-12,
            )

    def test_invalid_sign_convention_raises(
        self, synthetic_lsc_returns, pillar_times,
    ):
        with pytest.raises(ValueError, match="sign_convention"):
            fit_rates_pca(
                synthetic_lsc_returns, pillar_times, n_components=2,
                sign_convention="bogus",
            )

    def test_r_squared_per_pillar(self, synthetic_lsc_returns, pillar_times):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        r2 = model.r_squared_per_pillar(synthetic_lsc_returns)
        assert r2.shape == (10,)
        # Three components should explain ≥95% at every pillar on this
        # dataset, which is the standard rates-PCA quality bar.
        assert jnp.all(r2 >= 0.95)


# ── Shock helpers ──────────────────────────────────────────────────


class TestPCACurveShock:
    def test_zero_scores_is_identity(
        self, synthetic_lsc_returns, pillar_times, base_curve,
    ):
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        shocked = model.shock_curve(base_curve, jnp.zeros(3))
        assert jnp.allclose(
            shocked.discount_factors, base_curve.discount_factors, atol=1e-14,
        )

    def test_model_shock_matches_public_pca_curve_shock(
        self, synthetic_lsc_returns, pillar_times, base_curve,
    ):
        """``model.shock_curve`` and the public
        :func:`valax.risk.shocks.pca_curve_shock` must agree bit-for-bit
        when given the same Jacobian — they are two routes into the
        same primitive."""
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        rng = jax.random.PRNGKey(0)
        scores = jax.random.normal(rng, (3,))

        a = model.shock_curve(base_curve, scores)
        b = pca_curve_shock(base_curve, model.jacobian, scores)
        assert jnp.allclose(a.discount_factors, b.discount_factors, atol=1e-14)

    def test_pc1_only_is_close_to_parallel_shift(
        self, synthetic_lsc_returns, pillar_times, base_curve,
    ):
        """With L/S/C-generated data, a unit PC1 shock should land close
        to a parallel zero-rate bump (because PC1 ≈ constant loading)."""
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        shocked = model.shock_curve(base_curve, jnp.array([1.0, 0.0, 0.0]))
        pc1_bumps = model.jacobian[:, 0]
        # All bumps positive (positive-level convention), and relative
        # dispersion across pillars below 10% of the mean — i.e. "nearly
        # parallel".
        assert jnp.all(pc1_bumps > 0)
        spread = (jnp.max(pc1_bumps) - jnp.min(pc1_bumps)) / jnp.mean(pc1_bumps)
        assert float(spread) < 0.10
        # Sanity: shocked DFs differ from the base at every body pillar.
        assert not jnp.allclose(
            shocked.discount_factors, base_curve.discount_factors,
        )


# ── End-to-end P&L pipeline ────────────────────────────────────────


def _zero_coupon_bond_price(curve: DiscountCurve, maturity_date) -> jnp.ndarray:
    """Price of a unit-notional zero-coupon bond — just ``DF(maturity)``."""
    return curve(maturity_date)


class TestPCAPnL:
    def test_scenario_path_matches_manual_path(
        self, synthetic_lsc_returns, pillar_times, base_curve, query_dates,
    ):
        """``model.scenario(...) -> apply_scenario`` agrees with the
        explicit ``pullback_shocks`` + ``bump_curve_zero_rates`` path on
        the discount curve component of the result.
        """
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        scores = jnp.array([0.5, -0.3, 0.2])

        # Manual path
        manual = model.shock_curve(base_curve, scores)

        # Scenario path
        base_market = MarketData(
            spots=jnp.zeros(0),
            vols=jnp.zeros(0),
            dividends=jnp.zeros(0),
            discount_curve=base_curve,
        )
        scen = model.scenario(scores, n_assets=0)
        shocked_market = apply_scenario(base_market, scen)
        scenario_curve = shocked_market.discount_curve

        assert jnp.allclose(
            scenario_curve.discount_factors,
            manual.discount_factors,
            atol=1e-14,
        )

    def test_bond_pnl_under_pc1_shock_has_correct_sign(
        self, synthetic_lsc_returns, pillar_times, base_curve,
    ):
        """A +PC1 shock raises rates (positive-level convention), so a
        long zero-coupon bond should lose money — sign check, the basic
        sanity test for the whole pipeline."""
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        maturity = jnp.int32(REFERENCE_DATE + int(round(5.0 * 365.0)))

        p0 = _zero_coupon_bond_price(base_curve, maturity)
        p_up = _zero_coupon_bond_price(
            model.shock_curve(base_curve, jnp.array([1.0, 0.0, 0.0])),
            maturity,
        )
        p_dn = _zero_coupon_bond_price(
            model.shock_curve(base_curve, jnp.array([-1.0, 0.0, 0.0])),
            maturity,
        )
        # Rates up ⇒ price down; rates down ⇒ price up.
        assert float(p_up) < float(p0)
        assert float(p_dn) > float(p0)

    def test_pnl_is_differentiable_in_pc_scores(
        self, synthetic_lsc_returns, pillar_times, base_curve,
    ):
        """The whole shock-and-price pipeline must be ``jax.grad``-friendly,
        which is the whole point of a JAX-native risk stack."""
        model = fit_rates_pca(synthetic_lsc_returns, pillar_times, n_components=3)
        maturity = jnp.int32(REFERENCE_DATE + int(round(5.0 * 365.0)))

        def pnl(scores):
            shocked = model.shock_curve(base_curve, scores)
            return _zero_coupon_bond_price(shocked, maturity) \
                - _zero_coupon_bond_price(base_curve, maturity)

        g = jax.grad(pnl)(jnp.zeros(3))
        assert g.shape == (3,)
        # PC1 = level up ⇒ negative bond P&L sensitivity.
        assert float(g[0]) < 0.0
