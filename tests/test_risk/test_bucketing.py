"""Tests for risk-factor bucketing: linear aggregation and Jacobian reparameterization."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.market.data import MarketData
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.risk.bucketing import (
    BucketMap,
    BucketedLadder,
    aggregate,
    aggregate_covariance,
    aggregate_matrix,
    bucket_sensitivity_ladder,
    equal_weight_bucket_map,
    jacobian_from_fn,
    level_slope_curvature_jacobian,
    pca_jacobian,
    pullback_shocks,
    pushforward_scenario,
    pushforward_sensitivities,
    reparameterize_covariance,
    tenor_bucket_map,
)
from valax.risk.ladders import compute_ladder


# ── BucketMap basics ────────────────────────────────────────────────


class TestBucketMapBasics:
    def test_shape_and_attrs(self):
        A = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        bm = BucketMap(matrix=A, bucket_labels=("short", "long"))
        assert bm.n_buckets == 2
        assert bm.n_factors == 4
        assert bm.bucket_labels == ("short", "long")

    def test_aggregate_basic(self):
        A = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        bm = BucketMap(matrix=A)
        delta = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = aggregate(bm, delta)
        assert jnp.allclose(b, jnp.array([3.0, 7.0]))


# ── Linear duality ─────────────────────────────────────────────────


class TestLinearDuality:
    def test_pnl_invariance(self):
        """<δ_b, Δb> == <δ_x, Δx> when Δx = A^T Δb."""
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        delta_x = jax.random.normal(k1, (12,))
        bucket_shocks = jax.random.normal(k2, (4,))
        A = jax.random.normal(k3, (4, 12))
        bm = BucketMap(matrix=A)

        delta_b = aggregate(bm, delta_x)
        delta_x_from_b = pushforward_scenario(bm, bucket_shocks)

        pnl_factor = jnp.dot(delta_x, delta_x_from_b)
        pnl_bucket = jnp.dot(delta_b, bucket_shocks)
        assert jnp.isclose(pnl_factor, pnl_bucket, rtol=1e-10)


class TestAggregateCovariance:
    def test_psd_preserving(self):
        n = 6
        # Build an arbitrary PSD matrix
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (n, n))
        cov = M @ M.T + 0.1 * jnp.eye(n)

        A = jnp.array([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ])
        bm = BucketMap(matrix=A)
        cov_b = aggregate_covariance(bm, cov)

        assert cov_b.shape == (3, 3)
        # Symmetric
        assert jnp.allclose(cov_b, cov_b.T, atol=1e-10)
        # PSD: all eigenvalues ≥ 0 (up to numerical tolerance)
        eigvals = jnp.linalg.eigvalsh(cov_b)
        assert jnp.all(eigvals >= -1e-10)


# ── Tenor bucketing ───────────────────────────────────────────────


class TestTenorBucketIndicator:
    def test_nearest_assignment(self):
        # Pillars: 0.25, 0.5, 1, 2, 5, 10, 30
        # Buckets:        1,        5,      30
        pillars = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        edges = jnp.array([1.0, 5.0, 30.0])
        bm = tenor_bucket_map(pillars, edges, weight="indicator")
        # Pillars 0.25, 0.5, 1, 2 are nearest to bucket 1 (distance 0.75/0.5/0/1)
        # Pillar 5 → bucket 5; 10 → ... |10-5|=5, |10-30|=20 ⇒ bucket 5
        # Pillar 30 → bucket 30
        delta = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        bucketed = aggregate(bm, delta)
        # bucket "1": 1+2+3+4=10; bucket "5": 5+6=11; bucket "30": 7
        assert jnp.allclose(bucketed, jnp.array([10.0, 11.0, 7.0]))

    def test_indicator_matrix_is_one_hot(self):
        pillars = jnp.array([0.5, 1.5, 2.5])
        edges = jnp.array([0.0, 1.0, 2.0, 3.0])
        bm = tenor_bucket_map(pillars, edges, weight="indicator")
        # Each column should sum to exactly 1 and contain a single 1
        col_sums = jnp.sum(bm.matrix, axis=0)
        assert jnp.allclose(col_sums, 1.0)
        # Each column has exactly one 1.0 entry
        assert jnp.all(jnp.sum(bm.matrix == 1.0, axis=0) == 1)


class TestTenorBucketLinear:
    def test_weights_sum_to_one_per_pillar(self):
        pillars = jnp.array([0.5, 1.0, 1.5, 2.5, 4.0])
        edges = jnp.array([0.0, 1.0, 2.0, 5.0])
        bm = tenor_bucket_map(pillars, edges, weight="linear")
        col_sums = jnp.sum(bm.matrix, axis=0)
        assert jnp.allclose(col_sums, 1.0, atol=1e-10)

    def test_pillar_on_vertex_one_hot(self):
        """A pillar that coincides with a vertex should sit 100% on it."""
        pillars = jnp.array([1.0, 2.0])
        edges = jnp.array([0.0, 1.0, 2.0, 3.0])
        bm = tenor_bucket_map(pillars, edges, weight="linear")
        # Pillar 1.0 should be entirely at bucket 1
        col0 = bm.matrix[:, 0]
        assert jnp.isclose(col0[1], 1.0)
        assert jnp.isclose(jnp.sum(col0 ** 2), 1.0)

    def test_pillar_at_midpoint_splits_half_half(self):
        pillars = jnp.array([1.5])
        edges = jnp.array([0.0, 1.0, 2.0, 3.0])
        bm = tenor_bucket_map(pillars, edges, weight="linear")
        # Pillar 1.5 between buckets 1 and 2: weights 0.5 / 0.5
        col = bm.matrix[:, 0]
        assert jnp.isclose(col[1], 0.5)
        assert jnp.isclose(col[2], 0.5)
        assert jnp.isclose(col[0], 0.0)
        assert jnp.isclose(col[3], 0.0)


# ── Equal-weight (sector) buckets ─────────────────────────────────


class TestEqualWeightBucketMap:
    def test_one_hot_assignment(self):
        # 5 stocks across 2 sectors: [0,0,1,1,0]
        bm = equal_weight_bucket_map(
            group_membership=(0, 0, 1, 1, 0), n_buckets=2,
            bucket_labels=("tech", "energy"),
        )
        delta = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        agg = aggregate(bm, delta)
        # tech (0): 1+2+5 = 8; energy (1): 3+4 = 7
        assert jnp.allclose(agg, jnp.array([8.0, 7.0]))
        assert bm.bucket_labels == ("tech", "energy")


# ── Jacobian reparameterization ───────────────────────────────────


class TestJacobianDuality:
    def test_pushforward_pullback_consistency(self):
        """<J^T δ, Δb> == <δ, J Δb>."""
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        J = jax.random.normal(k1, (10, 3))
        delta = jax.random.normal(k2, (10,))
        bucket_shocks = jax.random.normal(k3, (3,))

        lhs = jnp.dot(pushforward_sensitivities(J, delta), bucket_shocks)
        rhs = jnp.dot(delta, pullback_shocks(J, bucket_shocks))
        assert jnp.isclose(lhs, rhs, rtol=1e-10)


class TestLevelSlopeCurvature:
    def test_shape_and_columns(self):
        t = jnp.linspace(0.5, 30.0, 10)
        J = level_slope_curvature_jacobian(t)
        assert J.shape == (10, 3)
        # Level is constant
        assert jnp.allclose(J[:, 0], 1.0)
        # Slope is monotone
        assert jnp.all(jnp.diff(J[:, 1]) > 0)

    def test_full_rank(self):
        t = jnp.linspace(0.5, 30.0, 10)
        J = level_slope_curvature_jacobian(t)
        assert jnp.linalg.matrix_rank(J) == 3


class TestPCAJacobian:
    def test_orthonormal_columns(self):
        key = jax.random.PRNGKey(2)
        X = jax.random.normal(key, (500, 8))
        J, eigvals, frac = pca_jacobian(X, n_components=4, center=True)
        # Columns should be orthonormal: J^T J ≈ I
        gram = J.T @ J
        assert jnp.allclose(gram, jnp.eye(4), atol=1e-8)
        # Eigenvalues sorted descending
        assert jnp.all(jnp.diff(eigvals) <= 0.0)
        # Fraction explained in [0, 1]
        assert 0.0 <= float(frac) <= 1.0

    def test_low_rank_reconstruction(self):
        """For a perfectly low-rank input the top-k PCs reconstruct exactly."""
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        # Build a rank-2 dataset: X = u v^T with two singular components
        n_obs, n_factors = 200, 6
        scores = jax.random.normal(k1, (n_obs, 2))
        loadings = jax.random.normal(k2, (2, n_factors))
        X = scores @ loadings  # exactly rank 2

        J, _, frac = pca_jacobian(X, n_components=2, center=False)
        assert float(frac) > 0.999

        # Project and reconstruct: X ≈ X J J^T (because J^T J = I)
        X_recon = X @ J @ J.T
        assert jnp.allclose(X, X_recon, atol=1e-6)


class TestReparameterizeCovariance:
    def test_diagonal_pca_covariance(self):
        """In PCA coordinates, covariance is diagonal with eigenvalues."""
        key = jax.random.PRNGKey(4)
        X = jax.random.normal(key, (1000, 5))
        cov = jnp.cov(X.T)
        J, eigvals, _ = pca_jacobian(X, n_components=5, center=True)
        cov_b = reparameterize_covariance(J, cov)
        # Diagonal should match the eigenvalues
        diag = jnp.diag(cov_b)
        assert jnp.allclose(diag, eigvals, atol=1e-6)
        # Off-diagonal close to zero
        off_diag = cov_b - jnp.diag(diag)
        assert jnp.max(jnp.abs(off_diag)) < 1e-6


class TestJacobianFromFn:
    def test_matches_linear_case(self):
        """For a linear b→x map, jacobian_from_fn should recover the matrix."""
        M = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(b):
            return M @ b

        b0 = jnp.zeros(2)
        J = jacobian_from_fn(f, b0)
        assert jnp.allclose(J, M)


# ── Aggregate matrix (cross-block bucketing) ──────────────────────


class TestAggregateMatrix:
    def test_bilateral_aggregation(self):
        # 4 rows × 6 cols → 2 row-buckets × 3 col-buckets
        M = jnp.arange(24, dtype=jnp.float64).reshape(4, 6)
        A_rows = jnp.array([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ])
        A_cols = jnp.array([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ])
        bm_r = BucketMap(matrix=A_rows)
        bm_c = BucketMap(matrix=A_cols)
        agg = aggregate_matrix(bm_r, M, bm_c)
        # Manual: A_rows @ M @ A_cols.T
        expected = A_rows @ M @ A_cols.T
        assert jnp.allclose(agg, expected)
        assert agg.shape == (2, 3)


# ── Bucket sensitivity ladder (end-to-end) ────────────────────────


def _bs_market_fn(option, market: MarketData) -> jnp.ndarray:
    from valax.risk.var import _extract_short_rate
    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(
        option, market.spots, market.vols, rate, market.dividends,
    )


@pytest.fixture
def two_asset_portfolio():
    instruments = EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([0.5, 1.0]),
        is_call=True,
    )
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2026, 7, 1),
        ymd_to_ordinal(2027, 1, 1),
        ymd_to_ordinal(2031, 1, 1),
    ])
    rate = 0.04
    times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    curve = DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref,
    )
    base = MarketData(
        spots=jnp.array([100.0, 100.0]),
        vols=jnp.array([0.2, 0.25]),
        dividends=jnp.array([0.0, 0.0]),
        discount_curve=curve,
    )
    return instruments, base


class TestBucketSensitivityLadder:
    def test_shapes(self, two_asset_portfolio):
        instruments, base = two_asset_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        # Bucket 4 pillars → 2 buckets (short / long)
        rate_bm = BucketMap(
            matrix=jnp.array([
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]),
            bucket_labels=("short", "long"),
        )
        # Bucket 2 assets → 1 sector
        spot_bm = equal_weight_bucket_map((0, 0), n_buckets=1)

        bucketed = bucket_sensitivity_ladder(
            ladder, rate_bucket=rate_bm, spot_bucket=spot_bm,
        )

        assert bucketed.delta_rate.shape == (2,)
        assert bucketed.delta_spot.shape == (1,)
        # Vol/div not bucketed → identity, retains n_assets = 2
        assert bucketed.delta_vol.shape == (2,)
        assert bucketed.cross_spot_rate.shape == (1, 2)
        assert bucketed.cross_vol_rate.shape == (2, 2)
        assert bucketed.rate_bucket_labels == ("short", "long")

    def test_aggregation_preserves_total_pnl(self, two_asset_portfolio):
        """A bucket shock pushed back to factor space gives the same PnL
        as the bucket-space PnL."""
        instruments, base = two_asset_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        rate_bm = BucketMap(
            matrix=jnp.array([
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]),
        )
        bucketed_dv01 = aggregate(rate_bm, ladder.delta_rate)
        bucket_shock = jnp.array([0.0010, 0.0025])  # +10 bp short, +25 bp long

        # PnL in bucket space
        pnl_bucket = jnp.dot(bucketed_dv01, bucket_shock)
        # PnL in raw factor space using the dual shock
        raw_shock = pushforward_scenario(rate_bm, bucket_shock)
        pnl_raw = jnp.dot(ladder.delta_rate, raw_shock)
        assert jnp.isclose(pnl_bucket, pnl_raw, rtol=1e-10)
