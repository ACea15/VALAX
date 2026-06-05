"""Tests that ground-truth model parameter samplers respect each
model's domain (Feller condition, ``|rho| < 1``, positivity, ...)."""

import jax.numpy as jnp

from valax.market.synthetic import (
    flat_discount_curve,
    sample_bs_params,
    sample_heston_params,
    sample_hull_white_params,
    sample_multi_asset_gbm_params,
    sample_sabr_params,
)


class TestBlackScholes:
    def test_vol_positive(self, seed_registry, default_synth_cfg):
        m = sample_bs_params(seed_registry, default_synth_cfg)
        assert float(m.vol) > 0


class TestHeston:
    def test_feller_condition_satisfied(
        self, seed_registry, default_synth_cfg
    ):
        m = sample_heston_params(seed_registry, default_synth_cfg)
        # 2 * kappa * theta >= xi^2
        slack = 2.0 * float(m.kappa) * float(m.theta) - float(m.xi) ** 2
        assert slack > 0, (
            f"Feller violated: 2*kappa*theta - xi^2 = {slack:g}"
        )

    def test_rho_in_open_unit_interval(
        self, seed_registry, default_synth_cfg
    ):
        m = sample_heston_params(seed_registry, default_synth_cfg)
        assert -1.0 < float(m.rho) < 1.0

    def test_all_positive_where_required(
        self, seed_registry, default_synth_cfg
    ):
        m = sample_heston_params(seed_registry, default_synth_cfg)
        for name in ("v0", "kappa", "theta", "xi"):
            assert float(getattr(m, name)) > 0, f"{name} not positive"


class TestSABR:
    def test_alpha_nu_positive(self, seed_registry, default_synth_cfg):
        m = sample_sabr_params(seed_registry, default_synth_cfg)
        assert float(m.alpha) > 0
        assert float(m.nu) > 0

    def test_beta_in_unit_interval(self, seed_registry, default_synth_cfg):
        m = sample_sabr_params(seed_registry, default_synth_cfg)
        assert 0.0 <= float(m.beta) <= 1.0

    def test_rho_in_open_unit_interval(
        self, seed_registry, default_synth_cfg
    ):
        m = sample_sabr_params(seed_registry, default_synth_cfg)
        assert -1.0 < float(m.rho) < 1.0


class TestHullWhite:
    def test_positivity(self, seed_registry):
        from valax.dates.daycounts import ymd_to_ordinal

        curve = flat_discount_curve(
            rate=0.03, reference_date=ymd_to_ordinal(2026, 1, 1)
        )
        m = sample_hull_white_params(seed_registry, curve)
        assert float(m.mean_reversion) > 0
        assert float(m.volatility) > 0


class TestMultiAssetGBM:
    def test_shapes(self, seed_registry, default_synth_cfg):
        m = sample_multi_asset_gbm_params(
            seed_registry, default_synth_cfg
        )
        n = default_synth_cfg.n_assets
        assert m.vols.shape == (n,)
        assert m.dividends.shape == (n,)
        assert m.correlation.shape == (n, n)

    def test_correlation_valid(self, seed_registry, default_synth_cfg):
        m = sample_multi_asset_gbm_params(
            seed_registry, default_synth_cfg
        )
        assert bool(jnp.allclose(jnp.diag(m.correlation), 1.0, atol=1e-8))
        min_eig = float(jnp.min(jnp.linalg.eigvalsh(m.correlation)))
        assert min_eig > -1e-8
