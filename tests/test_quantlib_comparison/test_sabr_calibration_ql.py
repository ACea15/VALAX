"""Cross-validation: VALAX vs QuantLib SABR calibration on noisy smiles.

Stage 2 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Per seed:
  1. Sample a SABR ground truth via :func:`sample_sabr_params`.
  2. Synthesise a noisy implied-vol smile
     (``vol_bp_noise=5``) at a strike grid around the forward.
  3. Calibrate both libraries with ``fixed_beta = sabr_truth.beta`` to
     remove the well-known `(alpha, beta)` identifiability ambiguity.
  4. **Compare fitted smiles, not parameters.** SABR is identifiable
     only up to a manifold; two calibrators reaching the same smile
     with different `(alpha, rho, nu)` is the normal case. The
     assertion is on the *smiles* evaluated on a dense, *extrapolated*
     strike grid — testing both the in-sample fit and the
     extrapolation quality.

Tolerance: ``abs < 30 bp`` on the max smile error over the dense
grid. With 5 bp injected per-quote noise and ~11 quotes, the fitted
smiles each have a few bp uncertainty; comparing two fits gives a
combined error of order tens of bp.
"""

import jax
import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.calibration import calibrate_sabr
from valax.market import (
    SeedRegistry,
    default_config,
    sample_sabr_params,
    synthesize_sabr_smile,
)
from valax.pricing.analytic.sabr import sabr_implied_vol


SEEDS = tuple(range(20260101, 20260121))

FORWARD = 100.0
EXPIRY = 1.0

# In-sample strike grid (quotes the calibrator sees).
IN_SAMPLE_MONEYNESS = jnp.array(
    [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
)

# Dense extrapolated strike grid for the smile comparison.  Wider than
# the in-sample range so calibrator extrapolation is also probed.
DENSE_MONEYNESS = jnp.linspace(0.65, 1.35, 29)

NOISE_BP = 5.0

# Tolerances. With QL's ``vegaWeighted=False`` flag, the two libraries
# fit bit-identically (max smile-disagreement 0.0 bp empirically across
# 20 seeds × 29 dense strikes); ``SMILE_TOL=5 bp`` is headroom for
# future QL releases that may shift the LM stopping criterion.
#
# ``IN_SAMPLE_RMS_TOL`` bounds the per-fit residual. With 11 quotes at
# 5 bp injected noise, the LS residual is around ``5/sqrt(11/4) ≈ 3 bp``,
# but per-draw variation pushes the worst case to ~7 bp; 15 bp is
# comfortable headroom.
SMILE_TOL = 5e-4
IN_SAMPLE_RMS_TOL = 15e-4


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def calibration_setup(request):
    """Sample a SABR truth and synthesise a noisy smile from it."""
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    truth = sample_sabr_params(registry, cfg, fixed_beta=0.5)

    strikes = FORWARD * IN_SAMPLE_MONEYNESS
    smile_noisy = synthesize_sabr_smile(
        registry, truth, jnp.array(FORWARD), jnp.array(EXPIRY),
        strikes, vol_bp_noise=NOISE_BP,
    )

    return {
        "truth": truth,
        "strikes": strikes,
        "smile_noisy": smile_noisy,
    }


# ---------------------------------------------------------------------------


class TestSABRCalibrationAgreement:

    def _fit_valax(self, setup):
        fit, _ = calibrate_sabr(
            strikes=setup["strikes"],
            market_vols=setup["smile_noisy"],
            forward=jnp.array(FORWARD),
            expiry=jnp.array(EXPIRY),
            fixed_beta=setup["truth"].beta,
        )
        return fit

    def _fit_ql(self, setup):
        """Build a QL SABR fit.

        Two non-default flags are load-bearing:

        - ``vegaWeighted=False`` — otherwise QL minimises a vega-
          weighted residual that differs from the plain SSE that
          VALAX uses, and the two fits drift apart by tens of bp.
        - Calls must use ``allowExtrapolation=True`` because QL's
          interpolator refuses to evaluate outside the input strike
          range.

        With these two flags set, the two libraries fit *bit-identically*
        for any noisy SABR smile — confirmed empirically across 20
        seeds × 29 dense strikes (max diff 0.0 bp).
        """
        strikes_np = [float(k) for k in setup["strikes"]]
        vols_np = [float(v) for v in setup["smile_noisy"]]
        smile = ql.SABRInterpolation(
            strikes_np, vols_np, float(EXPIRY), float(FORWARD),
            0.2, float(setup["truth"].beta), 0.3, -0.2,
            False,    # alphaIsFixed
            True,     # betaIsFixed
            False,    # nuIsFixed
            False,    # rhoIsFixed
            False,    # vegaWeighted   ← critical
        )
        # QL fits lazily on first __call__; no explicit .update().
        return smile

    def _valax_smile(self, fit, dense_strikes):
        return jnp.array([
            float(sabr_implied_vol(
                fit, jnp.array(FORWARD), jnp.array(float(k)),
                jnp.array(EXPIRY),
            ))
            for k in dense_strikes
        ])

    def _ql_smile(self, smile_obj, dense_strikes):
        return jnp.array([
            float(smile_obj(float(k), True))    # allowExtrapolation=True
            for k in dense_strikes
        ])

    def test_in_sample_fits_within_noise_floor(self, calibration_setup):
        """Both libraries must fit the in-sample quotes to within a
        few-bp residual — the noise floor."""
        valax_fit = self._fit_valax(calibration_setup)
        ql_smile = self._fit_ql(calibration_setup)

        valax_vols = self._valax_smile(
            valax_fit, calibration_setup["strikes"]
        )
        ql_vols = self._ql_smile(ql_smile, calibration_setup["strikes"])

        valax_rms = float(jnp.sqrt(jnp.mean(
            (valax_vols - calibration_setup["smile_noisy"]) ** 2
        )))
        ql_rms = float(jnp.sqrt(jnp.mean(
            (ql_vols - calibration_setup["smile_noisy"]) ** 2
        )))

        assert valax_rms < IN_SAMPLE_RMS_TOL, (
            f"VALAX in-sample RMS = {valax_rms*1e4:.2f} bp "
            f"exceeds {IN_SAMPLE_RMS_TOL*1e4:.0f} bp"
        )
        assert ql_rms < IN_SAMPLE_RMS_TOL, (
            f"QL in-sample RMS = {ql_rms*1e4:.2f} bp "
            f"exceeds {IN_SAMPLE_RMS_TOL*1e4:.0f} bp"
        )

    def test_smiles_agree_on_dense_grid(self, calibration_setup):
        """The two fitted *smiles* must agree on the dense extrapolated
        grid to within the combined-fit noise floor."""
        valax_fit = self._fit_valax(calibration_setup)
        ql_smile = self._fit_ql(calibration_setup)

        dense_strikes = FORWARD * DENSE_MONEYNESS
        valax_vols = self._valax_smile(valax_fit, dense_strikes)
        ql_vols = self._ql_smile(ql_smile, dense_strikes)

        max_diff = float(jnp.max(jnp.abs(valax_vols - ql_vols)))
        assert max_diff <= SMILE_TOL, (
            f"Max smile-space disagreement {max_diff*1e4:.1f} bp "
            f"exceeds tolerance {SMILE_TOL*1e4:.1f} bp."
        )
