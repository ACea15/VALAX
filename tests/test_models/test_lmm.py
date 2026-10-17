"""The LMM is curve-free: its initial forwards are read from a bound curve."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves, curve_requirements
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.rates import Caplet
from valax.models.lmm import (
    ExponentialCorrelation,
    LMMModel,
    PiecewiseConstantVol,
    build_lmm_model,
    compute_loading_matrix,
)
from valax.pricing.mc.lmm_paths import generate_lmm_paths

REF = int(ymd_to_ordinal(2025, 1, 1))
TENOR = jnp.array(
    [int(ymd_to_ordinal(y, 1, 1)) for y in range(2026, 2031)], dtype=jnp.int32
)


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + y, 1, 1)) for y in range(13)], dtype=jnp.int32
    )
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
        reference_date=jnp.int32(REF),
    )


def _free(**kw) -> LMMModel:
    return LMMModel(
        tenor_dates=TENOR,
        vol_structure=PiecewiseConstantVol(vols=0.2 * jnp.ones((4, 4))),
        corr_structure=ExponentialCorrelation(beta=jnp.array(0.1)),
        **kw,
    )


def test_constructed_curve_free():
    model = _free()
    assert model.initial_curve is None
    with pytest.raises(ValueError, match="not bound to a curve"):
        model.initial_forwards


def test_curve_derived_quantities():
    curve = _curve(0.04)
    model = bind_curves(_free(), curve)
    taus = year_fraction(TENOR[:-1], TENOR[1:], curve.day_count)
    np.testing.assert_array_equal(model.accrual_fractions, taus)
    np.testing.assert_allclose(
        model.initial_forwards,
        (curve(TENOR[:-1]) / curve(TENOR[1:]) - 1.0) / taus, rtol=0, atol=0,
    )
    np.testing.assert_array_equal(
        model.tenor_times, year_fraction(curve.reference_date, TENOR, curve.day_count)
    )
    assert float(model.initial_df) == float(curve(TENOR[0]))
    np.testing.assert_array_equal(
        model.loading_matrix,
        compute_loading_matrix(
            model.corr_structure.matrix(model.tenor_times[:-1]), None
        ),
    )
    assert model.day_count == curve.day_count


def test_build_lmm_model_is_bound_free_model():
    curve = _curve(0.04)
    built = build_lmm_model(
        curve, TENOR, _free().vol_structure, _free().corr_structure
    )
    bound = bind_curves(_free(), curve)
    a = generate_lmm_paths(built, 5, 200, jax.random.PRNGKey(0))
    b = generate_lmm_paths(bound, 5, 200, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(a.discount_factors, b.discount_factors)
    np.testing.assert_array_equal(a.forwards_at_tenors, b.forwards_at_tenors)


def test_requirements_curve_id_else_discount():
    caplet = Caplet(
        fixing_date=TENOR[0], start_date=TENOR[0], end_date=TENOR[1],
        strike=jnp.array(0.04), notional=jnp.array(1.0), discount_id="USD.SOFR.OIS",
    )
    assert curve_requirements(_free(), caplet) == {"initial_curve": "USD.SOFR.OIS"}
    assert curve_requirements(_free(curve_id="USD.SOFR.3M"), caplet) == {
        "initial_curve": "USD.SOFR.3M"
    }


def test_curve_free_models_stack_under_filter_vmap():
    """Two curve-free models with different vols batch like any pytree."""
    vols = jnp.stack([0.2 * jnp.ones((4, 4)), 0.3 * jnp.ones((4, 4))])
    models = eqx.filter_vmap(
        lambda v: LMMModel(
            tenor_dates=TENOR, vol_structure=PiecewiseConstantVol(vols=v),
            corr_structure=ExponentialCorrelation(beta=jnp.array(0.1)),
        )
    )(vols)
    assert models.vol_structure.vols.shape == (2, 4, 4)
    assert models.initial_curve is None


def test_correlation_reaches_the_paths():
    """The loadings derive from corr_structure, so its gradient is live."""
    curve = _curve(0.04)

    def mean_last_df(beta):
        model = bind_curves(
            LMMModel(
                tenor_dates=TENOR,
                vol_structure=PiecewiseConstantVol(vols=0.2 * jnp.ones((4, 4))),
                corr_structure=ExponentialCorrelation(beta=beta),
            ),
            curve,
        )
        result = generate_lmm_paths(model, 4, 500, jax.random.PRNGKey(1))
        return jnp.mean(result.forwards_at_fixing[:, -1])

    g = jax.grad(mean_last_df)(jnp.array(0.1))
    assert jnp.isfinite(g) and float(g) != 0.0
