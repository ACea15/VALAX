"""Guarded nested root-finds: a failed inner solve is NaN, never a raise.

The Jamshidian critical rate (Hull-White) and the Brigo-Mercurio exercise
boundary (G2++) are Newton solves nested inside the pricer and differentiated
implicitly.  Unguarded, an absurd or non-finite parameter made the inner
solve fail, and the implicit derivative then solved a non-finite linear
system — which *raises at runtime* inside ``jit`` / ``vmap`` (lineax's
non-finite check), aborting a whole calibration or book batch for one bad
trade.  The guard (a detached probe solve deciding ``ok``, then the
differentiable solve on ``where(ok, real, stand-in)``) turns that into a NaN
price and a NaN gradient for the bad entry only.

Anti-vacuity: each test first shows the failure path is really taken (NaN for
the absurd entry) and that the healthy entry is finite and unaffected.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.hull_white_swaptions import (
    hw_critical_rate,
    hw_swaption_price,
)

_ACT365 = "act_365"
_REF = int(ymd_to_ordinal(2025, 1, 1))
# Healthy, then absurd / non-finite volatilities that make the inner solve fail.
_HEALTHY_SIGMA = 0.01
_BAD_SIGMAS = (3.0, 10.0, float("nan"))


def _flat_curve(rate: float = 0.03, n_years: int = 21) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - _REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(_REF),
        day_count=_ACT365,
    )


_CURVE = _flat_curve()


def _swaption(expiry_y: int = 2, tenor_y: int = 5, strike: float = 0.03) -> Swaption:
    fixed = jnp.array(
        [int(ymd_to_ordinal(2025 + expiry_y + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(2025 + expiry_y, 1, 1))),
        fixed_dates=fixed,
        strike=jnp.asarray(strike),
        notional=jnp.array(1.0),
        is_payer=True,
        day_count=_ACT365,
    )


def _hw_price(sigma: jax.Array, swaption: Swaption) -> jax.Array:
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(0.05), volatility=sigma, initial_curve=_CURVE
    )
    return hw_swaption_price(swaption, model)


def _g2_price(sigma: jax.Array, swaption: Swaption) -> jax.Array:
    model = G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=sigma,
        volatility_ratio=jnp.array(0.8),
        correlation=jnp.array(-0.70),
        initial_curve=_CURVE,
    )
    return g2pp_swaption_price(swaption, model)


_PRICERS = pytest.mark.parametrize(
    "pricer", [_hw_price, _g2_price], ids=["hull_white", "g2pp"]
)


@_PRICERS
@pytest.mark.parametrize("bad_sigma", _BAD_SIGMAS)
def test_bad_sigma_is_nan_price_and_nan_gradient_under_jit(pricer, bad_sigma):
    swaption = _swaption()
    value_and_grad = eqx.filter_jit(jax.value_and_grad(pricer))
    value, grad = value_and_grad(jnp.asarray(bad_sigma), swaption)
    assert jnp.isnan(value)
    assert jnp.isnan(grad)
    # Same compiled function, healthy input: finite price and gradient.
    value, grad = value_and_grad(jnp.asarray(_HEALTHY_SIGMA), swaption)
    assert jnp.isfinite(value) and value > 0.0
    assert jnp.isfinite(grad) and grad > 0.0


@_PRICERS
def test_bad_trade_does_not_poison_a_vmapped_batch(pricer):
    swaption = _swaption()
    sigmas = jnp.array([_HEALTHY_SIGMA, *_BAD_SIGMAS, 2.0 * _HEALTHY_SIGMA])
    batched = eqx.filter_jit(
        jax.vmap(jax.value_and_grad(pricer), in_axes=(0, None))
    )
    values, grads = batched(sigmas, swaption)
    healthy = jnp.array([True] + [False] * len(_BAD_SIGMAS) + [True])
    assert bool(jnp.all(jnp.isnan(values[~healthy])))
    assert bool(jnp.all(jnp.isnan(grads[~healthy])))
    assert bool(jnp.all(jnp.isfinite(values[healthy])))
    assert bool(jnp.all(jnp.isfinite(grads[healthy])))
    # The healthy entries equal their own unbatched prices and gradients.
    single = eqx.filter_jit(jax.value_and_grad(pricer))
    for i in (0, len(sigmas) - 1):
        v, g = single(sigmas[i], swaption)
        assert float(values[i]) == pytest.approx(float(v), rel=1e-14, abs=0.0)
        assert float(grads[i]) == pytest.approx(float(g), rel=1e-12, abs=0.0)


@_PRICERS
def test_healthy_batch_matches_an_all_healthy_batch_exactly(pricer):
    """A bad neighbour changes nothing for the healthy entries of a batch."""
    swaption = _swaption()
    batched = eqx.filter_jit(
        jax.vmap(jax.value_and_grad(pricer), in_axes=(0, None))
    )
    clean = batched(jnp.array([_HEALTHY_SIGMA, 0.02]), swaption)
    mixed = batched(jnp.array([_HEALTHY_SIGMA, jnp.nan]), swaption)
    assert float(mixed[0][0]) == float(clean[0][0])
    assert float(mixed[1][0]) == float(clean[1][0])
    assert jnp.isnan(mixed[0][1]) and jnp.isnan(mixed[1][1])


def test_hw_critical_rate_failure_path_is_taken():
    """Direct check on the guarded solve: NaN on failure, a root otherwise."""
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(0.05),
        volatility=jnp.asarray(_HEALTHY_SIGMA),
        initial_curve=_CURVE,
    )
    times = jnp.array([3.0, 4.0, 5.0])
    flows = jnp.array([0.03, 0.03, 1.03])
    root = hw_critical_rate(model, jnp.asarray(2.0), times, flows)
    assert jnp.isfinite(root)
    # Non-finite cash flow: the probe fails and the result is NaN, without a
    # raise from the implicit derivative. The failed root's own gradient is
    # masked (finite, zero): the failure is signalled by the NaN *value*,
    # which makes every price built on it — and that price's gradient — NaN
    # (see the pricer-level tests above).
    bad = flows.at[0].set(jnp.nan)

    def f(c):
        return hw_critical_rate(model, jnp.asarray(2.0), times, c)

    assert jnp.isnan(f(bad))
    assert bool(jnp.all(jax.grad(f)(bad) == 0.0))
    # Healthy flows: a genuine, nonzero sensitivity.
    assert bool(jnp.all(jnp.isfinite(jax.grad(f)(flows))))
    assert bool(jnp.any(jax.grad(f)(flows) != 0.0))
