"""Barrier PDE (absorbing boundary) structural invariants.

PR-1 validates the continuously-monitored knock-out/knock-in via robust
structural relations (no analytic barrier pricer exists in VALAX yet); a
closed-form Reiner-Rubinstein cross-check is deferred to a later phase.
"""

import equinox as eqx
import jax.numpy as jnp

from valax.greeks.autodiff import greek
from valax.instruments.options import EquityBarrierOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.instruments.options import EuropeanOption
from valax.pricing.pde import PDEConfig, pde_price_dispatch

CFG = PDEConfig(n_spot=400, n_time=400)


def _model(vol=0.2, rate=0.05, dividend=0.0):
    return BlackScholesModel(
        vol=jnp.array(vol), rate=jnp.array(rate), dividend=jnp.array(dividend)
    )


def _barrier_price(instrument, spot, vol, rate, dividend):
    """Scalar pricing closure for the autodiff Greek engine."""
    model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
    return pde_price_dispatch(instrument, model, CFG, spot=spot).price


def _vanilla(S, K, is_call):
    opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(1.0), is_call=is_call)
    m = _model()
    return float(black_scholes_price(opt, jnp.array(S), m.vol, m.rate, m.dividend))


def test_knockout_between_zero_and_vanilla():
    ko = EquityBarrierOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), barrier=jnp.array(130.0),
        is_call=True, is_up=True, is_knock_in=False,
    )
    price = float(pde_price_dispatch(ko, _model(), CFG, spot=jnp.array(100.0)))
    vanilla = _vanilla(100.0, 100.0, True)
    assert 0.0 < price < vanilla


def test_in_out_parity():
    common = dict(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), barrier=jnp.array(130.0),
        is_call=True, is_up=True,
    )
    ko = EquityBarrierOption(is_knock_in=False, **common)
    ki = EquityBarrierOption(is_knock_in=True, **common)
    model = _model()
    ko_p = float(pde_price_dispatch(ko, model, CFG, spot=jnp.array(100.0)))
    ki_p = float(pde_price_dispatch(ki, model, CFG, spot=jnp.array(100.0)))
    vanilla = _vanilla(100.0, 100.0, True)
    assert abs((ko_p + ki_p) - vanilla) < 1e-3


def test_far_barrier_approaches_vanilla():
    vanilla = _vanilla(100.0, 100.0, True)
    near = EquityBarrierOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), barrier=jnp.array(115.0),
        is_call=True, is_up=True, is_knock_in=False,
    )
    far = EquityBarrierOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), barrier=jnp.array(200.0),
        is_call=True, is_up=True, is_knock_in=False,
    )
    model = _model()
    near_p = float(pde_price_dispatch(near, model, CFG, spot=jnp.array(100.0)))
    far_p = float(pde_price_dispatch(far, model, CFG, spot=jnp.array(100.0)))
    # A more distant up-and-out barrier is knocked out less often -> worth more,
    # and approaches the vanilla value.
    assert near_p < far_p < vanilla
    assert (vanilla - far_p) < (vanilla - near_p)


def test_filter_jit_smoke():
    ko = EquityBarrierOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), barrier=jnp.array(130.0),
        is_call=True, is_up=True, is_knock_in=False,
    )
    cfg = PDEConfig(n_spot=120, n_time=120)

    @eqx.filter_jit
    def priced(spot, vol, rate, dividend):
        model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
        return pde_price_dispatch(ko, model, cfg, spot=spot).price

    val = priced(jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
    assert float(val) > 0.0


# ── Second-order spot Greek (gamma) for barriers ─────────────────────
#
# Barriers build their mesh with ``uniform_linear_grid`` (one edge pinned to
# the barrier), so the spot-detachment lives in ``_barrier_bs`` itself: the
# grid is placed on ``stop_gradient(spot)`` while the read-off query stays live.
# These guard that barrier gamma flows correctly through the unified autodiff
# Greek engine (it was silently ~garbage before the fix).

class TestBarrierGamma:
    _ARGS = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))

    def _ko(self, **kw):
        base = dict(
            strike=jnp.array(100.0), expiry=jnp.array(1.0),
            is_call=True, is_up=True, is_knock_in=False,
        )
        base.update(kw)
        return EquityBarrierOption(**base)

    def test_gamma_is_finite_and_nonzero(self):
        ko = self._ko(barrier=jnp.array(130.0))
        gamma = float(greek(_barrier_price, "gamma", ko, *self._ARGS))
        assert jnp.isfinite(jnp.array(gamma))
        assert abs(gamma) > 1e-4

    def test_up_out_call_gamma_negative_below_barrier(self):
        """Up-and-out call value is concave near the barrier -> gamma < 0."""
        ko = self._ko(barrier=jnp.array(130.0))
        gamma = float(greek(_barrier_price, "gamma", ko, *self._ARGS))
        assert gamma < 0.0

    def test_in_out_gamma_parity(self):
        """Gamma is additive across the in/out decomposition (KO + KI = vanilla)."""
        common = dict(
            strike=jnp.array(100.0), expiry=jnp.array(1.0),
            barrier=jnp.array(130.0), is_call=True, is_up=True,
        )
        ko = EquityBarrierOption(is_knock_in=False, **common)
        ki = EquityBarrierOption(is_knock_in=True, **common)
        vanilla = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

        g_ko = float(greek(_barrier_price, "gamma", ko, *self._ARGS))
        g_ki = float(greek(_barrier_price, "gamma", ki, *self._ARGS))
        g_van = float(greek(black_scholes_price, "gamma", vanilla, *self._ARGS))
        assert abs((g_ko + g_ki) - g_van) < 5e-4

    def test_gamma_under_filter_jit(self):
        ko = self._ko(barrier=jnp.array(130.0))

        @eqx.filter_jit
        def gamma(instrument, spot, vol, rate, div):
            return greek(_barrier_price, "gamma", instrument, spot, vol, rate, div)

        g = float(gamma(ko, *self._ARGS))
        g_eager = float(greek(_barrier_price, "gamma", ko, *self._ARGS))
        assert abs(g - g_eager) < 1e-6
