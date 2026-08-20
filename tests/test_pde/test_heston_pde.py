"""Validation of the Heston ADI PDE pricer against independent references.

The 2-D ADI finite-difference price is checked against:

* ``heston_cos_price`` -- the semi-analytic Fang-Oosterlee COS oracle (effectively
  exact), across a strip of liquid strikes and for a Feller-violating parameter
  set;
* the Andersen-QE Monte-Carlo price (within a few standard errors);
* the Black-Scholes closed form in the ``xi -> 0`` (frozen-variance) limit;

plus internal consistency (Douglas / Craig-Sneyd / HV agreement), grid
convergence, autodiff Greeks and an ``eqx.filter_jit`` smoke test.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.heston import heston_cos_price
from valax.pricing.mc import MCConfig, mc_price_dispatch
from valax.pricing.pde import PDEConfig2D, pde_price_dispatch
from valax.pricing.pde.config import Scheme

SPOT, T, RATE, DIV = 100.0, 1.0, 0.03, 0.0


def _model(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04):
    return HestonModel(
        v0=jnp.array(v0),
        kappa=jnp.array(kappa),
        theta=jnp.array(theta),
        xi=jnp.array(xi),
        rho=jnp.array(rho),
        rate=jnp.array(RATE),
        dividend=jnp.array(DIV),
    )


def _cfg(n_x=300, n_y=120, n_time=150, y_max=0.7, y_scale=0.03, scheme=Scheme.HV):
    return PDEConfig2D(
        n_x=n_x,
        n_y=n_y,
        n_time=n_time,
        x_range=5.0,
        y_max=y_max,
        y_scale=y_scale,
        scheme=scheme,
    )


def _opt(strike, is_call=True):
    return EuropeanOption(
        strike=jnp.array(strike), expiry=jnp.array(T), is_call=is_call
    )


def _pde(opt, model, cfg):
    return float(pde_price_dispatch(opt, model, cfg, spot=jnp.array(SPOT)))


def _cos(opt, model):
    return float(
        heston_cos_price(opt, jnp.array(SPOT), model.rate, model.dividend, model)
    )


# ── Accuracy vs the COS oracle ───────────────────────────────────────


@pytest.mark.parametrize("strike", [90.0, 95.0, 100.0, 105.0, 110.0])
def test_matches_cos_across_strikes(strike):
    m, cfg = _model(), _cfg()
    opt = _opt(strike)
    pde, cos = _pde(opt, m, cfg), _cos(opt, m)
    rel = abs(pde - cos) / cos
    assert rel < 1e-3, f"K={strike}: pde={pde:.5f} cos={cos:.5f} rel={rel:.2e}"


def test_put_matches_cos():
    m, cfg = _model(), _cfg()
    opt = _opt(100.0, is_call=False)
    pde, cos = _pde(opt, m, cfg), _cos(opt, m)
    assert abs(pde - cos) / cos < 2e-3


def test_feller_violating_matches_cos():
    # 2*kappa*theta = 0.08 < xi^2 = 0.25  ->  Feller condition violated.
    m = _model(kappa=1.0, theta=0.04, xi=0.5, rho=-0.6)
    assert 2.0 * float(m.kappa) * float(m.theta) < float(m.xi) ** 2
    cfg = _cfg(y_max=0.7, y_scale=0.03)
    opt = _opt(100.0)
    pde, cos = _pde(opt, m, cfg), _cos(opt, m)
    assert abs(pde - cos) / cos < 5e-3


# ── Cross-checks: Monte-Carlo and the Black-Scholes limit ────────────


def test_matches_andersen_qe_mc():
    m, cfg = _model(), _cfg()
    opt = _opt(100.0)
    pde = _pde(opt, m, cfg)
    res = mc_price_dispatch(
        opt, m, MCConfig(n_paths=200_000, n_steps=100), jax.random.PRNGKey(0),
        spot=jnp.array(SPOT),
    )
    nse = abs(pde - float(res.price)) / float(res.stderr)
    assert nse < 4.0, f"pde={pde:.4f} mc={float(res.price):.4f} nse={nse:.2f}"


def test_small_xi_collapses_to_black_scholes():
    # Strong mean reversion + tiny vol-of-vol pins variance at v0 -> BSM(sqrt(v0)).
    v0 = 0.04
    m = _model(kappa=8.0, theta=v0, xi=0.02, rho=0.0, v0=v0)
    cfg = _cfg(y_max=0.4, y_scale=0.02)
    opt = _opt(100.0)
    pde = _pde(opt, m, cfg)
    bs = float(
        black_scholes_price(
            opt, jnp.array(SPOT), jnp.sqrt(jnp.array(v0)),
            jnp.array(RATE), jnp.array(DIV),
        )
    )
    assert abs(pde - bs) / bs < 5e-3


# ── Internal consistency, convergence, Greeks ────────────────────────


def test_schemes_agree():
    m = _model()
    opt = _opt(100.0)
    prices = {
        s.name: _pde(opt, m, _cfg(scheme=s))
        for s in (Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV)
    }
    lo, hi = min(prices.values()), max(prices.values())
    assert hi - lo < 3e-3, prices


def test_refinement_reduces_error():
    m = _model()
    opt = _opt(100.0)
    cos = _cos(opt, m)
    coarse = abs(_pde(opt, m, _cfg(n_x=120, n_y=48, n_time=80)) - cos)
    fine = abs(_pde(opt, m, _cfg(n_x=320, n_y=128, n_time=200)) - cos)
    assert fine < coarse


def test_greeks_are_sane():
    m, cfg = _model(), _cfg(n_x=200, n_y=80, n_time=120)
    opt = _opt(100.0)

    def price(spot):
        return pde_price_dispatch(opt, m, cfg, spot=spot).price

    delta = float(jax.grad(price)(jnp.array(SPOT)))
    gamma = float(jax.grad(jax.grad(price))(jnp.array(SPOT)))
    assert 0.4 < delta < 0.75, f"delta={delta}"
    assert gamma > 1e-4, f"gamma={gamma}"


def test_v0_vega_is_positive():
    m, cfg = _model(), _cfg(n_x=200, n_y=80, n_time=120)
    opt = _opt(100.0)

    def price(v0):
        model = _model(v0=v0)
        # v0 is a traced array here; rebuild model with it live.
        model = eqx.tree_at(lambda mm: mm.v0, m, v0)
        return pde_price_dispatch(opt, model, cfg, spot=jnp.array(SPOT)).price

    dvega = float(jax.grad(price)(jnp.array(0.04)))
    assert dvega > 0.0, f"d(price)/d(v0)={dvega}"


def test_filter_jit_smoke():
    m, cfg = _model(), _cfg(n_x=160, n_y=64, n_time=100)
    opt = _opt(100.0)

    @eqx.filter_jit
    def priced(spot):
        return pde_price_dispatch(opt, m, cfg, spot=spot).price

    assert float(priced(jnp.array(SPOT))) > 0.0
