"""Unit tests for the 2-D ADI stepper (:func:`solve_backward_2d`).

The correctness anchor is a *frozen-variance* Heston (``xi = 0``, ``kappa = 0``):
the variance dynamics vanish, so the 2-D PDE decouples across variance into
independent 1-D Black-Scholes problems with ``sigma^2 = v``. Reading the 2-D
solution at ``(ln spot, v0)`` must therefore reproduce the analytic BS price
with ``sigma = sqrt(v0)`` -- for every ADI scheme, and with the error shrinking
under refinement.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.boundary import apply_heston_variance_bc, heston_boundary
from valax.pricing.pde.coefficients import heston_operator_2d
from valax.pricing.pde.config import Scheme
from valax.pricing.pde.grids import log_spot_variance_grid, read_off_2d
from valax.pricing.pde.schemes2d import solve_backward_2d

SPOT, STRIKE, T, RATE, DIV = 100.0, 100.0, 1.0, 0.03, 0.01
V0 = 0.04  # sigma = 0.2


def _frozen_model():
    """Heston with no variance dynamics -> decouples to Black-Scholes(sqrt(v))."""
    return HestonModel(
        v0=jnp.array(V0),
        kappa=jnp.array(0.0),
        theta=jnp.array(V0),
        xi=jnp.array(0.0),
        rho=jnp.array(0.0),
        rate=jnp.array(RATE),
        dividend=jnp.array(DIV),
    )


def _price_frozen(scheme, n_x, n_y, n_time, is_call=True):
    m = _frozen_model()
    grid = log_spot_variance_grid(
        jnp.array(SPOT),
        jnp.array(T),
        jnp.array(V0),
        jnp.array(0.25),
        n_x=n_x,
        n_y=n_y,
        x_half_width=4.0,
        v_scale=0.02,
    )
    op = apply_heston_variance_bc(heston_operator_2d(m, grid), grid, m)
    bnd = heston_boundary(grid, jnp.array(STRIKE), m.rate, m.dividend, is_call)
    s = jnp.exp(grid.x.nodes)
    payoff = jnp.maximum(s - STRIKE, 0.0) if is_call else jnp.maximum(STRIKE - s, 0.0)
    terminal = jnp.broadcast_to(payoff[:, None], grid.shape)
    values = solve_backward_2d(
        op,
        bnd,
        terminal,
        expiry=jnp.array(T),
        n_time=n_time,
        scheme=scheme,
        theta=0.5,
        rannacher_steps=2,
    )
    return read_off_2d(grid, values, jnp.log(jnp.array(SPOT)), jnp.array(V0))


def _bs(is_call=True):
    opt = EuropeanOption(
        strike=jnp.array(STRIKE), expiry=jnp.array(T), is_call=is_call
    )
    return float(
        black_scholes_price(
            opt, jnp.array(SPOT), jnp.array(V0**0.5), jnp.array(RATE), jnp.array(DIV)
        )
    )


@pytest.mark.parametrize(
    "scheme", [Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV]
)
def test_frozen_variance_matches_black_scholes(scheme):
    price = float(_price_frozen(scheme, n_x=240, n_y=48, n_time=120))
    bs = _bs()
    assert abs(price - bs) / bs < 5e-3, f"{scheme}: pde={price:.5f} bs={bs:.5f}"


def test_frozen_variance_put_matches_black_scholes():
    price = float(_price_frozen(Scheme.CRAIG_SNEYD, 240, 48, 120, is_call=False))
    bs = _bs(is_call=False)
    assert abs(price - bs) / bs < 5e-3


def test_schemes_agree_on_frozen_variance():
    dg = float(_price_frozen(Scheme.DOUGLAS, 200, 40, 100))
    cs = float(_price_frozen(Scheme.CRAIG_SNEYD, 200, 40, 100))
    hv = float(_price_frozen(Scheme.HV, 200, 40, 100))
    assert abs(dg - cs) < 5e-3
    assert abs(dg - hv) < 5e-3


def test_refinement_reduces_error():
    bs = _bs()
    coarse = abs(float(_price_frozen(Scheme.DOUGLAS, 120, 24, 60)) - bs)
    fine = abs(float(_price_frozen(Scheme.DOUGLAS, 300, 60, 150)) - bs)
    assert fine < coarse + 1e-4


def test_rejects_non_adi_scheme():
    with pytest.raises(ValueError, match="ADI"):
        _price_frozen(Scheme.CRANK_NICOLSON, 60, 20, 30)


def test_full_heston_feller_violation_is_finite():
    """A Feller-violating full-Heston run must stay finite (no NaN/Inf)."""
    m = HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(1.0),
        theta=jnp.array(0.04),
        xi=jnp.array(0.9),  # 2*kappa*theta=0.08 < xi^2=0.81 -> Feller violated
        rho=jnp.array(-0.7),
        rate=jnp.array(RATE),
        dividend=jnp.array(DIV),
    )
    grid = log_spot_variance_grid(
        jnp.array(SPOT), jnp.array(T), jnp.array(0.04), jnp.array(0.5),
        n_x=120, n_y=48, x_half_width=4.0, v_scale=0.03,
    )
    op = apply_heston_variance_bc(heston_operator_2d(m, grid), grid, m)
    bnd = heston_boundary(grid, jnp.array(STRIKE), m.rate, m.dividend, True)
    s = jnp.exp(grid.x.nodes)
    terminal = jnp.broadcast_to(jnp.maximum(s - STRIKE, 0.0)[:, None], grid.shape)
    values = solve_backward_2d(
        op, bnd, terminal, expiry=jnp.array(T), n_time=100,
        scheme=Scheme.HV, theta=0.5, rannacher_steps=2,
    )
    assert bool(jnp.all(jnp.isfinite(values)))
    price = float(read_off_2d(grid, values, jnp.log(jnp.array(SPOT)), jnp.array(0.04)))
    assert jnp.isfinite(price) and price > 0.0


def test_filter_jit_smoke():
    """The whole solve is jit-compilable end to end."""
    price = eqx.filter_jit(_price_frozen)(Scheme.DOUGLAS, 100, 24, 50)
    assert bool(jnp.isfinite(price))
