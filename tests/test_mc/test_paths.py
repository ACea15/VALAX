"""Path generators: differentiability of the Heston QE scheme.

Regression for the double-``where`` NaN: the QE scheme evaluates both of its
branches and clamps radicands to zero, and a bare ``sqrt`` there made every
pathwise Heston parameter gradient NaN even though prices were fine.
"""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.mc.engine import MCConfig, mc_price

OPTION = EuropeanOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0))
CONFIG = MCConfig(n_paths=4_000, n_steps=50)
KEY = jax.random.PRNGKey(0)
FIELDS = ("v0", "kappa", "theta", "xi", "rho")


def _model(**kw):
    base = dict(
        v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.6, rate=0.03, dividend=0.0
    )
    base.update(kw)
    return HestonModel(**{k: jnp.asarray(v) for k, v in base.items()})


def _price(model):
    return mc_price(OPTION, jnp.asarray(100.0), model, CONFIG, KEY)


def test_heston_parameter_gradients_are_finite():
    grad = jax.grad(_price)(_model())
    for name in FIELDS:
        assert bool(jnp.isfinite(getattr(grad, name))), name


@pytest.mark.parametrize(
    "name, rtol, atol",
    [("v0", 0.05, 0.0), ("theta", 0.05, 0.0), ("xi", 0.0, 0.5), ("rho", 0.0, 0.5)],
)
def test_heston_gradients_match_analytic(name, rtol, atol):
    """Pathwise gradient vs the Fourier (COS) gradient, within MC error.

    Same-path finite differences are *not* a valid oracle here: the QE
    exponential branch has an atom at zero whose probability moves with the
    parameters, so a bump flips paths across it and the FD picks up jumps the
    pathwise derivative (correctly, in expectation) does not.
    """
    from valax.pricing.analytic.heston import heston_cos_price

    config = MCConfig(n_paths=40_000, n_steps=50)

    def mc(model):
        return mc_price(OPTION, jnp.asarray(100.0), model, config, KEY)

    def cos(model):
        return heston_cos_price(OPTION, jnp.asarray(100.0), model=model)

    pathwise = float(getattr(jax.grad(mc)(_model()), name))
    exact = float(getattr(jax.grad(cos)(_model()), name))
    assert pathwise == pytest.approx(exact, rel=rtol, abs=atol)


def test_jit_smoke():
    assert bool(jnp.isfinite(jax.jit(_price)(_model())))
