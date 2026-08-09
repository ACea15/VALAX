"""Benchmarks for the PDE second-order spot Greek (gamma).

Run with ``pytest --benchmark-only``. These exercise the read-off curvature
fix end-to-end: gamma is obtained through the *unified* autodiff Greek engine
(``jax.grad(jax.grad(pde_price))``), timed on a JIT-compiled call, and checked
for convergence against the Black-Scholes closed form as the spatial grid is
refined.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.greeks.autodiff import greek
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.solvers import pde_price, PDEConfig


_OPTION = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
_ARGS = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
_BS_GAMMA = float(greek(black_scholes_price, "gamma", _OPTION, *_ARGS))


def _jitted_gamma(n_spot: int, n_time: int):
    cfg = PDEConfig(n_spot=n_spot, n_time=n_time)

    @eqx.filter_jit
    def gamma(option, spot, vol, rate, div):
        return greek(lambda o, *a: pde_price(o, *a, cfg), "gamma", option, spot, vol, rate, div)

    # Warm up (compile) once outside the timed region.
    _ = gamma(_OPTION, *_ARGS)
    return gamma


@pytest.mark.parametrize("n_spot,n_time", [(100, 100), (200, 200), (400, 400)])
def test_gamma_throughput(benchmark, n_spot, n_time):
    """Time a compiled gamma evaluation and record its accuracy vs BS."""
    gamma = _jitted_gamma(n_spot, n_time)

    result = benchmark(lambda: float(gamma(_OPTION, *_ARGS)))

    err = abs(result - _BS_GAMMA)
    benchmark.extra_info["bs_gamma"] = _BS_GAMMA
    benchmark.extra_info["pde_gamma"] = result
    benchmark.extra_info["abs_error"] = err
    # Sanity: gamma is correct (and non-zero, unlike the old linear read-off).
    assert result > 1e-3
    assert err < 3.0e-4 + 0.02 * abs(_BS_GAMMA)


def test_gamma_convergence_profile(benchmark):
    """Benchmark that also asserts gamma error decreases as the grid refines."""

    def run():
        errs = {}
        for n in (50, 100, 200, 400):
            cfg = PDEConfig(n_spot=n, n_time=n)
            g = float(greek(lambda o, *a: pde_price(o, *a, cfg), "gamma", _OPTION, *_ARGS))
            errs[n] = abs(g - _BS_GAMMA)
        return errs

    errs = benchmark(run)
    benchmark.extra_info["errors_by_n"] = {str(k): v for k, v in errs.items()}

    # Monotone-ish improvement: the finest grid is the most accurate, and it
    # beats the coarsest by a clear margin.
    assert errs[400] <= errs[50]
    assert errs[400] < 1e-4
