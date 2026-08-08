"""High-level PDE drivers and the backward-compatible ``pde_price`` facade.

``pde_price`` retains its original signature and numerical behaviour (1-D
Crank-Nicolson in log-spot space) but is now a thin façade over the layered
core (:mod:`grids`, :mod:`operators`, :mod:`boundary`, :mod:`terminal`,
:mod:`schemes`). The multi-instrument entry point is
:func:`~valax.pricing.pde.dispatch.pde_price_dispatch`.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.pde.boundary import bs_european_boundary
from valax.pricing.pde.coefficients import bs_operator
from valax.pricing.pde.config import PDEConfig
from valax.pricing.pde.grids import read_off_1d, uniform_log_spot_grid
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme
from valax.pricing.pde.terminal import vanilla_terminal


def pde_price(
    option: EuropeanOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    config: PDEConfig = PDEConfig(),
) -> Float[Array, ""]:
    """Price a European option via Crank-Nicolson finite differences.

    Args:
        option: European option contract.
        spot: Current spot price.
        vol: Volatility.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        config: FD grid configuration.

    Returns:
        Option price.
    """
    model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
    grid = uniform_log_spot_grid(
        spot, vol, option.expiry, n=config.n_spot, half_width=config.spot_range
    )
    operator = bs_operator(model, grid)
    boundary = bs_european_boundary(
        grid, option.strike, rate, dividend, option.is_call
    )
    terminal = vanilla_terminal(grid, option.strike, option.is_call)
    values = solve_backward_1d(
        operator,
        boundary,
        terminal,
        expiry=option.expiry,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=config.rannacher_steps,
    )
    return read_off_1d(grid, values, jnp.log(spot))
