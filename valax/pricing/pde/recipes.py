"""Built-in PDE recipes, registered on the dispatcher at import time.

PR-1 covers single-asset equity instruments under
:class:`~valax.models.black_scholes.BlackScholesModel`:

- ``EuropeanOption`` — Crank-Nicolson.
- ``AmericanOption`` — CN + penalty-method free boundary.
- ``DigitalOption`` — CN + Rannacher start-up (damps the payoff discontinuity).
- ``EquityBarrierOption`` — CN with an absorbing barrier boundary; knock-ins via
  in/out parity.
"""

import jax.numpy as jnp

from valax.instruments.options import (
    AmericanOption,
    DigitalOption,
    EquityBarrierOption,
    EuropeanOption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.pde.boundary import (
    american_boundary,
    bs_european_boundary,
    digital_boundary,
    knockout_boundary,
)
from valax.pricing.pde.coefficients import bs_operator
from valax.pricing.pde.dispatch import PDEResult, register
from valax.pricing.pde.exercise import penalty_solver
from valax.pricing.pde.grids import (
    read_off_1d,
    uniform_linear_grid,
    uniform_log_spot_grid,
)
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme
from valax.pricing.pde.terminal import (
    digital_terminal,
    intrinsic_payoff,
    vanilla_terminal,
)


def _european_value(instrument, model, config, spot):
    """Continuation (European) value read off at ``spot`` — shared helper."""
    grid = uniform_log_spot_grid(
        spot, model.vol, instrument.expiry, n=config.n_spot, half_width=config.spot_range
    )
    operator = bs_operator(model, grid)
    boundary = bs_european_boundary(
        grid, instrument.strike, model.rate, model.dividend, instrument.is_call
    )
    terminal = vanilla_terminal(grid, instrument.strike, instrument.is_call)
    values = solve_backward_1d(
        operator,
        boundary,
        terminal,
        expiry=instrument.expiry,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=config.rannacher_steps,
    )
    return read_off_1d(grid, values, jnp.log(spot))


@register(EuropeanOption, BlackScholesModel)
def _european_bs(*, instrument, model, config, spot):
    return PDEResult(price=_european_value(instrument, model, config, spot))


@register(AmericanOption, BlackScholesModel)
def _american_bs(*, instrument, model, config, spot):
    grid = uniform_log_spot_grid(
        spot, model.vol, instrument.expiry, n=config.n_spot, half_width=config.spot_range
    )
    operator = bs_operator(model, grid)
    boundary = american_boundary(grid, instrument.strike, instrument.is_call)
    payoff = intrinsic_payoff(grid, instrument.strike, instrument.is_call)
    solver = penalty_solver(payoff, config.penalty_rho, config.penalty_iters)
    values = solve_backward_1d(
        operator,
        boundary,
        payoff,  # terminal condition == intrinsic payoff
        expiry=instrument.expiry,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=config.rannacher_steps,
        solver_fn=solver,
    )
    return PDEResult(price=read_off_1d(grid, values, jnp.log(spot)))


@register(DigitalOption, BlackScholesModel)
def _digital_bs(*, instrument, model, config, spot):
    grid = uniform_log_spot_grid(
        spot, model.vol, instrument.expiry, n=config.n_spot, half_width=config.spot_range
    )
    operator = bs_operator(model, grid)
    boundary = digital_boundary(instrument.payout, model.rate, instrument.is_call)
    terminal = digital_terminal(
        grid, instrument.strike, instrument.payout, instrument.is_call
    )
    # Force at least 2 Rannacher steps to damp the step discontinuity.
    rannacher = max(config.rannacher_steps, 2)
    values = solve_backward_1d(
        operator,
        boundary,
        terminal,
        expiry=instrument.expiry,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=rannacher,
    )
    return PDEResult(price=read_off_1d(grid, values, jnp.log(spot)))


@register(EquityBarrierOption, BlackScholesModel)
def _barrier_bs(*, instrument, model, config, spot):
    half = config.spot_range * model.vol * jnp.sqrt(instrument.expiry)
    x_center = jnp.log(spot)
    x_barrier = jnp.log(instrument.barrier)

    if instrument.is_up:
        lo, hi = x_center - half, x_barrier
        barrier_is_upper = True
    else:
        lo, hi = x_barrier, x_center + half
        barrier_is_upper = False

    grid = uniform_linear_grid(lo, hi, n=config.n_spot)
    operator = bs_operator(model, grid)
    vanilla_bnd = bs_european_boundary(
        grid, instrument.strike, model.rate, model.dividend, instrument.is_call
    )
    boundary = knockout_boundary(vanilla_bnd, barrier_is_upper=barrier_is_upper)
    terminal = vanilla_terminal(grid, instrument.strike, instrument.is_call)
    values = solve_backward_1d(
        operator,
        boundary,
        terminal,
        expiry=instrument.expiry,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=max(config.rannacher_steps, 2),
    )
    knockout = read_off_1d(grid, values, x_center)

    if instrument.is_knock_in:
        vanilla = _european_value(instrument, model, config, spot)
        return PDEResult(price=vanilla - knockout)
    return PDEResult(price=knockout)
