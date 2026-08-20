"""Built-in PDE recipes, registered on the dispatcher at import time.

PR-1 covers single-asset equity instruments under
:class:`~valax.models.black_scholes.BlackScholesModel`:

- ``EuropeanOption`` — Crank-Nicolson.
- ``AmericanOption`` — CN + penalty-method free boundary.
- ``DigitalOption`` — CN + Rannacher start-up (damps the payoff discontinuity).
- ``EquityBarrierOption`` — CN with an absorbing barrier boundary; knock-ins via
  in/out parity.

Local volatility (Dupire) adds a *time-dependent* operator recipe:

- ``EuropeanOption`` under ``LocalVolModel`` — CN with a per-step operator stack
  (:func:`~valax.pricing.pde.coefficients.lv_operator_stack`).

The Heston stochastic-volatility model adds a 2-D (ADI) recipe routed through
:class:`~valax.pricing.pde.config.PDEConfig2D`:

- ``EuropeanOption`` under ``HestonModel`` — Douglas / Craig-Sneyd / HV ADI on a
  log-spot :math:`\\times` variance grid
  (:func:`~valax.pricing.pde.schemes2d.solve_backward_2d`).

Accuracy note (FD Dupire vs the implied surface)
------------------------------------------------
Feeding the *continuous* Dupire local volatility into a *discrete* backward
finite-difference scheme does **not** reprice the input vanilla surface exactly
when the smile is skewed. The continuous Dupire formula is the inverse of the
continuous forward (Fokker-Planck) equation, which is **not** the adjoint of the
discrete backward operator, so a skew-dependent, grid-*independent* repricing
gap remains even in the mesh limit (it does not shrink as ``O(dx^2)``). This is
a well-known property of FD local-vol engines — QuantLib's
``FdBlackScholesVanillaEngine`` exhibits the same gap (empirically of the same
magnitude). Consequences worth knowing:

- **Flat and pure-term-structure (no-skew) surfaces**: the local vol is constant
  in log-spot, the scheme is exact, and the LV PDE reprices the surface (and
  agrees with Monte-Carlo) to grid tolerance.
- **Skewed surfaces**: the LV PDE is self-consistent and converges, agrees with
  an independent FD reference (and QuantLib) to grid tolerance, but reprices the
  vanilla surface / matches LV Monte-Carlo only near ATM; the gap grows into the
  wings with the skew. Monte-Carlo, which samples the true continuous SDE, is
  the faithful surface-repricer there. Closing the FD gap requires calibrating
  the local vol to the *discrete* forward operator (Andreasen-Huge) — see the
  research-ideas backlog rather than this recipe.
"""

import jax.numpy as jnp
from jax import lax

from valax.instruments.options import (
    AmericanOption,
    DigitalOption,
    EquityBarrierOption,
    EuropeanOption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.local_vol import LocalVolModel
from valax.pricing.pde.boundary import (
    american_boundary,
    apply_heston_variance_bc,
    bs_european_boundary,
    digital_boundary,
    heston_boundary,
    knockout_boundary,
)
from valax.pricing.pde.coefficients import (
    bs_operator,
    heston_operator_2d,
    lv_operator_stack,
)
from valax.pricing.pde.dispatch import PDEResult, register
from valax.pricing.pde.exercise import penalty_solver
from valax.pricing.pde.grids import (
    log_spot_variance_grid,
    read_off_1d,
    read_off_2d,
    uniform_linear_grid,
    uniform_log_spot_grid,
)
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme
from valax.pricing.pde.schemes2d import solve_backward_2d
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


def _lv_reference_vol(model, spot, expiry):
    """ATM-forward implied vol used only to scale the grid half-width.

    The grid is a numerical scaffold, so the width reference is detached from
    autodiff (spot and the surface query) — its sole job is to size a mesh wide
    enough to contain the relevant spot range. Uses the surface's own implied
    vol at the forward ``F(T) = S_0 exp((r - q) T)``.
    """
    mu = model.rate - model.dividend
    forward = lax.stop_gradient(spot) * jnp.exp(mu * expiry)
    return lax.stop_gradient(model.surface(forward, expiry))


@register(EuropeanOption, LocalVolModel)
def _european_lv(*, instrument, model, config, spot):
    ref_vol = _lv_reference_vol(model, spot, instrument.expiry)
    grid = uniform_log_spot_grid(
        spot, ref_vol, instrument.expiry, n=config.n_spot, half_width=config.spot_range
    )
    operator = lv_operator_stack(
        model, grid, spot, expiry=instrument.expiry, n_time=config.n_time
    )
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
    return PDEResult(price=read_off_1d(grid, values, jnp.log(spot)))


@register(EuropeanOption, HestonModel)
def _european_heston(*, instrument, model, config, spot):
    """European option under Heston via an ADI finite-difference solve.

    ``config`` is a :class:`~valax.pricing.pde.config.PDEConfig2D` (its
    ``scheme`` must be one of the ADI schemes). Grid *placement* is a numerical
    scaffold, so the spot and the variance centre / reference are detached from
    autodiff; the differentiable dependence lives entirely in the read-off
    queries — ``ln(spot)`` (delta / gamma) and the variance level ``v0``
    (v0-vega) — while the model parameters ``kappa, theta, xi, rho`` stay live
    through the operator coefficients.
    """
    # Detached references used only to size / place the mesh.
    v0_ref = lax.stop_gradient(model.v0)
    theta_ref = lax.stop_gradient(model.theta)
    vol_ref = jnp.sqrt(jnp.maximum(v0_ref, theta_ref))

    grid = log_spot_variance_grid(
        spot,
        instrument.expiry,
        v0_ref,
        config.y_max,
        n_x=config.n_x,
        n_y=config.n_y,
        x_half_width=config.x_range,
        v_scale=config.y_scale,
        vol_ref=vol_ref,
    )
    operator = apply_heston_variance_bc(
        heston_operator_2d(model, grid), grid, model
    )
    boundary = heston_boundary(
        grid, instrument.strike, model.rate, model.dividend, instrument.is_call
    )
    payoff = vanilla_terminal(grid.x, instrument.strike, instrument.is_call)
    terminal = jnp.broadcast_to(payoff[:, jnp.newaxis], grid.shape)
    values = solve_backward_2d(
        operator,
        boundary,
        terminal,
        expiry=instrument.expiry,
        n_time=config.n_time,
        scheme=config.scheme,
        theta=config.theta,
        rannacher_steps=config.rannacher_steps,
    )
    price = read_off_2d(grid, values, jnp.log(spot), model.v0)
    return PDEResult(price=price)


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
    # Grid *placement* is a numerical scaffold: detach ``spot`` so the mesh does
    # not co-move with it under autodiff (the requirement for a correct
    # second-order spot Greek, see ``uniform_log_spot_grid``). The read-off
    # query ``x_query`` below is kept live/differentiable; ``half`` stays live in
    # vol/expiry so vega/theta keep the grid-width sensitivity.
    x_center = jnp.log(lax.stop_gradient(spot))
    x_query = jnp.log(spot)
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
    knockout = read_off_1d(grid, values, x_query)

    if instrument.is_knock_in:
        vanilla = _european_value(instrument, model, config, spot)
        return PDEResult(price=vanilla - knockout)
    return PDEResult(price=knockout)
