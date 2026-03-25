"""Crank-Nicolson finite difference solver for the Black-Scholes PDE.

The BS PDE in log-spot space x = ln(S):

    dV/dt + (r - q - sigma^2/2) dV/dx + (sigma^2/2) d^2V/dx^2 - rV = 0

We step backward in time from the terminal payoff using Crank-Nicolson
(theta-scheme with theta=0.5) for unconditional stability and second-order
accuracy in both time and space.

The tridiagonal systems are solved via lineax for full JAX traceability.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


class PDEConfig(eqx.Module):
    """Finite difference grid configuration."""

    n_spot: int = eqx.field(static=True, default=200)    # spatial grid points
    n_time: int = eqx.field(static=True, default=200)    # time steps
    spot_range: float = eqx.field(static=True, default=4.0)  # log-spot range in std devs


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
    T = option.expiry
    K = option.strike
    N = config.n_spot  # spatial points (interior)
    M = config.n_time  # time steps

    # Log-spot grid centered at ln(spot)
    x_center = jnp.log(spot)
    x_width = config.spot_range * vol * jnp.sqrt(T)
    x_min = x_center - x_width
    x_max = x_center + x_width
    dx = (x_max - x_min) / (N + 1)
    dt = T / M

    # Interior grid points (exclude boundaries)
    x = x_min + dx * jnp.arange(1, N + 1)  # shape (N,)

    # Coefficients in log-spot space
    mu = rate - dividend - 0.5 * vol**2  # drift in log-space
    sigma2 = vol**2

    # Tridiagonal matrix coefficients
    alpha = dt * (sigma2 / (2 * dx**2) - mu / (2 * dx))  # lower diagonal
    beta = dt * (-sigma2 / dx**2 - rate)                   # main diagonal
    gamma = dt * (sigma2 / (2 * dx**2) + mu / (2 * dx))  # upper diagonal

    # Terminal payoff
    S_grid = jnp.exp(x)
    if option.is_call:
        V = jnp.maximum(S_grid - K, 0.0)
    else:
        V = jnp.maximum(K - S_grid, 0.0)

    # Boundary conditions (functions of time remaining tau)
    def boundary_lower(tau):
        """Value at x_min (S very small)."""
        S_lo = jnp.exp(x_min)
        if option.is_call:
            return jnp.array(0.0)
        else:
            return K * jnp.exp(-rate * tau) - S_lo * jnp.exp(-dividend * tau)

    def boundary_upper(tau):
        """Value at x_max (S very large)."""
        S_hi = jnp.exp(x_max)
        if option.is_call:
            return S_hi * jnp.exp(-dividend * tau) - K * jnp.exp(-rate * tau)
        else:
            return jnp.array(0.0)

    # Crank-Nicolson: theta = 0.5
    # (I - 0.5*A) V^{n} = (I + 0.5*A) V^{n+1} + boundary terms
    # where A is the tridiagonal operator

    # LHS matrix diagonals: I - 0.5*A
    lhs_lower = jnp.full(N - 1, -0.5 * alpha)
    lhs_diag = jnp.full(N, 1.0 - 0.5 * beta)
    lhs_upper = jnp.full(N - 1, -0.5 * gamma)

    # RHS matrix diagonals: I + 0.5*A
    rhs_lower = 0.5 * alpha
    rhs_diag = 1.0 + 0.5 * beta
    rhs_upper = 0.5 * gamma

    lhs_op = lx.TridiagonalLinearOperator(lhs_diag, lhs_lower, lhs_upper)
    solver = lx.Tridiagonal()

    def rhs_matvec(v):
        """Multiply (I + 0.5*A) @ v using tridiagonal structure."""
        result = rhs_diag * v
        result = result.at[1:].add(rhs_lower * v[:-1])
        result = result.at[:-1].add(rhs_upper * v[1:])
        return result

    def step(V, m):
        """One backward time step from m+1 to m."""
        tau_new = (M - m) * dt  # time remaining after this step
        tau_old = (M - m - 1) * dt

        rhs = rhs_matvec(V)

        # Add boundary contributions
        bc_lo_new = boundary_lower(tau_new)
        bc_lo_old = boundary_lower(tau_old)
        bc_hi_new = boundary_upper(tau_new)
        bc_hi_old = boundary_upper(tau_old)

        rhs = rhs.at[0].add(0.5 * alpha * bc_lo_old + 0.5 * alpha * bc_lo_new)
        rhs = rhs.at[-1].add(0.5 * gamma * bc_hi_old + 0.5 * gamma * bc_hi_new)

        sol = lx.linear_solve(lhs_op, rhs, solver=solver)
        return sol.value, None

    V, _ = jax.lax.scan(step, V, jnp.arange(M))

    # Interpolate to get price at exact spot
    log_spot = jnp.log(spot)
    return jnp.interp(log_spot, x, V)
