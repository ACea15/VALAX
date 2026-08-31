"""ADI time-stepping for 2-D (stochastic-volatility) finite-difference solvers.

Implements the Douglas, Craig-Sneyd and Hundsdorfer-Verwer alternating-direction
implicit (ADI) schemes of in't Hout & Foulon (2010) for the operator split
``F = A0 + A1 + A2`` produced by :mod:`valax.pricing.pde.operators2d`. The
backward evolution ``dU/dtau = F U`` is marched from the terminal payoff via a
single :func:`jax.lax.scan`. Each implicit stage inverts one direction only --
``(I - theta dt A1)`` along log-spot and ``(I - theta dt A2)`` along variance --
as a batch of independent tridiagonal solves (:func:`~valax.pricing.pde.linalg.tridiagonal_solve`
under :func:`jax.vmap`), never a full 2-D solve. The mixed term ``A0`` is applied
only explicitly.

Boundary handling mirrors the 1-D solver: the log-spot Dirichlet data enters as
an affine term ``g1(tau)`` weighted ``(1-theta)`` at the known level and
``theta`` at the solved level, so with ``A0 = A2 = 0`` every scheme collapses to
exactly the 1-D theta-method of :func:`~valax.pricing.pde.schemes.solve_backward_1d`.
The variance-axis boundary conditions are already baked into ``A2`` by
:func:`~valax.pricing.pde.boundary.apply_heston_variance_bc`.

A Rannacher start-up (a few leading fully-implicit Douglas steps) damps the
non-smooth payoff before the second-order scheme engages.

References:
    K. J. in't Hout and S. Foulon, "ADI finite difference schemes for option
    pricing in the Heston model with correlation", Int. J. Numer. Anal. Model.
    7 (2010) 303-320.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.pricing.pde.boundary import Boundary2D
from valax.pricing.pde.config import Scheme
from valax.pricing.pde.linalg import tridiagonal_solve
from valax.pricing.pde.operators2d import Operator2D

# A discrete-event hook: given the forward time-level index just solved and the
# solved field, return a possibly-modified field. Mirrors the 1-D stepper's
# ``event_fn`` so short-rate discounting and Bermudan exercise projections plug
# in the same way. See :func:`solve_backward_2d`.
EventFn = Callable[[Int[Array, ""], Float[Array, "n_x n_y"]], Float[Array, "n_x n_y"]]


def _batch_tridiag_solve(
    lower: Float[Array, "n_x n_y"],
    diag: Float[Array, "n_x n_y"],
    upper: Float[Array, "n_x n_y"],
    rhs: Float[Array, "n_x n_y"],
    *,
    axis: int,
) -> Float[Array, "n_x n_y"]:
    """Solve a tridiagonal system along ``axis``, batching the other axis.

    Args:
        lower, diag, upper: Per-node bands of shape ``(n_x, n_y)``. Only the
            interior of ``lower`` / ``upper`` (``[1:]`` / ``[:-1]`` along
            ``axis``) participates in the solve.
        rhs: Right-hand side of shape ``(n_x, n_y)``.
        axis: 0 to solve along log-spot (batch over variance), 1 to solve along
            variance (batch over log-spot).

    Returns:
        The solution field of shape ``(n_x, n_y)``.
    """
    batch_axis = 1 - axis

    def solve_line(lo, di, up, r):
        return tridiagonal_solve(lo[1:], di, up[:-1], r)

    return jax.vmap(solve_line, in_axes=batch_axis, out_axes=batch_axis)(
        lower, diag, upper, rhs
    )


def _implicit_x(
    op: Operator2D, theta_dt: Float[Array, ""], rhs: Float[Array, "n_x n_y"]
) -> Float[Array, "n_x n_y"]:
    """Apply ``(I - theta dt A1)^{-1}`` (implicit solve along log-spot)."""
    return _batch_tridiag_solve(
        -theta_dt * op.a1_lower,
        1.0 - theta_dt * op.a1_diag,
        -theta_dt * op.a1_upper,
        rhs,
        axis=0,
    )


def _implicit_v(
    op: Operator2D, theta_dt: Float[Array, ""], rhs: Float[Array, "n_x n_y"]
) -> Float[Array, "n_x n_y"]:
    """Apply ``(I - theta dt A2)^{-1}`` (implicit solve along variance)."""
    return _batch_tridiag_solve(
        -theta_dt * op.a2_lower,
        1.0 - theta_dt * op.a2_diag,
        -theta_dt * op.a2_upper,
        rhs,
        axis=1,
    )


def _x_boundary_term(
    op: Operator2D,
    lower_value: Float[Array, ""],
    upper_value: Float[Array, ""],
) -> Float[Array, "n_x n_y"]:
    """Log-spot Dirichlet ghost contribution to ``A1`` (rows 0 and n_x-1).

    The exterior nodes carry known values, so their coupling into the first and
    last log-spot rows is an explicit source term, broadcast across variance.
    """
    g = jnp.zeros(op.a1_diag.shape)
    g = g.at[0, :].add(op.a1_lower[0, :] * lower_value)
    g = g.at[-1, :].add(op.a1_upper[-1, :] * upper_value)
    return g


def solve_backward_2d(
    operator: Operator2D,
    boundary: Boundary2D,
    terminal: Float[Array, "n_x n_y"],
    *,
    expiry: Float[Array, ""],
    n_time: int,
    scheme: Scheme,
    theta: float,
    rannacher_steps: int,
    event_fn: Optional[EventFn] = None,
) -> Float[Array, "n_x n_y"]:
    """Backward-march the terminal field to ``t = 0`` with an ADI scheme.

    Args:
        operator: The split 2-D operator (with variance boundary rows already
            baked into ``A2``).
        boundary: Log-spot Dirichlet data as functions of time-remaining.
        terminal: Terminal (payoff) field of shape ``(n_x, n_y)``.
        expiry: Time to expiry ``T`` (``dt = T / n_time``).
        n_time: Number of backward time steps.
        scheme: ADI scheme (:attr:`~valax.pricing.pde.config.Scheme.DOUGLAS`,
            :attr:`~valax.pricing.pde.config.Scheme.CRAIG_SNEYD` or
            :attr:`~valax.pricing.pde.config.Scheme.HV`).
        theta: Implicitness parameter after the Rannacher start-up.
        rannacher_steps: Number of leading fully-implicit (``theta = 1``,
            plain Douglas) steps that damp the non-smooth payoff.
        event_fn: Optional discrete-event hook applied *after* each solved step,
            called as ``event_fn(level, values)`` where ``level`` is the
            **forward** time-level index of the level just solved (``0`` at
            ``t = 0`` up to ``n_time`` at expiry, so it runs over
            ``n_time - 1 ... 0``). It is a traced scalar, so the hook must
            select with :func:`jax.numpy.where` / array indexing rather than
            Python control flow. Used by short-rate recipes to apply the
            deterministic (spatially-uniform) part of the discount per step and
            to project Bermudan exercise. Same convention as
            :func:`~valax.pricing.pde.schemes.solve_backward_1d`.

    Returns:
        The solution field of shape ``(n_x, n_y)`` at ``t = 0``.

    Raises:
        ValueError: If ``scheme`` is not one of the three ADI schemes.
    """
    if not scheme.is_adi():
        raise ValueError(
            f"solve_backward_2d requires a 2-D ADI scheme (DOUGLAS, CRAIG_SNEYD "
            f"or HV); got {scheme}."
        )

    dt = expiry / n_time
    op = operator

    def step(u, inputs):
        m, theta_m, do_correction = inputs
        theta_dt = theta_m * dt
        # Backward march from expiry: the carry entering step ``m`` has ``m``
        # steps already taken (tau = m*dt; ``m = 0`` is the terminal payoff at
        # tau = 0) and the level being solved lies one step further back. Same
        # convention as the 1-D stepper, so the two agree when A0 = A2 = 0.
        # This carried the same mirrored-tau bug as the 1-D stepper (it was
        # written to mirror it) -- see entry 2 of
        # docs/architecture/numerical-pitfalls.md.
        tau_known = m * dt          # time-remaining at the known level
        tau_solved = (m + 1) * dt   # time-remaining at the solved level

        g_known = _x_boundary_term(
            op, boundary.x_lower_fn(tau_known), boundary.x_upper_fn(tau_known)
        )
        g_solved = _x_boundary_term(
            op, boundary.x_lower_fn(tau_solved), boundary.x_upper_fn(tau_solved)
        )
        d_bc = theta_dt * (g_solved - g_known)

        a0_u = op.apply_a0(u)
        a1_u = op.apply_a1(u)
        a2_u = op.apply_a2(u)
        f_u = a0_u + a1_u + a2_u + g_known

        # --- Douglas predictor. ---
        y0 = u + dt * f_u
        y1 = _implicit_x(op, theta_dt, y0 - theta_dt * a1_u + d_bc)
        y2 = _implicit_v(op, theta_dt, y1 - theta_dt * a2_u)

        if scheme == Scheme.CRAIG_SNEYD:
            # Corrector uses the mixed term re-evaluated at the Douglas result.
            y0_hat = y0 + 0.5 * dt * (op.apply_a0(y2) - a0_u)
            y1c = _implicit_x(op, theta_dt, y0_hat - theta_dt * a1_u + d_bc)
            y2c = _implicit_v(op, theta_dt, y1c - theta_dt * a2_u)
        elif scheme == Scheme.HV:
            # Corrector uses the full operator re-evaluated at the Douglas result;
            # the log-spot boundary term cancels between predictor and corrector.
            f_y2 = op.apply_a0(y2) + op.apply_a1(y2) + op.apply_a2(y2) + g_solved
            y0_hat = y0 + 0.5 * dt * (f_y2 - f_u)
            y1c = _implicit_x(op, theta_dt, y0_hat - theta_dt * op.apply_a1(y2))
            y2c = _implicit_v(op, theta_dt, y1c - theta_dt * op.apply_a2(y2))
        else:  # Scheme.DOUGLAS
            y2c = y2

        # Rannacher steps run plain (uncorrected) Douglas; select per step.
        u_new = jnp.where(do_correction, y2c, y2)
        if event_fn is not None:
            # Forward time-level index of the level just solved: the solved
            # level sits at time-remaining (m+1)*dt, i.e. forward level
            # n_time - m - 1. Same convention as the 1-D stepper.
            u_new = event_fn(n_time - m - 1, u_new)
        return u_new, None

    steps = jnp.arange(n_time)
    thetas = jnp.where(steps < rannacher_steps, 1.0, theta)
    do_correction = steps >= rannacher_steps
    u_final, _ = jax.lax.scan(step, terminal, (steps, thetas, do_correction))
    return u_final
