"""Theta-scheme backward time-stepping for 1-D solvers.

Implements the generic theta-scheme

.. math::

    (I - \\theta\\,\\Delta t\\,\\mathcal{L})\\,V^{n}
        = (I + (1-\\theta)\\,\\Delta t\\,\\mathcal{L})\\,V^{n+1} + \\text{BC},

with ``theta = 1/2`` recovering Crank-Nicolson and ``theta = 1`` fully-implicit
backward Euler. **Rannacher start-up** runs the first ``rannacher_steps`` steps
(nearest expiry) fully-implicit to damp oscillations from non-smooth terminal
data, then switches to the requested ``theta``.

The backward march is a single ``jax.lax.scan``. Two seams hook into it:

- ``solver_fn`` replaces the plain tridiagonal solve — this is how the American
  penalty method (:mod:`valax.pricing.pde.exercise`) plugs in;
- ``event_fn`` post-processes the field after each step, which is how *discrete*
  contractual events land: coupon payments and Bermudan / callable / puttable
  exercise projection.

**Time-dependent operators.** The ``operator`` argument may carry either 1-D
bands (length ``n``, constant in time — the original behaviour) or 2-D *stacked*
bands (shape ``(n_time, n)``, one row per backward step). The stacked form lets
the spatial operator vary from step to step, which is what local-volatility
(Dupire) pricing needs: the diffusion coefficient ``sigma^2_loc(x, tau)`` is
rebuilt at every time level. Row ``m`` of the stack is used at backward step
``m`` (nearest expiry first). See
:func:`~valax.pricing.pde.coefficients.lv_operator_stack`.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.pricing.pde.boundary import Boundary1D
from valax.pricing.pde.linalg import tridiagonal_matvec, tridiagonal_solve
from valax.pricing.pde.operators import Operator1D

StepSolver = Callable[
    [Float[Array, " n_minus_1"], Float[Array, " n"], Float[Array, " n_minus_1"], Float[Array, " n"]],
    Float[Array, " n"],
]

EventFn = Callable[[Int[Array, ""], Float[Array, " n"]], Float[Array, " n"]]


def solve_backward_1d(
    operator: Operator1D,
    boundary: Boundary1D,
    terminal: Float[Array, " n"],
    *,
    expiry: Float[Array, ""],
    n_time: int,
    theta: float,
    rannacher_steps: int = 0,
    solver_fn: Optional[StepSolver] = None,
    event_fn: Optional[EventFn] = None,
) -> Float[Array, " n"]:
    """Backward time-march the terminal condition to ``t = 0``.

    Args:
        operator: Per-row spatial operator coefficients ``L``. Bands may be
            1-D (length ``n``, constant in time) or 2-D *stacked* (shape
            ``(n_time, n)``, one row per backward step — row ``m`` used at
            step ``m``, nearest expiry first) for time-dependent operators
            such as local volatility.
        boundary: Dirichlet boundary values as functions of time-remaining.
        terminal: Terminal (payoff) values on the grid.
        expiry: Time to expiry ``T`` (``dt = T / n_time``).
        n_time: Number of backward time steps.
        theta: Implicitness parameter after Rannacher start-up.
        rannacher_steps: Number of leading fully-implicit steps.
        solver_fn: Optional per-step linear solver (defaults to the plain
            tridiagonal solve). Used to inject the American penalty method.
        event_fn: Optional discrete-event hook applied to the field *after*
            each backward step, as ``event_fn(level, values) -> values``.
            ``level`` is the **forward** time-level index of the level just
            solved, counting ``0`` at ``t = 0`` up to ``n_time`` at expiry, so
            it runs over ``n_time - 1 ... 0``. It is a traced scalar, so the
            hook must select with :func:`jax.numpy.where` / indexing into a
            per-level array rather than Python control flow. This is the seam
            for contractual events that land *between* time steps — coupon
            payments, and Bermudan / callable / puttable exercise projection
            (:func:`~valax.pricing.pde.exercise.explicit_project`). Continuous
            American exercise uses ``solver_fn`` instead.

    Returns:
        The solution field on the grid at ``t = 0``.
    """
    dt = expiry / n_time
    solve = solver_fn if solver_fn is not None else tridiagonal_solve

    a_lower = dt * operator.lower
    a_diag = dt * operator.diag
    a_upper = dt * operator.upper

    # Whether the operator carries per-step bands (shape (n_time, n)) rather
    # than a single set of bands (shape (n,)). Resolved at trace time from the
    # static rank, so the branch inside ``step`` is a Python conditional and
    # never becomes dynamic control flow.
    stacked = a_lower.ndim == 2

    def step(v, inputs):
        if stacked:
            m, theta_m, al, ad, au = inputs
        else:
            m, theta_m = inputs
            al, ad, au = a_lower, a_diag, a_upper
        # The scan marches *backward from expiry*: the carry entering step ``m``
        # is the level with ``m`` steps already taken, i.e. time-remaining
        # ``m*dt`` (``m = 0`` is the terminal payoff, tau = 0), and the level
        # being solved sits one step further back at ``(m+1)*dt``. The final
        # solved level therefore carries tau = ``n_time*dt = T``, which is the
        # t = 0 price returned below.
        #
        # These two lines previously read ``(n_time - m)*dt`` and
        # ``(n_time - m - 1)*dt`` -- exactly mirrored -- so every time-dependent
        # boundary was evaluated with the wrong discount factor. It hid for a
        # long time because the default 4-sigma domain put the resulting error
        # (4.5e-4 on an ATM Black-Scholes call) inside every tolerance in the
        # repo, including the QuantLib comparisons. See entry 2 of
        # docs/architecture/numerical-pitfalls.md, and the invariance guards in
        # tests/test_pde/test_schemes.py::TestBoundaryTimeDirection and
        # tests/test_pde/test_crank_nicolson.py::TestPDEDomainWidthIndependence.
        tau_known = m * dt          # time-remaining at the known level
        tau_solved = (m + 1) * dt   # time-remaining at the solved level

        expl = 1.0 - theta_m
        # RHS = (I + (1-theta) A) v
        rhs = tridiagonal_matvec(
            expl * al[1:],
            1.0 + expl * ad,
            expl * au[:-1],
            v,
        )
        # Boundary contributions (explicit at the known level, implicit at the
        # solved level).
        bc_lo_known = boundary.lower_fn(tau_known)
        bc_lo_solved = boundary.lower_fn(tau_solved)
        bc_hi_known = boundary.upper_fn(tau_known)
        bc_hi_solved = boundary.upper_fn(tau_solved)
        rhs = rhs.at[0].add(
            expl * al[0] * bc_lo_known + theta_m * al[0] * bc_lo_solved
        )
        rhs = rhs.at[-1].add(
            expl * au[-1] * bc_hi_known + theta_m * au[-1] * bc_hi_solved
        )

        # LHS = (I - theta A)
        lhs_lower = -theta_m * al[1:]
        lhs_diag = 1.0 - theta_m * ad
        lhs_upper = -theta_m * au[:-1]

        v_new = solve(lhs_lower, lhs_diag, lhs_upper, rhs)
        if event_fn is not None:
            # Forward time-level index of the level just solved: the solved
            # level sits at time-remaining (m+1)*dt, i.e. forward level
            # n_time - m - 1.
            v_new = event_fn(n_time - m - 1, v_new)
        return v_new, None

    steps = jnp.arange(n_time)
    thetas = jnp.where(steps < rannacher_steps, 1.0, theta)
    if stacked:
        xs = (steps, thetas, a_lower, a_diag, a_upper)
    else:
        xs = (steps, thetas)
    v_final, _ = jax.lax.scan(step, terminal, xs)
    return v_final


def theta_for_scheme(scheme) -> float:
    """Map a 1-D :class:`~valax.pricing.pde.config.Scheme` to its theta value.

    Fully-implicit -> ``1.0``; Crank-Nicolson -> ``0.5``. The 2-D ADI schemes
    (:attr:`~valax.pricing.pde.config.Scheme.DOUGLAS`,
    :attr:`~valax.pricing.pde.config.Scheme.CRAIG_SNEYD`,
    :attr:`~valax.pricing.pde.config.Scheme.HV`) are **not** valid here: they
    split a multi-dimensional operator across axes and are handled by the 2-D
    ADI stepper, so passing one to the 1-D solver is a configuration error.

    Args:
        scheme: The requested time-stepping scheme.

    Returns:
        The implicitness parameter ``theta`` for the 1-D theta-scheme.

    Raises:
        ValueError: If ``scheme`` is a 2-D ADI scheme, which has no 1-D
            theta interpretation and would otherwise silently degrade to
            Crank-Nicolson.
    """
    from valax.pricing.pde.config import Scheme

    if scheme.is_adi():
        raise ValueError(
            f"{scheme} is a 2-D ADI scheme and cannot drive the 1-D "
            "theta-scheme solver; use Scheme.IMPLICIT or Scheme.CRANK_NICOLSON "
            "for 1-D problems, or route the model through the 2-D ADI stepper."
        )
    if scheme == Scheme.IMPLICIT:
        return 1.0
    return 0.5
