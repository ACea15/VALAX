"""Configuration objects for the PDE pricing subsystem.

The configs are :class:`equinox.Module` pytrees with grid sizes, scheme
selection, and exercise style stored as ``static`` fields so they specialise
the compiled graph (mirroring ``MCConfig`` / ``BinomialConfig``).

:class:`PDEConfig` keeps its original three fields (``n_spot``, ``n_time``,
``spot_range``) first and defaulted identically, so every existing call site
and test remains valid; ``scheme``, ``exercise`` and ``rannacher_steps`` are
additive with backward-compatible defaults.
"""

import enum

import equinox as eqx


class Scheme(enum.Enum):
    """Time-stepping / operator-splitting scheme.

    Attributes:
        IMPLICIT: Fully-implicit backward Euler (theta = 1).
        CRANK_NICOLSON: Theta = 1/2, second-order in time.
        DOUGLAS: 2-D ADI, first-order in the mixed term (PR-2).
        CRAIG_SNEYD: 2-D ADI, second-order in the mixed term (PR-2).
        HV: 2-D ADI, Hundsdorfer-Verwer (PR-2).
    """

    IMPLICIT = "implicit"
    CRANK_NICOLSON = "cn"
    DOUGLAS = "douglas"
    CRAIG_SNEYD = "craig_sneyd"
    HV = "hv"


class Exercise(enum.Enum):
    """Exercise style applied during the backward sweep.

    Attributes:
        EUROPEAN: No early exercise.
        AMERICAN: Continuous free boundary via the penalty method.
        BERMUDAN: Discrete exercise dates via explicit projection.
    """

    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"


class PDEConfig(eqx.Module):
    """Finite-difference grid configuration for 1-D solvers.

    Attributes:
        n_spot: Number of interior spatial grid points.
        n_time: Number of backward time steps.
        spot_range: Grid half-width in units of ``vol * sqrt(T)`` around
            ``ln(spot)``.
        scheme: Time-stepping scheme (default Crank-Nicolson).
        exercise: Exercise style (default European).
        rannacher_steps: Number of fully-implicit start-up steps used to damp
            oscillations from non-smooth terminal data (0 = disabled).
        penalty_rho: Penalty coefficient for the American penalty method.
        penalty_iters: Fixed penalty iterations per time step (kept static so
            the whole solve remains differentiable).
    """

    n_spot: int = eqx.field(static=True, default=200)
    n_time: int = eqx.field(static=True, default=200)
    spot_range: float = eqx.field(static=True, default=4.0)
    scheme: Scheme = eqx.field(static=True, default=Scheme.CRANK_NICOLSON)
    exercise: Exercise = eqx.field(static=True, default=Exercise.EUROPEAN)
    rannacher_steps: int = eqx.field(static=True, default=0)
    penalty_rho: float = eqx.field(static=True, default=1.0e6)
    penalty_iters: int = eqx.field(static=True, default=5)


class PDEConfig2D(eqx.Module):
    """Finite-difference grid configuration for 2-D (ADI) solvers.

    Reserved for the PR-2 stochastic-volatility / two-asset solvers; defined
    here so the public config surface is stable from PR-1.

    Attributes:
        n_x: Interior grid points along the first axis (e.g. log-spot).
        n_y: Interior grid points along the second axis (e.g. variance).
        n_time: Number of backward time steps.
        x_range: Half-width of the first axis in std-dev units.
        y_max: Upper bound of the second axis (e.g. maximum variance).
        scheme: ADI scheme (default Craig-Sneyd).
        exercise: Exercise style (default European).
        theta: Implicitness parameter for the per-axis solves.
        rannacher_steps: Fully-implicit start-up steps.
    """

    n_x: int = eqx.field(static=True, default=128)
    n_y: int = eqx.field(static=True, default=64)
    n_time: int = eqx.field(static=True, default=200)
    x_range: float = eqx.field(static=True, default=4.0)
    y_max: float = eqx.field(static=True, default=1.0)
    scheme: Scheme = eqx.field(static=True, default=Scheme.CRAIG_SNEYD)
    exercise: Exercise = eqx.field(static=True, default=Exercise.EUROPEAN)
    theta: float = eqx.field(static=True, default=0.5)
    rannacher_steps: int = eqx.field(static=True, default=2)
