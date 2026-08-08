"""Model -> PDE operator coefficient adapters.

Maps a VALAX model onto the drift/diffusion/discount coefficients that
:func:`~valax.pricing.pde.operators.build_operator_1d` expects. PR-1 covers the
Black-Scholes model in log-spot space; further models (local vol, Heston, SLV,
Hull-White) are added in later phases.
"""

from valax.models.black_scholes import BlackScholesModel
from valax.pricing.pde.grids import Grid1D
from valax.pricing.pde.operators import Operator1D, build_operator_1d


def bs_operator(model: BlackScholesModel, grid: Grid1D) -> Operator1D:
    """Build the log-spot Black-Scholes operator for ``model`` on ``grid``.

    In log-spot space ``x = ln S`` the drift is ``r - q - sigma^2 / 2``, the
    diffusion coefficient is ``sigma^2`` and the discount rate is ``r``.

    Args:
        model: Black-Scholes model parameters.
        grid: The log-spot grid.

    Returns:
        The assembled :class:`Operator1D`.
    """
    drift = model.rate - model.dividend - 0.5 * model.vol**2
    diffusion = model.vol**2
    return build_operator_1d(grid, drift=drift, diffusion=diffusion, discount=model.rate)
