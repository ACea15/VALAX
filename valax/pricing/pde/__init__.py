"""Finite difference PDE solvers."""

from valax.pricing.pde.solvers import pde_price
from valax.pricing.pde.config import PDEConfig, PDEConfig2D, Scheme, Exercise
from valax.pricing.pde.dispatch import (
    PDEResult,
    pde_price_dispatch,
    register,
    registered_recipes,
)

# Imports for side effects: populate the dispatcher registry.
from valax.pricing.pde import recipes  # noqa: F401,E402
from valax.pricing.pde import hull_white  # noqa: F401,E402
