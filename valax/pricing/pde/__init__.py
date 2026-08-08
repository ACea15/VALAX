"""Finite difference PDE solvers."""

from valax.pricing.pde.solvers import pde_price
from valax.pricing.pde.config import PDEConfig, PDEConfig2D, Scheme, Exercise
from valax.pricing.pde.dispatch import (
    PDEResult,
    pde_price_dispatch,
    register,
    registered_recipes,
)

# Import for side effects: populate the dispatcher registry.
from valax.pricing.pde import recipes  # noqa: F401,E402
