"""SABR-based volatility surface: one SABR model per expiry slice.

Industry standard for rates vol surfaces (swaption cubes) and common
for equity. Each expiry has its own (alpha, beta, rho, nu) parameters;
queries at intermediate expiries interpolate the parameters linearly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.calibration.sabr import calibrate_sabr


class SABRVolSurface(eqx.Module):
    """Volatility surface built from SABR models at discrete expiries.

    At each expiry slice, a calibrated SABR model defines the smile.
    Queries at intermediate expiries use linearly interpolated SABR
    parameters. All parameters are differentiable leaves.

    Attributes:
        expiries: Sorted expiry grid (year fractions).
        forwards: Forward price at each expiry.
        alphas: SABR alpha at each expiry.
        betas: SABR beta at each expiry.
        rhos: SABR rho at each expiry.
        nus: SABR nu at each expiry.
    """

    expiries: Float[Array, " n_expiries"]
    forwards: Float[Array, " n_expiries"]
    alphas: Float[Array, " n_expiries"]
    betas: Float[Array, " n_expiries"]
    rhos: Float[Array, " n_expiries"]
    nus: Float[Array, " n_expiries"]

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolate SABR implied vol at (strike, expiry).

        Linearly interpolates SABR parameters and the forward to the
        query expiry, then evaluates the Hagan formula.
        """
        alpha = jnp.interp(expiry, self.expiries, self.alphas)
        beta = jnp.interp(expiry, self.expiries, self.betas)
        rho = jnp.interp(expiry, self.expiries, self.rhos)
        nu = jnp.interp(expiry, self.expiries, self.nus)
        forward = jnp.interp(expiry, self.expiries, self.forwards)

        model = SABRModel(alpha=alpha, beta=beta, rho=rho, nu=nu)
        return sabr_implied_vol(model, forward, strike, expiry)


def calibrate_sabr_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    fixed_beta: Float[Array, ""] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 256,
) -> SABRVolSurface:
    """Calibrate a SABR surface by fitting each expiry slice independently.

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry.
        forwards: Forward price at each expiry.
        expiries: Expiry grid (year fractions).
        fixed_beta: If provided, beta is fixed across all slices.
        solver: Optimizer for each slice.
        max_steps: Max iterations per slice.

    Returns:
        A fitted ``SABRVolSurface``.
    """
    n = len(expiries)
    alphas = []
    betas = []
    rhos = []
    nus = []

    for i in range(n):
        model, _ = calibrate_sabr(
            strikes=strikes_per_expiry[i],
            market_vols=market_vols_per_expiry[i],
            forward=forwards[i],
            expiry=expiries[i],
            fixed_beta=fixed_beta,
            solver=solver,
            max_steps=max_steps,
        )
        alphas.append(model.alpha)
        betas.append(model.beta)
        rhos.append(model.rho)
        nus.append(model.nu)

    return SABRVolSurface(
        expiries=expiries,
        forwards=forwards,
        alphas=jnp.stack(alphas),
        betas=jnp.stack(betas),
        rhos=jnp.stack(rhos),
        nus=jnp.stack(nus),
    )
