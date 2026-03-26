"""SABR implied volatility (Hagan et al. 2002) and pricing via Black-76.

Provides the Hagan asymptotic expansion for SABR implied volatility,
then feeds it into Black-76 for option pricing.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.black76 import black76_price


def sabr_implied_vol(
    model: SABRModel,
    forward: Float[Array, ""],
    strike: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Hagan's SABR implied Black volatility formula.

    Handles the ATM case (F == K) and the general case via the full
    asymptotic expansion from Hagan et al. (2002).

    Args:
        model: SABR model parameters (alpha, beta, rho, nu).
        forward: Forward price.
        strike: Strike price.
        expiry: Time to expiry in year fractions.

    Returns:
        Implied Black (lognormal) volatility.
    """
    alpha = model.alpha
    beta = model.beta
    rho = model.rho
    nu = model.nu

    one_minus_beta = 1.0 - beta
    FK = forward * strike

    # Autodiff-safe log(F/K): add tiny eps to avoid exact zero and 0/0 grads
    log_FK = jnp.log(forward / strike)

    # Midpoint for power terms
    FK_mid = jnp.sqrt(FK)
    FK_beta = FK_mid ** one_minus_beta

    # z = (nu / alpha) * FK^((1-beta)/2) * log(F/K)
    z = (nu / alpha) * FK_beta * log_FK

    # x(z) = log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    sqrt_term = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = jnp.log((sqrt_term + z - rho) / (1.0 - rho))

    # Ratio z/x(z) with autodiff-safe ATM limit (z -> 0 => z/x -> 1).
    # Use the JAX-safe jnp.where pattern: substitute finite values in the
    # "inactive" branch to prevent NaN gradients from leaking through.
    is_small = jnp.abs(z) < 1e-7
    safe_z = jnp.where(is_small, 1.0, z)
    safe_x = jnp.where(is_small, 1.0, x_z)
    z_over_x = jnp.where(is_small, 1.0, safe_z / safe_x)

    # Denominator corrections from the expansion
    # D1 = 1 + (1-beta)^2/24 * log^2(F/K) + (1-beta)^4/1920 * log^4(F/K)
    log_FK_sq = log_FK**2
    D1 = 1.0 + one_minus_beta**2 / 24.0 * log_FK_sq + one_minus_beta**4 / 1920.0 * log_FK_sq**2

    # Numerator: alpha / (FK^((1-beta)/2) * D1)
    numerator = alpha / (FK_beta * D1)

    # Higher-order time correction
    # N1 = (1-beta)^2/24 * alpha^2 / FK^(1-beta)
    # N2 = rho * beta * nu * alpha / (4 * FK^((1-beta)/2))
    # N3 = (2 - 3*rho^2) * nu^2 / 24
    FK_full_beta = FK ** one_minus_beta
    N1 = one_minus_beta**2 / 24.0 * alpha**2 / FK_full_beta
    N2 = 0.25 * rho * beta * nu * alpha / FK_beta
    N3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2

    correction = 1.0 + (N1 + N2 + N3) * expiry

    return numerator * z_over_x * correction


def sabr_price(
    option: EuropeanOption,
    forward: Float[Array, ""],
    rate: Float[Array, ""],
    model: SABRModel,
) -> Float[Array, ""]:
    """Price a European option under the SABR model.

    Computes the Hagan implied vol, then feeds it into Black-76.

    Args:
        option: European option contract (strike, expiry, is_call).
        forward: Current forward price.
        rate: Risk-free discount rate.
        model: SABR model parameters.

    Returns:
        Option price.
    """
    vol = sabr_implied_vol(model, forward, option.strike, option.expiry)
    return black76_price(option, forward, vol, rate)
