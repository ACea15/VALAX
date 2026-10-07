"""SABR implied volatility (Hagan et al. 2002) and pricing.

Provides the Hagan asymptotic expansions for SABR implied volatility in both
the lognormal (Black-76) and normal (Bachelier) quoting conventions, then feeds
them into the corresponding closed-form pricer. The normal expansion additionally
supports a displacement (shift) so that zero and negative strikes/forwards --
routine in interest-rate markets -- price finitely, where the lognormal
expansion cannot run at all.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.black76 import black76_price
from valax.pricing.analytic.bachelier import bachelier_price


#: Crossover below which :func:`_z_over_x` uses its series rather than the
#: closed form. Chosen by balancing the two error sources rather than by picking
#: a small-looking number: the series truncates at order ``z**3``, while the
#: closed form loses precision like ``eps / z``, because ``x(z) -> z`` makes the
#: ``log`` and the division cancel. Setting ``z**3 ~ eps / z`` gives
#: ``z ~ eps**0.25 ~ 1e-4``, where both branches are accurate to about 1e-12.
_ATM_Z_TOL = 1e-4


def _z_over_x(
    z: Float[Array, ""],
    rho: Float[Array, ""],
) -> Float[Array, ""]:
    r"""The ratio :math:`z / x(z)` of Hagan's expansion, correct at the money.

    .. math::

        x(z) = \log\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho},
        \qquad \frac{z}{x(z)} \xrightarrow[z \to 0]{} 1.

    Both SABR expansions need this ratio, and both hit :math:`z = 0` exactly at
    the at-the-money strike, where the closed form is :math:`0/0`.

    The obvious guard -- ``jnp.where(|z| < tol, 1.0, z / x(z))`` -- gets the
    *value* right and the **derivative wrong**. A constant has zero derivative,
    so differentiating through it discards the whole :math:`z/x` contribution;
    since :math:`z \propto \log(F/K)`, that silently zeroes part of
    :math:`\partial\sigma/\partial F` at the money. The error is invisible in
    prices and fatal to a backbone delta.

    The fix is to switch to the Taylor series rather than to a constant, so both
    branches are smooth functions of ``z`` that agree to :math:`O(z^3)` *and*
    have the right slope:

    .. math::

        \frac{z}{x(z)} = 1 - \frac{\rho}{2} z
                         + \left(\frac{1}{6} - \frac{\rho^2}{4}\right) z^2
                         + O(z^3).

    The outer ``jnp.where`` still substitutes finite values into the inactive
    branch, which is what keeps the large-``z`` path from leaking NaN gradients.

    Args:
        z: Hagan's moneyness variable; zero at the money.
        rho: SABR correlation.

    Returns:
        The ratio :math:`z / x(z)`, with correct value and derivatives at and
        near :math:`z = 0`.
    """
    sqrt_term = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = jnp.log((sqrt_term + z - rho) / (1.0 - rho))

    is_small = jnp.abs(z) < _ATM_Z_TOL
    safe_z = jnp.where(is_small, 1.0, z)
    safe_x = jnp.where(is_small, 1.0, x_z)
    series = 1.0 - 0.5 * rho * z + (1.0 / 6.0 - 0.25 * rho**2) * z**2
    return jnp.where(is_small, series, safe_z / safe_x)


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

    # Ratio z/x(z), series-continued at the money so the *derivatives* are
    # right there too -- see :func:`_z_over_x`.
    z_over_x = _z_over_x(z, rho)

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


def sabr_normal_implied_vol(
    model: SABRModel,
    forward: Float[Array, ""],
    strike: Float[Array, ""],
    expiry: Float[Array, ""],
    shift: Float[Array, ""] = 0.0,
) -> Float[Array, ""]:
    r"""Hagan's SABR implied *normal* (Bachelier) volatility formula.

    Companion to :func:`sabr_implied_vol` for the normal quoting convention used
    by interest-rate desks. A displacement ``shift`` shifts both the forward and
    strike (the SABR process is applied to ``F + shift``), so that zero and
    negative rates -- where the lognormal expansion is undefined -- remain
    finite. Because the normal volatility is invariant under a common shift of
    ``F`` and ``K``, the returned value is fed directly to
    :func:`valax.pricing.analytic.bachelier.bachelier_price` with the unshifted
    forward and strike.

    The expansion is the normal-vol analogue of Hagan et al. (2002); it reduces
    to :math:`\sigma_N = \alpha` exactly in the arithmetic-Brownian-motion limit
    :math:`\beta = 0,\ \nu \to 0`.

    Args:
        model: SABR model parameters (alpha, beta, rho, nu).
        forward: Forward price/rate.
        strike: Strike.
        expiry: Time to expiry in year fractions.
        shift: Displacement added to forward and strike (default 0).

    Returns:
        Implied normal (absolute) volatility.

    References:
        Hagan, Kumar, Lesniewski, Woodward (2002), "Managing Smile Risk".
    """
    alpha = model.alpha
    beta = model.beta
    rho = model.rho
    nu = model.nu

    f = forward + shift
    K = strike + shift

    one_minus_beta = 1.0 - beta
    FK = f * K
    log_FK = jnp.log(f / K)
    log_FK_sq = log_FK**2

    # (FK)^((1-beta)/2) appears in z and in the beta-nu cross term.
    FK_half_beta = FK ** (0.5 * one_minus_beta)

    # z = (nu / alpha) * (FK)^((1-beta)/2) * log(F/K); x(z) as in the paper.
    z = (nu / alpha) * FK_half_beta * log_FK

    # z/x(z), series-continued at the money so the derivatives are right there
    # too -- see :func:`_z_over_x`.
    z_over_x = _z_over_x(z, rho)

    # Leading factor alpha * (FK)^(beta/2) times the moneyness series ratio.
    # Numerator series has coefficients (1/24, 1/1920); denominator carries the
    # (1-beta) powers. For beta = 0 the two series cancel identically.
    prefactor = alpha * (FK ** (0.5 * beta))
    num_series = 1.0 + log_FK_sq / 24.0 + log_FK_sq**2 / 1920.0
    den_series = (
        1.0
        + one_minus_beta**2 / 24.0 * log_FK_sq
        + one_minus_beta**4 / 1920.0 * log_FK_sq**2
    )

    # Time correction bracket.
    FK_one_minus_beta = FK ** one_minus_beta
    N1 = -beta * (2.0 - beta) / 24.0 * alpha**2 / FK_one_minus_beta
    N2 = 0.25 * rho * beta * nu * alpha / FK_half_beta
    N3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    correction = 1.0 + (N1 + N2 + N3) * expiry

    return prefactor * (num_series / den_series) * z_over_x * correction


def sabr_price_bachelier(
    option: EuropeanOption,
    forward: Float[Array, ""],
    rate: Float[Array, ""],
    model: SABRModel,
    shift: Float[Array, ""] = 0.0,
) -> Float[Array, ""]:
    """Price a European option under SABR via the normal (Bachelier) expansion.

    Computes the Hagan normal implied vol, then feeds it into Bachelier. Unlike
    :func:`sabr_price`, this handles zero and negative strikes/forwards (via
    ``shift``), as required for interest-rate options.

    Args:
        option: European option contract (strike, expiry, is_call).
        forward: Current forward price/rate.
        rate: Risk-free discount rate.
        model: SABR model parameters.
        shift: Displacement added to forward and strike (default 0).

    Returns:
        Option price.
    """
    vol = sabr_normal_implied_vol(model, forward, option.strike, option.expiry, shift)
    return bachelier_price(option, forward, vol, rate)
