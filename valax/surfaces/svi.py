"""SVI (Stochastic Volatility Inspired) volatility surface.

Gatheral's SVI parameterization of total implied variance. Five
parameters per expiry slice define the smile shape. The raw SVI
formula is:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where k = log(K/F) is log-moneyness, and w is total implied variance
(implied_vol^2 * T). Implied vol = sqrt(w / T).

SVI is popular for equity surfaces because:
- It fits market smiles well with only 5 parameters
- It can be made arbitrage-free (calendar spread + butterfly) with
  parameter constraints
- The functional form has good extrapolation properties in the wings
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix
from jaxtyping import Float
from jax import Array

from valax.calibration.transforms import (
    TransformSpec,
    positive,
    correlation,
    model_to_unconstrained,
    unconstrained_to_model,
)


class SVISlice(eqx.Module):
    """SVI parameters for a single expiry.

    Attributes:
        a: Overall variance level.
        b: Slope of the wings (b >= 0).
        rho: Left/right asymmetry (-1 < rho < 1).
        m: Horizontal translation of the smile vertex.
        sigma: Smoothing around the vertex (sigma > 0).
    """

    a: Float[Array, ""]
    b: Float[Array, ""]
    rho: Float[Array, ""]
    m: Float[Array, ""]
    sigma: Float[Array, ""]


SVI_TRANSFORMS: dict[str, TransformSpec] = {
    "a": TransformSpec(
        to_unconstrained=lambda x: x,
        from_unconstrained=lambda x: x,
    ),
    "b": positive(),
    "rho": correlation(),
    "m": TransformSpec(
        to_unconstrained=lambda x: x,
        from_unconstrained=lambda x: x,
    ),
    "sigma": positive(),
}


def svi_total_variance(
    params: SVISlice,
    log_moneyness: Float[Array, ""],
) -> Float[Array, ""]:
    """SVI total implied variance w(k).

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    k_shifted = log_moneyness - params.m
    return params.a + params.b * (
        params.rho * k_shifted + jnp.sqrt(k_shifted**2 + params.sigma**2)
    )


def svi_implied_vol(
    params: SVISlice,
    forward: Float[Array, ""],
    strike: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Implied vol from SVI: sqrt(w(k) / T)."""
    k = jnp.log(strike / forward)
    w = svi_total_variance(params, k)
    # Clamp w to be non-negative for numerical safety
    w = jnp.maximum(w, 1e-10)
    return jnp.sqrt(w / expiry)


class SVIVolSurface(eqx.Module):
    """Volatility surface with SVI parameterization at each expiry.

    All SVI parameters are stored as vectors of length ``n_expiries``.
    Queries at intermediate expiries linearly interpolate total variance
    (not implied vol) to preserve calendar-spread arbitrage-freeness.

    Attributes:
        expiries: Sorted expiry grid (year fractions).
        forwards: Forward price at each expiry.
        a_vec: SVI ``a`` at each expiry.
        b_vec: SVI ``b`` at each expiry.
        rho_vec: SVI ``rho`` at each expiry.
        m_vec: SVI ``m`` at each expiry.
        sigma_vec: SVI ``sigma`` at each expiry.
    """

    expiries: Float[Array, " n_expiries"]
    forwards: Float[Array, " n_expiries"]
    a_vec: Float[Array, " n_expiries"]
    b_vec: Float[Array, " n_expiries"]
    rho_vec: Float[Array, " n_expiries"]
    m_vec: Float[Array, " n_expiries"]
    sigma_vec: Float[Array, " n_expiries"]

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolate SVI implied vol at (strike, expiry).

        Computes total variance at each expiry slice for the given strike,
        then linearly interpolates total variance across expiries. This
        preserves the calendar spread no-arbitrage condition (total
        variance must be non-decreasing in expiry).
        """
        forward = jnp.interp(expiry, self.expiries, self.forwards)
        k = jnp.log(strike / forward)

        # Total variance at each slice for this log-moneyness
        def _w_at_slice(a, b, rho, m, sigma, fwd):
            k_slice = jnp.log(strike / fwd)
            params = SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma)
            return svi_total_variance(params, k_slice)

        w_vec = jax.vmap(_w_at_slice)(
            self.a_vec, self.b_vec, self.rho_vec,
            self.m_vec, self.sigma_vec, self.forwards,
        )

        # Interpolate total variance in expiry
        w = jnp.interp(expiry, self.expiries, w_vec)
        w = jnp.maximum(w, 1e-10)
        return jnp.sqrt(w / expiry)


def calibrate_svi_slice(
    strikes: Float[Array, " n"],
    market_vols: Float[Array, " n"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    initial_guess: SVISlice | None = None,
    weights: Float[Array, " n"] | None = None,
    max_steps: int = 256,
) -> tuple[SVISlice, optimistix.Solution]:
    """Calibrate SVI parameters to a single smile.

    Args:
        strikes: Strike grid.
        market_vols: Observed implied vols.
        forward: Forward price.
        expiry: Time to expiry.
        initial_guess: Starting SVI parameters. If None, uses heuristic.
        weights: Per-strike weights.
        max_steps: Max optimizer iterations.

    Returns:
        (fitted_params, solution).
    """
    if weights is None:
        weights = jnp.ones_like(strikes)

    if initial_guess is None:
        atm_var = jnp.median(market_vols) ** 2 * expiry
        initial_guess = SVISlice(
            a=atm_var,
            b=jnp.array(0.1),
            rho=jnp.array(-0.2),
            m=jnp.array(0.0),
            sigma=jnp.array(0.1),
        )

    transforms = dict(SVI_TRANSFORMS)
    y0 = model_to_unconstrained(initial_guess, transforms)

    def residuals(raw_params, args):
        transforms_, template, fwd, exp, ks, mkt_vols, ws = args
        params = unconstrained_to_model(raw_params, transforms_, template)
        model_vols = jax.vmap(
            lambda K: svi_implied_vol(params, fwd, K, exp)
        )(ks)
        return ws * (model_vols - mkt_vols)

    args = (transforms, initial_guess, forward, expiry, strikes, market_vols, weights)
    opt = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optimistix.least_squares(residuals, opt, y0, args=args, max_steps=max_steps)

    fitted = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted, sol


def calibrate_svi_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    max_steps: int = 256,
) -> SVIVolSurface:
    """Calibrate an SVI surface by fitting each expiry slice independently.

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry.
        forwards: Forward price at each expiry.
        expiries: Expiry grid (year fractions).
        max_steps: Max iterations per slice.

    Returns:
        A fitted ``SVIVolSurface``.
    """
    n = len(expiries)
    a_list, b_list, rho_list, m_list, sigma_list = [], [], [], [], []

    for i in range(n):
        params, _ = calibrate_svi_slice(
            strikes=strikes_per_expiry[i],
            market_vols=market_vols_per_expiry[i],
            forward=forwards[i],
            expiry=expiries[i],
            max_steps=max_steps,
        )
        a_list.append(params.a)
        b_list.append(params.b)
        rho_list.append(params.rho)
        m_list.append(params.m)
        sigma_list.append(params.sigma)

    return SVIVolSurface(
        expiries=expiries,
        forwards=forwards,
        a_vec=jnp.stack(a_list),
        b_vec=jnp.stack(b_list),
        rho_vec=jnp.stack(rho_list),
        m_vec=jnp.stack(m_list),
        sigma_vec=jnp.stack(sigma_list),
    )
