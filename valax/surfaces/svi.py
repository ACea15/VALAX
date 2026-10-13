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
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult
from valax.calibration.settings import FIT_SETTINGS, SolverSettings
from valax.calibration.solve import (
    period_labels,
    solve_batched_level,
    solve_level,
    stack_slices,
)
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

        # Interpolate total variance in expiry. Below the first slice
        # we extrapolate linearly through the origin so that implied
        # vol stays constant (matches the first-slice IV) as T → 0,
        # rather than diverging. This also keeps ∂w/∂T positive for
        # Dupire below the first slice. Above the last slice we keep
        # ``jnp.interp``'s flat-w extrapolation.
        w_interior = jnp.interp(expiry, self.expiries, w_vec)
        w_below = w_vec[0] * expiry / self.expiries[0]
        w = jnp.where(expiry < self.expiries[0], w_below, w_interior)
        w = jnp.maximum(w, 1e-10)
        return jnp.sqrt(w / expiry)

    def total_variance(
        self,
        log_moneyness: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Total variance ``w(k, T)`` for Dupire / SLV consumers.

        Consistent with ``__call__`` via the identity

            total_variance(log(K / F_T), T) == __call__(K, T) ** 2 * T,

        where ``F_T = interp(T, expiries, forwards)``. Concretely we
        convert ``k`` back to an absolute strike using the
        query-expiry forward, then evaluate each slice's SVI total
        variance at that slice's own log-moneyness and linearly
        interpolate across expiries.

        Args:
            log_moneyness: ``k = log(K / F_T)`` for the query expiry.
            expiry: Year fraction.

        Returns:
            Scalar total variance ``w = sigma_IV^2 * T``, clamped at
            ``1e-10`` for numerical safety (matches ``__call__``).
        """
        forward_q = jnp.interp(expiry, self.expiries, self.forwards)
        strike = forward_q * jnp.exp(log_moneyness)

        def _w_at_slice(a, b, rho, m, sigma, fwd_slice):
            k_slice = jnp.log(strike / fwd_slice)
            params = SVISlice(a=a, b=b, rho=rho, m=m, sigma=sigma)
            return svi_total_variance(params, k_slice)

        w_vec = jax.vmap(_w_at_slice)(
            self.a_vec, self.b_vec, self.rho_vec,
            self.m_vec, self.sigma_vec, self.forwards,
        )
        # Linear-in-T extrapolation through the origin below the first
        # slice (constant IV in the T → 0 limit). Matches ``__call__``.
        w_interior = jnp.interp(expiry, self.expiries, w_vec)
        w_below = w_vec[0] * expiry / self.expiries[0]
        w = jnp.where(expiry < self.expiries[0], w_below, w_interior)
        return jnp.maximum(w, 1e-10)


def _svi_guess(
    market_vols: Float[Array, " n"],
    expiry: Float[Array, ""],
) -> SVISlice:
    """Heuristic SVI seed: flat at the median total variance."""
    atm_var = jnp.median(market_vols) ** 2 * expiry
    return SVISlice(
        a=atm_var,
        b=jnp.array(0.1),
        rho=jnp.array(-0.2),
        m=jnp.array(0.0),
        sigma=jnp.array(0.1),
    )


def _svi_assemble(theta: dict, _inputs: LevelInputs) -> SVISlice:
    """Unconstrained parameters -> :class:`SVISlice`."""
    template = SVISlice(
        a=jnp.asarray(0.0), b=jnp.asarray(0.0), rho=jnp.asarray(0.0),
        m=jnp.asarray(0.0), sigma=jnp.asarray(0.0),
    )
    return unconstrained_to_model(theta, SVI_TRANSFORMS, template)


def svi_slice_residual(theta: dict, inputs: LevelInputs) -> Float[Array, " n"]:
    """Weighted vol residual of one SVI slice.

    ``inputs.quotes`` are the market vols; ``inputs.fixed`` carries
    ``forward``, ``expiry``, ``strikes`` and ``weights``.

    Args:
        theta: Unconstrained SVI parameters.
        inputs: The slice's level inputs.

    Returns:
        ``weights * (svi_vol - market_vol)``.
    """
    params = _svi_assemble(theta, inputs)
    fx = inputs.fixed
    model_vols = jax.vmap(
        lambda k: svi_implied_vol(params, fx["forward"], k, fx["expiry"])
    )(fx["strikes"])
    return fx["weights"] * (model_vols - inputs.quotes)


def calibrate_svi_slice(
    strikes: Float[Array, " n"],
    market_vols: Float[Array, " n"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    *,
    initial_guess: SVISlice | None = None,
    weights: Float[Array, " n"] | None = None,
    settings: SolverSettings | None = None,
    name: str = "svi",
) -> CalibrationResult:
    """Calibrate SVI parameters to a single smile.

    A weighted least-squares level in implied-vol space (unconstrained raw
    SVI parameters, see ``SVI_TRANSFORMS``). Non-arbitrage constraints are
    not imposed here.

    Args:
        strikes: Strike grid.
        market_vols: Observed implied vols.
        forward: Forward price.
        expiry: Time to expiry.
        initial_guess: Starting SVI parameters. If None, uses heuristic.
        weights: Per-strike weights.
        settings: Solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`SVISlice` is ``result.value``.
    """
    settings = FIT_SETTINGS if settings is None else settings
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    inputs = LevelInputs(
        quotes=jnp.asarray(market_vols, dtype=jnp.float64),
        fixed={
            "forward": jnp.asarray(forward, dtype=jnp.float64),
            "expiry": jnp.asarray(expiry, dtype=jnp.float64),
            "strikes": strikes,
            "weights": jnp.ones_like(strikes) if weights is None
            else jnp.asarray(weights, dtype=jnp.float64),
        },
    )
    if initial_guess is None:
        initial_guess = _svi_guess(inputs.quotes, inputs.fixed["expiry"])
    theta0 = jax.lax.stop_gradient(
        model_to_unconstrained(initial_guess, SVI_TRANSFORMS)
    )
    return solve_level(
        name, "least_squares", svi_slice_residual, theta0, inputs,
        _svi_assemble, settings,
    )


def calibrate_svi_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    *,
    weights_per_expiry: list[Float[Array, " n_k"]] | None = None,
    settings: SolverSettings | None = None,
    name: str = "svi_surface",
) -> CalibrationResult:
    """Calibrate an SVI surface by fitting each expiry slice independently.

    One batched level (:func:`~valax.calibration.solve.solve_batched_level`):
    the slices are solved separately but in one ``vmap``, each reports its
    own convergence, and a failed slice is named by expiry in
    ``result.check()``. Ragged strike grids are padded with zero-weight
    entries (:func:`~valax.calibration.solve.stack_slices`).

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry.
        forwards: Forward price at each expiry.
        expiries: Expiry grid (year fractions).
        weights_per_expiry: Optional per-strike weights, one array per expiry.
        settings: Solver settings shared by every slice; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`SVIVolSurface` is ``result.value``.
    """
    settings = FIT_SETTINGS if settings is None else settings
    strikes, vols, weights = stack_slices(
        strikes_per_expiry, market_vols_per_expiry, weights_per_expiry
    )
    expiries = jnp.asarray(expiries, dtype=jnp.float64)
    forwards = jnp.asarray(forwards, dtype=jnp.float64)
    inputs = LevelInputs(
        quotes=vols,
        fixed={
            "forward": forwards, "expiry": expiries,
            "strikes": strikes, "weights": weights,
        },
    )

    def node_inputs(inp: LevelInputs) -> LevelInputs:
        return inp

    def assemble(theta: dict, inp: LevelInputs) -> SVIVolSurface:
        p = jax.vmap(lambda t: _svi_assemble(t, inp))(theta)
        return SVIVolSurface(
            expiries=inp.fixed["expiry"], forwards=inp.fixed["forward"],
            a_vec=p.a, b_vec=p.b, rho_vec=p.rho, m_vec=p.m, sigma_vec=p.sigma,
        )

    theta0 = jax.lax.stop_gradient(
        jax.vmap(
            lambda v, t: model_to_unconstrained(_svi_guess(v, t), SVI_TRANSFORMS)
        )(vols, expiries)
    )
    return solve_batched_level(
        name, "least_squares", svi_slice_residual, theta0, inputs,
        node_inputs, assemble, settings,
        labels=period_labels(expiries),
    )
