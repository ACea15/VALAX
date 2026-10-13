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

from valax.calibration.result import CalibrationResult
from valax.calibration.sabr import sabr_batch_level, stacked_sabr_inputs
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import period_labels
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol


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

    def total_variance(
        self,
        log_moneyness: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Total variance ``w(k, T)`` for Dupire / SLV consumers.

        Consistent with ``__call__`` via the identity

            total_variance(log(K / F_T), T) == __call__(K, T) ** 2 * T,

        where ``F_T = interp(T, expiries, forwards)``.

        Args:
            log_moneyness: ``k = log(K / F_T)``.
            expiry: Year fraction.

        Returns:
            Scalar total variance ``w = sigma_IV^2 * T``.
        """
        forward_q = jnp.interp(expiry, self.expiries, self.forwards)
        strike = forward_q * jnp.exp(log_moneyness)
        iv = self.__call__(strike, expiry)
        return iv * iv * expiry


def calibrate_sabr_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    *,
    fixed_beta: Float[Array, ""] | float | None = 0.5,
    weights_per_expiry: list[Float[Array, " n_k"]] | None = None,
    settings: SolverSettings | None = None,
    name: str = "sabr_surface",
) -> CalibrationResult:
    """Calibrate a SABR surface by fitting each expiry slice independently.

    One batched level (:func:`~valax.calibration.sabr.sabr_batch_level`):
    every slice is its own least-squares fit with its own convergence flag,
    solved together in one ``vmap``, and a failed slice is named by expiry in
    ``result.check()``.

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry.
        forwards: Forward price at each expiry.
        expiries: Expiry grid (year fractions).
        fixed_beta: Beta held fixed across all slices — 0.5 by default, as
            for :func:`~valax.calibration.sabr.calibrate_sabr` and the chain
            nodes; ``None`` calibrates it per slice.
        weights_per_expiry: Optional per-strike weights, one array per expiry.
        settings: Per-slice solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`SABRVolSurface` is ``result.value``.
    """
    expiries = jnp.asarray(expiries, dtype=jnp.float64)
    inputs = stacked_sabr_inputs(
        strikes_per_expiry, market_vols_per_expiry, forwards, expiries,
        weights_per_node=weights_per_expiry, fixed_beta=fixed_beta,
    )

    def value_fn(models: SABRModel, nodes, _inputs) -> SABRVolSurface:
        return SABRVolSurface(
            expiries=nodes.fixed["expiry"],
            forwards=nodes.fixed["forward"],
            alphas=models.alpha,
            betas=models.beta,
            rhos=models.rho,
            nus=models.nu,
        )

    return sabr_batch_level(
        name, inputs, lambda inp: inp, value_fn,
        labels=period_labels(expiries),
        fit_beta=fixed_beta is None,
        settings=settings,
    )
