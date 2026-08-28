"""SABR swaption cube: one SABR model per (expiry, tenor) node.

The interest-rate analogue of :class:`valax.surfaces.sabr_surface.SABRVolSurface`,
lifted from a single expiry axis to the two *schedule* axes of a swaption cube --
option **expiry** and underlying-swap **tenor**. Each grid node carries a
calibrated SABR model ``(alpha, beta, rho, nu)`` and its forward par swap rate;
queries at intermediate (expiry, tenor) bilinearly interpolate those parameters
and then evaluate the Hagan smile **analytically** in strike.

The strike dimension is deliberately *not* tabulated: SABR provides the smile in
closed form, so interpolating across strike would introduce kinks, break wing
extrapolation, and forfeit the negative/zero-strike capability of the normal
expansion. Only the parameters vary across the (expiry, tenor) grid.

A static ``is_normal`` flag selects the quoting convention: lognormal (Black-76,
:func:`sabr_implied_vol`) or normal (Bachelier, :func:`sabr_normal_implied_vol`).
The normal branch additionally honours a displacement ``shift`` so that zero and
negative rates price finitely.

This object is intentionally distinct from ``SABRVolSurface`` (which is
expiry x strike and feeds equity Dupire/SLV consumers); the surface is left
untouched.
"""

import functools

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.calibration.sabr import calibrate_sabr
from valax.surfaces._interp import bilinear_2d


class SwaptionCube(eqx.Module):
    """SABR swaption cube indexed by (expiry, tenor, strike).

    SABR parameters and forward swap rates live on a rectangular
    ``(n_expiries, n_tenors)`` grid; a query bilinearly interpolates each
    parameter to ``(expiry, tenor)`` and evaluates the Hagan implied vol at
    ``strike``.

    Attributes:
        expiries: Sorted option-expiry grid (year fractions).
        tenors: Sorted underlying-swap-tenor grid (years).
        forwards: Forward par swap rate at each grid node.
        alphas: SABR alpha at each grid node.
        betas: SABR beta at each grid node.
        rhos: SABR rho at each grid node.
        nus: SABR nu at each grid node.
        shift: Displacement applied to forward and strike in the normal
            branch (ignored when ``is_normal`` is False). Default 0.
        is_normal: If True, ``__call__`` returns normal (Bachelier) vol via
            :func:`sabr_normal_implied_vol`; otherwise lognormal (Black-76) vol
            via :func:`sabr_implied_vol`.
    """

    expiries: Float[Array, " n_expiries"]
    tenors: Float[Array, " n_tenors"]
    forwards: Float[Array, "n_expiries n_tenors"]
    alphas: Float[Array, "n_expiries n_tenors"]
    betas: Float[Array, "n_expiries n_tenors"]
    rhos: Float[Array, "n_expiries n_tenors"]
    nus: Float[Array, "n_expiries n_tenors"]
    shift: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))
    is_normal: bool = eqx.field(static=True, default=False)

    def _interp(
        self,
        values: Float[Array, "n_expiries n_tenors"],
        expiry: Float[Array, ""],
        tenor: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Bilinearly interpolate a node grid to ``(expiry, tenor)``.

        ``values`` is stored ``(n_expiries, n_tenors)`` == ``(n_y, n_x)`` with
        expiry as the outer (y) axis and tenor as the inner (x) axis, matching
        the :func:`bilinear_2d` convention.
        """
        return bilinear_2d(values, self.tenors, self.expiries, tenor, expiry)

    def model_at(
        self,
        expiry: Float[Array, ""],
        tenor: Float[Array, ""],
    ) -> SABRModel:
        """Interpolated SABR model at ``(expiry, tenor)``.

        Args:
            expiry: Option expiry (year fraction).
            tenor: Underlying-swap tenor (years).

        Returns:
            A :class:`SABRModel` with bilinearly interpolated parameters.
        """
        return SABRModel(
            alpha=self._interp(self.alphas, expiry, tenor),
            beta=self._interp(self.betas, expiry, tenor),
            rho=self._interp(self.rhos, expiry, tenor),
            nu=self._interp(self.nus, expiry, tenor),
        )

    def forward_at(
        self,
        expiry: Float[Array, ""],
        tenor: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolated forward par swap rate at ``(expiry, tenor)``."""
        return self._interp(self.forwards, expiry, tenor)

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
        tenor: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Implied volatility at ``(strike, expiry, tenor)``.

        Bilinearly interpolates the SABR parameters and forward swap rate to
        ``(expiry, tenor)``, then evaluates the Hagan expansion at ``strike`` in
        the cube's quoting convention.

        Args:
            strike: Option strike (swap rate).
            expiry: Option expiry (year fraction).
            tenor: Underlying-swap tenor (years).

        Returns:
            Implied volatility (normal or lognormal per ``is_normal``).
        """
        model = self.model_at(expiry, tenor)
        forward = self.forward_at(expiry, tenor)
        if self.is_normal:
            return sabr_normal_implied_vol(model, forward, strike, expiry, self.shift)
        return sabr_implied_vol(model, forward, strike, expiry)


def calibrate_swaption_cube(
    strikes_per_node: list[list[Float[Array, " n_k"]]],
    market_vols_per_node: list[list[Float[Array, " n_k"]]],
    forwards: Float[Array, "n_expiries n_tenors"],
    expiries: Float[Array, " n_expiries"],
    tenors: Float[Array, " n_tenors"],
    is_normal: bool = False,
    shift: Float[Array, ""] = 0.0,
    fixed_beta: Float[Array, ""] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 256,
) -> SwaptionCube:
    """Calibrate a swaption cube by fitting each (expiry, tenor) slice.

    Loops the rectangular grid, calling :func:`calibrate_sabr` per node in the
    requested quoting convention, and stacks the fitted parameters into
    ``(n_expiries, n_tenors)`` grids. Fitting one smile slice at a time keeps
    each call a single-instrument least-squares problem, avoiding the
    ``optimistix`` Levenberg-Marquardt ``List arity mismatch`` that arises when
    a residual closes over a sequence of instrument pytrees.

    Both quoting conventions calibrate to machine precision on self-consistent
    data. The ``is_normal`` flag must match the convention of the input quotes:
    fitting normal-quoted vols with the lognormal formula (or vice versa) is a
    model/convention mismatch that will not converge -- see
    ``examples/pitfalls/01_normal_sabr_calibration_divergence.py``.

    Args:
        strikes_per_node: Nested list ``[i][j]`` of strike arrays for expiry
            ``i`` and tenor ``j``.
        market_vols_per_node: Nested list ``[i][j]`` of observed vol arrays,
            matching ``strikes_per_node``, in the ``is_normal`` convention.
        forwards: Forward par swap rate at each node.
        expiries: Expiry grid (year fractions).
        tenors: Tenor grid (years).
        is_normal: Calibrate against normal (Bachelier) quotes if True, else
            lognormal (Black-76).
        shift: Displacement for the normal expansion (used only when
            ``is_normal`` is True).
        fixed_beta: If provided, beta is fixed across all nodes.
        solver: Per-slice optimizer (see :func:`calibrate_sabr`).
        max_steps: Max iterations per slice.

    Returns:
        A fitted :class:`SwaptionCube`.
    """
    n_e = len(expiries)
    n_t = len(tenors)

    if is_normal:
        vol_fn = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))
    else:
        vol_fn = sabr_implied_vol

    alpha_rows, beta_rows, rho_rows, nu_rows = [], [], [], []
    for i in range(n_e):
        alphas, betas, rhos, nus = [], [], [], []
        for j in range(n_t):
            model, _ = calibrate_sabr(
                strikes=strikes_per_node[i][j],
                market_vols=market_vols_per_node[i][j],
                forward=forwards[i, j],
                expiry=expiries[i],
                fixed_beta=fixed_beta,
                solver=solver,
                max_steps=max_steps,
                vol_fn=vol_fn,
                is_normal=is_normal,
            )
            alphas.append(model.alpha)
            betas.append(model.beta)
            rhos.append(model.rho)
            nus.append(model.nu)
        alpha_rows.append(jnp.stack(alphas))
        beta_rows.append(jnp.stack(betas))
        rho_rows.append(jnp.stack(rhos))
        nu_rows.append(jnp.stack(nus))

    return SwaptionCube(
        expiries=expiries,
        tenors=tenors,
        forwards=forwards,
        alphas=jnp.stack(alpha_rows),
        betas=jnp.stack(beta_rows),
        rhos=jnp.stack(rho_rows),
        nus=jnp.stack(nu_rows),
        shift=jnp.asarray(shift),
        is_normal=is_normal,
    )
