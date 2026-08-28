"""SABR caplet/floorlet (optionlet) volatility surface: expiry x strike.

The interest-rate counterpart to :class:`valax.surfaces.sabr_surface.SABRVolSurface`.
An *optionlet* is a single-period caplet/floorlet, so its underlying is one
forward rate and the surface has **no tenor axis** (contrast the
:class:`valax.surfaces.swaption_cube.SwaptionCube`): it is expiry x strike, with
one calibrated SABR model per optionlet expiry. Queries at intermediate expiries
linearly interpolate the SABR parameters and forward, then evaluate the Hagan
smile analytically in strike.

A static ``is_normal`` flag selects the quoting convention (lognormal Black-76 vs
normal Bachelier); the normal branch honours a displacement ``shift`` so zero and
negative rates price finitely. ``SABRVolSurface`` is left untouched (it carries a
``total_variance`` method for equity Dupire/SLV consumers and must not be
repurposed).
"""

import functools

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.calibration.sabr import calibrate_sabr


class OptionletVolSurface(eqx.Module):
    """Caplet/floorlet volatility surface built from per-expiry SABR models.

    Attributes:
        expiries: Sorted optionlet-expiry grid (year fractions).
        forwards: Forward rate for the optionlet at each expiry.
        alphas: SABR alpha at each expiry.
        betas: SABR beta at each expiry.
        rhos: SABR rho at each expiry.
        nus: SABR nu at each expiry.
        shift: Displacement applied to forward and strike in the normal branch
            (ignored when ``is_normal`` is False). Default 0.
        is_normal: If True, ``__call__`` returns normal (Bachelier) vol via
            :func:`sabr_normal_implied_vol`; otherwise lognormal (Black-76) vol
            via :func:`sabr_implied_vol`.
    """

    expiries: Float[Array, " n_expiries"]
    forwards: Float[Array, " n_expiries"]
    alphas: Float[Array, " n_expiries"]
    betas: Float[Array, " n_expiries"]
    rhos: Float[Array, " n_expiries"]
    nus: Float[Array, " n_expiries"]
    shift: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))
    is_normal: bool = eqx.field(static=True, default=False)

    def model_at(self, expiry: Float[Array, ""]) -> SABRModel:
        """Interpolated SABR model at ``expiry`` (linear in each parameter)."""
        return SABRModel(
            alpha=jnp.interp(expiry, self.expiries, self.alphas),
            beta=jnp.interp(expiry, self.expiries, self.betas),
            rho=jnp.interp(expiry, self.expiries, self.rhos),
            nu=jnp.interp(expiry, self.expiries, self.nus),
        )

    def forward_at(self, expiry: Float[Array, ""]) -> Float[Array, ""]:
        """Interpolated forward rate at ``expiry``."""
        return jnp.interp(expiry, self.expiries, self.forwards)

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Implied volatility at ``(strike, expiry)``.

        Linearly interpolates the SABR parameters and forward to ``expiry``,
        then evaluates the Hagan expansion at ``strike`` in the surface's
        quoting convention.

        Args:
            strike: Optionlet strike (rate).
            expiry: Optionlet expiry (year fraction).

        Returns:
            Implied volatility (normal or lognormal per ``is_normal``).
        """
        model = self.model_at(expiry)
        forward = self.forward_at(expiry)
        if self.is_normal:
            return sabr_normal_implied_vol(model, forward, strike, expiry, self.shift)
        return sabr_implied_vol(model, forward, strike, expiry)


def build_sabr_caplet_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    is_normal: bool = False,
    shift: Float[Array, ""] = 0.0,
    fixed_beta: Float[Array, ""] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 256,
) -> OptionletVolSurface:
    """Strip a caplet vol surface by SABR-calibrating each expiry slice.

    The convention-aware analogue of
    :func:`valax.surfaces.sabr_surface.calibrate_sabr_surface`: it fits one SABR
    smile per optionlet expiry (single-instrument least-squares, so it never
    triggers the ``optimistix`` sequence-arity trap) and stacks the parameters.

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry, in
            the ``is_normal`` convention.
        forwards: Forward rate at each expiry.
        expiries: Expiry grid (year fractions).
        is_normal: Calibrate against normal (Bachelier) quotes if True, else
            lognormal (Black-76). Must match the convention of the input quotes.
        shift: Displacement for the normal expansion (used only when
            ``is_normal`` is True).
        fixed_beta: If provided, beta is fixed across all slices.
        solver: Per-slice optimizer (see :func:`calibrate_sabr`).
        max_steps: Max iterations per slice.

    Returns:
        A fitted :class:`OptionletVolSurface`.
    """
    if is_normal:
        vol_fn = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))
    else:
        vol_fn = sabr_implied_vol

    alphas, betas, rhos, nus = [], [], [], []
    for i in range(len(expiries)):
        model, _ = calibrate_sabr(
            strikes=strikes_per_expiry[i],
            market_vols=market_vols_per_expiry[i],
            forward=forwards[i],
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

    return OptionletVolSurface(
        expiries=expiries,
        forwards=forwards,
        alphas=jnp.stack(alphas),
        betas=jnp.stack(betas),
        rhos=jnp.stack(rhos),
        nus=jnp.stack(nus),
        shift=jnp.asarray(shift),
        is_normal=is_normal,
    )
