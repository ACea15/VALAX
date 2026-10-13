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

Static ``dynamics`` / ``quote_form`` fields record *what happens to the smile
when the curve moves* -- whether the stored (fitted) forward or the live one
enters the Hagan evaluation. See :mod:`valax.surfaces.dynamics` for the
mechanics and the
[vol-surfaces guide](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves)
for the derivation and the chosen default.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.calibration.result import CalibrationResult
from valax.calibration.sabr import (
    sabr_batch_level,
    sabr_vol_fn,
    stacked_sabr_inputs,
)
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import period_labels
from valax.surfaces.dynamics import resolve_smile_query, validate_smile_dynamics


class OptionletVolSurface(eqx.Module):
    """Caplet/floorlet volatility surface built from per-expiry SABR models.

    Attributes:
        expiries: Sorted optionlet-expiry grid (year fractions).
        forwards: The forward rate each optionlet was **fitted** at
            (:math:`F_0`) -- not the live forward, which pricers supply at
            query time.
        alphas: SABR alpha at each expiry.
        betas: SABR beta at each expiry.
        rhos: SABR rho at each expiry.
        nus: SABR nu at each expiry.
        shift: Displacement applied to forward and strike in the normal branch
            (ignored when ``is_normal`` is False). Default 0.
        is_normal: If True, ``__call__`` returns normal (Bachelier) vol via
            :func:`sabr_normal_implied_vol`; otherwise lognormal (Black-76) vol
            via :func:`sabr_implied_vol`.
        dynamics: Smile-dynamics convention -- which forward the Hagan
            evaluation uses when the curve has moved away from the fitted
            forward. One of
            :data:`~valax.surfaces.dynamics.SMILE_DYNAMICS`, defaulting to
            ``"sticky_strike"`` (the convention VALAX has always applied, so
            existing prices are preserved exactly). Static.
        quote_form: How the calibrating quote's strike was expressed, one of
            :data:`~valax.surfaces.dynamics.QUOTE_FORMS`. It carries no
            query-time behaviour of its own -- ``__call__`` always takes an
            absolute strike -- but it is validated against ``dynamics``,
            because the dynamics is a consequence of what the quote holds
            fixed. Static.
    """

    expiries: Float[Array, " n_expiries"]
    forwards: Float[Array, " n_expiries"]
    alphas: Float[Array, " n_expiries"]
    betas: Float[Array, " n_expiries"]
    rhos: Float[Array, " n_expiries"]
    nus: Float[Array, " n_expiries"]
    shift: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))
    is_normal: bool = eqx.field(static=True, default=False)
    dynamics: str = eqx.field(static=True, default="sticky_strike")
    quote_form: str = eqx.field(static=True, default="absolute_strike")

    def __check_init__(self) -> None:
        """Validate the smile-dynamics configuration.

        Raises:
            ValueError: If ``dynamics`` or ``quote_form`` is unrecognised, or
                the pair is contradictory.
        """
        validate_smile_dynamics(
            "OptionletVolSurface", self.dynamics, self.quote_form
        )

    def model_at(self, expiry: Float[Array, ""]) -> SABRModel:
        """Interpolated SABR model at ``expiry`` (linear in each parameter)."""
        return SABRModel(
            alpha=jnp.interp(expiry, self.expiries, self.alphas),
            beta=jnp.interp(expiry, self.expiries, self.betas),
            rho=jnp.interp(expiry, self.expiries, self.rhos),
            nu=jnp.interp(expiry, self.expiries, self.nus),
        )

    def forward_at(self, expiry: Float[Array, ""]) -> Float[Array, ""]:
        """Interpolated *fitted* forward rate at ``expiry``.

        This is :math:`F_0` -- the forward each optionlet was calibrated at --
        not the live forward the curve implies now, which curve-aware pricers
        supply via ``__call__(..., forward=...)``.

        Args:
            expiry: Optionlet expiry (year fraction).

        Returns:
            Linearly interpolated fitted forward.
        """
        return jnp.interp(expiry, self.expiries, self.forwards)

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
        *,
        forward: Float[Array, ""] | None = None,
    ) -> Float[Array, ""]:
        """Implied volatility at ``(strike, expiry)``.

        Linearly interpolates the SABR parameters -- and the fitted forward
        :math:`F_0` -- to ``expiry``, applies the surface's ``dynamics``
        convention to pick the strike and forward the expansion is evaluated
        at, then evaluates Hagan in the surface's quoting convention.

        Under the default ``dynamics="sticky_strike"`` the live ``forward`` is
        deliberately not consulted. The other two conventions require it; see
        the [smile-dynamics section](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves)
        of the vol-surfaces guide, which owns the derivation and the default.
        The off-node mixed-order caveat documented there applies here too, with
        linear rather than bilinear parameter interpolation.

        Args:
            strike: Optionlet strike (absolute rate), whatever the
                ``quote_form`` the surface was fitted from.
            expiry: Optionlet expiry (year fraction).
            forward: Optional live period forward computed off the curves,
                supplied by the vol-source protocol of
                :mod:`valax.pricing.analytic.rates_smile`. Required by
                ``dynamics`` in ``{"sticky_moneyness", "sabr_backbone"}``;
                ignored by ``"sticky_strike"``.

        Returns:
            Implied volatility (normal or lognormal per ``is_normal``).

        Raises:
            ValueError: If ``dynamics`` needs the live forward and none was
                supplied.
        """
        model = self.model_at(expiry)
        strike_eff, anchor = resolve_smile_query(
            self.dynamics,
            strike,
            self.forward_at(expiry),
            forward,
            owner="OptionletVolSurface",
        )
        if self.is_normal:
            return sabr_normal_implied_vol(
                model, anchor, strike_eff, expiry, self.shift
            )
        return sabr_implied_vol(model, anchor, strike_eff, expiry)


def build_sabr_caplet_surface(
    strikes_per_expiry: list[Float[Array, " n_k"]],
    market_vols_per_expiry: list[Float[Array, " n_k"]],
    forwards: Float[Array, " n_expiries"],
    expiries: Float[Array, " n_expiries"],
    *,
    is_normal: bool = False,
    shift: Float[Array, ""] = 0.0,
    fixed_beta: Float[Array, ""] | None = None,
    weights_per_expiry: list[Float[Array, " n_k"]] | None = None,
    settings: SolverSettings | None = None,
    dynamics: str = "sticky_strike",
    quote_form: str = "absolute_strike",
    name: str = "optionlet_surface",
) -> CalibrationResult:
    """Fit a caplet vol surface by SABR-calibrating each expiry slice.

    The convention-aware analogue of
    :func:`valax.surfaces.sabr_surface.calibrate_sabr_surface`: one SABR smile
    per optionlet expiry, solved as one batched level
    (:func:`~valax.calibration.sabr.sabr_batch_level`) so a failed slice is
    named by expiry in ``result.check()``.

    Args:
        strikes_per_expiry: List of strike arrays, one per expiry.
        market_vols_per_expiry: List of observed vol arrays, one per expiry, in
            the ``is_normal`` convention.
        forwards: Forward rate at each expiry; recorded on the surface as the
            forward each slice was fitted at.
        expiries: Expiry grid (year fractions).
        is_normal: Calibrate against normal (Bachelier) quotes if True, else
            lognormal (Black-76). Must match the convention of the input quotes.
        shift: Displacement for the normal expansion (used only when
            ``is_normal`` is True).
        fixed_beta: If provided, beta is fixed across all slices.
        weights_per_expiry: Optional per-strike weights, one array per expiry.
        settings: Per-slice solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        dynamics: Smile-dynamics convention recorded on the returned surface;
            see :class:`OptionletVolSurface`. Defaults to ``"sticky_strike"``.
        quote_form: Quote form recorded on the returned surface. Defaults to
            ``"absolute_strike"``, matching the absolute
            ``strikes_per_expiry`` this routine fits.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`OptionletVolSurface` is ``result.value``.

    Raises:
        ValueError: If ``dynamics`` and ``quote_form`` are inconsistent.
    """
    validate_smile_dynamics("OptionletVolSurface", dynamics, quote_form)
    expiries = jnp.asarray(expiries, dtype=jnp.float64)
    inputs = stacked_sabr_inputs(
        strikes_per_expiry, market_vols_per_expiry, forwards, expiries,
        weights_per_node=weights_per_expiry, fixed_beta=fixed_beta,
    )

    def value_fn(models: SABRModel, nodes, _inputs) -> OptionletVolSurface:
        return OptionletVolSurface(
            expiries=nodes.fixed["expiry"],
            forwards=nodes.fixed["forward"],
            alphas=models.alpha,
            betas=models.beta,
            rhos=models.rho,
            nus=models.nu,
            shift=jnp.asarray(shift),
            is_normal=is_normal,
            dynamics=dynamics,
            quote_form=quote_form,
        )

    return sabr_batch_level(
        name, inputs, lambda inp: inp, value_fn,
        labels=period_labels(expiries),
        fit_beta=fixed_beta is None,
        vol_fn=sabr_vol_fn(is_normal, shift),
        is_normal=is_normal,
        settings=settings,
    )
