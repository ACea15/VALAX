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

A second pair of static fields, ``dynamics`` and ``quote_form``, records *what
happens to the smile when the curve moves* -- whether the stored (fitted)
forward or the live one enters the Hagan evaluation. See
:mod:`valax.surfaces.dynamics` for the mechanics and the
[vol-surfaces guide](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves)
for the derivation and the chosen default.

This object is intentionally distinct from ``SABRVolSurface`` (which is
expiry x strike and feeds equity Dupire/SLV consumers); the surface is left
untouched.
"""

import functools
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction
from valax.instruments.rates import Swaption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.calibration.sabr import calibrate_sabr
from valax.surfaces._interp import bilinear_2d
from valax.surfaces.dynamics import (
    default_dynamics_for,
    resolve_smile_query,
    validate_smile_dynamics,
)


#: Tolerance (in years) on the within-column agreement of derived tenors. A
#: 5-year schedule spans 1825 or 1826 days depending on where the leap years
#: fall, so the derived tenor of a "5Y" column varies by a day across expiries;
#: anything beyond this is a mis-filed node rather than a calendar artefact.
_TENOR_AXIS_TOL = 7.0 / 365.0


class SwaptionCube(eqx.Module):
    """SABR swaption cube indexed by (expiry, tenor, strike).

    SABR parameters and fitted forward swap rates live on a rectangular
    ``(n_expiries, n_tenors)`` grid; a query bilinearly interpolates each
    parameter to ``(expiry, tenor)`` and evaluates the Hagan implied vol at
    ``strike``, under the smile-dynamics convention named by ``dynamics``.

    Attributes:
        expiries: Sorted option-expiry grid (year fractions).
        tenors: Sorted underlying-swap-tenor grid (years).
        forwards: The forward par swap rate each node was **fitted** at
            (:math:`F_0`) -- not the live forward, which pricers supply at
            query time.
        alphas: SABR alpha at each grid node.
        betas: SABR beta at each grid node.
        rhos: SABR rho at each grid node.
        nus: SABR nu at each grid node.
        shift: Displacement applied to forward and strike in the normal
            branch (ignored when ``is_normal`` is False). Default 0.
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
    tenors: Float[Array, " n_tenors"]
    forwards: Float[Array, "n_expiries n_tenors"]
    alphas: Float[Array, "n_expiries n_tenors"]
    betas: Float[Array, "n_expiries n_tenors"]
    rhos: Float[Array, "n_expiries n_tenors"]
    nus: Float[Array, "n_expiries n_tenors"]
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
        validate_smile_dynamics("SwaptionCube", self.dynamics, self.quote_form)

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
        """Interpolated *fitted* forward par swap rate at ``(expiry, tenor)``.

        This is :math:`F_0` -- the forward each node was calibrated at -- not
        the live forward the curve implies now. Curve-aware pricers supply the
        latter at query time via ``__call__(..., forward=...)``.

        Args:
            expiry: Option expiry (year fraction).
            tenor: Underlying-swap tenor (years).

        Returns:
            Bilinearly interpolated fitted forward.
        """
        return self._interp(self.forwards, expiry, tenor)

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
        tenor: Float[Array, ""],
        *,
        forward: Float[Array, ""] | None = None,
    ) -> Float[Array, ""]:
        """Implied volatility at ``(strike, expiry, tenor)``.

        Bilinearly interpolates the SABR parameters -- and the fitted forward
        :math:`F_0` -- to ``(expiry, tenor)``, applies the cube's ``dynamics``
        convention to pick the strike and forward the expansion is evaluated
        at, then evaluates Hagan in the cube's quoting convention.

        Under the default ``dynamics="sticky_strike"`` the live ``forward`` is
        deliberately not consulted, so the quote at an absolute strike is
        invariant under a curve move. The other two conventions require it; see
        the [smile-dynamics section](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves)
        of the vol-surfaces guide, which owns the derivation, the default and
        the off-node caveat.

        .. warning::

            At an intermediate ``(expiry, tenor)`` the SABR parameters are
            bilinearly interpolated while ``forward`` is exact, so an off-node
            query is **mixed-order**: since :math:`\\alpha` is the volatility of
            :math:`F^\\beta`, an off-node ATM query under ``"sabr_backbone"``
            will not equal the bilinear interpolation of the neighbouring ATM
            vols.

        Args:
            strike: Option strike (absolute swap rate), whatever the
                ``quote_form`` the cube was fitted from.
            expiry: Option expiry (year fraction).
            tenor: Underlying-swap tenor (years).
            forward: Optional live forward par swap rate computed off the
                curves, supplied by the vol-source protocol of
                :mod:`valax.pricing.analytic.rates_smile`. Required by
                ``dynamics`` in ``{"sticky_moneyness", "sabr_backbone"}``;
                ignored by ``"sticky_strike"``.

        Returns:
            Implied volatility (normal or lognormal per ``is_normal``).

        Raises:
            ValueError: If ``dynamics`` needs the live forward and none was
                supplied.
        """
        model = self.model_at(expiry, tenor)
        strike_eff, anchor = resolve_smile_query(
            self.dynamics,
            strike,
            self.forward_at(expiry, tenor),
            forward,
            owner="SwaptionCube",
        )
        if self.is_normal:
            return sabr_normal_implied_vol(
                model, anchor, strike_eff, expiry, self.shift
            )
        return sabr_implied_vol(model, anchor, strike_eff, expiry)


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
    dynamics: str = "sticky_strike",
    quote_form: str = "absolute_strike",
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
    ``examples/pitfalls/01_sabr_vol_convention_mismatch.py``.

    Args:
        strikes_per_node: Nested list ``[i][j]`` of strike arrays for expiry
            ``i`` and tenor ``j``.
        market_vols_per_node: Nested list ``[i][j]`` of observed vol arrays,
            matching ``strikes_per_node``, in the ``is_normal`` convention.
        forwards: Forward par swap rate at each node -- caller-supplied, and
            recorded on the cube as the forward each node was fitted at.
        expiries: Expiry grid (year fractions).
        tenors: Tenor grid (years).
        is_normal: Calibrate against normal (Bachelier) quotes if True, else
            lognormal (Black-76).
        shift: Displacement for the normal expansion (used only when
            ``is_normal`` is True).
        fixed_beta: If provided, beta is fixed across all nodes.
        solver: Per-slice optimizer (see :func:`calibrate_sabr`).
        max_steps: Max iterations per slice.
        dynamics: Smile-dynamics convention recorded on the returned cube; see
            :class:`SwaptionCube`. Defaults to ``"sticky_strike"``.
        quote_form: Quote form recorded on the returned cube; see
            :class:`SwaptionCube`. Defaults to ``"absolute_strike"``, matching
            the absolute ``strikes_per_node`` this routine fits.

    Returns:
        A fitted :class:`SwaptionCube`.

    Raises:
        ValueError: If ``dynamics`` and ``quote_form`` are inconsistent.
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
        dynamics=dynamics,
        quote_form=quote_form,
    )


# ── Curve-aware calibration ───────────────────────────────────────────

def _validate_node_grid(swaptions: Sequence[Sequence[Swaption]]) -> None:
    """Check a nested node grid is rectangular and non-empty.

    Args:
        swaptions: Nested sequence ``[i][j]`` of representative swaptions.

    Raises:
        ValueError: If the grid is empty or its rows have unequal length.
    """
    if len(swaptions) == 0 or len(swaptions[0]) == 0:
        raise ValueError(
            "The swaption node grid is empty; it must be a non-empty "
            "[n_expiries][n_tenors] nested sequence."
        )
    n_t = len(swaptions[0])
    for i, row in enumerate(swaptions):
        if len(row) != n_t:
            raise ValueError(
                f"The swaption node grid is not rectangular: row {i} has "
                f"{len(row)} nodes but row 0 has {n_t}. A cube needs one "
                f"swaption per (expiry, tenor) node."
            )


def swaption_node_coordinates(
    swaptions: Sequence[Sequence[Swaption]],
    curve: DiscountCurve,
) -> tuple[Float[Array, " n_expiries"], Float[Array, " n_tenors"]]:
    """Derive a cube's ``(expiries, tenors)`` axes from its node schedules.

    Uses exactly the coordinate a curve-aware pricer computes --
    ``year_fraction(reference_date, expiry_date)`` and
    ``year_fraction(expiry_date, fixed_dates[-1])``, matching
    :func:`valax.pricing.analytic.rates_smile.swaption_price` -- so a cube
    calibrated here is queried *on* its nodes rather than a hair off them.

    The expiry axis is taken from each row and required to agree exactly within
    the row (all nodes of a row share an expiry date). The tenor axis is taken
    from the first row; later rows may differ by up to a week because a 5-year
    schedule spans 1825 or 1826 days depending on the leap years it covers.

    Args:
        swaptions: Nested sequence ``[i][j]`` of representative swaptions, one
            per node. Only their schedules and day counts are read.
        curve: Curve supplying the valuation (reference) date.

    Returns:
        Tuple ``(expiries, tenors)`` of year-fraction axes.

    Raises:
        ValueError: If the grid is not rectangular, a row's nodes disagree on
            the expiry year fraction, or a column's tenors disagree by more
            than a week (a mis-filed node rather than a calendar artefact).
    """
    _validate_node_grid(swaptions)

    def _expiry_of(s: Swaption) -> float:
        return float(
            year_fraction(curve.reference_date, s.expiry_date, s.day_count)
        )

    def _tenor_of(s: Swaption) -> float:
        return float(
            year_fraction(s.expiry_date, s.fixed_dates[-1], s.day_count)
        )

    expiries = []
    for i, row in enumerate(swaptions):
        values = [_expiry_of(s) for s in row]
        if max(values) != min(values):
            raise ValueError(
                f"Row {i} of the swaption node grid spans several expiries "
                f"({min(values)} .. {max(values)} years). Every node of a row "
                f"must share one option expiry; the tenor varies along the row."
            )
        expiries.append(values[0])

    tenors = [_tenor_of(s) for s in swaptions[0]]
    for i, row in enumerate(swaptions[1:], start=1):
        for j, s in enumerate(row):
            drift = abs(_tenor_of(s) - tenors[j])
            if drift > _TENOR_AXIS_TOL:
                raise ValueError(
                    f"Node ({i}, {j}) of the swaption grid has underlying "
                    f"tenor {_tenor_of(s):.4f}y but column {j} was set to "
                    f"{tenors[j]:.4f}y by row 0 (drift {drift:.4f}y > "
                    f"{_TENOR_AXIS_TOL:.4f}y). Columns must share one "
                    f"underlying tenor."
                )

    return jnp.asarray(expiries), jnp.asarray(tenors)


def swaption_node_forwards_and_annuities(
    swaptions: Sequence[Sequence[Swaption]],
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> tuple[
    Float[Array, "n_expiries n_tenors"], Float[Array, "n_expiries n_tenors"]
]:
    """Per-node forward par swap rates and discount annuities, off the curves.

    The curve-derived half of a curve-aware cube calibration. Calls
    :func:`valax.pricing.analytic.swaptions.swaption_forward_and_annuity` per
    node, so :math:`S_x` and :math:`A_d` are computed by the same code the
    pricer uses -- and are differentiable functions of the curve pillars, which
    is what makes :math:`\\partial\\theta_v / \\partial\\theta_c` non-zero.

    The annuities are not needed to fit vol *quotes*, but they are needed to
    convert premium quotes, so both grids are returned together.

    Args:
        swaptions: Nested sequence ``[i][j]`` of representative swaptions.
        curve: Discount (OIS / collateral) curve.
        forward_curve: Optional tenor / projection curve for the underlying
            swaps' floating legs. Defaults to ``curve`` (single-curve).

    Returns:
        Tuple ``(forwards, annuities)``, each ``(n_expiries, n_tenors)``.

    Raises:
        ValueError: If the node grid is not rectangular.
    """
    _validate_node_grid(swaptions)
    forward_rows, annuity_rows = [], []
    for row in swaptions:
        pairs = [
            swaption_forward_and_annuity(s, curve, forward_curve) for s in row
        ]
        forward_rows.append(jnp.stack([f for f, _ in pairs]))
        annuity_rows.append(jnp.stack([a for _, a in pairs]))
    return jnp.stack(forward_rows), jnp.stack(annuity_rows)


def calibrate_swaption_cube_from_curves(
    swaptions: Sequence[Sequence[Swaption]],
    strikes_per_node: Sequence[Sequence[Float[Array, " n_k"]]],
    market_vols_per_node: Sequence[Sequence[Float[Array, " n_k"]]],
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
    *,
    quote_form: str = "absolute_strike",
    dynamics: str | None = None,
    is_normal: bool = False,
    shift: Float[Array, ""] = 0.0,
    fixed_beta: Float[Array, ""] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 256,
) -> SwaptionCube:
    r"""Calibrate a swaption cube with the forwards derived from the curves.

    The curve-aware counterpart of :func:`calibrate_swaption_cube`, which takes
    a caller-supplied ``forwards`` array. That array is where the dual-curve
    chain used to break: a forward handed in from outside is a constant as far
    as the fit is concerned, so :math:`\partial\theta_v/\partial\theta_c = 0`
    and no curve risk could ever reach the volatility layer. Here the grid axes
    *and* the at-the-money anchors come from the node schedules and the curves
    themselves, computed by the same
    :func:`~valax.pricing.analytic.swaptions.swaption_forward_and_annuity` the
    pricer uses -- so the fitted parameters are differentiable functions of the
    curve pillars, and a node reprices exactly because its grid coordinate is
    the one the pricer will ask for.

    This is also the prerequisite for ``dynamics="sabr_backbone"``: that
    convention only reproduces the calibrated market vol at a node when
    :math:`F = F_0` there, which holds precisely when the fitted forward came
    from the curve and schedule conventions the pricer uses.

    Strikes may be given as absolute rates or as offsets to each node's own
    at-the-money forward (``quote_form="atm_offset"``), which is how swaption
    matrices are usually quoted; the offsets are converted to absolute strikes
    per node using that node's derived :math:`F_0`.

    Args:
        swaptions: Nested sequence ``[i][j]`` of representative swaptions, one
            per ``(expiry, tenor)`` node. Only the schedule, day count and
            (for the premium routes) notional are read -- the ``strike`` field
            is ignored, since the smile spans many strikes per node.
        strikes_per_node: Nested sequence ``[i][j]`` of strike arrays. Absolute
            rates when ``quote_form="absolute_strike"``, offsets to that node's
            forward when ``quote_form="atm_offset"``.
        market_vols_per_node: Nested sequence ``[i][j]`` of observed vol
            arrays matching ``strikes_per_node``, in the ``is_normal``
            convention.
        curve: Discount (OIS / collateral) curve.
        forward_curve: Optional tenor / projection curve for the underlying
            swaps. Defaults to ``curve`` (single-curve setup).
        quote_form: ``"absolute_strike"`` (default) or ``"atm_offset"``;
            recorded on the returned cube.
        dynamics: Smile-dynamics convention recorded on the returned cube.
            Defaults to the one ``quote_form`` implies (see
            :func:`~valax.surfaces.dynamics.default_dynamics_for`);
            ``"sabr_backbone"`` is never selected implicitly.
        is_normal: Calibrate against normal (Bachelier) quotes if True, else
            lognormal (Black-76). Must match the convention of the quotes.
        shift: Displacement for the normal expansion (used only when
            ``is_normal`` is True).
        fixed_beta: If provided, beta is fixed across all nodes.
        solver: Per-slice optimizer (see
            :func:`valax.calibration.sabr.calibrate_sabr`).
        max_steps: Max iterations per slice.

    Returns:
        A fitted :class:`SwaptionCube` whose ``forwards``, ``expiries`` and
        ``tenors`` are all curve- and schedule-derived.

    Raises:
        ValueError: If the node grid is not rectangular or its axes are
            inconsistent (see :func:`swaption_node_coordinates`), if the quote
            grids do not match the node grid, or if ``dynamics`` and
            ``quote_form`` contradict each other.
    """
    _validate_node_grid(swaptions)
    n_e, n_t = len(swaptions), len(swaptions[0])
    for name, grid in (
        ("strikes_per_node", strikes_per_node),
        ("market_vols_per_node", market_vols_per_node),
    ):
        if len(grid) != n_e or any(len(row) != n_t for row in grid):
            raise ValueError(
                f"{name} must match the swaption node grid's shape "
                f"({n_e}, {n_t})."
            )

    if dynamics is None:
        dynamics = default_dynamics_for(quote_form)
    validate_smile_dynamics("SwaptionCube", dynamics, quote_form)

    expiries, tenors = swaption_node_coordinates(swaptions, curve)
    forwards, _ = swaption_node_forwards_and_annuities(
        swaptions, curve, forward_curve
    )

    if quote_form == "atm_offset":
        absolute_strikes = [
            [forwards[i, j] + jnp.asarray(strikes_per_node[i][j])
             for j in range(n_t)]
            for i in range(n_e)
        ]
    else:
        absolute_strikes = [
            [jnp.asarray(strikes_per_node[i][j]) for j in range(n_t)]
            for i in range(n_e)
        ]

    return calibrate_swaption_cube(
        strikes_per_node=absolute_strikes,
        market_vols_per_node=market_vols_per_node,
        forwards=forwards,
        expiries=expiries,
        tenors=tenors,
        is_normal=is_normal,
        shift=shift,
        fixed_beta=fixed_beta,
        solver=solver,
        max_steps=max_steps,
        dynamics=dynamics,
        quote_form=quote_form,
    )


def calibrate_swaption_cube_from_graph(
    swaptions: Sequence[Sequence[Swaption]],
    strikes_per_node: Sequence[Sequence[Float[Array, " n_k"]]],
    market_vols_per_node: Sequence[Sequence[Float[Array, " n_k"]]],
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
    **kwargs: Any,
) -> SwaptionCube:
    """:func:`calibrate_swaption_cube_from_curves` resolved on a curve graph.

    The dual-curve entry point: the discounting and projection curves are
    looked up on a bootstrapped :class:`~valax.curves.graph.CurveGraph`, so the
    cube is anchored on the same multi-curve snapshot the book is priced from.

    Args:
        swaptions: Nested sequence ``[i][j]`` of representative swaptions.
        strikes_per_node: Nested sequence of strike arrays.
        market_vols_per_node: Nested sequence of observed vol arrays.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the projection curve for the underlying
            swaps' floating legs.
        **kwargs: Forwarded to :func:`calibrate_swaption_cube_from_curves`.

    Returns:
        A fitted :class:`SwaptionCube`.
    """
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return calibrate_swaption_cube_from_curves(
        swaptions,
        strikes_per_node,
        market_vols_per_node,
        discount_curve,
        forward_curve,
        **kwargs,
    )
