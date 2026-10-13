r"""Node factories: the calibration levels a chain is assembled from.

Each factory returns a :class:`~valax.calibration.chain.CalibrationNode`. The
recipe is always the same three functions, which is the whole contract for
adding a new level:

1. a **residual** ``(theta, inputs) -> vector`` — square for an exact level,
   the fit residual for a least-squares one;
2. an **assemble** map ``(theta, inputs) -> object`` from the unknowns to the
   calibrated object (curve graph, surface, model);
3. an **install** map ``(market, object) -> market`` — or ``None`` for a
   model-scope node, whose object is the ``model`` argument of ``price``.

:func:`~valax.calibration.chain.solve_level` does the rest: solving with the
configured :class:`~valax.calibration.settings.SolverSettings`, reporting
convergence, and exposing the level to both differentiation paths.

Factories provided here span the three tiers of market:

* :func:`market_inputs_node` — explicit spots, dividends, flat rates and flat
  vols (the constant-parameter Black-Scholes market, as a chain);
* :func:`curve_graph_node` — joint single- or multi-curve bootstrap (level 1a);
* :func:`sabr_smile_node` — SABR smiles per expiry on forwards read from the
  upstream market (level 1b);
* :func:`swaption_cube_node`, :func:`caplet_strip_node` and
  :func:`optionlet_sabr_node` — the rates vol levels (swaption cube; cap
  strip → optionlet SABR), forwards and annuities from the upstream curves
  (level 1b);
* :func:`price_fit_node` — any model with a registered pricer, fitted to a
  basket through :func:`valax.core.pricing.price` (level 1b quote conversion
  or level 2 model calibration — the same factory serves both scopes);
* :func:`swaption_basket_node` with :func:`coterminal_swaptions` — a rates
  model fitted per trade to a swaption basket priced off the calibrated cube
  (level 2), batched over same-shaped baskets with ``eqx.filter_vmap``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.calibration.chain import (
    CalibrationNode,
    explicit_level,
    install_curves,
    install_market_fields,
    install_surface,
    solve_level,
)
from valax.calibration.level import LevelInputs
from valax.calibration.settings import (
    EXACT_SETTINGS,
    FIT_SETTINGS,
    SolverSettings,
)
from valax.calibration.transforms import (
    TransformSpec,
    correlation,
    positive,
)
from valax.core.market_state import MarketState, validate_surface_id
from valax.curves.bootstrap_graph import (
    curve_graph_from_log_dfs,
    curve_quote_vector,
    curve_residual_fn,
    initial_log_dfs,
    validate_curve_system,
)
from valax.curves.bootstrap_proto import BootstrapInstrument
from valax.curves.discount import flat_curve
from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveSpec
from valax.market.quotes import QuoteSet
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.surfaces.constant import ConstantVol
from valax.surfaces.sabr_surface import SABRVolSurface


# ── Tier 0: explicit market inputs ───────────────────────────────────


def market_inputs_node(
    name: str = "market_inputs",
    *,
    spots: Mapping[str, str] | None = None,
    dividends: Mapping[str, str] | None = None,
    flat_rates: Mapping[str, str] | None = None,
    flat_vols: Mapping[str, str] | None = None,
    reference_date: int | Int[Array, ""] = 0,
    day_count: str = "act_365",
) -> CalibrationNode:
    """Explicit level installing quoted scalars directly into the market.

    The constant-parameter market as a chain: a quoted spot is the spot, a
    quoted rate is a flat curve, a quoted vol is a flat surface. Quote risk
    through this node is exactly the textbook delta / rho / vega / dividend
    sensitivity.

    Args:
        name: Node name.
        spots: ``asset_id -> quote_key`` for spot levels.
        dividends: ``asset_id -> quote_key`` for continuous dividend yields.
        flat_rates: ``curve_id -> quote_key`` for flat continuously-compounded
            curves (use ``"_default_"`` for the single-curve sentinel).
        flat_vols: ``surface_id -> quote_key`` for constant vol surfaces
            (e.g. ``"SPX.VOL"`` or ``"_default_"``).
        reference_date: Valuation date ordinal of the flat curves.
        day_count: Day count of the flat curves.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.
    """
    spots = dict(spots or {})
    dividends = dict(dividends or {})
    flat_rates = dict(flat_rates or {})
    flat_vols = dict(flat_vols or {})
    keys = tuple(
        dict.fromkeys(
            [*spots.values(), *dividends.values(),
             *flat_rates.values(), *flat_vols.values()]
        )
    )

    def assemble(_theta: None, inputs: LevelInputs) -> dict:
        q = inputs.quotes
        return {
            "spots": {a: q[k] for a, k in spots.items()},
            "dividends": {a: q[k] for a, k in dividends.items()},
            "curves": {
                c: flat_curve(q[k], reference_date, day_count=day_count)
                for c, k in flat_rates.items()
            },
            "surfaces": {s: ConstantVol(vol=q[k]) for s, k in flat_vols.items()},
        }

    def calibrate_fn(inputs: LevelInputs):
        return explicit_level(name, inputs, assemble)

    return CalibrationNode(
        name=name,
        quote_keys=keys,
        fixed_keys=(),
        calibrate_fn=calibrate_fn,
        install_fn=install_market_fields,
        needs_upstream=False,
    )


# ── Tier 1/2: curve bootstrap (level 1a) ─────────────────────────────


def instrument_quotes(
    instruments: Sequence[BootstrapInstrument],
    quote_keys: Sequence[str],
) -> QuoteSet:
    """Read the quote values embedded in bootstrap instruments.

    Args:
        instruments: Bootstrap quotes (each carries its primary quote).
        quote_keys: Identifiers, one per instrument, in the same order.

    Returns:
        A :class:`~valax.market.quotes.QuoteSet`.

    Raises:
        ValueError: If the lengths differ.
    """
    if len(quote_keys) != len(instruments):
        raise ValueError(
            f"instrument_quotes: {len(quote_keys)} keys for "
            f"{len(instruments)} instruments."
        )
    return QuoteSet.from_vector(quote_keys, curve_quote_vector(instruments))


def curve_graph_node(
    name: str,
    reference_date: int | Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
    quote_keys: Sequence[str],
    *,
    fixings: FixingHistory | None = None,
    initial_guess: Mapping[str, Float[Array, " n"]] | None = None,
    settings: SolverSettings | None = None,
) -> CalibrationNode:
    """Joint single- or multi-curve bootstrap as a calibration level.

    The same residual as
    :func:`~valax.curves.bootstrap_graph.bootstrap_curve_graph`
    (:func:`~valax.curves.bootstrap_graph.curve_residual_fn`), but with the
    quotes read from the :class:`~valax.market.quotes.QuoteSet` and the eager
    validation done here, once, so the solve itself is traceable. A single
    ``CurveSpec`` is the single-curve market; several, coupled through
    ``curves_touched``, are the multi-curve one — same node.

    Args:
        name: Node name.
        reference_date: Valuation date ordinal.
        curve_specs: Curves to solve for.
        instruments: Bootstrap instruments (their embedded quote values are
            replaced by the ``QuoteSet`` values at calibration time).
        quote_keys: One quote identifier per instrument, same order.
        fixings: Optional realised-fixings registry.
        initial_guess: Optional per-curve log-DF guesses.
        settings: Solver settings; defaults to :data:`EXACT_SETTINGS`.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If the system is not square or not closed, or the number
            of keys does not match the number of instruments.
    """
    specs = tuple(curve_specs)
    insts = tuple(instruments)
    keys = tuple(quote_keys)
    validate_curve_system(specs, insts)
    if len(keys) != len(insts):
        raise ValueError(
            f"curve_graph_node({name!r}): {len(keys)} quote keys for "
            f"{len(insts)} instruments."
        )
    settings = EXACT_SETTINGS if settings is None else settings
    residual = curve_residual_fn(reference_date, specs, insts, fixings=fixings)
    theta0 = initial_log_dfs(reference_date, specs, initial_guess)

    def residual_fn(theta: Float[Array, " n"], inputs: LevelInputs):
        return residual(theta, inputs.quotes.vector(keys))

    def assemble(theta: Float[Array, " n"], _inputs: LevelInputs):
        return curve_graph_from_log_dfs(theta, specs, reference_date)

    def calibrate_fn(inputs: LevelInputs):
        return solve_level(
            name, "root", residual_fn, theta0, inputs, assemble, settings,
            labels=keys,
        )

    return CalibrationNode(
        name=name,
        quote_keys=keys,
        fixed_keys=(),
        calibrate_fn=calibrate_fn,
        install_fn=install_curves,
        needs_upstream=False,
    )


# ── Level 1b: SABR smiles on upstream forwards ───────────────────────


def equity_forward(
    asset_id: str | None = None,
    curve_id: str | None = None,
) -> Callable[[MarketState, Float[Array, ""]], Float[Array, ""]]:
    """Forward reader ``F(T) = S e^{-qT} / P(0, T)`` from the upstream market.

    Args:
        asset_id: Spot / dividend key (``None`` for the sentinel).
        curve_id: Discount curve id (``None`` for the default curve).

    Returns:
        ``forward(market, expiry) -> F``, differentiable in the market.
    """

    def forward(market: MarketState, expiry: Float[Array, ""]) -> Float[Array, ""]:
        from valax.core.binding import project_rate

        curve = market.curve(curve_id)
        rate = project_rate(curve, expiry)
        spot = market.spot(asset_id)
        dividend = market.dividend(asset_id)
        return spot * jnp.exp((rate - dividend) * expiry)

    return forward


def sabr_smile_node(
    surface_id: str,
    *,
    expiries: Float[Array, " n_exp"],
    strikes: Float[Array, "n_exp n_k"],
    quote_keys: Sequence[Sequence[str]],
    forward_fn: Callable[[MarketState, Float[Array, ""]], Float[Array, ""]],
    beta: float = 0.5,
    beta_key: str | None = None,
    weights: Float[Array, "n_exp n_k"] | None = None,
    settings: SolverSettings | None = None,
    name: str | None = None,
) -> CalibrationNode:
    r"""SABR smile per expiry, fitted on forwards read from the upstream market.

    Level 1b of the chain. Each expiry's ``(alpha, rho, nu)`` is fitted to the
    quoted lognormal vols by weighted least squares; the forwards come from
    the curves and spots already in the market, so the fit is coupled to the
    upstream levels and the adjoint carries vega *and* the forward's curve and
    spot risk. The result is a
    :class:`~valax.surfaces.sabr_surface.SABRVolSurface`.

    Beta is held fixed. Pass ``beta_key`` to read it from the fixed inputs
    ``p`` — its sensitivity is then reported by the risk functions.

    Args:
        surface_id: Key the surface is installed under (e.g. ``"SPX.VOL"``).
        expiries: Expiry grid (year fractions), sorted.
        strikes: Absolute strikes, one row per expiry.
        quote_keys: Quote identifiers, same shape as ``strikes``.
        forward_fn: ``(market, expiry) -> forward``; see
            :func:`equity_forward`.
        beta: Fixed beta when ``beta_key`` is ``None``.
        beta_key: Fixed-input identifier for beta.
        weights: Per-quote least-squares weights; defaults to ones.
        settings: Solver settings; defaults to :data:`FIT_SETTINGS`.
        name: Node name; defaults to ``surface_id``.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    expiries = jnp.asarray(expiries, dtype=jnp.float64)
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    n_exp, n_k = strikes.shape
    if expiries.shape != (n_exp,):
        raise ValueError(
            f"sabr_smile_node: {expiries.shape[0]} expiries but strikes has "
            f"{n_exp} rows."
        )
    key_grid = tuple(tuple(row) for row in quote_keys)
    if len(key_grid) != n_exp or any(len(r) != n_k for r in key_grid):
        raise ValueError("sabr_smile_node: quote_keys must match strikes' shape.")
    flat_keys = tuple(k for row in key_grid for k in row)
    w = jnp.ones_like(strikes) if weights is None else jnp.asarray(weights)
    settings = FIT_SETTINGS if settings is None else settings
    transforms: dict[str, TransformSpec] = {
        "alpha": positive(),
        "rho": correlation(),
        "nu": positive(),
    }

    def _beta(inputs: LevelInputs) -> Float[Array, ""]:
        if beta_key is None:
            return jnp.asarray(beta)
        return inputs.fixed[beta_key]

    def _forwards(inputs: LevelInputs) -> Float[Array, " n_exp"]:
        return jnp.stack([forward_fn(inputs.upstream, t) for t in expiries])

    def _params(theta: dict) -> dict:
        return {k: transforms[k].from_unconstrained(v) for k, v in theta.items()}

    def residual_fn(theta: dict, inputs: LevelInputs) -> Float[Array, " n"]:
        p = _params(theta)
        b = _beta(inputs)
        fwd = _forwards(inputs)
        market_vols = inputs.quotes.vector(flat_keys).reshape(n_exp, n_k)

        def slice_vols(alpha, rho, nu, f, t, ks):
            m = SABRModel(alpha=alpha, beta=b, rho=rho, nu=nu)
            return jax.vmap(lambda k: sabr_implied_vol(m, f, k, t))(ks)

        model_vols = jax.vmap(slice_vols)(
            p["alpha"], p["rho"], p["nu"], fwd, expiries, strikes
        )
        return (w * (model_vols - market_vols)).reshape(-1)

    def assemble(theta: dict, inputs: LevelInputs) -> SABRVolSurface:
        p = _params(theta)
        b = _beta(inputs)
        return SABRVolSurface(
            expiries=expiries,
            forwards=_forwards(inputs),
            alphas=p["alpha"],
            betas=jnp.full((n_exp,), b),
            rhos=p["rho"],
            nus=p["nu"],
        )

    def initial_theta(inputs: LevelInputs) -> dict:
        b = _beta(inputs)
        fwd = _forwards(inputs)
        market_vols = inputs.quotes.vector(flat_keys).reshape(n_exp, n_k)
        atm = jnp.median(market_vols, axis=1)
        alpha0 = atm * fwd ** (1.0 - b)
        guess = {
            "alpha": alpha0,
            "rho": jnp.full((n_exp,), -0.2),
            "nu": jnp.full((n_exp,), 0.4),
        }
        # The guess only seeds the solver; keep it out of the gradient.
        return jax.lax.stop_gradient(
            {k: transforms[k].to_unconstrained(v) for k, v in guess.items()}
        )

    def calibrate_fn(inputs: LevelInputs):
        return solve_level(
            name or surface_id,
            "least_squares",
            residual_fn,
            initial_theta(inputs),
            inputs,
            assemble,
            settings,
        )

    return CalibrationNode(
        name=name or surface_id,
        quote_keys=flat_keys,
        fixed_keys=() if beta_key is None else (beta_key,),
        calibrate_fn=calibrate_fn,
        install_fn=install_surface(surface_id),
        needs_upstream=True,
    )


# ── Level 1b (rates): swaption cube and cap strip on upstream curves ──


def _curves_reader(
    discount_id: str | None, forward_id: str | None
) -> Callable[[MarketState], tuple]:
    """``market -> (discount_curve, forward_curve | None)``."""

    def read(market: MarketState) -> tuple:
        forward = None if forward_id is None else market.curve(forward_id)
        return market.curve(discount_id), forward

    return read


def _padded_keys(
    key_rows: Sequence[Sequence[str]], width: int
) -> tuple[str, ...]:
    """Flatten ragged per-node key rows, padding each by its last key.

    Matches :func:`~valax.calibration.solve.stack_slices`, which pads the
    numbers the same way with zero weight — so a padded quote is read but
    contributes (and receives) nothing.
    """
    out: list[str] = []
    for row in key_rows:
        row = tuple(row)
        out.extend(row + (row[-1],) * (width - len(row)))
    return tuple(out)


def swaption_cube_node(
    surface_id: str,
    *,
    swaptions: Sequence[Sequence[Any]],
    strikes_per_node: Sequence[Sequence[Float[Array, " n_k"]]],
    quote_keys: Sequence[Sequence[Sequence[str]]],
    discount_id: str | None,
    reference_date: int | Int[Array, ""],
    forward_id: str | None = None,
    quote_form: str = "absolute_strike",
    dynamics: str | None = None,
    is_normal: bool = False,
    shift: float = 0.0,
    beta: float | None = 0.5,
    beta_key: str | None = None,
    weights_per_node: Sequence[Sequence[Float[Array, " n_k"]]] | None = None,
    settings: SolverSettings | None = None,
    name: str | None = None,
) -> CalibrationNode:
    r"""SABR swaption cube on the upstream curves, as a chain level.

    The rates counterpart of :func:`sabr_smile_node`: one SABR smile per
    ``(expiry, tenor)`` cell, fitted to quoted vols at the node's strikes,
    with each cell's forward par swap rate recomputed *inside* the level
    from the curves already in the market (the pricer's own
    :func:`~valax.pricing.analytic.swaptions.swaption_forward_and_annuity`).
    The fit is therefore a differentiable function of the curve pillars, and
    :func:`~valax.greeks.quote_risk.chain_adjoint` pushes a swaption's vega
    to the vol quotes *and* the forward's curve risk to the curve quotes —
    curve → vol → price in one sweep. Shares its implementation with
    :func:`~valax.surfaces.swaption_cube.calibrate_swaption_cube_from_curves`
    (:func:`~valax.surfaces.swaption_cube.swaption_cube_level`); cells are
    solved independently and a failed cell is named (``"5Yx10Y"``).

    Args:
        surface_id: Key the cube is installed under (e.g. ``"USD.SWPN.VOL"``).
        swaptions: Nested ``[i][j]`` representative swaptions (schedules
            define the axes and forwards; strikes ignored).
        strikes_per_node: Nested ``[i][j]`` strike arrays — absolute, or
            offsets to the node forward under ``quote_form="atm_offset"``.
        quote_keys: Nested ``[i][j]`` sequences of quote identifiers, one per
            strike.
        discount_id: Discount curve id in the upstream market.
        reference_date: Valuation date ordinal, used eagerly to derive the
            cube's expiry / tenor axes.
        forward_id: Projection curve id (``None`` for single-curve).
        quote_form: ``"absolute_strike"`` or ``"atm_offset"``.
        dynamics: Smile dynamics recorded on the cube.
        is_normal: Normal (Bachelier) quotes if True, else lognormal.
        shift: Displacement of the normal expansion.
        beta: Fixed SABR beta, or ``None`` to calibrate beta per cell.
        beta_key: Fixed-input identifier for beta (overrides ``beta``); its
            sensitivity is reported.
        weights_per_node: Optional nested per-strike weights.
        settings: Per-cell solver settings.
        name: Node name; defaults to ``surface_id``.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If the grid is not rectangular, the key / strike grids
            disagree, or the axes are inconsistent.
    """
    from valax.calibration.solve import stack_slices
    from valax.surfaces.swaption_cube import (
        _validate_node_grid,
        swaption_cube_level,
        swaption_node_coordinates,
    )

    validate_surface_id(surface_id)
    _validate_node_grid(swaptions)
    n_e, n_t = len(swaptions), len(swaptions[0])
    flat_strikes = [jnp.asarray(k) for row in strikes_per_node for k in row]
    key_rows = [tuple(keys) for row in quote_keys for keys in row]
    if len(flat_strikes) != n_e * n_t or len(key_rows) != n_e * n_t:
        raise ValueError(
            f"swaption_cube_node: strikes_per_node and quote_keys must match "
            f"the ({n_e}, {n_t}) node grid."
        )
    for i, (k, keys) in enumerate(zip(flat_strikes, key_rows)):
        if k.shape[0] != len(keys):
            raise ValueError(
                f"swaption_cube_node: node {i} has {k.shape[0]} strikes but "
                f"{len(keys)} quote keys."
            )
    strikes, _, weights = stack_slices(
        flat_strikes, flat_strikes,
        None if weights_per_node is None
        else [w for row in weights_per_node for w in row],
    )
    width = strikes.shape[1]
    padded = _padded_keys(key_rows, width)
    keys = tuple(dict.fromkeys(padded))
    expiries, tenors = swaption_node_coordinates(
        swaptions, flat_curve(0.0, reference_date)
    )
    curves = _curves_reader(discount_id, forward_id)
    node_name = name or surface_id

    if beta_key is not None:
        beta_fn = lambda inp: inp.fixed[beta_key]  # noqa: E731
    elif beta is not None:
        beta_fn = lambda inp: jnp.asarray(beta)  # noqa: E731
    else:
        beta_fn = None

    def calibrate_fn(inputs: LevelInputs):
        return swaption_cube_level(
            node_name, swaptions, inputs, curves,
            strikes=strikes, weights=weights,
            vols_fn=lambda inp: inp.quotes.vector(padded).reshape(
                n_e * n_t, width
            ),
            beta_fn=beta_fn,
            expiries=expiries, tenors=tenors, quote_form=quote_form,
            dynamics=dynamics, is_normal=is_normal, shift=shift,
            settings=settings,
        )

    return CalibrationNode(
        name=node_name,
        quote_keys=keys,
        fixed_keys=() if beta_key is None else (beta_key,),
        calibrate_fn=calibrate_fn,
        install_fn=install_surface(surface_id),
        needs_upstream=True,
    )


def caplet_strip_node(
    strip_id: str,
    *,
    caps: Sequence[Any],
    strikes: Float[Array, " n_strikes"],
    quote_keys: Sequence[Sequence[str]],
    discount_id: str | None,
    reference_date: int | Int[Array, ""],
    forward_id: str | None = None,
    is_normal: bool = False,
    settings: SolverSettings | None = None,
    name: str | None = None,
) -> CalibrationNode:
    r"""Cap-matrix strip on the upstream curves, as a chain level.

    Strips every strike column of a flat-vol cap matrix into per-optionlet
    vols (:func:`~valax.surfaces.caplet_stripping.strip_columns_level`: one
    sequential bootstrap per column, ``vmap``-ed, each column reporting its
    own convergence). Forwards are projected and payoffs discounted off the
    curves already in the market, so the stripped vols — and everything
    fitted on them — carry the curves' risk. The batched
    :class:`~valax.surfaces.caplet_stripping.CapletVolCurve` is installed
    under ``strip_id`` for a downstream :func:`optionlet_sabr_node`.

    Args:
        strip_id: Surface key the strip is installed under.
        caps: Nested cap schedule, ascending in maturity (strikes ignored).
        strikes: Quoted strike grid (one strip column per strike).
        quote_keys: ``[k][i]`` quote identifiers — strike ``k``, cap ``i``.
        discount_id: Discount curve id in the upstream market.
        reference_date: Valuation date ordinal, used eagerly to reject
            optionlets that have already fixed.
        forward_id: Projection curve id (``None`` for single-curve).
        is_normal: Bachelier (True) or Black-76 (False) quotes.
        settings: Per-column solver settings.
        name: Node name; defaults to ``strip_id``.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If ``quote_keys`` does not match ``(len(strikes),
            len(caps))`` or the strip is invalid.
    """
    from valax.surfaces.caplet_stripping import (
        _validate_strip,
        strip_columns_level,
    )

    validate_surface_id(strip_id)
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    n_k, n_c = int(strikes.shape[0]), len(caps)
    key_grid = tuple(tuple(row) for row in quote_keys)
    if len(key_grid) != n_k or any(len(r) != n_c for r in key_grid):
        raise ValueError(
            f"caplet_strip_node: quote_keys must have shape ({n_k}, {n_c})."
        )
    for k in range(n_k):
        struck = [eqx.tree_at(lambda c: c.strike, cap, strikes[k]) for cap in caps]
        _validate_strip(
            struck, flat_curve(0.0, reference_date), jnp.ones(n_c), None
        )
    flat_keys = tuple(k for row in key_grid for k in row)
    curves = _curves_reader(discount_id, forward_id)
    node_name = name or strip_id

    def calibrate_fn(inputs: LevelInputs):
        return strip_columns_level(
            node_name, tuple(caps), inputs, curves, strikes=strikes,
            quotes_fn=lambda inp: inp.quotes.vector(flat_keys).reshape(n_k, n_c),
            is_normal=is_normal, settings=settings,
        )

    return CalibrationNode(
        name=node_name,
        quote_keys=flat_keys,
        fixed_keys=(),
        calibrate_fn=calibrate_fn,
        install_fn=install_surface(strip_id),
        needs_upstream=True,
    )


def optionlet_sabr_node(
    surface_id: str,
    *,
    strip_id: str,
    beta: float | None = 0.5,
    beta_key: str | None = None,
    is_normal: bool = False,
    shift: float = 0.0,
    dynamics: str = "sticky_strike",
    settings: SolverSettings | None = None,
    name: str | None = None,
) -> CalibrationNode:
    r"""SABR optionlet surface fitted on an upstream cap strip.

    The second rates-vol level of the cap chain: one SABR smile per
    optionlet expiry, fitted to the vols the upstream
    :func:`caplet_strip_node` installed under ``strip_id`` (strike axis = the
    strip's strikes), anchored on the strip's curve-projected forwards. The
    level has **no quotes of its own** — its inputs are entirely upstream —
    so :func:`~valax.greeks.quote_risk.chain_adjoint` pulls a caplet's vega
    through it into the strip level and from there onto the cap quotes and
    the curves. Expiries are fitted independently and named on failure.

    Args:
        surface_id: Key the :class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`
            is installed under.
        strip_id: Surface key of the upstream batched strip.
        beta: Fixed SABR beta, or ``None`` to calibrate it per expiry.
        beta_key: Fixed-input identifier for beta (overrides ``beta``).
        is_normal: Normal (Bachelier) strip vols if True, else lognormal.
        shift: Displacement of the normal SABR expansion.
        dynamics: Smile dynamics recorded on the surface.
        settings: Per-expiry solver settings.
        name: Node name; defaults to ``surface_id``.

    Returns:
        A market-scope :class:`~valax.calibration.chain.CalibrationNode`.
    """
    from valax.calibration.sabr import sabr_batch_level, sabr_vol_fn
    from valax.calibration.solve import period_labels
    from valax.surfaces.optionlet_surface import OptionletVolSurface

    validate_surface_id(surface_id)
    validate_surface_id(strip_id)
    node_name = name or surface_id

    def _beta(inp: LevelInputs) -> Float[Array, ""]:
        if beta_key is not None:
            return inp.fixed[beta_key]
        return jnp.asarray(0.5 if beta is None else beta)

    def node_inputs(inp: LevelInputs) -> LevelInputs:
        strip = inp.upstream.surfaces[strip_id]
        vols = strip.vols.T                      # (n_expiries, n_strikes)
        n_exp, n_k = vols.shape
        fixed = {
            "forward": strip.forwards[0],
            "expiry": strip.expiries[0],
            "strikes": jnp.broadcast_to(strip.strike, (n_exp, n_k)),
            "weights": jnp.ones((n_exp, n_k)),
        }
        if beta is not None or beta_key is not None:
            fixed["beta"] = jnp.broadcast_to(_beta(inp), (n_exp,))
        return LevelInputs(quotes=vols, fixed=fixed)

    def value_fn(models, nodes, _inp) -> OptionletVolSurface:
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
        )

    def calibrate_fn(inputs: LevelInputs):
        expiries = inputs.upstream.surfaces[strip_id].expiries[0]
        return sabr_batch_level(
            node_name, inputs, node_inputs, value_fn,
            labels=period_labels(expiries),
            fit_beta=beta is None and beta_key is None,
            vol_fn=sabr_vol_fn(is_normal, shift),
            is_normal=is_normal,
            settings=settings,
        )

    return CalibrationNode(
        name=node_name,
        quote_keys=(),
        fixed_keys=() if beta_key is None else (beta_key,),
        calibrate_fn=calibrate_fn,
        install_fn=install_surface(surface_id),
        needs_upstream=True,
    )


# ── Any registered model, fitted through price() ─────────────────────


#: Quote forms understood by :func:`price_fit_node` by name.
QUOTE_FORMS = ("price", "lognormal_vol")


def _target_price(
    quote_form: str | Callable[[Any, Float[Array, ""], MarketState], Float[Array, ""]],
    instrument: Any,
    quote: Float[Array, ""],
    market: MarketState,
) -> Float[Array, ""]:
    """Convert one quote to the price it implies for its instrument."""
    from valax.core.pricing import price

    if callable(quote_form):
        return quote_form(instrument, quote, market)
    if quote_form == "price":
        return quote
    # "lognormal_vol": Black-Scholes at the quoted vol; rate and dividend are
    # bound from the market exactly as for any other model.
    from valax.models.black_scholes import BlackScholesModel

    return price(instrument, BlackScholesModel(vol=quote), market).price


def price_fit_node(
    name: str,
    *,
    template: eqx.Module,
    transforms: Mapping[str, TransformSpec],
    basket: Sequence[Any],
    quote_keys: Sequence[str],
    quote_form: str | Callable = "price",
    weights: Float[Array, " n"] | None = None,
    fixed_fields: Mapping[str, str] | None = None,
    install_fn: Callable[[MarketState, Any], MarketState] | None = None,
    method: str = "auto",
    config: Any = None,
    settings: SolverSettings | None = None,
) -> CalibrationNode:
    r"""Fit any model with a registered pricer to a basket, through ``price``.

    The generic calibration level: residual
    :math:`r_k = w_k\,[\,\mathrm{price}(I_k, M(\theta), \text{market}) - T_k(q_k)\,]`,
    where :math:`T_k` turns quote :math:`q_k` into a target price
    (``quote_form``). Because it reuses the pricer registry, calibration and
    pricing share one implementation, and any (instrument, model, method)
    triple reachable from :func:`valax.core.pricing.price` can be calibrated
    with no new code.

    The same factory serves both scopes: leave ``install_fn=None`` for a
    model-scope node (level 2, e.g. Heston to an equity surface, Hull-White to
    a co-terminal basket) or pass one to install the fitted object into the
    market (level 1b, e.g. converting a premium quote into a vol).

    Args:
        name: Node name.
        template: Model instance providing the structure and every field not
            calibrated (fields left ``None`` are bound from the market at
            pricing time, as usual).
        transforms: ``field -> TransformSpec`` for the calibrated fields; the
            solver works in the unconstrained space.
        basket: Calibration instruments.
        quote_keys: One quote identifier per basket instrument.
        quote_form: ``"price"`` (quote is the premium), ``"lognormal_vol"``
            (quote is a Black-Scholes implied vol, converted with the bound
            market rate and dividend), or a callable
            ``(instrument, quote, market) -> price``.
        weights: Per-instrument weights; defaults to ones. Use ``1/vega`` to
            make price residuals vol-like.
        fixed_fields: ``field -> fixed-input key``: model fields held fixed at
            a value read from ``p`` (their sensitivities are reported).
        install_fn: Market installer for a market-scope node; ``None`` for a
            model-scope node.
        method: Pricing method for the basket.
        config: Pricing config for the basket.
        settings: Solver settings; defaults to :data:`FIT_SETTINGS`.

    Returns:
        A :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If the basket and keys differ in length, or the quote
            form is unknown.
    """
    basket = tuple(basket)
    keys = tuple(quote_keys)
    if len(basket) != len(keys):
        raise ValueError(
            f"price_fit_node({name!r}): {len(keys)} keys for "
            f"{len(basket)} instruments."
        )
    if not callable(quote_form) and quote_form not in QUOTE_FORMS:
        raise ValueError(
            f"price_fit_node({name!r}): unknown quote_form {quote_form!r}; "
            f"use one of {QUOTE_FORMS} or a callable."
        )
    transforms = dict(transforms)
    fixed_fields = dict(fixed_fields or {})
    w = jnp.ones((len(basket),)) if weights is None else jnp.asarray(weights)
    settings = FIT_SETTINGS if settings is None else settings
    theta0 = {
        f: transforms[f].to_unconstrained(jnp.asarray(getattr(template, f)))
        for f in transforms
    }

    def assemble(theta: dict, inputs: LevelInputs) -> Any:
        model = template
        for f, spec in transforms.items():
            model = eqx.tree_at(
                lambda m, _f=f: getattr(m, _f), model,
                spec.from_unconstrained(theta[f]),
            )
        for f, key in fixed_fields.items():
            model = eqx.tree_at(
                lambda m, _f=f: getattr(m, _f), model, inputs.fixed[key],
                is_leaf=lambda x: x is None,
            )
        return model

    def residual_fn(theta: dict, inputs: LevelInputs) -> Float[Array, " n"]:
        from valax.core.pricing import price

        model = assemble(theta, inputs)
        market = inputs.upstream
        r = []
        for inst, key in zip(basket, keys):
            model_price = price(
                inst, model, market, method=method, config=config
            ).price
            target = _target_price(quote_form, inst, inputs.quotes[key], market)
            r.append(model_price - target)
        return w * jnp.stack(r)

    def calibrate_fn(inputs: LevelInputs):
        return solve_level(
            name, "least_squares", residual_fn, theta0, inputs, assemble,
            settings,
        )

    return CalibrationNode(
        name=name,
        quote_keys=keys,
        fixed_keys=tuple(dict.fromkeys(fixed_fields.values())),
        calibrate_fn=calibrate_fn,
        install_fn=install_fn,
        needs_upstream=True,
    )


# ── Level 2 (rates): per-trade model fitted to a swaption basket ─────


def coterminal_swaptions(
    bermudan: Any,
    *,
    strike: Float[Array, ""] | None = None,
) -> tuple:
    """The co-terminal European basket of a Bermudan swaption (eager).

    One European swaption per exercise date, each exercising into the
    remaining swap (the Bermudan's fixed schedule after that date) — the
    standard calibration basket of a one-factor model for a Bermudan, whose
    exercise value at each date is exactly such a swap.

    Args:
        bermudan: A :class:`~valax.instruments.rates.BermudanSwaption` with
            concrete dates.
        strike: Basket strike; defaults to the Bermudan's strike.

    Returns:
        One :class:`~valax.instruments.rates.Swaption` per exercise date that
        leaves at least one coupon, in exercise order.
    """
    import numpy as np

    from valax.instruments.rates import Swaption

    fixed = np.asarray(bermudan.fixed_dates)
    basket = []
    for e in np.asarray(bermudan.exercise_dates):
        k = int(np.searchsorted(fixed, e, side="right"))
        if k >= fixed.shape[0]:
            continue
        basket.append(Swaption(
            expiry_date=jnp.asarray(e, dtype=jnp.int32),
            fixed_dates=jnp.asarray(fixed[k:], dtype=jnp.int32),
            strike=bermudan.strike if strike is None else jnp.asarray(strike),
            notional=bermudan.notional,
            is_payer=bermudan.is_payer,
            day_count=bermudan.day_count,
            discount_id=bermudan.discount_id,
            forward_id=bermudan.forward_id,
        ))
    return tuple(basket)


def swaption_basket_node(
    name: str,
    *,
    template: eqx.Module,
    basket: Sequence[Any],
    cube_id: str,
    transforms: Mapping[str, TransformSpec],
    fixed_fields: Mapping[str, str] | None = None,
    target_keys: Sequence[str] | None = None,
    weights: Float[Array, " n"] | None = None,
    relative: bool = True,
    method: str = "analytic",
    config: Any = None,
    settings: SolverSettings | None = None,
) -> CalibrationNode:
    r"""A pricing model fitted per trade to a swaption basket, off the cube.

    Level 2 of the rates chain: the model (e.g. a curve-free
    :class:`~valax.models.hull_white.HullWhiteModel`) is fitted to the
    basket's prices *implied by the calibrated market* — each target
    :math:`T_k` is the basket swaption priced off the cube installed under
    ``cube_id`` and the curves (the smile-aware
    :func:`~valax.pricing.analytic.rates_smile.swaption_price`) — and each
    model price goes through :func:`valax.core.pricing.price` (``method``,
    the exact Jamshidian route for Hull-White by default), which binds the
    model's curve from the same market:

    .. math::

        r_k(\theta) = w_k\,\frac{\mathrm{price}(I_k, M(\theta),
        \text{market}) - T_k(\text{market}) - s_k}{\max(|T_k|, 10^{-8})}

    The level reads the market only (its upstream), so
    :func:`~valax.greeks.quote_risk.chain_adjoint` pushes the trade's model
    sensitivity :math:`\lambda_m` through the cube level onto the vol quotes
    and through the forwards / discounting onto the curve quotes.

    Passing ``target_keys`` adds one held-fixed **target shift** :math:`s_k`
    per basket instrument (value ``0``): its reported fixed-input
    sensitivity :math:`\bar p_k = \partial V / \partial T_k` is the
    trade's exposure to each basket instrument's price — the **vega
    profile across the basket**, available on both differentiation paths
    and checkable by bumping the shift.

    Baskets may be traced (closed over inside ``eqx.filter_vmap``), so
    same-shaped per-trade baskets batch into one calibration.

    Args:
        name: Node name.
        template: Model providing structure and every field not calibrated;
            leave its curve unbound (``price`` binds it).
        basket: Calibration swaptions (e.g. :func:`coterminal_swaptions`).
        cube_id: Surface id of the calibrated swaption cube.
        transforms: ``field -> TransformSpec`` of the calibrated fields.
        fixed_fields: ``field -> fixed-input key`` held fixed at ``p``.
        target_keys: One fixed-input key per basket instrument for the
            target shifts (pass zeros in ``fixed``); ``None`` for none.
        weights: Per-instrument weights; default ones.
        relative: Divide each residual by the target price.
        method: Pricing method of the basket under the model.
        config: Pricing config of the basket.
        settings: Solver settings; defaults to
            :data:`~valax.calibration.basket.BASKET_SETTINGS`.

    Returns:
        A model-scope :class:`~valax.calibration.chain.CalibrationNode`.

    Raises:
        ValueError: If ``target_keys`` does not match the basket length.
    """
    from valax.calibration.basket import BASKET_SETTINGS

    basket = tuple(basket)
    n = len(basket)
    targets_keyed = tuple(target_keys) if target_keys is not None else ()
    if target_keys is not None and len(targets_keyed) != n:
        raise ValueError(
            f"swaption_basket_node({name!r}): {len(targets_keyed)} target keys "
            f"for {n} basket instruments."
        )
    transforms = dict(transforms)
    fixed_fields = dict(fixed_fields or {})
    w = jnp.ones((n,)) if weights is None else jnp.asarray(weights)
    settings = BASKET_SETTINGS if settings is None else settings
    theta0 = {
        f: transforms[f].to_unconstrained(jnp.asarray(getattr(template, f)))
        for f in transforms
    }

    def assemble(theta: dict, inputs: LevelInputs) -> Any:
        model = template
        for f, spec in transforms.items():
            model = eqx.tree_at(
                lambda m, _f=f: getattr(m, _f), model,
                spec.from_unconstrained(theta[f]),
            )
        for f, key in fixed_fields.items():
            model = eqx.tree_at(
                lambda m, _f=f: getattr(m, _f), model, inputs.fixed[key],
                is_leaf=lambda x: x is None,
            )
        return model

    def targets(inputs: LevelInputs) -> Float[Array, " n"]:
        from valax.core.resolve import discount_id_of, forward_id_of
        from valax.pricing.analytic.rates_smile import swaption_price

        market = inputs.upstream
        cube = market.surfaces[cube_id]
        out = []
        for k, inst in enumerate(basket):
            fwd_id = forward_id_of(inst)
            t = swaption_price(
                inst, market.curve(discount_id_of(inst)), cube,
                None if fwd_id is None else market.curve(fwd_id),
            )
            if targets_keyed:
                t = t + inputs.fixed[targets_keyed[k]]
            out.append(t)
        return jnp.stack(out)

    def residual_fn(theta: dict, inputs: LevelInputs) -> Float[Array, " n"]:
        from valax.core.pricing import price

        model = assemble(theta, inputs)
        market = inputs.upstream
        prices = jnp.stack([
            price(inst, model, market, method=method, config=config).price
            for inst in basket
        ])
        t = targets(inputs)
        scale = jnp.maximum(jnp.abs(t), 1e-8) if relative else jnp.ones_like(t)
        return w * (prices - t) / scale

    def calibrate_fn(inputs: LevelInputs):
        return solve_level(
            name, "least_squares", residual_fn, theta0, inputs, assemble,
            settings, labels=targets_keyed,
        )

    return CalibrationNode(
        name=name,
        quote_keys=(),
        fixed_keys=tuple(dict.fromkeys([*fixed_fields.values(), *targets_keyed])),
        calibrate_fn=calibrate_fn,
        install_fn=None,
        needs_upstream=True,
    )


__all__ = [
    "EXACT_SETTINGS",
    "FIT_SETTINGS",
    "QUOTE_FORMS",
    "curve_graph_node",
    "equity_forward",
    "instrument_quotes",
    "market_inputs_node",
    "price_fit_node",
    "caplet_strip_node",
    "coterminal_swaptions",
    "optionlet_sabr_node",
    "sabr_smile_node",
    "swaption_basket_node",
    "swaption_cube_node",
]
