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
* :func:`price_fit_node` — any model with a registered pricer, fitted to a
  basket through :func:`valax.core.pricing.price` (level 1b quote conversion
  or level 2 model calibration — the same factory serves both scopes).
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
from valax.calibration.settings import SolverSettings
from valax.calibration.transforms import (
    TransformSpec,
    correlation,
    positive,
)
from valax.core.market_state import MarketState
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


#: Default settings of an exact (square) level: Newton to 1e-10.
EXACT_SETTINGS = SolverSettings(method="newton", rtol=1e-10, atol=1e-10)

#: Default settings of a least-squares level: Levenberg-Marquardt to 1e-8.
#: Looser than :data:`EXACT_SETTINGS` on purpose: a fit's residual does not
#: vanish, and price-based fits (Fourier / numerical pricers) have a noise
#: floor near 1e-10 that a tighter step tolerance cannot get under — the
#: solver would report non-convergence at a perfectly good optimum.
FIT_SETTINGS = SolverSettings(
    method="levenberg_marquardt", rtol=1e-8, atol=1e-8, max_steps=512
)


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
            name, "root", residual_fn, theta0, inputs, assemble, settings
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


__all__ = [
    "EXACT_SETTINGS",
    "FIT_SETTINGS",
    "QUOTE_FORMS",
    "curve_graph_node",
    "equity_forward",
    "instrument_quotes",
    "market_inputs_node",
    "price_fit_node",
    "sabr_smile_node",
]
