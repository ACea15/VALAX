"""Repricing and P&L over a :class:`MarketState`, through ``price(...)``.

The counterpart of :mod:`valax.risk.var` for the container the unified
pricer takes.  Where ``var.py`` takes a caller-supplied
``pricing_fn(instrument, MarketData)`` and bridges the loose-scalar
convention through its own adapter, every function here goes through the
one front door, :func:`valax.core.pricing.price`, so a risk number and a
front-office price are produced by *the same call*.  That is what makes
"every Greek is one ``jax.grad`` through one function" true rather than
aspirational (``PRICING_ONTOLOGY.md`` § 0.1).

Homogeneous groups (the unit of work)
-------------------------------------
``price`` dispatches on ``(type(instrument), type(model), method)``, so
one ``vmap`` covers one **homogeneous group** and a heterogeneous book is
a sum over groups.  The grouping key is stricter than "instrument class":

    (instrument type, its static field values, model type and its
     statics, method, config)

``EuropeanOption.is_call`` is ``eqx.field(static=True)``, so calls and
puts cannot share a stacked pytree — a group is exactly what
:func:`valax.market.synthetic.portfolio.sample_option_portfolio` already
returns separately.  ``method`` and ``config`` are keyword arguments here
rather than fields of a group object: they mirror ``price``'s own
signature, and reifying the group as a pytree is the *aggregation*
layer's decision, not this module's.

Three additive quantities, and no risk measure
----------------------------------------------
The primitives return the three things that aggregate across groups by
addition — the scalar PV (:func:`group_pv`), the gradient *pytree*
(:func:`group_market_gradient`), and the ``(n_scenarios,)`` P&L vector
(:func:`group_scenario_pnl`).  Deliberately **not** a VaR number: a
quantile of a sum is not a sum of quantiles, so
:func:`valax.risk.var.value_at_risk` / ``expected_shortfall`` and the
FRTB PLA tests apply to the *aggregated* P&L vector, at the top only.

Because every group differentiates w.r.t. the same ``MarketState``
pytree, per-group gradients share a structure and combine with
``jax.tree_util.tree_map(operator.add, ...)`` — aggregation happens in
the market's own coordinate system, which is why
:mod:`valax.risk.bucketing` stays a coordinate change applied *after*
aggregation rather than a prerequisite for it.

Common random numbers (MC groups)
---------------------------------
``key`` is threaded, unchanged, into the base valuation and into every
shocked revaluation.  Drawing a fresh key per revaluation would make an
MC group's scenario P&L pure sampling noise — analytic pricers have
hidden this so far.  :func:`group_scenario_pnl` therefore closes over one
key for the whole call; pass a different key per *group*, never per
scenario.

Quantity and sign are not here
------------------------------
``price`` returns a *unit* price and instruments are data-only, so
:func:`group_prices` is the per-instrument vector a position-aware layer
weights.  Nothing in this module carries a position size.

See also:
    :mod:`valax.risk.var` — the legacy ``MarketData`` / ``MarketScenario``
    path, frozen for one deprecation cycle.  It cannot be re-expressed in
    terms of this module: an opaque ``pricing_fn`` carries no model, and
    ``price`` needs one.
"""

import warnings
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.market.keyed_scenario import KeyedScenario
from valax.risk.state_shocks import shock


# ── Group validation (pre-trace, user boundary) ──────────────────────


def group_size(instruments: Any) -> int:
    """Number of instruments in a stacked group, validated.

    A group is one instrument pytree whose array leaves all carry the
    same leading batch axis.  Checked at the user boundary so a badly
    formed group fails with a readable message instead of a tracer error
    from inside ``vmap``.

    Uniformity of the *static* fields is not checked here because it
    cannot fail here: differing statics change the pytree treedef, so
    such instruments cannot be stacked into one pytree in the first
    place.  That is the grouping key doing its job upstream.

    Args:
        instruments: Batched instrument pytree with a leading batch dim.

    Returns:
        The group size ``n_instruments``.

    Raises:
        ValueError: If the pytree has no array leaves, holds a scalar
            leaf (the signature of a single unstacked instrument), or
            its leaves disagree on the leading axis.
    """
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(instruments)
              if eqx.is_array(leaf)]
    if not leaves:
        raise ValueError(
            "group_size: the instrument pytree has no array leaves, so it "
            "carries no batch axis. Stack a group with "
            "jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *instruments)."
        )

    scalar = [leaf for leaf in leaves if jnp.ndim(leaf) == 0]
    if scalar:
        raise ValueError(
            f"group_size: {len(scalar)} leaf/leaves of the instrument "
            f"pytree are scalars, which means this is a single instrument "
            f"rather than a stacked group. Price it with "
            f"valax.core.price directly, or stack it with "
            f"jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *instruments)."
        )

    sizes = {int(jnp.shape(leaf)[0]) for leaf in leaves}
    if len(sizes) != 1:
        raise ValueError(
            f"group_size: instrument leaves disagree on the leading batch "
            f"axis (found sizes {sorted(sizes)}). Every leaf of a stacked "
            f"group must share one n_instruments axis."
        )
    return sizes.pop()


#: Scenario field → the model field that shadows it.  A model carrying its
#: own scalar for one of these reads *that* value, not the market's, so a
#: shock to the corresponding market object cannot reach the price.
_SHADOWED_BY = (
    ("curve_shocks", "rate", "the discount curve"),
    ("vol_shocks", "vol", "the vol surface"),
    ("dividend_shocks", "dividend", "the dividend yield"),
)


def _shadowed_shocks(model: Any, scenario: KeyedScenario) -> list[str]:
    """Scenario fields whose shocks the model's own parameters shadow.

    Detected structurally, from the model pytree's fields — not guessed.
    A model that carries an intrinsic scalar ``rate`` / ``vol`` /
    ``dividend`` (the equity family: ``BlackScholesModel``,
    ``HestonModel``, ``LocalVolModel``, ``SLVModel``,
    ``MultiAssetGBMModel``) reads that parameter, so the corresponding
    market object is not an input to the price.  A model that references
    the market by id instead (``HullWhiteModel`` / ``G2PPModel``, both
    ``curve_id``) or no model at all (curve-discounted) is unaffected.

    Args:
        model: The group's model pytree, or ``None``.
        scenario: The keyed scenario about to be applied.

    Returns:
        Human-readable descriptions of the shadowed shocks, empty when
        every shock in the scenario can reach the price.
    """
    if model is None:
        return []
    shadowed = []
    for field, param, what in _SHADOWED_BY:
        if not getattr(scenario, field):
            continue
        if getattr(model, param, None) is not None:
            shadowed.append(
                f"{field} (shadowed by {type(model).__name__}.{param}, so "
                f"{what} is not an input to this price)"
            )
    return shadowed


def _warn_shadowed_shocks(
    model: Any,
    scenario: KeyedScenario,
    caller: str,
) -> None:
    """Warn that a shock cannot reach the price it was aimed at.

    Called at the user boundary (pre-trace) so a group whose model
    shadows the market cannot report a silent **zero** P&L for a shock
    that was meant to move it.  The same failure class as the dropped
    ``curve_graph`` that :func:`valax.risk.var._warn_frozen_graph_curves`
    guards on the legacy path: a plausible number, no error.

    The shadowed risk is not lost — it is on the *model* gradient
    (:func:`group_model_gradient`) rather than the market gradient. What
    removes the split entirely is full curve de-embedding
    (``PRICING_ONTOLOGY.md`` § 0.3).

    Args:
        model: The group's model pytree, or ``None``.
        scenario: The keyed scenario about to be applied.
        caller: Name of the calling entry point, quoted in the message.
    """
    shadowed = _shadowed_shocks(model, scenario)
    if not shadowed:
        return
    warnings.warn(
        f"{caller}: this group's model carries its own parameters, so "
        f"{len(shadowed)} of the scenario's shock families cannot reach "
        f"the price and contribute exactly zero P&L — "
        f"{'; '.join(shadowed)}. That risk is on the model, not the "
        f"market: use group_model_gradient for it, or build the model "
        f"from the market so the shock flows through.",
        UserWarning,
        stacklevel=3,
    )


# ── Group valuation ──────────────────────────────────────────────────


def group_prices(
    instruments: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, " n_instruments"]:
    """Unit prices of one homogeneous group, one ``vmap`` over ``price``.

    The per-instrument vector a position-aware layer weights by quantity
    and sign; :func:`group_pv` is its sum.  ``market``, ``model`` and
    ``key`` are shared across the group, so an MC group prices every
    instrument on the *same* paths.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None`` for
            curve-discounted instruments.
        market: The calibrated market.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"`` and shared by the
            whole group.

    Returns:
        Unit prices, one per instrument.
    """
    def _price_one(inst: Any) -> Float[Array, ""]:
        return price(
            inst, model, market, method=method, config=config, key=key
        ).price

    return eqx.filter_vmap(_price_one)(instruments)


def group_pv(
    instruments: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, ""]:
    """Present value of one homogeneous group.

    The scalar every sensitivity in this module differentiates, and the
    first of the three quantities that aggregate additively across
    groups.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None``.
        market: The calibrated market.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"``.

    Returns:
        Sum of the group's unit prices.
    """
    return jnp.sum(
        group_prices(
            instruments, model, market,
            method=method, config=config, key=key,
        )
    )


# ── Scenario repricing ───────────────────────────────────────────────


def group_reprice(
    instruments: Any,
    model: Any,
    market: MarketState,
    scenario: KeyedScenario,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, ""]:
    """Reprice one group under a single keyed scenario.

    Applies ``scenario`` with :func:`valax.risk.state_shocks.shock` and
    revalues.  Because the scenario is keyed by identifier, an OIS move
    and a projection-curve (basis) move are distinct, separately
    expressible shocks on the same :class:`~valax.curves.graph.CurveGraph`
    — the thing an unkeyed ``MarketScenario.rate_shocks`` vector cannot
    say and which
    :func:`valax.risk.var.reprice_under_scenario` therefore warns about.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None``.
        market: Base calibrated market.
        scenario: Keyed shocks to apply.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"``.  Reuse the *same*
            key here as for the base valuation.

    Returns:
        Shocked present value of the group.

    Raises:
        KeyError: If a shock identifier addresses no curve, surface,
            spot or dividend on ``market``.
        TypeError: If handed a legacy
            :class:`~valax.market.scenario.MarketScenario`; lift it with
            :func:`valax.market.keyed_scenario.keyed_from_market_scenario`.

    Warns:
        UserWarning: If ``model`` carries its own scalar ``rate`` /
            ``vol`` / ``dividend``, which shadows the corresponding market
            object so that shock contributes exactly zero P&L.
    """
    _warn_shadowed_shocks(model, scenario, "group_reprice")
    shocked = shock(market, scenario)
    return group_pv(
        instruments, model, shocked, method=method, config=config, key=key,
    )


def group_scenario_pnl(
    instruments: Any,
    model: Any,
    market: MarketState,
    scenarios: KeyedScenario,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, " n_scenarios"]:
    """P&L vector of one group over a batched keyed scenario set.

    ``pnl[i] = group_pv(shock(market, scenario_i)) - group_pv(market)``,
    vmapped over the leading scenario axis of a pytree built by
    :func:`valax.market.keyed_scenario.stack_keyed_scenarios`.

    The second of the three additive quantities: P&L **vectors** add
    across groups, and a risk measure is applied once to the aggregate.
    Summing per-group VaRs instead would produce a plausible wrong
    number, silently — a quantile of a sum is not a sum of quantiles.

    ``key`` is closed over, so the base valuation and every shocked
    revaluation share one set of random draws.  Without that an MC
    group's P&L is sampling noise rather than scenario response; see
    ``tests/test_risk/test_state_var.py::TestCommonRandomNumbers``.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None``.
        market: Base calibrated market.
        scenarios: Keyed scenarios stacked along a leading
            ``n_scenarios`` axis.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key for the whole group, required for ``method="mc"``.

    Returns:
        P&L vector of shape ``(n_scenarios,)``.

    Warns:
        UserWarning: If ``model`` carries its own scalar ``rate`` /
            ``vol`` / ``dividend``, which shadows the corresponding market
            object so that shock contributes exactly zero P&L.
    """
    _warn_shadowed_shocks(model, scenarios, "group_scenario_pnl")

    base_value = group_pv(
        instruments, model, market, method=method, config=config, key=key,
    )

    def _single_pnl(scenario: KeyedScenario) -> Float[Array, ""]:
        # Inlined rather than delegated to ``group_reprice`` so the
        # shadowed-shock warning fires once, above, naming this entry
        # point — not once per trace of the vmapped body.
        shocked = shock(market, scenario)
        return group_pv(
            instruments, model, shocked,
            method=method, config=config, key=key,
        ) - base_value

    return jax.vmap(_single_pnl)(scenarios)


# ── Gradients ────────────────────────────────────────────────────────


def group_market_gradient(
    instruments: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> MarketState:
    """``∇`` of the group PV w.r.t. the market — a ``MarketState`` pytree.

    The group-level analogue of
    :func:`valax.greeks.market_greeks.market_gradient`: every
    inexact-array leaf of ``market`` (each curve's ``discount_factors``,
    every surface node, spots, dividends) is replaced by the whole
    group's sensitivity to it; integer and static leaves are ``None``.

    The third additive quantity, and the one that makes multi-curve risk
    fall out for free: the returned pytree carries a separate cotangent
    for *every named curve* on the graph, so a projection curve's risk is
    reported in its own right rather than folded into one rate block.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None``.
        market: The calibrated market.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"``.

    Returns:
        A ``MarketState``-shaped gradient pytree.
    """
    def fn(m: MarketState) -> Float[Array, ""]:
        return group_pv(
            instruments, model, m, method=method, config=config, key=key,
        )

    return eqx.filter_grad(fn)(market)


def group_model_gradient(
    instruments: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Any:
    """``∇`` of the group PV w.r.t. the model's calibrated parameters.

    The group-level analogue of
    :func:`valax.greeks.market_greeks.model_gradient`.  Model greeks
    aggregate additively only across groups that *share* a model; a book
    spanning several models yields one gradient pytree per model, which
    is a keyed union rather than a sum.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group.
        market: The calibrated market.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"``.

    Returns:
        A model-shaped gradient pytree.

    Raises:
        ValueError: If ``model`` is ``None`` — a curve-discounted trade
            has no model parameters.  Use :func:`group_market_gradient`
            for its curve sensitivities.
    """
    if model is None:
        raise ValueError(
            "group_model_gradient needs a model; a model-free "
            "(curve-discounted) group has no model parameters. Use "
            "group_market_gradient for its curve sensitivities."
        )

    def fn(mod: Any) -> Float[Array, ""]:
        return group_pv(
            instruments, mod, market, method=method, config=config, key=key,
        )

    return eqx.filter_grad(fn)(model)
