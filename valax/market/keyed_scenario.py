"""Identifier-keyed market scenarios for :class:`MarketState`.

:class:`~valax.market.scenario.MarketScenario` is four **flat, unkeyed**
vectors laid out positionally over a
:class:`~valax.market.data.MarketData` — ``spot_shocks[n_assets]``,
``vol_shocks[n_assets]``, ``rate_shocks[n_pillars]``,
``dividend_shocks[n_assets]``.  That representation cannot address the
container the unified pricer actually takes: a
:class:`~valax.core.market_state.MarketState` keys curves, surfaces and
spots **by identifier**, holds a whole
:class:`~valax.curves.graph.CurveGraph` (so "rates" is not one block) and
stores vols as surface *objects* (so a vol shock is not one scalar per
asset).  One unkeyed pillar vector cannot even say *which* curve of a
dual-curve book it moves.

:class:`KeyedScenario` closes that gap.  It is a sparse, additive
description of a market move:

* every field is a ``dict`` from identifier to shock magnitude;
* an **absent key means no shock** — ``KeyedScenario()`` is the no-op
  scenario, and a single-curve / single-asset market needs no ids at all
  thanks to the ``"_default_"`` sentinel;
* an **unknown key is an error**, never a silent no-op (see
  :func:`valax.risk.state_shocks.shock`).

Apply one with :func:`valax.risk.state_shocks.shock`, which returns a new
``MarketState`` (never mutates) and is differentiable in both the base
market and the shock magnitudes.

Relationship to the legacy view
-------------------------------

``MarketScenario`` stays the legacy view for one deprecation cycle: the
whole ``valax/risk/var.py`` VaR framework and the scenario generators in
``valax/risk/scenarios.py`` consume it unchanged.
``examples/19_unified_book_workflow.py`` shows the keyed path end to end
(PCA and basis scenarios on a dual-curve book).  :func:`keyed_from_market_scenario` lifts one into a
``KeyedScenario`` so the two representations can be cross-checked against
each other (and so a legacy generator can feed the keyed path).
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from valax.curves.graph import _DEFAULT_CURVE_ID
from valax.market.scenario import MarketScenario


#: Sentinel identifier that makes single-curve / single-asset markets
#: zero-configuration.  Shared with ``CurveGraph`` and ``MarketState``.
DEFAULT_ID = _DEFAULT_CURVE_ID


class KeyedScenario(eqx.Module):
    """Identifier-keyed market shocks for a :class:`MarketState`.

    Each field is a mapping from identifier to a shock magnitude; keys
    absent from a mapping are not shocked.  Magnitudes are deltas, never
    levels, and every one of them is a differentiable leaf.

    Attributes:
        spot_shocks: Asset id → spot shock.  Additive, or a return when
            ``multiplicative`` is set.
        vol_shocks: Surface id → additive **parallel** vol shift, applied
            by :func:`valax.risk.state_shocks.shift_surface`.  An asset
            id ``"SPX"`` also resolves the conventional surface key
            ``"SPX.VOL"``.
        curve_shocks: Curve id → per-pillar additive zero-rate bumps for
            that curve of the :class:`~valax.curves.graph.CurveGraph`,
            applied by :func:`valax.risk.shocks.bump_graph_curve`.  This
            is what makes an OIS move and a projection-curve (basis) move
            two distinct, separately expressible scenarios.
        dividend_shocks: Asset id → additive dividend-yield bump.
        multiplicative: If True, ``spot_shocks`` are returns
            (``new = old * (1 + shock)``); all other shocks are always
            additive.  Static field.
    """

    spot_shocks: dict = eqx.field(default_factory=dict)
    vol_shocks: dict = eqx.field(default_factory=dict)
    curve_shocks: dict = eqx.field(default_factory=dict)
    dividend_shocks: dict = eqx.field(default_factory=dict)
    multiplicative: bool = eqx.field(static=True, default=False)


def zero_keyed_scenario() -> KeyedScenario:
    """Create the no-op keyed scenario.

    Every mapping is empty, so :func:`valax.risk.state_shocks.shock`
    returns the base market untouched.  Provided for symmetry with
    :func:`valax.market.scenario.zero_scenario`, which needs factor
    counts; the keyed representation is sparse and needs none.

    Returns:
        An empty :class:`KeyedScenario`.
    """
    return KeyedScenario()


def keyed_from_market_scenario(
    scenario: MarketScenario,
    asset_ids: tuple[str, ...] | list[str] | None = None,
    curve_id: str = DEFAULT_ID,
    surface_ids: tuple[str, ...] | list[str] | None = None,
) -> KeyedScenario:
    """Lift a legacy :class:`MarketScenario` into a keyed scenario.

    The positional vectors are zipped against ``asset_ids`` and the one
    unkeyed ``rate_shocks`` block is attributed to ``curve_id`` — the
    decision the flat representation leaves implicit and which a keyed
    scenario has to state.  Vol shocks become per-surface *parallel
    shifts*, which is exactly the legacy semantics (``vols + vol_shocks``
    on one scalar per asset) generalised to surface objects.

    Args:
        scenario: The legacy scenario to lift.
        asset_ids: Asset identifiers, one per entry of the per-asset
            vectors.  Defaults to ``ASSET_0 …``, matching
            :meth:`valax.core.market_state.MarketState.from_market_data`.
        curve_id: Curve the ``rate_shocks`` block belongs to.  Defaults
            to the ``"_default_"`` sentinel, i.e. the single-curve
            market's only curve.
        surface_ids: Surface identifiers for the vol shocks.  Defaults to
            ``"<ASSET>.VOL"``, the key ``from_market_data`` files
            per-asset vols under.

    Returns:
        An equivalent :class:`KeyedScenario`.

    Raises:
        ValueError: If ``asset_ids`` or ``surface_ids`` has a different
            length from the scenario's per-asset vectors.
    """
    n_assets = int(jnp.asarray(scenario.spot_shocks).shape[0])
    if asset_ids is None:
        asset_ids = tuple(f"ASSET_{i}" for i in range(n_assets))
    elif len(asset_ids) != n_assets:
        raise ValueError(
            f"asset_ids has length {len(asset_ids)} but the scenario "
            f"carries {n_assets} assets."
        )
    if surface_ids is None:
        surface_ids = tuple(f"{aid}.VOL" for aid in asset_ids)
    elif len(surface_ids) != n_assets:
        raise ValueError(
            f"surface_ids has length {len(surface_ids)} but the scenario "
            f"carries {n_assets} assets."
        )

    return KeyedScenario(
        spot_shocks={
            aid: scenario.spot_shocks[i] for i, aid in enumerate(asset_ids)
        },
        vol_shocks={
            sid: scenario.vol_shocks[i] for i, sid in enumerate(surface_ids)
        },
        curve_shocks={curve_id: scenario.rate_shocks},
        dividend_shocks={
            aid: scenario.dividend_shocks[i]
            for i, aid in enumerate(asset_ids)
        },
        multiplicative=scenario.multiplicative,
    )


def stack_keyed_scenarios(scenarios: list[KeyedScenario]) -> KeyedScenario:
    """Stack keyed scenarios along a leading scenario axis.

    Every leaf gains a leading ``n_scenarios`` dimension, giving the
    keyed analogue of :class:`~valax.market.scenario.ScenarioSet` — a
    single pytree ready for ``jax.vmap`` over the scenario axis.  All
    scenarios must shock the same identifiers (a missing key is a
    *structural* difference, not a zero).

    Args:
        scenarios: Non-empty list of scenarios with identical key sets
            and matching ``multiplicative`` flags.

    Returns:
        One :class:`KeyedScenario` whose leaves are batched.

    Raises:
        ValueError: If the list is empty, the key sets differ, or the
            ``multiplicative`` flags disagree.
    """
    if not scenarios:
        raise ValueError("stack_keyed_scenarios needs at least one scenario.")

    fields = ("spot_shocks", "vol_shocks", "curve_shocks", "dividend_shocks")
    first = scenarios[0]
    for other in scenarios[1:]:
        if other.multiplicative != first.multiplicative:
            raise ValueError(
                "stack_keyed_scenarios: scenarios disagree on "
                "`multiplicative`; they would not share a P&L convention."
            )
        for field in fields:
            keys_a = set(getattr(first, field))
            keys_b = set(getattr(other, field))
            if keys_a != keys_b:
                raise ValueError(
                    f"stack_keyed_scenarios: {field} key sets differ "
                    f"({sorted(keys_a)} vs {sorted(keys_b)}). Pass an "
                    f"explicit zero for ids a scenario leaves untouched."
                )

    stacked: dict[str, Any] = {}
    for field in fields:
        stacked[field] = {
            key: jnp.stack(
                [jnp.asarray(getattr(s, field)[key]) for s in scenarios]
            )
            for key in getattr(first, field)
        }
    return KeyedScenario(multiplicative=first.multiplicative, **stacked)
