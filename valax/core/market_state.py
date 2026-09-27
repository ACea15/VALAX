"""``MarketState`` — the calibrated market container.

One pytree holding every *calibrated* market object a pricer can need:
a :class:`~valax.curves.graph.CurveGraph`, keyed volatility surfaces,
spots, dividends, realised fixings, and the credit / inflation / FX
corners.  It is the ``market`` argument of the unified pricing
signature

.. code-block:: python

    price(instrument, model, market, *, method="auto", config=None, key=None)

described in ``PRICING_ONTOLOGY.md``.

Relationship to :class:`~valax.market.data.MarketData`
------------------------------------------------------

``MarketState`` **generalises** ``MarketData`` rather than renaming it:

============  ================================  ==============================
Aspect        ``MarketData``                    ``MarketState``
============  ================================  ==============================
curves        one curve + optional graph        ``CurveGraph`` primary
vols          one **scalar** per asset          ``dict[str, surface]``
spots         vectors, positional               keyed by asset id
credit/infl.  —                                 keyed curves
============  ================================  ==============================

Per resolved question Q4, ``MarketData`` survives one minor release as a
**two-way view**: :meth:`MarketState.from_market_data` and
:meth:`MarketState.to_market_data` are lossless for everything the older
container can express (scalar vols round-trip through
:class:`~valax.surfaces.constant.ConstantVol`), so the ``risk/var.py``
scenario framework keeps working untouched while new code migrates.

Single-curve stays zero-config
------------------------------

Multi-curve is opt-in.  A single-curve market is a one-entry graph under
the grandfathered ``"_default_"`` sentinel — the same sentinel the
single-curve bootstrap already emits — and every id field defaults to it,
so a vanilla trade needs no selector configuration at all:

.. code-block:: python

    market = MarketState.single_curve(curve, spot=jnp.array(100.0))

Identifier alphabets
--------------------

Curve ids follow the alphabet frozen in MC-Curves-2
(``<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]``) and are validated where curves
are *built*, not here.  Surface ids follow the alphabet frozen by
resolved question Q3 — ``<UNDERLYING>.<KIND>[.<QUALIFIER>]`` with
``KIND`` one of ``VOL``, ``LV``, ``LEVERAGE``, ``CUBE``, ``OPTIONLET``
(e.g. ``SPX.VOL``, ``EURUSD.VOL``, ``USD.SOFR.CUBE``) — and *are*
validated here, because ``MarketState`` is where surfaces are first
keyed.  The ``"_default_"`` sentinel is accepted everywhere.

Surfaces themselves stay **duck-typed**: anything implementing the
existing vol-source protocol (``__call__`` and/or
``total_variance(log_moneyness, expiry)``) is accepted, which is what
lets ``GridVolSurface``, SVI, ``SABRVolSurface``, ``LeverageGrid``,
``SwaptionCube`` and ``ConstantVol`` coexist in one container.
"""

from __future__ import annotations

import re
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory, empty_fixing_history
from valax.curves.graph import CurveGraph, _DEFAULT_CURVE_ID
from valax.surfaces.constant import ConstantVol


#: Sentinel identifier used by every selector default.  Shared with the
#: single-curve bootstrap path so a single-curve market resolves with no
#: configuration.
DEFAULT_ID = _DEFAULT_CURVE_ID

#: Surface-kind labels of the frozen surface-id alphabet (Q3).
_SURFACE_KINDS = ("VOL", "LV", "LEVERAGE", "CUBE", "OPTIONLET")

#: Regex enforcing ``<UNDERLYING>.<KIND>[.<QUALIFIER>]`` on surface ids.
#: ``UNDERLYING`` may itself be dotted (``USD.SOFR``) so rates cubes reuse
#: the curve alphabet's leading fields.
_SURFACE_ID_RE = re.compile(
    r"^[A-Z][A-Z0-9_]*(\.[A-Z][A-Z0-9_]*)*"
    rf"\.({'|'.join(_SURFACE_KINDS)})"
    r"(\.[A-Za-z0-9_]+)?$"
)


def validate_surface_id(surface_id: str) -> None:
    """Validate ``surface_id`` against the frozen surface alphabet.

    Args:
        surface_id: The key a surface is filed under in
            :attr:`MarketState.surfaces`.

    Raises:
        ValueError: If the identifier is neither the ``"_default_"``
            sentinel nor a match for ``<UNDERLYING>.<KIND>``
            ``[.<QUALIFIER>]``.
    """
    if surface_id == DEFAULT_ID:
        return
    if not _SURFACE_ID_RE.fullmatch(surface_id):
        raise ValueError(
            f"Invalid surface identifier {surface_id!r}. Expected the "
            f"pattern <UNDERLYING>.<KIND>[.<QUALIFIER>] with KIND in "
            f"{_SURFACE_KINDS} (e.g. 'SPX.VOL', 'EURUSD.VOL', "
            f"'USD.SOFR.CUBE'), or the sentinel {DEFAULT_ID!r}."
        )


class MarketState(eqx.Module):
    """Calibrated market container: curves, surfaces, spots, fixings.

    Every array leaf is differentiable, and the nested curves/surfaces
    are themselves pytrees, so ``jax.grad`` of a pricing function that
    takes a ``MarketState`` yields sensitivities to every pillar, every
    surface node, and every spot simultaneously.  Scenario shocks return
    a *new* ``MarketState`` (``eqx.tree_at``), never mutate one.

    Attributes:
        curves: Identifier-keyed discount/forward curves.  A single-curve
            market is a one-entry graph under ``"_default_"``.
        surfaces: Identifier-keyed vol surfaces / cubes / leverage grids
            (duck-typed; see module docstring).
        spots: Asset-id-keyed spot levels.
        dividends: Asset-id-keyed continuous dividend yields.
        fixings: Realised-fixing registry for seasoned floating legs.
        credit_curves: Identifier-keyed survival curves.
        inflation_curves: Identifier-keyed inflation curves.
        fx_spots: Pair-keyed FX spot rates (e.g. ``"EURUSD"``).
        asset_ids: Canonical asset ordering.  Defines the vector layout
            used by :meth:`to_market_data` and by vectorised risk code;
            empty means "sorted order of ``spots``".  Static field.
    """

    curves: CurveGraph
    surfaces: dict = eqx.field(default_factory=dict)
    spots: dict = eqx.field(default_factory=dict)
    dividends: dict = eqx.field(default_factory=dict)
    fixings: FixingHistory = eqx.field(default_factory=empty_fixing_history)
    credit_curves: dict = eqx.field(default_factory=dict)
    inflation_curves: dict = eqx.field(default_factory=dict)
    fx_spots: dict = eqx.field(default_factory=dict)
    asset_ids: tuple = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        """Validate surface identifiers against the frozen alphabet."""
        for surface_id in self.surfaces:
            validate_surface_id(surface_id)

    # ── Constructors ─────────────────────────────────────────────────

    @classmethod
    def single_curve(
        cls,
        discount_curve: DiscountCurve,
        *,
        spot: Float[Array, ""] | None = None,
        vol: Float[Array, ""] | Any = None,
        dividend: Float[Array, ""] | None = None,
        fixings: FixingHistory | None = None,
    ) -> "MarketState":
        """Build a zero-configuration single-curve, single-asset market.

        Everything is filed under the ``"_default_"`` sentinel, which is
        what a bare instrument's selectors resolve to, so the resulting
        market prices a vanilla trade with no ids specified anywhere.

        Args:
            discount_curve: The one curve; used for discounting and, when
                a trade names no projection curve, for forward projection.
            spot: Optional spot level of the single underlying.
            vol: Optional volatility — a scalar (wrapped into a
                :class:`~valax.surfaces.constant.ConstantVol`) or any
                duck-typed surface object.
            dividend: Optional continuous dividend yield.
            fixings: Optional realised-fixing registry.

        Returns:
            A :class:`MarketState`.
        """
        surfaces: dict = {}
        if vol is not None:
            # A duck-typed surface is callable; a scalar is not.
            surfaces[DEFAULT_ID] = (
                vol if callable(vol) else ConstantVol(vol=jnp.asarray(vol))
            )
        return cls(
            curves=CurveGraph(curves={DEFAULT_ID: discount_curve}),
            surfaces=surfaces,
            spots={} if spot is None else {DEFAULT_ID: jnp.asarray(spot)},
            dividends=(
                {} if dividend is None
                else {DEFAULT_ID: jnp.asarray(dividend)}
            ),
            fixings=empty_fixing_history() if fixings is None else fixings,
            asset_ids=() if spot is None else (DEFAULT_ID,),
        )

    @classmethod
    def from_market_data(
        cls,
        market_data: Any,
        asset_ids: tuple | None = None,
    ) -> "MarketState":
        """Lift a legacy :class:`~valax.market.data.MarketData` (Q4 view).

        The per-asset scalar ``vols`` become
        :class:`~valax.surfaces.constant.ConstantVol` surfaces keyed
        ``"<ASSET>.VOL"``, so the round trip through
        :meth:`to_market_data` is lossless.  When ``market_data`` carries
        a ``curve_graph`` it becomes the state's graph (with the legacy
        ``discount_curve`` added under the sentinel if absent);
        otherwise the single ``discount_curve`` is the whole graph.

        Args:
            market_data: The legacy container.
            asset_ids: Optional canonical asset names, one per entry of
                ``market_data.spots``.  Defaults to ``ASSET_0 …``.

        Returns:
            An equivalent :class:`MarketState`.

        Raises:
            ValueError: If ``asset_ids`` length does not match the
                number of assets.
        """
        n_assets = int(jnp.asarray(market_data.spots).shape[0])
        if asset_ids is None:
            asset_ids = tuple(f"ASSET_{i}" for i in range(n_assets))
        elif len(asset_ids) != n_assets:
            raise ValueError(
                f"asset_ids has length {len(asset_ids)} but MarketData "
                f"carries {n_assets} assets."
            )

        if market_data.curve_graph is None:
            curves = {DEFAULT_ID: market_data.discount_curve}
        else:
            curves = dict(market_data.curve_graph.curves)
            curves.setdefault(DEFAULT_ID, market_data.discount_curve)

        return cls(
            curves=CurveGraph(curves=curves),
            surfaces={
                f"{aid}.VOL": ConstantVol(vol=market_data.vols[i])
                for i, aid in enumerate(asset_ids)
            },
            spots={aid: market_data.spots[i] for i, aid in enumerate(asset_ids)},
            dividends={
                aid: market_data.dividends[i]
                for i, aid in enumerate(asset_ids)
            },
            asset_ids=tuple(asset_ids),
        )

    # ── Views ────────────────────────────────────────────────────────

    def to_market_data(self) -> Any:
        """Project back onto a legacy :class:`MarketData` (Q4 view).

        Assets are laid out in :attr:`asset_ids` order (or sorted spot
        keys if that is empty).  A surface contributes a scalar vol when
        it is a :class:`~valax.surfaces.constant.ConstantVol`; richer
        surfaces are sampled at-the-money is *not* attempted — they
        cannot be represented by ``MarketData``'s scalar-per-asset
        field, so this raises rather than silently flattening a smile.

        Returns:
            A :class:`~valax.market.data.MarketData`.

        Raises:
            ValueError: If an asset's surface is not a ``ConstantVol``
                (i.e. the state is strictly richer than ``MarketData``),
                or if the state holds no discount curve.
        """
        from valax.market.data import MarketData

        ids = self.asset_ids or tuple(sorted(self.spots))
        vols = []
        for aid in ids:
            surface = self.surfaces.get(f"{aid}.VOL", self.surfaces.get(aid))
            if not isinstance(surface, ConstantVol):
                raise ValueError(
                    f"MarketState.to_market_data: asset {aid!r} carries "
                    f"{type(surface).__name__ if surface is not None else 'no'}"
                    f" volatility, which MarketData's scalar-per-asset "
                    f"`vols` field cannot represent.  Keep using "
                    f"MarketState, or flatten the surface explicitly."
                )
            vols.append(jnp.asarray(surface.vol))

        return MarketData(
            spots=jnp.stack([jnp.asarray(self.spots[a]) for a in ids])
            if ids else jnp.zeros((0,)),
            vols=jnp.stack(vols) if ids else jnp.zeros((0,)),
            dividends=jnp.stack(
                [jnp.asarray(self.dividends.get(a, 0.0)) for a in ids]
            ) if ids else jnp.zeros((0,)),
            discount_curve=self.discount_curve,
            curve_graph=self.curves,
        )

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def discount_curve(self) -> DiscountCurve:
        """The sentinel curve, or the only curve if there is exactly one.

        Raises:
            ValueError: If the graph is empty, or holds several curves
                and none under the sentinel — in which case the caller
                must name the curve it means.
        """
        if DEFAULT_ID in self.curves:
            return self.curves[DEFAULT_ID]
        if len(self.curves.curves) == 1:
            return next(iter(self.curves.values()))
        raise ValueError(
            f"MarketState holds {len(self.curves.curves)} curves and none "
            f"under the {DEFAULT_ID!r} sentinel; name the curve explicitly "
            f"(available: {sorted(self.curves.keys())})."
        )

    def curve(self, curve_id: str | None = None) -> DiscountCurve:
        """Look up a curve by id; ``None`` means the default curve.

        Args:
            curve_id: Identifier, or ``None`` / the sentinel for the
                default curve.

        Returns:
            The :class:`DiscountCurve`.

        Raises:
            KeyError: If ``curve_id`` is absent from the graph.
        """
        if curve_id is None or curve_id == DEFAULT_ID:
            return self.discount_curve
        if curve_id not in self.curves:
            raise KeyError(
                f"Curve {curve_id!r} not in MarketState "
                f"(available: {sorted(self.curves.keys())})."
            )
        return self.curves[curve_id]

    def surface(self, surface_id: str | None = None) -> Any:
        """Look up a vol surface by id; ``None`` means the default one.

        Args:
            surface_id: Identifier, or ``None`` / the sentinel for the
                default surface.

        Returns:
            The surface object (duck-typed).

        Raises:
            KeyError: If ``surface_id`` is absent.
        """
        key = DEFAULT_ID if surface_id is None else surface_id
        if key not in self.surfaces:
            raise KeyError(
                f"Surface {key!r} not in MarketState "
                f"(available: {sorted(self.surfaces)})."
            )
        return self.surfaces[key]

    def spot(self, asset_id: str | None = None) -> Float[Array, ""]:
        """Look up a spot level by asset id.

        Args:
            asset_id: Identifier, or ``None`` / the sentinel.  When the
                state holds exactly one asset, that asset answers the
                sentinel.

        Returns:
            The spot level.

        Raises:
            KeyError: If the asset is absent (or ambiguous).
        """
        return self._lookup_asset(self.spots, asset_id, "spot")

    def dividend(self, asset_id: str | None = None) -> Float[Array, ""]:
        """Look up a continuous dividend yield by asset id.

        Args:
            asset_id: Identifier, or ``None`` / the sentinel.

        Returns:
            The dividend yield; ``0.0`` when the asset exists but carries
            no dividend entry.

        Raises:
            KeyError: If the asset is unknown and no default exists.
        """
        try:
            return self._lookup_asset(self.dividends, asset_id, "dividend")
        except KeyError:
            if not self.dividends:
                return jnp.asarray(0.0)
            raise

    def _lookup_asset(
        self,
        table: dict,
        asset_id: str | None,
        what: str,
    ) -> Float[Array, ""]:
        """Shared sentinel-aware lookup for the per-asset tables."""
        key = DEFAULT_ID if asset_id is None else asset_id
        if key in table:
            return jnp.asarray(table[key])
        if key == DEFAULT_ID and len(table) == 1:
            return jnp.asarray(next(iter(table.values())))
        raise KeyError(
            f"No {what} for asset {key!r} in MarketState "
            f"(available: {sorted(table)})."
        )

    @property
    def reference_date(self) -> Int[Array, ""]:
        """Valuation date, taken from the default curve.

        Returns:
            The curve's ``reference_date`` ordinal.
        """
        return self.discount_curve.reference_date
