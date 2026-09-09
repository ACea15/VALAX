r"""Cross-engine price reconciliation for short-rate (and other) instruments.

The value of having four numerical routes to the same model — analytic, PDE,
Monte Carlo and (where it exists) QuantLib — is *triangulation*: a price that
agrees across independent engines is a price worth trusting, and a
single-engine calibration is only as credible as its agreement with the
others. :func:`price_all_engines` runs whichever engines are available for a
given ``(instrument, model)`` and returns them in one table, so tests and
examples can assert agreement and users can see the trade-offs (speed, bias,
optionality coverage) side by side.

Engines are **best-effort**: an engine with no registered recipe for the pair
(e.g. Monte-Carlo caplets under Hull-White, or any QuantLib engine when the
package is absent) is simply omitted rather than raising, so the same call
works across the whole instrument matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import jax

from valax.pricing.mc import MCConfig, mc_price_dispatch
from valax.pricing.pde import pde_price_dispatch

# An analytic / QuantLib oracle is either a pre-computed price or a callable
# ``(instrument, model) -> price``.
Oracle = Union[float, Callable[[Any, Any], Any]]


@dataclass(frozen=True)
class EngineQuote:
    """A single engine's price (and Monte-Carlo error, when applicable).

    Attributes:
        price: The engine's price estimate.
        stderr: Monte-Carlo standard error, or ``None`` for deterministic
            engines (analytic, PDE, QuantLib).
    """

    price: float
    stderr: Optional[float] = None


@dataclass(frozen=True)
class ReconciliationTable:
    """Prices for one instrument across every available engine.

    Attributes:
        quotes: Mapping of engine name (``"analytic"``, ``"pde"``, ``"mc"``,
            ``"quantlib"``) to its :class:`EngineQuote`.
        reference: The engine name used as the accuracy reference when
            computing differences (the first present of analytic, quantlib,
            pde).
    """

    quotes: dict[str, EngineQuote]
    reference: str

    def price(self, engine: str) -> float:
        """Return the price reported by ``engine``."""
        return self.quotes[engine].price

    def abs_diff(self, engine: str, reference: Optional[str] = None) -> float:
        """Absolute price difference of ``engine`` from the reference engine."""
        ref = reference if reference is not None else self.reference
        return abs(self.quotes[engine].price - self.quotes[ref].price)

    def rel_diff(self, engine: str, reference: Optional[str] = None) -> float:
        """Relative price difference of ``engine`` from the reference engine.

        Guards a near-zero reference with a unit floor so the ratio stays
        finite for deep-out-of-the-money prices.
        """
        ref = reference if reference is not None else self.reference
        base = abs(self.quotes[ref].price)
        return self.abs_diff(engine, ref) / max(base, 1.0)

    def max_rel_diff(self, reference: Optional[str] = None) -> float:
        """Largest relative difference of any non-reference engine."""
        ref = reference if reference is not None else self.reference
        others = [e for e in self.quotes if e != ref]
        return max((self.rel_diff(e, ref) for e in others), default=0.0)

    def as_dict(self) -> dict[str, float]:
        """Return a plain ``engine -> price`` mapping."""
        return {name: q.price for name, q in self.quotes.items()}

    def __str__(self) -> str:  # pragma: no cover - formatting only
        ref = self.reference
        lines = [f"{'engine':<10}{'price':>18}{'stderr':>14}{'rel_diff':>12}"]
        for name, q in self.quotes.items():
            se = "" if q.stderr is None else f"{q.stderr:.6g}"
            rd = "" if name == ref else f"{self.rel_diff(name, ref):.3e}"
            lines.append(f"{name:<10}{q.price:>18.8f}{se:>14}{rd:>12}")
        lines.append(f"(reference engine: {ref})")
        return "\n".join(lines)


def _resolve_oracle(oracle: Oracle, instrument: Any, model: Any) -> float:
    """Evaluate a float-or-callable oracle to a plain float."""
    value = oracle(instrument, model) if callable(oracle) else oracle
    return float(value)


def price_all_engines(
    instrument: Any,
    model: Any,
    *,
    pde_config: Any = None,
    mc_config: Optional[MCConfig] = None,
    mc_key: Optional[jax.Array] = None,
    mc_market_args: Optional[dict[str, Any]] = None,
    analytic: Optional[Oracle] = None,
    quantlib: Optional[Oracle] = None,
) -> ReconciliationTable:
    """Price ``instrument`` under ``model`` on every available engine.

    Each engine is attempted independently and omitted (rather than raising) if
    it has no registered recipe for the pair or its optional dependency is
    missing — so the same call spans the whole instrument matrix.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree.
        pde_config: PDE grid configuration (:class:`~valax.pricing.pde.PDEConfig`
            or ``PDEConfig2D``). If ``None`` the PDE engine is skipped.
        mc_config: Monte-Carlo configuration. If ``None`` (or ``mc_key`` is
            ``None``) the MC engine is skipped.
        mc_key: JAX PRNG key for the Monte-Carlo engine.
        mc_market_args: Extra keyword arguments forwarded to the MC recipe
            (e.g. ``{"n_steps": 200}``).
        analytic: A closed-form price, or a callable ``(instrument, model) ->
            price``. Included as the ``"analytic"`` row when given.
        quantlib: A QuantLib price, or a callable ``(instrument, model) ->
            price``. Included as the ``"quantlib"`` row when given; a callable
            that raises (e.g. QuantLib not installed) is silently skipped.

    Returns:
        A :class:`ReconciliationTable`. The reference engine for differences is
        the first present of ``analytic``, ``quantlib``, ``pde``.

    Raises:
        ValueError: If no engine at all could be evaluated.
    """
    quotes: dict[str, EngineQuote] = {}

    if analytic is not None:
        quotes["analytic"] = EngineQuote(
            _resolve_oracle(analytic, instrument, model)
        )

    if pde_config is not None:
        try:
            result = pde_price_dispatch(instrument, model, pde_config)
            quotes["pde"] = EngineQuote(float(result.price))
        except ValueError:
            pass

    if mc_config is not None and mc_key is not None:
        try:
            result = mc_price_dispatch(
                instrument, model, mc_config, mc_key,
                **(mc_market_args or {}),
            )
            quotes["mc"] = EngineQuote(
                float(result.price), float(result.stderr)
            )
        except ValueError:
            pass

    if quantlib is not None:
        try:
            quotes["quantlib"] = EngineQuote(
                _resolve_oracle(quantlib, instrument, model)
            )
        except Exception:  # noqa: BLE001 - QuantLib absent / unsupported
            pass

    if not quotes:
        raise ValueError(
            "price_all_engines could not evaluate any engine for "
            f"({type(instrument).__name__}, {type(model).__name__}). "
            "Pass a pde_config, an mc_config + mc_key, or an analytic oracle."
        )

    for candidate in ("analytic", "quantlib", "pde", "mc"):
        if candidate in quotes:
            reference = candidate
            break

    return ReconciliationTable(quotes=quotes, reference=reference)
