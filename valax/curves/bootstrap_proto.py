"""Protocol for bootstrap instruments.

Each bootstrap instrument carries a market quote and provides one
residual on the curve graph: a quote like "the 5Y SOFR OIS swap rate is
3.85%" maps to one equation that the calibrated curve must satisfy.
The bootstrapper collects residuals from all instruments and solves for
the log-discount-factors that zero them out simultaneously.

The protocol is **structural** — any class implementing the right shape
qualifies, no inheritance required.  This matches the JAX/equinox style
and keeps the central bootstrap solver free of ``isinstance`` dispatch:
to add a new instrument type, define a class with ``curves_touched``
and ``residual``, and the joint solver picks it up.

The protocol is tagged ``@runtime_checkable`` so calling code can do
``isinstance(inst, BootstrapInstrument)`` for guard clauses, but this
is intentionally a weak check (it only verifies attribute presence,
not signatures) — the contract is enforced by the residual unit tests
on each implementation.

See ``production.md`` §11.3 for design rationale and ``theory.md`` §3.7,
§3.8 for the no-arbitrage residuals each instrument family imposes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float, Int
from jax import Array

from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveGraph


@runtime_checkable
class BootstrapInstrument(Protocol):
    """A market quote that imposes one residual on the curve graph.

    Implementations must:

    * Carry a ``curves_touched: tuple[str, ...]`` attribute (typically
      a static :class:`equinox` field) listing the identifiers of every
      curve the residual depends on.  The joint solver uses this to
      construct the sparse Jacobian structure.

    * Provide a ``residual(graph, fixings, ref_date) -> Float[Array, ""]``
      method that returns zero when the graph correctly reprices the
      market quote.

    The residual must be **differentiable** with respect to graph leaves
    and the instrument's own quote field, so ``jax.grad`` can flow
    through the bootstrap (for sensitivity analysis) and so the joint
    solver can compute its Newton Jacobian via ``jax.jacobian``.

    The residual must be **JIT-compatible**: no Python-level branching
    on traced values, no scipy / numpy calls, no ``isinstance`` checks
    on JAX arrays.  Branch on the static ``curves_touched`` tuple or on
    static instrument fields if needed.

    The ``fixings`` argument is unconditional: instruments that do not
    consume fixings simply ignore it.  This avoids a parallel
    ``residual_no_fixings`` overload and keeps the dispatch surface flat.
    """

    curves_touched: tuple[str, ...]

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Return the repricing residual for this instrument.

        Args:
            graph: Calibrated (or in-progress) curve graph.  Only the
                curves listed in ``self.curves_touched`` are read.
            fixings: Realised-fixings registry.  Unused if the
                instrument has no float legs or all fixings are forward.
            ref_date: Valuation date as an ordinal integer scalar.

        Returns:
            A scalar ``Float[Array, ""]`` — zero when the graph reprices
            the quote exactly, non-zero otherwise.  Sign convention is
            instrument-specific but stable per type (documented on each
            implementation).
        """
        ...
