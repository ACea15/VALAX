"""Reserved exception types for arbitrage / no-arbitrage diagnostics.

These exception types are *reserved*: they exist so synthetic-data
arbitrage tests can name the error the library *would* raise once
detection is implemented. None of them is raised by the library today
(see ``tests/test_market/test_arbitrage_handling.py``, which uses
``pytest.mark.xfail`` to track the gap).

Hierarchy
---------
::

    ValueError
    └── ArbitrageError
        ├── NonPSDCorrelationError
        ├── ButterflyArbError
        ├── CalendarArbError
        ├── PutCallParityError
        ├── NonConvexSmileError
        └── InconsistentQuotesError

Design notes
------------
- All errors subclass :class:`ValueError` so existing user code that
  guards against bad inputs with ``except ValueError`` still catches
  them.
- Each error carries an optional ``magnitude`` and ``location`` payload
  so callers can decide whether to raise, warn-and-regularise, or
  silently project back into the no-arbitrage region.
- These errors must **not** be raised inside JIT-traced code (they
  raise Python exceptions which break tracing).  Use them only at
  construction / validation boundaries.
"""

from __future__ import annotations


class ArbitrageError(ValueError):
    """Base class for static-arbitrage diagnostic errors.

    Attributes:
        magnitude: Numerical severity of the violation (e.g., minimum
            eigenvalue for non-PSD, max negative density for butterfly
            arb).  Optional.
        location: Where in the input the violation was detected
            (e.g., strike index, slice index, quote name).  Optional.
    """

    def __init__(
        self,
        message: str,
        *,
        magnitude: float | None = None,
        location: object | None = None,
    ) -> None:
        super().__init__(message)
        self.magnitude = magnitude
        self.location = location


class NonPSDCorrelationError(ArbitrageError):
    """A correlation matrix is not positive semi-definite.

    Typically raised when ``min(eigvalsh(C)) < -tol``.  Multi-asset MC
    via Cholesky factorisation will fail (or worse: silently produce
    biased paths) on such a matrix.
    """


class ButterflyArbError(ArbitrageError):
    """Butterfly arbitrage detected on an implied-vol smile / price grid.

    Equivalent to a negative risk-neutral density at some strike, i.e.
    ``d^2 C / dK^2 < 0`` for European call prices.
    """


class CalendarArbError(ArbitrageError):
    """Calendar-spread arbitrage detected across two expiries.

    Total implied variance ``w(k, T) = sigma(k, T)^2 * T`` must be
    non-decreasing in ``T`` at every fixed log-moneyness ``k``.
    """


class PutCallParityError(ArbitrageError):
    """A call/put pair violates put-call parity beyond a tolerance.

    For European options under deterministic rates and dividends:
    ``C - P = S * exp(-q*T) - K * exp(-r*T)``.
    """


class NonConvexSmileError(ArbitrageError):
    """A call-price-vs-strike curve is not convex.

    Equivalent to butterfly arbitrage at the discrete strike level;
    kept separate so detectors that work directly on the price grid
    can raise a more specific error.
    """


class InconsistentQuotesError(ArbitrageError):
    """A set of bootstrap quotes is mutually inconsistent.

    For example, two overlapping swap quotes that imply incompatible
    par rates for the same maturity bucket.
    """


__all__ = [
    "ArbitrageError",
    "NonPSDCorrelationError",
    "ButterflyArbError",
    "CalendarArbError",
    "PutCallParityError",
    "NonConvexSmileError",
    "InconsistentQuotesError",
]
