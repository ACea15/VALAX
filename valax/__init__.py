"""VALAX: JAX-native quantitative finance valuation engine."""

import jax
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

# Reserved arbitrage / no-arbitrage exception types.  Not raised by the
# library today; see ``tests/test_market/test_arbitrage_handling.py``
# for the detection backlog.
from valax.core.diagnostics import (
    ArbitrageError,
    ButterflyArbError,
    CalendarArbError,
    InconsistentQuotesError,
    NonConvexSmileError,
    NonPSDCorrelationError,
    PutCallParityError,
)

__all__ = [
    "__version__",
    "ArbitrageError",
    "ButterflyArbError",
    "CalendarArbError",
    "InconsistentQuotesError",
    "NonConvexSmileError",
    "NonPSDCorrelationError",
    "PutCallParityError",
]
