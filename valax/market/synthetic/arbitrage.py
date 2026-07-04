"""Deliberate arbitrage injection for stress-testing the library.

Every function in this module takes *valid* synthetic data and
returns a *minimally-invalid* version of it, together with an
``ArbDiagnosis`` describing what was injected and how strongly.

The associated tests (``tests/test_market/test_arbitrage_handling.py``)
feed each broken object into every plausible consumer and assert one
of three outcomes:

1. **Detect**: the consumer raises one of the reserved exceptions in
   :mod:`valax.core.diagnostics`.  *Preferred.*
2. **Regularise**: the consumer returns a sanitised result and emits
   a warning.  *Acceptable.*
3. **Silently mishandle**: the consumer accepts the bad data.  Marked
   ``xfail(strict=True)`` with a reason — green once detection lands.

This makes the xfail set a public, machine-readable backlog of missing
safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


# Tag for the kind of arbitrage injected.  Plain string to keep the
# diagnosis JSON-serialisable; the matching exception type in
# :mod:`valax.core.diagnostics` is named after the same kind.
ArbKind = Literal[
    "non_psd_correlation",
    "butterfly",
    "calendar",
    "put_call_parity",
    "non_convex_smile",
    "inconsistent_quotes",
    "negative_density",
    "basket_variance",
]


@dataclass(frozen=True)
class ArbDiagnosis:
    """Description of a deliberate arbitrage injection.

    Attributes:
        kind: Which family of arbitrage was injected.
        magnitude: Severity, in the most natural units for the kind
            (e.g., minimum eigenvalue for non-PSD; min(d²C/dK²) for
            butterfly).
        location: Where in the input it was injected (index, name…).
        note: Free-text human description.
    """

    kind: ArbKind
    magnitude: float
    location: object | None = None
    note: str = ""


# ── Correlation matrix ────────────────────────────────────────────


def inject_non_psd_correlation(
    correlation: Float[Array, "n n"],
    eps: float = 0.05,
) -> tuple[Float[Array, "n n"], ArbDiagnosis]:
    """Subtract a scaled identity from a valid correlation matrix
    so its minimum eigenvalue drops below zero.

    Args:
        correlation: A valid (symmetric, unit-diagonal, PSD) correlation
            matrix.
        eps: Amount subtracted from each diagonal entry.  The resulting
            matrix has unit diagonal restored, but its eigenvalues
            collectively shift, producing a negative one.

    Returns:
        Tuple ``(perturbed, diagnosis)``.  The perturbed matrix has
        unit diagonal but is *not* PSD; its minimum eigenvalue is
        recorded in ``diagnosis.magnitude``.
    """
    n = correlation.shape[0]
    # Shrink off-diagonals toward zero by (1 + eps); this preserves
    # unit diagonal but inflates correlations beyond the original
    # range and can yield non-PSD matrices.
    off = correlation - jnp.eye(n)
    perturbed = jnp.eye(n) + (1.0 + eps) * off
    # Force unit diagonal exactly.
    perturbed = perturbed.at[jnp.diag_indices(n)].set(1.0)
    min_eig = float(jnp.min(jnp.linalg.eigvalsh(perturbed)))
    return perturbed, ArbDiagnosis(
        kind="non_psd_correlation",
        magnitude=min_eig,
        location=None,
        note=f"Inflated off-diagonals by 1+{eps}.",
    )


# ── Smile / surface ───────────────────────────────────────────────


def inject_butterfly_arb(
    strikes: Float[Array, " n"],
    vols: Float[Array, " n"],
    k_index: int,
    bump: float = -0.05,
) -> tuple[Float[Array, " n"], ArbDiagnosis]:
    """Bump a single point on a vol smile downward so the implied
    call-price grid becomes locally non-convex.

    Args:
        strikes: Strikes (only used for the location annotation).
        vols: Implied vols at those strikes (any maturity slice).
        k_index: Index of the vol to bump.
        bump: Additive bump to ``vols[k_index]`` (negative to create
            a downward dent that breaks convexity).

    Returns:
        Tuple ``(perturbed_vols, diagnosis)``.

    Notes:
        Whether this *actually* triggers a negative density depends on
        the rest of the smile; the magnitude reported in the diagnosis
        is just ``bump`` for reference.
    """
    perturbed = vols.at[k_index].add(bump)
    return perturbed, ArbDiagnosis(
        kind="butterfly",
        magnitude=float(bump),
        location={"k_index": int(k_index), "strike": float(strikes[k_index])},
        note="Single-knot downward bump on the smile.",
    )


def inject_non_convex_smile(
    strikes: Float[Array, " n"],
    vols: Float[Array, " n"],
    k_index: int,
    bump: float = 0.10,
) -> tuple[Float[Array, " n"], ArbDiagnosis]:
    """Spike a single vol *up* by ``bump`` so the resulting call-price
    grid violates convexity in strike.

    Distinct from :func:`inject_butterfly_arb` only in the sign of the
    typical bump and the kind label — convenience for tests that want
    to drive a non-convexity detector specifically.
    """
    perturbed = vols.at[k_index].add(bump)
    return perturbed, ArbDiagnosis(
        kind="non_convex_smile",
        magnitude=float(bump),
        location={"k_index": int(k_index), "strike": float(strikes[k_index])},
        note="Single-knot upward spike breaks call-price convexity.",
    )


def inject_calendar_arb(
    total_variances: Float[Array, " n_expiries"],
    i: int,
    j: int,
) -> tuple[Float[Array, " n_expiries"], ArbDiagnosis]:
    """Swap the total variance at two expiries to force calendar arb.

    Args:
        total_variances: ``w(T_k) = sigma(T_k)^2 * T_k`` at a fixed
            log-moneyness, sorted by ``T_k`` ascending.
        i: Lower expiry index (must satisfy ``i < j``). Swapping a
            larger-T entry into an earlier-T slot makes ``w`` no
            longer monotone non-decreasing.
        j: Upper expiry index (must satisfy ``i < j``).

    Returns:
        Tuple ``(perturbed, diagnosis)``.
    """
    if not (0 <= i < j < total_variances.shape[0]):
        raise ValueError(
            f"Require 0 <= i < j < n; got i={i}, j={j}, n={total_variances.shape[0]}"
        )
    perturbed = total_variances.at[i].set(total_variances[j]).at[j].set(
        total_variances[i]
    )
    drop = float(total_variances[j] - total_variances[i])
    return perturbed, ArbDiagnosis(
        kind="calendar",
        magnitude=drop,
        location={"i": int(i), "j": int(j)},
        note="Swapped total variances at two expiries.",
    )


# ── Price strip ───────────────────────────────────────────────────


def inject_pcp_violation(
    call_prices: Float[Array, " n"],
    put_prices: Float[Array, " n"],
    bp: float = 50.0,
) -> tuple[
    tuple[Float[Array, " n"], Float[Array, " n"]],
    ArbDiagnosis,
]:
    """Inflate all call prices by ``bp`` basis points of spot so
    ``C - P`` no longer matches the discounted forward minus strike.

    Args:
        call_prices: Call price strip.
        put_prices: Matching put price strip (same strikes, same expiry).
        bp: Basis points to add to each call price, expressed relative
            to the call price itself.

    Returns:
        Tuple ``((perturbed_calls, puts), diagnosis)``.
    """
    perturbed_calls = call_prices * (1.0 + bp * 1e-4)
    return (perturbed_calls, put_prices), ArbDiagnosis(
        kind="put_call_parity",
        magnitude=float(bp),
        location=None,
        note=f"Calls inflated by {bp} bp; puts unchanged.",
    )


def inject_negative_density(
    strikes: Float[Array, " n"],
    call_prices: Float[Array, " n"],
    k_index: int,
    bump: float = 0.5,
) -> tuple[Float[Array, " n"], ArbDiagnosis]:
    """Spike one call price so the second strike difference goes negative.

    Args:
        strikes: Strikes (sorted ascending).
        call_prices: Call prices at those strikes.
        k_index: Index of the call price to spike.
        bump: Multiplicative bump (1.5 means +50%).
    """
    perturbed = call_prices.at[k_index].multiply(1.0 + bump)
    return perturbed, ArbDiagnosis(
        kind="negative_density",
        magnitude=float(bump),
        location={"k_index": int(k_index), "strike": float(strikes[k_index])},
        note=f"Call price at index {k_index} multiplied by {1.0 + bump}.",
    )


# ── Bootstrap quotes ──────────────────────────────────────────────


def inject_inconsistent_bootstrap_quotes(
    quotes: Float[Array, " n"],
    bp_offset: float = 10.0,
    index: int = -1,
) -> tuple[Float[Array, " n"], ArbDiagnosis]:
    """Add ``bp_offset`` basis points to one quote so the bootstrap
    cannot reproduce both it and its neighbours.

    Args:
        quotes: Array of par rates (continuous or simply-compounded).
        bp_offset: Offset to add at ``index``, in basis points.
        index: Which quote to perturb (default: last).
    """
    n = quotes.shape[0]
    idx = index if index >= 0 else n + index
    perturbed = quotes.at[idx].add(bp_offset * 1e-4)
    return perturbed, ArbDiagnosis(
        kind="inconsistent_quotes",
        magnitude=float(bp_offset),
        location={"index": int(idx)},
        note=f"Quote {idx} offset by {bp_offset} bp.",
    )


# ── Basket variance ───────────────────────────────────────────────


def inject_basket_variance_violation(
    correlation: Float[Array, "n n"],
    weight_index_i: int,
    weight_index_j: int,
    new_value: float = 1.5,
) -> tuple[Float[Array, "n n"], ArbDiagnosis]:
    """Force one off-diagonal correlation beyond ``[-1, 1]`` so basket
    variance can exceed the Cauchy-Schwarz upper bound.

    Args:
        correlation: Valid correlation matrix.
        weight_index_i: Row index of the off-diagonal entry to set.
        weight_index_j: Column index of the off-diagonal entry to set.
        new_value: Value to insert (default 1.5 — clearly invalid).

    Returns:
        Tuple ``(perturbed, diagnosis)``.
    """
    if weight_index_i == weight_index_j:
        raise ValueError("i and j must differ for an off-diagonal injection.")
    perturbed = (
        correlation
        .at[weight_index_i, weight_index_j].set(new_value)
        .at[weight_index_j, weight_index_i].set(new_value)
    )
    return perturbed, ArbDiagnosis(
        kind="basket_variance",
        magnitude=float(new_value),
        location={"i": int(weight_index_i), "j": int(weight_index_j)},
        note=f"Off-diagonal set to {new_value}.",
    )


__all__ = [
    "ArbDiagnosis",
    "ArbKind",
    "inject_non_psd_correlation",
    "inject_butterfly_arb",
    "inject_non_convex_smile",
    "inject_calendar_arb",
    "inject_pcp_violation",
    "inject_negative_density",
    "inject_inconsistent_bootstrap_quotes",
    "inject_basket_variance_violation",
]
