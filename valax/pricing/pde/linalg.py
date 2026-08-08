"""Reusable tridiagonal linear algebra for finite-difference solvers.

This is the single place in VALAX that touches :mod:`lineax`. Extracting the
tridiagonal solve and the matrix-vector product here keeps the dependency
surface small and lets the numerics be unit-tested in isolation; the 2-D ADI
solvers (PR-2) reuse :func:`tridiagonal_solve` per axis.

A tridiagonal operator is represented by three bands:

- ``lower``: sub-diagonal, length ``n - 1`` (multiplies ``v[:-1]``).
- ``diag``: main diagonal, length ``n``.
- ``upper``: super-diagonal, length ``n - 1`` (multiplies ``v[1:]``).
"""

import lineax as lx
from jaxtyping import Float
from jax import Array


def tridiagonal_solve(
    lower: Float[Array, " n_minus_1"],
    diag: Float[Array, " n"],
    upper: Float[Array, " n_minus_1"],
    rhs: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve the tridiagonal system ``A x = rhs`` via ``lineax.Tridiagonal``.

    Args:
        lower: Sub-diagonal band (length ``n - 1``).
        diag: Main diagonal (length ``n``).
        upper: Super-diagonal band (length ``n - 1``).
        rhs: Right-hand side (length ``n``).

    Returns:
        The solution vector ``x`` (length ``n``).
    """
    operator = lx.TridiagonalLinearOperator(diag, lower, upper)
    solution = lx.linear_solve(operator, rhs, solver=lx.Tridiagonal())
    return solution.value


def tridiagonal_matvec(
    lower: Float[Array, " n_minus_1"],
    diag: Float[Array, " n"],
    upper: Float[Array, " n_minus_1"],
    v: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Multiply a tridiagonal operator by a vector: ``A @ v``.

    Args:
        lower: Sub-diagonal band (length ``n - 1``).
        diag: Main diagonal (length ``n``).
        upper: Super-diagonal band (length ``n - 1``).
        v: Vector to multiply (length ``n``).

    Returns:
        The product ``A @ v`` (length ``n``).
    """
    result = diag * v
    result = result.at[1:].add(lower * v[:-1])
    result = result.at[:-1].add(upper * v[1:])
    return result
