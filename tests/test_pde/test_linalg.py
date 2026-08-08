"""Unit tests for the tridiagonal linear-algebra helpers."""

import jax
import jax.numpy as jnp

from valax.pricing.pde.linalg import tridiagonal_matvec, tridiagonal_solve


def _dense(lower, diag, upper):
    n = diag.shape[0]
    A = jnp.diag(diag)
    A = A.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(lower)
    A = A.at[jnp.arange(0, n - 1), jnp.arange(1, n)].set(upper)
    return A


def test_solve_matches_dense():
    key = jax.random.PRNGKey(0)
    n = 25
    k1, k2, k3, k4 = jax.random.split(key, 4)
    lower = jax.random.normal(k1, (n - 1,))
    upper = jax.random.normal(k2, (n - 1,))
    # Diagonally dominant -> well-conditioned.
    diag = 5.0 + jnp.abs(jax.random.normal(k3, (n,)))
    rhs = jax.random.normal(k4, (n,))

    x = tridiagonal_solve(lower, diag, upper, rhs)
    x_ref = jnp.linalg.solve(_dense(lower, diag, upper), rhs)
    assert jnp.allclose(x, x_ref, atol=1e-10)


def test_matvec_matches_dense():
    key = jax.random.PRNGKey(1)
    n = 25
    k1, k2, k3, k4 = jax.random.split(key, 4)
    lower = jax.random.normal(k1, (n - 1,))
    upper = jax.random.normal(k2, (n - 1,))
    diag = jax.random.normal(k3, (n,))
    v = jax.random.normal(k4, (n,))

    y = tridiagonal_matvec(lower, diag, upper, v)
    y_ref = _dense(lower, diag, upper) @ v
    assert jnp.allclose(y, y_ref, atol=1e-10)
