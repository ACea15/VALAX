"""JAX type aliases for VALAX."""

from jaxtyping import Float, Int, Bool
from jax import Array

Scalar = Float[Array, ""]
Vec = Float[Array, " n"]
Mat = Float[Array, "n m"]
DateArray = Int[Array, "..."]
BoolArray = Bool[Array, "..."]
