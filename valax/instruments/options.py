"""Option instrument definitions."""

import equinox as eqx
from jaxtyping import Float
from jax import Array


class EuropeanOption(eqx.Module):
    """European option contract.

    Data-only pytree — no pricing logic. Pass to a pricing function
    along with market data to get a price.
    """

    strike: Float[Array, ""]
    expiry: Float[Array, ""]  # time to expiry in year fractions
    is_call: bool = eqx.field(static=True, default=True)
