"""Tests for the keyed quote container ``QuoteSet``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.market import QuoteSet


@pytest.fixture
def quotes():
    return QuoteSet.from_dict({"USD.OIS.1Y": 0.045, "SPX.SPOT": 100.0})


class TestConstruction:
    def test_from_dict_casts_to_float(self, quotes):
        assert quotes["SPX.SPOT"].dtype == jnp.float64
        assert quotes.keys() == ("SPX.SPOT", "USD.OIS.1Y")

    def test_from_vector_roundtrip(self):
        q = QuoteSet.from_vector(("a", "b"), jnp.array([1.0, 2.0]))
        assert float(q["b"]) == 2.0
        assert jnp.allclose(q.vector(("b", "a")), jnp.array([2.0, 1.0]))

    def test_from_vector_length_mismatch(self):
        with pytest.raises(ValueError, match="2 keys"):
            QuoteSet.from_vector(("a", "b"), jnp.array([1.0]))

    def test_missing_key_names_available(self, quotes):
        with pytest.raises(KeyError, match="available"):
            quotes["EUR.OIS.1Y"]


class TestFunctionalUpdates:
    def test_bump_returns_new_set(self, quotes):
        bumped = quotes.bump("SPX.SPOT", 1.0)
        assert float(bumped["SPX.SPOT"]) == 101.0
        assert float(quotes["SPX.SPOT"]) == 100.0

    def test_select_and_merge(self, quotes):
        sub = quotes.select(["SPX.SPOT"])
        assert sub.keys() == ("SPX.SPOT",)
        merged = sub.merge(QuoteSet.from_dict({"SPX.SPOT": 99.0, "X": 1.0}))
        assert float(merged["SPX.SPOT"]) == 99.0 and "X" in merged

    def test_add_treats_missing_as_zero(self, quotes):
        total = quotes + QuoteSet.from_dict({"SPX.SPOT": 1.0, "NEW": 2.0})
        assert float(total["SPX.SPOT"]) == 101.0
        assert float(total["NEW"]) == 2.0
        assert float(total["USD.OIS.1Y"]) == pytest.approx(0.045)

    def test_empty_vector(self):
        assert QuoteSet().vector().shape == (0,)


class TestPytree:
    def test_grad_is_quote_shaped(self, quotes):
        g = jax.grad(lambda q: q["SPX.SPOT"] ** 2 + 3.0 * q["USD.OIS.1Y"])(quotes)
        assert isinstance(g, QuoteSet)
        assert float(g["SPX.SPOT"]) == pytest.approx(200.0)
        assert float(g["USD.OIS.1Y"]) == pytest.approx(3.0)

    def test_vmap_over_scenarios(self, quotes):
        batch = jax.tree.map(lambda v: v + jnp.arange(3.0), quotes)
        out = jax.vmap(lambda q: q["SPX.SPOT"])(batch)
        assert jnp.allclose(out, jnp.array([100.0, 101.0, 102.0]))

    def test_jit_smoke(self, quotes):
        f = eqx.filter_jit(lambda q: q.bump("SPX.SPOT", 1.0)["SPX.SPOT"])
        assert float(f(quotes)) == 101.0
