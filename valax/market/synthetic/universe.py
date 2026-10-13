r"""Synthetic quote universes: truth → quotes → calibration chain.

The rest of :mod:`valax.market.synthetic` produces calibrated objects (curves,
model parameters) directly. This module goes one step further back and
produces what a desk actually observes — **market quotes** — from a sampled
ground truth, together with the calibration chain that consumes them:

1. **Sample a true market**: an OIS curve (Nelson-Siegel-Svensson), projection
   curves (OIS plus a smooth tenor basis), spot, dividend, a SABR smile per
   expiry, and optionally a true pricing model.
2. **Quote it with the library's own code**: curve quotes are the par values
   that zero each bootstrap instrument's residual on the true curves
   (:func:`valax.curves.par_quote`); smile quotes are the true smile's implied
   vols; premium quotes are :func:`valax.core.pricing.price` under the true
   model.
3. **Add noise** (Gaussian, in basis points of rate or vol, or relative for
   premiums).
4. **Return the quotes with their chain nodes.** Quote keys and nodes come from
   the same specification, so a snapshot's quotes always match the chain that
   calibrates them.

Because the true curves are represented exactly in the bootstrap's own basis
(log-linear discount factors on the quote maturities), a noise-free snapshot
calibrates back to the truth to solver tolerance — the strongest end-to-end
check of the calibration chain, and the reason the examples and tests built on
this module can assert truth recovery rather than mere self-consistency.

Noise model
-----------
Today every quote receives independent Gaussian noise. Planned refinements
(deliberately not implemented yet, kept here so the extension point is
obvious):

* **bid/ask** — quote ``(bid, ask)`` pairs whose half-spread widens in the
  wings and at long expiries; the mid becomes the quote and ``1/spread`` the
  least-squares weight passed to the smile and model nodes;
* **correlated noise** — noise shared across a curve (a stale feed shifts a
  whole strip) rather than i.i.d. per quote;
* **stale and missing quotes** — flagged through
  :class:`~valax.market.provenance.QuoteProvenance` (``stale=True``) or dropped
  from the ``QuoteSet``, exercising the chain's handling of absent quotes.

Stream names (``SeedRegistry`` keys) are listed on each function, so a snapshot
is reproducible from ``(master_seed, library_version)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.bootstrap_graph import par_quote
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph, CurveSpec
from valax.curves.instruments import DepositRate, IborSwapRate, OISSwapRate
from valax.instruments.options import AsianOption, EquityBarrierOption, EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.quotes import QuoteSet
from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.curves import _nss_zero_rate, _sample_nss_params
from valax.market.synthetic.seeds import SeedRegistry
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.surfaces.sabr_surface import SABRVolSurface

if TYPE_CHECKING:  # import cycles: core / calibration import valax.market
    from valax.calibration.chain import CalibrationChain, CalibrationNode
    from valax.core.market_state import MarketState


DEFAULT_ID = "_default_"

#: Days per year used to place synthetic pillar and coupon dates.
_DAYS_PER_YEAR = 365


def _uniform(key: Array, low: float, high: float) -> Float[Array, ""]:
    return jax.random.uniform(key, (), dtype=jnp.float64, minval=low, maxval=high)


def _gaussian_noise(
    registry: SeedRegistry, stream: str, shape: tuple, scale: float
) -> Float[Array, "..."]:
    """i.i.d. Gaussian noise; exactly zero when ``scale == 0``.

    Extension point for the bid/ask and correlated-noise models described in
    the module docstring.
    """
    if scale == 0.0:
        return jnp.zeros(shape)
    return scale * jax.random.normal(registry.key(stream), shape, dtype=jnp.float64)


# ── Rates world ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RatesWorldSpec:
    """Shape of a synthetic single-currency rates market.

    Attributes:
        currency: Currency code used in quote keys.
        ois_id: Curve id of the OIS discount curve (``"_default_"`` for a
            single-curve market that bare instruments resolve to).
        projection_tenors: Projection curves to build, e.g. ``("3M",)``;
            each becomes curve ``"<CCY>.TERM.<tenor>"`` quoted by IBOR swaps.
            Empty for a single-curve (OIS-only) market.
        pillar_years: Integer quote maturities in years. The first is a
            deposit on the OIS curve; the rest are annual-coupon swaps.
        basis_bp_range: Range of the long-end tenor basis (bp) sampled per
            projection curve.
        basis_slope_bp_range: Range of the short-minus-long basis shape (bp).
        zero_basis: Force every projection curve to equal the OIS curve
            (the single-curve limit, with multi-curve plumbing).
    """

    currency: str = "USD"
    ois_id: str = "USD.SOFR.OIS"
    projection_tenors: tuple[str, ...] = ("3M",)
    pillar_years: tuple[int, ...] = (1, 2, 3, 5, 7, 10)
    basis_bp_range: tuple[float, float] = (5.0, 25.0)
    basis_slope_bp_range: tuple[float, float] = (-10.0, 10.0)
    zero_basis: bool = False

    def projection_id(self, tenor: str) -> str:
        """Curve id of the projection curve for ``tenor``."""
        return f"{self.currency}.TERM.{tenor}"


@dataclass(frozen=True)
class RatesWorld:
    """A sampled true rates market.

    Attributes:
        spec: The specification it was sampled from.
        reference_date: Valuation date ordinal.
        day_count: Day count of every curve and quote.
        truth: True curves, one per id, on the quote-maturity pillars.
    """

    spec: RatesWorldSpec
    reference_date: Int[Array, ""]
    day_count: str
    truth: CurveGraph

    @property
    def pillar_dates(self) -> Int[Array, " n"]:
        """Quote-maturity pillar ordinals (excluding the reference date)."""
        years = jnp.asarray(self.spec.pillar_years, dtype=jnp.int32)
        return self.reference_date + years * _DAYS_PER_YEAR

    @property
    def curve_ids(self) -> tuple[str, ...]:
        """OIS id followed by every projection id."""
        return (self.spec.ois_id,) + tuple(
            self.spec.projection_id(t) for t in self.spec.projection_tenors
        )

    def truth_market(self) -> "MarketState":
        """The true curves as a :class:`~valax.core.market_state.MarketState`."""
        from valax.core.market_state import MarketState

        return MarketState(curves=self.truth)


def _curve_on_pillars(
    ref: Int[Array, ""],
    pillar_dates: Int[Array, " n"],
    zero_rates: Float[Array, " n"],
    years: Float[Array, " n"],
    day_count: str,
) -> DiscountCurve:
    dfs = jnp.concatenate([jnp.ones(1), jnp.exp(-zero_rates * years)])
    return DiscountCurve(
        pillar_dates=jnp.concatenate([ref[None], pillar_dates]),
        discount_factors=dfs,
        reference_date=ref,
        day_count=day_count,
    )


def sample_rates_world(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    spec: RatesWorldSpec = RatesWorldSpec(),
) -> RatesWorld:
    """Sample a true OIS curve and its projection curves.

    The OIS zero curve is Nelson-Siegel-Svensson (``cfg.nss_param_ranges``);
    each projection curve adds a smooth tenor basis
    ``b(t) = b_inf + b_slope * (1 - exp(-t/2)) / (t/2)``. Both are evaluated
    on the quote-maturity pillars only, so the truth lies exactly in the
    bootstrap's representation (log-linear discount factors between quote
    maturities) and a noise-free snapshot recalibrates to it.

    Stream names: ``"synthetic.curve.nss.params"``,
    ``"synthetic.universe.basis.<tenor>"``.

    Args:
        registry: Seed registry.
        cfg: Supplies the reference date, day count and NSS ranges.
        spec: Market shape.

    Returns:
        The :class:`RatesWorld`.

    Raises:
        ValueError: If ``pillar_years`` is empty, unsorted or non-positive.
    """
    years_t = tuple(spec.pillar_years)
    if not years_t or list(years_t) != sorted(set(years_t)) or years_t[0] < 1:
        raise ValueError(
            "RatesWorldSpec.pillar_years must be strictly increasing "
            f"positive integers, got {years_t}."
        )
    ref = jnp.asarray(cfg.reference_date, dtype=jnp.int32)
    years = jnp.asarray(years_t, dtype=jnp.float64)
    pillars = ref + jnp.asarray(years_t, dtype=jnp.int32) * _DAYS_PER_YEAR
    # Pillar times in the curve day count (act_365 with 365-day years: exact).
    from valax.dates.daycounts import year_fraction

    times = year_fraction(ref, pillars, cfg.day_count)

    ois_zero = _nss_zero_rate(years, *_sample_nss_params(registry, cfg))
    curves = {spec.ois_id: _curve_on_pillars(ref, pillars, ois_zero, times, cfg.day_count)}
    for tenor in spec.projection_tenors:
        if spec.zero_basis:
            basis = jnp.zeros_like(years)
        else:
            k1, k2 = jax.random.split(registry.key(f"synthetic.universe.basis.{tenor}"))
            b_inf = _uniform(k1, *spec.basis_bp_range) * 1e-4
            b_slope = _uniform(k2, *spec.basis_slope_bp_range) * 1e-4
            x = years / 2.0
            basis = b_inf + b_slope * (1.0 - jnp.exp(-x)) / x
        curves[spec.projection_id(tenor)] = _curve_on_pillars(
            ref, pillars, ois_zero + basis, times, cfg.day_count
        )
    return RatesWorld(
        spec=spec, reference_date=ref, day_count=cfg.day_count,
        truth=CurveGraph(curves=curves),
    )


@dataclass(frozen=True)
class CurveQuoteSet:
    """Synthetic curve quotes and the chain node that bootstraps them.

    Attributes:
        quotes: Noisy quotes (what the chain is calibrated from).
        clean: Noise-free par quotes on the true curves.
        keys: Quote identifiers, in node order.
        tenors: ``key -> maturity in years`` (for scenario shapes).
        instruments: Bootstrap instruments, in node order.
        specs: Curve specs of the joint solve.
        node: Ready-made ``curve_graph_node``.
    """

    quotes: QuoteSet
    clean: QuoteSet
    keys: tuple[str, ...]
    tenors: dict[str, float]
    instruments: tuple
    specs: tuple[CurveSpec, ...]
    node: "CalibrationNode"


def _annual(ref: Int[Array, ""], n_years: int, per_year: int = 1) -> Int[Array, " n"]:
    k = jnp.arange(1, n_years * per_year + 1, dtype=jnp.int32)
    return ref + (k * _DAYS_PER_YEAR) // per_year


def synthesize_curve_quote_set(
    registry: SeedRegistry,
    world: RatesWorld,
    *,
    bp_noise: float = 0.0,
    node_name: str | None = None,
) -> CurveQuoteSet:
    """Par quotes of the standard strip on the true curves, plus their node.

    OIS curve: a deposit to the first pillar and annual OIS swaps to the
    rest. Each projection curve: IBOR swaps (annual fixed, float at the curve
    tenor) to every pillar. Keys are ``"<CCY>.OIS.<n>Y"`` and
    ``"<CCY>.IRS<tenor>.<n>Y"``.

    Stream name: ``"synthetic.universe.curve_noise.<curve_id>"``.

    Args:
        registry: Seed registry.
        world: The true rates market.
        bp_noise: Standard deviation of i.i.d. Gaussian quote noise, in basis
            points. ``0`` gives exact par quotes.
        node_name: Name of the curve node; defaults to ``"<CCY>_curves"``.

    Returns:
        The :class:`CurveQuoteSet`.
    """
    from valax.calibration.nodes import curve_graph_node

    spec, ref, dc = world.spec, world.reference_date, world.day_count
    years = spec.pillar_years
    pillars = world.pillar_dates
    ccy = spec.currency

    insts: list = []
    keys: list[str] = []
    tenors: dict[str, float] = {}
    for i, y in enumerate(years):
        if i == 0:
            inst = DepositRate(
                start_date=ref, end_date=pillars[0], rate=jnp.asarray(0.0),
                day_count=dc, curves_touched=(spec.ois_id,),
            )
        else:
            inst = OISSwapRate(
                start_date=ref, fixed_dates=_annual(ref, y), rate=jnp.asarray(0.0),
                day_count=dc, curves_touched=(spec.ois_id,),
            )
        insts.append(inst)
        keys.append(f"{ccy}.OIS.{y}Y")
        tenors[keys[-1]] = float(y)
    for tenor in spec.projection_tenors:
        per_year = {"1M": 12, "3M": 4, "6M": 2, "12M": 1, "1Y": 1}.get(tenor)
        if per_year is None:
            raise ValueError(f"Unsupported projection tenor {tenor!r}.")
        proj = spec.projection_id(tenor)
        for y in years:
            float_dates = _annual(ref, y, per_year)
            fixing = jnp.concatenate([ref[None], float_dates[:-1]])
            insts.append(
                IborSwapRate(
                    start_date=ref, fixed_dates=_annual(ref, y),
                    float_dates=float_dates, fixing_dates=fixing,
                    rate=jnp.asarray(0.0), fixed_day_count=dc, float_day_count=dc,
                    curves_touched=(spec.ois_id, proj), index_id=proj,
                )
            )
            keys.append(f"{ccy}.IRS{tenor}.{y}Y")
            tenors[keys[-1]] = float(y)

    clean_vec = jnp.stack([par_quote(inst, world.truth, ref) for inst in insts])
    noise = jnp.concatenate([
        _gaussian_noise(
            registry, f"synthetic.universe.curve_noise.{cid}", (len(years),),
            bp_noise * 1e-4,
        )
        for cid in world.curve_ids
    ])
    specs = tuple(
        CurveSpec(curve_id=cid, currency=ccy, pillar_dates=pillars, day_count=dc)
        for cid in world.curve_ids
    )
    node = curve_graph_node(
        node_name or f"{ccy.lower()}_curves", ref, specs, insts, keys
    )
    return CurveQuoteSet(
        quotes=QuoteSet.from_vector(keys, clean_vec + noise),
        clean=QuoteSet.from_vector(keys, clean_vec),
        keys=tuple(keys),
        tenors=tenors,
        instruments=tuple(insts),
        specs=specs,
        node=node,
    )


# ── Equity world ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EquityWorldSpec:
    """Shape of a synthetic single-underlying equity market.

    The discount curve is the rates world's OIS curve; for the bare
    instruments of :mod:`valax.instruments.options` to resolve it, build the
    rates world with ``ois_id="_default_"``.

    Attributes:
        asset_id: Spot / dividend key (``"_default_"`` for bare instruments).
        surface_id: Key of the SABR surface (``"_default_"`` or
            ``"<UNDERLYING>.VOL"``).
        quote_prefix: Prefix of every quote key, e.g. ``"SPX"``.
        expiries: Smile expiries (year fractions), sorted.
        moneyness: Strikes as fractions of each expiry's true forward.
        beta: SABR beta (held fixed; a fixed input of the chain).
        spot_range: Range of the sampled spot.
        div_range: Range of the sampled continuous dividend yield.
        atm_vol_range: Range of the short- and long-dated ATM vols.
    """

    asset_id: str = DEFAULT_ID
    surface_id: str = DEFAULT_ID
    quote_prefix: str = "SPX"
    expiries: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    moneyness: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    beta: float = 0.5
    spot_range: tuple[float, float] = (80.0, 120.0)
    div_range: tuple[float, float] = (0.0, 0.03)
    atm_vol_range: tuple[float, float] = (0.15, 0.30)


@dataclass(frozen=True)
class EquityWorld:
    """A sampled true equity market on top of a rates world.

    Attributes:
        spec: The specification.
        rates: The underlying rates world (discounting and forwards).
        spot: True spot.
        dividend: True continuous dividend yield.
        forwards: True forward per expiry.
        strikes: Quoted absolute strikes, ``(n_expiries, n_moneyness)``.
        smile: True SABR surface.
    """

    spec: EquityWorldSpec
    rates: RatesWorld
    spot: Float[Array, ""]
    dividend: Float[Array, ""]
    forwards: Float[Array, " n_exp"]
    strikes: Float[Array, "n_exp n_k"]
    smile: SABRVolSurface

    def truth_market(self) -> "MarketState":
        """True curves, spot, dividend and smile in one market."""
        from valax.calibration.chain import install_market_fields

        market = self.rates.truth_market()
        return install_market_fields(market, {
            "spots": {self.spec.asset_id: self.spot},
            "dividends": {self.spec.asset_id: self.dividend},
            "surfaces": {self.spec.surface_id: self.smile},
        })


def sample_equity_world(
    registry: SeedRegistry,
    rates: RatesWorld,
    spec: EquityWorldSpec = EquityWorldSpec(),
) -> EquityWorld:
    """Sample spot, dividend and a SABR smile term structure.

    Smile parameters vary smoothly with expiry: the ATM vol moves from a short
    to a long level, ``rho`` mean-reverts from a steep short-dated skew, and
    ``nu`` decays like :math:`T^{-0.3}`. Forwards are
    :math:`S e^{(r(T) - q) T}` with :math:`r(T)` the true OIS zero rate.

    Stream names: ``"synthetic.universe.equity.<field>"``.

    Args:
        registry: Seed registry.
        rates: The rates world providing the discount curve.
        spec: Market shape.

    Returns:
        The :class:`EquityWorld`.
    """
    from valax.core.binding import project_rate

    keys = jax.random.split(registry.key("synthetic.universe.equity.params"), 7)
    spot = _uniform(keys[0], *spec.spot_range)
    div = _uniform(keys[1], *spec.div_range)
    atm_short = _uniform(keys[2], *spec.atm_vol_range)
    atm_long = _uniform(keys[3], *spec.atm_vol_range)
    rho_short = _uniform(keys[4], -0.8, -0.4)
    rho_long = _uniform(keys[5], -0.5, -0.1)
    nu_1y = _uniform(keys[6], 0.3, 0.8)

    t = jnp.asarray(spec.expiries, dtype=jnp.float64)
    curve = rates.truth[rates.spec.ois_id]
    r = jnp.stack([project_rate(curve, ti) for ti in t])
    forwards = spot * jnp.exp((r - div) * t)
    w = jnp.exp(-t / 1.0)
    atm = atm_long + (atm_short - atm_long) * w
    beta = jnp.asarray(spec.beta)
    smile = SABRVolSurface(
        expiries=t,
        forwards=forwards,
        alphas=atm * forwards ** (1.0 - beta),
        betas=jnp.full_like(t, spec.beta),
        rhos=rho_long + (rho_short - rho_long) * w,
        nus=nu_1y * t ** (-0.3),
    )
    strikes = forwards[:, None] * jnp.asarray(spec.moneyness)[None, :]
    return EquityWorld(
        spec=spec, rates=rates, spot=spot, dividend=div,
        forwards=forwards, strikes=strikes, smile=smile,
    )


def _expiry_label(t: float) -> str:
    months = round(t * 12)
    return f"{months // 12}Y" if months % 12 == 0 else f"{months}M"


@dataclass(frozen=True)
class SmileQuoteSet:
    """Synthetic equity quotes (spot, dividend, smile) and their nodes.

    Attributes:
        quotes: Noisy quotes (spot and dividend are quoted exactly).
        clean: Noise-free quotes.
        fixed: Held-fixed inputs (SABR beta).
        vol_keys: Smile quote keys, shape ``(n_expiries, n_moneyness)``.
        coords: ``vol key -> (expiry, moneyness)`` (for scenario shapes).
        spot_key: Spot quote key.
        dividend_key: Dividend quote key.
        beta_key: Fixed-input key of SABR beta.
        inputs_node: Explicit node installing spot and dividend.
        smile_node: ``sabr_smile_node`` on the upstream forwards.
    """

    quotes: QuoteSet
    clean: QuoteSet
    fixed: QuoteSet
    vol_keys: tuple[tuple[str, ...], ...]
    coords: dict[str, tuple[float, float]]
    spot_key: str
    dividend_key: str
    beta_key: str
    inputs_node: "CalibrationNode"
    smile_node: "CalibrationNode"


def synthesize_smile_quote_set(
    registry: SeedRegistry,
    world: EquityWorld,
    *,
    vol_bp_noise: float = 0.0,
) -> SmileQuoteSet:
    """Implied-vol quotes of the true smile, plus spot / dividend quotes.

    Keys: ``"<P>.SPOT"``, ``"<P>.DIV"``, ``"<P>.VOL.<expiry>.<moneyness%>"``
    and the fixed input ``"<P>.SABR.BETA"``, with ``<P>`` the quote prefix.

    Stream name: ``"synthetic.universe.vol_noise"``.

    Args:
        registry: Seed registry.
        world: The true equity market.
        vol_bp_noise: Standard deviation of i.i.d. Gaussian vol noise, in vol
            basis points (``10`` = 0.10 vol points).

    Returns:
        The :class:`SmileQuoteSet`.
    """
    from valax.calibration.nodes import (
        equity_forward,
        market_inputs_node,
        sabr_smile_node,
    )

    spec, p = world.spec, world.spec.quote_prefix
    vol_keys = tuple(
        tuple(f"{p}.VOL.{_expiry_label(t)}.{round(m * 100)}" for m in spec.moneyness)
        for t in spec.expiries
    )
    coords = {
        k: (t, m)
        for t, row in zip(spec.expiries, vol_keys)
        for m, k in zip(spec.moneyness, row)
    }
    s = world.smile

    def slice_vols(i: int) -> Float[Array, " n_k"]:
        model = SABRModel(alpha=s.alphas[i], beta=s.betas[i], rho=s.rhos[i], nu=s.nus[i])
        return jax.vmap(
            lambda k: sabr_implied_vol(model, s.forwards[i], k, s.expiries[i])
        )(world.strikes[i])

    clean_vols = jnp.stack([slice_vols(i) for i in range(len(spec.expiries))])
    noise = _gaussian_noise(
        registry, "synthetic.universe.vol_noise", clean_vols.shape, vol_bp_noise * 1e-4
    )
    flat_keys = [k for row in vol_keys for k in row]
    spot_key, div_key, beta_key = f"{p}.SPOT", f"{p}.DIV", f"{p}.SABR.BETA"
    base = {spot_key: world.spot, div_key: world.dividend}
    clean = QuoteSet.from_vector(flat_keys, clean_vols.reshape(-1)).with_values(base)
    quotes = QuoteSet.from_vector(
        flat_keys, (clean_vols + noise).reshape(-1)
    ).with_values(base)

    inputs = market_inputs_node(
        f"{p.lower()}_inputs",
        spots={spec.asset_id: spot_key},
        dividends={spec.asset_id: div_key},
    )
    smile = sabr_smile_node(
        spec.surface_id,
        expiries=jnp.asarray(spec.expiries),
        strikes=world.strikes,
        quote_keys=vol_keys,
        forward_fn=equity_forward(
            spec.asset_id if spec.asset_id != DEFAULT_ID else None,
            world.rates.spec.ois_id,
        ),
        beta_key=beta_key,
        name=f"{p.lower()}_smile",
    )
    return SmileQuoteSet(
        quotes=quotes,
        clean=clean,
        fixed=QuoteSet.from_dict({beta_key: spec.beta}),
        vol_keys=vol_keys,
        coords=coords,
        spot_key=spot_key,
        dividend_key=div_key,
        beta_key=beta_key,
        inputs_node=inputs,
        smile_node=smile,
    )


def synthesize_model_quotes(
    registry: SeedRegistry,
    model: Any,
    basket: Sequence[Any],
    keys: Sequence[str],
    market: "MarketState",
    *,
    quote_form: str = "price",
    rel_noise: float = 0.0,
    stream_name: str = "synthetic.universe.model_noise",
) -> QuoteSet:
    """Quotes of a basket priced under a *true* model.

    Use it to test model-level calibration (``price_fit_node``) for parameter
    recovery: quotes generated here are exactly reproducible by the model at
    its true parameters.

    Args:
        registry: Seed registry.
        model: The true model.
        basket: Calibration instruments.
        keys: One quote key per instrument.
        market: Market to price on (usually a world's ``truth_market()``).
        quote_form: ``"price"`` (premiums) or ``"lognormal_vol"``
            (Black-Scholes implied vols, inverted with the market's bound rate
            and dividend — the same convention ``price_fit_node`` uses).
        rel_noise: Relative Gaussian noise on premiums, or absolute vol noise
            in vol units for ``"lognormal_vol"``.
        stream_name: Seed stream for the noise.

    Returns:
        The :class:`~valax.market.quotes.QuoteSet`.

    Raises:
        ValueError: On an unknown quote form or a length mismatch.
    """
    from valax.core.pricing import price
    from valax.models.black_scholes import BlackScholesModel

    if len(basket) != len(keys):
        raise ValueError("synthesize_model_quotes: basket and keys differ in length.")
    if quote_form not in ("price", "lognormal_vol"):
        raise ValueError(f"Unknown quote_form {quote_form!r}.")
    premiums = jnp.stack([price(inst, model, market).price for inst in basket])
    if quote_form == "price":
        noise = _gaussian_noise(registry, stream_name, premiums.shape, rel_noise)
        return QuoteSet.from_vector(keys, premiums * (1.0 + noise))

    def implied(inst: Any, target: Float[Array, ""]) -> Float[Array, ""]:
        def f(v):
            return price(inst, BlackScholesModel(vol=v), market).price - target

        vol = jnp.asarray(0.2)
        for _ in range(30):  # Newton; eager, synthetic data only
            vol = vol - f(vol) / jnp.maximum(jax.grad(f)(vol), 1e-12)
        return vol

    vols = jnp.stack([implied(i, p) for i, p in zip(basket, premiums)])
    noise = _gaussian_noise(registry, stream_name, vols.shape, rel_noise)
    return QuoteSet.from_vector(keys, vols + noise)


# ── Rates volatility world: swaption cube and cap matrix ─────────────


@dataclass(frozen=True)
class RatesVolSpec:
    """Shape of a synthetic rates volatility market on a rates world.

    Swaptions: one SABR smile per ``(expiry, tenor)`` cell, quoted as vols at
    ATM-offset strikes. Caps: a flat-vol matrix over cap maturities × strikes
    (offsets around the longest cap's ATM forward, fixed as absolute strikes),
    generated from per-bucket caplet vols. Quotes are **normal** (Bachelier)
    vols with a normal SABR (``beta = 0``) by default — the rates-desk
    convention, well defined at any rate level, including the low and negative
    rates the Nelson-Siegel-Svensson sampler produces.

    Attributes:
        cube_id: Surface id of the swaption cube.
        expiry_years: Swaption expiries (integer years).
        tenor_years: Underlying swap tenors (integer years).
        strike_offsets_bp: Quoted strikes as offsets to each cell's forward.
        is_normal: Normal (Bachelier) vol quotes if True, else lognormal.
        shift: Displacement of the normal SABR expansion, so strikes and
            forwards down to ``-shift`` are admissible.
        beta: SABR beta of the true cube and caplets (held fixed by the
            chain).
        atm_vol_range: Range of the true ATM vol per cell, in the quoting
            convention (absolute rate vol when normal).
        rho_range: Range of the true SABR rho per cell.
        nu_range: Range of the true SABR nu per cell.
        strip_id: Surface id of the stripped cap matrix.
        optionlet_id: Surface id of the SABR optionlet surface fitted on it.
        cap_years: Quoted cap maturities (integer years, quarterly
            optionlets, first fixing one quarter after the valuation date).
        cap_strike_offsets_bp: Cap strikes as offsets to the longest cap's
            ATM forward (rounded to 1bp, then held as absolute strikes).
        caplet_vol_range: Range of the true ATM caplet vol per bucket.
    """

    cube_id: str = "USD.CUBE"
    expiry_years: tuple[int, ...] = (1, 2, 5)
    tenor_years: tuple[int, ...] = (2, 5, 10)
    strike_offsets_bp: tuple[float, ...] = (-100.0, -50.0, 0.0, 50.0, 100.0)
    is_normal: bool = True
    shift: float = 0.03
    beta: float = 0.0
    atm_vol_range: tuple[float, float] = (0.0060, 0.0120)
    rho_range: tuple[float, float] = (-0.40, -0.05)
    nu_range: tuple[float, float] = (0.25, 0.50)
    strip_id: str = "USD.OPTIONLET.STRIP"
    optionlet_id: str = "USD.OPTIONLET"
    cap_years: tuple[int, ...] = (1, 2, 3, 5)
    cap_strike_offsets_bp: tuple[float, ...] = (-50.0, 0.0, 50.0, 100.0)
    caplet_vol_range: tuple[float, float] = (0.0060, 0.0120)


@dataclass(frozen=True)
class RatesVolWorld:
    """A sampled true rates-volatility market.

    Attributes:
        spec: The specification it was sampled from.
        rates: The underlying rates world (true curves).
        swaptions: ``[i][j]`` representative swaptions of the cube cells.
        cube: The true :class:`~valax.surfaces.swaption_cube.SwaptionCube`.
        caps: Quoted caps, ascending in maturity (strike field unused).
        cap_strikes: Absolute strikes of the cap matrix.
        strip: The true batched caplet vols (one
            :class:`~valax.surfaces.caplet_stripping.CapletVolCurve` row per
            cap strike) — piecewise constant per bucket, so a noise-free
            strip recovers it exactly.
    """

    spec: RatesVolSpec
    rates: RatesWorld
    swaptions: tuple
    cube: Any
    caps: tuple
    cap_strikes: Any
    strip: Any

    @property
    def discount_id(self) -> str:
        """Discount (OIS) curve id."""
        return self.rates.spec.ois_id

    @property
    def forward_id(self) -> str:
        """Projection curve id (the first projection tenor, else OIS)."""
        tenors = self.rates.spec.projection_tenors
        return self.rates.spec.projection_id(tenors[0]) if tenors else self.discount_id

    def truth_market(self) -> "MarketState":
        """True curves, cube and caplet strip as a ``MarketState``."""
        from valax.core.market_state import MarketState

        return MarketState(
            curves=self.rates.truth,
            surfaces={self.spec.cube_id: self.cube, self.spec.strip_id: self.strip},
        )


def _swaption_grid(world: RatesWorld, spec: RatesVolSpec) -> tuple:
    """Annual-pay payer swaptions on the (expiry, tenor) grid."""
    from valax.instruments.rates import Swaption

    ref, dc = world.reference_date, world.day_count
    grid = []
    for e in spec.expiry_years:
        expiry = ref + e * _DAYS_PER_YEAR
        grid.append(tuple(
            Swaption(
                expiry_date=expiry,
                fixed_dates=_annual(expiry, t),
                strike=jnp.asarray(0.0),
                notional=jnp.asarray(1.0),
                is_payer=True,
                day_count=dc,
            )
            for t in spec.tenor_years
        ))
    return tuple(grid)


def _cap_strip_caps(world: RatesWorld, spec: RatesVolSpec) -> tuple:
    """Nested quarterly caps, first fixing one quarter after the valuation date."""
    from valax.instruments.rates import Cap

    ref, dc = world.reference_date, world.day_count
    n_max = 4 * max(spec.cap_years)
    edges = ref + (jnp.arange(1, n_max + 2, dtype=jnp.int32) * _DAYS_PER_YEAR) // 4
    caps = []
    for y in spec.cap_years:
        n = 4 * y - 1  # drop the spot-starting period (market convention)
        caps.append(Cap(
            fixing_dates=edges[:n], start_dates=edges[:n], end_dates=edges[1:n + 1],
            strike=jnp.asarray(0.0), notional=jnp.asarray(1_000_000.0),
            is_cap=True, day_count=dc,
        ))
    return tuple(caps)


def sample_rates_vol_world(
    registry: SeedRegistry,
    rates: RatesWorld,
    spec: RatesVolSpec = RatesVolSpec(),
) -> RatesVolWorld:
    """Sample a true SABR swaption cube and true caplet vols on a rates world.

    Cube: one ATM vol level per world (uniform in ``atm_vol_range``) with a
    ±5% uniform dispersion per cell, and rho and nu drawn uniformly per cell,
    with alpha from
    the leading-order backbone (``vol_atm * F^(-beta)`` normal,
    ``vol_atm * F^(1-beta)`` lognormal) on the cell's true forward (computed with
    the pricer's own forward / annuity off the true curves). Caps: per strike
    and bucket, the lognormal SABR vol of a per-bucket smile at the bucket's
    last optionlet — one vol per bucket, the representation the strip
    solves for.

    Stream names: ``"synthetic.universe.cube"``,
    ``"synthetic.universe.caplets"``.

    Args:
        registry: Seed registry.
        rates: The true rates world (curves).
        spec: Volatility market shape.

    Returns:
        The :class:`RatesVolWorld`.
    """
    from valax.surfaces.caplet_stripping import (
        CapletVolCurve,
        _strip_plan,
        optionlet_expiries,
        optionlet_forwards,
    )
    from valax.calibration.sabr import sabr_vol_fn
    from valax.surfaces.swaption_cube import (
        SwaptionCube,
        swaption_node_coordinates,
        swaption_node_forwards_and_annuities,
    )

    vol_fn = sabr_vol_fn(spec.is_normal, spec.shift)

    def backbone_alpha(atm_vol, forward):
        if spec.is_normal:
            return atm_vol * (forward + spec.shift) ** (-spec.beta)
        return atm_vol * forward ** (1.0 - spec.beta)

    disc = rates.truth[rates.spec.ois_id]
    tenors = rates.spec.projection_tenors
    fwd_curve = rates.truth[rates.spec.projection_id(tenors[0])] if tenors else None

    swaptions = _swaption_grid(rates, spec)
    expiries, tenor_axis = swaption_node_coordinates(swaptions, disc)
    forwards, _ = swaption_node_forwards_and_annuities(swaptions, disc, fwd_curve)
    shape = forwards.shape
    k = jax.random.split(registry.key("synthetic.universe.cube"), 4)
    # One ATM level per world with a ±5% cell-level dispersion: a realistic,
    # smooth surface (independent per-cell levels would be a jagged term
    # structure no one-factor model could fit to a co-terminal basket).
    level = jax.random.uniform(k[0], (), minval=spec.atm_vol_range[0], maxval=spec.atm_vol_range[1])
    atm = level * (1.0 + 0.05 * jax.random.uniform(k[3], shape, minval=-1.0, maxval=1.0))
    rho = jax.random.uniform(k[1], shape, minval=spec.rho_range[0], maxval=spec.rho_range[1])
    nu = jax.random.uniform(k[2], shape, minval=spec.nu_range[0], maxval=spec.nu_range[1])
    beta = jnp.full(shape, spec.beta)
    cube = SwaptionCube(
        expiries=expiries, tenors=tenor_axis, forwards=forwards,
        alphas=backbone_alpha(atm, forwards), betas=beta, rhos=rho, nus=nu,
        shift=jnp.asarray(spec.shift), is_normal=spec.is_normal,
        quote_form="atm_offset", dynamics="sticky_moneyness",
    )

    caps = _cap_strip_caps(rates, spec)
    longest = caps[-1]
    index, ends = _strip_plan(caps)
    opt_expiries = optionlet_expiries(longest, disc)
    opt_forwards = optionlet_forwards(longest, disc, fwd_curve)
    last = jnp.asarray([e - 1 for e in ends])
    n_b = len(ends)
    kc = jax.random.split(registry.key("synthetic.universe.caplets"), 3)
    b_atm = jax.random.uniform(kc[0], (n_b,), minval=spec.caplet_vol_range[0], maxval=spec.caplet_vol_range[1])
    b_rho = jax.random.uniform(kc[1], (n_b,), minval=spec.rho_range[0], maxval=spec.rho_range[1])
    b_nu = jax.random.uniform(kc[2], (n_b,), minval=spec.nu_range[0], maxval=spec.nu_range[1])
    f_b, t_b = opt_forwards[last], opt_expiries[last]
    atm_cap = float(jnp.round(jnp.mean(opt_forwards) * 1e4)) * 1e-4
    strikes = atm_cap + jnp.asarray(spec.cap_strike_offsets_bp) * 1e-4

    def bucket_vol(b, strike):
        model = SABRModel(
            alpha=backbone_alpha(b_atm[b], f_b[b]), beta=jnp.asarray(spec.beta),
            rho=b_rho[b], nu=b_nu[b],
        )
        return vol_fn(model, f_b[b], strike, t_b[b])

    bucket_vols = jnp.stack([
        jnp.stack([bucket_vol(b, s) for b in range(n_b)]) for s in strikes
    ])  # (n_strikes, n_buckets)
    n_k = strikes.shape[0]
    strip = CapletVolCurve(
        expiries=jnp.broadcast_to(opt_expiries, (n_k,) + opt_expiries.shape),
        vols=bucket_vols[:, index],
        strike=strikes,
        forwards=jnp.broadcast_to(opt_forwards, (n_k,) + opt_forwards.shape),
        bucket_ends=ends,
        is_normal=spec.is_normal,
    )
    return RatesVolWorld(
        spec=spec, rates=rates, swaptions=swaptions, cube=cube,
        caps=caps, cap_strikes=strikes, strip=strip,
    )


@dataclass(frozen=True)
class RatesVolQuoteSet:
    """Synthetic swaption-vol and cap-vol quotes and their chain nodes.

    Attributes:
        quotes: Noisy vol quotes.
        clean: Noise-free vol quotes.
        fixed: Held-fixed inputs (the cube's SABR beta).
        swaption_keys: ``[i][j][k]`` swaption vol keys.
        cap_keys: ``[strike][cap]`` flat cap vol keys.
        beta_key: Fixed-input key of the cube's SABR beta.
        cube_node: :func:`~valax.calibration.nodes.swaption_cube_node`.
        strip_node: :func:`~valax.calibration.nodes.caplet_strip_node`.
        optionlet_node: :func:`~valax.calibration.nodes.optionlet_sabr_node`.
    """

    quotes: QuoteSet
    clean: QuoteSet
    fixed: QuoteSet
    swaption_keys: tuple
    cap_keys: tuple
    beta_key: str
    cube_node: "CalibrationNode"
    strip_node: "CalibrationNode"
    optionlet_node: "CalibrationNode"


def _flat_cap_vol(
    cap: Any, curve: Any, forward_curve: Any, target: Any, is_normal: bool
) -> Any:
    """Flat vol repricing ``target`` (eager Newton; synthetic data only)."""
    from valax.pricing.analytic.caplets import cap_price_bachelier, cap_price_black76

    pricer = cap_price_bachelier if is_normal else cap_price_black76

    def f(v):
        return pricer(cap, curve, v, forward_curve) - target

    vol = jnp.asarray(0.008 if is_normal else 0.25)
    for _ in range(40):
        vol = vol - f(vol) / jnp.maximum(jax.grad(f)(vol), 1e-12)
    return vol


def synthesize_rates_vol_quote_set(
    registry: SeedRegistry,
    world: RatesVolWorld,
    *,
    vol_bp_noise: float = 0.0,
) -> RatesVolQuoteSet:
    """Vol quotes of the true cube and cap matrix, plus their nodes.

    Swaption keys ``"<CCY>.SWPN.<e>Yx<t>Y.<offset bp>"`` (vol at
    ``F + offset``); cap keys ``"<CCY>.CAP.<n>Y.<strike bp>"`` (flat vol
    repricing the cap under the true caplet vols), both in the spec's quoting
    convention; fixed input
    ``"<CCY>.SWPN.SABR.BETA"``.

    Stream names: ``"synthetic.universe.swaption_vol_noise"``,
    ``"synthetic.universe.cap_vol_noise"``.

    Args:
        registry: Seed registry.
        world: The true rates-volatility market.
        vol_bp_noise: Standard deviation of i.i.d. Gaussian vol noise, in vol
            basis points.

    Returns:
        The :class:`RatesVolQuoteSet`.
    """
    from valax.calibration.nodes import (
        caplet_strip_node,
        optionlet_sabr_node,
        swaption_cube_node,
    )
    from valax.calibration.sabr import sabr_vol_fn
    from valax.pricing.analytic.caplets import cap_price_bachelier, cap_price_black76

    spec, rates = world.spec, world.rates
    vol_fn = sabr_vol_fn(spec.is_normal, spec.shift)
    cap_pricer = cap_price_bachelier if spec.is_normal else cap_price_black76
    ccy = rates.spec.currency
    disc = rates.truth[world.discount_id]
    fwd = rates.truth[world.forward_id] if world.forward_id != world.discount_id else None
    offsets = jnp.asarray(spec.strike_offsets_bp) * 1e-4
    cube = world.cube

    swpn_keys, swpn_vols = [], []
    for i, e in enumerate(spec.expiry_years):
        row_k = []
        for j, t in enumerate(spec.tenor_years):
            keys = tuple(f"{ccy}.SWPN.{e}Yx{t}Y.{int(o):+d}" for o in spec.strike_offsets_bp)
            model = SABRModel(alpha=cube.alphas[i, j], beta=cube.betas[i, j],
                              rho=cube.rhos[i, j], nu=cube.nus[i, j])
            f0 = cube.forwards[i, j]
            swpn_vols.append(jax.vmap(
                lambda o: vol_fn(model, f0, f0 + o, cube.expiries[i])
            )(offsets))
            row_k.append(keys)
        swpn_keys.append(tuple(row_k))
    swpn_flat = [k for row in swpn_keys for keys in row for k in keys]
    swpn_clean = jnp.concatenate(swpn_vols)

    # Cap keys name the maturity exactly as the strip labels its buckets
    # ("cap 2Y" <-> "USD.CAP.2Y.<bp>"), measured from the valuation date.
    from valax.surfaces.caplet_stripping import cap_strip_labels

    maturities = [
        label.removeprefix("cap ")
        for label in cap_strip_labels(world.caps, rates.reference_date)
    ]
    cap_keys, cap_vols = [], []
    for s_idx, strike in enumerate(world.cap_strikes):
        row = []
        for cap, maturity in zip(world.caps, maturities):
            struck = _restrike(cap, strike)
            n = int(struck.fixing_dates.shape[0])
            target = cap_pricer(struck, disc, world.strip.vols[s_idx, :n], fwd)
            cap_vols.append(
                _flat_cap_vol(struck, disc, fwd, target, spec.is_normal)
            )
            row.append(f"{ccy}.CAP.{maturity}.{round(float(strike) * 1e4)}")
        cap_keys.append(tuple(row))
    cap_flat = [k for row in cap_keys for k in row]
    cap_clean = jnp.stack(cap_vols)

    swpn_noise = _gaussian_noise(
        registry, "synthetic.universe.swaption_vol_noise", swpn_clean.shape,
        vol_bp_noise * 1e-4,
    )
    cap_noise = _gaussian_noise(
        registry, "synthetic.universe.cap_vol_noise", cap_clean.shape,
        vol_bp_noise * 1e-4,
    )
    clean = QuoteSet.from_vector(swpn_flat, swpn_clean).merge(
        QuoteSet.from_vector(cap_flat, cap_clean)
    )
    quotes = QuoteSet.from_vector(swpn_flat, swpn_clean + swpn_noise).merge(
        QuoteSet.from_vector(cap_flat, cap_clean + cap_noise)
    )
    beta_key = f"{ccy}.SWPN.SABR.BETA"
    fwd_id = None if world.forward_id == world.discount_id else world.forward_id
    cube_node = swaption_cube_node(
        spec.cube_id,
        swaptions=world.swaptions,
        strikes_per_node=[[offsets for _ in row] for row in world.swaptions],
        quote_keys=swpn_keys,
        discount_id=world.discount_id,
        forward_id=fwd_id,
        reference_date=rates.reference_date,
        quote_form="atm_offset",
        is_normal=spec.is_normal,
        shift=spec.shift,
        beta_key=beta_key,
        name=f"{ccy.lower()}_swaption_cube",
    )
    strip_node = caplet_strip_node(
        spec.strip_id,
        caps=world.caps,
        strikes=world.cap_strikes,
        quote_keys=cap_keys,
        discount_id=world.discount_id,
        forward_id=fwd_id,
        reference_date=rates.reference_date,
        is_normal=spec.is_normal,
        name=f"{ccy.lower()}_cap_strip",
    )
    optionlet_node = optionlet_sabr_node(
        spec.optionlet_id,
        strip_id=spec.strip_id,
        beta=spec.beta,
        is_normal=spec.is_normal,
        shift=spec.shift,
        name=f"{ccy.lower()}_optionlet_sabr",
    )
    return RatesVolQuoteSet(
        quotes=quotes,
        clean=clean,
        fixed=QuoteSet.from_dict({beta_key: spec.beta}),
        swaption_keys=tuple(swpn_keys),
        cap_keys=tuple(cap_keys),
        beta_key=beta_key,
        cube_node=cube_node,
        strip_node=strip_node,
        optionlet_node=optionlet_node,
    )


def _restrike(cap: Any, strike: float) -> Any:
    """``cap`` re-struck at ``strike``."""
    import equinox as eqx

    return eqx.tree_at(lambda c: c.strike, cap, jnp.asarray(strike))


# ── Snapshots ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyntheticSnapshot:
    """Everything an end-to-end run needs: quotes, chain and the truth.

    Attributes:
        quotes: Noisy market quotes.
        clean: Noise-free market quotes.
        fixed: Held-fixed inputs.
        chain: The calibration chain consuming ``quotes`` and ``fixed``.
        truth_market: The true calibrated market.
        parts: The component quote sets (``"curves"``, ``"smile"``), for keys,
            coordinates and nodes.
    """

    quotes: QuoteSet
    clean: QuoteSet
    fixed: QuoteSet
    chain: "CalibrationChain"
    truth_market: "MarketState"
    parts: dict = field(default_factory=dict)


def rates_snapshot(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    spec: RatesWorldSpec = RatesWorldSpec(),
    *,
    bp_noise: float = 0.0,
) -> tuple[SyntheticSnapshot, RatesWorld]:
    """Sample a rates world and quote it.

    Args:
        registry: Seed registry.
        cfg: Reference date, day count, NSS ranges.
        spec: Rates market shape.
        bp_noise: Curve quote noise in basis points.

    Returns:
        ``(snapshot, world)``.
    """
    from valax.calibration.chain import CalibrationChain

    world = sample_rates_world(registry, cfg, spec)
    curves = synthesize_curve_quote_set(registry, world, bp_noise=bp_noise)
    snap = SyntheticSnapshot(
        quotes=curves.quotes,
        clean=curves.clean,
        fixed=QuoteSet(),
        chain=CalibrationChain(nodes=(curves.node,)),
        truth_market=world.truth_market(),
        parts={"curves": curves},
    )
    return snap, world


def equity_snapshot(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    equity_spec: EquityWorldSpec = EquityWorldSpec(),
    *,
    rates_spec: RatesWorldSpec | None = None,
    bp_noise: float = 0.0,
    vol_bp_noise: float = 0.0,
) -> tuple[SyntheticSnapshot, EquityWorld]:
    """Sample an equity world (OIS curve, spot, dividend, smile) and quote it.

    The chain is curves → spot/dividend → SABR smile on the curve-implied
    forwards.

    Args:
        registry: Seed registry.
        cfg: Reference date, day count, NSS ranges.
        equity_spec: Equity market shape.
        rates_spec: Rates shape; defaults to a single OIS curve under the
            ``"_default_"`` sentinel so bare equity instruments resolve it.
        bp_noise: Curve quote noise in basis points.
        vol_bp_noise: Vol quote noise in vol basis points.

    Returns:
        ``(snapshot, world)``.
    """
    from valax.calibration.chain import CalibrationChain

    if rates_spec is None:
        rates_spec = RatesWorldSpec(ois_id=DEFAULT_ID, projection_tenors=())
    rates = sample_rates_world(registry, cfg, rates_spec)
    world = sample_equity_world(registry, rates, equity_spec)
    curves = synthesize_curve_quote_set(registry, rates, bp_noise=bp_noise)
    smile = synthesize_smile_quote_set(registry, world, vol_bp_noise=vol_bp_noise)
    snap = SyntheticSnapshot(
        quotes=curves.quotes.merge(smile.quotes),
        clean=curves.clean.merge(smile.clean),
        fixed=smile.fixed,
        chain=CalibrationChain(nodes=(curves.node, smile.inputs_node, smile.smile_node)),
        truth_market=world.truth_market(),
        parts={"curves": curves, "smile": smile},
    )
    return snap, world


def rates_vol_snapshot(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    rates_spec: RatesWorldSpec = RatesWorldSpec(),
    vol_spec: RatesVolSpec = RatesVolSpec(),
    *,
    bp_noise: float = 0.0,
    vol_bp_noise: float = 0.0,
) -> tuple[SyntheticSnapshot, RatesVolWorld]:
    """Sample a rates world with a swaption cube and a cap matrix, and quote it.

    The chain is curves → swaption cube (on the curve-implied forwards) and
    curves → cap strip → SABR optionlet surface. A noise-free snapshot
    recovers the true curves, cube and caplet strip to solver tolerance
    (the optionlet SABR fit is a smoothing of the strip, not a truth).

    Args:
        registry: Seed registry.
        cfg: Reference date, day count, NSS ranges.
        rates_spec: Rates market shape (needs a projection tenor of
            ``"3M"`` for quarterly caps to project off the right curve; with
            none, caps and swaptions project off the OIS curve).
        vol_spec: Volatility market shape.
        bp_noise: Curve quote noise in basis points.
        vol_bp_noise: Vol quote noise in vol basis points.

    Returns:
        ``(snapshot, world)``.
    """
    from valax.calibration.chain import CalibrationChain

    rates = sample_rates_world(registry, cfg, rates_spec)
    world = sample_rates_vol_world(registry, rates, vol_spec)
    curves = synthesize_curve_quote_set(registry, rates, bp_noise=bp_noise)
    vols = synthesize_rates_vol_quote_set(registry, world, vol_bp_noise=vol_bp_noise)
    snap = SyntheticSnapshot(
        quotes=curves.quotes.merge(vols.quotes),
        clean=curves.clean.merge(vols.clean),
        fixed=vols.fixed,
        chain=CalibrationChain(nodes=(
            curves.node, vols.cube_node, vols.strip_node, vols.optionlet_node,
        )),
        truth_market=world.truth_market(),
        parts={"curves": curves, "vols": vols},
    )
    return snap, world


# ── Quote scenarios ──────────────────────────────────────────────────


def sample_quote_scenarios(
    registry: SeedRegistry,
    quotes: QuoteSet,
    n_scenarios: int,
    *,
    rate_tenors: dict[str, float] | None = None,
    vol_coords: dict[str, tuple[float, float]] | None = None,
    spot_keys: Sequence[str] = (),
    rate_level_bp: float = 10.0,
    rate_slope_bp: float = 5.0,
    vol_level_bp: float = 100.0,
    vol_skew_bp: float = 50.0,
    spot_rel: float = 0.02,
    spot_vol_corr: float = -0.7,
    stream_name: str = "synthetic.universe.scenarios",
) -> QuoteSet:
    """Stacked quote scenarios with a few interpretable factors.

    * Rates: a parallel move plus a slope move linear in maturity (10Y pivot).
    * Vols: a parallel move plus a skew move linear in moneyness, with the
      parallel vol move correlated with the spot return (``spot_vol_corr``).
    * Spots: multiplicative lognormal returns.

    Keys absent from every group are left unchanged. The result has a leading
    scenario axis on every value and feeds ``jax.vmap`` over
    :func:`valax.calibration.chain.calibrate_market` directly.

    Args:
        registry: Seed registry.
        quotes: Base quotes.
        n_scenarios: Number of scenarios.
        rate_tenors: ``rate key -> maturity in years``.
        vol_coords: ``vol key -> (expiry, moneyness)``.
        spot_keys: Spot quote keys.
        rate_level_bp: Standard deviation of the parallel rate move (bp).
        rate_slope_bp: Standard deviation of the slope move at 10Y (bp).
        vol_level_bp: Standard deviation of the parallel vol move (vol bp).
        vol_skew_bp: Standard deviation of the skew move per 20% moneyness.
        spot_rel: Standard deviation of the spot log-return.
        spot_vol_corr: Correlation of spot return and parallel vol move.
        stream_name: Seed stream.

    Returns:
        A :class:`~valax.market.quotes.QuoteSet` whose values have shape
        ``(n_scenarios,)``.
    """
    z = jax.random.normal(registry.key(stream_name), (5, n_scenarios), dtype=jnp.float64)
    rate_level, rate_slope, spot_z, vol_ind, vol_skew = z
    vol_level = spot_vol_corr * spot_z + jnp.sqrt(1.0 - spot_vol_corr ** 2) * vol_ind
    rate_tenors = rate_tenors or {}
    vol_coords = vol_coords or {}
    spot_keys = set(spot_keys)

    out = {}
    for k, v in quotes.values.items():
        if k in rate_tenors:
            shape = (rate_tenors[k] - 10.0) / 10.0
            out[k] = v + 1e-4 * (rate_level_bp * rate_level + rate_slope_bp * shape * rate_slope)
        elif k in vol_coords:
            _, m = vol_coords[k]
            out[k] = v + 1e-4 * (vol_level_bp * vol_level + vol_skew_bp * (m - 1.0) / 0.2 * vol_skew)
        elif k in spot_keys:
            out[k] = v * jnp.exp(spot_rel * spot_z)
        else:
            out[k] = jnp.broadcast_to(v, (n_scenarios,))
    return QuoteSet(values=out)


# ── Products on a synthetic world ────────────────────────────────────


def sample_equity_trades(
    registry: SeedRegistry,
    world: EquityWorld,
    *,
    n_per_kind: int = 2,
    kinds: Sequence[str] = ("european", "barrier", "asian"),
    expiry_range: tuple[float, float] = (0.3, 1.5),
    moneyness_range: tuple[float, float] = (0.9, 1.1),
    barrier_smoothing: float = 0.5,
) -> list:
    """Sample equity trades on a world's underlying.

    Strikes are drawn relative to spot; barriers are up-and-out at 120–140% of
    spot with sigmoid smoothing so pathwise Greeks exist.

    Stream name: ``"synthetic.universe.trades.<kind>"``.

    Args:
        registry: Seed registry.
        world: The equity world.
        n_per_kind: Trades per kind.
        kinds: Any of ``"european"``, ``"barrier"``, ``"asian"``.
        expiry_range: Expiry range (years).
        moneyness_range: Strike / spot range.
        barrier_smoothing: Sigmoid width for barriers (price units).

    Returns:
        A list of instruments.

    Raises:
        ValueError: On an unknown kind.
    """
    trades: list = []
    for kind in kinds:
        k = jax.random.split(registry.key(f"synthetic.universe.trades.{kind}"), 3)
        t = jax.random.uniform(k[0], (n_per_kind,), minval=expiry_range[0], maxval=expiry_range[1])
        m = jax.random.uniform(k[1], (n_per_kind,), minval=moneyness_range[0], maxval=moneyness_range[1])
        b = jax.random.uniform(k[2], (n_per_kind,), minval=1.2, maxval=1.4)
        uid = world.spec.asset_id
        for i in range(n_per_kind):
            strike, expiry = world.spot * m[i], t[i]
            if kind == "european":
                trades.append(EuropeanOption(strike=strike, expiry=expiry, underlying_id=uid))
            elif kind == "barrier":
                trades.append(EquityBarrierOption(
                    strike=strike, expiry=expiry, barrier=world.spot * b[i],
                    is_call=True, is_up=True, is_knock_in=False,
                    smoothing=barrier_smoothing, underlying_id=uid,
                ))
            elif kind == "asian":
                trades.append(AsianOption(strike=strike, expiry=expiry, underlying_id=uid))
            else:
                raise ValueError(f"Unknown trade kind {kind!r}.")
    return trades


def sample_swap_book(
    registry: SeedRegistry,
    world: RatesWorld,
    *,
    n_swaps: int = 6,
    tenor: str | None = None,
    notional_range: tuple[float, float] = (1e6, 5e7),
    off_market_bp: float = 50.0,
) -> list[InterestRateSwap]:
    """Sample annual-coupon swaps that discount on OIS and project on a tenor curve.

    Maturities are drawn from the world's pillar years; fixed rates are the
    true par rate plus a Gaussian off-market spread; direction is random.

    Stream name: ``"synthetic.universe.swaps"``.

    Args:
        registry: Seed registry.
        world: The rates world.
        n_swaps: Number of swaps.
        tenor: Projection tenor; defaults to the first in the spec (or the
            OIS curve itself if the world is single-curve).
        notional_range: Notional range.
        off_market_bp: Standard deviation of the fixed-rate offset (bp).

    Returns:
        A list of :class:`~valax.instruments.rates.InterestRateSwap`.
    """
    from valax.core.market_state import MarketState
    from valax.core.pricing import price

    spec, ref = world.spec, world.reference_date
    if tenor is None and spec.projection_tenors:
        tenor = spec.projection_tenors[0]
    forward_id = spec.projection_id(tenor) if tenor else None
    k = jax.random.split(registry.key("synthetic.universe.swaps"), 4)
    mat_idx = jax.random.randint(k[0], (n_swaps,), 0, len(spec.pillar_years))
    notionals = jax.random.uniform(k[1], (n_swaps,), minval=notional_range[0], maxval=notional_range[1])
    offsets = off_market_bp * 1e-4 * jax.random.normal(k[2], (n_swaps,), dtype=jnp.float64)
    payer = jax.random.bernoulli(k[3], 0.5, (n_swaps,))
    market = MarketState(curves=world.truth)

    swaps = []
    for i in range(n_swaps):
        years = spec.pillar_years[int(mat_idx[i])]
        template = InterestRateSwap(
            start_date=ref, fixed_dates=_annual(ref, years),
            fixed_rate=jnp.asarray(0.0), notional=notionals[i],
            pay_fixed=bool(payer[i]), day_count=world.day_count,
            discount_id=spec.ois_id, forward_id=forward_id,
        )
        # Par rate: PV is affine in the fixed rate, so two prices suffice.
        pv0 = price(template, None, market).price
        bumped = InterestRateSwap(
            start_date=ref, fixed_dates=_annual(ref, years),
            fixed_rate=jnp.asarray(1e-2), notional=notionals[i],
            pay_fixed=bool(payer[i]), day_count=world.day_count,
            discount_id=spec.ois_id, forward_id=forward_id,
        )
        pv1 = price(bumped, None, market).price
        par = -pv0 * 1e-2 / (pv1 - pv0)
        swaps.append(InterestRateSwap(
            start_date=ref, fixed_dates=_annual(ref, years),
            fixed_rate=par + offsets[i], notional=notionals[i],
            pay_fixed=bool(payer[i]), day_count=world.day_count,
            discount_id=spec.ois_id, forward_id=forward_id,
        ))
    return swaps


__all__ = [
    "CurveQuoteSet",
    "EquityWorld",
    "EquityWorldSpec",
    "RatesWorld",
    "RatesVolQuoteSet",
    "RatesVolSpec",
    "RatesVolWorld",
    "RatesWorldSpec",
    "SmileQuoteSet",
    "SyntheticSnapshot",
    "equity_snapshot",
    "rates_snapshot",
    "rates_vol_snapshot",
    "sample_equity_trades",
    "sample_equity_world",
    "sample_quote_scenarios",
    "sample_rates_vol_world",
    "sample_rates_world",
    "sample_swap_book",
    "synthesize_curve_quote_set",
    "synthesize_model_quotes",
    "synthesize_rates_vol_quote_set",
    "synthesize_smile_quote_set",
]
