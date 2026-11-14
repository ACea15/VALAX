# Rates: Products and Models

[← Models & Theory index](../index.md)

This section is the theory map for **interest-rate products**: what they are,
how they split into the part that needs only a curve and the part that needs a
model, and where each piece of the maths lives. It owns no derivation and no
status table — it routes you to the page that does.

| If you want… | Go to |
|---|---|
| which instrument is priced by which engine, and its status | [Instrument Coverage Matrix](../../applications/instrument-pricing.md#interest-rate-derivatives) |
| an instrument's payoff, cashflows and how-to recipe | [Fixed Income](../../guide/fixed-income.md), [Callable/Puttable Bonds](../../guide/callable-bonds.md), [Interest Rate Exotics](../../guide/rates-exotics.md) |
| a worked end-to-end calibration-to-risk example | [Tutorial — Rates End-to-End](../../guide/tutorial-rates.md) |

---

## 1 · Three layers of rates products

Rates products fall into three layers. Each adds one modelling ingredient to
the one below, which is also the order to read the theory in.

| Layer | Products | What prices it | Theory |
|---|---|---|---|
| **1. Curve-only (linear)** | Zero-coupon and fixed-rate bonds, floating-rate notes, interest-rate / OIS / amortising / seasoned / basis swaps, cross-currency swaps, total-return swaps | **Discounting** projected cashflows on the curves. No volatility, no model: the floating leg telescopes against the discount curve. | [Curve Framework (§3)](../curves.md#3-curve-framework), esp. [§3.2 multi-curve](../curves.md#32-single-curve-vs-multi-curve-framework) |
| **2. Vanilla optionality** | Caplets, caps/floors, European swaptions | **A choice of numéraire** that makes the forward or swap rate a martingale, collapsing the price to Black-76 / Bachelier. Needs a volatility surface (optionlet surface, swaption cube), not a dynamic model. | [Interest-Rate Derivatives (§9.1–9.3)](../rate-derivatives.md#91-change-of-numeraire-and-the-forward-measure) |
| **3. Model-dependent** | Bermudan swaptions, callable/puttable bonds, CMS swaps and caps/floors, CMS-spread swaps, range accruals | **A term-structure model**: early exercise needs the joint dynamics of the curve; CMS needs a convexity adjustment; spread products need decorrelation between tenors. | [§9.4–9.6](../rate-derivatives.md#94-bermudan-swaptions-and-early-exercise) and the model chapters below |

Layers 2 and 3 together are the **rate derivatives**; layer 1 is their input and
their hedge.

### How the layers connect

- The curves of layer 1 are *inputs* to everything above: the discount curve
  supplies the numéraire and the projection curve supplies the forwards.
  Whether those are one curve or several is the subject of
  [Why Multi-Curve?](../../guide/why-multicurve.md).
- A layer-2 swaption is the building block of layer 3: the market quotes
  swaptions, so models are **calibrated to swaptions** and then used to price
  the exotics. The calibration side is in
  [Calibration Theory](../calibration.md) and
  [Calibration Chains](../calibration-chains.md).
- The price of a layer-1 swap is the *underlying* of a swaption, and
  put–call parity ties caps/floors back to it
  ([§9.2](../rate-derivatives.md#92-caps-floors-and-the-caplet-formula)).

---

## 2 · Rate derivatives

The derivatives section starts from the model-free numéraire machinery and then
moves to the models and numerical engines, in the order they are needed:

| Step | Page | What it adds |
|---|---|---|
| 1 | [Interest-Rate Derivatives (§9)](../rate-derivatives.md) | Forward measures, caplets, swaptions, Bermudans, CMS convexity, and which model to choose |
| 2 | [Hull-White MC: Exact Conditional Simulation](../hull-white-mc.md) | One-factor Gaussian short rate: exact path simulation |
| 3 | [Hull-White Swaptions: Jamshidian Decomposition](../hull-white-swaptions.md) | The closed-form European swaption that makes Hull-White calibratable |
| 4 | [Hull-White Finite Differences](../hull-white-pde.md) | Bermudans and callables by backward PDE with an exercise projection |
| 5 | [Time-Dependent Volatility](../time-dependent-volatility.md) | Piecewise-constant σ(t) shared by Hull-White, G2++ and Cheyette; the general tree |
| 6 | [G2++: Two-Factor Gaussian](../g2pp.md) | Imperfect correlation between tenors; CMS-spread products |
| 7 | [Cheyette: Quasi-Gaussian Skew](../cheyette.md) | State-dependent volatility, i.e. a rate skew, in a low-dimensional HJM |
| 8 | [Multi-Curve Short Rates](../multi-curve-short-rate.md) | The deterministic tenor basis that lets one exact-fit model price with OIS discounting and a distinct projection curve |

The background models these pages build on are in
[Stochastic Models](../stochastic-models.md): [SABR §2.5](../stochastic-models.md#25-sabr),
[LMM §2.6](../stochastic-models.md#26-libor-market-model-lmm-bgm) and
[Hull-White §2.8](../stochastic-models.md#28-hull-white-one-factor-short-rate-model).
