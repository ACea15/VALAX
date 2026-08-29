# Models & Theory

This document links the mathematical foundations of quantitative finance to the specific models and implementations in VALAX. The [User Guide](../guide/analytical.md) shows *how* to use each pricing function; this document explains *why* the formulas hold, what assumptions they rest on, and where those assumptions break down.

Throughout, we reference VALAX modules by path (e.g., `valax/pricing/analytic/black_scholes.py`) so you can trace each formula to its implementation.

---

## Contents

This page is the index for the theory library; each chapter lives in its
own page under `theory/`.


- [1. Foundational Framework](foundations.md#1-foundational-framework)
  - [1.1 No-Arbitrage, Martingales, and Risk-Neutral Pricing](foundations.md#11-no-arbitrage-martingales-and-risk-neutral-pricing)
  - [1.2 Itô's Lemma](foundations.md#12-itos-lemma)
  - [1.3 Girsanov's Theorem and Measure Change](foundations.md#13-girsanovs-theorem-and-measure-change)
  - [1.4 The Feynman-Kac Theorem](foundations.md#14-the-feynman-kac-theorem)
- [2. Stochastic Models](stochastic-models.md#2-stochastic-models)
  - [2.1 Black-Scholes / Geometric Brownian Motion](stochastic-models.md#21-black-scholes-geometric-brownian-motion)
  - [2.2 Black-76 (Futures / Forwards)](stochastic-models.md#22-black-76-futures-forwards)
  - [2.3 Bachelier (Normal Model)](stochastic-models.md#23-bachelier-normal-model)
  - [2.4 Heston Stochastic Volatility](stochastic-models.md#24-heston-stochastic-volatility)
  - [2.5 SABR](stochastic-models.md#25-sabr)
  - [2.6 LIBOR Market Model (LMM / BGM)](stochastic-models.md#26-libor-market-model-lmm-bgm)
  - [2.7 Garman-Kohlhagen (FX Options)](stochastic-models.md#27-garman-kohlhagen-fx-options)
  - [2.8 Hull-White One-Factor Short-Rate Model](stochastic-models.md#28-hull-white-one-factor-short-rate-model)
  - [2.9 Two-Asset Correlated BSM and Spread Options](stochastic-models.md#29-two-asset-correlated-bsm-and-spread-options)
- [3. Curve Framework](curves.md#3-curve-framework)
  - [3.1 Discount Factors, Zero Rates, and Forward Rates](curves.md#31-discount-factors-zero-rates-and-forward-rates)
  - [3.2 Single-Curve vs. Multi-Curve Framework](curves.md#32-single-curve-vs-multi-curve-framework)
  - [3.3 Curve Bootstrapping](curves.md#33-curve-bootstrapping)
  - [3.4 Interpolation Methods](curves.md#34-interpolation-methods)
  - [3.5 Day Count Conventions](curves.md#35-day-count-conventions)
  - [3.6 Inflation Curves and Breakeven Pricing](curves.md#36-inflation-curves-and-breakeven-pricing)
  - [3.7 No-Arbitrage Relations Across Curves](curves.md#37-no-arbitrage-relations-across-curves)
  - [3.8 Joint Multi-Curve Calibration](curves.md#38-joint-multi-curve-calibration)
  - [3.9 Futures, Convexity Adjustment, and Fixings](curves.md#39-futures-convexity-adjustment-and-fixings)
- [4. Volatility](volatility.md#4-volatility)
  - [4.1 Implied Volatility](volatility.md#41-implied-volatility)
  - [4.2 The Volatility Surface](volatility.md#42-the-volatility-surface)
  - [4.3 SVI Parameterization](volatility.md#43-svi-parameterization)
  - [4.4 Local Volatility (Dupire)](volatility.md#44-local-volatility-dupire)
  - [4.5 Stochastic-Local Volatility](volatility.md#45-stochastic-local-volatility)
- [5. Pricing Methods](pricing-methods.md#5-pricing-methods)
  - [5.1 Analytical (Closed-Form)](pricing-methods.md#51-analytical-closed-form)
  - [5.2 PDE (Finite Differences)](pricing-methods.md#52-pde-finite-differences)
  - [5.3 Monte Carlo Simulation](pricing-methods.md#53-monte-carlo-simulation)
  - [5.4 Lattice (Binomial Trees)](pricing-methods.md#54-lattice-binomial-trees)
- [6. Greeks and Automatic Differentiation](greeks.md#6-greeks-and-automatic-differentiation)
  - [6.1 Greeks as Derivatives](greeks.md#61-greeks-as-derivatives)
  - [6.2 Automatic Differentiation](greeks.md#62-automatic-differentiation)
  - [6.3 Pathwise Method for MC Greeks](greeks.md#63-pathwise-method-for-mc-greeks)
- [7. Risk Measures](risk-measures.md#7-risk-measures)
  - [7.1 Value at Risk (VaR)](risk-measures.md#71-value-at-risk-var)
  - [7.2 Expected Shortfall (CVaR)](risk-measures.md#72-expected-shortfall-cvar)
  - [7.3 P&L Attribution](risk-measures.md#73-pl-attribution)
  - [7.4 Sensitivity Ladders](risk-measures.md#74-sensitivity-ladders)
  - [7.5 P&L Vectors: Hypothetical, Risk-Theoretical, Actual](risk-measures.md#75-pl-vectors-hypothetical-risk-theoretical-actual)
  - [7.6 VaR Backtesting](risk-measures.md#76-var-backtesting)
  - [7.7 FRTB P&L Attribution Test](risk-measures.md#77-frtb-pl-attribution-test)
  - [7.8 Risk Bucketing: Linear and Jacobian Transformations](risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations)
- [8. Calibration Theory](calibration.md#8-calibration-theory)
  - [8.1 The Calibration Problem](calibration.md#81-the-calibration-problem)
  - [8.2 Levenberg-Marquardt Algorithm](calibration.md#82-levenberg-marquardt-algorithm)
  - [8.3 Parameter Constraints and Transforms](calibration.md#83-parameter-constraints-and-transforms)
  - [8.4 Identifiability and Ill-Conditioning](calibration.md#84-identifiability-and-ill-conditioning)
- [9. Interest-Rate Derivatives](rate-derivatives.md#9-interest-rate-derivatives)
  - [9.1 Change of Numeraire and the Forward Measure](rate-derivatives.md#91-change-of-numeraire-and-the-forward-measure)
  - [9.2 Caps, Floors, and the Caplet Formula](rate-derivatives.md#92-caps-floors-and-the-caplet-formula)
  - [9.3 Swaptions and the Annuity Measure](rate-derivatives.md#93-swaptions-and-the-annuity-measure)
  - [9.4 Bermudan Swaptions and Early Exercise](rate-derivatives.md#94-bermudan-swaptions-and-early-exercise)
  - [9.5 CMS and Convexity Adjustments](rate-derivatives.md#95-cms-and-convexity-adjustments)
  - [9.6 Model Choice for Rate Derivatives](rate-derivatives.md#96-model-choice-for-rate-derivatives)
- [References](references.md#references)

Rates deep-dives (own pages): [Hull-White Monte Carlo](hull-white-mc.md) · [Hull-White Swaptions (Jamshidian)](hull-white-swaptions.md) · [Hull-White Finite Differences](hull-white-pde.md).
