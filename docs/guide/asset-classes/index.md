# Asset Classes

Each instrument's payoff or cashflow maths and its how-to recipe live on exactly one
asset-class guide below.

| Asset class | Guide | Covers |
|---|---|---|
| Fixed income | [Fixed Income](../fixed-income.md) | Date utilities, discount curves, bonds, FRNs, OIS and multi-leg swaps, duration and key-rate risk. |
| Bonds with optionality | [Callable/Puttable/FRN/Convertible Bonds](../callable-bonds.md) | Floating-rate, callable, puttable and convertible bonds. |
| Interest rate exotics | [Interest Rate Exotics](../rates-exotics.md) | OIS, cross-currency, total return, CMS swaps, CMS caps/floors, range accruals. |
| FX | [FX Derivatives](../fx.md) | Quanto options, TARFs, FX swaps (FX forwards and vanilla options are in [Analytical Pricing](../analytical.md#fx-options-garman-kohlhagen)). |
| Inflation | [Inflation Derivatives](../inflation.md) | Zero-coupon and year-on-year inflation swaps, inflation caps/floors. |
| Equity | [Equity Exotics](../equity-exotics.md) | Digital, compound, chooser, spread, autocallable, worst-of basket and cliquet options. |
| Credit | [Credit Derivatives](../credit.md) | CDS, CDO tranches and survival curves. |

**Instrument × solver status** is stated once, in the
[coverage matrix](../../applications/instrument-pricing.md#coverage-matrix). The
numerical engines that price these instruments are described under
[Numerical Methods](../numerical-methods/index.md).
