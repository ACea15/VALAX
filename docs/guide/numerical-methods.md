# Numerical Methods

VALAX prices through four numerical methods. Each has its own guide covering the
engine mechanics, the recipes it supports and the vanilla contract for that engine.

| Method | Guide | Covers |
|---|---|---|
| Analytical | [Analytical Pricing](analytical.md) | Closed-form engines: Black-Scholes-Merton, Black-76, Bachelier, SABR, Garman-Kohlhagen. |
| Monte Carlo | [Monte Carlo](monte-carlo.md) | Path generators, payoff functions and the `(instrument, model)` dispatcher. |
| PDE | [PDE Solvers](pde.md) | Finite-difference solvers: Crank-Nicolson in 1-D, ADI in 2-D. |
| Lattice | [Binomial Trees](lattice.md) | CRR binomial tree and Hull-White trinomial tree. |

**Which instruments each method covers** is stated once, in the solver columns of the
[coverage matrix](../applications/instrument-pricing.md#coverage-matrix). The
guides above deliberately do not repeat that status. For the underlying maths see
[Pricing Methods](../theory/pricing-methods.md), and for how the kernels are routed
through `price()` see the [unified pricing tutorial](tutorial-unified-pricing.md).

Each instrument's payoff maths lives on its asset-class guide; see
[Asset Classes](asset-classes.md).
