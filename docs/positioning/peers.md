# Peer Tools & Market Segments

*How VALAX compares with the other systems that fill the pricing-and-risk "third ring" — QuantLib, vendor platforms, bank internal libraries, prop and buy-side stacks — and where, across those market segments, an open, differentiable, GPU-vectorised engine is most likely to be useful.*

---

## 1. Peer tools — what else fills the third ring today

VALAX competes for the same job as a small set of well-known systems. The honest comparison is:

| Tool | Origin | Strengths | Where VALAX is structurally different |
|---|---|---|---|
| **QuantLib** | Open source, C++ (1999–) | Enormous instrument and convention breadth; the de-facto cross-check oracle. | Bumped Greeks, mutable engine pattern, CPU-only, no autodiff. VALAX validates against it (see [QuantLib Validation Pyramid](../architecture/quantlib-validation-pyramid.md)) but does not inherit its architecture. |
| **Murex MX.3 / Calypso / Summit / Numerix CrossAsset / FINCAD** | Commercial sell-side platforms | Battle-tested coverage of structured products, lifecycle, integrations, XVA modules. | Closed, vendor-bound, finite-difference Greeks, no GPU/autodiff story. Multi-million-dollar licences. |
| **Bank internal libraries** (JPM Athena, GS SecDB/Slang, BAML Quartz, MS Hydra) | In-house, Python or Slang/Python | Tight integration with golden trade source; firm-specific governance baked in. | Built over 20+ years on mutable object graphs; the *consistency tax* described in [Positioning § 3](index.md#3-what-actually-kills-bank-programmes-ranked) is exactly what motivates VALAX. |
| **Hedge-fund / prop "quant kitchens"** (e.g. Bloomberg DLib, hedge-fund internal pricers) | Mixed, often Python over C++ kernels | Lean, focused on the strategies actually traded; fast iteration. | Often missing rigorous calibration, audit, governance for sell-side use — and almost never autodiff-native or GPU-vectorised across instruments × scenarios. **The picture is more nuanced than the table suggests — see § 2 below.** |
| **Domain-specific JAX libraries** (`diffrax`, `optimistix`, `equinox`, `numpyro`, `flax`) | Open source, JAX ecosystem | The libraries VALAX is built *on*. | Not pricers; they are the differentiable maths VALAX composes into pricers. |

The pattern is consistent: every existing tool in the third ring made architectural choices that pre-date modern autodiff and accelerators. VALAX's bet is that a pure-functional, differentiable, GPU-first kernel — written today, with twenty-five years of QuantLib coverage as a validation oracle — is a strictly better foundation for the next twenty years of pricing-and-risk work than retrofitting any of the above.

For the full *why* of that bet, see [Positioning § 4](index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys).

## 2. Prop houses and HFT — narrower scope, different economics

A caveat up front: **prop trading and HFT are deliberately secretive industries**, and almost nothing about what's actually inside the major firms is public. What follows is a synthesis of what is plausibly inferable from public job postings, conference talks, alumni-authored blog posts, court filings, and the trading behaviour observable in the market itself. Treat it as informed guesswork, not insider knowledge. Specific firms are named as *representatives of a category*, not as claims about their internal stacks.

With that caveat, the popular picture — *"the big banks have decade-old libraries with Heston and Black-Scholes, but so do the prop houses and the HFT firms"* — is likely partly right and mostly misleading. The space appears to be sharply stratified, and the stratification matters for placing VALAX honestly. Roughly four groups:

**1. Bank-style libraries.** Athena (JPM), SecDB/Slang (Goldman), Quartz (BAML), Hydra (Morgan Stanley); or the vendor equivalents Murex MX.3, Calypso, Numerix CrossAsset. These are the best-documented of the four groups because banks publish papers, alumni write books (e.g. Joshi, Hull, the Athena/SecDB literature), and regulators force disclosure of broad architectural shape. They cover hundreds of instrument types across every asset class, with decades of accumulated convention handling, model governance baked in, and integration into golden-source trade stores. They are slow to change, expensive to run, and structurally what VALAX is designed to challenge. The § 1 comparison stands for this group.

**2. Prop options market makers — likely narrow, fast, and proprietary.** Optiver, IMC, SIG, Akuna, Wolverine, CTC, Flow Traders, the listed-options books at Jane Street and Citadel Securities. These firms make markets in listed options at scale, and several things about the business *suggest* they build their own pricing infrastructure rather than license a bank-style library:

- Their product universe — listed equity / index / ETF / futures options — is narrow compared to a bank's cross-asset book. The maths needed (Black-Scholes, CRR or trinomial for American, a smile model, dividends, borrow rates, maybe a jump term for short-dated) is a small, stable syllabus that has not fundamentally changed since the 1990s.
- Their latency requirements are well below what any bank or vendor library targets, which makes a tuned in-house pricer plausibly easier than retrofitting.
- Public job postings at these firms routinely advertise C++ / quant-developer roles building "low-latency pricing infrastructure", "vol surface fitting", and "Greeks computation". This is consistent with in-house libraries; it isn't proof of them.
- The economics appear to support building rather than licensing — these are profitable firms with quant headcounts in the tens to low hundreds, against revenue lines large enough to absorb the investment.

What is much less clear from the outside is *how much* of their stack is bespoke versus glued together from open-source components and proprietary calibration on top. **VALAX is most likely not a direct hot-path competitor for this group** — firms that have already invested in a tuned, latency-optimised pricer are unlikely to swap it for a general-purpose library. The more honest pitch is *augmentation*: VALAX for the **batch overnight risk run**, **portfolio-level Greeks across the book**, and the **what-if / scenario layer** that a latency-tuned hot-path pricer is typically not designed for. The hot path stays theirs; the slow path is where an open, differentiable, GPU-vectorised engine has something distinctive to offer.

**3. Pure HFT — probably a different conversation entirely.** Virtu, Tower, Hudson River Trading, Two Sigma Securities, XTX, Jump's HFT book, the cash-equity / futures / ETF businesses inside Citadel Securities. For firms in this group whose primary business is cash equities, futures basis, ETF-vs-constituent arbitrage, or rates cash-vs-futures arb, public discussion of their stacks (conference talks, ex-employee accounts, court documents from various lawsuits) consistently points away from option-pricing-model machinery and towards *fair-value models* — predictors of the next mid from order-book state and microstructure features, often implemented in lean C++ or pinned to FPGA. Risk in this world tends to be positional ("flat by close") rather than Greek-based, and the "valuation" of a position is typically the venue mid rather than a model output. **It is hard to see a natural role for VALAX in this style of business**, and we should not claim one. When a firm in this category also runs a listed-options book — Virtu and Citadel Securities both do — that book is widely understood to be run separately, on infrastructure that looks much more like group 2.

**4. The buy-side spectrum — heterogeneous and the most VALAX-relevant.** Hedge funds and asset managers vary enormously, and unlike banks they face essentially no public-disclosure pressure on their internal stacks. The picture below is a rough sketch of categories rather than a claim about any individual firm:

- **Vol-arb / dispersion / structured-note funds** (Capstone, Parallax, the vol pods inside multi-strats) — likely run proprietary pricers, narrowly scoped, with the same economics as group 2.
- **Rates RV and macro funds** (Brevan, Rokos, Element, Caxton) — plausibly a Python calibration layer on top of QuantLib or a wrapped bank library, plus a handful of proprietary models for the specific trades they put on, though the mix varies.
- **Convertible arb, distressed credit, structured-credit funds** — coverage ranges from senior-quant hand-rolled pricers to licensed Numerix / Kamakura installs to thin wrappers around vendor analytics. Smaller funds in this space generally cannot afford to build *or* license fully and likely live with the resulting precision loss.
- **Multi-strat pod shops** (Millennium, P72, Balyasny, ExodusPoint, Schonfeld) — the public picture, reinforced by their hiring patterns, is that there is no firm-wide pricing library; each pod brings or builds its own tools, and the firm provides risk aggregation on top, often based on pod-reported sensitivities rather than independent repricing.
- **Systematic equity / CTA funds** (Two Sigma, AQR, Renaissance, Winton, Man AHL) — to the extent these firms trade options at all, the pricer is generally not the part of the stack they invest in; their proprietary code is the *signal*, not the valuation.

**Where VALAX likely fits.** Pulling the four groups together: banks have built their libraries over decades because regulation and product breadth force them to cover everything; prop options market makers appear to have built narrower ones because a small team can plausibly own the whole syllabus for a focused product set; pure HFT firms don't seem to need pricing libraries in the bank sense at all; and the buy-side picture is patchy. The audience for whom **building** a bank-style stack is uneconomical and **licensing** Numerix or Murex is overkill — multi-strat hedge-fund pods, mid-sized vol funds, sophisticated family offices, new prop desks spinning up a derivatives book, and arguably the *slow-path risk and what-if layer* sitting behind a group-2 prop options market maker — is the audience where a credible open, differentiable, GPU-vectorised third-ring component appears most likely to be the missing option today. That is where we expect VALAX to have its strongest pull. Whether any specific firm sees it that way is, of course, something only that firm can tell us.

---

## 3. Buy-side counterpart — what changes at a hedge fund or asset manager

The [positioning map](index.md#2-the-bank-hedge-fund-stack-at-a-glance) is bank-flavoured because banks publish the most about their stacks. The same picture holds on the buy side with three substitutions:

1. **Trade volumes and product zoo are narrower.** A vol-arb fund cares about a few hundred listed options and variance swaps; an LDI manager cares about IR/inflation swaps and bonds. Coverage requirements are smaller, but the maths is the same.
2. **Regulatory ring is lighter but not empty.** No FRTB IMA, but SIMM for uncleared margin, SEC/CFTC reporting for swaps, and AIFMD/UCITS risk reporting in Europe. The bucketed-sensitivities outputs from VALAX feed these the same way they feed bank reports.
3. **The biggest win shifts from XVA to differentiable strategy research.** Buy-side workflows that today are prohibitively expensive — gradient-based portfolio construction with derivatives in the loop, deep hedging of structured-note books, Bayesian posteriors on calibrated parameters — are the natural consumers of VALAX's frontier layer (roadmap P5). See [Vision §3 and §4](../vision.md) for the table mapping buy-side workflows to VALAX features.
