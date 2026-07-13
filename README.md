This repository contains the source of the **O25 Cosmochrony paper**  
*Systematic Pair-Level Campaign for δpair:
Convergence, Inter-Pair Concentration, and Normalization Structure*.

This work extends the **spectral admissibility sub-programme** by providing
the first **systematic numerical validation of the pair observable**
introduced and structurally stabilised in **O16–O24**.

It addresses the final numerical question left open after **O24**:

> Can the capacity exponent δpair be robustly measured across primes,
> and if so, what controls its apparent variation?

## Quick Summary

δpair does not vary across primes.

The apparent drift comes from finite-size effects in the observable:
measurements are controlled by a BFS window whose depth is not yet asymptotic.

As a result, standard extrapolations in q are structurally misleading.

The correct scaling variable is not q, but n₁(q)/q.

Once this is accounted for, all values are consistent with the expected range.

## Context

**O16–O24** established that:

- the physically relevant observable is the canonical pair-level quantity  
  $\sigma_{\mathrm{pair}}^{\mathrm{can}}(n)$
- the exponent $\delta_{\mathrm{pair}} \approx 7.44$ lies within the admissible window  
  $[7.4, 10.6]$
- the transfer chain  
  $c_{\mathrm{BI}} \to \delta_{\mathrm{pair}} \to \beta^*$  
  holds unconditionally (**O24**)

However:

- δpair had only been computed for **single pairs per prime**
- no systematic inter-pair analysis was available
- observed variations across primes remained unexplained
- extrapolation towards an asymptotic value δ∞ appeared unstable

This defines the scope of **O25**.

## Core Result

The paper establishes that:

> The apparent variability of δpair across primes is not physical,
> but a finite-size normalization effect controlled by the BFS window depth.

Three key results are obtained:

- **convergence** of δpair(q) across primes
- **inter-pair concentration** → δpair is a structural invariant
- **non-identifiability of δ∞** via naive extrapolation in q

The correct asymptotic variable is identified as:

$$
\frac{n_1(q)}{q}
$$

not q itself.

## Main Structural Results

### 1. Inter-pair concentration

*Result.* The variance of δpair across conjugate pairs vanishes with q.

Thus:

- δpair is not a block-dependent quantity
- it is a **structural invariant of the Weil representation**

### 2. Convergence of δpair(q)

*Result.* The sequence $\bar{\delta}_{\mathrm{pair}}(q)$ stabilises with q.

Thus:

- fluctuations observed in earlier works were finite-sample effects
- convergence is robust across all pairs

### 3. Degeneracy of empirical fits

*Result.* Multiple functional forms fit the data equally well:

- $\delta_\infty + a / \log q$
- $\delta_\infty + a / (\log q)^2$
- $\delta_\infty + a / q^\alpha$

but yield incompatible δ∞.

Thus:

- extrapolation in q is **structurally ill-posed**
- identifiability of δ∞ fails

### 4. Origin of the obstruction

The leading correction behaves as:

$$
\frac{\log q}{\log n_1(q)} \approx 1
$$

with:

- $n_1(q) = \Theta(q)$ (O9)

Thus:

- finite-size corrections are nearly constant in log q
- multiple asymptotic models become indistinguishable

The obstruction is therefore:

> structural, not numerical.

### 5. Normalization correction (O14)

Applying the O14 correction:

$\delta_{\mathrm{corr}}(q) = \delta_{\mathrm{pair}}(q) - \eta \frac{\log q}{\log n_1(q)}$

with $\eta = 1/2$ yields:

- $\delta_{\mathrm{corr}}(q) \in [7.7, 9.0]$

Thus:

- all corrected values fall within the admissible window
- the apparent downward drift is a normalization artefact

### 6. Identification of the correct asymptotic variable

*Result.* The controlling quantity is:

$$
\frac{n_1(q)}{q}
$$

Thus:

- q is not the correct scaling parameter
- convergence is governed by BFS window depth

## Foundational Chain from the Substrate

The derivation is fully internal:

Born–Infeld admissibility  
$\to$ canonical pair observable (O16–O21)  
$\to$ projection locking (O22)  
$\to$ quaternionic maximality (O23)  
$\to$ rank stability (O24)  
$\to$ pair-level numerical convergence (O25)  
$\to$ normalization structure  
$\to$ identification of asymptotic variable

No external fitting assumption is required.

## Mathematical Role of O25

**O25** provides the numerical closure of the admissibility framework:

- it validates δpair across all conjugate pairs
- it proves inter-pair concentration
- it explains the instability observed in O12–O13
- it identifies the structural origin of finite-size effects
- it invalidates naive extrapolation in q
- it isolates the correct asymptotic variable

More precisely, the paper:

- performs the first full pair-level campaign
- measures δpair distributions per prime
- establishes convergence and variance collapse
- proves degeneracy of empirical fits
- derives the normalization correction mechanism
- identifies $n_1(q)/q$ as the controlling quantity

## Epistemic Structure of the Paper

### Established input

- canonical pair observable (**O16–O21**)
- projection locking (**O22**)
- quaternionic maximality (**O23**)
- rank invariance (**O24**)
- normalization correction (**O14**)
- Heisenberg/Weil pipeline

### New results

- full pair-level numerical campaign
- inter-pair concentration result
- degeneracy of extrapolation fits
- identification of normalization structure
- asymptotic variable shift (q → n₁(q)/q)

### Remaining open problems

- analytical derivation of $n_1(q)/q$
- validation of $\eta = 1/2$ in full pipeline
- large-q regime (q ≥ 211 and beyond)
- next-to-leading order corrections
- analytical determination of δ∞

## Interpretation of the Result

The conceptual shift is decisive:

- previous view: δpair varies with q
- O25: δpair is stable, variation is artefactual

Thus:

- fluctuations are not physical
- they are induced by normalization and window effects

The key insight is:

> asymptotic behaviour is controlled by observable geometry,
> not by the external parameter q.

## Structural Role of O25

**O25** completes the sequence:

- **O16**: pair observable
- **O17–O19**: fibre structure and normalization
- **O20–O21**: persistence and intrinsic saturation
- **O22**: projection locking
- **O23**: quaternionic threshold
- **O24**: rank stability
- **O25**: numerical convergence and asymptotic structure

Thus:

- the observable is validated
- the variability is explained
- the asymptotic obstruction is identified
- the correct scaling variable is isolated

## What O25 Adds

- full pair-level numerical validation
- inter-pair concentration
- explanation of δ instability
- structural origin of finite-size corrections
- degeneracy of extrapolation
- identification of the true asymptotic variable

## Outcome

The spectral admissibility framework is now:

- structurally grounded (**O24**)
- numerically validated (**O25**)
- normalization-aware
- asymptotically well-posed (at the structural level)

The remaining task is now purely analytical.

## Residual Open Problems

### Asymptotic ratio

Determine the limit of:

$$
n_1(q)/q \to \alpha
$$

### Next-order corrections

Derive subleading corrections to δpair(q).

### Large-q regime

Extend computations to larger primes.

### Analytical δ∞

Compute δ∞ without numerical extrapolation.

### Pipeline closure

Fully integrate normalization and asymptotic structure.

## Status

The programme is now:

- structurally closed (**O24**)
- numerically validated (**O25**)
- free of extrapolation artefacts
- ready for analytical completion

## Repository Structure

```text
paper/
├── out/      # Compiled O25 PDF
├── tex/      # LaTeX sources
└── README.md
```
# Citation

If you reference this work, please cite:

J. Beau
Systematic Pair-Level Campaign for δpair:
Convergence, Inter-Pair Concentration, and Normalization Structure
Zenodo, 2026.

# Acknowledgements

Portions of the derivations, conceptual synthesis, structural organisation,
and editorial refinement benefited from iterative interactions with large
language models used as analytical assistants.

All theoretical results, computations, and interpretations remain the sole
responsibility of the author.

# Contributions

This repository is intended as a research reference.

Critical feedback, independent verification, and further analysis of:

- pair-level observables
- normalization structure
- asymptotic scaling
- BFS window effects
- spectral admissibility

are welcome.

Please open an issue to discuss conceptual points, technical details, or
possible extensions.
