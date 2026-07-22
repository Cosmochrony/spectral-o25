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

At fixed $q$, the pair observable concentrates strongly across conjugate pairs.
Across primes, its fitted exponent drifts over the accessible finite windows, and the
available data do not identify a unique asymptote.

As a result, standard extrapolations in q are structurally misleading.

The candidate finite-window control variable is $n_1(q)/q$ rather than $q$ alone.

The raw large-$q$ values enter the historical phenomenological window, while the
earlier O14 correction eventually overcorrects.

## Context

**O16–O24** established that:

- the physically relevant observable is the canonical pair-level quantity  
  $\sigma_{\mathrm{pair}}^{\mathrm{can}}(n)$
- the reference value $\delta_{\mathrm{pair}} \approx 7.44$ lies within the historical
  phenomenological window
  $[7.4, 10.6]$
- vertical fibre structure does not change the observable rank (**O24**)

The further map
$\beta^*=1/(\delta_{\mathrm{pair}}+1/2)$ is not a native Heisenberg law: it imports a
changing-degree LPS growth equation into the fixed-degree Heisenberg cascade. O25 therefore
retains values produced by that reciprocal only as phenomenological diagnostics.

However:

- δpair had only been computed for **single pairs per prime**
- no systematic inter-pair analysis was available
- observed variations across primes remained unexplained
- extrapolation towards an asymptotic value δ∞ appeared unstable

This defines the scope of **O25**.

## Core Result

The paper establishes that:

> The pair statistic concentrates at fixed $q$, while its cross-prime drift remains
> entangled with estimator and BFS-window effects.

Three key results are obtained:

- **reproducible extraction** of δpair(q) across primes
- **inter-pair concentration** → δpair is a stable fixed-$q$ pair statistic
- **non-identifiability of δ∞** via naive extrapolation in q

The candidate variable controlling the finite-window analysis is:

$$
\frac{n_1(q)}{q}
$$

not q itself.

## Main Structural Results

### 1. Inter-pair concentration

*Result.* The variance of δpair across conjugate pairs decreases over the sampled primes.

Thus:

- δpair is not dominated by the choice of conjugate pair at fixed $q$
- the concentration is compatible with O24 rank stability

### 2. Cross-prime drift of δpair(q)

*Result.* The sequence $\bar{\delta}_{\mathrm{pair}}(q)$ is reproducibly measured but
does not determine a unique asymptote on the accessible prime range.

Thus:

- pair-to-pair fluctuations shrink over the sampled range
- the cross-prime asymptote remains unresolved

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

This small-$q$ endpoint diagnostic was historically consistent with the phenomenological
window. The large-$q$ extension shows that it eventually overcorrects, so it is not a native
exponent conversion and does not establish that the drift is purely a normalization artefact.

### 6. Candidate asymptotic variable

*Diagnostic.* The quantity to track is:

$$
\frac{n_1(q)}{q}
$$

Thus the BFS window depth must be tracked explicitly; the present finite range does not
prove a unique asymptotic scaling law.

## Foundational Chain from the Substrate

The observable-side chain is internal:

Born–Infeld admissibility  
$\to$ canonical pair observable (O16–O21)  
$\to$ projection locking (O22)  
$\to$ quaternionic maximality (O23)  
$\to$ rank stability (O24)  
$\to$ pair-level numerical convergence (O25)  
$\to$ normalization structure  
$\to$ identification of asymptotic variable

No external fitting assumption is required for the reported pair statistics.
This chain does not derive the separate LPS-to-Heisenberg capacity-to-rate transfer.

## Mathematical Role of O25

**O25** provides the numerical closure of the admissibility framework:

- it validates δpair across all conjugate pairs
- it proves inter-pair concentration
- it explains the instability observed in O12–O13
- it identifies the structural origin of finite-size effects
- it invalidates naive extrapolation in q
- it isolates the candidate finite-window control variable

More precisely, the paper:

- performs the first full pair-level campaign
- measures δpair distributions per prime
- establishes convergence and variance collapse
- proves degeneracy of empirical fits
- derives the normalization correction mechanism
- identifies $n_1(q)/q$ as the quantity to test in a future asymptotic analysis

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
- O25: the fixed-$q$ pair statistic concentrates, while cross-prime variation remains unresolved

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
- the finite-window variability is quantified
- the asymptotic obstruction is identified
- a candidate scaling variable is isolated

## What O25 Adds

- full pair-level numerical validation
- inter-pair concentration
- quantification of δ instability
- structural origin of finite-size corrections
- degeneracy of extrapolation
- identification of the candidate finite-window control variable

## Outcome

The spectral admissibility framework is now:

- structurally grounded (**O24**)
- numerically validated (**O25**)
- normalization-aware
- explicit about its unresolved asymptotic regime

The remaining tasks include an asymptotic analysis of the estimator and any future native
growth law; the legacy reciprocal is not an available closure.

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
