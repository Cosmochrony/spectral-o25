# o_stability_diagnostic.py
#
# Diagnostic: mode-level stability along the BFS cascade
#
# For each prime q and each conjugate pair (c, q-c), tracks sigma_c(n) as a
# function of BFS depth n and tests whether the "effective projective rank"
# I(U_t) -- the number of modes whose sigma_can_pair(n) exceeds a threshold --
# is monotone increasing (no regression).
#
# The test addresses the stability condition required for the identification
#   time ~ monotone ordering of I(U_t)
# which is conjectured in the companion conceptual development.
#
# Output:
#   - Per-prime trace plots: sigma_pair(n) vs n for each pair (c, q-c)
#   - Crossing events: first n at which sigma_pair(n) falls below threshold
#   - Regression count: number of pairs that regress after crossing
#   - Summary table: per-prime stability verdict
#
# Usage:
#   python o_stability_diagnostic.py --npz-dir ./npz --primes 29 61 101
#   (npz files produced by o25_paired_pipeline.py)

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Threshold: fraction of the global max of sigma_pair(n) that counts as
# "having crossed the admissibility threshold".  Two regimes tested:
#   - absolute: sigma_pair(n) > THRESHOLD_ABS
#   - relative: sigma_pair(n) > THRESHOLD_REL * max_n(sigma_pair)
THRESHOLD_REL = 0.5   # fraction of per-pair maximum
THRESHOLD_ABS = 1e-3  # absolute floor (avoids numerical noise)

# Minimum BFS depth before we start tracking crossings
N_BURN_IN = 2

# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def load_pair_data(npz_path: Path) -> dict:
    """Load the npz file produced by o25_paired_pipeline.py.

    Actual keys used:
        q                : scalar int
        ns               : (n_depths,)         -- BFS depths
        pairs            : (n_pairs, 2)         -- (c, q-c) indices
        sigma_pair_mean  : (n_pairs, n_depths)  -- per-pair sigma traces
        n0, n1           : scalar ints          -- fitting window bounds
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {}
    result["q"]    = int(data["q"])
    result["depths"] = data["ns"]                  # shape (n_depths,)
    result["pairs"]  = data["pairs"]               # shape (n_pairs, 2)
    # sigma_pair_mean is (n_pairs, n_depths) -- per-pair traces
    result["sigma_pair_all"]  = data["sigma_pair_mean"]   # (n_pairs, n_depths)
    result["sigma_pair_mean"] = data["sigma_pair_mean"].mean(axis=0)  # global mean
    result["n0"] = int(data["n0"])
    result["n1"] = int(data["n1"])
    return result


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def analyze_stability(sigma_traces: np.ndarray, depths: np.ndarray,
                      threshold_rel: float = THRESHOLD_REL,
                      threshold_abs: float = THRESHOLD_ABS,
                      n_burn_in: int = N_BURN_IN) -> dict:
    """Analyze stability of mode crossings.

    Parameters
    ----------
    sigma_traces : (n_pairs, n_depths) array
        Per-pair sigma_pair(n) traces.
    depths : (n_depths,) array
        BFS depths corresponding to columns.

    Returns
    -------
    dict with keys:
        n_pairs           : int
        n_crossings       : number of pairs that crossed threshold
        n_regressions     : number of pairs that regressed after crossing
        regression_pairs  : list of pair indices that regressed
        first_crossing_n  : array (n_pairs,) -- depth of first crossing, nan if none
        stability_ratio   : n_crossings / n_pairs  (fraction that crossed)
        monotone_ratio    : (n_crossings - n_regressions) / n_crossings
    """
    n_pairs, n_depths = sigma_traces.shape
    burn_mask = depths >= n_burn_in

    first_crossing_n = np.full(n_pairs, np.nan)
    n_regressions = 0
    regression_pairs = []

    for i in range(n_pairs):
        trace = sigma_traces[i]
        # Per-pair adaptive threshold
        thr = max(threshold_abs, threshold_rel * np.nanmax(trace))
        # Find first crossing after burn-in
        crossed = np.where(burn_mask & (trace > thr))[0]
        if len(crossed) == 0:
            continue
        first_idx = crossed[0]
        first_crossing_n[i] = depths[first_idx]
        # Check for regression: any subsequent point below threshold
        post = trace[first_idx + 1:]
        if np.any(post < thr):
            n_regressions += 1
            regression_pairs.append(i)

    n_crossings = int(np.sum(~np.isnan(first_crossing_n)))
    stability_ratio = n_crossings / n_pairs if n_pairs > 0 else 0.0
    monotone_ratio = ((n_crossings - n_regressions) / n_crossings
                      if n_crossings > 0 else float("nan"))

    return {
        "n_pairs": n_pairs,
        "n_crossings": n_crossings,
        "n_regressions": n_regressions,
        "regression_pairs": regression_pairs,
        "first_crossing_n": first_crossing_n,
        "stability_ratio": stability_ratio,
        "monotone_ratio": monotone_ratio,
    }


def analyze_mean_monotonicity(sigma_mean: np.ndarray,
                              depths: np.ndarray,
                              n_burn_in: int = N_BURN_IN) -> dict:
    """Check monotonicity of the mean trace I(U_t) = sigma_pair_mean(n).

    Returns
    -------
    dict with keys:
        n_steps           : total steps after burn-in
        n_decreases       : number of steps where mean decreased
        max_decrease      : largest decrease (absolute)
        is_monotone       : bool -- no decrease observed
        monotone_fraction : fraction of steps with non-decrease
    """
    mask = depths >= n_burn_in
    trace = sigma_mean[mask]
    diffs = np.diff(trace)
    n_decreases = int(np.sum(diffs < 0))
    max_decrease = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    return {
        "n_steps": len(diffs),
        "n_decreases": n_decreases,
        "max_decrease": max_decrease,
        "is_monotone": n_decreases == 0,
        "monotone_fraction": (len(diffs) - n_decreases) / len(diffs)
                             if len(diffs) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_traces(sigma_traces: np.ndarray, depths: np.ndarray,
                q: int, out_dir: Path,
                n_highlight: int = 5):
    """Plot per-pair sigma_pair(n) traces for a given prime."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n_pairs = sigma_traces.shape[0]
    # Background: all pairs in grey
    for i in range(n_pairs):
        ax.semilogy(depths, sigma_traces[i], color="grey",
                    alpha=0.15, linewidth=0.6)
    # Highlight a few pairs
    idxs = np.linspace(0, n_pairs - 1, min(n_highlight, n_pairs), dtype=int)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idxs)))
    for idx, col in zip(idxs, colors):
        ax.semilogy(depths, sigma_traces[idx], color=col,
                    linewidth=1.4, label=f"pair {idx}")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\sigma_{\mathrm{pair}}(n)$")
    ax.set_title(f"Per-pair traces  $q = {q}$  (all pairs grey, sample highlighted)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"traces_q{q}.pdf"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def plot_crossing_histogram(first_crossing_n: np.ndarray,
                            depths: np.ndarray, q: int,
                            out_dir: Path):
    """Histogram of first-crossing depths."""
    valid = first_crossing_n[~np.isnan(first_crossing_n)]
    if len(valid) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(depths.min(), depths.max() + 2) - 0.5
    ax.hist(valid, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("First crossing depth $n$")
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"Distribution of first threshold crossings  $q = {q}$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"crossings_q{q}.pdf"
    fig.savefig(str(path))
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mode stability diagnostic for O-series npz outputs"
    )
    parser.add_argument("--npz-dir", type=str, default="o25_outputs",
                        help="Directory containing npz files (default: ./o25_outputs)")
    parser.add_argument("--primes", type=int, nargs="+",
                        default=[29, 61, 101],
                        help="Primes to analyze")
    parser.add_argument("--out-dir", type=str, default="stability_out",
                        help="Output directory for plots and summary")
    parser.add_argument("--threshold-rel", type=float, default=THRESHOLD_REL,
                        help="Relative threshold (fraction of per-pair max)")
    parser.add_argument("--threshold-abs", type=float, default=THRESHOLD_ABS,
                        help="Absolute floor threshold")
    parser.add_argument("--burn-in", type=int, default=N_BURN_IN,
                        help="Minimum BFS depth before tracking")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  O-series mode stability diagnostic")
    print(f"  threshold_rel = {args.threshold_rel}  "
          f"threshold_abs = {args.threshold_abs}  "
          f"burn_in = {args.burn_in}")
    print(f"{'='*60}\n")

    summary_rows = []

    for q in args.primes:
        # Try both naming conventions used by o25_paired_pipeline.py
        candidates = [
            npz_dir / f"q{q}_o25.npz",
            npz_dir / f"paired_q{q}.npz",
            npz_dir / f"o25_q{q}.npz",
            npz_dir / f"q{q}_paired.npz",
        ]
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path is None:
            print(f"[q={q}] No npz file found in {npz_dir}  (tried {candidates})")
            continue

        print(f"[q={q}] Loading {npz_path.name} ...")
        data = load_pair_data(npz_path)
        depths = data["depths"]
        sigma_mean = data["sigma_pair_mean"]

        # --- Mean monotonicity test ---
        mono = analyze_mean_monotonicity(sigma_mean, depths, args.burn_in)
        print(f"  Mean trace monotonicity:")
        print(f"    steps after burn-in : {mono['n_steps']}")
        print(f"    decreases           : {mono['n_decreases']}")
        print(f"    max decrease        : {mono['max_decrease']:.3e}")
        print(f"    monotone fraction   : {mono['monotone_fraction']:.4f}")
        print(f"    strictly monotone   : {mono['is_monotone']}")

        row = {
            "q": q,
            "mean_monotone": mono["is_monotone"],
            "mean_mono_frac": mono["monotone_fraction"],
            "mean_n_decreases": mono["n_decreases"],
            "mean_max_decrease": mono["max_decrease"],
        }

        # --- Per-pair stability test ---
        if data["sigma_pair_all"] is not None:
            traces = data["sigma_pair_all"]
            stab = analyze_stability(traces, depths,
                                     args.threshold_rel,
                                     args.threshold_abs,
                                     args.burn_in)
            print(f"  Per-pair stability:")
            print(f"    n_pairs             : {stab['n_pairs']}")
            print(f"    crossed threshold   : {stab['n_crossings']} "
                  f"({100*stab['stability_ratio']:.1f}%)")
            print(f"    regressions         : {stab['n_regressions']}")
            print(f"    monotone ratio      : {stab['monotone_ratio']:.4f}")
            if stab["n_regressions"] > 0:
                print(f"    *** REGRESSIONS DETECTED — "
                      f"pairs: {stab['regression_pairs'][:10]} "
                      f"{'...' if len(stab['regression_pairs']) > 10 else ''}")
            else:
                print(f"    No regressions detected — stability condition holds.")

            row.update({
                "n_pairs": stab["n_pairs"],
                "n_crossings": stab["n_crossings"],
                "n_regressions": stab["n_regressions"],
                "monotone_ratio": stab["monotone_ratio"],
            })

            # Plots
            plot_traces(traces, depths, q, out_dir)
            plot_crossing_histogram(stab["first_crossing_n"], depths, q, out_dir)
        else:
            print(f"  Per-pair traces not available in npz "
                  f"(sigma_pair_all absent) — mean-only analysis.")
            row.update({
                "n_pairs": "n/a",
                "n_crossings": "n/a",
                "n_regressions": "n/a",
                "monotone_ratio": "n/a",
            })

        summary_rows.append(row)
        print()

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    header = (f"{'q':>6}  {'mean_mono':>9}  {'mono_frac':>9}  "
              f"{'n_decr':>6}  {'pairs':>6}  "
              f"{'crossed':>7}  {'regress':>7}  {'mono_r':>7}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['q']:>6}  "
            f"{'YES' if r['mean_monotone'] else 'NO':>9}  "
            f"{r['mean_mono_frac']:>9.4f}  "
            f"{r['mean_n_decreases']:>6}  "
            f"{str(r.get('n_pairs','n/a')):>6}  "
            f"{str(r.get('n_crossings','n/a')):>7}  "
            f"{str(r.get('n_regressions','n/a')):>7}  "
            f"{str(r.get('monotone_ratio','n/a')):>7}"
        )
    print()
    print("Interpretation:")
    print("  mean_mono = YES  : mean I(U_t) is strictly non-decreasing — (A) holds")
    print("  mono_r = 1.0     : no mode regressed after crossing — (C) holds")
    print("  mono_r < 1.0     : regressions detected — stability condition fails")
    print()
    print(f"Plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()