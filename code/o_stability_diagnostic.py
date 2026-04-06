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

def cumulative_capacity(sigma_traces: np.ndarray) -> np.ndarray:
    """Compute I(n) = sigma_pair(0) - sigma_pair(n) for each pair.

    I(n) is the cumulative projected capacity up to BFS depth n.
    It is the quantity that should be monotone increasing if the
    identification  time ~ ordering of I(U_t)  holds.

    Parameters
    ----------
    sigma_traces : (n_pairs, n_depths) or (n_depths,) array

    Returns
    -------
    I : same shape as sigma_traces
    """
    if sigma_traces.ndim == 1:
        return sigma_traces[0] - sigma_traces
    return sigma_traces[:, 0:1] - sigma_traces  # broadcast over depths


def analyze_stability(sigma_traces: np.ndarray, depths: np.ndarray,
                      n_burn_in: int = N_BURN_IN) -> dict:
    """Analyze monotonicity and regression of I(n) per pair.

    I(n) = sigma_pair(0) - sigma_pair(n) is the cumulative projected
    capacity.  We test:
      (A) I(n) is non-decreasing at every step (no regression)
      (C) Once I(n) reaches a local maximum it does not decrease

    Parameters
    ----------
    sigma_traces : (n_pairs, n_depths) array
    depths       : (n_depths,) array

    Returns
    -------
    dict with per-pair and aggregate statistics on I(n).
    """
    n_pairs, n_depths = sigma_traces.shape
    I = cumulative_capacity(sigma_traces)   # (n_pairs, n_depths)

    burn_mask = depths >= n_burn_in
    I_burn = I[:, burn_mask]               # (n_pairs, n_burn_depths)
    depths_burn = depths[burn_mask]

    # Per-pair: count steps where I decreases
    diffs = np.diff(I_burn, axis=1)        # (n_pairs, n_burn_depths - 1)
    n_decreases_per_pair = (diffs < 0).sum(axis=1)
    max_decrease_per_pair = np.where(diffs.min(axis=1) < 0,
                                     diffs.min(axis=1), 0.0)

    regression_pairs = list(np.where(n_decreases_per_pair > 0)[0])
    n_regressions = len(regression_pairs)
    n_steps = I_burn.shape[1] - 1

    # Monotone fraction across all pairs and steps
    total_steps = n_pairs * n_steps
    total_decreases = int(n_decreases_per_pair.sum())
    monotone_fraction = (total_steps - total_decreases) / total_steps \
                        if total_steps > 0 else float("nan")

    return {
        "n_pairs": n_pairs,
        "n_steps": n_steps,
        "n_regressions": n_regressions,
        "regression_pairs": regression_pairs,
        "n_decreases_per_pair": n_decreases_per_pair,
        "max_decrease_per_pair": max_decrease_per_pair,
        "total_decreases": total_decreases,
        "monotone_fraction": monotone_fraction,
        "is_monotone": n_regressions == 0,
        "I": I,
        "depths_burn": depths_burn,
        "I_burn": I_burn,
    }


def analyze_mean_monotonicity(sigma_mean: np.ndarray,
                              depths: np.ndarray,
                              n_burn_in: int = N_BURN_IN) -> dict:
    """Check monotonicity of I_mean(n) = sigma_pair_mean(0) - sigma_pair_mean(n).

    This is the mean cumulative projected capacity across all pairs.
    """
    I_mean = cumulative_capacity(sigma_mean)   # (n_depths,)
    mask = depths >= n_burn_in
    trace = I_mean[mask]
    diffs = np.diff(trace)
    n_decreases = int(np.sum(diffs < 0))
    max_decrease = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    return {
        "I_mean": I_mean,
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
    """Plot per-pair I(n) = sigma_pair(0) - sigma_pair(n) traces."""
    I = cumulative_capacity(sigma_traces)   # (n_pairs, n_depths)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: raw sigma_pair(n)
    ax = axes[0]
    for i in range(I.shape[0]):
        ax.semilogy(depths, sigma_traces[i], color="grey",
                    alpha=0.15, linewidth=0.6)
    idxs = np.linspace(0, I.shape[0] - 1, min(n_highlight, I.shape[0]),
                       dtype=int)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idxs)))
    for idx, col in zip(idxs, colors):
        ax.semilogy(depths, sigma_traces[idx], color=col,
                    linewidth=1.4, label=f"pair {idx}")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\sigma_{\mathrm{pair}}(n)$  (residual capacity)")
    ax.set_title(f"Residual capacity  $q = {q}$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    # Right: I(n) = sigma(0) - sigma(n)  -- should be monotone increasing
    ax = axes[1]
    for i in range(I.shape[0]):
        ax.plot(depths, I[i], color="grey", alpha=0.15, linewidth=0.6)
    for idx, col in zip(idxs, colors):
        ax.plot(depths, I[idx], color=col,
                linewidth=1.4, label=f"pair {idx}")
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$I(n) = \sigma_{\mathrm{pair}}(0) - \sigma_{\mathrm{pair}}(n)$")
    ax.set_title(f"Cumulative projected capacity  $q = {q}$\n"
                 r"(should be $\nearrow$ monotone — time proxy)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"traces_q{q}.pdf"
    fig.savefig(str(path))
    plt.close(fig)
    return path


def plot_crossing_histogram(first_crossing_n: np.ndarray,
                            depths: np.ndarray, q: int,
                            out_dir: Path):
    """Histogram of first-crossing depths — kept for compatibility."""
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
    parser.add_argument("--burn-in", type=int, default=N_BURN_IN,
                        help="Minimum BFS depth before tracking (default: 2)")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  O-series mode stability diagnostic")
    print(f"  Testing I(n) = sigma_pair(0) - sigma_pair(n)  [cumulative capacity]")
    print(f"  burn_in = {args.burn_in}")
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

        # --- Mean monotonicity test on I_mean(n) ---
        mono = analyze_mean_monotonicity(sigma_mean, depths, args.burn_in)
        print(f"  I_mean(n) = sigma_pair_mean(0) - sigma_pair_mean(n):")
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

        # --- Per-pair stability test on I(n) ---
        if data["sigma_pair_all"] is not None:
            traces = data["sigma_pair_all"]
            stab = analyze_stability(traces, depths, args.burn_in)
            print(f"  Per-pair I(n) stability:")
            print(f"    n_pairs             : {stab['n_pairs']}")
            print(f"    pairs with regression: {stab['n_regressions']} "
                  f"({100*(1 - stab['monotone_fraction']):.1f}% of steps regress)")
            print(f"    monotone fraction   : {stab['monotone_fraction']:.4f}")
            print(f"    strictly monotone   : {stab['is_monotone']}")
            if stab["n_regressions"] > 0:
                print(f"    *** REGRESSIONS in pairs: "
                      f"{stab['regression_pairs'][:10]}"
                      f"{'...' if len(stab['regression_pairs']) > 10 else ''}")
                worst = stab["n_decreases_per_pair"].argmax()
                print(f"    worst pair {worst}: "
                      f"{stab['n_decreases_per_pair'][worst]} decreases, "
                      f"max drop = {stab['max_decrease_per_pair'][worst]:.3e}")
            else:
                print(f"    No regressions — I(n) strictly non-decreasing "
                      f"for all pairs. Stability condition holds.")

            row.update({
                "n_pairs": stab["n_pairs"],
                "n_regressions": stab["n_regressions"],
                "pair_mono_frac": stab["monotone_fraction"],
                "is_monotone": stab["is_monotone"],
            })

            # Plots — now shows both sigma_pair(n) and I(n)
            plot_traces(traces, depths, q, out_dir)
        else:
            print(f"  Per-pair traces not available — mean-only analysis.")
            row.update({
                "n_pairs": "n/a",
                "n_regressions": "n/a",
                "pair_mono_frac": "n/a",
                "is_monotone": "n/a",
            })

        summary_rows.append(row)
        print()

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    def _fmt(v):
        if v is None or v == "n/a":
            return "n/a"
        if isinstance(v, str):
            return v
        return f"{v:.4f}"

    header = (f"{'q':>6}  {'I_mean_mono':>11}  {'I_mean_frac':>11}  "
              f"{'pairs':>6}  {'regress':>7}  {'pair_frac':>9}  {'mono_all':>8}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['q']:>6}  "
            f"{'YES' if r['mean_monotone'] else 'NO':>11}  "
            f"{r['mean_mono_frac']:>11.4f}  "
            f"{str(r.get('n_pairs','n/a')):>6}  "
            f"{str(r.get('n_regressions','n/a')):>7}  "
            f"{_fmt(r.get('pair_mono_frac')):>9}  "
            f"{str(r.get('is_monotone','n/a')):>8}"
        )
    print()
    print("Interpretation:")
    print("  I_mean_mono = YES : mean I(n) = sigma(0)-sigma(n) is non-decreasing — (A) holds")
    print("  pair_frac = 1.0   : I(n) non-decreasing for all pairs at every step — (C) holds")
    print("  mono_all = True   : no pair ever regressed — stability condition fully holds")
    print()
    print(f"Plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()