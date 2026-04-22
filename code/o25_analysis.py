"""
o25_analysis.py
===============
Produces the three figures for SpectralO25.

Figure 1 — sigma_pair(n) log-log profiles
    For each prime, mean sigma_pair over all pairs vs BFS depth n,
    with OLS fit overlaid.  Illustrates the power-law decay and the
    fitting window [n0, n1].

Figure 2 — Inter-pair delta_pair distribution
    For each prime, scatter plot of delta_pair per pair (sorted),
    with mean and +/- 1 sigma bands.  Shows concentration with q.

Figure 3 — Convergence and normalization correction
    Left:  raw dpairmean(q) vs q with empirical fits (1/log q, 1/(log q)^2)
    Right: dpairmean(q) and delta_corr(q) = dpairmean - eta*log(q)/log(n1)
           both plotted vs q, with admissible window [7.4, 10.6] shaded.

USAGE
-----
python o25_analysis.py                          # reads from o25_outputs/
python o25_analysis.py --data-dir /path/to/npz
python o25_analysis.py --out-dir /path/to/figs

REQUIRES
--------
numpy, matplotlib, scipy
"""

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# PARAMETERS
# ============================================================

DEFAULT_DATA_DIR = pathlib.Path("o25_outputs")
DEFAULT_OUT_DIR  = pathlib.Path("o25_figures")

PRIMES   = [29, 61, 101, 151, 211]
COLORS   = {29: "#4477aa", 61: "#ee6677", 101: "#228833", 151: "#ccbb44", 211: "#aa3377"}
MARKERS  = {29: "o", 61: "s", 101: "^", 151: "D", 211: "P"}

TARGET_LO, TARGET_HI = 7.4, 10.6
ETA = 0.5   # O14 normalization exponent


# ============================================================
# DATA LOADING
# ============================================================

def load_results(data_dir):
    """Load all available o25 npz files. Returns dict q -> data."""
    data = {}
    for q in PRIMES:
        path = data_dir / f"q{q}_o25.npz"
        if path.exists():
            z = np.load(path)
            data[q] = z
        else:
            print(f"  WARNING: {path} not found, skipping q={q}")
    return data


# ============================================================
# FIGURE 1: sigma_pair log-log profiles
# ============================================================

def figure1_sigma_pair_profiles(data, out_dir):
    """
    Log-log decay of mean sigma_pair(n) for each prime.
    One panel per prime, 2x2 layout.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    # Hide the unused 6th panel
    axes[5].set_visible(False)

    for ax, q in zip(axes, PRIMES):
        if q not in data:
            ax.set_visible(False)
            continue
        z   = data[q]
        ns  = z["ns"].astype(float)
        n0, n1 = int(z["n0"]), int(z["n1"])
        spm = z["sigma_pair_mean"]          # (n_pairs, n_shells)
        grand_mean = np.mean(spm, axis=0)   # (n_shells,)

        col = COLORS[q]

        # Plot mean sigma_pair
        mask_pos = grand_mean > 1e-15
        ax.loglog(ns[mask_pos] + 1, grand_mean[mask_pos],
                  "o", color=col, ms=5, alpha=0.8, label=r"$\bar\sigma_{\rm pair}(n)$")

        # OLS fit in window [n0, n1]
        win = (ns >= n0) & (ns <= n1) & (grand_mean > 1e-15)
        if win.sum() >= 2:
            log_n = np.log(ns[win] + 1)
            log_s = np.log(grand_mean[win])
            coef  = np.polyfit(log_n, log_s, 1)
            delta_fit = -coef[0]
            n_fit = np.linspace(ns[win][0], ns[win][-1], 60)
            s_fit = np.exp(np.polyval(coef, np.log(n_fit + 1)))
            ax.loglog(n_fit + 1, s_fit, "--", color=col, lw=1.5,
                      label=rf"$n^{{-{delta_fit:.2f}}}$")

        # Shade fitting window
        ax.axvspan(ns[n0] + 1, ns[n1] + 1, alpha=0.10, color=col)

        ax.set_xlabel(r"$n + 1$", fontsize=9)
        ax.set_ylabel(r"$\bar\sigma_{\rm pair}(n)$", fontsize=9)
        ax.set_title(rf"$q = {q}$,  window $[{n0},{n1}]$", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(r"Figure 1: Mean $\sigma_{\rm pair}(n)$ --- O25",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"o25_fig1_sigma_pair.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: o25_fig1_sigma_pair.pdf / .png")


# ============================================================
# FIGURE 2: inter-pair delta_pair distribution
# ============================================================

def figure2_distribution(data, out_dir):
    """
    Scatter of delta_pair per pair (sorted by value) for each prime.
    Shows concentration: variance decreases with q.
    """
    available = [q for q in PRIMES if q in data]
    if not available:
        print("  figure2: no data available, skipped.")
        return
    fig, axes = plt.subplots(1, len(available),
                             figsize=(3.2 * len(available), 4.5),
                             sharey=True)
    if len(available) == 1:
        axes = [axes]

    ax_iter = iter(axes)
    for q in PRIMES:
        if q not in data:
            continue
        ax  = next(ax_iter)
        z   = data[q]
        dp  = z["delta_pair_mean"]       # (n_pairs,)
        dp_valid = dp[np.isfinite(dp)]
        dp_sorted = np.sort(dp_valid)
        n_pairs = len(dp_sorted)

        col = COLORS[q]
        mk  = MARKERS[q]

        ax.scatter(range(n_pairs), dp_sorted, color=col, s=30,
                   marker=mk, alpha=0.7, zorder=3)

        mean_dp = float(np.mean(dp_sorted))
        std_dp  = float(np.std(dp_sorted))
        ax.axhline(mean_dp, color=col, lw=1.5, ls="-",
                   label=rf"mean $= {mean_dp:.3f}$")
        ax.axhspan(mean_dp - std_dp, mean_dp + std_dp,
                   alpha=0.15, color=col,
                   label=rf"$\pm\sigma = {std_dp:.3f}$")
        ax.axhspan(TARGET_LO, TARGET_HI, alpha=0.08, color="green")
        ax.axhline(TARGET_LO, color="green", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(TARGET_HI, color="green", ls="--", lw=1.0, alpha=0.7)

        ax.set_xlabel("pair rank (sorted)", fontsize=9)
        ax.set_title(rf"$q = {q}$  ($n_\mathrm{{pairs}} = {n_pairs}$)",
                     fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\delta_{\rm pair}$ per conjugate pair", fontsize=9)
    axes[0].text(0.01, (TARGET_LO + TARGET_HI) / 2,
                 r"target $[7.4,\,10.6]$",
                 transform=axes[0].get_yaxis_transform(),
                 va="center", ha="left", fontsize=7.5,
                 color="green", alpha=0.8)

    fig.suptitle(r"Figure 2: Full distribution of $\delta_{\rm pair}$ across pairs --- O25",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"o25_fig2_distribution.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: o25_fig2_distribution.pdf / .png")


# ============================================================
# FIGURE 3: convergence and normalization correction
# ============================================================

def figure3_convergence(data, out_dir):
    """
    Left:  raw dpairmean(q) with empirical fits 1/log(q) and 1/(log(q))^2.
    Right: raw vs O14-corrected dpairmean(q), admissible window shaded.
    """
    qs_avail   = sorted([q for q in PRIMES if q in data])
    if not qs_avail:
        print("  figure3: no data available, skipped.")
        return
    qs_fit     = [q for q in qs_avail if q >= 61]  # q=29 excluded from fits

    qs_arr     = np.array(qs_avail, dtype=float)
    qs_fit_arr = np.array(qs_fit,   dtype=float)

    dp_arr   = np.array([float(np.nanmean(data[q]["delta_pair_mean"]))
                         for q in qs_avail])
    dp_fit   = np.array([float(np.nanmean(data[q]["delta_pair_mean"]))
                         for q in qs_fit])
    dp_std   = np.array([float(np.nanstd(data[q]["delta_pair_mean"]))
                         for q in qs_avail])

    n1_arr   = np.array([int(data[q]["n1"]) for q in qs_avail], dtype=float)
    n1_fit   = np.array([int(data[q]["n1"]) for q in qs_fit],   dtype=float)

    # O14 correction
    corr_factor = np.log(qs_arr) / np.log(n1_arr)
    dp_corr     = dp_arr - ETA * corr_factor
    corr_factor_fit = np.log(qs_fit_arr) / np.log(n1_fit)
    dp_corr_fit     = dp_fit - ETA * corr_factor_fit

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left panel: raw data + empirical fits ────────────────────────────
    for q, dp, std in zip(qs_avail, dp_arr, dp_std):
        col = COLORS[q]
        mk  = MARKERS[q]
        ax1.errorbar(q, dp, yerr=std, fmt=mk, color=col, ms=7,
                     capsize=4, lw=1.5, zorder=3,
                     label=rf"$q={q}$: ${dp:.3f}\pm{std:.3f}$")

    # Fit 1/log(q)
    try:
        f1 = lambda q, d_inf, a: d_inf + a / np.log(q)
        p1, _ = curve_fit(f1, qs_fit_arr, dp_fit, p0=[5., 20.])
        q_plot = np.linspace(50, 220, 200)
        ax1.plot(q_plot, f1(q_plot, *p1), "--", color="gray", lw=1.2,
                 label=rf"$\delta_\infty + a/\!\log q$  ($\delta_\infty={p1[0]:.2f}$)")
    except Exception:
        pass

    # Fit 1/(log q)^2
    try:
        f2 = lambda q, d_inf, a: d_inf + a / np.log(q)**2
        p2, _ = curve_fit(f2, qs_fit_arr, dp_fit, p0=[7., 40.])
        ax1.plot(q_plot, f2(q_plot, *p2), ":", color="gray", lw=1.2,
                 label=rf"$\delta_\infty + a/(\!\log q)^2$  ($\delta_\infty={p2[0]:.2f}$)")
    except Exception:
        pass

    ax1.axhspan(TARGET_LO, TARGET_HI, alpha=0.10, color="green",
                label=r"target $[7.4,\,10.6]$")
    ax1.axhline(TARGET_LO, color="green", ls="--", lw=1.0, alpha=0.7)
    ax1.axhline(TARGET_HI, color="green", ls="--", lw=1.0, alpha=0.7)
    ax1.set_xlabel(r"prime $q$", fontsize=10)
    ax1.set_ylabel(r"$\bar\delta_{\rm pair}(q)$", fontsize=10)
    ax1.set_title("(A) Raw $\\bar\\delta_{\\rm pair}$ and empirical fits", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(20, 225)

    # ── Right panel: raw vs corrected ────────────────────────────────────
    for q, dp, dp_c, std in zip(qs_avail, dp_arr, dp_corr, dp_std):
        col = COLORS[q]
        mk  = MARKERS[q]
        ax2.errorbar(q, dp, yerr=std, fmt=mk, color=col, ms=7,
                     capsize=4, lw=1.5, alpha=0.5)
        ax2.plot(q, dp_c, mk, color=col, ms=9, markeredgewidth=1.5,
                 markeredgecolor="k", zorder=4)

    # Legend proxies
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               ms=7, alpha=0.5,
               label=r"raw $\bar\delta_{\rm pair}$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               ms=9, markeredgewidth=1.5, markeredgecolor="k",
               label=r"$\delta_{\rm corr} = \bar\delta_{\rm pair}"
                      r" - \eta\,\frac{\log q}{\log n_1}$"),
    ]
    ax2.axhspan(TARGET_LO, TARGET_HI, alpha=0.10, color="green",
                label=r"target $[7.4,\,10.6]$")
    ax2.axhline(TARGET_LO, color="green", ls="--", lw=1.0, alpha=0.7)
    ax2.axhline(TARGET_HI, color="green", ls="--", lw=1.0, alpha=0.7)
    ax2.set_xlabel(r"prime $q$", fontsize=10)
    ax2.set_ylabel(r"$\delta$", fontsize=10)
    ax2.set_title(r"(B) Raw vs O14-corrected ($\eta = 1/2$)", fontsize=10)
    ax2.legend(handles=proxies + [
        matplotlib.patches.Patch(alpha=0.10, color="green",
                                 label=r"target $[7.4,\,10.6]$")],
               fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(20, 225)

    fig.suptitle(r"Figure 3: Convergence of $\delta_{\rm pair}$ and normalization correction --- O25",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"o25_fig3_convergence.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved: o25_fig3_convergence.pdf / .png")


# ============================================================
# PRINT SUMMARY TABLE (for paper verification)
# ============================================================

def print_summary(data):
    print()
    print("=== O25 Summary ===")
    print(f"{'q':>5}  {'pairs':>6}  {'M':>4}  {'mean':>8}  {'std':>7}  "
          f"{'cv%':>5}  {'n0':>3}  {'n1':>3}  {'n1/q':>7}  "
          f"{'eta*corr':>9}  {'d_corr':>8}")
    print("-" * 80)
    for q in PRIMES:
        if q not in data:
            continue
        z    = data[q]
        dp   = z["delta_pair_mean"]
        n0, n1 = int(z["n0"]), int(z["n1"])
        npairs  = int(z["pairs"].shape[0])
        M       = int(z["M_per_pair"])
        mean_dp = float(np.nanmean(dp))
        std_dp  = float(np.nanstd(dp))
        cv      = 100 * std_dp / mean_dp if mean_dp > 0 else float("nan")
        corr    = ETA * np.log(q) / np.log(n1)
        d_corr  = mean_dp - corr
        print(f"{q:>5}  {npairs:>6}  {M:>4}  {mean_dp:>8.4f}  {std_dp:>7.4f}  "
              f"{cv:>5.1f}  {n0:>3}  {n1:>3}  {n1/q:>7.4f}  "
              f"{corr:>9.4f}  {d_corr:>8.4f}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    import matplotlib.patches  # needed for legend proxy in fig3

    parser = argparse.ArgumentParser(
        description="O25 analysis: produces all paper figures"
    )
    parser.add_argument("--data-dir", type=pathlib.Path,
                        default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir",  type=pathlib.Path,
                        default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    data = load_results(args.data_dir)
    print(f"Loaded: {sorted(data.keys())}")

    print_summary(data)

    print("Generating figures...")
    figure1_sigma_pair_profiles(data, args.out_dir)
    figure2_distribution(data, args.out_dir)
    figure3_convergence(data, args.out_dir)

    print()
    print(f"All figures saved to: {args.out_dir}")


if __name__ == "__main__":
    import matplotlib.patches
    main()