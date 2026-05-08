"""
o25_scaling_plot.py
===================
Convergence curve of delta_pair vs q from O25 npz files.

Reads all q*_o25.npz files in the specified directory, extracts
delta_pair_mean, delta_pair_std, delta_pair_global, and M_per_pair,
then plots:
  - delta_mean (mean over pairs and blocks) with error bars
  - delta_global (OLS fit on mean trajectory)
  - fit delta_mean ~ delta_inf + C/sqrt(q) with extrapolated limit
  - theoretical window [7.4, 10.6] from O10
  - prediction delta_pair ~ 7.44 from O24

USAGE
-----
  python o25_scaling_plot.py                    # reads ./o25_outputs/
  python o25_scaling_plot.py --dir /path/to/npz
  python o25_scaling_plot.py --dir . --out scaling.pdf
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_all(npz_dir):
    """Load all q*_o25.npz files, return sorted list of dicts."""
    pattern = str(Path(npz_dir) / "q*_o25.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No q*_o25.npz files found in {npz_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        q  = int(d["q"])
        M  = int(d["M_per_pair"])
        n0 = int(d["n0"])
        n1 = int(d["n1"])

        dg  = float(d["delta_pair_global"])
        r2g = float(d["r2_global"])

        # Per-pair mean exponents (may contain NaN for pairs that didn't converge)
        dm_arr = d["delta_pair_mean"].astype(float)   # (P,)
        ds_arr = d["delta_pair_std"].astype(float)    # (P,)
        valid  = np.isfinite(dm_arr)
        n_valid = int(valid.sum())
        n_total = len(dm_arr)

        dm  = float(np.nanmean(dm_arr)) if n_valid > 0 else np.nan
        # Standard error on the mean across pairs
        se  = float(np.nanstd(dm_arr) / np.sqrt(n_valid)) if n_valid > 1 else np.nan

        # Count non-zero sigma_pair values within the fitting window
        sigma_pm = d["sigma_pair_mean"]    # (P, N_shells)
        ns_arr   = d["ns"]                 # (N_shells,)
        in_win   = (ns_arr >= n0) & (ns_arr <= n1)
        # Mean sigma_pair across pairs, within window
        sig_win  = sigma_pm[:, in_win].mean(axis=0)
        n_nonzero_win = int(np.sum(sig_win > 0))

        records.append(dict(
            q=q, M=M, n0=n0, n1=n1,
            delta_global=dg, r2_global=r2g,
            delta_mean=dm, delta_mean_se=se,
            n_valid=n_valid, n_total=n_total,
            n_nonzero_win=n_nonzero_win,
            path=path,
        ))
        flag = "  [EXCLUDED: <3 non-zero pts in window]" if n_nonzero_win < 3 else ""
        print(f"  q={q:3d}  M={M:3d}  window=[{n0},{n1}]  "
              f"delta_global={dg:.3f}  delta_mean={dm:.3f}+/-{se:.3f}  "
              f"({n_valid}/{n_total} pairs)  {n_nonzero_win} pts in window{flag}")

    return sorted(records, key=lambda r: r["q"])


def fit_scaling(records, key="delta_mean"):
    """Fit delta(q) ~ delta_inf + C/sqrt(q), return (delta_inf, C, q_fit, y_fit)."""
    rows = [(r["q"], r[key]) for r in records
            if np.isfinite(r.get(key, np.nan))]
    if len(rows) < 3:
        return None, None, None, None
    qs = np.array([r[0] for r in rows], dtype=float)
    ys = np.array([r[1] for r in rows], dtype=float)
    x  = 1.0 / np.sqrt(qs)
    p  = np.polyfit(x, ys, 1)          # p[0]=C, p[1]=delta_inf
    q_dense = np.linspace(qs.min(), max(qs.max(), 500), 300)
    y_fit   = p[1] + p[0] / np.sqrt(q_dense)
    return p[1], p[0], q_dense, y_fit


def make_plot(records, out_path):
    qs      = np.array([r["q"]             for r in records])
    dg      = np.array([r["delta_global"]  for r in records])
    dm      = np.array([r["delta_mean"]    for r in records])
    dm_se   = np.array([r["delta_mean_se"] for r in records])

    # Exclude structurally unreliable primes from fits.
    # Criterion: fewer than half the pairs valid, OR window < 5 points,
    # OR n_valid/n_total < 0.9 (too many failed pairs).
    def is_reliable(r):
        frac_valid = r["n_valid"] / r["n_total"] if r["n_total"] > 0 else 0
        n_win = r["n1"] - r["n0"] + 1
        return frac_valid >= 0.9 and n_win >= 5

    records_fit = [r for r in records if is_reliable(r)]
    excluded    = [r["q"] for r in records if not is_reliable(r)]
    if excluded:
        for r in records:
            if not is_reliable(r):
                frac = r["n_valid"]/r["n_total"]
                nwin = r["n1"] - r["n0"] + 1
                print(f"  Excluded q={r['q']:3d}: "
                      f"{r['n_valid']}/{r['n_total']} pairs ({frac:.0%}), "
                      f"window={nwin} pts  [unreliable]")

    # Fit on delta_global (monotone, robust).
    # Fit on delta_mean only if C > 0 (physically meaningful decreasing trend).
    inf_g, C_g, q_fit, y_fit_g = fit_scaling(records_fit, key="delta_global")
    inf_m, C_m, _,     y_fit_m = fit_scaling(records_fit, key="delta_mean")
    show_mean_fit = (inf_m is not None and C_m is not None and C_m > 0)

    O10_lo, O10_hi = 7.4, 10.6
    O24_pred = 7.44

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Theoretical band
    ax.axhspan(O10_lo, O10_hi, color="green", alpha=0.10,
               label=r"O10 window $[7.4,\,10.6]$")
    ax.axhline(O24_pred, color="green", lw=1.2, ls="--",
               label=fr"O24 prediction $\delta_{{\mathrm{{pair}}}} \approx {O24_pred}$")

    # Fit on delta_global
    if y_fit_g is not None:
        ax.plot(q_fit, y_fit_g, color="steelblue", lw=1.0, ls="--", alpha=0.6,
                label=fr"Fit $\delta_{{\mathrm{{global}}}}$: "
                      fr"$\hat{{\delta}}_{{\infty}} = {inf_g:.2f}$, $C={C_g:.1f}$")
        # Note: axhline at inf_g suppressed -- may fall outside O10 window

    # Fit on delta_mean only if monotonically decreasing (C > 0)
    valid_m = np.isfinite(dm)
    if show_mean_fit:
        ax.plot(q_fit, y_fit_m, color="tomato", lw=1.0, ls="--", alpha=0.6,
                label=fr"Fit $\delta_{{\mathrm{{mean}}}}$: "
                      fr"$\hat{{\delta}}_{{\infty}} = {inf_m:.2f}$, $C={C_m:.1f}$")
        ax.axhline(inf_m, color="tomato", lw=0.7, ls=":", alpha=0.5)

    # Split into fit-included and excluded primes
    excl_mask = np.array([not is_reliable(r)
                          for r in records])

    # delta_global: included (solid) and excluded (open)
    if np.any(~excl_mask):
        ax.scatter(qs[~excl_mask], dg[~excl_mask], marker="s", s=45,
                   color="steelblue", zorder=4,
                   label=r"$\delta_{\mathrm{global}}$ (OLS on mean trajectory)")
    if np.any(excl_mask):
        ax.scatter(qs[excl_mask], dg[excl_mask], marker="s", s=45,
                   facecolors="none", edgecolors="steelblue",
                   linewidths=1.5, zorder=4, alpha=0.5,
                   label=r"$\delta_{\mathrm{global}}$ (excluded from fit)")
    ax.plot(qs, dg, color="steelblue", lw=0.9, alpha=0.5)

    # delta_mean with 2SE error bars — exclude unreliable primes
    incl_m = valid_m & ~excl_mask
    excl_m = valid_m & excl_mask
    if np.any(incl_m):
        ax.errorbar(qs[incl_m], dm[incl_m], yerr=2*dm_se[incl_m],
                    fmt="o", color="tomato", capsize=3, capthick=1.2,
                    lw=1.2, ms=5, zorder=5,
                    label=r"$\delta_{\mathrm{mean}}$ (mean over pairs, $\pm 2\,\mathrm{SE}$)")
        ax.plot(qs[incl_m], dm[incl_m], color="tomato", lw=0.9, alpha=0.5)
    if np.any(excl_m):
        ax.errorbar(qs[excl_m], dm[excl_m], yerr=2*dm_se[excl_m],
                    fmt="o", color="tomato", capsize=3, capthick=1.2,
                    lw=1.2, ms=3, zorder=4, alpha=0.35,
                    markerfacecolor="none", markeredgecolor="tomato")

    # Annotate window and M on delta_global points
    for r in records:
        label = f"$[{r['n0']},{r['n1']}]$\n$M={r['M']}$"
        ax.annotate(label, xy=(r["q"], r["delta_global"]),
                    xytext=(5, 7), textcoords="offset points",
                    fontsize=6.5, color="steelblue", alpha=0.85)
    # Flag q=29 delta_mean if anomalous (non-monotone due to small window/few pairs)
    r29 = next((r for r in records if r["q"] == 29), None)
    if r29 is not None and np.isfinite(r29["delta_mean"]):
        n_valid, n_total = r29["n_valid"], r29["n_total"]
        if n_valid < n_total:
            ax.annotate(fr"$({n_valid}/{n_total}$ pairs)",
                        xy=(r29["q"], r29["delta_mean"]),
                        xytext=(6, -14), textcoords="offset points",
                        fontsize=6.5, color="tomato", alpha=0.8)

    ax.set_xlabel(r"Prime $q$", fontsize=12)
    ax.set_ylabel(r"$\delta_{\mathrm{pair}}$", fontsize=12)
    ax.set_title(r"Convergence of $\delta_{\mathrm{pair}}$ with $q$ (O25 data)",
                 fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.set_xlim(0, max(qs) + 30)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    # Print fit summary
    if inf_g is not None:
        print(f"\nFit delta_global ~ {inf_g:.3f} + {C_g:.2f}/sqrt(q)")
        print(f"  Extrapolated limit:        {inf_g:.3f}")
        print(f"  O24 prediction:            {O24_pred}")
        print(f"  Gap:                      {inf_g - O24_pred:+.3f}")
    if inf_m is not None:
        print(f"\nFit delta_mean   ~ {inf_m:.3f} + {C_m:.2f}/sqrt(q)")
        print(f"  Extrapolated limit:        {inf_m:.3f}")
        print(f"  Gap:                      {inf_m - O24_pred:+.3f}")


def main():
    p = argparse.ArgumentParser(
        description="O25 delta_pair convergence plot from npz files")
    p.add_argument("--dir", default="o25_outputs",
                   help="Directory containing q*_o25.npz files")
    p.add_argument("--out", default="o25_scaling.pdf",
                   help="Output figure path")
    args = p.parse_args()

    print(f"Loading from: {args.dir}")
    records = load_all(args.dir)
    print(f"\nLoaded {len(records)} primes: {[r['q'] for r in records]}")
    make_plot(records, args.out)


if __name__ == "__main__":
    main()