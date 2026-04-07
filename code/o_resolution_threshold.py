# o_resolution_threshold.py
#
# Extract n* : the BFS shell depth at which sigma_pair(n) transitions
# from fast decay to slow decay (the "knee" of the curve).
#
# Two modes:
#   Mode A (threshold) : n* = first n where sigma_pair(n) < threshold_rel
#                        * sigma_pair(0).  Simple but sensitive to threshold.
#
#   Mode B (knee)      : n* = shell at the inflection point of log(sigma_pair)
#                        vs n, detected via maximum of |d^2 log(sigma)/dn^2|.
#                        Threshold-independent; finds the structural transition.
#
# Default: Mode B (knee detection).
#
# Usage:
#   python o_resolution_threshold.py --npz-dir ./o25_outputs --primes 29 61 101
#   python o_resolution_threshold.py --mode threshold --threshold-rel 0.01 ...

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

THRESHOLD_REL = 0.01
N_BURN_IN     = 2
PLATEAU_REL   = 0.99
SMOOTH_WIN    = 3      # smoothing window for knee detection (in shells)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def arithmetic_class(c: int, q: int) -> dict:
    """Classify c modulo arithmetic properties of Z/qZ.

    Returns
    -------
    dict with:
        is_qr      : bool   -- c is a quadratic residue mod q
        is_primitive: bool  -- c is a primitive root mod q
        ord_c      : int    -- multiplicative order of c mod q
        c_mod_small: dict   -- c mod 2,3,4,6 (small structure)
        leg        : int    -- Legendre symbol (c|q) : +1, -1, or 0
    """
    # Legendre symbol via Euler's criterion (q prime)
    leg = pow(int(c), (q - 1) // 2, q)
    if leg == q - 1:
        leg = -1   # maps q-1 → -1

    is_qr = (leg == 1)

    # Multiplicative order of c mod q
    order = 1
    val   = c % q
    cur   = val
    while cur != 1:
        cur = (cur * val) % q
        order += 1
        if order > q:          # safety
            order = -1
            break

    is_primitive = (order == q - 1)

    return {
        "leg":          leg,
        "is_qr":        is_qr,
        "is_primitive": is_primitive,
        "ord_c":        order,
        "c_mod4":       c % 4,
        "c_mod6":       c % 6,
    }

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pair_data(npz_path: Path) -> dict:
    """Load npz produced by o25_paired_pipeline.py."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "q":              int(data["q"]),
        "depths":         data["ns"],
        "pairs":          data["pairs"],
        "sigma_pair_all": data["sigma_pair_mean"],
        "n0":             int(data["n0"]),
        "n1":             int(data["n1"]),
    }

# ---------------------------------------------------------------------------
# Knee detection
# ---------------------------------------------------------------------------

def find_knee(trace: np.ndarray, depths: np.ndarray,
              n_burn_in: int = N_BURN_IN,
              smooth_win: int = SMOOTH_WIN) -> dict:
    """Find the knee of log(sigma_pair) vs n.

    The knee is the shell where the second derivative of log(sigma) is
    maximally negative — i.e. the point where the decay rate changes
    most abruptly from fast to slow.

    Returns
    -------
    dict with nstar, nstar_idx, method='knee', and diagnostics.
    """
    mask = depths >= n_burn_in
    d    = depths[mask]
    s    = trace[mask]

    # Avoid log(0)
    s_safe = np.where(s > 0, s, np.nan)
    log_s  = np.log(s_safe)

    # Smooth to reduce noise
    if smooth_win > 1 and len(log_s) > smooth_win:
        kernel = np.ones(smooth_win) / smooth_win
        log_s_sm = np.convolve(log_s, kernel, mode="valid")
        d_sm     = d[smooth_win//2 : smooth_win//2 + len(log_s_sm)]
    else:
        log_s_sm = log_s
        d_sm     = d

    if len(log_s_sm) < 3:
        return {"nstar": None, "nstar_idx": None, "method": "knee",
                "crossed": False, "d2": None}

    # First and second discrete derivatives of log(sigma)
    d1 = np.diff(log_s_sm)
    d2 = np.diff(d1)

    if len(d2) == 0 or np.all(np.isnan(d2)):
        return {"nstar": None, "nstar_idx": None, "method": "knee",
                "crossed": False, "d2": None}

    # Knee = FIRST local maximum of |d2| above a noise floor.
    # We do not take the global argmax (which may find a late secondary coude);
    # instead we scan left-to-right and stop at the first peak.
    d2_abs  = np.abs(d2)
    finite  = np.isfinite(d2_abs)
    if not np.any(finite):
        return {"nstar": None, "nstar_idx": None, "method": "knee",
                "crossed": False, "d2": d2}

    # Noise floor: 10% of the finite max
    noise_floor = 0.10 * float(np.nanmax(np.where(finite, d2_abs, np.nan)))

    # Find first local peak: d2_abs[i] > d2_abs[i-1] and d2_abs[i] >= d2_abs[i+1]
    # and above noise floor
    knee_local = None
    for i in range(1, len(d2_abs) - 1):
        if (finite[i] and d2_abs[i] > noise_floor
                and d2_abs[i] > d2_abs[i - 1]
                and d2_abs[i] >= d2_abs[i + 1]):
            knee_local = i
            break

    # Fallback: global argmax if no local peak found
    if knee_local is None:
        knee_local = int(np.nanargmax(np.where(finite, d2_abs, np.nan)))
    # Map back to original depth array
    knee_depth = d_sm[knee_local + 1]   # +1 for double-diff offset
    # Find closest index in full depths array
    knee_idx   = int(np.argmin(np.abs(depths - knee_depth)))

    sigma0 = trace[0]
    I      = sigma0 - trace
    I_max  = float(I[-1])

    return {
        "nstar":          int(knee_depth),
        "nstar_idx":      knee_idx,
        "sigma_at_nstar": float(trace[knee_idx]),
        "I_at_nstar":     float(I[knee_idx]),
        "I_max":          I_max,
        "plateau_frac":   float(I[knee_idx] / I_max) if I_max > 0 else float("nan"),
        "crossed":        True,
        "method":         "knee",
        "d2":             d2,
        "d_sm":           d_sm,
    }


def find_nstar_threshold(trace: np.ndarray, depths: np.ndarray,
                         threshold_rel: float = THRESHOLD_REL,
                         n_burn_in: int = N_BURN_IN) -> dict:
    """Original threshold-based n* detection."""
    sigma0   = trace[0]
    I        = sigma0 - trace
    I_max    = float(I[-1])
    thr      = threshold_rel * sigma0
    burn_mask = depths >= n_burn_in
    crossed   = np.where(burn_mask & (trace < thr))[0]

    if len(crossed) == 0:
        return {"nstar": None, "nstar_idx": None, "method": "threshold",
                "crossed": False, "I_max": I_max,
                "I_at_nstar": float(I[-1]), "plateau_frac": 1.0,
                "sigma_at_nstar": None}
    idx = crossed[0]
    return {
        "nstar":          int(depths[idx]),
        "nstar_idx":      int(idx),
        "sigma_at_nstar": float(trace[idx]),
        "I_at_nstar":     float(I[idx]),
        "I_max":          I_max,
        "plateau_frac":   float(I[idx] / I_max) if I_max > 0 else float("nan"),
        "crossed":        True,
        "method":         "threshold",
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_nstar(sigma_traces: np.ndarray, depths: np.ndarray,
               nstar_results: list, q: int, out_dir: Path,
               n_highlight: int = 5):
    I_all  = sigma_traces[:, 0:1] - sigma_traces
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    idxs   = np.linspace(0, len(nstar_results) - 1,
                         min(n_highlight, len(nstar_results)), dtype=int)
    colors = plt.cm.tab10(np.linspace(0, 1, len(idxs)))

    # Panel 1: log sigma_pair(n) with n* marked
    ax = axes[0]
    for i in range(sigma_traces.shape[0]):
        ax.semilogy(depths, sigma_traces[i], color="grey",
                    alpha=0.12, linewidth=0.6)
    for idx, col in zip(idxs, colors):
        ax.semilogy(depths, sigma_traces[idx], color=col,
                    linewidth=1.4, label=f"pair {idx}")
        r = nstar_results[idx]
        if r["crossed"] and r["nstar"] is not None:
            ax.axvline(r["nstar"], color=col, linewidth=0.9,
                       linestyle=":", alpha=0.8)
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$\sigma_{\mathrm{pair}}(n)$")
    ax.set_title(f"Residual capacity + $n^*$  ($q={q}$)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: I(n) with n* onset
    ax = axes[1]
    for i in range(I_all.shape[0]):
        ax.plot(depths, I_all[i], color="grey", alpha=0.12, linewidth=0.6)
    for idx, col in zip(idxs, colors):
        ax.plot(depths, I_all[idx], color=col,
                linewidth=1.4, label=f"pair {idx}")
        r = nstar_results[idx]
        if r["crossed"] and r["nstar"] is not None:
            ax.axvline(r["nstar"], color=col, linewidth=0.9,
                       linestyle=":", alpha=0.8)
            ax.scatter([r["nstar"]], [r["I_at_nstar"]],
                       color=col, s=30, zorder=5)
    ax.set_xlabel("BFS depth $n$")
    ax.set_ylabel(r"$I(n)$")
    ax.set_title(f"Cumulative capacity + $n^*$  ($q={q}$)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: distribution of n* coloured by arithmetic class
    ax = axes[2]
    nstars_qr   = [r["nstar"] for r in nstar_results
                   if r["crossed"] and r["nstar"] is not None
                   and r["arith"]["is_qr"]]
    nstars_nqr  = [r["nstar"] for r in nstar_results
                   if r["crossed"] and r["nstar"] is not None
                   and not r["arith"]["is_qr"]]
    nstars_valid = [r["nstar"] for r in nstar_results
                    if r["crossed"] and r["nstar"] is not None]
    if nstars_valid:
        bins = np.arange(min(nstars_valid) - 0.5,
                         max(nstars_valid) + 1.5)
        if nstars_qr:
            ax.hist(nstars_qr,  bins=bins, color="steelblue",
                    alpha=0.7, label="QR (leg=+1)",  edgecolor="white")
        if nstars_nqr:
            ax.hist(nstars_nqr, bins=bins, color="tomato",
                    alpha=0.7, label="NQR (leg=-1)", edgecolor="white")
        ax.set_xlabel("$n^*$ (knee depth)")
        ax.set_ylabel("Number of pairs")
        ax.set_title(f"Distribution of $n^*$  ($q={q}$)\n"
                     f"mean={np.mean(nstars_valid):.2f}  "
                     f"std={np.std(nstars_valid):.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 4: scatter n* vs leg*ord(c)
    ax = axes[3]
    crossed_r = [r for r in nstar_results
                 if r["crossed"] and r["nstar"] is not None]
    if crossed_r:
        xs_qr  = [r["arith"]["leg"] * r["arith"]["ord_c"] for r in crossed_r
                  if r["arith"]["is_qr"]]
        ys_qr  = [r["nstar"] for r in crossed_r if r["arith"]["is_qr"]]
        xs_nqr = [r["arith"]["leg"] * r["arith"]["ord_c"] for r in crossed_r
                  if not r["arith"]["is_qr"]]
        ys_nqr = [r["nstar"] for r in crossed_r if not r["arith"]["is_qr"]]
        if xs_qr:
            ax.scatter(xs_qr,  ys_qr,  color="steelblue", s=40,
                       alpha=0.8, label="QR (leg=+1)", zorder=3)
        if xs_nqr:
            ax.scatter(xs_nqr, ys_nqr, color="tomato",    s=40,
                       alpha=0.8, label="NQR (leg=-1)", zorder=3)
        # Trend line over all points
        all_x = np.array([r["arith"]["leg"] * r["arith"]["ord_c"]
                          for r in crossed_r], dtype=float)
        all_y = np.array([r["nstar"] for r in crossed_r], dtype=float)
        if len(all_x) > 2:
            corr = np.corrcoef(all_x, all_y)[0, 1]
            m, b = np.polyfit(all_x, all_y, 1)
            xfit = np.linspace(all_x.min(), all_x.max(), 100)
            ax.plot(xfit, m * xfit + b, "k--", linewidth=1.0,
                    label=f"fit  r={corr:.3f}")
        ax.set_xlabel(r"$\mathrm{leg}(c|q) \times \mathrm{ord}(c)$")
        ax.set_ylabel("$n^*$")
        ax.set_title(f"$n^*$ vs combined variable  ($q={q}$)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"nstar_q{q}.pdf"
    fig.savefig(str(path))
    plt.close(fig)
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract n* via knee detection or threshold"
    )
    parser.add_argument("--npz-dir",       type=str, default="o25_outputs")
    parser.add_argument("--primes",        type=int, nargs="+",
                        default=[29, 61, 101])
    parser.add_argument("--out-dir",       type=str, default="stability_out")
    parser.add_argument("--mode",          type=str, default="knee",
                        choices=["knee", "threshold"],
                        help="knee (default) or threshold")
    parser.add_argument("--threshold-rel", type=float, default=THRESHOLD_REL)
    parser.add_argument("--smooth-win",    type=int,   default=SMOOTH_WIN,
                        help="Smoothing window for knee detection")
    parser.add_argument("--burn-in",       type=int,   default=N_BURN_IN)
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Resolution threshold diagnostic  — mode: {args.mode}")
    if args.mode == "threshold":
        print(f"  threshold = {args.threshold_rel:.3f} x sigma_pair(0)")
    else:
        print(f"  knee detection  (smooth_win = {args.smooth_win}, first local peak)")
    print(f"  burn_in = {args.burn_in}")
    print(f"{'='*65}\n")

    all_summary = []

    for q in args.primes:
        candidates = [
            npz_dir / f"q{q}_o25.npz",
            npz_dir / f"paired_q{q}.npz",
            npz_dir / f"o25_q{q}.npz",
        ]
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path is None:
            print(f"[q={q}] No npz found.")
            continue

        data   = load_pair_data(npz_path)
        depths = data["depths"]
        traces = data["sigma_pair_all"]
        pairs  = data["pairs"]

        print(f"[q={q}] {traces.shape[0]} pairs, {traces.shape[1]} depths")

        results = []
        for i in range(len(traces)):
            if args.mode == "knee":
                r = find_knee(traces[i], depths,
                              n_burn_in=args.burn_in,
                              smooth_win=args.smooth_win)
            else:
                r = find_nstar_threshold(traces[i], depths,
                                         threshold_rel=args.threshold_rel,
                                         n_burn_in=args.burn_in)
            r["pair_idx"] = i
            r["c"]        = int(pairs[i, 0])
            r["qmc"]      = int(pairs[i, 1])
            r["arith"]    = arithmetic_class(r["c"], q)
            results.append(r)

        # Per-pair table with arithmetic columns
        print(f"  {'pair':>5}  {'(c,q-c)':>12}  {'n*':>5}  "
              f"{'leg':>4}  {'ord(c)':>7}  {'prim':>5}  "
              f"{'c%4':>4}  {'sigma(n*)':>10}  {'plateau':>8}")
        print(f"  {'-'*80}")
        for r in results:
            a   = r["arith"]
            ns  = str(r["nstar"]) if r["crossed"] else "---"
            sig = (f"{r['sigma_at_nstar']:.2e}"
                   if r["crossed"] else "---")
            atp = ("YES" if (r["crossed"] and
                             r["plateau_frac"] >= PLATEAU_REL)
                   else "NO")
            print(f"  {r['pair_idx']:>5}  "
                  f"({r['c']:>3},{r['qmc']:>3})  "
                  f"{ns:>5}  "
                  f"{a['leg']:>4}  "
                  f"{a['ord_c']:>7}  "
                  f"{'YES' if a['is_primitive'] else 'no':>5}  "
                  f"{a['c_mod4']:>4}  "
                  f"{sig:>10}  "
                  f"{atp:>8}")

        # Summary with arithmetic breakdown
        crossed  = [r for r in results if r["crossed"] and r["nstar"] is not None]
        nstars   = np.array([r["nstar"] for r in crossed], dtype=float)
        pfrac    = np.array([r["plateau_frac"] for r in crossed])

        print(f"\n  Summary q={q}:")
        print(f"    pairs with n*     : {len(crossed)} / {len(results)}")
        if len(crossed):
            print(f"    n*  mean ± std    : {nstars.mean():.2f} ± {nstars.std():.2f}")
            print(f"    n*  range         : [{int(nstars.min())}, {int(nstars.max())}]")
            print(f"    plateau_frac mean : {pfrac.mean():.4f}")

            # Arithmetic split: QR vs NQR
            nstar_qr  = [r["nstar"] for r in crossed if r["arith"]["is_qr"]]
            nstar_nqr = [r["nstar"] for r in crossed if not r["arith"]["is_qr"]]
            if nstar_qr:
                print(f"    n* QR  (leg=+1)   : mean={np.mean(nstar_qr):.2f}  "
                      f"n={len(nstar_qr)}")
            if nstar_nqr:
                print(f"    n* NQR (leg=-1)   : mean={np.mean(nstar_nqr):.2f}  "
                      f"n={len(nstar_nqr)}")

            # Arithmetic split: primitive root vs non-primitive
            nstar_prim  = [r["nstar"] for r in crossed if r["arith"]["is_primitive"]]
            nstar_nprim = [r["nstar"] for r in crossed if not r["arith"]["is_primitive"]]
            if nstar_prim:
                print(f"    n* primitive root : mean={np.mean(nstar_prim):.2f}  "
                      f"n={len(nstar_prim)}")
            if nstar_nprim:
                print(f"    n* non-primitive  : mean={np.mean(nstar_nprim):.2f}  "
                      f"n={len(nstar_nprim)}")

            # Order-based split: group by ord(c) ranges
            orders = np.array([r["arith"]["ord_c"] for r in crossed])
            legs   = np.array([r["arith"]["leg"]   for r in crossed])
            print(f"    ord(c) range      : [{orders.min()}, {orders.max()}]  "
                  f"mean={orders.mean():.1f}")
            # Correlations
            if len(orders) > 2:
                corr_ord = np.corrcoef(nstars, orders)[0, 1]
                corr_leg = np.corrcoef(nstars, legs)[0, 1]
                combined = legs * orders
                corr_comb = np.corrcoef(nstars, combined)[0, 1]
                print(f"    corr(n*, ord(c))  : {corr_ord:.3f}")
                print(f"    corr(n*, leg)      : {corr_leg:.3f}")
                print(f"    corr(n*, leg*ord)  : {corr_comb:.3f}  ← combined")
        print()

        all_summary.append({
            "q": q, "n_pairs": len(results), "n_crossed": len(crossed),
            "nstar_mean": float(nstars.mean()) if len(crossed) else float("nan"),
            "nstar_std":  float(nstars.std())  if len(crossed) else float("nan"),
            "pfrac_mean": float(pfrac.mean())  if len(crossed) else float("nan"),
        })

        plot_nstar(traces, depths, results, q, out_dir)

    # Cross-prime summary
    print(f"\n{'='*65}")
    print("  Cross-prime summary")
    print(f"{'='*65}")
    print(f"  {'q':>6}  {'crossed':>8}  {'n* mean':>8}  "
          f"{'n* std':>8}  {'plat_frac':>10}")
    print(f"  {'-'*48}")
    for s in all_summary:
        print(f"  {s['q']:>6}  {s['n_crossed']:>8}  "
              f"{s['nstar_mean']:>8.2f}  {s['nstar_std']:>8.2f}  "
              f"{s['pfrac_mean']:>10.4f}")
    print()
    print("Interpretation (knee mode):")
    print("  n* = shell of maximal change in decay rate of sigma_pair(n).")
    print("  Stable n* across pairs : structural property of q.")
    print("  Growing n* with q      : resolution threshold reached later")
    print("                           for larger groups.")
    print("  QR/NQR split in n*     : arithmetic origin of bimodality.")
    print("  corr(n*, ord(c)) ~ 1   : multiplicative order drives the knee depth.")
    print(f"\nPlots saved to: {out_dir}/")


if __name__ == "__main__":
    main()

#
# Since c_chi and lambda_rho are not yet independently fixed in the
# O-series pipeline, this script operates in two modes:
#
#   Mode A (relative) : n* = first n where sigma_pair(n) < threshold_rel
#                       * max_n(sigma_pair).  Default threshold_rel = 0.01.
#                       Identifies the crossing as a fraction of initial
#                       capacity, independent of absolute normalisation.
#
#   Mode B (absolute) : n* = first n where sigma_pair(n) < threshold_abs.
#                       Useful when the BI bound can be estimated.
#
# For each prime q and each pair (c, q-c):
#   - finds n*
#   - checks that I(n*) is already close to its plateau (saturation check)
#   - reports whether n* coincides with the plateau onset of I(n)
#
# Output:
#   - Table of n* per pair and per prime
#   - Plot: sigma_pair(n) with n* marked + I(n) with plateau onset marked
#   - Summary: is n* a stable marker across pairs within a prime?
#
# Usage:
#   python o_resolution_threshold.py --npz-dir ./o25_outputs --primes 29 61 101

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters (all at top for easy tuning)
# ---------------------------------------------------------------------------

THRESHOLD_REL   = 0.01   # n* = first n where sigma < this * sigma(0)
THRESHOLD_ABS   = None   # if set, overrides THRESHOLD_REL
N_BURN_IN       = 2      # ignore first N_BURN_IN depths