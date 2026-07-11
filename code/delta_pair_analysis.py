"""delta_pair_analysis.py -- knee-trimmed interior fit of delta_pair(q) and convergence test.

Reads the production paired checkpoints o25_outputs/q{q}_o25.npz (grand-mean sigma_pair)
and, if present, the large-q campaign dpc_out/dpc_q{q}.npz (grand_sigma_pair), forms a
single homogeneous delta_pair(q) sequence, and tests convergence to delta_inf.

Observable: sigma_pair(n), grand mean over pairs (and samples).  Fit convention (O16/O25):
    log sigma_pair = -delta_pair * log(n+1) + const,  on a window of shells.

Two windows per q:
  - FULL     [n0, n1]      : n0..n1 = find_fitting_window (last n with sigma>EPS_SAT).
  - INTERIOR [n0, n1-TRIM] : knee-trimmed -- drop the TRIM saturation-rolloff shells whose
                             local log-log slope steepens (the "knee").  This removes the
                             downward bias the saturation knee induces on the OLS slope.
The gap delta_full - delta_interior measures the knee bias; it should shrink with q as the
power-law interior widens.

Convergence: fit delta_interior(q) ~ delta_inf + C / sqrt(q)  (O25 ansatz), report delta_inf
and beta* = 1 / (delta_inf + 1/2).

USAGE
  python delta_pair_analysis.py [--dpc-dir dpc_out] [--trim 2] [--out delta_pair_convergence.pdf]
"""
import os, sys, glob, json, argparse, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from spectral_O12 import find_fitting_window, EPS_SAT

EPS_PAIR = 1e-15


def canonical_window(q, ns, sigma_single=None):
    """Canonical single-block O12 window (n0,n1).

    Priority: the n1_scaling_out/n1_q{q}.json file (find_fitting_window on the single-block
    sigma_bar, the rule used uniformly across q in {29..601}); fall back to recomputing
    find_fitting_window on a provided single-block sigma if the JSON is absent.
    """
    here = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    jp = here / "n1_scaling_out" / f"n1_q{q}.json"
    if jp.exists():
        d = json.loads(jp.read_text())
        return int(d["n0"]), int(d["n1"])
    if sigma_single is not None:
        n0, n1 = find_fitting_window(ns[1:], sigma_single[1:], q)
        return max(int(n0), 1), int(n1)
    return None, None


def fit_slope(ns, sigma, n0, n1):
    """OLS delta = -slope of log(sigma) vs log(n+1) on [n0,n1]; returns (delta,R2,npts)."""
    n0 = max(n0, 1)
    n1 = min(n1, len(sigma) - 1)
    if n1 - n0 < 1:
        return np.nan, np.nan, 0
    nn = ns[n0:n1 + 1].astype(float)
    ss = sigma[n0:n1 + 1].astype(float)
    m = (nn > 0) & (ss > EPS_PAIR)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())
    x = np.log(nn[m] + 1.0)
    y = np.log(ss[m])
    c = np.polyfit(x, y, 1)
    r = y - np.polyval(c, x)
    ssr = np.sum(r ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ssr / sst if sst > 1e-15 else np.nan
    return float(-c[0]), float(r2), int(m.sum())


def load_records(dpc_dir, trim):
    recs = {}
    here = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    # production checkpoints
    for p in sorted(glob.glob(str(here / "o25_outputs" / "q*_o25.npz"))):
        if ".v1." in p:
            continue
        z = np.load(p, allow_pickle=True)
        q = int(z["q"])
        sp = z["sigma_pair_mean"].mean(axis=0)
        ns = z["ns"]
        ssingle = z["sigma_c_mean"].mean(axis=0) if "sigma_c_mean" in z.files else None
        recs[q] = dict(q=q, ns=ns, sigma=sp, source="prod", sigma_single=ssingle,
                       M=int(z["M_per_pair"]), K=int(z["pairs"].shape[0]))
    # large-q campaign (override/add)
    if dpc_dir:
        for p in sorted(glob.glob(str(pathlib.Path(dpc_dir) / "dpc_q*.npz"))):
            z = np.load(p, allow_pickle=True)
            q = int(z["q"])
            ssingle = z["sigma_c_mean"].mean(axis=0) if "sigma_c_mean" in z.files else None
            recs[q] = dict(q=q, ns=z["ns"], sigma=z["grand_sigma_pair"], source="dpc",
                           sigma_single=ssingle, M=int(z["M"]), K=int(z["K"]))
    # windows + fits
    out = []
    for q in sorted(recs):
        r = recs[q]
        ns, sp = r["ns"], r["sigma"]
        n0, n1 = canonical_window(q, ns, r.get("sigma_single"))
        if n0 is None:
            n0, n1 = find_fitting_window(ns[1:], sp[1:], q)
            n0 = max(n0, 1)
        n1 = min(n1, len(sp) - 1)
        d_full, r2_full, np_full = fit_slope(ns, sp, n0, n1)
        n1t = max(n0 + 1, n1 - trim)
        d_int, r2_int, np_int = fit_slope(ns, sp, n0, n1t)
        r.update(n0=n0, n1=n1, n1t=n1t,
                 d_full=d_full, r2_full=r2_full,
                 d_int=d_int, r2_int=r2_int, gap=d_full - d_int)
        out.append(r)
    return out


def fit_conv(qs, ds, form="sqrt"):
    """delta ~ delta_inf + C*g(q), g in {1/sqrt(q), 1/q, log q / q}.
    Returns (delta_inf, C, R2)."""
    qs = np.asarray(qs, float); ds = np.asarray(ds, float)
    m = np.isfinite(ds)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    g = {"sqrt": 1.0 / np.sqrt(qs[m]), "inv": 1.0 / qs[m],
         "invlog": np.log(qs[m]) / qs[m]}[form]
    p = np.polyfit(g, ds[m], 1)
    yh = p[1] + p[0] * g
    r2 = 1.0 - np.sum((ds[m] - yh) ** 2) / np.sum((ds[m] - ds[m].mean()) ** 2)
    return float(p[1]), float(p[0]), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpc-dir", default="dpc_out")
    ap.add_argument("--trim", type=int, default=2, help="knee shells to trim from window tail")
    ap.add_argument("--out", default=None, help="optional PDF path for the convergence plot")
    a = ap.parse_args()

    recs = load_records(a.dpc_dir if os.path.isdir(a.dpc_dir) else None, a.trim)

    print(f"{'q':>5} {'src':>4} {'K':>4} {'M':>3} {'window':>9} {'int':>9} "
          f"{'d_full':>7} {'d_int':>7} {'gap':>6} {'R2int':>6}")
    print("-" * 78)
    for r in recs:
        print(f"{r['q']:>5} {r['source']:>4} {r['K']:>4} {r['M']:>3} "
              f"[{r['n0']:>2},{r['n1']:>2}]   [{r['n0']:>2},{r['n1t']:>2}]   "
              f"{r['d_full']:>7.3f} {r['d_int']:>7.3f} {r['gap']:>6.3f} {r['r2_int']:>6.3f}")

    qs = [r["q"] for r in recs]
    d_full = [r["d_full"] for r in recs]
    d_int = [r["d_int"] for r in recs]

    print("\nConvergence fits  delta ~ delta_inf + C*g(q)  (asymptotic-regime scan):")
    forms = [("1/sqrt(q)", "sqrt"), ("1/q", "inv"), ("log q / q", "invlog")]
    qsa = np.asarray(qs, float)
    best = {}
    for label, series in (("full", d_full), ("interior", d_int)):
        sa = np.asarray(series, float)
        print(f"  {label}:")
        for qmin in [29, 101, 151, 211]:
            msk = qsa >= qmin
            if msk.sum() < 3:
                continue
            rows = []
            for nm, fm in forms:
                di, C, r2 = fit_conv(qsa[msk], sa[msk], fm)
                rows.append((nm, fm, di, r2))
            good = [r for r in rows if np.isfinite(r[3])]
            bn, bfm, bdi, br2 = max(good, key=lambda r: r[3])
            detail = "  ".join(f"{nm}:{di:.2f}(R2={r2:.3f},b*={1/(di+.5):.3f})"
                               for nm, fm, di, r2 in rows)
            print(f"    q>={qmin:>3} (n={int(msk.sum())}): {detail}")
            print(f"              -> best {bn}: delta_inf={bdi:.3f}  beta*={1/(bdi+0.5):.4f}")
            if qmin == 101:
                best[label] = (bn, bdi, br2, bfm)
    print(f"\n  O24 prediction delta_pair ~ 7.44  ->  beta* = {1.0/(7.44+0.5):.4f}")
    print(f"  O10 window [7.4,10.6]            ->  beta* in "
          f"[{1.0/(10.6+0.5):.4f},{1.0/(7.4+0.5):.4f}]")
    # for the figure, use the best interior fit on the asymptotic regime (q>=101)
    if "interior" in best:
        bn_i, inf_i, _, best_form_i = best["interior"]
        msk = np.asarray(qs, float) >= 101
        inf_i, C_i, _ = fit_conv(np.asarray(qs, float)[msk],
                                 np.asarray(d_int, float)[msk], best_form_i)
        best_label_i = bn_i
    else:
        inf_i, C_i, best_form_i, best_label_i = np.nan, np.nan, "sqrt", "?"

    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        qa = np.array(qs, float)
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhspan(7.4, 10.6, color="green", alpha=0.10, label="O10 window [7.4, 10.6]")
        ax.axhline(7.44, color="green", ls="--", lw=1, label="O24 prediction 7.44")
        ax.plot(qa, d_full, "s-", color="steelblue", ms=5, label=r"$\delta_{\rm full}$")
        ax.plot(qa, d_int, "o-", color="tomato", ms=5, label=r"$\delta_{\rm interior}$ (knee-trimmed)")
        qd = np.linspace(qa.min(), max(qa.max(), 601), 200)
        if np.isfinite(inf_i):
            gd = {"sqrt": 1/np.sqrt(qd), "inv": 1/qd, "invlog": np.log(qd)/qd}[best_form_i]
            ax.plot(qd, inf_i + C_i * gd, ":", color="tomato", lw=1,
                    label=fr"best fit q$\geq$101 ({best_label_i}) $\delta_\infty={inf_i:.2f}$")
            ax.axhline(inf_i, color="tomato", lw=0.6, ls=":")
        ax.set_xlabel("prime q"); ax.set_ylabel(r"$\delta_{\rm pair}$")
        ax.set_title(r"Convergence of $\delta_{\rm pair}(q)$ (knee-trimmed interior)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(a.out, dpi=150)
        print(f"\nFigure: {a.out}")


if __name__ == "__main__":
    main()
