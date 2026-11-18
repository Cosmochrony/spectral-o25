"""o25_scaling_extended_reconstruct.py -- reconstructed generator for o25_scaling_extended.pdf.

Provenance
----------
The original generator of the extended-campaign figure (created 3 July 2026, first committed in 00ad91e)
was not recovered. This script is an independent reconstruction written on 18 November 2026 from the
preserved checkpoints. It is not the original code. It reproduces every published value of the figure and
of the extended-campaign table to their printed precision; the historical label "8.09" at q = 401 was a
double rounding of the tabulated 8.085, the unrounded value being 8.0848.

Inputs (all in data/extended/, checksums in data/extended/SHA256SUMS, provenance in PROVENANCE.md):
  q{29,61,101,151,211}_o25.npz   full-graph production checkpoints, M = 50 samples per pair
  dpc_q{307,401,601}.npz          capped-BFS campaign checkpoints (M = 16, 8, 8)
  n1_q{307,401,601}.json          canonical single-block fitting windows (n0, n1)

Quantities:
  delta_global(q): stored OLS exponent for q <= 211; for q >= 307, OLS of log grand_sigma_pair
                   against log(n + 1) on the window [n0, n1] of n1_q{q}.json.
  delta_corr(q):   per-pair mean exponent minus (1/2) log q / log n1  (O14 correction, eta = 1/2).
  delta_inf:       intercepts of delta_global ~ delta_inf + C g(q), g = 1/sqrt(q) and 1/q, on q >= 101.

Labelling: the interval [7.4, 10.6] and the value 7.44 are drawn as historical comparisons only. They are
not conditions derived from O6, which proves only beta_eff <= 1/(alpha + 1/2). No data are changed.
Value labels use one rule: Python format "{:.2f}" applied to the unrounded value.

Usage (from code/):
  python o25_scaling_extended_reconstruct.py            # write the figure and check all values
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "extended"
FIG = HERE / "o25_scaling_extended.pdf"
VALUES = HERE / "o25_scaling_extended_values.json"

Q_PROD = (29, 61, 101, 151, 211)
Q_CAPPED = (307, 401, 601)

# Published values of SpectralO25 v1.5, with explicit tolerances.
# A value printed with k decimals gets tolerance 0.5e-k.  The O14-correction table prints delta_corr as the
# difference of two already rounded columns (mean and eta log q / log n1), so its delta_corr tolerance is the
# sum of their two rounding tolerances, 1e-3; the two columns are also checked on their own at 5e-4.
# q = 29 has no printed delta_corr (it is only plotted); it is checked on the figure, not here.
EXPECTED = {
    # figure labels (q <= 151) and Table "Extended campaign" (q >= 211)
    "delta_global": {29: (12.51, 5e-3), 61: (11.49, 5e-3), 101: (10.70, 5e-3), 151: (9.53, 5e-3),
                     211: (8.760, 5e-4), 307: (8.254, 5e-4), 401: (8.085, 5e-4), 601: (7.610, 5e-4)},
    # Table "O14 normalization correction" (q = 61..211), columns mean and eta log q / log n1
    "delta_mean": {61: (9.983, 5e-4), 101: (9.548, 5e-4), 151: (9.125, 5e-4), 211: (8.784, 5e-4),
                   307: (8.261, 5e-4), 401: (8.090, 5e-4), 601: (7.618, 5e-4)},
    "correction": {61: (0.988, 5e-4), 101: (0.962, 5e-4), 151: (0.978, 5e-4), 211: (1.014, 5e-4)},
    # O14-correction table (q = 61..211, 3 decimals) and Table "Extended campaign" (q >= 307, 2 decimals)
    "delta_corr": {61: (8.995, 1e-3), 101: (8.586, 1e-3), 151: (8.147, 1e-3), 211: (7.771, 1e-3),
                   307: (7.23, 5e-3), 401: (7.03, 5e-3), 601: (6.53, 5e-3)},
    "window": {307: (3, 16), 401: (4, 17), 601: (4, 19)},
    # figure legend, one decimal
    "delta_inf": {"sqrt": (5.4, 5e-2), "inv": (7.1, 5e-2)},
}


def verify_inputs():
    lines = (DATA / "SHA256SUMS").read_text().split("\n")
    for line in filter(None, lines):
        digest, name = line.split()
        path = DATA / name
        if not path.exists():
            sys.exit(f"missing input {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            sys.exit(f"checksum mismatch for {path}")


def ols_exponent(ns, sigma, n0, n1):
    x = np.log(ns[n0:n1 + 1].astype(float) + 1.0)
    y = np.log(sigma[n0:n1 + 1].astype(float))
    return float(-np.polyfit(x, y, 1)[0])


def compute():
    rows = []
    for q in Q_PROD:
        d = np.load(DATA / f"q{q}_o25.npz", allow_pickle=False)
        rows.append(dict(q=q, delta_global=float(d["delta_pair_global"]),
                         delta_mean=float(np.nanmean(d["delta_pair_mean"])),
                         n0=int(d["n0"]), n1=int(d["n1"]), source="full graph, M=50"))
    for q in Q_CAPPED:
        z = np.load(DATA / f"dpc_q{q}.npz", allow_pickle=False)
        w = json.loads((DATA / f"n1_q{q}.json").read_text())
        n0, n1 = int(w["n0"]), int(w["n1"])
        per_pair = [ols_exponent(z["ns"], s, n0, n1) for s in z["sigma_pair_mean"]]
        rows.append(dict(q=q, delta_global=ols_exponent(z["ns"], z["grand_sigma_pair"], n0, n1),
                         delta_mean=float(np.mean(per_pair)), n0=n0, n1=n1,
                         source=f"capped BFS, M={int(z['M'])}"))
    for r in rows:
        r["correction"] = 0.5 * np.log(r["q"]) / np.log(r["n1"])
        r["delta_corr"] = r["delta_mean"] - r["correction"]
    q = np.array([r["q"] for r in rows], float)
    dg = np.array([r["delta_global"] for r in rows])
    mask = q >= 101
    fits = {}
    for name, g in (("sqrt", lambda x: 1.0 / np.sqrt(x)), ("inv", lambda x: 1.0 / x)):
        slope, intercept = np.polyfit(g(q[mask]), dg[mask], 1)
        fits[name] = dict(delta_inf=float(intercept), C=float(slope))
    return rows, fits


def check(rows, fits):
    failures = []
    by_q = {r["q"]: r for r in rows}
    for key in ("delta_global", "delta_mean", "correction", "delta_corr"):
        for q, (ref, tol) in EXPECTED[key].items():
            if abs(by_q[q][key] - ref) > tol:
                failures.append(f"{key}(q={q}) = {by_q[q][key]:.6f}, expected {ref} +/- {tol}")
    for q, win in EXPECTED["window"].items():
        if (by_q[q]["n0"], by_q[q]["n1"]) != win:
            failures.append(f"window(q={q}) = {(by_q[q]['n0'], by_q[q]['n1'])}, expected {win}")
    for name, (ref, tol) in EXPECTED["delta_inf"].items():
        if abs(fits[name]["delta_inf"] - ref) > tol:
            failures.append(f"delta_inf[{name}] = {fits[name]['delta_inf']:.4f}, expected {ref} +/- {tol}")
    n_checks = sum(len(EXPECTED[k]) for k in EXPECTED)
    return failures, n_checks


def plot(rows, fits):
    q = np.array([r["q"] for r in rows], float)
    dg = np.array([r["delta_global"] for r in rows])
    dc = np.array([r["delta_corr"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axhspan(7.4, 10.6, color="green", alpha=0.10,
               label=r"historical comparison range $[7.4,\,10.6]$ (not derived)")
    ax.axhline(7.44, color="green", ls="--", lw=1.5,
               label=r"O24 value $\delta_{\mathrm{pair}}\approx 7.44$ (historical comparison)")
    qq = np.linspace(101, 650, 300)
    for name, col, lab in (("sqrt", "tab:blue", r"$1/\sqrt{q}$ fit ($\delta_\infty$ = %.1f)"),
                           ("inv", "indigo", r"$1/q$ fit ($\delta_\infty$ = %.1f)")):
        f = fits[name]
        g = 1.0 / np.sqrt(qq) if name == "sqrt" else 1.0 / qq
        ax.plot(qq, f["delta_inf"] + f["C"] * g, ls=":", color=col, lw=1.3, label=lab % f["delta_inf"])
    ax.plot(q, dg, "o-", color="tomato", lw=1.8, ms=7, label=r"$\delta_{\mathrm{global}}$ (raw)")
    ax.plot(q, dc, "s--", color="grey", lw=1.1, ms=6, label=r"$\delta_{\mathrm{corr}}$ (O14, $\eta=1/2$)")
    for x, y in zip(q, dg):
        ax.annotate("{:.2f}".format(y), (x, y), textcoords="offset points", xytext=(4, 8),
                    fontsize=7, color="tomato")
    ax.axvline(211, color="k", ls=":", lw=0.7)
    ax.text(105, 12.4, "original (M=50)", fontsize=7.5, color="0.35")
    ax.text(218, 12.4, "extension (capped BFS)", fontsize=7.5, color="0.35")
    ax.set_xlim(0, 660)
    ax.set_ylim(6, 13)
    ax.set_xlabel(r"prime $q$")
    ax.set_ylabel(r"$\delta_{\mathrm{pair}}$")
    ax.set_title(r"Raw exponent $\delta_{\mathrm{global}}(q)$ and O14-corrected values (extended campaign)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def main():
    verify_inputs()
    rows, fits = compute()
    plot(rows, fits)
    VALUES.write_text(json.dumps(dict(rows=rows, fits=fits, numpy=np.__version__,
                                      matplotlib=matplotlib.__version__), indent=1) + "\n")
    for r in rows:
        print(f"q={r['q']:>3}  window=[{r['n0']},{r['n1']}]  delta_global={r['delta_global']:.4f}  "
              f"delta_corr={r['delta_corr']:.4f}  ({r['source']})")
    for name, f in fits.items():
        print(f"fit {name}: delta_inf={f['delta_inf']:.4f}")
    failures, n_checks = check(rows, fits)
    if failures:
        print("\n".join(failures))
        sys.exit(f"{len(failures)} of {n_checks} value checks failed")
    print(f"all {n_checks} value checks passed")


if __name__ == "__main__":
    main()
