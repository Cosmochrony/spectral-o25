"""
o25_paired_pipeline.py
======================
O25 numerical campaign: systematic delta_pair computation across all
conjugate pairs (c, q-c) and all target primes q.

GROUNDING IN THE O-SERIES
--------------------------
Every design decision traces to a specific paper:

Observable (O12, spectral_O12.py line 288):
    sigma_c(n) = delta_r_n / |S_n|
where delta_r_n is the number of Weil fingerprint vectors in shell S_n
that are NEW (linearly independent of the Gram-Schmidt span built over
shells 0,...,n-1).  This is the exact observable computed by
compute_block_capacity() in spectral_O12.py.

Block parameterisation (O12):
    c_block = (c1, c2, c3) in (Z/qZ)^3, all nonzero, c1+c2+c3 != 0 (mod q).
    c1 is the central character; c2, c3 are sampled independently.
    Two blocks with the same c1 = c yield different sigma_c values
    (different amplitude factor r(c,q) from O17/O19), but the same
    asymptotic exponent delta_c (O17 Corollary 2.3).

Pair observable (O16):
    sigma_pair(n) = sigma_c(n) * sigma_{q-c}(n)
Two blocks are drawn independently for c and q-c, with c1=c and c1=q-c
respectively, reproducing the O16 protocol.

Regression convention (O16_pair_observable.py, function fit_delta):
    log(sigma_pair) = -delta_pair * log(n+1) + const
The +1 shift is the convention used throughout O16, which produces the
published value delta_pair ~ 7.44 for the target pairs at q in {29, 61}.

Fitting window (O12, stored in q<q>_o12.npz):
    [n0, n1] as determined by find_fitting_window() in spectral_O12.py.
    Values: {29:(2,5), 61:(2,7), 101:(3,10), 151:(3,12), 211:(3,13)}.

WHAT O25 ADDS
-------------
O16 computed sigma_pair for one specific pair per prime (where two
blocks with conjugate c1 happened to appear in the O12 sample).
O25 computes sigma_pair for ALL (q-1)/2 conjugate pairs, with M
independent block samples per pair, giving:
  - the full distribution of delta_pair across pairs and blocks
  - the mean and inter-pair variance
  - the scaling law delta_pair(q) as a function of q

USAGE
-----
python o25_paired_pipeline.py                   # all default primes
python o25_paired_pipeline.py --primes 101 151 211
python o25_paired_pipeline.py --primes 307 --M 20 --n-max 70
python o25_paired_pipeline.py --primes 61 --M 50 --force

REQUIRES
--------
spectral_O12.py in the same directory (or on PYTHONPATH).
"""

import argparse
import pathlib
import sys
import time
import os
import traceback

# Fix BLAS/OpenMP threading to 1 before any numpy import.
# Each joblib worker process is already a separate core; internal
# multithreading from numpy/BLAS would cause over-subscription
# (n_jobs workers × N BLAS threads >> available cores).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from spectral_O12 import (
        build_generators,
        bfs_shells,
        compute_block_capacity,
        find_fitting_window,
    )
except Exception as e:
    print(f"ERROR importing spectral_O12: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# PARAMETERS (all at top for reproducibility)
# ============================================================

DEFAULT_PRIMES = [29, 61, 101, 151, 211]

# M_PER_PAIR: number of independent block samples per conjugate pair (c, q-c).
# Each sample draws independent (c2,c3) for c and independent (c2',c3') for q-c,
# reproducing the O16 protocol.  M=20 gives a stable mean; M=50 for publication.
M_PER_PAIR_DEFAULT = 20

# Fitting windows from O12 npz files (find_fitting_window in spectral_O12.py).
# These reproduce the exact windows used in O12/O13/O16.
WINDOW_O12 = {
    29:  (2,  5),
    61:  (2,  7),
    101: (3, 10),
    151: (3, 12),
    211: (3, 13),
}

# BFS fraction: stop BFS at max_fraction * q^3 nodes.
# For small q: 0.99 (full graph).  For larger q: reduce to limit memory/time.
BFS_FRAC = {
    29:  0.99,
    61:  0.99,
    101: 0.99,
    151: 0.29,
    211: 0.11,
    307: 0.08,
    401: 0.05,
}
BFS_FRAC_FALLBACK = 0.05

# n_max cap on BFS depth per block (from O12 paper params).
N_MAX_BLOCK = {
    29:   8,
    61:  10,
    101: 20,
    151: 37,
    211: 50,
    307: 70,
    401: 80,
}
N_MAX_FALLBACK = 60

DEFAULT_SEED  = 42
OUTPUT_DIR    = pathlib.Path("o25_outputs")

# Number of parallel workers for the pair loop.
# -1 = use all available CPU cores (recommended on multi-core machines).
# 1  = sequential (original behaviour, useful for debugging).
N_JOBS_DEFAULT = -1

# Minimum sigma_pair value to include in the log-log fit (avoid numerical noise)
EPS_PAIR = 1e-15


# ============================================================
# BLOCK SAMPLING WITH FIXED c1
# ============================================================

def sample_block_with_c1(c1, q, rng, max_attempts=2000):
    """
    Sample a generic block (c1, c2, c3) with c1 fixed and c2, c3 random.
    Generic condition: all ci != 0 and c1+c2+c3 != 0 (mod q).
    """
    for _ in range(max_attempts):
        c2 = int(rng.integers(1, q))
        c3 = int(rng.integers(1, q))
        if (c1 + c2 + c3) % q != 0:
            return np.array([c1, c2, c3], dtype=np.int64)
    raise RuntimeError(f"Cannot sample generic block with c1={c1}, q={q}")


# ============================================================
# FITTING: O16 CONVENTION log(n+1)
# ============================================================

def fit_delta_pair(sigma_pair, ns, n0, n1):
    """
    OLS fit: log(sigma_pair) = -delta_pair * log(n+1) + const
    on window [n0, n1], skipping values below EPS_PAIR.

    This is the exact convention of O16_pair_observable.py (function fit_delta),
    which uses np.log(nf[mask] + 1) rather than np.log(nf[mask]).
    The +1 shift is responsible for the published delta_pair ~ 7.44 at q in {29,61}.

    Returns (delta_pair, R2) or (nan, nan) if fewer than 2 valid points.
    """
    n_arr = ns[n0:n1+1].astype(float)
    s_arr = sigma_pair[n0:min(n1+1, len(sigma_pair))]
    # Pad with zeros if sigma_pair is shorter than the window
    if len(s_arr) < len(n_arr):
        s_arr = np.concatenate([s_arr, np.zeros(len(n_arr) - len(s_arr))])
    mask = (n_arr > 0) & (s_arr > EPS_PAIR)
    if mask.sum() < 2:
        return np.nan, np.nan
    log_n = np.log(n_arr[mask] + 1)
    log_s = np.log(s_arr[mask])
    coef  = np.polyfit(log_n, log_s, 1)
    resid = log_s - np.polyval(coef, log_n)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((log_s - log_s.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan
    return float(-coef[0]), float(r2)


# ============================================================
# MAIN COMPUTATION: ONE PRIME
# ============================================================

def _compute_one_pair(idx, c, qc, shells, ns, q, gens, n_max_block,
                      n0, n1, M, base_seed):
    """
    Compute M samples of sigma_pair for the conjugate pair (c, q-c).
    Called in parallel by run_one_prime; must be a module-level function
    (required by multiprocessing pickling).

    Uses a per-pair seed derived from base_seed so results are reproducible
    regardless of the number of workers.
    """
    rng = np.random.default_rng(base_seed + idx * 997 + c * 7)
    n_shells = len(shells)
    sp_accumulator = np.zeros(n_shells)
    delta_arr        = np.full(M, np.nan)
    r2_arr           = np.full(M, np.nan)
    sc_accumulator   = np.zeros(n_shells)
    sqmc_accumulator = np.zeros(n_shells)

    for m in range(M):
        cb_c  = sample_block_with_c1(c,  q, rng)
        cb_qc = sample_block_with_c1(qc, q, rng)

        sv_c,  _, _, _ = compute_block_capacity(
            shells, cb_c,  q, gens, n_max=n_max_block)
        sv_qc, _, _, _ = compute_block_capacity(
            shells, cb_qc, q, gens, n_max=n_max_block)

        n_sh = min(len(sv_c), len(sv_qc))
        sp   = sv_c[:n_sh] * sv_qc[:n_sh]
        sp_accumulator[:n_sh]   += sp
        sc_accumulator[:n_sh]   += sv_c[:n_sh]
        sqmc_accumulator[:n_sh] += sv_qc[:n_sh]

        d, r2 = fit_delta_pair(sp, ns, n0, min(n1, n_sh - 1))
        delta_arr[m] = d
        r2_arr[m]    = r2

    return idx, sp_accumulator / M, sc_accumulator / M, sqmc_accumulator / M, delta_arr, r2_arr


def run_one_prime(q, M=M_PER_PAIR_DEFAULT, seed=DEFAULT_SEED,
                  bfs_frac=None, n_max_block=None, n0=None, n1=None,
                  n_jobs=N_JOBS_DEFAULT, auto_window=False, verbose=True):
    """
    O25 computation for prime q: all (q-1)//2 conjugate pairs, M samples each.

    Parameters
    ----------
    q            : prime
    M            : samples per pair
    seed         : RNG seed (each pair gets a deterministic derived seed)
    bfs_frac     : BFS fraction (default from BFS_FRAC table)
    n_max_block  : n_max per block (default from N_MAX_BLOCK table)
    n0, n1       : fitting window (default from WINDOW_O12 table)
    n_jobs       : parallel workers (-1 = all cores, 1 = sequential)
    auto_window  : if True, calibrate [n0,n1] from actual sigma_bar after BFS,
                   ignoring WINDOW_O12 table (use with --bfs-frac 0.99)
    verbose      : print progress

    Returns
    -------
    dict with all results (see save_npz for the stored fields)
    """
    if bfs_frac    is None: bfs_frac    = BFS_FRAC.get(q, BFS_FRAC_FALLBACK)
    if n_max_block is None: n_max_block = N_MAX_BLOCK.get(q, N_MAX_FALLBACK)
    # Window: use WINDOW_O12 table if available and auto_window not requested,
    # otherwise calibrate from the actual sigma_bar after BFS (find_fitting_window).
    use_table_window = (not auto_window) and (q in WINDOW_O12)
    if use_table_window:
        n0, n1 = WINDOW_O12[q]

    t0   = time.perf_counter()

    if verbose:
        print(f"  q={q}: BFS (frac={bfs_frac}, n_max_block={n_max_block})...")

    gens   = build_generators(q)
    shells = bfs_shells(None, None, gens, q, bfs_frac)
    ns     = np.arange(len(shells), dtype=np.int64)
    shell_sizes = np.array([len(s) for s in shells], dtype=np.int64)

    if verbose:
        print(f"  q={q}: {len(shells)} shells, |G_q_partial|={shell_sizes.sum()}")

    # Window calibration
    if not use_table_window:
        # Estimate sigma_bar from a small probe sample (5 random blocks)
        # to feed find_fitting_window, without running the full M*n_pairs jobs.
        probe_rng = np.random.default_rng(seed + 999999)
        probe_sigmas = []
        probe_gens = build_generators(q)
        for _ in range(5):
            c_probe = int(probe_rng.integers(1, q))
            c2 = int(probe_rng.integers(1, q))
            c3 = int(probe_rng.integers(1, q))
            while (c_probe + c2 + c3) % q == 0:
                c3 = int(probe_rng.integers(1, q))
            cb = np.array([c_probe, c2, c3], dtype=np.int64)
            sv, _, _, _ = compute_block_capacity(
                shells, cb, q, probe_gens, n_max=n_max_block)
            pad = len(shells) - len(sv)
            if pad > 0:
                sv = np.concatenate([sv, np.zeros(pad)])
            probe_sigmas.append(sv[:len(shells)])
        sigma_bar_probe = np.mean(probe_sigmas, axis=0)
        n0, n1 = find_fitting_window(ns[1:], sigma_bar_probe[1:], q)
        n0 = max(n0, 1)
        n1 = min(n1, len(shells) - 1)
        if verbose:
            print(f"  q={q}: auto-calibrated window=[{n0},{n1}]"
                  f" (from {len(shells)} shells, bfs_frac={bfs_frac})")
    else:
        if verbose:
            print(f"  q={q}: window=[{n0},{n1}] (from WINDOW_O12 table)")

    # All conjugate pairs c in {1, ..., (q-1)//2}
    pairs     = [(c, q - c) for c in range(1, (q - 1) // 2 + 1)]
    n_pairs   = len(pairs)
    pairs_arr = np.array(pairs, dtype=np.int64)

    if verbose:
        import os
        n_cores = os.cpu_count() if n_jobs == -1 else abs(n_jobs)
        print(f"  q={q}: {n_pairs} pairs x M={M} samples  "
              f"(n_jobs={n_jobs}, ~{n_cores} workers)...")

    # Parallel computation: each pair is independent (O24 verticality).
    #
    # Progress: pairs are processed in fixed-size batches of PROGRESS_BATCH,
    # independent of n_cores, so the first log line appears quickly even on
    # machines with many cores (avoids waiting for a full wave of 32 or 120
    # simultaneous jobs before any output).
    #
    # Backend: loky shares the shells object via mmap on POSIX, avoiding
    # repeated pickling of the large shells list for each pair.
    PROGRESS_BATCH = 5

    job_results   = []
    n_done        = 0
    t_pairs_start = time.perf_counter()

    for batch_start in range(0, n_pairs, PROGRESS_BATCH):
        batch = list(enumerate(pairs))[batch_start:batch_start + PROGRESS_BATCH]
        batch_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_compute_one_pair)(
                idx, c, qc, shells, ns, q, gens, n_max_block, n0, n1, M, seed
            )
            for idx, (c, qc) in batch
        )
        job_results.extend(batch_results)
        n_done += len(batch)
        if verbose:
            elapsed = time.perf_counter() - t_pairs_start
            rate    = n_done / elapsed
            eta_s   = (n_pairs - n_done) / rate if rate > 0 else float('inf')
            eta_str = (f"{eta_s/3600:.1f}h" if eta_s >= 3600
                       else f"{eta_s/60:.0f}min")
            done_deltas = [np.nanmean(r[4]) for r in batch_results
                           if np.any(np.isfinite(r[4]))]
            preview = (f"  batch mean={np.mean(done_deltas):.3f}"
                       if done_deltas else "")
            print(f"  q={q}: {n_done}/{n_pairs} pairs done  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta_str}{preview}",
                  flush=True)

    # Reassemble results (order may differ from submission order)
    sigma_pair_mean  = np.zeros((n_pairs, len(shells)))
    sigma_c_mean     = np.zeros((n_pairs, len(shells)))
    sigma_qmc_mean   = np.zeros((n_pairs, len(shells)))
    delta_samples    = np.full((n_pairs, M), np.nan)
    r2_samples       = np.full((n_pairs, M), np.nan)

    for idx, sp_mean, sc_mean, sqmc_mean, d_arr, r2_arr in job_results:
        sigma_pair_mean[idx]  = sp_mean
        sigma_c_mean[idx]     = sc_mean
        sigma_qmc_mean[idx]   = sqmc_mean
        delta_samples[idx]    = d_arr
        r2_samples[idx]       = r2_arr

    # Per-pair statistics
    delta_pair_mean = np.nanmean(delta_samples, axis=1)
    delta_pair_std  = np.nanstd(delta_samples,  axis=1)
    r2_mean         = np.nanmean(r2_samples,    axis=1)

    # Global delta_pair: fit on grand mean sigma_pair across all pairs
    grand_mean = np.mean(sigma_pair_mean, axis=0)
    delta_global, r2_global = fit_delta_pair(grand_mean, ns, n0, n1)

    dt = time.perf_counter() - t0

    if verbose:
        n_valid = np.sum(np.isfinite(delta_pair_mean))
        print(f"  q={q}: delta_pair(mean/pairs)={np.nanmean(delta_pair_mean):.4f} "
              f"+/- {np.nanstd(delta_pair_mean):.4f}  "
              f"({n_valid}/{n_pairs} valid pairs)  "
              f"[global={delta_global:.4f}, R2={r2_global:.4f}]  "
              f"time={dt:.1f}s")

    return dict(
        q=q, seed=seed, M_per_pair=M,
        bfs_frac=bfs_frac, n_max_block=n_max_block,
        pairs=pairs_arr,
        ns=ns, shell_sizes=shell_sizes,
        n0=n0, n1=n1,
        sigma_pair_mean=sigma_pair_mean,
        sigma_c_mean=sigma_c_mean,
        sigma_qmc_mean=sigma_qmc_mean,
        delta_pair_samples=delta_samples,
        delta_pair_mean=delta_pair_mean,
        delta_pair_std=delta_pair_std,
        r2_mean=r2_mean,
        delta_pair_global=delta_global,
        r2_global=r2_global,
        wall_time_s=dt,
    )


# ============================================================
# SAVE / LOAD
# ============================================================

def save_npz(res, out_path):
    """Save result dict to .npz checkpoint."""
    np.savez(
        out_path,
        q                  = np.int64(res["q"]),
        seed               = np.int64(res["seed"]),
        M_per_pair         = np.int64(res["M_per_pair"]),
        bfs_frac           = np.float64(res["bfs_frac"]),
        n_max_block        = np.int64(res["n_max_block"]),
        pairs              = res["pairs"].astype(np.int64),
        ns                 = res["ns"].astype(np.int64),
        shell_sizes        = res["shell_sizes"].astype(np.int64),
        n0                 = np.int64(res["n0"]),
        n1                 = np.int64(res["n1"]),
        sigma_pair_mean    = res["sigma_pair_mean"].astype(np.float64),
        sigma_c_mean       = res["sigma_c_mean"].astype(np.float64),
        sigma_qmc_mean     = res["sigma_qmc_mean"].astype(np.float64),
        delta_pair_samples = res["delta_pair_samples"].astype(np.float64),
        delta_pair_mean    = res["delta_pair_mean"].astype(np.float64),
        delta_pair_std     = res["delta_pair_std"].astype(np.float64),
        r2_mean            = res["r2_mean"].astype(np.float64),
        delta_pair_global  = np.float64(res["delta_pair_global"]),
        r2_global          = np.float64(res["r2_global"]),
        wall_time_s        = np.float64(res["wall_time_s"]),
    )
    print(f"  Saved: {out_path}")


def verify_npz(path):
    """Quick sanity check: load and print key results."""
    z  = np.load(path)
    q  = int(z["q"])
    n0 = int(z["n0"])
    n1 = int(z["n1"])
    dg = float(z["delta_pair_global"])
    r2 = float(z["r2_global"])
    dm = float(np.nanmean(z["delta_pair_mean"]))
    ds = float(np.nanstd(z["delta_pair_mean"]))
    np_ = int(z["pairs"].shape[0])
    M  = int(z["M_per_pair"])
    print(f"  [verify] q={q}  n_pairs={np_}  M={M}  window=[{n0},{n1}]  "
          f"delta_pair(mean/pairs)={dm:.4f}+/-{ds:.4f}  "
          f"[global={dg:.4f}, R2={r2:.4f}]")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="O25: systematic delta_pair across all pairs and primes"
    )
    parser.add_argument(
        "--primes", type=int, nargs="+",
        default=DEFAULT_PRIMES,
        help=f"Primes to compute (default: {DEFAULT_PRIMES})"
    )
    parser.add_argument(
        "--M", type=int, default=M_PER_PAIR_DEFAULT,
        help=f"Samples per pair (default: {M_PER_PAIR_DEFAULT})"
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=OUTPUT_DIR,
    )
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--bfs-frac",    type=float, default=None,
                        help="BFS fraction (overrides table)")
    parser.add_argument("--n-max",       type=int,   default=None,
                        help="n_max per block (overrides table)")
    parser.add_argument("--n-jobs",      type=int,   default=N_JOBS_DEFAULT,
                        help=f"Parallel workers (-1=all cores, default: {N_JOBS_DEFAULT})")
    parser.add_argument("--auto-window", action="store_true",
                        help="Calibrate [n0,n1] from actual BFS data "
                             "(required when --bfs-frac overrides the table)")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute even if output exists")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("o25_paired_pipeline.py")
    print("======================")
    print(f"Primes     : {args.primes}")
    print(f"M per pair : {args.M}")
    print(f"n_jobs     : {args.n_jobs}")
    print(f"bfs_frac   : {args.bfs_frac if args.bfs_frac else 'from table'}")
    print(f"window     : {'auto-calibrate' if args.auto_window else 'from WINDOW_O12 table'}")
    print(f"Seed       : {args.seed}")
    print(f"Output dir : {args.out_dir}")
    print()

    total_t0 = time.perf_counter()
    summary  = []

    for q in sorted(set(args.primes)):
        out_path = args.out_dir / f"q{q}_o25.npz"
        print(f"--- q={q} ---")

        if out_path.exists() and not args.force:
            print(f"  Exists: {out_path}  (--force to recompute)")
            verify_npz(out_path)
            z = np.load(out_path)
            summary.append((
                q,
                float(z["delta_pair_global"]), float(z["r2_global"]),
                float(np.nanmean(z["delta_pair_mean"])),
                float(np.nanstd(z["delta_pair_mean"])),
                int(z["n0"]), int(z["n1"]),
            ))
            print()
            continue

        res = run_one_prime(
            q=q, M=args.M, seed=args.seed,
            bfs_frac=args.bfs_frac,
            n_max_block=args.n_max,
            n_jobs=args.n_jobs,
            auto_window=args.auto_window,
            verbose=True,
        )
        save_npz(res, out_path)
        verify_npz(out_path)
        summary.append((
            q,
            res["delta_pair_global"], res["r2_global"],
            float(np.nanmean(res["delta_pair_mean"])),
            float(np.nanstd(res["delta_pair_mean"])),
            res["n0"], res["n1"],
        ))
        print()

    total_dt = time.perf_counter() - total_t0

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"{'q':>6}  {'mean(pairs)':>12}  {'std':>8}  {'global':>10}  {'R2':>8}  {'window':>10}")
    print("-" * 74)
    for q, dg, r2, dm, ds, n0, n1 in summary:
        print(f"{q:>6}  {dm:>12.4f}  {ds:>8.4f}  {dg:>10.4f}  {r2:>8.4f}  [{n0},{n1}]")
    print("=" * 74)
    print(f"Total time: {total_dt:.1f}s")
    print()
    print("Next: python o25_analysis.py  (figures and scaling law)")


if __name__ == "__main__":
    main()