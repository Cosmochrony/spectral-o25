"""n1_parallel.py -- multicore driver for the BFS saturation depth n1(q).

Same faithful measurement as n1_resumable.py (validated to reproduce the production O28 n1 on
q in {29,61,101,151,211} -> {4,8,11,13,14}), but runs the N_PROBE=5 probe blocks of each prime
in parallel via joblib. Intended for a real multicore machine with no per-call time limit, so
no checkpointing is used: each prime is computed in one shot and its n1 written to JSON.

n1 = find_fitting_window over sigma_bar = mean of 5 probe blocks (rng seed+999999),
sigma(n)=delta_r(n)/|S_n|, EPS_SAT=1e-3, absolute 1e-10 Gram-Schmidt threshold.

Usage (uses all cores; ~5 blocks run concurrently per prime):
  python n1_parallel.py --primes 401 503 601 --jobs -1 --out-dir n1_scaling_out

To re-validate the protocol first:
  python n1_parallel.py --primes 29 61 101 151 211        # expect n1 = 4 8 11 13 14
"""
import os
# Force single-threaded BLAS in every (worker) process BEFORE numpy imports its backend.
# Parallelism is across the 5 probe blocks (joblib); nested BLAS threads inside each worker
# oversubscribe the cores and, on Apple Accelerate (NEWLAPACK) under loky, segfault in
# cblas_zhemm during the complex matmul of rank_update. One BLAS thread per worker fixes it.
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, pathlib, sys, time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import (build_generators, heisenberg_mul_batch,
                          fingerprint_vectors_batch, find_fitting_window, EPS_SAT)

DEFAULT_SEED = 42
N_PROBE = 5
EPS_GS = 1e-10
CHUNK = 1200


def build_bfs(q, n_cap):
    gens = [np.array(g, np.int64) for g in build_generators(q)]
    visited = np.array([0], np.int64)
    frontier = np.array([[0, 0, 0]], np.int64)
    shells = [frontier]
    for _ in range(n_cap):
        nb = np.concatenate([heisenberg_mul_batch(frontier, g, q) for g in gens], 0)
        keys = np.unique((nb[:, 0] * q + nb[:, 1]) * q + nb[:, 2])
        new = keys[~np.isin(keys, visited, assume_unique=True)]
        if new.size == 0:
            break
        visited = np.union1d(visited, new)
        a = new // (q * q); b = (new // q) % q; g = new % q
        frontier = np.stack([a, b, g], 1).astype(np.int64)
        shells.append(frontier)
    return shells


def _cmm(A, B):
    """Complex A @ B routed through real (dgemm) matmuls.
    Apple Accelerate's complex matmul (zgemm / CDOUBLE_matmul) segfaults on large complex
    operands (confirmed: EXC_BAD_ACCESS in libBLAS at q=601); the real path is robust."""
    Ar, Ai = A.real, A.imag
    Br, Bi = B.real, B.imag
    return (Ar @ Br - Ai @ Bi) + 1j * (Ar @ Bi + Ai @ Br)


def rank_update(basis, vecs, q):
    if basis.shape[0] >= q:
        return basis, 0
    if basis.shape[0] > 0:
        resid = vecs - _cmm(_cmm(vecs, basis.conj().T), basis)
    else:
        resid = vecs
    surv = resid[np.linalg.norm(resid, axis=1) > EPS_GS]
    dr = 0
    for w in surv:
        if basis.shape[0] >= q:
            break
        if basis.shape[0] > 0:
            w = w - _cmm(basis.T, _cmm(basis.conj(), w[:, None]))[:, 0]
        nrm = np.linalg.norm(w)
        if nrm > EPS_GS:
            basis = np.vstack([basis, (w / nrm)[None, :]]); dr += 1
    return basis, dr


def one_block(shells, cb, q, n_proc):
    gens_arr = np.array(build_generators(q), np.int64)
    basis = np.empty((0, q), np.complex128)
    sigma = [0.0]
    # Adaptive chunk: cap peak memory ~ chunk*64*q*16 bytes per worker (shrinks for large q).
    chunk_sz = max(200, min(CHUNK, 200000 // q))
    for sidx in range(1, n_proc + 1):
        if sidx >= len(shells):
            break
        shell = shells[sidx]; dr = 0
        for s in range(0, len(shell), chunk_sz):
            chunk = np.asarray(shell[s:s + chunk_sz], np.int64)
            vecs = fingerprint_vectors_batch(chunk, np.array(cb, np.int64), gens_arr, q)
            basis, d = rank_update(basis, vecs, q); dr += d
            if basis.shape[0] >= q:
                break
        sigma.append(dr / len(shell))
        if basis.shape[0] >= q:
            break
    return sigma, basis.shape[0]


def probe_blocks(q, seed):
    rng = np.random.default_rng(seed + 999999); out = []
    for _ in range(N_PROBE):
        c = [int(rng.integers(1, q)) for _ in range(3)]
        while sum(c) % q == 0:
            c[2] = int(rng.integers(1, q))
        out.append(c)
    return out


def measure(q, seed, safety, jobs, out_dir):
    t0 = time.perf_counter()
    n_cap = int(np.ceil(safety * np.sqrt(q))) + 6
    n_proc = int(np.ceil(np.sqrt(q))) + 2
    shells = build_bfs(q, n_cap)
    n_shells = len(shells)
    blocks = probe_blocks(q, seed)
    out = Parallel(n_jobs=jobs)(delayed(one_block)(shells, cb, q, n_proc) for cb in blocks)
    sigs, ranks = [], []
    for sv, rk in out:
        sv = np.array(sv); ranks.append(rk)
        if len(sv) < n_shells:
            sv = np.concatenate([sv, np.zeros(n_shells - len(sv))])
        sigs.append(sv[:n_shells])
    sigma_bar = np.mean(sigs, 0)
    ns = np.arange(n_shells)
    n0, n1 = find_fitting_window(ns[1:], sigma_bar[1:], q)
    n0 = max(n0, 1); n1 = min(n1, n_shells - 1)
    res = {"q": q, "n0": int(n0), "n1": int(n1), "n_proc": n_proc, "n_shells": n_shells,
           "min_rank": int(min(ranks)), "saturated": bool(n1 < n_shells - 1),
           "sigma_bar": [float(x) for x in sigma_bar]}
    (out_dir / f"n1_q{q}.json").write_text(json.dumps(res, indent=2))
    print(f"  q={q}: n1={n1}  n1/sqrt(q)={n1/np.sqrt(q):.3f}  "
          f"rank={min(ranks)}/{q}  ({time.perf_counter()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--safety", type=float, default=1.4)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("n1_scaling_out"))
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in sorted(set(a.primes)):
        cp = a.out_dir / f"n1_q{q}.json"
        if cp.exists():
            r = json.loads(cp.read_text())
            if "n1" in r:
                print(f"  q={q}: cached n1={r['n1']}  (delete {cp.name} to recompute)")
                rows.append(r); continue
        rows.append(measure(q, a.seed, a.safety, a.jobs, a.out_dir))
    print("\n q     n1   n1/q     n1/sqrt(q)")
    for r in rows:
        print(f" {r['q']:<5} {r['n1']:<4} {r['n1']/r['q']:<8.4f} "
              f"{r['n1']/np.sqrt(r['q']):.3f}")


if __name__ == "__main__":
    main()
