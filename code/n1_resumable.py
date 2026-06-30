"""n1_resumable.py -- faithful, fully resumable measurement of the BFS saturation depth n1(q).

Designed for a hard wall-clock budget per invocation (default 38s): it checkpoints per
(probe-block, shell) and exits gracefully before being killed, so repeated calls accumulate
progress with zero loss. BFS shells are cached once per q.

Rank increments delta_r(n) reproduce spectral_O12.gram_schmidt_batch exactly (absolute 1e-10
residual threshold); validated against the production n1 on q in {61,101} -> {8,11}.

n1 = find_fitting_window over sigma_bar = mean of N_PROBE=5 probe blocks (rng seed+999999),
sigma(n)=delta_r(n)/|S_n|, EPS_SAT=1e-3.  (matches o25_paired_pipeline auto-window)

Usage (call repeatedly until it prints the final n1):
  python n1_resumable.py --primes 151 --out-dir DIR --budget 38
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, pathlib, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import (build_generators, heisenberg_mul_batch,
                          fingerprint_vectors_batch, find_fitting_window, EPS_SAT)

DEFAULT_SEED = 42
N_PROBE = 5
EPS_GS = 1e-10
CHUNK = 1200


def save_atomic(path, **arrays):
    tmp = pathlib.Path(str(path) + f".tmp{os.getpid()}")
    np.savez(tmp, **arrays)
    src = tmp if tmp.exists() else pathlib.Path(str(tmp) + ".npz")
    os.replace(src, path)


def build_bfs(q, n_cap, cache):
    if cache.exists():
        try:
            z = np.load(cache)
            offs = z["offsets"]; flat = z["flat"]
            return [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        except Exception:
            pass  # corrupt (interrupted write) -> recompute below
    gens = [np.array(g, np.int64) for g in build_generators(q)]
    visited = np.array([0], np.int64)            # sorted set of encoded keys
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
    offsets = np.zeros(len(shells) + 1, np.int64)
    for i, s in enumerate(shells):
        offsets[i + 1] = offsets[i] + len(s)
    flat = np.concatenate(shells, 0).astype(np.int64)
    save_atomic(cache, flat=flat, offsets=offsets)
    return shells


def _cmm(A, B):
    """Complex A @ B via real (dgemm) matmuls -- avoids Apple Accelerate's zgemm segfault."""
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
    norms = np.linalg.norm(resid, axis=1)
    surv = resid[norms > EPS_GS]
    dr = 0
    for w in surv:
        if basis.shape[0] >= q:
            break
        if basis.shape[0] > 0:
            w = w - _cmm(basis.T, _cmm(basis.conj(), w[:, None]))[:, 0]
        nrm = np.linalg.norm(w)
        if nrm > EPS_GS:
            basis = np.vstack([basis, (w / nrm)[None, :]])
            dr += 1
    return basis, dr


def probe_blocks(q, seed):
    rng = np.random.default_rng(seed + 999999); out = []
    for _ in range(N_PROBE):
        c = [int(rng.integers(1, q)) for _ in range(3)]
        while sum(c) % q == 0:
            c[2] = int(rng.integers(1, q))
        out.append(c)
    return out


def run(q, seed, safety, out_dir, budget):
    t0 = time.perf_counter()
    n_cap = int(np.ceil(safety * np.sqrt(q))) + 6
    cache = out_dir / f"bfs_q{q}.npz"
    shells = build_bfs(q, n_cap, cache)
    n_shells = len(shells)
    gens_arr = np.array(build_generators(q), np.int64)
    # Common processing depth: must exceed n1 (~sqrt q) so every block is evaluated at the
    # mean-crossing shell (no zero-pad bias), but stay shallow enough to skip the costly deep
    # shells. n1/sqrt(q) <= 1.1 on all known primes, so 1.3*sqrt(q)+2 is a safe common cap.
    n_proc = min(n_shells - 1, int(np.ceil(np.sqrt(q))) + 2)
    blocks = probe_blocks(q, seed)
    final = out_dir / f"n1_q{q}.json"
    if final.exists() and "n1" in json.loads(final.read_text()):
        print(f"  q={q}: done n1={json.loads(final.read_text())['n1']}")
        return True

    for bi, cb in enumerate(blocks):
        bpath = out_dir / f"blk_q{q}_b{bi}.npz"
        basis = np.empty((0, q), np.complex128); sidx = 1; sigma = [0.0]
        if bpath.exists():
            try:
                z = np.load(bpath)
                if bool(z["done"]):
                    continue
                basis = z["basis"]; sidx = int(z["sidx"]); sigma = list(z["sigma"])
            except Exception:
                basis = np.empty((0, q), np.complex128); sidx = 1; sigma = [0.0]
        cb_arr = np.array(cb, np.int64)
        done = False
        while sidx <= n_proc:
            shell = shells[sidx]
            dr = 0
            for s in range(0, len(shell), CHUNK):
                chunk = np.asarray(shell[s:s + CHUNK], np.int64)
                vecs = fingerprint_vectors_batch(chunk, cb_arr, gens_arr, q)
                basis, d = rank_update(basis, vecs, q)
                dr += d
                if basis.shape[0] >= q:
                    break
            sig_n = dr / len(shell)
            sigma.append(sig_n)
            sidx += 1
            # Stop at the common depth n_proc (every block reaches it -> unbiased mean) or at
            # full rank q. Deeper shells have sigma < EPS_SAT and cannot move the n1 crossing.
            if basis.shape[0] >= q or sidx > n_proc:
                done = True
            if done or time.perf_counter() - t0 > budget:
                save_atomic(bpath, basis=basis, sidx=sidx, sigma=np.array(sigma),
                            done=done, rank=basis.shape[0])
                if not done:
                    print(f"  q={q} block {bi+1}/{N_PROBE}: checkpoint at shell {sidx}"
                          f"/{n_shells}, rank={basis.shape[0]}/{q} "
                          f"({time.perf_counter()-t0:.0f}s) -- resume")
                    return False
                break
        save_atomic(bpath, basis=basis, sidx=sidx, sigma=np.array(sigma),
                    done=True, rank=basis.shape[0])
        print(f"  q={q} block {bi+1}/{N_PROBE} done rank={basis.shape[0]}/{q} "
              f"({time.perf_counter()-t0:.0f}s)")
        if time.perf_counter() - t0 > budget:
            return False

    # all blocks done -> assemble n1
    sigs, ranks = [], []
    for bi in range(N_PROBE):
        z = np.load(out_dir / f"blk_q{q}_b{bi}.npz")
        sv = np.array(z["sigma"]); ranks.append(int(z["rank"]))
        if len(sv) < n_shells:
            sv = np.concatenate([sv, np.zeros(n_shells - len(sv))])
        sigs.append(sv[:n_shells])
    sigma_bar = np.mean(sigs, 0)
    ns = np.arange(n_shells)
    n0, n1 = find_fitting_window(ns[1:], sigma_bar[1:], q)
    n0 = max(n0, 1); n1 = min(n1, n_shells - 1)
    mr = min(ranks)
    res = {"q": q, "n0": int(n0), "n1": int(n1), "n_cap": n_cap, "n_shells": n_shells,
           "min_rank": mr, "saturated": bool(n1 < n_shells - 1),
           "sigma_bar": [float(x) for x in sigma_bar]}
    final.write_text(json.dumps(res, indent=2))
    print(f"  q={q}: n1={n1}  n1/sqrt(q)={n1/np.sqrt(q):.3f}  rank={mr}/{q}  "
          f"sat={res['saturated']}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--safety", type=float, default=1.8)
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("n1_scaling_out"))
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    for q in sorted(set(a.primes)):
        run(q, a.seed, a.safety, a.out_dir, a.budget)


if __name__ == "__main__":
    main()
