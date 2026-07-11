"""delta_pair_convergence_campaign.py -- self-consistent delta_pair(q) campaign to large q,
with live ETA and full shell-level resume.

Goal (rate-side front "close beta*"):
    Test whether the interior exponent delta_pair(q) converges as q -> infinity, and
    extract delta_inf, hence beta* = 1/(delta_inf + 1/2).

WHY A NEW CAMPAIGN
------------------
The production paired pipeline builds the BFS graph up to frac*q^3 nodes, infeasible at
q in {307,401,503,601}.  But sigma_pair(n) = sigma_c(n)*sigma_{q-c}(n) only depends on
shells 0..n with n in the fitting interior (n <~ 1.1*sqrt(q)).  So we cap the BFS at
n_cap ~ safety*sqrt(q) shells (n1_resumable.build_bfs), which yields IDENTICAL early shells
and therefore IDENTICAL sigma_c values (validated: max abs diff = 0.0), at a tiny fraction
of the cost.  Depth is capped at n1+2 (n1 from n1_scaling_out) so deep post-knee shells are
not computed.

Per-block capacity uses spectral_O12.gram_schmidt_batch (EPS_GS=1e-10), starting at shell 0
(its fingerprint vectors seed the basis), so sigma_c(n)=dr_n/|S_n| is the production
observable (validated identical to compute_block_capacity, n>=0).

RESUME + ETA
------------
The parallel unit is a single block (there are K*(... )*M*2 of them per q).  Each block
checkpoints after every shell to out_dir/blk_q{q}_{id}.npz, so an interrupted run resumes
exactly where it stopped -- re-run the SAME command.  A timed probe block prints an upfront
wall-time estimate; thereafter ETA is updated live as blocks complete.  A finished prime
writes out_dir/dpc_q{q}.npz and its block files are removed; on restart finished primes are
skipped.

USAGE (re-run the same line to resume)
  python delta_pair_convergence_campaign.py --primes 307 401 503 601 \
         --K 24 --M 16 --out-dir dpc_out --n-jobs -1
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ACCELERATE_MAX_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, pathlib, sys, time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import build_generators, fingerprint_vectors_batch, gram_schmidt_batch
from n1_resumable import build_bfs

DEFAULT_SEED = 42
CHUNK = 1200
EPS_PAIR = 1e-15


def fmt_eta(s):
    if s < 0 or not np.isfinite(s):
        return "?"
    if s >= 3600:
        return f"{s/3600:.1f}h"
    if s >= 60:
        return f"{s/60:.0f}min"
    return f"{s:.0f}s"


def n1_window_end(q):
    jp = pathlib.Path(__file__).parent / "n1_scaling_out" / f"n1_q{q}.json"
    if jp.exists():
        return int(json.loads(jp.read_text())["n1"])
    return None


def sample_block_with_c1(c1, q, rng, max_attempts=2000):
    for _ in range(max_attempts):
        c2 = int(rng.integers(1, q)); c3 = int(rng.integers(1, q))
        if (c1 + c2 + c3) % q != 0:
            return np.array([c1, c2, c3], dtype=np.int64)
    raise RuntimeError(f"cannot sample generic block c1={c1} q={q}")


def block_list(q, K, M, seed):
    """Deterministic block_id -> c_block and the pair/sample/role layout (o25 pair-seed)."""
    half = (q - 1) // 2
    cs = (list(range(1, half + 1)) if K >= half
          else sorted(set(int(round(x)) for x in np.linspace(1, half, K))))
    blocks, layout = {}, []
    for idx, c in enumerate(cs):
        qc = q - c
        rng = np.random.default_rng(seed + idx * 997 + c * 7)
        for m in range(M):
            cb_c = sample_block_with_c1(c, q, rng)
            cb_qc = sample_block_with_c1(qc, q, rng)
            bidc, bidq = f"{idx}_{m}_c", f"{idx}_{m}_q"
            blocks[bidc] = cb_c; blocks[bidq] = cb_qc
            layout.append((idx, m, c, qc, bidc, bidq))
    return cs, blocks, layout


def _block_done(path):
    if not path.exists():
        return False
    try:
        return bool(np.load(path)["done"])
    except Exception:
        return False


def compute_block(bid, c_block, shells, gens_arr, q, n_proc, out_dir, save_interval=10.0):
    """Run one block's Gram-Schmidt to n_proc, time-checkpointing (save if >save_interval s
    since last save, plus at the end).  Cheap shallow shells trigger no I/O; expensive deep
    shells checkpoint every shell.  Resumes from out_dir/blk_q{q}_{bid}.npz if present."""
    bpath = out_dir / f"blk_q{q}_{bid}.npz"
    basis = np.empty((0, q), np.complex128); sidx = 0; sigma = []
    if bpath.exists():
        try:
            z = np.load(bpath)
            if bool(z["done"]):
                return bid
            basis = z["basis"]; sidx = int(z["sidx"]); sigma = list(z["sigma"])
        except Exception:
            basis = np.empty((0, q), np.complex128); sidx = 0; sigma = []
    cb = np.asarray(c_block, np.int64)
    bmat = basis if basis.shape[0] > 0 else None

    def save(done):
        bm = bmat if bmat is not None else np.empty((0, q), np.complex128)
        tmp = pathlib.Path(str(bpath) + f".tmp{os.getpid()}")
        np.savez(tmp, basis=bm, sidx=sidx, sigma=np.array(sigma), done=done)
        os.replace(tmp if tmp.exists() else pathlib.Path(str(tmp) + ".npz"), bpath)

    last_save = time.perf_counter()
    while sidx <= n_proc and sidx < len(shells):
        shell = np.asarray(shells[sidx], np.int64); dr = 0
        for s in range(0, len(shell), CHUNK):
            vecs = fingerprint_vectors_batch(shell[s:s + CHUNK], cb, gens_arr, q)
            bmat, d = gram_schmidt_batch(bmat, vecs); dr += d
            if bmat is not None and bmat.shape[0] >= q:
                break
        sigma.append(dr / len(shell)); sidx += 1
        rk = 0 if bmat is None else bmat.shape[0]
        if rk >= q:
            break
        if time.perf_counter() - last_save > save_interval:
            save(done=False); last_save = time.perf_counter()
    save(done=True)
    return bid


def run_one_prime(q, K, M, seed, safety, n_jobs, out_dir, verbose=True):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"dpc_q{q}.npz"
    if final.exists():
        if verbose:
            print(f"  q={q}: DONE (exists {final.name}, skip)", flush=True)
        return final
    t0 = time.perf_counter()
    gens = build_generators(q); gens_arr = np.array(gens, np.int64)
    n_cap = int(np.ceil(safety * np.sqrt(q))) + 6
    shells = build_bfs(q, n_cap, out_dir / f"bfs_q{q}.npz")
    n_sh = len(shells)
    n1 = n1_window_end(q)
    n_proc = min(n_sh - 1, (n1 + 2) if n1 is not None
                 else int(np.ceil(1.2 * np.sqrt(q))) + 2)
    shell_sizes = np.array([len(s) for s in shells], np.int64)
    cs, blocks, layout = block_list(q, K, M, seed)
    Kp = len(cs)
    bids = list(blocks.keys())
    total = len(bids)
    remaining = [b for b in bids if not _block_done(out_dir / f"blk_q{q}_{b}.npz")]
    already = total - len(remaining)
    if verbose:
        print(f"  q={q}: pairs={Kp} M={M} -> {total} blocks (resume: {already} done), "
              f"n_proc={n_proc}, shells={n_sh}, sum|S|={shell_sizes.sum()} "
              f"(BFS {time.perf_counter()-t0:.1f}s)", flush=True)

    nj = os.cpu_count() if n_jobs in (-1, None) else abs(n_jobs)
    if remaining:
        # Timed probe on the deepest-cost block (use the first remaining) for an upfront ETA.
        tprobe = time.perf_counter()
        compute_block(remaining[0], blocks[remaining[0]], shells, gens_arr, q, n_proc, out_dir)
        probe_s = time.perf_counter() - tprobe
        rem2 = [b for b in remaining[1:] if not _block_done(out_dir / f"blk_q{q}_{b}.npz")]
        est = probe_s * (len(rem2) / max(nj, 1))
        if verbose:
            print(f"  q={q}: probe block {probe_s:.1f}s; ~{len(rem2)} blocks left on "
                  f"{nj} cores -> est wall ~{fmt_eta(est)}", flush=True)
        # Live ETA as blocks complete.
        trun = time.perf_counter(); done_run = 0
        gen = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
            delayed(compute_block)(b, blocks[b], shells, gens_arr, q, n_proc, out_dir)
            for b in rem2)
        n_rem = len(rem2)
        step = max(1, n_rem // 20)  # ~20 ETA updates
        for _ in gen:
            done_run += 1
            if verbose and (done_run % step == 0 or done_run == n_rem):
                el = time.perf_counter() - trun
                rate = done_run / el if el > 0 else 0
                eta = (n_rem - done_run) / rate if rate > 0 else float('inf')
                print(f"  q={q}: {already+1+done_run}/{total} blocks "
                      f"({el/60:.1f}min, ETA {fmt_eta(eta)})", flush=True)

    # assemble
    def load_sigma(bid):
        sv = np.array(np.load(out_dir / f"blk_q{q}_{bid}.npz")["sigma"], float)
        if len(sv) < n_sh:
            sv = np.concatenate([sv, np.zeros(n_sh - len(sv))])
        return sv[:n_sh]
    sp = np.zeros((Kp, n_sh)); sc = np.zeros((Kp, n_sh)); sq = np.zeros((Kp, n_sh))
    cnt = np.zeros(Kp)
    for (idx, m, c, qc, bidc, bidq) in layout:
        vc = load_sigma(bidc); vq = load_sigma(bidq)
        sp[idx] += vc * vq; sc[idx] += vc; sq[idx] += vq; cnt[idx] += 1
    for i in range(Kp):
        if cnt[i] > 0:
            sp[i] /= cnt[i]; sc[i] /= cnt[i]; sq[i] /= cnt[i]
    grand = sp.mean(axis=0)
    ns = np.arange(n_sh, dtype=np.int64)
    np.savez(final,
             q=np.int64(q), K=np.int64(Kp), M=np.int64(M), seed=np.int64(seed),
             safety=np.float64(safety), n_proc=np.int64(n_proc),
             ns=ns, shell_sizes=shell_sizes,
             pairs=np.array([(c, q - c) for c in cs], np.int64),
             sigma_pair_mean=sp, sigma_c_mean=sc, sigma_qmc_mean=sq,
             grand_sigma_pair=grand, wall_time_s=np.float64(time.perf_counter() - t0))
    # cleanup block files
    for b in bids:
        p = out_dir / f"blk_q{q}_{b}.npz"
        if p.exists():
            try: p.unlink()
            except Exception: pass
    if verbose:
        nz = int(np.sum(grand > EPS_PAIR))
        print(f"  q={q}: DONE Kp={Kp} M={M} nz_grand={nz} "
              f"wall={ (time.perf_counter()-t0)/60:.1f}min -> {final.name}", flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+",
                    default=[29, 61, 101, 151, 211, 307, 401, 503, 601])
    ap.add_argument("--K", type=int, default=24, help="conjugate pairs per prime")
    ap.add_argument("--M", type=int, default=16, help="samples per pair")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--safety", type=float, default=1.6)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("dpc_out"))
    a = ap.parse_args()
    print("delta_pair_convergence_campaign (resumable, live ETA)")
    print(f"primes={a.primes} K={a.K} M={a.M} seed={a.seed} safety={a.safety} "
          f"n_jobs={a.n_jobs}", flush=True)
    for q in sorted(set(a.primes)):
        run_one_prime(q, a.K, a.M, a.seed, a.safety, a.n_jobs, a.out_dir)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
