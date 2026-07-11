"""delta_pair_resumable.py -- shell-level resumable delta_pair campaign for ONE large prime.

Computes the grand-mean pair observable sigma_pair(n) = <sigma_c(n) * sigma_{q-c}(n)> over
K conjugate pairs x M samples, to a capped depth n_proc, with per-block shell-level
checkpoints and a hard wall-clock budget per invocation.  Call repeatedly until it prints
DONE; designed for environments with a short per-call time cap.

Capacity per block uses the validated rank machinery from n1_resumable (rank_update, EPS_GS
=1e-10), which reproduces spectral_O12.gram_schmidt_batch exactly -> sigma_c(n)=dr_n/|S_n|
is the production observable.  BFS shells are the capped build_bfs (identical early shells).

Block list is precomputed deterministically with the o25 pair-seed scheme
(rng = default_rng(seed + idx*997 + c*7); draw cb_c then cb_qc per sample), so results are
reproducible and comparable to o25_paired_pipeline.

USAGE (repeat until DONE):
  python delta_pair_resumable.py --q 307 --K 4 --M 4 --n-proc 13 --budget 40 --out-dir dpc_out
On completion writes dpc_out/dpc_q{q}.npz (grand_sigma_pair, sigma_c_mean, sigma_qmc_mean).
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


def sample_block_with_c1(c1, q, rng, max_attempts=2000):
    for _ in range(max_attempts):
        c2 = int(rng.integers(1, q)); c3 = int(rng.integers(1, q))
        if (c1 + c2 + c3) % q != 0:
            return np.array([c1, c2, c3], dtype=np.int64)
    raise RuntimeError(f"cannot sample generic block c1={c1} q={q}")


def block_list(q, K, M, seed):
    """Deterministic (block_id -> c_block) plus the pair/sample/role index map."""
    half = (q - 1) // 2
    if K >= half:
        cs = list(range(1, half + 1))
    else:
        cs = sorted(set(int(round(x)) for x in np.linspace(1, half, K)))
    blocks = {}   # bid -> c_block (np.int64[3])
    layout = []   # (idx, m, c, qc, bid_c, bid_q)
    for idx, c in enumerate(cs):
        qc = q - c
        rng = np.random.default_rng(seed + idx * 997 + c * 7)
        for m in range(M):
            cb_c = sample_block_with_c1(c, q, rng)
            cb_qc = sample_block_with_c1(qc, q, rng)
            bid_c = f"{idx}_{m}_c"; bid_q = f"{idx}_{m}_q"
            blocks[bid_c] = cb_c; blocks[bid_q] = cb_qc
            layout.append((idx, m, c, qc, bid_c, bid_q))
    return cs, blocks, layout


def compute_block(bid, c_block, shells, gens_arr, q, n_proc, out_dir, budget, t0):
    """Advance one block's GS to n_proc or until budget; checkpoint sigma/basis/sidx."""
    bpath = out_dir / f"blk_q{q}_{bid}.npz"
    # Start at shell 0 (the origin): its fingerprint vectors seed the Gram-Schmidt basis,
    # exactly as spectral_O12.compute_block_capacity does.  Skipping it shifts every dr_n.
    basis = np.empty((0, q), np.complex128); sidx = 0; sigma = []
    if bpath.exists():
        try:
            z = np.load(bpath)
            if bool(z["done"]):
                return bid, list(z["sigma"]), True
            basis = z["basis"]; sidx = int(z["sidx"]); sigma = list(z["sigma"])
        except Exception:
            basis = np.empty((0, q), np.complex128); sidx = 0; sigma = []
    cb = np.asarray(c_block, np.int64)
    bmat = basis if basis.shape[0] > 0 else None
    done = False
    while sidx <= n_proc and sidx < len(shells):
        shell = np.asarray(shells[sidx], np.int64)
        dr = 0
        for s in range(0, len(shell), CHUNK):
            vecs = fingerprint_vectors_batch(shell[s:s + CHUNK], cb, gens_arr, q)
            bmat, d = gram_schmidt_batch(bmat, vecs)  # production GS (EPS_GS, single pass)
            dr += d
            if bmat is not None and bmat.shape[0] >= q:
                break
        sigma.append(dr / len(shell))
        sidx += 1
        rk = 0 if bmat is None else bmat.shape[0]
        if rk >= q or sidx > n_proc:
            done = True
        if done or time.perf_counter() - t0 > budget:
            break
    basis = bmat if bmat is not None else np.empty((0, q), np.complex128)
    if sidx > n_proc or basis.shape[0] >= q:
        done = True
    tmp = pathlib.Path(str(bpath) + f".tmp{os.getpid()}")
    np.savez(tmp, basis=basis, sidx=sidx, sigma=np.array(sigma), done=done)
    os.replace(tmp if tmp.exists() else pathlib.Path(str(tmp) + ".npz"), bpath)
    return bid, sigma, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--n-proc", type=int, default=None, help="cap depth (default: n1 from json or 1.2 sqrt q+2)")
    ap.add_argument("--safety", type=float, default=1.6)
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("dpc_out"))
    a = ap.parse_args()
    q = a.q
    a.out_dir.mkdir(parents=True, exist_ok=True)
    final = a.out_dir / f"dpc_q{q}.npz"
    if final.exists():
        print(f"q={q}: DONE (exists {final.name})"); return
    t0 = time.perf_counter()
    gens = build_generators(q); gens_arr = np.array(gens, np.int64)
    n_cap = int(np.ceil(a.safety * np.sqrt(q))) + 6
    shells = build_bfs(q, n_cap, a.out_dir / f"bfs_q{q}.npz")
    # depth
    if a.n_proc is not None:
        n_proc = a.n_proc
    else:
        jp = pathlib.Path(__file__).parent / "n1_scaling_out" / f"n1_q{q}.json"
        n_proc = (int(json.loads(jp.read_text())["n1"]) if jp.exists()
                  else int(np.ceil(1.2 * np.sqrt(q))) + 2)
    n_proc = min(n_proc, len(shells) - 1)
    shell_sizes = np.array([len(s) for s in shells], np.int64)

    cs, blocks, layout = block_list(q, a.K, a.M, a.seed)
    bids = list(blocks.keys())
    # which remain
    def is_done(bid):
        p = a.out_dir / f"blk_q{q}_{bid}.npz"
        if not p.exists():
            return False
        try:
            return bool(np.load(p)["done"])
        except Exception:
            return False
    remaining = [b for b in bids if not is_done(b)]
    print(f"q={q} n_proc={n_proc} shells={len(shells)} blocks={len(bids)} "
          f"remaining={len(remaining)} budget={a.budget}s", flush=True)
    if remaining:
        Parallel(n_jobs=a.n_jobs, backend="loky", verbose=0)(
            delayed(compute_block)(b, blocks[b], shells, gens_arr, q, n_proc,
                                   a.out_dir, a.budget, t0)
            for b in remaining)
        still = [b for b in bids if not is_done(b)]
        print(f"q={q}: advanced; remaining now {len(still)} "
              f"(elapsed {time.perf_counter()-t0:.0f}s)", flush=True)
        if still:
            print("NOT DONE -- call again"); return

    # assemble
    n_sh = len(shells)
    def load_sigma(bid):
        sv = np.array(np.load(a.out_dir / f"blk_q{q}_{bid}.npz")["sigma"], float)
        if len(sv) < n_sh:
            sv = np.concatenate([sv, np.zeros(n_sh - len(sv))])
        return sv[:n_sh]
    Kp = len(cs)
    sp = np.zeros((Kp, n_sh)); sc = np.zeros((Kp, n_sh)); sq = np.zeros((Kp, n_sh))
    cnt = np.zeros(Kp)
    for (idx, m, c, qc, bid_c, bid_q) in layout:
        vc = load_sigma(bid_c); vq = load_sigma(bid_q)
        sp[idx] += vc * vq; sc[idx] += vc; sq[idx] += vq; cnt[idx] += 1
    for i in range(Kp):
        if cnt[i] > 0:
            sp[i] /= cnt[i]; sc[i] /= cnt[i]; sq[i] /= cnt[i]
    grand = sp.mean(axis=0)
    ns = np.arange(n_sh, dtype=np.int64)
    np.savez(final,
             q=np.int64(q), K=np.int64(Kp), M=np.int64(a.M), seed=np.int64(a.seed),
             safety=np.float64(a.safety), n_proc=np.int64(n_proc),
             ns=ns, shell_sizes=shell_sizes,
             pairs=np.array([(c, q - c) for c in cs], np.int64),
             sigma_pair_mean=sp, sigma_c_mean=sc, sigma_qmc_mean=sq,
             grand_sigma_pair=grand,
             wall_time_s=np.float64(time.perf_counter() - t0))
    print(f"q={q}: DONE -> {final.name} (Kp={Kp} M={a.M} n_proc={n_proc})", flush=True)


if __name__ == "__main__":
    main()
