"""Exact ball-growth constant C_Heis in |B_n| ~ C_Heis n^4 for Cay(Heis3(Z/qZ),{X^±,Y^±}).
Pure BFS with an integer-set frontier (memory-light); q large so no wraparound to depth n_max.
Confirms homogeneous dimension D = 4 (Bass-Guivarch) and C_Heis ~ 0.427."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import build_generators, heisenberg_mul_batch


def ball_counts(q, n_max):
    visited = set([0]); frontier = np.array([[0, 0, 0]], np.int64); B = [1]; cum = 1
    gens = [np.array(g, np.int64) for g in build_generators(q)]
    for _ in range(1, n_max + 1):
        nb = np.concatenate([heisenberg_mul_batch(frontier, g, q) for g in gens], 0)
        keys = np.unique((nb[:, 0] * q + nb[:, 1]) * q + nb[:, 2])
        fresh = np.fromiter((k for k in keys.tolist() if k not in visited), np.int64)
        if fresh.size == 0:
            break
        visited.update(fresh.tolist())
        a = fresh // (q * q); b = (fresh // q) % q; g = fresh % q
        frontier = np.stack([a, b, g], 1).astype(np.int64); cum += fresh.size; B.append(cum)
    return np.array(B)


if __name__ == "__main__":
    q, n_max = 20011, 40
    B = ball_counts(q, n_max); n = np.arange(len(B))
    for nn in [20, 25, 30, 35, 40]:
        if nn < len(B):
            print(f" n={nn:>3} |B_n|={B[nn]:>9} B/n^4={B[nn]/nn**4:.5f} "
                  f"S/n^3={(B[nn]-B[nn-1])/nn**3:.5f}")
    m = (n >= 15) & (n < len(B))
    p = np.polyfit(np.log(n[m]), np.log(B[m]), 1); C = np.exp(p[1])
    print(f"\n power fit |B_n|~C n^p: p={p[0]:.4f} C={C:.5f}")
    print(f" kappa_naive = C^(-1/4) = {C**-0.25:.4f}  (if |B_n1| ~ q^2 exactly)")
