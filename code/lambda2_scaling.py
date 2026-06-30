"""Spectral gap lambda_2 (algebraic connectivity) of Cay(Heis3(Z/qZ),{X^±,Y^±}).
Tests the diffusive scaling lambda_2 ~ 4*pi^2 / q^2 (gap of the horizontal abelianisation torus
(Z/qZ)^2), which sets the q^2 coverage scale used in the n1 = Theta(sqrt q) derivation.
Result: lambda_2 * q^2 -> ~39, rising toward 4*pi^2 = 39.478."""
import numpy as np, sys, os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_O12 import build_generators


def gap(q):
    idx = lambda a, b, g: (a * q + b) * q + g
    N = q ** 3; gens = build_generators(q)
    rows = []; cols = []
    for a in range(q):
        for b in range(q):
            for g in range(q):
                u = idx(a, b, g)
                for (ga, gb, gg) in gens:
                    rows.append(u); cols.append(idx((a + ga) % q, (b + gb) % q, (g + gg + a * gb) % q))
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    L = csr_matrix((deg, (range(N), range(N))), shape=(N, N)) - A
    ev = eigsh(L, k=2, which='SM', return_eigenvectors=False, maxiter=5000)
    return sorted(ev)[1]


if __name__ == "__main__":
    print(" q   lambda2      lambda2*q^2   (4 pi^2 = %.4f)" % (4 * np.pi ** 2))
    for q in [5, 7, 11, 13, 17, 19]:
        l2 = gap(q)
        print(f" {q:<3} {l2:.6f}   {l2 * q * q:.4f}")
