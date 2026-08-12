"""
bb_code.py
==========
Construction of a Bivariate Bicycle (BB) code from (ell, m, A, B), following
Sec. 2.3.1 of the outline:

    rho: x^i y^j -> circ_ell(x^i) tensor circ_m(y^j)
    HX = [A | B],   HZ = [B^T | A^T]
    n = 2*ell*m,    k = n - rank(HX) - rank(HZ)   (over F2)

Also provides brute-force minimum-distance search (feasible only for small n)
and GF(2) rank / row-space-equality routines used throughout the automorphism
detection code.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np

from bb_ring import Ring


# ============================================================================
#  GF(2) linear algebra
# ============================================================================

def gf2_rref_rank(M: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2) via Gaussian elimination (XOR pivoting)."""
    M = (M.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            M[[rank, pivot]] = M[[pivot, rank]]
        pivot_row = M[rank].copy()
        mask = M[:, col].astype(bool)
        mask[rank] = False
        M[mask] ^= pivot_row
        rank += 1
        if rank == rows:
            break
    return rank


def same_rowspace(M1: np.ndarray, M2: np.ndarray) -> bool:
    """True iff the F2 row spaces of M1 and M2 (same #columns) coincide."""
    if M1.shape[1] != M2.shape[1]:
        return False
    r1 = gf2_rref_rank(M1)
    r2 = gf2_rref_rank(M2)
    if r1 != r2:
        return False
    stacked = np.vstack([M1, M2])
    return gf2_rref_rank(stacked) == r1


def gf2_nullspace_basis(M: np.ndarray) -> np.ndarray:
    """Basis (rows) of the right nullspace of M over GF(2)."""
    M = (M.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    A = M.copy()
    pivots = []
    r = 0
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if A[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[f] = 1
        for ridx, pc in enumerate(pivots):
            if A[ridx, f]:
                vec[pc] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), dtype=np.uint8)


# ============================================================================
#  BB code
# ============================================================================

@dataclass
class BBCode:
    ring: Ring
    A: np.ndarray
    B: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.ell = self.ring.ell
        self.m = self.ring.m
        self.n = 2 * self.ell * self.m
        self._rhoA = self._circulant_lift(self.A)
        self._rhoB = self._circulant_lift(self.B)
        self.HX = np.hstack([self._rhoA, self._rhoB])
        self.HZ = np.hstack([self._rhoB.T, self._rhoA.T])

    def _circulant_lift(self, f: np.ndarray) -> np.ndarray:
        """rho(f): the (ell*m)x(ell*m) circulant matrix of multiplication by f."""
        ell, m = self.ell, self.m
        n = ell * m
        terms = self.ring.to_terms(f)
        Mat = np.zeros((n, n), dtype=np.uint8)
        for r in range(n):
            a, b = divmod(r, m)
            for (di, dj) in terms:
                c = ((a + di) % ell) * m + ((b + dj) % m)
                Mat[r, c] ^= 1
        return Mat

    # ------------------------------------------------------------- checks
    def css_commutes(self) -> bool:
        prod = (self.HX.astype(np.uint32) @ self.HZ.T.astype(np.uint32)) % 2
        return bool(np.all(prod == 0))

    def rank_HX(self) -> int:
        return gf2_rref_rank(self.HX)

    def rank_HZ(self) -> int:
        return gf2_rref_rank(self.HZ)

    def k(self) -> int:
        return self.n - self.rank_HX() - self.rank_HZ()

    def params_str(self) -> str:
        return f"[[{self.n},{self.k()},d]]  (ell={self.ell}, m={self.m})"

    # --------------------------------------------------- full stabilizer S
    def stabilizer_matrix(self) -> np.ndarray:
        """
        The 2*ell*m x 4*ell*m matrix representing the module
        M = R.gX + R.gZ  <=  N = R^4,
        gX = (A,B,0,0), gZ = (0,0,bar(B),bar(A)),
        as block_diag(HX, HZ). Column blocks are ordered [A|B|barB_slot|barA_slot]
        i.e. slot1=A, slot2=B, slot3=barB, slot4=barA (matching Sec 3 / Sec 4
        of the outline: gX=(f1,f2,0,0), gZ=(0,0, barf2, barf1)).
        """
        ell_m = self.ell * self.m
        Z = np.zeros((ell_m, ell_m), dtype=np.uint8)
        top = np.hstack([self._rhoA, self._rhoB, Z, Z])
        rho_barB = self._circulant_lift(self.ring.bar(self.B))
        rho_barA = self._circulant_lift(self.ring.bar(self.A))
        bottom = np.hstack([Z, Z, rho_barB, rho_barA])
        return np.vstack([top, bottom])

    # --------------------------------------------------------- brute force d
    def min_distance_bruteforce(self, max_weight_search: int = None):
        """
        Exact minimum distance via brute-force enumeration of the quotient
        ker(HX)/rowspace(HZ) (X-type logical weight) — only feasible for
        small n (n <~ 24-30). Returns None if infeasible / not attempted.
        """
        n = self.n
        if n > 26:
            return None
        kerHX = gf2_nullspace_basis(self.HX)
        rsHZ = self.HZ
        dimK = kerHX.shape[0]
        if dimK == 0:
            return None
        best = n + 1
        # enumerate all F2 combinations of kernel basis vectors (2^dimK),
        # skip those that lie in rowspace(HZ) (pure stabilizers => not logical)
        rank_HZ = gf2_rref_rank(rsHZ)
        for bits in range(1, 1 << dimK):
            vec = np.zeros(n, dtype=np.uint8)
            for b in range(dimK):
                if bits & (1 << b):
                    vec ^= kerHX[b]
            w = int(vec.sum())
            if w == 0 or w >= best:
                continue
            # is vec in rowspace(HZ)? -> pure stabilizer, not a logical rep
            test = np.vstack([rsHZ, vec[None, :]])
            if gf2_rref_rank(test) == rank_HZ:
                continue
            best = w
        return best if best <= n else None
