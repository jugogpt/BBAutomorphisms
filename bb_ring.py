"""
bb_ring.py
==========
Ambient ring R_{l,m} = F2[x,y]/(x^l-1, y^m-1) for Bivariate Bicycle (BB) codes,
following "Automorphisms of Bivariate Bicycle Codes via Refined CRT Components
and Local Transfer Data".

A polynomial in R_{l,m} is represented as a dense l x m numpy array over F2
(entry [i,j] = coefficient of x^i y^j).

This module also implements the ring-automorphism catalog of our paper:
    1. Multipliers        psi_(jx,jy): x -> x^jx, y -> y^jy
    2. Partial folds       theta_x, theta_y
    3. Full fold (antipode) iota = theta_x . theta_y
    4. Shears               Hom(Z_l, Z_m) ~= Z_d,  d = gcd(l,m)
    5. Transpose            tau: x <-> y   (only when l == m)

Every automorphism is represented uniformly as a function
    phi: (i, j) -> (i', j')
acting on exponent pairs (taken mod (l,m) by the caller), together with a
human-readable name, so that the detection/inverse-design code can treat the
whole catalog uniformly.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from typing import Callable, List, Tuple
import numpy as np

Term = Tuple[int, int]


class Ring:
    """The ambient ring R_{l,m} = F2[x,y]/(x^l-1, y^m-1)."""

    def __init__(self, ell: int, m: int):
        self.ell = ell
        self.m = m

    # ---------------------------------------------------------------- basics
    def zero(self) -> np.ndarray:
        return np.zeros((self.ell, self.m), dtype=np.uint8)

    def from_terms(self, terms: List[Term]) -> np.ndarray:
        """Build a polynomial (dense array) from a list of (i,j) exponents.
        Repeated terms cancel mod 2 (F2 group algebra)."""
        p = self.zero()
        for (i, j) in terms:
            p[i % self.ell, j % self.m] ^= 1
        return p

    def to_terms(self, A: np.ndarray) -> List[Term]:
        return [tuple(int(x) for x in t) for t in np.argwhere(A)]

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(A, B)

    def mul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Circular (group-algebra) convolution over F2."""
        ell, m = self.ell, self.m
        C = np.zeros((ell, m), dtype=np.uint8)
        idxA = np.argwhere(A)
        idxB = np.argwhere(B)
        for (i, j) in idxA:
            # shift B by (i,j) and XOR into C
            shifted = np.roll(np.roll(B, i, axis=0), j, axis=1)
            C ^= shifted
        return C

    def bar(self, A: np.ndarray) -> np.ndarray:
        """Antipode / bar involution: f(x,y) -> f(x^-1, y^-1)."""
        return np.roll(np.roll(A[::-1, ::-1], 1, axis=0), 1, axis=1)

    def weight(self, A: np.ndarray) -> int:
        return int(A.sum())

    def equal(self, A: np.ndarray, B: np.ndarray) -> bool:
        return bool(np.array_equal(A, B))

    # ------------------------------------------------------------- printing qualities
    def poly_str(self, A: np.ndarray) -> str:
        terms = self.to_terms(A)
        if not terms:
            return "0"
        parts = []
        for (i, j) in sorted(terms):
            if i == 0 and j == 0:
                parts.append("1")
                continue
            s = ""
            if i == 1:
                s += "x"
            elif i > 0:
                s += f"x^{i}"
            if j == 1:
                s += "y"
            elif j > 0:
                s += f"y^{j}"
            parts.append(s)
        return " + ".join(parts)

    # ---------------------------------------------------- applying automorphism for later testing
    def apply_auto(self, A: np.ndarray, phi: "RingAuto") -> np.ndarray:
        ell, m = self.ell, self.m
        B = self.zero()
        for (i, j) in self.to_terms(A):
            i2, j2 = phi.map(i, j)
            B[i2 % ell, j2 % m] ^= 1
        return B

# ======= Ring automorphism catalog 

@dataclass(frozen=True)
class RingAuto:
    """A ring automorphism of R_{l,m} as an exponent-lattice map."""
    name: str
    map: Callable[[int, int], Tuple[int, int]]

    def compose(self, other: "RingAuto") -> "RingAuto":
        """Return self . other  (apply `other` first, then `self`)."""
        return RingAuto(
            name=f"({self.name}) o ({other.name})",
            map=lambda i, j, s=self, o=other: s.map(*o.map(i, j)),
        )


def identity_auto() -> RingAuto:
    return RingAuto("id", lambda i, j: (i, j))


def multiplier(ell: int, m: int, jx: int, jy: int) -> RingAuto:
    """psi_(jx,jy): x -> x^jx, y -> y^jy.  Requires jx in (Z/l)*, jy in (Z/m)*."""
    if gcd(jx, ell) != 1 or gcd(jy, m) != 1:
        raise ValueError("multiplier exponents must be units mod (ell,m)")
    return RingAuto(f"mult({jx},{jy})", lambda i, j: (i * jx, j * jy))


def theta_x() -> RingAuto:
    """Partial fold: x -> x^-1, y -> y."""
    return RingAuto("theta_x", lambda i, j: (-i, j))


def theta_y() -> RingAuto:
    """Partial fold: x -> x, y -> y^-1."""
    return RingAuto("theta_y", lambda i, j: (i, -j))


def full_fold() -> RingAuto:
    """Antipode iota = theta_x . theta_y, the multiplier (-1,-1)."""
    return RingAuto("iota", lambda i, j: (-i, -j))


def transpose_auto() -> RingAuto:
    """tau: x <-> y.  Only a ring automorphism of R_{l,m} when l == m."""
    return RingAuto("tau", lambda i, j: (j, i))


def shear_family(ell: int, m: int) -> List[RingAuto]:
    """
    Shears: Hom(Z_l, Z_m) ~= Z_d, d = gcd(l,m).
    x -> x y^c,  y -> y,   c a multiple of m/d,  c in {0, m/d, ..., (d-1) m/d}.
    On exponents: (i,j) -> (i, j + c*i mod m).
    Returns the d automorphisms (including identity, s=0).
    """
    d = gcd(ell, m)
    out = []
    for s in range(d):
        c = s * (m // d)
        out.append(RingAuto(f"shear_x(c={c})", lambda i, j, c=c: (i, j + c * i)))
    return out


def shear_family_transpose(ell: int, m: int) -> List[RingAuto]:
    """
    The "transpose" variant of the shear family: y -> y x^c', x -> x,
    c' a multiple of l/d.  On exponents: (i,j) -> (i + c'*j, j).
    """
    d = gcd(ell, m)
    out = []
    for s in range(d):
        cp = s * (ell // d)
        out.append(RingAuto(f"shear_y(c={cp})", lambda i, j, cp=cp: (i + cp * j, j)))
    return out


def unit_multipliers(ell: int, m: int) -> List[Tuple[int, int]]:
    units_l = [a for a in range(1, ell + 1) if gcd(a, ell) == 1]
    units_m = [b for b in range(1, m + 1) if gcd(b, m) == 1]
    return [(a, b) for a in units_l for b in units_m]


def full_catalog(ell: int, m: int, include_shears: bool = True) -> List[RingAuto]:
    """
    Build a (finite, reasonably sized) catalog of ring automorphisms to scan:
    all multipliers, composed with {id, theta_x, theta_y, iota}, composed with
    the shear family (if d=gcd(l,m)>1), and with the transpose when l==m.
    """
    cat: List[RingAuto] = []
    folds = [identity_auto(), theta_x(), theta_y(), full_fold()]
    shears = shear_family(ell, m) if include_shears else [identity_auto()]
    for (jx, jy) in unit_multipliers(ell, m):
        mult = multiplier(ell, m, jx, jy)
        for fold in folds:
            for sh in shears:
                cat.append(mult.compose(fold).compose(sh))
    if ell == m:
        tau = transpose_auto()
        extra = []
        for a in cat:
            extra.append(tau.compose(a))
        cat = cat + extra
    return cat


# ============================================================================
#  Twisted-torus reduction utility (Smith normal form for 2x2 lattices)
# ============================================================================

def reduce_twisted_torus(a1: Term, a2: Term):
    """
    Given a "twisted torus" defined by two lattice vectors a1, a2 in Z^2
    (i.e. relations x^a1[0] y^a1[1] = 1, x^a2[0] y^a2[1] = 1), find a
    unimodular change of exponent-coordinates (i,j) -> (i,j).V taking the
    lattice L = span_Z{a1,a2} to a standard rectangular lattice d1 Z x d2 Z,
    i.e. reduce the BB-like code on the twisted torus to the standard
    R_{d1,d2} presentation used throughout this module.

    Returns (ell, m, V) where V is the 2x2 unimodular integer transform such
    that new_exponents = old_exponents @ V, and ell*m = |det([a1;a2])|.
    Implemented via the standard "global smallest nonzero pivot" 2x2 Smith
    normal form algorithm (see in-line comments); verified against random
    lattices in scripts/test_twisted_torus.py.
    """
    M = np.array([a1, a2], dtype=int)
    # V tracks the unimodular column (= exponent-coordinate) transform, so
    # that new_exponents = old_exponents @ V.  Row operations (relabelling
    # which lattice generator is "first") do not affect the exponent-space
    # transform and so never touch V.
    V = np.eye(2, dtype=int)

    # Global-smallest-pivot reduction (standard, provably terminating 2x2
    # Smith-normal-form algorithm): repeatedly bring the smallest nonzero
    # entry to position (0,0) via row/column swaps, then use it to reduce
    # the rest of its row (column op -> updates V) and column (row op ->
    # never touches V) via floor division. Each such reduction step turns
    # a target entry into a remainder strictly smaller in absolute value
    # than the current pivot, so the global minimum nonzero |entry| is
    # non-increasing and strictly decreases infinitely often; this rules
    # out the 2-phase-alternation cycling that a naive scheme can hit on
    # degenerate inputs (e.g. equal-magnitude entries).
    max_iter = 1000
    for _ in range(max_iter):
        if M[0, 1] == 0 and M[1, 0] == 0:
            break
        nz = [(abs(int(M[r, c])), r, c) for r in range(2) for c in range(2) if M[r, c] != 0]
        pivot_abs, pr, pc = min(nz)
        if pr == 1:
            M[[0, 1], :] = M[[1, 0], :]
        if pc == 1:
            M[:, [0, 1]] = M[:, [1, 0]]
            V[:, [0, 1]] = V[:, [1, 0]]
        if M[0, 1] != 0:
            q = M[0, 1] // M[0, 0]
            M[:, 1] -= q * M[:, 0]
            V[:, 1] -= q * V[:, 0]
        if M[1, 0] != 0:
            q = M[1, 0] // M[0, 0]
            M[1, :] -= q * M[0, :]
    else:
        raise RuntimeError("2x2 Smith-normal-form reduction did not converge")

    d1, d2 = abs(int(M[0, 0])), abs(int(M[1, 1]))
    detV = int(round(np.linalg.det(V)))
    assert detV in (1, -1), f"V not unimodular, det={detV}"
    return d1, d2, V


def transform_terms(terms: List[Term], V: np.ndarray) -> List[Term]:
    out = []
    for (i, j) in terms:
        v = np.array([i, j]) @ V
        out.append((int(v[0]), int(v[1])))
    return out
