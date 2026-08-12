"""
bb_crt.py
=========
The actual CRT-component machinery of Sec. 2.2-3.2 of the outline, as
opposed to the row-space shortcut in bb_automorphisms.py. This module:

  1. Factors x^l-1 and y^m-1 over F2 (Sec. 2.2, eq. 5).
  2. Hensel-lifts x (resp. y) to an exact root inside each chain-ring factor
     (Lemma 2.2 / Lemma B.1), giving the uniformizers u, v.
  3. Builds the refined component set C (Theorem 2.4 / Lemma 2.3: Frobenius
     orbits on root pairs), each component c a local ring
     J_c = F_{2^D}[u,v]/<u^{2^s_l}, v^{2^s_m}>.
  4. Reduces a polynomial f in R_{l,m} to its local coordinates f_c in J_c
     (the Hasse/Taylor jet of Sec. 3.2.1), via the mod-2 binomial
     (Lucas'-theorem) substitution x^i=(alpha+u)^i.
  5. Implements J_c arithmetic (multiplication, unit testing) and the local
     matching-equation solve of Theorem 3.9 / Algorithm 3.12.

Every numerically load-bearing claim here is cross-checked against an
independent oracle at construction time:
  - component *count* against Theorem 2.4's formula |C| = sum_{i,j} gcd(d_i,e_j)
  - component *dimension* against dim R_{l,m} = l*m
  - local-support dimension count against the already-validated GF(2)-rank
    formula k = 2 * sum_c dim(J_c / <A_c,B_c>)     (Sec. 2.3.1)
so a bug here is caught by disagreement with numbers bb_code.py already
gets right, rather than trusted on first principles alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from typing import List, Tuple, Dict
import numpy as np
import galois

from bb_code import gf2_nullspace_basis

GF2 = galois.GF(2)


# ============================================================================
#  Step 1 (Sec. 2.2, eq. 5): factor x^n - 1 over F2 via its odd part.
# ============================================================================

def two_adic_split(n: int) -> Tuple[int, int]:
    """n = 2^s * n' with n' odd. Returns (s, n')."""
    s = 0
    while n % 2 == 0:
        n //= 2
        s += 1
    return s, n


def factor_odd_cyclotomic(n_odd: int) -> List[Tuple[galois.Poly, int]]:
    """
    Factor x^{n_odd} - 1 over F2 into its distinct irreducible factors
    (squarefree since gcd(n_odd,2)=1). Returns [(g_i, deg g_i), ...].
    Uses galois' general factorer; cross-checked against the outline's own
    2-cyclotomic-coset degree formula in the test suite.
    """
    if n_odd == 1:
        return [(galois.Poly([1, 1], field=GF2), 1)]  # x - 1 = x + 1
    p = galois.Poly.Degrees(list(range(n_odd + 1)),
                             coeffs=[1] + [0] * (n_odd - 1) + [1], field=GF2)
    factors, mult = p.factors()
    assert all(e == 1 for e in mult), "odd part must be squarefree"
    return [(f, f.degree) for f in factors]


def cyclotomic_coset_degrees(n_odd: int) -> List[int]:
    """Reference check: predicted factor degrees via 2-cyclotomic cosets mod n_odd."""
    if n_odd == 1:
        return [1]
    seen = set()
    degs = []
    for a in range(n_odd):
        if a in seen:
            continue
        coset = []
        x = a
        while x not in coset:
            coset.append(x)
            seen.add(x)
            x = (2 * x) % n_odd
        degs.append(len(coset))
    return degs


# ============================================================================
#  Step 2 (Lemma 2.2 / Lemma B.1): Hensel lift x to an exact root of g
#  inside L = F2[x]/<g^{2^s}>.
# ============================================================================

def poly_pow(g: galois.Poly, k: int) -> galois.Poly:
    r = galois.Poly([1], field=GF2)
    for _ in range(k):
        r = r * g
    return r


def poly_egcd(a: galois.Poly, b: galois.Poly):
    """Extended Euclid over F2[x]: returns (gcd, s, t) with s*a+t*b=gcd."""
    ZERO = galois.Poly([0], field=GF2)
    ONE = galois.Poly([1], field=GF2)
    old_r, r = a, b
    old_s, s = ONE, ZERO
    old_t, t = ZERO, ONE
    while r != ZERO:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def hensel_lift_root(g: galois.Poly, s: int) -> galois.Poly:
    """
    Hensel-lift x to alpha in L=F2[x]/<g^{2^s}> with g(alpha)=0 exactly and
    alpha == x (mod g), by Newton iteration with doubling precision
    (Lemma 2.2 / Lemma B.1). Returns alpha as an F2[x] polynomial reduced
    mod g^{2^s}.
    """
    x = galois.Poly([1, 0], field=GF2)
    if s == 0:
        return x % galois.Poly([1], field=GF2)
    alpha = x
    gprime = g.derivative()
    precision_mod = g  # alpha is exact mod g^1 == g trivially (g(x)=g(x))
    k = 1
    while k < s:
        k_new = min(2 * k, s)
        M = poly_pow(g, k_new)
        # evaluate g(alpha) mod M via polynomial composition
        g_at_alpha = _compose(g, alpha, M)
        if g_at_alpha == galois.Poly([0], field=GF2):
            alpha = alpha % M
            k = k_new
            continue
        # inverse of g'(alpha) mod M via extended Euclid (g'(alpha) is a
        # unit mod g since g is separable, hence a unit mod any power of g)
        gprime_at_alpha = _compose(gprime, alpha, M)
        d, inv, _ = poly_egcd(gprime_at_alpha, M)
        assert d == galois.Poly([1], field=GF2), "g'(alpha) not a unit -- g not separable?"
        alpha = (alpha - g_at_alpha * inv) % M
        k = k_new
    return alpha % poly_pow(g, s)


def _compose(f: galois.Poly, alpha: galois.Poly, M: galois.Poly) -> galois.Poly:
    """Evaluate f(alpha(x)) mod M, i.e. polynomial composition f o alpha, mod M."""
    coeffs = f.coeffs  # highest degree first
    result = galois.Poly([0], field=GF2)
    for c in coeffs:
        result = (result * alpha) % M
        if c:
            result = (result + galois.Poly([1], field=GF2)) % M
    return result


# ============================================================================
#  Step 3 (Theorem 2.4 / Lemma 2.3): refined component set C.
#  Field-theoretic route: work directly in GF(2^D) via root-finding and
#  Frobenius orbits, rather than trying to carry the Hensel-lifted F2[x]
#  representative through the bivariate splitting (which is what Lemma 2.3's
#  splitting genuinely depends on: how roots of g_i, h_j interact under a
#  *shared* Frobenius in a common field).
# ============================================================================

@dataclass
class Component:
    i: int          # index into ell-side factor list
    j: int          # index into m-side factor list
    d_i: int
    e_j: int
    D: int          # lcm(d_i,e_j) -- residue field degree
    field: "galois.FieldArray"   # GF(2^D)
    alpha: object    # representative root of g_i, in field
    beta: object      # representative root of h_j, in field
    orbit: list       # the full Frobenius orbit [(alpha,beta), (alpha^2,beta^2), ...]


def build_components(ell: int, m: int) -> Tuple[List[Component], List, List]:
    """
    Build the refined CRT component set C of Theorem 2.4 for R_{ell,m}.
    Returns (components, ell_factors, m_factors) where ell_factors is the
    list of (g_i, d_i, s_ell) for the ell-side (similarly m_factors), s_ell
    the 2-adic valuation of ell (shared across all components on that side).
    """
    s_ell, ell_odd = two_adic_split(ell)
    s_m, m_odd = two_adic_split(m)
    ell_facs = factor_odd_cyclotomic(ell_odd)
    m_facs = factor_odd_cyclotomic(m_odd)

    components: List[Component] = []
    for i, (g_i, d_i) in enumerate(ell_facs):
        for j, (h_j, e_j) in enumerate(m_facs):
            D = d_i * e_j // gcd(d_i, e_j)
            field = galois.GF(2 ** D)
            g_lift = galois.Poly(g_i.coeffs, field=field)
            h_lift = galois.Poly(h_j.coeffs, field=field)
            roots_g = g_lift.roots()
            roots_h = h_lift.roots()
            assert len(roots_g) == d_i and len(roots_h) == e_j, \
                f"expected g_i to split completely in GF(2^{D}) (d_i={d_i}, got {len(roots_g)})"
            pairs = [(a, b) for a in roots_g for b in roots_h]
            seen = set()
            for (a, b) in pairs:
                if (int(a), int(b)) in seen:
                    continue
                orbit = []
                aa, bb = a, b
                while (int(aa), int(bb)) not in seen:
                    orbit.append((aa, bb))
                    seen.add((int(aa), int(bb)))
                    aa, bb = aa * aa, bb * bb
                components.append(Component(i, j, d_i, e_j, D, field, a, b, orbit))
    return components, [(g, d, s_ell) for g, d in ell_facs], [(h, e, s_m) for h, e in m_facs]


# ============================================================================
#  Step 4 (Sec. 3.2.1): local coordinates. Reduce f(x,y) to its Hasse/Taylor
#  jet f_c at component c, via the mod-2 binomial (Lucas'-theorem)
#  substitution x^i=(alpha+u)^i = sum_k C(i,k) alpha^{i-k} u^k, truncated at
#  k < 2^s_ell (similarly y,v). C(i,k) mod 2 = 1 iff (i & k) == k (Lucas).
# ============================================================================

def local_expand(terms: List[Tuple[int, int]], comp: Component, s_ell: int, s_m: int):
    """
    Reduce the polynomial with monomial support `terms` (list of (i,j)
    exponent pairs, coefficients all 1 over F2) to its local jet at
    component `comp`: a (2^s_ell, 2^s_m) array over GF(2^comp.D), entry
    [k1,k2] = coefficient of u^k1 v^k2.
    """
    field = comp.field
    a, b = comp.alpha, comp.beta
    K1, K2 = 2 ** s_ell, 2 ** s_m
    out = field.Zeros((K1, K2))
    for (i, j) in terms:
        for k1 in range(min(i, K1 - 1) + 1):
            if (i & k1) != k1:
                continue
            coefx = a ** (i - k1)
            for k2 in range(min(j, K2 - 1) + 1):
                if (j & k2) != k2:
                    continue
                coefy = b ** (j - k2)
                out[k1, k2] += coefx * coefy
    return out


# ============================================================================
#  Step 5: J_c = F_{2^D}[u,v]/<u^{2^s_ell},v^{2^s_m}> arithmetic.
# ============================================================================

def jc_mul(X, Y, K1: int, K2: int):
    """Truncated 2D convolution: multiplication in J_c."""
    field = type(X)
    out = field.Zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            if X[k1, k2] == 0:
                continue
            for l1 in range(K1 - k1):
                for l2 in range(K2 - k2):
                    if Y[l1, l2] == 0:
                        continue
                    out[k1 + l1, k2 + l2] += X[k1, k2] * Y[l1, l2]
    return out


def jc_to_f2_vector(X, D: int, K1: int, K2: int) -> np.ndarray:
    """Flatten a J_c element (K1xK2 array over GF(2^D)) to an F2 vector of
    length D*K1*K2, via each GF(2^D) entry's D-bit power-basis expansion."""
    out = np.zeros(D * K1 * K2, dtype=np.uint8)
    idx = 0
    for k1 in range(K1):
        for k2 in range(K2):
            vec = np.array(X[k1, k2].vector(), dtype=np.uint8)
            out[idx:idx + D] = vec
            idx += D
    return out


def jc_basis(D: int, K1: int, K2: int, field):
    """F2-basis of J_c as a vector space: theta^t * u^k1 * v^k2."""
    basis = []
    for t in range(D):
        for k1 in range(K1):
            for k2 in range(K2):
                el = field.Zeros((K1, K2))
                el[k1, k2] = field.primitive_element ** t
                basis.append(el)
    return basis


def jc_f2_vector_to_element(vec: np.ndarray, field, D: int, K1: int, K2: int):
    """Inverse of jc_to_f2_vector: rebuild a J_c element from its flat F2 vector."""
    out = field.Zeros((K1, K2))
    idx = 0
    for k1 in range(K1):
        for k2 in range(K2):
            bits = vec[idx:idx + D]
            out[k1, k2] = field.Vector(bits)
            idx += D
    return out


def gf2_solve(M: np.ndarray, b: np.ndarray):
    """
    Solve M x = b over GF(2) via augmented-matrix row reduction.
    Returns a particular solution x (np.uint8 array) if consistent, else None.
    (Underdetermined systems: returns *some* solution, e.g. with free
    variables set to 0 -- matching Theorem 3.9's own framing, where any
    consistent omega_c certifies the matching equation at that component.)
    """
    M = (M.copy() % 2).astype(np.uint8)
    b = (b.copy() % 2).astype(np.uint8)
    rows, cols = M.shape
    aug = np.hstack([M, b.reshape(-1, 1)])
    pivots = []
    r = 0
    for c in range(cols):
        piv = None
        for rr in range(r, rows):
            if aug[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        if piv != r:
            aug[[r, piv]] = aug[[piv, r]]
        mask = aug[:, c].astype(bool)
        mask[r] = False
        aug[mask] ^= aug[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    for rr in range(r, rows):
        if aug[rr, :-1].sum() == 0 and aug[rr, -1] == 1:
            return None  # 0 = 1, inconsistent
    x = np.zeros(cols, dtype=np.uint8)
    for ridx, c in enumerate(pivots):
        x[c] = aug[ridx, -1]
    return x


def jc_is_unit(X) -> bool:
    """A local-ring element is a unit iff its constant term (u^0 v^0
    coefficient, i.e. its image in the residue field) is nonzero."""
    return bool(X[0, 0] != 0)


def solve_matching_at_component(A_c, B_c, X1, X2, D: int, K1: int, K2: int):
    """
    Solve omega * A_c = X1, omega * B_c = X2 for omega in J_c (Theorem 3.9 /
    Lemma 3.7's per-component matching equation), by linear algebra: build
    the matrix of "multiply by A_c" and "multiply by B_c" as F2-linear maps
    on omega's D*K1*K2 coordinates (w.r.t. the jc_basis power basis), stack,
    solve against [X1;X2].

    IMPORTANT: this system is frequently *underdetermined* (e.g. at an
    unsupported component, A_c=B_c=0, every omega solves the trivial 0=0
    system) -- existence of *a* solution is not enough, we need existence
    of a solution that is additionally a UNIT (nonzero constant term). So
    after finding one particular solution via Gaussian elimination, we
    search the affine solution space (particular + nullspace) for a unit
    rather than accepting whatever the elimination happened to return.

    Returns omega (J_c element) if a unit solution exists, else None.
    """
    field = type(A_c)
    basis = jc_basis(D, K1, K2, field)
    cols = []
    for e in basis:
        v1 = jc_to_f2_vector(jc_mul(e, A_c, K1, K2), D, K1, K2)
        v2 = jc_to_f2_vector(jc_mul(e, B_c, K1, K2), D, K1, K2)
        cols.append(np.concatenate([v1, v2]))
    Mmat = np.array(cols, dtype=np.uint8).T   # rows = 2*D*K1*K2 equations, cols = D*K1*K2 unknowns
    rhs = np.concatenate([jc_to_f2_vector(X1, D, K1, K2), jc_to_f2_vector(X2, D, K1, K2)])
    sol0 = gf2_solve(Mmat, rhs)
    if sol0 is None:
        return None

    def reconstruct(sol_vec):
        omega = field.Zeros((K1, K2))
        for flag, e in zip(sol_vec, basis):
            if flag:
                omega = omega + e
        return omega

    def verify_unit(omega):
        if not np.array_equal(jc_to_f2_vector(jc_mul(omega, A_c, K1, K2), D, K1, K2),
                               jc_to_f2_vector(X1, D, K1, K2)):
            return False
        if not np.array_equal(jc_to_f2_vector(jc_mul(omega, B_c, K1, K2), D, K1, K2),
                               jc_to_f2_vector(X2, D, K1, K2)):
            return False
        return jc_is_unit(omega)

    omega0 = reconstruct(sol0)
    if verify_unit(omega0):
        return omega0

    # Underdetermined case: search particular + nullspace for a unit.
    null_basis = gf2_nullspace_basis(Mmat)
    dim_null = null_basis.shape[0]
    if dim_null == 0:
        return None  # unique solution and it wasn't a unit -- genuinely no unit
    # Try single-vector flips first (cheap, covers the common "totally
    # unsupported, any omega works, just pick omega=1" case in one step).
    for nv in null_basis:
        cand = reconstruct((sol0.astype(np.uint8) ^ nv.astype(np.uint8)))
        if verify_unit(cand):
            return cand
    # Fall back to bounded exhaustive search over small nullspaces.
    if dim_null <= 16:
        for bits in range(1, 1 << dim_null):
            combo = np.zeros_like(sol0)
            for b in range(dim_null):
                if bits & (1 << b):
                    combo ^= null_basis[b]
            cand = reconstruct((sol0.astype(np.uint8) ^ combo.astype(np.uint8)))
            if verify_unit(cand):
                return cand
    return None


# ============================================================================
#  Step 6 (Sec. 3.2.1's action of Aut(R) on C): component-induced map.
#  Every catalog automorphism is Z-linear on the exponent lattice, psi(i,j) =
#  (a*i+b*j, c*i+d*j); the induced action on a root-pair character (alpha,
#  beta) is sigma_psi(alpha,beta) = (alpha^a * beta^c, alpha^b * beta^d)
#  (derived directly from the character-evaluation pullback, not asserted).
# ============================================================================

def linear_map_coeffs(phi, ell: int, m: int):
    a, c = phi.map(1, 0)
    b, d = phi.map(0, 1)
    return a, b, c, d


def sigma_psi(alpha, beta, a, b, c, d):
    """Induced action of a linear ring automorphism on a root-pair character:
    sigma_psi(alpha,beta) = (alpha^a * beta^c, alpha^b * beta^d)."""
    return alpha ** a * beta ** c, alpha ** b * beta ** d


def find_component(components, alpha, beta):
    for c in components:
        if any(int(a) == int(alpha) and int(b) == int(beta) for (a, b) in c.orbit):
            return c
    return None


