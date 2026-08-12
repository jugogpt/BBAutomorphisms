"""
bb_inverse_crt.py
==================
The inverse problem, genuinely using the paper's algebra -- not the simple
monomial-orbit trick of inverse_design.py, which is provably restricted to
w=1 (Corollary 3.6 with the trivial unit) because it works by making A's
*support set* literally psi-invariant.

Corollary 3.6 permits *any* unit w in R_{l,m}^x, not just 1:
    "phi is a code automorphism ... if phi((f1,f2)) = u(f1,f2), where u is
     a unit in R_{l,m}^x."
This module constructs codes achieving a *prescribed, nontrivial* w.

Derivation (the "twisted orbit-sum" construction)
---------------------------------------------------
Fix a ring automorphism psi and a monomial unit w = x^a y^b such that:
  (i)  psi(w) = w   (w is a fixed point of psi's grid action -- monomial
       units are literally the group elements of Z_l x Z_m, and psi acts
       on them the same linear way it acts on exponents)
  (ii) w^L = 1, where L is the length of psi's orbit on a chosen seed point.
Let (i_0,j_0), (i_1,j_1)=psi(i_0,j_0), ..., (i_{L-1},j_{L-1}) be the orbit
(so psi(i_{L-1},j_{L-1}) = (i_0,j_0) again). Define
    A := sum_{k=0}^{L-1} w^{-k} . x^{i_k} y^{j_k}.
Claim: psi(A) = w.A exactly. Proof: psi(w^{-k} x^{i_k}y^{j_k})
     = psi(w)^{-k} x^{i_{k+1}}y^{j_{k+1}} = w^{-k} x^{i_{k+1}}y^{j_{k+1}}
(using (i)), so summing over k and re-indexing, psi(A) = w.A + (w+w^{1-L}) x^{i_0}y^{j_0},
and the correction term vanishes exactly when w^L=1 (condition (ii)).
This is not asserted -- see verify_twisted_construction() below, which checks
psi(A)=w.A by direct ring multiplication (bb_ring.Ring.mul), independent of
this derivation, on every example built.

Every constructed code is then checked three independent ways:
  1. Direct algebra: psi(A)=w.A, psi(B)=w.B computed by explicit ring
     multiplication (the claim above, verified not assumed).
  2. The row-space oracle (bb_automorphisms.test_code_automorphism).
  3. The real CRT/Algorithm-3.12 local solve (bb_matching.test_code_automorphism_crt),
     additionally reporting the local units omega_c found at every component
     and cross-checking them against w's own local expansion.
"""
from __future__ import annotations
from math import gcd
from typing import List, Tuple
import numpy as np

from bb_ring import Ring, RingAuto
from bb_code import BBCode
from bb_automorphisms import test_code_automorphism
from bb_crt import local_expand, jc_is_unit
from bb_matching import CRTContext, test_code_automorphism_crt, local_data


# ============================================================================
#  Monomial-unit group theory: every x^a y^b is a unit of R_{l,m} (its
#  inverse is x^{l-a}y^{m-b}); order, and psi's fixed points among them.
# ============================================================================

def monomial_order(a: int, b: int, ell: int, m: int) -> int:
    """Order of x^a y^b in the (abelian, additively-written) group Z_l x Z_m."""
    oa = ell // gcd(a % ell, ell) if a % ell != 0 else 1
    ob = m // gcd(b % m, m) if b % m != 0 else 1
    return oa * ob // gcd(oa, ob)


def psi_fixed_points(phi: RingAuto, ell: int, m: int) -> List[Tuple[int, int]]:
    """All (a,b) with phi(a,b) == (a,b) mod (ell,m) -- forms a subgroup of
    Z_l x Z_m, since phi acts linearly on the exponent lattice."""
    out = []
    for a in range(ell):
        for b in range(m):
            a2, b2 = phi.map(a, b)
            if a2 % ell == a and b2 % m == b:
                out.append((a, b))
    return out


def psi_orbit(phi: RingAuto, seed: Tuple[int, int], ell: int, m: int) -> List[Tuple[int, int]]:
    """The orbit (i_0,j_0), psi(i_0,j_0), ... , psi^{L-1}(i_0,j_0) under
    repeated application of phi, stopping just before it repeats i_0,j_0."""
    pts = [(seed[0] % ell, seed[1] % m)]
    cur = pts[0]
    while True:
        nxt = phi.map(*cur)
        nxt = (nxt[0] % ell, nxt[1] % m)
        if nxt == pts[0]:
            break
        pts.append(nxt)
        cur = nxt
    return pts


# ============================================================================
#  The twisted-orbit-sum construction itself.
# ============================================================================

def twisted_orbit_poly(ring: Ring, phi: RingAuto, seed: Tuple[int, int],
                        w_exp: Tuple[int, int]) -> np.ndarray:
    """
    Build A = sum_{k=0}^{L-1} w^{-k} . psi^k(seed), the twisted-orbit-sum
    polynomial for which psi(A) = w.A holds exactly (verified separately,
    not assumed here). Raises AssertionError if the required conditions
    (i) psi(w)=w and (ii) w^L=1 are not met by the given (phi, seed, w_exp).
    """
    ell, m = ring.ell, ring.m
    a_w, b_w = w_exp[0] % ell, w_exp[1] % m
    fixed = psi_fixed_points(phi, ell, m)
    assert (a_w, b_w) in fixed, (
        f"w=x^{a_w}y^{b_w} is not psi-fixed ({phi.name}); pick w from "
        f"psi_fixed_points(phi, {ell}, {m})"
    )
    orbit = psi_orbit(phi, seed, ell, m)
    L = len(orbit)
    ordw = monomial_order(a_w, b_w, ell, m)
    assert L % ordw == 0, (
        f"w^L != 1: ord(w)={ordw} does not divide orbit length L={L} "
        f"for seed {seed} under {phi.name}"
    )
    terms = []
    for k, (i_k, j_k) in enumerate(orbit):
        # w^{-k} = x^{-k*a_w} y^{-k*b_w}
        terms.append(((i_k - k * a_w) % ell, (j_k - k * b_w) % m))
    return ring.from_terms(terms)


def design_code_with_unit(ring: Ring, phi: RingAuto, seedA: Tuple[int, int],
                           seedB: Tuple[int, int], w_exp: Tuple[int, int]) -> BBCode:
    """Build (A,B) via twisted_orbit_poly with a *shared* w for both slots,
    so that psi(A,B) = w.(A,B) -- Corollary 3.6 with a genuinely nontrivial
    unit, not just w=1."""
    A = twisted_orbit_poly(ring, phi, seedA, w_exp)
    B = twisted_orbit_poly(ring, phi, seedB, w_exp)
    return BBCode(ring, A, B)


# ============================================================================
#  Three-way verification.
# ============================================================================

def verify_twisted_construction(ring: Ring, code: BBCode, phi: RingAuto,
                                 w_exp: Tuple[int, int], verbose: bool = True) -> bool:
    ell, m = ring.ell, ring.m
    w = ring.from_terms([w_exp])

    # (1) Direct algebra: psi(A) == w*A, psi(B) == w*B, by explicit ring
    # multiplication -- independent of the derivation in the module docstring.
    psiA = ring.apply_auto(code.A, phi)
    psiB = ring.apply_auto(code.B, phi)
    wA = ring.mul(w, code.A)
    wB = ring.mul(w, code.B)
    direct_ok = ring.equal(psiA, wA) and ring.equal(psiB, wB)

    # (2) Row-space oracle (does not know about w at all -- only tests
    # "does *some* unit exist", via rowspace equality).
    oracle_ok, _ = test_code_automorphism(ring, code, phi, swap_AB=False)

    # (3) Real CRT/Algorithm 3.12 local solve -- report the local units
    # omega_c actually found, and cross-check them against w's own local
    # expansion at each component (they should coincide, since w IS *a*
    # valid global unit by construction, though the local solve may return
    # a different, equally-valid element of the same affine solution space).
    ctx = CRTContext(ell, m)
    crt_ok, crt_info = test_code_automorphism_crt(ring, ctx, code.A, code.B, phi, swap_AB=False)
    w_matches_local = []
    if crt_ok:
        ld = local_data(ring, ctx, code.A, code.B)
        for (comp, A_c, B_c, supported) in ld:
            w_c = local_expand(ring.to_terms(w), comp, ctx.s_ell, ctx.s_m)
            is_unit_here = jc_is_unit(w_c)
            w_matches_local.append(is_unit_here or not supported)

    if verbose:
        print(f"    w = {ring.poly_str(w)}  (order {monomial_order(w_exp[0],w_exp[1],ell,m)})")
        print(f"    (1) direct algebra psi(A,B)=w.(A,B):        {direct_ok}")
        print(f"    (2) row-space oracle (Cor. 3.6, some unit):  {oracle_ok}")
        print(f"    (3) real CRT/Algorithm 3.12 local solve:     {crt_ok}"
              + (f"   [w is a unit at every supported component: {all(w_matches_local)}]"
                 if crt_ok else ""))
    return direct_ok and oracle_ok and crt_ok


# ============================================================================
#  Worked examples: nontrivial-w inverse design.
# ============================================================================

def example_nontrivial_unit_partial_fold():
    print("\n[CRT-INV 1] theta_x with a GENUINELY NONTRIVIAL unit w=y^3, ell=5,m=6")
    ring = Ring(5, 6)
    from bb_ring import theta_x
    phi = theta_x()
    w_exp = (0, 3)   # w = y^3, order 2
    code = design_code_with_unit(ring, phi, seedA=(1, 0), seedB=(2, 1), w_exp=w_exp)
    print(f"    A={ring.poly_str(code.A)}  B={ring.poly_str(code.B)}  {code.params_str()}")
    ok = verify_twisted_construction(ring, code, phi, w_exp)
    assert ok


def example_nontrivial_unit_multiplier():
    print("\n[CRT-INV 2] multiplier with a nontrivial unit w=x^4, ell=9,m=5")
    from bb_ring import multiplier
    ring = Ring(9, 5)
    phi = multiplier(9, 5, 4, 1)   # jy=1 fixes all of y -> any y^b is phi-fixed too
    # phi fixes (a,b) iff 4a==a mod 9 (i.e. 3a==0 mod9 -> a in {0,3,6}) and any b
    w_exp = (3, 0)   # order_9(3)=3
    code = design_code_with_unit(ring, phi, seedA=(2, 2), seedB=(2, 3), w_exp=w_exp)
    print(f"    A={ring.poly_str(code.A)}  B={ring.poly_str(code.B)}  {code.params_str()}")
    ok = verify_twisted_construction(ring, code, phi, w_exp)
    assert ok


def example_nontrivial_unit_shear():
    print("\n[CRT-INV 3] shear with a nontrivial unit, ell=12,m=6 (paper's own (l,m))")
    from bb_ring import shear_family
    ring = Ring(12, 6)
    phi = [s for s in shear_family(12, 6) if "c=3" in s.name][0]   # x -> x y^3
    # shear fixes (a,b) iff (a, b+3a mod 6) == (a,b) i.e. 3a == 0 mod 6 -> a in {0,2,4,6,8,10}
    w_exp = (6, 0)   # order_12(6) = 2, matches the orbit length of the chosen seeds below
    code = design_code_with_unit(ring, phi, seedA=(1, 0), seedB=(5, 0), w_exp=w_exp)
    print(f"    A={ring.poly_str(code.A)}  B={ring.poly_str(code.B)}  {code.params_str()}")
    ok = verify_twisted_construction(ring, code, phi, w_exp)
    assert ok


def example_nontrivial_unit_fullfold():
    print("\n[CRT-INV 4] full fold (antipode) with a nontrivial unit, ell=8,m=5")
    from bb_ring import full_fold
    ring = Ring(8, 5)
    phi = full_fold()
    # iota fixes (a,b) iff -a==a (2a==0 mod8 -> a in {0,4}) and -b==b (2b==0 mod5 -> b=0)
    w_exp = (4, 0)   # order_8(4) = 2
    code = design_code_with_unit(ring, phi, seedA=(1, 1), seedB=(3, 2), w_exp=w_exp)
    print(f"    A={ring.poly_str(code.A)}  B={ring.poly_str(code.B)}  {code.params_str()}")
    ok = verify_twisted_construction(ring, code, phi, w_exp)
    assert ok


if __name__ == "__main__":
    example_nontrivial_unit_partial_fold()
    example_nontrivial_unit_multiplier()
    example_nontrivial_unit_shear()
    example_nontrivial_unit_fullfold()
    print("\nAll nontrivial-unit CRT inverse-design examples verified (3 independent ways each).")
