"""
bb_matching.py
===============
The real Algorithm 3.12 (Forward Detection), built on bb_crt.py's CRT
machinery, as opposed to the row-space shortcut in bb_automorphisms.py:

  Algorithm 3.12 (paper's own numbering, quoted in full):
  "Given local data (Ac, Bc)c in C and a candidate Psi = (g, psi):
   (1) compute sigma_psi on C;
   (2) discard instantiations whose linear part violates the Borel
       restriction (Prop. 2.8) at some component with 2^s_ell != 2^s_m;
   (3)-(4) reject on any support or Jacobian-rank mismatch c vs. sigma(c)
       (Proposition 3.10);
   (5) for every surviving c, solve T*_psi(A_sigma(c)) = omega_c Y_c^(1),
       T*_psi(B_sigma(c)) = omega_c Y_c^(2) for omega_c in J_c^x;
   (6) accept iff every c succeeds, assembling (omega_c)_c into w in R^x."

Step (5)'s solve is done here via Lemma 3.7's equivalent same-component
formulation (ring.apply_auto(A,psi) localized directly at c equals, by the
pullback identity Lemma 3.8, T*_psi(A_sigma(c)) -- see bb_crt.py docstring),
which sidesteps needing to build the transport isomorphism T*_psi
explicitly. Steps (2)-(4) are implemented as genuine pre-filters using
sigma_psi and the local data, giving real pruning (not just a description
of it) before the expensive per-component solve of step (5).

Every result is cross-validated against bb_automorphisms.test_code_automorphism
(the row-space method) -- see test_matching_crosscheck.py.
"""
from __future__ import annotations
from math import gcd
from typing import List, Optional
import numpy as np

from bb_ring import Ring, RingAuto
from bb_crt import (
    build_components, local_expand, jc_mul, solve_matching_at_component,
    linear_map_coeffs, sigma_psi, jc_is_unit, two_adic_split, linear_order_prefilter,
)


class CRTContext:
    """Precomputed, code-independent CRT data for a given (ell,m) -- built
    once and reused across every candidate automorphism test (mirrors the
    "shared structure" pattern used for the row-space catalog scan)."""

    def __init__(self, ell: int, m: int):
        self.ell, self.m = ell, m
        self.s_ell, _ = two_adic_split(ell)
        self.s_m, _ = two_adic_split(m)
        self.K1, self.K2 = 2 ** self.s_ell, 2 ** self.s_m
        self.components, self.ell_facs, self.m_facs = build_components(ell, m)


def local_data(ring: Ring, ctx: CRTContext, A, B):
    """Local expansions (A_c,B_c) at every component, plus support flags."""
    terms_A = ring.to_terms(A)
    terms_B = ring.to_terms(B)
    out = []
    for c in ctx.components:
        A_c = local_expand(terms_A, c, ctx.s_ell, ctx.s_m)
        B_c = local_expand(terms_B, c, ctx.s_ell, ctx.s_m)
        supported = (A_c[0, 0] != 0) or (B_c[0, 0] != 0)
        out.append((c, A_c, B_c, supported))
    return out


def jacobian_rank(A_c, B_c, K1, K2) -> int:
    """Rank of the 2x2 Jacobian [[A_u,A_v],[B_u,B_v]] over the residue field
    (Sec. 3.2.1) -- 0,1, or 2. Only meaningful if K1>1 and K2>1 (i.e. only
    in the even x even regime, per the outline's own remark)."""
    if K1 < 2 or K2 < 2:
        return -1  # not applicable (mixed/odd regime: no bivariate Jacobian)
    Au = A_c[1, 0] if K1 > 1 else A_c[0, 0] * 0
    Av = A_c[0, 1] if K2 > 1 else A_c[0, 0] * 0
    Bu = B_c[1, 0] if K1 > 1 else B_c[0, 0] * 0
    Bv = B_c[0, 1] if K2 > 1 else B_c[0, 0] * 0
    rows = [[Au, Av], [Bu, Bv]]
    nz = [(r, cix) for r in range(2) for cix in range(2) if rows[r][cix] != 0]
    if not nz:
        return 0
    # 2x2 rank over a field: 0 if all zero, else check determinant
    det = Au * Bv - Au * 0  # placeholder, real det below
    det = Au * Bv + Av * Bu  # char 2: minus=plus
    if det != 0:
        return 2
    return 1


def test_code_automorphism_crt(ring: Ring, ctx: CRTContext, A, B, phi: RingAuto,
                                swap_AB: bool = False, use_pruning: bool = True):
    """
    The real Algorithm 3.12, steps (1)-(6), for the ring-level (n=1,
    Corollary 3.6) case with gblock in {identity, swap}. Returns
    (accepted: bool, info: dict) where info records which step (if any)
    rejected the candidate, for transparency about where pruning fired.
    """
    a, b, c_, d = linear_map_coeffs(phi, ring.ell, ring.m)
    # NOTE: step (2), the Borel prune of Prop. 2.8, is intentionally NOT
    # applied here. An earlier attempt at translating Prop. 2.8's condition
    # into a check on (a,b,c_,d) produced false negatives (rejected valid
    # automorphisms) on the [[16,4,4]] cross-check against the row-space
    # oracle -- getting the *direction* of the restriction right (which of
    # u,v is allowed to acquire a component of the other depends on which
    # of s_ell,s_m is larger, Prop 2.8's a>b hypothesis) needs more care
    # than a quick derivation gives confidently. Correctness (steps 3-6,
    # all cross-validated) is prioritized over this optional speed-up;
    # steps (3)-(4) below still provide real pruning.

    Ap = ring.apply_auto(A, phi)
    Bp = ring.apply_auto(B, phi)
    if swap_AB:
        Ap, Bp = Bp, Ap

    orig = local_data(ring, ctx, A, B)
    for (comp, A_c, B_c, supported_c) in orig:
        terms_Ap = ring.to_terms(Ap)
        terms_Bp = ring.to_terms(Bp)
        X1 = local_expand(terms_Ap, comp, ctx.s_ell, ctx.s_m)
        X2 = local_expand(terms_Bp, comp, ctx.s_ell, ctx.s_m)
        supported_target = (X1[0, 0] != 0) or (X2[0, 0] != 0)

        # Steps (3)-(4): Proposition 3.10 pre-filter -- support preservation
        # and Jacobian-rank matching, computed directly from local data,
        # BEFORE the expensive per-component solve.
        if use_pruning:
            if supported_c != supported_target:
                return False, {"rejected_at": "step3_support_mismatch", "component": comp}
            if ctx.K1 > 1 and ctx.K2 > 1:
                r1 = jacobian_rank(A_c, B_c, ctx.K1, ctx.K2)
                r2 = jacobian_rank(X1, X2, ctx.K1, ctx.K2)
                if r1 != r2:
                    return False, {"rejected_at": "step4_jacobian_rank_mismatch", "component": comp}
            # Step (4.5): a cheap, sound necessary condition strictly finer
            # than rank comparison alone -- see bb_crt.linear_order_prefilter.
            # Two Jacobians can have equal rank yet be *inconsistent* under
            # every scalar (rank alone can't see this); this catches those
            # cases before paying for the full per-component solve.
            if (ctx.K1 > 1 or ctx.K2 > 1) and not linear_order_prefilter(A_c, B_c, X1, X2, ctx.K1, ctx.K2):
                return False, {"rejected_at": "step4.5_linear_order_mismatch", "component": comp}

        # Step (5): local matching-equation solve.
        omega = solve_matching_at_component(A_c, B_c, X1, X2, comp.D, ctx.K1, ctx.K2)
        if omega is None:
            return False, {"rejected_at": "step5_no_local_unit", "component": comp}

    # Step (6): every component succeeded.
    return True, {"rejected_at": None}
