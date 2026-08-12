"""
test_crt_kcheck.py
====================
Cross-check: Sec 2.3.1 states
    k = 2 * sum_{c in C} dim(J_c / <f1,c, f2,c>)
Compute the RHS purely via the new CRT machinery (bb_crt.py: factorization,
Hensel lifting, component construction, local expansion, local-ideal
dimension via GF(2)-linear algebra on J_c's F2-basis) and compare against
the LHS as already computed (independently, via GF(2) rank of H_X,H_Z) by
bb_code.py -- the tool that's been validated against four published codes.
Agreement here validates everything in bb_crt.py *except* the automorphism
matching-equation solve (component construction, Hensel lifting, local
expansion, and local-ideal linear algebra all participate in computing k).
"""
import numpy as np
from bb_crt import build_components, local_expand, jc_mul, jc_to_f2_vector, jc_basis
from bb_ring import Ring
from bb_code import BBCode, gf2_rref_rank


def local_ideal_dim(A_c, B_c, D, K1, K2):
    """dim_F2 <A_c,B_c> as an ideal of J_c, via GF(2)-linear span."""
    basis = jc_basis(D, K1, K2, type(A_c))
    rows = []
    for e in basis:
        rows.append(jc_to_f2_vector(jc_mul(A_c, e, K1, K2), D, K1, K2))
        rows.append(jc_to_f2_vector(jc_mul(B_c, e, K1, K2), D, K1, K2))
    M = np.array(rows, dtype=np.uint8)
    return gf2_rref_rank(M)


def k_via_crt(ell, m, A_terms, B_terms):
    from bb_crt import two_adic_split
    s_ell, _ = two_adic_split(ell)
    s_m, _ = two_adic_split(m)
    K1, K2 = 2 ** s_ell, 2 ** s_m
    comps, _, _ = build_components(ell, m)
    total = 0
    for c in comps:
        A_c = local_expand(A_terms, c, s_ell, s_m)
        B_c = local_expand(B_terms, c, s_ell, s_m)
        dim_Jc = c.D * K1 * K2
        dim_ideal = local_ideal_dim(A_c, B_c, c.D, K1, K2)
        total += (dim_Jc - dim_ideal)
    return 2 * total


def run():
    cases = [
        ("[[18,4,4]]", 3, 3, [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0), (2, 0)]),
        ("[[16,4,4]]", 2, 4, [(0, 0), (0, 1), (0, 3), (1, 3)], [(0, 0), (0, 1), (0, 3), (1, 1)]),
        ("[[72,12,6]] BB6", 6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
        ("[[144,12,12]] gross", 12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    ]
    for name, ell, m, A_terms, B_terms in cases:
        ring = Ring(ell, m)
        A = ring.from_terms(A_terms)
        B = ring.from_terms(B_terms)
        code = BBCode(ring, A, B)
        k_ground_truth = code.k()
        k_crt = k_via_crt(ell, m, A_terms, B_terms)
        ok = (k_ground_truth == k_crt)
        print(f"{name:24s} (ell={ell},m={m}): k_GF2rank={k_ground_truth}  k_CRT={k_crt}  MATCH={ok}")
        assert ok, f"MISMATCH for {name}"
    print("\nAll k cross-checks passed: CRT component/local-expansion pipeline verified.")


if __name__ == "__main__":
    run()
