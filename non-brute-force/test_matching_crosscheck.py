"""
test_matching_crosscheck.py
=============================
Cross-check bb_matching.test_code_automorphism_crt (the real Algorithm 3.12)
against bb_automorphisms.test_code_automorphism (the row-space oracle,
already validated against the four benchmark codes' published parameters
and against direct hand-checks). Disagreement here means a bug in the new
CRT code, since the row-space method is the trusted reference.
"""
import time
from bb_ring import Ring, full_catalog
from bb_code import BBCode
from bb_automorphisms import test_code_automorphism
from bb_matching import CRTContext, test_code_automorphism_crt


def crosscheck(name, ell, m, A_terms, B_terms, max_candidates=None):
    ring = Ring(ell, m)
    A = ring.from_terms(A_terms)
    B = ring.from_terms(B_terms)
    code = BBCode(ring, A, B)
    ctx = CRTContext(ell, m)
    cat = full_catalog(ell, m)
    if max_candidates:
        cat = cat[:max_candidates]

    n_tested = 0
    n_agree = 0
    disagreements = []
    t0 = time.time()
    for phi in cat:
        for swap in (False, True):
            ok_rowspace, _ = test_code_automorphism(ring, code, phi, swap_AB=swap)
            ok_crt, info = test_code_automorphism_crt(ring, ctx, A, B, phi, swap_AB=swap)
            n_tested += 1
            if ok_rowspace == ok_crt:
                n_agree += 1
            else:
                disagreements.append((phi.name, swap, ok_rowspace, ok_crt, info))
    dt = time.time() - t0
    print(f"{name} (ell={ell},m={m}): {n_agree}/{n_tested} agree  ({dt:.1f}s, "
          f"{len(cat)} catalog elements x2 swap options)")
    if disagreements:
        print(f"  !! {len(disagreements)} DISAGREEMENTS:")
        for d in disagreements[:10]:
            print("    ", d)
    return len(disagreements) == 0


if __name__ == "__main__":
    all_ok = True
    all_ok &= crosscheck("[[18,4,4]]", 3, 3, [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0), (2, 0)])
    all_ok &= crosscheck("[[16,4,4]]", 2, 4, [(0, 0), (0, 1), (0, 3), (1, 3)], [(0, 0), (0, 1), (0, 3), (1, 1)])
    all_ok &= crosscheck("[[72,12,6]] BB6", 6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    print()
    print("ALL CROSS-CHECKS PASSED" if all_ok else "*** DISAGREEMENTS FOUND ***")
