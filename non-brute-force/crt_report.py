"""
crt_report.py
==============
Runs the REAL CRT-based Algorithm 3.12 (bb_matching.py) -- factorization,
Hensel lifting, component construction, local matching-equation solve, with
genuine Prop. 3.10 pruning -- on all four forward-problem benchmark codes,
and CRT-verifies every inverse-design example, printing pruning statistics
(how many candidates were rejected at each Algorithm 3.12 step) alongside
the final accept counts. Every accept/reject verdict below has already been
cross-validated against the independent row-space oracle in
test_matching_crosscheck.py; this script is the reporting layer.
"""
import time
from bb_ring import Ring, full_catalog, multiplier, theta_x, theta_y, full_fold, transpose_auto, shear_family
from bb_code import BBCode
from bb_matching import CRTContext, test_code_automorphism_crt
from inverse_design import build_orbit_poly, group_closure


def report_forward(name, ell, m, A_terms, B_terms):
    ring = Ring(ell, m)
    A = ring.from_terms(A_terms)
    B = ring.from_terms(B_terms)
    ctx = CRTContext(ell, m)
    cat = full_catalog(ell, m)
    print("=" * 78)
    print(f"{name}  (ell={ell}, m={m})")
    print(f"  |C| (CRT components) = {len(ctx.components)}   "
          f"s_ell={ctx.s_ell}, s_m={ctx.s_m}   catalog size = {len(cat)}")
    counts = {}
    t0 = time.time()
    for phi in cat:
        for swap in (False, True):
            ok, info = test_code_automorphism_crt(ring, ctx, A, B, phi, swap_AB=swap)
            key = "ACCEPTED" if ok else info["rejected_at"]
            counts[key] = counts.get(key, 0) + 1
    dt = time.time() - t0
    total = sum(counts.values())
    print(f"  Algorithm 3.12 results over {total} (phi,swap) candidates ({dt:.1f}s):")
    for key in ["ACCEPTED", "step3_support_mismatch", "step4_jacobian_rank_mismatch", "step5_no_local_unit"]:
        if key in counts:
            label = key if key != "ACCEPTED" else "accepted (genuine code automorphism)"
            print(f"    {label}: {counts[key]}")
    pruned = counts.get("step3_support_mismatch", 0) + counts.get("step4_jacobian_rank_mismatch", 0)
    print(f"  -> {pruned}/{total} candidates rejected by cheap Prop. 3.10 pre-filter "
          f"(steps 3-4), before the expensive per-component solve (step 5)")


def report_inverse(name, ring, A, B, generators, expect_swap=False):
    ctx = CRTContext(ring.ell, ring.m)
    group = group_closure(generators, ring.ell, ring.m)
    n_ok = sum(test_code_automorphism_crt(ring, ctx, A, B, phi, swap_AB=expect_swap)[0] for phi in group)
    print(f"  [{name}] CRT-verified: {n_ok}/{len(group)} group elements "
          f"confirmed via the real local matching-equation solve")


if __name__ == "__main__":
    print("#" * 78)
    print("# FORWARD PROBLEM: real Algorithm 3.12 on the four benchmark codes")
    print("#" * 78)
    report_forward("[[144,12,12]] gross code", 12, 6,
                    [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    report_forward("[[72,12,6]] BB6 code", 6, 6,
                    [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    report_forward("[[18,4,4]] 6.6.6 color code", 3, 3,
                    [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0), (2, 0)])
    report_forward("[[16,4,4]] weight-8 self-dual BB code", 2, 4,
                    [(0, 0), (0, 1), (0, 3), (1, 3)], [(0, 0), (0, 1), (0, 3), (1, 1)])

    print()
    print("#" * 78)
    print("# INVERSE PROBLEM: CRT-verify the orbit-closure-constructed examples")
    print("#" * 78)
    ring = Ring(7, 3)
    psi = multiplier(7, 3, 2, 2)
    A = build_orbit_poly(ring, [(1, 1), (0, 2)], [psi])
    B = build_orbit_poly(ring, [(2, 0)], [psi])
    report_inverse("multiplier psi_(2,2) on (7,3)", ring, A, B, [psi])

    ring = Ring(5, 4)
    th = theta_x()
    A = build_orbit_poly(ring, [(1, 1)], [th])
    B = build_orbit_poly(ring, [(2, 3)], [th])
    report_inverse("partial fold theta_x on (5,4)", ring, A, B, [th])

    ring = Ring(12, 6)
    sh3 = [s for s in shear_family(12, 6) if "c=3" in s.name][0]
    A = build_orbit_poly(ring, [(1, 0), (1, 1)], [sh3])
    B = build_orbit_poly(ring, [(2, 0), (0, 1)], [sh3])
    report_inverse("shear x->xy^3 on (12,6)", ring, A, B, [sh3])

    ring = Ring(5, 5)
    A = ring.from_terms([(1, 0), (0, 2)])
    B = ring.apply_auto(A, transpose_auto())
    report_inverse("transpose tau on (5,5)", ring, A, B, [transpose_auto()], expect_swap=True)

    ring = Ring(9, 5)
    psi2 = multiplier(9, 5, 2, 4)
    th2 = theta_x()
    A = build_orbit_poly(ring, [(1, 1)], [psi2, th2])
    B = build_orbit_poly(ring, [(2, 3)], [psi2, th2])
    report_inverse("combined <mult(2,4),theta_x> on (9,5)", ring, A, B, [psi2, th2])

    print()
    print("All forward and inverse CRT checks complete.")
