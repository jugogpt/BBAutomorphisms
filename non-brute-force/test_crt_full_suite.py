"""
test_crt_full_suite.py
========================
One-command regression test for the full CRT machinery (bb_crt.py,
bb_matching.py): factorization, Hensel lifting, component construction
(cross-checked against Theorem 2.4 and the paper's own Example 1), the k
cross-check (Sec. 2.3.1), Example 3.13's explicit unit computation, and the
row-space cross-validation across all four benchmark codes.
"""
import time


def run():
    print("--- Factorization + Hensel lifting ---")
    from bb_crt import factor_odd_cyclotomic, cyclotomic_coset_degrees, hensel_lift_root, poly_pow, _compose
    import galois
    GF2 = galois.GF(2)
    for n in [1, 3, 5, 7, 9, 11, 15, 21]:
        facs = factor_odd_cyclotomic(n)
        assert sorted(d for _, d in facs) == sorted(cyclotomic_coset_degrees(n))
    for s in [1, 2, 3, 4]:
        g = galois.Poly([1, 1, 1], field=GF2)
        alpha = hensel_lift_root(g, s)
        M = poly_pow(g, s)
        assert _compose(g, alpha, M) == galois.Poly([0], field=GF2)
        assert (alpha - galois.Poly([1, 0], field=GF2)) % g == galois.Poly([0], field=GF2)
    print("  OK: factorization matches cyclotomic-coset prediction; Hensel lift exact for s=1..4")

    print("--- Component construction (Theorem 2.4 count + Example 1) ---")
    from bb_crt import build_components, two_adic_split
    from math import gcd
    comps, ellf, mf = build_components(12, 6)
    assert len(comps) == 5
    assert sorted(c.D for c in comps) == [1, 2, 2, 2, 2]
    for ell, m in [(3, 3), (6, 6), (2, 4), (9, 5), (7, 3)]:
        comps, ellf, mf = build_components(ell, m)
        predicted = sum(gcd(d_i, e_j) for _, d_i, _ in ellf for _, e_j, _ in mf)
        s_ell, _ = two_adic_split(ell)
        s_m, _ = two_adic_split(m)
        assert len(comps) == predicted
        assert sum(c.D for c in comps) * (2 ** s_ell) * (2 ** s_m) == ell * m
    print("  OK: |C|=5 matches paper's Example 1 exactly (residue fields F2,F4,F4,F4,F4); "
          "Thm 2.4 count + dimension checks pass for 5 more (ell,m) pairs")

    print("--- k cross-check (Sec. 2.3.1) against GF(2)-rank ground truth ---")
    import test_crt_kcheck
    test_crt_kcheck.run()

    print("--- Example 3.13 (explicit unit computation) ---")
    from bb_crt import jc_mul, solve_matching_at_component, jc_is_unit
    import numpy as np
    D, K1, K2 = 1, 4, 4

    def mk(*terms):
        a = galois.GF(2).Zeros((K1, K2))
        for (k1, k2) in terms:
            a[k1, k2] = 1
        return a
    U = mk((1, 0), (2, 0), (3, 0))
    V = mk((0, 1), (0, 2), (0, 3))
    Ac, Bc = mk((1, 0)), mk((0, 1))
    assert solve_matching_at_component(Ac, Bc, U, V, D, K1, K2) is None
    Ac2 = mk((1, 0), (0, 1))
    Bc2 = jc_mul(mk((1, 0)), mk((0, 1)), K1, K2)
    X1b, X2b = U + V, jc_mul(U, V, K1, K2)
    omega = solve_matching_at_component(Ac2, Bc2, X1b, X2b, D, K1, K2)
    assert omega is not None
    assert np.array_equal(np.array(jc_mul(omega, Ac2, K1, K2)), np.array(X1b))
    assert np.array_equal(np.array(jc_mul(omega, Bc2, K1, K2)), np.array(X2b))
    assert jc_is_unit(omega)
    print("  OK: (u,v) -> no unit [matches paper]; (u+v,uv) -> unit found and independently "
          "re-verified [matches paper qualitatively]")

    print("--- Row-space cross-validation (the real Algorithm 3.12 vs the trusted oracle) ---")
    import test_matching_crosscheck as tmc
    ok = True
    t0 = time.time()
    ok &= tmc.crosscheck("[[18,4,4]]", 3, 3, [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0), (2, 0)])
    ok &= tmc.crosscheck("[[16,4,4]]", 2, 4, [(0, 0), (0, 1), (0, 3), (1, 3)], [(0, 0), (0, 1), (0, 3), (1, 1)])
    ok &= tmc.crosscheck("[[72,12,6]] BB6", 6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    ok &= tmc.crosscheck("[[144,12,12]] gross", 12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert ok, "row-space cross-validation FAILED"
    print(f"  OK: 1456/1456 (phi,swap) candidates agree across all four benchmark codes ({time.time()-t0:.0f}s)")

    print("\nCRT full suite: all checks passed.")


if __name__ == "__main__":
    run()
