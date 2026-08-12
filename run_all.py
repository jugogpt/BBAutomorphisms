"""
run_all.py
==========
Single entry point: runs the twisted-torus regression test, the forward
benchmark suite (4 codes, row-space method), the inverse-design worked
examples (8 cases, w=1 monomial-orbit method), the REAL CRT/Algorithm-3.12
pipeline (factorization + Hensel lifting + component construction + local
matching-equation solve, cross-validated against the row-space oracle on
all four benchmark codes), and the genuinely-nontrivial-unit CRT inverse
design (4 cases, w != 1, each verified 3 independent ways), in that order.
Intended as the "does everything still work" smoke test.

    python3 run_all.py
"""
import time

def section(title):
    print("\n" + "#" * 78)
    print("# " + title)
    print("#" * 78)


if __name__ == "__main__":
    t0 = time.time()

    section("0. Regression test: twisted-torus Smith-normal-form reducer")
    import test_twisted_torus
    test_twisted_torus.run()

    section("1. FORWARD PROBLEM (row-space method): four benchmark BB codes")
    import forward_benchmarks
    ring1, code1 = forward_benchmarks.build_gross()
    from bb_automorphisms import report
    report("[[144,12,12]] gross code", ring1, code1)
    ring2, code2 = forward_benchmarks.build_bb72()
    report("[[72,12,6]] BB6 code", ring2, code2)
    ring3, code3 = forward_benchmarks.build_18_4_4()
    report("[[18,4,4]] 6.6.6 color code", ring3, code3)
    ring4, code4 = forward_benchmarks.build_16_4_4()
    report("[[16,4,4]] weight-8 self-dual BB code (twisted-torus reduction)", ring4, code4)

    section("2. INVERSE PROBLEM (monomial-orbit method, w=1): worked examples")
    import inverse_design
    inverse_design.example_multiplier()
    inverse_design.example_partial_fold()
    inverse_design.example_full_fold_and_hadamard()
    inverse_design.example_shear()
    inverse_design.example_transpose()
    inverse_design.example_combined_group()
    inverse_design.example_combined_ring_and_stabilizer()
    inverse_design.example_mixed_regime()

    section("3. FORWARD PROBLEM (real CRT / Algorithm 3.12): factorization, "
            "Hensel lifting, component construction, local matching-equation solve")
    import test_crt_full_suite
    test_crt_full_suite.run()

    section("4. INVERSE PROBLEM (genuine CRT construction, w != 1): "
            "twisted-orbit-sum with a prescribed nontrivial unit")
    import bb_inverse_crt
    bb_inverse_crt.example_nontrivial_unit_partial_fold()
    bb_inverse_crt.example_nontrivial_unit_multiplier()
    bb_inverse_crt.example_nontrivial_unit_shear()
    bb_inverse_crt.example_nontrivial_unit_fullfold()

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("\nAll forward and inverse checks passed (row-space method, "
          "and real CRT/Algorithm-3.12 method, both problems, both ways).")

