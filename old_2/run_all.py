"""
run_all.py
==========
Single entry point: runs the twisted-torus regression test, the forward
benchmark suite (4 codes), and the inverse-design worked examples (8 cases),
in that order. Intended as the "does everything still work" smoke test.

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

    section("1. FORWARD PROBLEM: four benchmark BB codes")
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

    section("2. INVERSE PROBLEM: automorphism design worked examples")
    import inverse_design
    inverse_design.example_multiplier()
    inverse_design.example_partial_fold()
    inverse_design.example_full_fold_and_hadamard()
    inverse_design.example_shear()
    inverse_design.example_transpose()
    inverse_design.example_combined_group()
    inverse_design.example_combined_ring_and_stabilizer()
    inverse_design.example_mixed_regime()

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print("\nAll forward and inverse checks passed.")
