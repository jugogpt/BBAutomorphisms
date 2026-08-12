"""
test_twisted_torus.py
======================
Regression test for bb_ring.reduce_twisted_torus (the 2x2 Smith-normal-form
routine used to reduce a twisted-torus BB-like code presentation to the
outline's standard rectangular R_{ell,m} presentation).

Run directly:  python3 test_twisted_torus.py
"""
import random
import numpy as np

from bb_ring import reduce_twisted_torus


def run(n_trials: int = 5000, coord_range: int = 25, seed: int = 0) -> None:
    random.seed(seed)
    trials = 0
    fails = 0
    for _ in range(n_trials):
        a1 = (random.randint(-coord_range, coord_range), random.randint(-coord_range, coord_range))
        a2 = (random.randint(-coord_range, coord_range), random.randint(-coord_range, coord_range))
        det = a1[0] * a2[1] - a1[1] * a2[0]
        if det == 0:
            continue
        trials += 1
        try:
            d1, d2, V = reduce_twisted_torus(a1, a2)
        except Exception as e:
            print("FAIL (exception):", a1, a2, e)
            fails += 1
            continue
        if d1 * d2 != abs(det):
            print("FAIL (det mismatch):", a1, a2, d1, d2, det)
            fails += 1
            continue
        detV = round(np.linalg.det(V))
        if detV not in (1, -1):
            print("FAIL (V not unimodular):", a1, a2, V)
            fails += 1
            continue
        v1 = np.array(a1) @ V
        v2 = np.array(a2) @ V
        ok = (v1[0] % d1 == 0 and v1[1] % d2 == 0 and v2[0] % d1 == 0 and v2[1] % d2 == 0)
        if not ok:
            print("FAIL (lattice image not rectangular):", a1, a2, d1, d2, v1, v2)
            fails += 1
    print(f"test_twisted_torus: {trials - fails}/{trials} random lattices reduced correctly.")
    assert fails == 0, f"{fails} failures out of {trials} trials"

    # The specific case this toolkit relies on for the [[16,4,4]] benchmark.
    d1, d2, V = reduce_twisted_torus((0, 4), (2, 2))
    assert (d1, d2) == (2, 4)
    assert np.array_equal(V, np.array([[1, -1], [0, 1]]))
    print("test_twisted_torus: [[16,4,4]] twisted-torus case reproduced exactly.")


if __name__ == "__main__":
    run()
