"""
bb_automorphisms.py
====================
Computational detection of BB-code automorphisms and fold-transversal
symmetries, implementing (numerically, without the CRT machinery) the two
matching theorems of the outline:

  * Theorem 3.1 / Corollary 3.6 (n=1, cyclic-source case):
        sigma(M) = M  <=>  psi(f1,f2) = w.(f1,f2),  w in R^x
    Verified here by comparing the F2 row-space of HX = [A|B] before and
    after applying the candidate ring automorphism psi to (A,B) (and
    optionally swapping the A/B slots). Row-space equality is *equivalent*
    to "differ by a unit that assembles from a per-component unit" because
    HX's row space literally *is* R.(A,B) in coordinates -- this sidesteps
    needing the explicit CRT component decomposition to certify units.

  * Theorem 3.2 (n=4, 2-generated case), covering the gblock catalog of
    Sec. 3.1.2-3.1.4 (block-diagonal / anti-block-diagonal / lower
    triangular gblock matrices, i.e. H-type, CX-fold-type, and CZ/S-fold
    type symmetries) composed with Phi = psi (+) psi (+) psi (+) psi
    (the psi_L = psi_R restriction the outline adopts throughout Sec. 3):
        sigma(M) = M  <=>  the two (2lm x 4lm) generator matrices S, S'
        (before/after applying gblock . Phi) have the same F2 row space.

Both tests reduce to GF(2) row-space equality, which we compute exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from bb_ring import Ring, RingAuto, full_catalog
from bb_code import BBCode, same_rowspace, gf2_rref_rank


# ============================================================================
#  gblock catalog (Sec. 3.1.2 - 3.1.4), as 4x4 F2 matrices acting on the
#  4 slots (A, B, barB, barA) of N = R^4.  Tensored with I_{ell*m} to act
#  on the full 4*ell*m - dim space.
# ============================================================================

I4 = np.eye(4, dtype=np.uint8)

# eq (34): swap-fold  I tensor X   (pure qubit swap on the (i, i+lm) fold)
GBLOCK_SWAP_FOLD = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.uint8)

# eq (35): CX-fold (A upper block diagonal / D lower block diagonal)
GBLOCK_CX_FOLD = np.array([
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
], dtype=np.uint8)

# eq (36): transversal-Hadamard type (anti-block-diagonal), two variants
GBLOCK_HADAMARD = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.uint8)

GBLOCK_HADAMARD_SWAP = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=np.uint8)

# eq (37): CZ-fold (lower triangular, all B_i = 0)
GBLOCK_CZ_FOLD = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
], dtype=np.uint8)

GBLOCK_CATALOG = {
    "identity (pure code automorphism)": I4,
    "swap-fold I(x)X  [eq.34, pure permutation]": GBLOCK_SWAP_FOLD,
    "CX-fold  [eq.35]": GBLOCK_CX_FOLD,
    "Hadamard-fold, no swap [eq.36a]": GBLOCK_HADAMARD,
    "Hadamard-fold, with swap [eq.36b]": GBLOCK_HADAMARD_SWAP,
    "CZ-fold  [eq.37]": GBLOCK_CZ_FOLD,
}


# ============================================================================
#  Ring-level (Corollary 3.6) detection: psi (+) psi diagonal, gblock in
#  {identity, plain A<->B swap}.
# ============================================================================

def test_code_automorphism(ring: Ring, code: BBCode, phi: RingAuto,
                            swap_AB: bool = False,
                            rankHX: Optional[int] = None) -> Tuple[bool, dict]:
    """
    Test whether Psi = (swap?, phi applied diagonally to both slots) is a
    code automorphism, i.e. whether it preserves the X-stabilizer module
    M_X = R.(A,B)  (equivalently, by the CSS/bar structure, also M_Z).
    This directly implements Corollary 3.6.
    """
    Ap = ring.apply_auto(code.A, phi)
    Bp = ring.apply_auto(code.B, phi)
    if swap_AB:
        Ap, Bp = Bp, Ap
    HXp = np.hstack([code._circulant_lift(Ap), code._circulant_lift(Bp)])
    if rankHX is not None:
        r2 = gf2_rref_rank(HXp)
        ok = (r2 == rankHX) and (gf2_rref_rank(np.vstack([code.HX, HXp])) == rankHX)
    else:
        ok = same_rowspace(code.HX, HXp)
    info = {"phi": phi.name, "swap_AB": swap_AB,
            "A'": ring.poly_str(Ap), "B'": ring.poly_str(Bp)}
    return ok, info


def scan_code_automorphisms(ring: Ring, code: BBCode,
                             include_shears: bool = True,
                             also_try_swap: bool = True) -> List[dict]:
    """
    Scan the full ring-automorphism catalog (Sec. 2.2) for code automorphisms
    of `code` in the psi_L = psi_R, gblock in {id, swap} regime.
    Returns the list of hits (as info dicts) -- this is the forward-problem
    "Automorphism Detection" of Sec. 3.1.
    """
    hits = []
    cat = full_catalog(ring.ell, ring.m, include_shears=include_shears)
    swap_options = [False, True] if also_try_swap else [False]
    rankHX = code.rank_HX()
    for phi in cat:
        for swap in swap_options:
            ok, info = test_code_automorphism(ring, code, phi, swap_AB=swap, rankHX=rankHX)
            if ok:
                hits.append(info)
    return hits


# ============================================================================
#  Full stabilizer-level (Theorem 3.2) detection, covering the gblock
#  catalog of Sec 3.1.2-3.1.4, restricted to psi_L = psi_R (Phi = psi^{(+)4}).
# ============================================================================

def _permute_columns_block(S: np.ndarray, ell_m: int, phi: RingAuto,
                            ell: int, m: int) -> np.ndarray:
    """Apply the grid automorphism phi to each of the 4 ell*m column-blocks
    of S (i.e. Phi = psi (+) psi (+) psi (+) psi acting on N=R^4)."""
    # Build the permutation of the ell*m coordinate (row-major index a*m+b)
    perm = np.zeros(ell_m, dtype=np.int64)
    for a in range(ell):
        for b in range(m):
            a2, b2 = phi.map(a, b)
            perm[(a2 % ell) * m + (b2 % m)] = a * m + b
    # perm[new_index] = old_index  =>  new_col[new_index] = old_col[old_index]
    out = np.zeros_like(S)
    for block in range(4):
        cols = slice(block * ell_m, (block + 1) * ell_m)
        out[:, cols] = S[:, cols][:, perm]
    return out


def _apply_gblock_columns(S: np.ndarray, ell_m: int, gblock: np.ndarray) -> np.ndarray:
    """Mix the 4 column-blocks of S according to the 4x4 F2 matrix gblock
    (new_block_j = XOR over i with gblock[j,i]=1 of block_i)."""
    blocks = [S[:, j * ell_m:(j + 1) * ell_m] for j in range(4)]
    new_blocks = []
    for jrow in range(4):
        acc = np.zeros((S.shape[0], ell_m), dtype=np.uint8)
        for i in range(4):
            if gblock[jrow, i]:
                acc ^= blocks[i]
        new_blocks.append(acc)
    return np.hstack(new_blocks)


def test_stabilizer_symmetry(ring: Ring, code: BBCode, phi: RingAuto,
                              gblock: np.ndarray, S: Optional[np.ndarray] = None,
                              rankS: Optional[int] = None) -> bool:
    """
    Theorem 3.2 test: is sigma = gblock . (psi (+) psi (+) psi (+) psi) an
    automorphism of the full stabilizer module M = R.gX + R.gZ subset N=R^4?
    """
    ell_m = ring.ell * ring.m
    if S is None:
        S = code.stabilizer_matrix()
    S1 = _permute_columns_block(S, ell_m, phi, ring.ell, ring.m)
    S2 = _apply_gblock_columns(S1, ell_m, gblock)
    if rankS is not None:
        r2 = gf2_rref_rank(S2)
        if r2 != rankS:
            return False
        return gf2_rref_rank(np.vstack([S, S2])) == rankS
    return same_rowspace(S, S2)


def scan_stabilizer_symmetries(ring: Ring, code: BBCode,
                                include_shears: bool = True) -> List[dict]:
    """
    Scan (ring automorphism) x (gblock catalog) for full stabilizer-module
    symmetries -- the psi_L=psi_R restriction of Sec. 3.1's detection
    problem, covering pure code automorphisms, transversal Hadamard/CX-fold
    (Sec 3.1.2/3.1.3), and CZ/S-fold (Sec 3.1.4) simultaneously.
    """
    hits = []
    cat = full_catalog(ring.ell, ring.m, include_shears=include_shears)
    S = code.stabilizer_matrix()
    rankS = gf2_rref_rank(S)
    for gname, gmat in GBLOCK_CATALOG.items():
        for phi in cat:
            if test_stabilizer_symmetry(ring, code, phi, gmat, S=S, rankS=rankS):
                hits.append({"gblock": gname, "phi": phi.name})
    return hits


# ============================================================================
#  Pretty-printer for a forward-problem report
# ============================================================================

def report(name: str, ring: Ring, code: BBCode, include_shears: bool = True,
           full_stabilizer_scan: bool = True) -> None:
    print("=" * 78)
    print(f"{name}   ({code.params_str()})")
    print(f"  A = {ring.poly_str(code.A)}")
    print(f"  B = {ring.poly_str(code.B)}")
    print(f"  CSS commutes: {code.css_commutes()}")
    d = code.min_distance_bruteforce()
    if d is not None:
        print(f"  distance (bruteforce) = {d}")
    cat = full_catalog(ring.ell, ring.m, include_shears=include_shears)
    print(f"  ring-automorphism group scanned (genuine BFS closure, Sec 2.2 catalog): {len(cat)} elements")
    hits = scan_code_automorphisms(ring, code, include_shears=include_shears)
    no_swap = [h for h in hits if not h["swap_AB"]]
    with_swap = [h for h in hits if h["swap_AB"]]
    print(f"  code automorphisms found (ring-level, Cor. 3.6): {len(no_swap)}/{len(cat)} "
          f"without A<->B swap, {len(with_swap)}/{len(cat)} with swap")
    if full_stabilizer_scan:
        shits = scan_stabilizer_symmetries(ring, code, include_shears=include_shears)
        print(f"  full stabilizer-module symmetries found (Thm 3.2, gblock catalog, "
              f"{len(cat)} phi's x {len(GBLOCK_CATALOG)} gblocks = {len(cat)*len(GBLOCK_CATALOG)} tested): "
              f"{len(shits)}")
        gfam = {}
        for h in shits:
            gfam.setdefault(h["gblock"], 0)
            gfam[h["gblock"]] += 1
        for g, cnt in sorted(gfam.items()):
            print(f"    - {g}: {cnt} hit(s)")



