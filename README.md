
```markdown
██████╗ ██████╗      ██████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██████╔╝██████╔╝    ██║     ██║   ██║██║  ██║█████╗
██╔══██╗██╔══██╗    ██║     ██║   ██║██║  ██║██╔══╝
██████╔╝██████╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗
╚═════╝ ╚═════╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝

 █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗██╗███████╗███╗   ███╗███████╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║██║██╔════╝████╗ ████║██╔════╝
███████║██║   ██║   ██║   ██║   ██║██╔████╔██║██║   ██║██████╔╝██████╔╝███████║██║███████╗██╔████╔██║███████╗
██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██║   ██║██╔══██╗██║╔══  ██╔══██║██║╚════██║██║╚██╔╝██║╚════██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║██║███████║██║ ╚═╝ ██║███████║
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝╚══════╝
```

# Overview 

Computational verification scripts for work exploring **Automorphisms of Bivariate Bicycle
Codes via a refined CRT components and local transfer data"** ring-representation approach.
Here we implement the **forward problem** (detect automorphisms of known BB codes)
and the **inverse problem** (design BB codes with prescribed automorphism
groups) in Pure Python + NumPy (should also work in a SAGE notebook, though)

## Project Contents

| File | Use in verification |
|---|---|
| `bb_ring.py` | Creates instances of the mathematical objects important to the forward and inverse algorithms: The ambient ring `R_{l,m} = F2[x,y]/(x^l-1,y^m-1)`; the full ring-automorphism catalog (multipliers, partial folds, full fold, shears, transpose); a 2×2 Smith-normal-form routine for reducing twisted torus presentations to the standard rectangular one. |
| `bb_code.py` | This script builds `H_X=[A\|B]`, `H_Z=[B^T\|A^T]`, code `n,k` via exact GF(2) rank (n = number of physical qubits, k = number of logical qubits), brute-force minimum distance for small codes, and the full 2-generator stabilizer matrix `S = block_diag(H_X,H_Z)` as in our paper outline (section 2 I believe) |
| `bb_automorphisms.py` | This is the script for our automorphism  detection engine (forward problem). Implements Corollary 3.6 (ring-level, cyclic-source case) and Theorem 3.2 (full 4-slot stabilizer-module case) as GF(2) row-space-equality tests, plus the concrete `gblock` catalog of eqs. (34)–(37) (swap-fold, CX-fold, Hadamard-fold ×2, CZ-fold). `report(...)` prints a full forward-problem scan for one code. |
| `forward_benchmarks.py` | Builds and scans the four requested benchmark codes. |
| `inverse_design.py` | Orbit-closure construction (a constructive proof of Corollary 3.6 with `w=1`) plus 8 worked examples: one per catalog symmetry, a multi-symmetry combination, a ring+stabilizer combination, and a "mixed" CRT-regime example. |
| `test_twisted_torus.py` | Regression test for the Smith-normal-form reducer (~5000 random lattices + the exact `[[16,4,4]]` case). |
| `run_all.py` | Runs everything above in one command. |

In order to run everything and test yourself, paste
```
python3 run_all.py

```
Individual pieces:
```
python3 test_twisted_torus.py
python3 forward_benchmarks.py
python3 inverse_design.py
```

## But how does this python correspond back to our math?

**Ring and polynomials.** A polynomial in `R_{l,m}` is a dense `l×m` NumPy
0/1 array (`Ring.from_terms`, `Ring.mul` via circular convolution,
`Ring.bar` for the antipode `f(x,y) → f(x⁻¹,y⁻¹)`).

**Ring-automorphism catalog (Sec. 2.2).** Every entry — multiplier
`ψ_(jx,jy)`, partial folds `θx, θy`, full fold `ι`, the shear family
(size `d = gcd(l,m)`, `x ↦ x y^{s·m/d}`), and the transpose `τ` (when
`l=m`) — is represented uniformly as a function on exponent pairs
`(i,j) ↦ (i',j')`. `full_catalog(l,m)` builds every multiplier composed
with every fold and every shear (and, when `l=m`, also composed with `τ`):
192 elements for `(l,m)=(12,6)`, 96 for `(3,3)`, etc. — small enough to
brute-force scan exhaustively.

**Forward detection = row-space equality (Corollary 3.6 / Theorem 3.2).**
Rather than reconstructing the CRT component decomposition symbolically
(Sec. 2.2–2.3, which needs Hensel lifts and factoring `x^l−1` over `F2`),
detection is done by a numerically *equivalent* test that sidesteps that
machinery entirely:

- **Ring level (`test_code_automorphism`, Cor. 3.6):** apply the candidate
  `ψ` to `(A,B)` (optionally swapping slots), rebuild `H_X' = [ψ(A)\|ψ(B)]`,
  and check `rowspace(H_X) = rowspace(H_X')` over GF(2). Since
  `rowspace(H_X) = R·(A,B)` *by definition* of the cyclic-submodule picture,
  this is exactly the statement `ψ(A,B) = w·(A,B)` for some unit `w`
  (Theorem 3.1/3.5's "generators of a cyclic module over a finite local
  ring differ by a unit," applied globally via `R ≅ ∏ J_c`).
- **Full stabilizer level (`test_stabilizer_symmetry`, Thm 3.2):** build
  `S = block_diag(H_X,H_Z)` (the 2-generator matrix for
  `M = R·g_X ⊕ R·g_Z`, `g_X=(A,B,0,0)`, `g_Z=(0,0,B̄,Ā)`), apply `Φ = ψ⊕ψ⊕ψ⊕ψ`
  to the four column-blocks, then apply a 4×4 `gblock` matrix (eqs. 34–37)
  mixing the blocks, and again test row-space equality against the
  original `S`. This directly tests the `ψ_L=ψ_R` restriction of Theorem
  3.2, covering pure code automorphisms, transversal-Hadamard/CX-fold
  (Sec. 3.1.2–3.1.3), and CZ/S-fold (Sec. 3.1.4) symmetries in one sweep.

GF(2) rank and row-space equality are computed exactly via Gaussian
elimination (`bb_code.gf2_rref_rank`, `same_rowspace`). We hoped to avoid floating point and heuristics here.

**Inverse design = orbit closure (Sec. 3.4, "Automorphism Design").**
Given target ring automorphisms `φ_1,…,φ_k`, generate the finite subgroup
`Γ = ⟨φ_1,…,φ_k⟩` of `Sym(Z_l×Z_m)` by BFS, then set `A := Σ_{orbit}` for
any seed monomials — i.e. `A`'s support is a union of full `Γ`-orbits. Then
`φ(A) = A` **exactly** for every `φ ∈ Γ`, not just up to a unit: this is a
constructive instance of Corollary 3.6 with `w=1`, and it requires no
search. A second free trick: since every catalog automorphism is a
`Z`-linear map on the exponent lattice and `bar = −Id` is central in
`GL_2(Z)`, `φ∘bar = bar∘φ` always — so setting `B := bar(A)` simultaneously
gives (a) the same ring automorphism group on `B` and (b) a Hadamard-fold
stabilizer symmetry (`bar` + swap), for free. Every claim produced this way
is then independently re-verified with the *same* detection routines used
on the forward-problem benchmarks (`verify_group_invariance`), so the
"proof by construction" is never taken on faith.

## The forward-problem benchmark codes we use

| Code | `(l,m)` | `A` | `B` | Regime (Thm 2.1) | Source |
|---|---|---|---|---|---|
| `[[144,12,12]]` gross | `(12,6)` | `x³+y+y²` | `y³+x+x²` | even×even, `s_l=2,s_m=1` | Bravyi et al., arXiv:2308.07915 |
| `[[72,12,6]]` BB6 | `(6,6)` | `x³+y+y²` | `y³+x+x²` | even×even, `s_l=s_m=1` | same |
| `[[18,4,4]]` 6.6.6 color code | `(3,3)` | `x+1+y²` | `y+1+x²` | odd×odd, semisimple/PIGA | Wang et al., arXiv:2505.09684; errorcorrectionzoo.org/c/stab_18_4_4 |
| `[[16,4,4]]` weight-8 self-dual BB | `(2,4)`* | `1+y+y³+xy³` | `1+y+y³+xy` | even×even, `s_l=1,s_m=2` | Liang & Chen, arXiv:2510.05211 (twisted-torus reduction, see below) |

All four are verified from scratch: `H_X H_Z^T = 0` (CSS), `n = 2lm`,
`k = n − rank(H_X) − rank(H_Z)` over GF(2) exactly matches the published
`k`, and — for the two small codes — brute-force minimum distance over
`ker(H_X)/rowspace(H_Z)` reproduces the published `d = 4` exactly.

### Special exception for`[[16,4,4]]`

errorcorrectionzoo's own `[[16,4,4]]` "symplectic-double" code is a
*different* object — a concatenation of the `[[4,2,2]]` code with its
symplectic double — and is not natively presented as a straight
`R_{l,m}`-type BB code, so it doesn't fit this paper's ambient-ring
framework directly. Instead we use a genuinely-BB, same-parameter code:
Liang & Chen's weight-8 self-dual BB code on a **twisted torus**,
`f(x,y) = 1+x+y+y⁻¹` with lattice basis vectors `a1=(0,4)`, `a2=(2,2)`
(i.e. relations `y⁴=1`, `x²y²=1`). `bb_ring.reduce_twisted_torus` finds the
unimodular exponent-coordinate change `V=[[1,-1],[0,1]]` (via a 2×2 Smith
normal form) that turns this into the standard rectangular presentation
`(l,m)=(2,4)` with `A=1+y+y³+xy³`, `B=1+y+y³+xy = bar(A)`; the code
construction confirms `n=16, k=4, d=4` exactly, matching the published
parameters. This is a useful side benefit of the toolkit: any BB-family
code presented on a non-rectangular ("twisted") torus can be mechanically
reduced to this paper's `R_{l,m}` presentation, which is a prerequisite
for applying the CRT machinery of Sec. 2.2 at all.

## Summary of forward-problem results 

Running `forward_benchmarks.py` scans the full ring-automorphism catalog
(with shears) and the full `gblock` catalog against each code:

- **gross `[[144,12,12]]`:** 4 ring-level automorphisms found (multiplier,
  `θx`, `θy`, full fold, each up to the trivial identity/swap variants), 8
  full stabilizer-level symmetries (identity + Hadamard-fold-with-swap).
- **BB6 `[[72,12,6]]`:** 8 ring-level hits (adds a nontrivial transpose,
  since `l=m=6`), 16 stabilizer-level hits (adds the pure swap-fold and
  Hadamard-fold-without-swap, absent for the gross code since `l≠m` there).
- **color code `[[18,4,4]]`:** 16 ring-level hits, 40 stabilizer-level hits
  — richest symmetry group of the four, consistent with it being the
  self-dual, odd×odd (semisimple) case where `A,B` are literally
  transposes of one another (`B(x,y)=A(y,x)`).
- **`[[16,4,4]]`:** 16 ring-level hits (including a genuine shear symmetry,
  since `gcd(2,4)=2>1`), 40 stabilizer-level hits.

Two important caveats about these counts, stated plainly: (1) the reported
numbers are **raw hits** from the scan (they include the identity, and
compositions that may correspond to the same physical permutation counted
more than once under different `(φ, gblock)` labels) — turning this into a
literal `|Aut|` would need deduplication by resulting physical permutation,
which this toolkit does not attempt; (2) the scan restricts to `ψ_L=ψ_R`
(Sec. 3's stated scope throughout Sec. 3.1–3.2) and to the six concrete
`gblock` matrices from Sec. 3.1.2–3.1.4, not the full unconstrained
symmetry group.

## Inverse-problem worked examples (summary)

`inverse_design.py` builds and *independently verifies* 8 codes:

1. **Single symmetry — multiplier** `ψ_(2,2)` on `(7,3)`: `[[42,6,d]]`.
2. **Single symmetry — partial fold** `θx` on `(5,4)`: `[[40,8,d]]`.
3. **Single symmetry — full fold** `ι` on `(9,5)`, with the free
   `B:=bar(A)` trick also producing a Hadamard-fold stabilizer symmetry:
   `[[90,2,d]]`.
4. **Single symmetry — shear** `x↦xy³` on `(12,6)` (the outline's own
   example, `d=gcd(12,6)=6`): `[[144,16,d]]`.
5. **Single symmetry — transpose** `τ` on `(5,5)`, mirroring the real
   `[[18,4,4]]` code's `B(x,y)=A(y,x)` structure: `[[50,2,d]]`.
6. **Multiple symmetries at once** — the order-12 subgroup generated by
   `⟨mult(2,4), θx⟩` on `(9,5)`, all 12 elements independently verified as
   code automorphisms of a single code: `[[90,74,d]]`.
7. **Variation — ring + stabilizer symmetry together** on `(8,6)` (even×even):
   a multiplier subgroup plus the free bar/Hadamard trick simultaneously;
   160 total full-stabilizer symmetries found by the scan.
8. **Completing the trichotomy — mixed regime** (`l` even, `m` odd) on
   `(8,5)`, not present among the four fixed benchmark codes: `[[80,16,d]]`.

Every example is checked two ways: (a) by construction (orbit-closure
guarantees `φ(A)=A,φ(B)=B` exactly), and (b) by independently re-running
the same `test_code_automorphism` / `scan_stabilizer_symmetries` routines
used on the forward-problem benchmarks — i.e. the construction is not
trusted blindly.

## Known limitations / honest scope

- The CRT component decomposition (`R_{l,m} ≅ ∏ J_c`, Hensel lifts,
  Frobenius orbits) from Sec. 2.2 is **not** implemented; detection instead
  uses the numerically equivalent row-space test, which is exact but
  doesn't expose the component-wise structure (units `ω_c`, transport maps
  `T*_ψ`, etc.) that Sec. 3.2's *proof* machinery is built on.
- `ψ_L ≠ ψ_R` (Sec. 3.3) is out of scope, matching the outline's own
  stated restriction through Sec. 3.
- Minimum distance is only computed by brute force for `n ≲ 26`; the two
  large codes' published distances (`d=12`, `d=6`) are taken from the
  literature, not re-derived here.
- The `gblock` catalog covers exactly the six matrices given explicitly in
  Sec. 3.1.2–3.1.4 (eqs. 34–37 plus identity); "combining gate types"
  (Sec. 3.1.5) and monomial-offset CX variants are not enumerated.
