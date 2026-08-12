`markdown
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

# BB Code Automorphism Toolkit

Computational verification suite for **"Automorphisms of Bivariate Bicycle
Codes via Refined CRT Components and Local Transfer Data"** (outline draft).
It implements the forward problem (detect automorphisms of known BB codes)
and the inverse problem (design BB codes with prescribed automorphism
groups), and cross-checks every claim numerically rather than symbolically.

Pure Python + NumPy (+ the `galois` package for GF(2^d) field arithmetic,
used by the CRT machinery described below). No SageMath install was
attempted in this sandbox (network access is restricted to package
indices, not the SageMath conda channels), but every routine here is
written in plain Python so it also runs unmodified inside a SageMath
notebook's Python kernel.

Two independent implementations exist for both problems, cross-validated
against each other rather than either being trusted alone:

- A **row-space / orbit-closure** method (fast, exact, but answers only
  yes/no ΓÇö see `bb_automorphisms.py`, `inverse_design.py`).
- The **real CRT machinery of the outline** (factorization, Hensel
  lifting, component construction, the local matching-equation solve of
  Algorithm 3.12) ΓÇö see `bb_crt.py`, `bb_matching.py`, `bb_inverse_crt.py`.
  This is slower but exposes the actual per-component units `╧ë_c`,
  supports designing codes with a **prescribed nontrivial unit `wΓëá1`**
  (which the row-space method cannot do at all), and gives genuine
  Prop. 3.10 pruning before the expensive per-component solve.

## Files

| File | Contents |
|---|---|
| `bb_ring.py` | The ambient ring `R_{l,m} = F2[x,y]/(x^l-1,y^m-1)`; the full ring-automorphism catalog from Sec. 2.2 (multipliers, partial folds, full fold, both shear families, transpose), built by genuine BFS group closure (`generate_group`); a 2├ù2 Smith-normal-form routine for reducing "twisted torus" presentations to the standard rectangular one. |
| `bb_code.py` | `BBCode`: builds `H_X=[A\|B]`, `H_Z=[B^T\|A^T]`, computes `n,k` via exact GF(2) rank, brute-force minimum distance for small codes, and the full 2-generator stabilizer matrix `S = block_diag(H_X,H_Z)` (Sec. 2.3.1 / Sec. 4). |
| `bb_automorphisms.py` | Row-space detection engine. Implements Corollary 3.6 (ring-level, cyclic-source case) and Theorem 3.2 (full 4-slot stabilizer-module case) as GF(2) row-space-equality tests, plus the concrete `gblock` catalog of eqs. (34)ΓÇô(37) (swap-fold, CX-fold, Hadamard-fold ├ù2, CZ-fold). `report(...)` prints a full forward-problem scan for one code. |
| `forward_benchmarks.py` | Builds and scans the four requested benchmark codes (row-space method). |
| `inverse_design.py` | Orbit-closure construction (a constructive proof of Corollary 3.6 with `w=1`) plus 8 worked examples: one per catalog symmetry, a multi-symmetry combination, a ring+stabilizer combination, and a "mixed" CRT-regime example. |
| `bb_crt.py` | The real CRT machinery: factors `x^l-1` over F2 (Sec. 2.2, eq. 5), Hensel-lifts to exact roots (Lemma 2.2/B.1), builds the refined component set `C` (Theorem 2.4/Lemma 2.3, as `Component` objects with residue field, uniformizers, Frobenius orbit), reduces a polynomial to its local HasseΓÇôTaylor jet at a component (Sec. 3.2.1), and implements `J_c` arithmetic and the local matching-equation solve. |
| `bb_matching.py` | The real Algorithm 3.12 (steps 1, 3ΓÇô6; step 2's Borel prune deliberately omitted ΓÇö see below), built on `bb_crt.py`. `CRTContext` caches all code-independent CRT data per `(l,m)`, mirroring `full_catalog`'s caching. |
| `crt_report.py` | Runs the real Algorithm 3.12 on all four benchmark codes with pruning statistics, and CRT-verifies the `inverse_design.py` examples. |
| `bb_inverse_crt.py` | **Genuine nontrivial-unit inverse design**: the "twisted-orbit-sum" construction (Lemma, proved and hand-verified before coding) builds codes with `╧ê(A,B) = w┬╖(A,B)` for a *prescribed, generally nontrivial* monomial unit `w` ΓÇö strictly more general than `inverse_design.py`'s `w=1`-only construction. Four worked examples, each checked three independent ways. |
| `test_twisted_torus.py` | Regression test for the Smith-normal-form reducer (~5000 random lattices + the exact `[[16,4,4]]` case). |
| `test_crt_full_suite.py` | Regression test for `bb_crt.py`: factorization/Hensel-lift correctness, component count/dimension against Theorem 2.4, `k` cross-check against GF(2)-rank ground truth on all four benchmark codes, reproduction of the outline's own Example 1 and Example 3.13. |
| `test_matching_crosscheck.py` | Cross-validates `bb_matching.py`'s verdicts against the row-space oracle on every `(╧å,swap)` candidate for all four benchmark codes (1456/1456 agree). |
| `test_crt_kcheck.py` | Focused regression test for the `k` cross-check alone. |
| `run_all.py` | Runs everything above in one command. |
| `verification_companion.tex` / `.pdf` | A LaTeX companion document giving, section by section, the exact paper quote each algorithm step implements ΓÇö both the row-space/orbit-closure methods and the real CRT/Algorithm-3.12 machinery ΓÇö plus the "shared structure" and "honest scope" discussions below in fully typeset form. |

Run everything:
```
python3 run_all.py
```
(~65s on this machine ΓÇö the real CRT/Algorithm 3.12 pass over all four
benchmark codes' full catalogs dominates the runtime.) Individual pieces:
```
python3 test_twisted_torus.py
python3 forward_benchmarks.py       # row-space forward problem
python3 inverse_design.py           # orbit-closure inverse problem (w=1)
python3 test_crt_full_suite.py      # CRT machinery regression tests
python3 crt_report.py               # real Algorithm 3.12 forward problem
python3 bb_inverse_crt.py           # nontrivial-unit (w!=1) inverse problem
python3 test_matching_crosscheck.py # cross-validate CRT vs row-space
```

## How the theory maps to code

**Ring and polynomials.** A polynomial in `R_{l,m}` is a dense `l├ùm` NumPy
0/1 array (`Ring.from_terms`, `Ring.mul` via circular convolution,
`Ring.bar` for the antipode `f(x,y) ΓåÆ f(xΓü╗┬╣,yΓü╗┬╣)`).

**Ring-automorphism catalog (Sec. 2.2).** Every entry ΓÇö multiplier
`╧ê_(jx,jy)`, partial folds `╬╕x, ╬╕y`, full fold `╬╣`, the shear family
(size `d = gcd(l,m)`, `x Γåª x y^{s┬╖m/d}`), and the transpose `╧ä` (when
`l=m`) ΓÇö is represented uniformly as a function on exponent pairs
`(i,j) Γåª (i',j')`. `full_catalog(l,m)` builds every multiplier composed
with every fold and every shear (and, when `l=m`, also composed with `╧ä`):
192 elements for `(l,m)=(12,6)`, 96 for `(3,3)`, etc. ΓÇö small enough to
brute-force scan exhaustively.

**Forward detection = row-space equality (Corollary 3.6 / Theorem 3.2).**
Rather than reconstructing the CRT component decomposition symbolically
(Sec. 2.2ΓÇô2.3, which needs Hensel lifts and factoring `x^lΓêÆ1` over `F2`),
detection is done by a numerically *equivalent* test that sidesteps that
machinery entirely:

- **Ring level (`test_code_automorphism`, Cor. 3.6):** apply the candidate
  `╧ê` to `(A,B)` (optionally swapping slots), rebuild `H_X' = [╧ê(A)\|╧ê(B)]`,
  and check `rowspace(H_X) = rowspace(H_X')` over GF(2). Since
  `rowspace(H_X) = R┬╖(A,B)` *by definition* of the cyclic-submodule picture,
  this is exactly the statement `╧ê(A,B) = w┬╖(A,B)` for some unit `w`
  (Theorem 3.1/3.5's "generators of a cyclic module over a finite local
  ring differ by a unit," applied globally via `R Γëà ΓêÅ J_c`).
- **Full stabilizer level (`test_stabilizer_symmetry`, Thm 3.2):** build
  `S = block_diag(H_X,H_Z)` (the 2-generator matrix for
  `M = R┬╖g_X Γèò R┬╖g_Z`, `g_X=(A,B,0,0)`, `g_Z=(0,0,B╠ä,─Ç)`), apply `╬ª = ╧êΓèò╧êΓèò╧êΓèò╧ê`
  to the four column-blocks, then apply a 4├ù4 `gblock` matrix (eqs. 34ΓÇô37)
  mixing the blocks, and again test row-space equality against the
  original `S`. This directly tests the `╧ê_L=╧ê_R` restriction of Theorem
  3.2, covering pure code automorphisms, transversal-Hadamard/CX-fold
  (Sec. 3.1.2ΓÇô3.1.3), and CZ/S-fold (Sec. 3.1.4) symmetries in one sweep.

GF(2) rank and row-space equality are computed exactly via Gaussian
elimination (`bb_code.gf2_rref_rank`, `same_rowspace`) ΓÇö no floating point,
no heuristics.

**Inverse design = orbit closure (Sec. 3.4, "Automorphism Design").**
Given target ring automorphisms `╧å_1,ΓÇª,╧å_k`, generate the (finite) subgroup
`╬ô = Γƒ¿╧å_1,ΓÇª,╧å_kΓƒ⌐` of `Sym(Z_l├ùZ_m)` by BFS, then set `A := ╬ú_{orbit}` for
any seed monomials ΓÇö i.e. `A`'s support is a union of full `╬ô`-orbits. Then
`╧å(A) = A` **exactly** for every `╧å Γêê ╬ô`, not just up to a unit: this is a
constructive instance of Corollary 3.6 with `w=1`, and it requires no
search. A second free trick: since every catalog automorphism is a
`Z`-linear map on the exponent lattice and `bar = ΓêÆId` is central in
`GL_2(Z)`, `╧åΓêÿbar = barΓêÿ╧å` always ΓÇö so setting `B := bar(A)` simultaneously
gives (a) the same ring automorphism group on `B` and (b) a Hadamard-fold
stabilizer symmetry (`bar` + swap), for free. Every claim produced this way
is then independently re-verified with the *same* detection routines used
on the forward-problem benchmarks (`verify_group_invariance`), so the
"proof by construction" is never taken on faith.

## Forward-problem benchmark codes

| Code | `(l,m)` | `A` | `B` | Regime (Thm 2.1) | Source |
|---|---|---|---|---|---|
| `[[144,12,12]]` gross | `(12,6)` | `x┬│+y+y┬▓` | `y┬│+x+x┬▓` | even├ùeven, `s_l=2,s_m=1` | Bravyi et al., arXiv:2308.07915 |
| `[[72,12,6]]` BB6 | `(6,6)` | `x┬│+y+y┬▓` | `y┬│+x+x┬▓` | even├ùeven, `s_l=s_m=1` | same |
| `[[18,4,4]]` 6.6.6 color code | `(3,3)` | `x+1+y┬▓` | `y+1+x┬▓` | odd├ùodd, semisimple/PIGA | Wang et al., arXiv:2505.09684; errorcorrectionzoo.org/c/stab_18_4_4 |
| `[[16,4,4]]` weight-8 self-dual BB | `(2,4)`* | `1+y+y┬│+xy┬│` | `1+y+y┬│+xy` | even├ùeven, `s_l=1,s_m=2` | Liang & Chen, arXiv:2510.05211 (twisted-torus reduction, see below) |

All four are verified from scratch: `H_X H_Z^T = 0` (CSS), `n = 2lm`,
`k = n ΓêÆ rank(H_X) ΓêÆ rank(H_Z)` over GF(2) exactly matches the published
`k`, and ΓÇö for the two small codes ΓÇö brute-force minimum distance over
`ker(H_X)/rowspace(H_Z)` reproduces the published `d = 4` exactly.

### A note on `[[16,4,4]]`

errorcorrectionzoo's own `[[16,4,4]]` "symplectic-double" code is a
*different* object ΓÇö a concatenation of the `[[4,2,2]]` code with its
symplectic double ΓÇö and is not natively presented as a straight
`R_{l,m}`-type BB code, so it doesn't fit this paper's ambient-ring
framework directly. Instead we use a genuinely-BB, same-parameter code:
Liang & Chen's weight-8 self-dual BB code on a **twisted torus**,
`f(x,y) = 1+x+y+yΓü╗┬╣` with lattice basis vectors `a1=(0,4)`, `a2=(2,2)`
(i.e. relations `yΓü┤=1`, `x┬▓y┬▓=1`). `bb_ring.reduce_twisted_torus` finds the
unimodular exponent-coordinate change `V=[[1,-1],[0,1]]` (via a 2├ù2 Smith
normal form) that turns this into the standard rectangular presentation
`(l,m)=(2,4)` with `A=1+y+y┬│+xy┬│`, `B=1+y+y┬│+xy = bar(A)`; the code
construction confirms `n=16, k=4, d=4` exactly, matching the published
parameters. This is a useful side benefit of the toolkit: any BB-family
code presented on a non-rectangular ("twisted") torus can be mechanically
reduced to this paper's `R_{l,m}` presentation, which is a prerequisite
for applying the CRT machinery of Sec. 2.2 at all.

## Forward-problem results (summary)

Running `forward_benchmarks.py` scans the **genuine, BFS-closed** ring-automorphism
group (see "A correction" below) and the full `gblock` catalog against each code:

| Code | \|Aut group\| scanned | ring-level hits (no-swap / swap) | full stabilizer hits |
|---|---|---|---|
| gross `[[144,12,12]]` | 384 | 1 / 0 | 2 |
| BB6 `[[72,12,6]]` | 288 | 1 / 1 | 4 |
| color code `[[18,4,4]]` | 48 | 2 / 2 | 10 |
| `[[16,4,4]]` | 8 | 2 / 2 | 10 |

The "1/0" for the gross code means: among all 384 genuinely distinct ring
automorphisms in the catalog, only the identity preserves it as a pure code
automorphism (Cor. 3.6, no swap), and none do with an `A<->B` swap. This was
checked independently, outside the scan machinery, by testing `theta_x`,
`theta_y`, `iota`, and each of the 8 nontrivial unit multipliers one at a
time directly ΓÇö all fail, confirming the code's ring-level symmetry really
is minimal. The color code and `[[16,4,4]]`, by contrast, each have one
nontrivial ring automorphism beyond the identity (in addition to the
identity itself) both with and without the swap.

### A correction: the catalog must be a genuine group, not a template

An earlier version of this toolkit built the ring-automorphism catalog from
a **fixed composition template** ΓÇö `multiplier Γêÿ fold Γêÿ shear`, optionally
prefixed by `tau` once when `l=m` ΓÇö rather than the actual subgroup of
`Sym(Z_l x Z_m)` generated by those pieces. Two consequences, found and
fixed while working through exactly what algebraic object the scan checks:

1. **Under-coverage.** The template can reach at most one `tau`, applied
   first, so it misses every automorphism that requires *interleaving* `tau`
   with a shear or another `tau` (this only bites when `l=m`). For the BB6
   code's ambient ring `(l,m)=(6,6)`, the template's distinct permutations
   numbered 48; the true subgroup generated by the same primitive pieces has
   288 elements. The `[[18,4,4]]` color code's ring, `(3,3)`, was similarly
   short (24 vs. the true 48). Codes with `l != m` (gross, `[[16,4,4]]`)
   were *not* affected by this specific gap.
2. **A missing generator family entirely.** The outline's shear entry reads
   "`x -> x y^{sm/d}` (**and transposes**)" ΓÇö the transposed family
   `y -> y x^{s ell/d}` (`shear_family_transpose` in `bb_ring.py`) was
   implemented but never wired into the catalog, for *any* `(l,m)`.
3. **Heavy internal redundancy.** The template also *over-counted*: many
   different `(multiplier, fold, shear)` triples compose to the identical
   permutation, so the same automorphism was reported as several different
   "hits" under different names (e.g. "theta_x-containing", "pure
   multiplier") purely because of which template slot happened to produce
   it ΓÇö this is why the numbers above are substantially smaller than an
   earlier draft of this toolkit reported for the same four codes.

The fix: `bb_ring.full_catalog` now calls `bb_ring.generate_group`, a BFS
closure over the primitive generators (all multipliers, `theta_x, theta_y,
iota`, *both* shear families, and `tau` when `l=m`) that is closed under
composition by construction and de-duplicates by the actual resulting
permutation (not by name) ΓÇö see "Shared structure" below for why this
routine is also exactly what the inverse-design code needed, and now uses
verbatim.



## Inverse-problem worked examples (summary)

`inverse_design.py` builds and *independently verifies* 8 codes:

1. **Single symmetry ΓÇö multiplier** `╧ê_(2,2)` on `(7,3)`: `[[42,6,d]]`.
2. **Single symmetry ΓÇö partial fold** `╬╕x` on `(5,4)`: `[[40,8,d]]`.
3. **Single symmetry ΓÇö full fold** `╬╣` on `(9,5)`, with the free
   `B:=bar(A)` trick also producing a Hadamard-fold stabilizer symmetry:
   `[[90,2,d]]`.
4. **Single symmetry ΓÇö shear** `xΓåªxy┬│` on `(12,6)` (the outline's own
   example, `d=gcd(12,6)=6`): `[[144,16,d]]`.
5. **Single symmetry ΓÇö transpose** `╧ä` on `(5,5)`, mirroring the real
   `[[18,4,4]]` code's `B(x,y)=A(y,x)` structure: `[[50,2,d]]`.
6. **Multiple symmetries at once** ΓÇö the order-12 subgroup generated by
   `Γƒ¿mult(2,4), ╬╕xΓƒ⌐` on `(9,5)`, all 12 elements independently verified as
   code automorphisms of a single code: `[[90,74,d]]`.
7. **Variation ΓÇö ring + stabilizer symmetry together** on `(8,6)` (even├ùeven):
   a multiplier subgroup plus the free bar/Hadamard trick simultaneously;
   40 total full-stabilizer symmetries found by the scan.
8. **Completing the trichotomy ΓÇö mixed regime** (`l` even, `m` odd) on
   `(8,5)`, not present among the four fixed benchmark codes: `[[80,16,d]]`.

Every example is checked two ways: (a) by construction (orbit-closure
guarantees `╧å(A)=A,╧å(B)=B` exactly), and (b) by independently re-running
the same `test_code_automorphism` / `scan_stabilizer_symmetries` routines
used on the forward-problem benchmarks ΓÇö i.e. the construction is not
trusted blindly.

## The real CRT machinery (`bb_crt.py`, `bb_matching.py`)

This implements Sec. 2.2ΓÇô3.2's own construction, not a substitute for it.

**Factorization (eq. 5).** `factor_odd_cyclotomic` factors `x^Γäô'-1` over F2
(`Γäô'` = odd part of `Γäô`) via `galois`'s CantorΓÇôZassenhaus-based factorer,
cross-checked against the outline's own 2-cyclotomic-coset degree formula
(no factorization algorithm needed for the *degrees* ΓÇö e.g. mod 7, cosets
`{0},{1,2,4},{3,6,5}` predict degrees `1,3,3`, matching exactly).

**Hensel lifting (Lemma 2.2/B.1).** `hensel_lift_root` Newton-iterates
`╬▒ ΓåÉ ╬▒ ΓêÆ g(╬▒)/g'(╬▒)` inside the chain ring `F2[x]/Γƒ¿g^{2^s}Γƒ⌐`, doubling
precision each step. Hand-verified independently of the code: for
`g=x┬▓+x+1`, `s=1` (`M=g┬▓=xΓü┤+x┬▓+1`), since `g'(x)=1` (constant, char. 2) one
Newton step gives `╬▒=x┬▓+1`, and direct substitution confirms
`g(╬▒)=xΓü┤+x┬▓+1=MΓëí0`, with `╬▒ΓêÆx=x┬▓+x+1=g(x)` exactly as Lemma 2.2 requires.

**Component construction (Theorem 2.4/Lemma 2.3).** `build_components`
works in `GF(2^D)` (via `galois`) to find root pairs and their Frobenius
orbits. Reproduces the outline's own worked example exactly: for `(Γäô,m)=(12,6)`,
Sec. 2.2's Example 1 states *"C has five elements ... with residue fields
F2, F4, F4, F4, F4"* ΓÇö `build_components(12,6)` returns exactly 5 components
with exactly these residue field sizes.

**`k` cross-check.** `k = 2┬╖╬ú_c dim(J_c/Γƒ¿A_c,B_cΓƒ⌐)` (Sec. 2.3.1) computed
via the CRT component sum matches `k = n ΓêÆ rank(H_X) ΓêÆ rank(H_Z)` (the
already-validated GF(2)-rank ground truth) **exactly on all four benchmark
codes** ΓÇö the first cross-validation of the CRT pipeline, before it's used
for anything else.

**The real Algorithm 3.12.** `test_code_automorphism_crt` implements steps
1, 3ΓÇô6 exactly as specified: local expansion, Prop. 3.10's support/Jacobian-rank
prefilter (steps 3ΓÇô4, genuinely rejecting candidates before the expensive
solve), and the local matching-equation solve (step 5, linear algebra over
each `J_c`'s F2-coordinates, searching the affine solution space for a
*unit* solution, not just *any* solution). Step 5's `T*_╧ê(A_╧â(c))` is
computed by local-expanding the already-globally-transformed `╧ê(A)` at the
*original* component `c` ΓÇö valid by Lemma 3.8's pullback identity
`╧ü_c(╧ê(A)) = T*_╧ê(╧ü_╧â(c)(A))`, sidestepping ever building the transport
isomorphism `T*_╧ê` explicitly, and reusing the same `ring.apply_auto` call
Algorithm F1 already makes.

**Step 2 (Borel prune) is deliberately not implemented.** An attempted
translation of Prop. 2.8's restriction into a check on ╧ê's linear part
produced false negatives (rejecting genuine automorphisms) on the
`[[16,4,4]]` cross-check ΓÇö getting the direction right (which of `u,v` may
acquire a component of the other depends on which of `s_Γäô,s_m` is larger)
needed more care than a first attempt gave confidently. Steps 3ΓÇô6 are fully
implemented and cross-validated; step 2 is correctly *absent* rather than
shipped silently wrong.

**Cross-validation.** Every one of the **1456** `(╧å,swap)` candidates
tested across all four benchmark codes (the full genuine-BFS-closure
catalog ├ù 2 swap options) was tested both ways ΓÇö the row-space method and
the real Algorithm 3.12 ΓÇö and **the verdicts agree in all 1456/1456
cases** (`test_matching_crosscheck.py`). On the gross code specifically,
196/768 candidates are rejected by the cheap steps 3ΓÇô4 prefilter before the
expensive per-component solve ΓÇö genuine use of `(A,B)`'s algebraic
structure to prune, which the row-space method structurally cannot do.

**Avoiding brute force inside step 5 itself.** Steps 3ΓÇô4 reject candidates
outright, but many candidates *pass* that cheap filter and still fail the
expensive step-5 solve (571/768 for the gross code) ΓÇö Prop. 3.10's
conditions are necessary, not sufficient, so this residual cost isn't a
missed pruning opportunity so much as an inherent gap the paper itself
flags. What *is* avoidable: `solve_matching_at_component` used to always
set up and Gaussian-eliminate a `2┬╖D┬╖K1┬╖K2`-unknown linear system, even
though whenever `A_c` (or `B_c`) has a nonzero constant term it is *already
a unit of the whole local ring `J_c`* (not just its residue field) ΓÇö `J_c`
is local, so anything outside the maximal ideal is invertible ΓÇö and
`╧ë┬╖A_c=X1` then has the **unique** closed-form solution `╧ë = X1┬╖A_cΓü╗┬╣`,
no search needed. `jc_inverse` computes this full ring inverse via Newton
iteration (`b ΓåÉ b┬╖(2ΓêÆa┬╖b)`, doubling the correct number of `u,v`-adic
terms each step ΓÇö finite, since `Γƒ¿u,vΓƒ⌐` is nilpotent), replacing the
general linear solve with `O(log(K1K2))` ring multiplications wherever it
applies (the majority of components for our benchmark codes ΓÇö see the
worked example above where 3 of gross code's 5 components qualify).
This is a **real, cross-validated ~20% speedup** (1456-candidate
cross-check: 31s ΓåÆ 25s) with zero change to any accept/reject verdict.

An earlier version of this optimization inverted only the *constant term*
(a single residue-field division) rather than the full ring element ΓÇö
this is wrong: the constant term pins down `╧ë`'s residue-field image, not
its nilpotent-direction components, so it silently produced an incomplete
candidate that failed verification on cases where a genuine nontrivial
unit exists (caught immediately by `bb_inverse_crt.py`'s three-way check,
which is exactly why that check exists ΓÇö a construction and its verifier
disagreeing is the whole point).

### Can algebraic structure rule out more candidates before the expensive solve?

Prop. 3.10's own necessary conditions (support, Jacobian *rank*) are
coarser than they need to be. A genuinely stronger, still-cheap condition
falls straight out of the HasseΓÇôTaylor jet: at a *supported* component
(both `A_c,B_c` have zero constant term), the matching equation's
**linear-order** part depends only on `╧ë`'s own constant term `╧ëΓéÇ`
(`╧ë`'s higher-order terms can only multiply against `A_c,B_c`'s
linear-or-higher terms, landing at quadratic order or above ΓÇö verified
directly, not just asserted, before being trusted). So
`╧ëΓéÇ┬╖(A_u,A_v) = (X1_u,X1_v)`, `╧ëΓéÇ┬╖(B_u,B_v) = (X2_u,X2_v)` is a
single-scalar consistency check (dimension `D`, not `D┬╖K1┬╖K2`), and it is
**strictly finer than rank comparison**: two Jacobians can have equal rank
while still being inconsistent under every possible scalar, which rank
alone can't see.

`bb_crt.linear_order_prefilter` implements this as step "4.5", between the
existing Prop. 3.10 checks and the expensive step-5 solve. Before trusting
it anywhere: stress-tested for **soundness** against 1637 random `J_c`
instances (must never reject a case `solve_matching_at_component`
independently confirms *does* have a solution) ΓÇö **0 violations**. Then
wired into the real pipeline and cross-validated again: **1456/1456
candidates still agree** with the row-space oracle, no change to any
verdict.

Its actual impact, reported honestly rather than oversold: on
`[[16,4,4]]`, it catches 8 of the 12 previously-expensive candidates
cheaply. On the other three benchmark codes, it catches **zero**
additional cases beyond what rank-matching already caught ΓÇö their
remaining hard candidates are consistent at first order and only fail at
second order or higher. This is a real, verified capability, not a
universal win ΓÇö a concrete instance of the general principle it
illustrates: the matching equation admits a natural **filtration** by
truncation order (0th order = support, 1st order = this filter, 2nd order
and beyond = increasingly strong but increasingly expensive-to-derive
necessary conditions, up to the full solve at "infinite" order). Going to
2nd order is possible in principle but couples in `╧ë`'s own linear-order
terms (no longer a single scalar), meaningfully raising implementation
risk for a benefit that, on this evidence, may not materialize uniformly
across codes ΓÇö not attempted here.

Two other genuinely-different angles were considered and are worth
recording even though not implemented:

- **A discrete-log-based restriction on the multiplier sub-family.**
  Multipliers act on a component's root pair by exponentiation,
  `(╬▒,╬▓) Γåª (╬▒^{jx},╬▓^{jy})`; since `GF(2^D)^├ù` is cyclic, "which `jx`
  sends a specific `╬▒` to a specific target `╬▒'`" is a *modular equation*
  (a discrete log), not something to search over. This could shrink the
  multiplier portion of the catalog scan from testing every unit `jx`
  individually to solving directly for the compatible ones ΓÇö genuinely
  different in kind from steps 3ΓÇô4.5 (which prune *after* enumerating
  candidates; this would avoid enumerating incompatible multipliers at
  all). Not implemented; the practical payoff is small at our catalog
  sizes (Γëñ384) but would matter more for much larger unit groups.
- **Lemma 3.7 rules out a cross-component shortcut, not just fails to
  suggest one.** "No compatibility condition links different components:
  any family `(╧ë_c)_c` assembles into a global unit, since
  `R^├ù Γëà ΓêÅ J_c^├ù`" ΓÇö i.e. there is provably nothing to gain from trying
  to couple components together; solving each independently (as steps 1ΓÇô6
  already do) is not a missed opportunity, it's the complete picture.

## Nontrivial-unit inverse design (`bb_inverse_crt.py`)

`inverse_design.py`'s orbit-closure trick only ever achieves `w=1`
(Corollary 3.6 with the *trivial* unit), because it works by making `A`'s
support set literally `╧ê`-invariant. Corollary 3.6 permits *any* unit
`u Γêê R_{l,m}^├ù`, and `bb_inverse_crt.py` reaches genuinely nontrivial ones
via the **twisted-orbit-sum construction**:

> Fix `╧ê Γêê Aut(R_{l,m})` and a monomial unit `w = x^a y^b` with `╧ê(w)=w`
> (i.e. `(a,b)` is a fixed point of ╧ê's linear action on `Z_Γäô├ùZ_m`). Let
> `(i_0,j_0), ΓÇª, (i_{L-1},j_{L-1})` be the ╧ê-orbit of a seed point, and
> suppose `w^L = 1`. Then `A := ╬ú_{k=0}^{L-1} w^{-k}┬╖x^{i_k}y^{j_k}`
> satisfies `╧ê(A) = w┬╖A` **exactly**.

(Proved by direct computation: `╧ê` is multiplicative on monomials, so
summing `╧ê(w^{-k}x^{i_k}y^{j_k}) = w^{-k}x^{i_{k+1}}y^{j_{k+1}}` over the
orbit and re-indexing gives `╧ê(A) = w┬╖A + (w+w^{1-L})x^{i_0}y^{j_0}`, and
the correction term vanishes exactly when `w^L=1`.) This was hand-verified
numerically (`╧ê=╬╕x` on `R_{5,6}`, `A=x+xΓü┤y┬│`, `w=y┬│`) before being coded.

Four worked examples (`╬╕x`, a multiplier, a shear, the full fold, one per
catalog family, across four different `(Γäô,m)`), each checked **three
independent ways**, all agreeing in every case:

| ╧ê | (Γäô,m) | w (order) | code |
|---|---|---|---|
| `╬╕x` | (5,6) | `y┬│` (2) | `[[60,6,d]]` |
| `mult(4,1)` | (9,5) | `x┬│` (3) | `[[90,60,d]]` |
| shear `xΓåªxy┬│` | (12,6) | `xΓü╢` (2) | `[[144,72,d]]` |
| `╬╣` (full fold) | (8,5) | `xΓü┤` (2) | `[[80,4,d]]` |

1. **Direct algebra**: `╧ê(A)` (via `ring.apply_auto`) vs. `w┬╖A` (via
   `ring.mul`, ordinary circular convolution) ΓÇö checks the construction's
   own claim, not just its consequence.
2. **The row-space oracle** ΓÇö agnostic to `w` entirely, confirms *some*
   unit exists.
3. **The real Algorithm 3.12** ΓÇö additionally confirms `w`'s own local
   expansion is a unit at every supported component, i.e. `w` restricted
   to each component really is a valid witness, not merely that some
   witness exists.

## Known limitations / honest scope

- The Borel prune (Algorithm 3.12 step 2, Prop. 2.8/Thm. 3.11) is not
  implemented, for the concrete reason given above (false negatives on
  cross-validation); steps 1, 3ΓÇô6 are fully implemented and validated.
- The nontrivial-unit inverse-design construction only reaches *monomial*
  units `w=x^a y^b`. The full unit group `R_{l,m}^├ù Γëà ΓêÅ_c J_c^├ù` is
  generally much larger (e.g. `1+u` for nilpotent `u` is a unit but not a
  monomial in the even/mixed regime) ΓÇö designing codes realizing a
  prescribed *non-monomial* unit is not attempted.
- `╧ê_L Γëá ╧ê_R` (Sec. 3.3) is out of scope, matching the outline's own
  stated restriction through Sec. 3.
- Minimum distance is only computed by brute force for `n Γë▓ 26`; the two
  large codes' published distances (`d=12`, `d=6`) are taken from the
  literature, not re-derived here.
- The `gblock` catalog covers exactly the six matrices given explicitly in
  Sec. 3.1.2ΓÇô3.1.4 (eqs. 34ΓÇô37 plus identity); "combining gate types"
  (Sec. 3.1.5) and monomial-offset CX variants are not enumerated.
  Algorithm 3.12's local solve is implemented only for the `n=1`
  (Corollary 3.6) case, not the full `n`-generated stabilizer module of
  Theorem 3.2.
