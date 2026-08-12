import numpy as np

# Twisted torus for [[16,4,4]] self-dual BB code (Liang & Chen, arXiv:2510.05211):
# f(x,y) = 1 + x + y + y^-1, lattice vectors a1=(0,4), a2=(2,2)
# Reduce to standard R_{ell,m} presentation via unimodular column transform V
# new coords (a,b) = (i,j) . V,  V = [[1,-1],[0,1]]  (det=1)

V = np.array([[1,-1],[0,1]])
a1 = np.array([0,4])
a2 = np.array([2,2])
print("a1 V =", a1 @ V)   # should be (0,4) -> relation b=0 mod 4
print("a2 V =", a2 @ V)   # should be (2,0) -> relation a=0 mod 2

f_terms = [(0,0),(1,0),(0,1),(0,-1)]
new_f_terms = [tuple(np.array(t) @ V) for t in f_terms]
print("f terms ->", new_f_terms)

# bar(f) terms (x^-1,y^-1 substitution)
barf_terms = [(-i,-j) for (i,j) in f_terms]
new_barf_terms = [tuple(np.array(t) @ V) for t in barf_terms]
print("bar(f) terms ->", new_barf_terms)

ell, m = 2, 4
def reduce_mod(t):
    return (t[0] % ell, t[1] % m)

A_terms = sorted(set(reduce_mod(t) for t in new_f_terms))
B_terms = sorted(set(reduce_mod(t) for t in new_barf_terms))
print("A terms mod (2,4):", A_terms)
print("B terms mod (2,4):", B_terms)
