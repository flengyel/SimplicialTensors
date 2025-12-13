# petersen_coset_enumeration.sage
#
# Goal: Enumerate labelings of the Petersen graph modulo automorphisms,
# i.e. compute a right transversal for Aut(P) in S_10, and use it to compute
# the full rank distribution of the mod-2 adjacency boundary:
#
#   ∂T := sum_{k=0}^{9} d_k(T)  in M_9(F_2),
#   where (d_k T)[u,v] = T[δ_k(u), δ_k(v)] and δ_k skips k.
#
# This reduces 10! = 3,628,800 labelings to 10!/|Aut(P)| = 30,240 coset reps
# since Aut(P) has order 120.

from sage.all import *

def delta(k, idx):
    # coface δ_k: [N-2] -> [N-1] skipping k (Sage indices 0..)
    return idx if idx < k else idx + 1

def face_map_adj(T, k):
    # T is an NxN matrix over GF(2), returns (N-1)x(N-1)
    N = T.nrows()
    dim = N - 1
    out = matrix(GF(2), dim, dim)
    for u in range(dim):
        ru = delta(k, u)
        for v in range(dim):
            rv = delta(k, v)
            out[u, v] = T[ru, rv]
    return out

def boundary_mod2(T):
    N = T.nrows()
    dim = N - 1
    B = zero_matrix(GF(2), dim, dim)
    for k in range(N):
        B += face_map_adj(T, k)
    return B

def permute_adjacency(T, perm):
    # perm is a list of length N giving new order of vertices (0-based).
    # Return P T P^T via simultaneous row/col permutation.
    N = T.nrows()
    return matrix(GF(2), [[T[perm[i], perm[j]] for j in range(N)] for i in range(N)])

# --- Build Petersen graph and its adjacency matrix over GF(2) ---
P = graphs.PetersenGraph()
# Ensure vertices are 0..9 in a fixed order:
P = P.relabel({v:i for i,v in enumerate(sorted(P.vertices()))}, inplace=False)
N = P.order()

T0 = matrix(GF(2), P.adjacency_matrix())

# --- Automorphism group and coset reps of S_N / Aut(P) ---
Aut = P.automorphism_group()   # subgroup of SymmetricGroup(N) (acts on 0..N-1)
print("N =", N)
print("|Aut(P)| =", Aut.order())

# Use GAP via libgap to get a right transversal efficiently.
# (This pattern is standard in Sage for large coset enumerations.)
Sgap = libgap.SymmetricGroup(N)
Agap = libgap(Aut)   # convert Sage PermutationGroup -> GAP group

cosets = Sgap.RightCosets(Agap)
reps_gap = [c.Representative() for c in cosets]
print("Number of right cosets =", len(reps_gap))
print("Expected =", factorial(N)//Aut.order())

# Convert a GAP permutation into a 0-based python list perm[0..N-1]
def gap_perm_to_list(g):
    # GAP permutations act on {1..N}; convert to list of images of 1..N, then shift to 0..N-1
    imgs = [Integer(g.Image(i)) for i in range(1, N+1)]
    return [i-1 for i in imgs]

# --- Compute rank distribution over all labelings using coset reps ---
counts = {}
for g in reps_gap:
    perm = gap_perm_to_list(g)
    Tp = permute_adjacency(T0, perm)
    Bp = boundary_mod2(Tp)
    r = Bp.rank()
    counts[r] = counts.get(r, 0) + 1

# Each coset corresponds to Aut.order() permutations with the same labeled adjacency matrix.
weighted = {r: c * Aut.order() for r, c in counts.items()}

print("\nUnweighted counts over coset reps (sum = #cosets):")
print(sorted(counts.items()))
print("\nWeighted counts over all permutations in S_N (sum = N!):")
print(sorted(weighted.items()))
print("\nCheck totals:", sum(weighted.values()), "vs", factorial(N))
