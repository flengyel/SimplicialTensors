import networkx as nx
import numpy as np
import itertools
from collections import Counter
import time

# ==========================================
# 1. CORE MATH: DSTM Boundary (Mod 2)
# ==========================================

def boundary_rank_mod2(T):
    """
    Computes rank_GF2( sum_k d_k(T) ).
    d_k(T) is the principal submatrix deleting index k.
    """
    N = T.shape[0]
    dim = N - 1
    # B accumulates the sum (XOR) of faces
    B = np.zeros((dim, dim), dtype=np.uint8)
    
    for k in range(N):
        # Create face k by deleting row/col k
        # Using boolean masking is fast
        mask = np.ones(N, dtype=bool)
        mask[k] = False
        face = T[mask][:, mask]
        B ^= face
        
    # GF(2) Rank via Gaussian Elimination
    A = B.copy()
    rows, cols = A.shape
    r = 0
    rank = 0
    for c in range(cols):
        if r >= rows: break
        
        # Find pivot
        pivot_rows = np.where(A[r:, c] == 1)[0] + r
        if len(pivot_rows) == 0:
            continue
            
        pivot = pivot_rows[0]
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            
        # Eliminate rows below
        # Vectorized XOR
        pivot_vec = A[r]
        rows_to_elim = np.where(A[r+1:, c] == 1)[0] + r + 1
        if len(rows_to_elim) > 0:
            A[rows_to_elim] ^= pivot_vec
            
        r += 1
        rank += 1
    return rank

# ==========================================
# 2. INVARIANT CALCULATOR
# ==========================================

def compute_rank_spectrum(G, name):
    """
    Computes the multiset of ranks over ALL N! permutations.
    """
    A = nx.to_numpy_array(G, dtype=np.uint8)
    N = G.number_of_nodes()
    
    print(f"Analyzing {name} (N={N})...")
    ranks = []
    
    # Iterate over all permutations
    # For N=6, 720 iters (instant).
    # For N=10, 3.6M iters (approx 60-90s).
    for p in itertools.permutations(range(N)):
        # Apply permutation P A P^T
        # Equivalent to reordering rows/cols
        Ap = A[np.ix_(p, p)]
        ranks.append(boundary_rank_mod2(Ap))
        
    return Counter(ranks)

# ==========================================
# 3. EXPERIMENT: Cospectral Pair
# ==========================================

def run_cospectral_experiment():
    print("\n--- EXPERIMENT A: Cospectral Split (N=6) ---")
    print("Searching Graph Atlas for non-isomorphic cospectral pair...")
    
    # Load 6-vertex graphs from Atlas
    # Note: graph_atlas_g() returns graphs with varied N. Filter for N=6.
    atlas = nx.graph_atlas_g()
    graphs_n6 = [g for g in atlas if len(g) == 6]
    
    # Group by Spectrum
    spectra_map = {}
    pair = None
    
    for i, g in enumerate(graphs_n6):
        # Compute eigenvalues (rounded to avoid float drift)
        evals = np.linalg.eigvalsh(nx.to_numpy_array(g))
        sig = tuple(sorted(np.round(evals, 5)))
        
        if sig in spectra_map:
            prev_g = spectra_map[sig]
            if not nx.is_isomorphic(g, prev_g):
                print(f"Found cospectral pair! Atlas IDs unknown, using internal indices.")
                pair = (prev_g, g)
                break
        else:
            spectra_map[sig] = g
            
    if not pair:
        print("Could not find pair in Atlas subset.")
        return

    G1, G2 = pair
    
    # Compute DSTM Invariants
    dist1 = compute_rank_spectrum(G1, "Graph A")
    dist2 = compute_rank_spectrum(G2, "Graph B")
    
    print(f"\nResults:")
    print(f"Graph A Spectrum: {dict(sorted(dist1.items()))}")
    print(f"Graph B Spectrum: {dict(sorted(dist2.items()))}")
    
    if dist1 != dist2:
        print("\n[SUCCESS] The graphs are cospectral but have different DSTM Rank Spectra.")
        # Check for Rank 0
        has_zero_1 = 0 in dist1
        has_zero_2 = 0 in dist2
        if has_zero_1 != has_zero_2:
            print(f"[INSIGHT] Rank 0 Split: Graph {'A' if has_zero_1 else 'B'} admits trivial boundary; the other does not.")
            # Check counts against your paper data
            # Your G115: {0: 12, 2: 244, 4: 464}
            # Your G117: {2: 224, 4: 496}
            print("Compare these counts to G115/G117 data.")

# ==========================================
# 4. EXPERIMENT: Petersen Proof
# ==========================================

def run_petersen_proof():
    print("\n--- EXPERIMENT B: Petersen Exhaustive Proof (N=10) ---")
    P = nx.petersen_graph()
    A = nx.to_numpy_array(P, dtype=np.uint8)
    
    print("Checking ALL 3,628,800 permutations for Rank 0...")
    print("(This usually takes 1-2 minutes. Please wait.)")
    
    start = time.time()
    found_zero = False
    
    # Generator to avoid memory overhead
    perms = itertools.permutations(range(10))
    
    for i, p in enumerate(perms):
        Ap = A[np.ix_(p, p)]
        r = boundary_rank_mod2(Ap)
        
        if r == 0:
            found_zero = True
            print(f"Found Rank 0 at permutation {p}!")
            break
            
        if i % 500000 == 0 and i > 0:
            print(f"  Checked {i} permutations...", end='\r')
            
    dt = time.time() - start
    print(f"\nFinished in {dt:.2f}s.")
    
    if not found_zero:
        print("[THEOREM] The Petersen Graph has NO labeling with Rank 0 boundary.")
        print("          It is intrinsically DSTM-obstructed.")
    else:
        print("[RESULT] Petersen allows Rank 0.")

if __name__ == "__main__":
    run_cospectral_experiment()
    run_petersen_proof()