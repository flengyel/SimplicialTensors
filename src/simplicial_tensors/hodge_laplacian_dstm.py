"""
Hodge Laplacian for Diagonal Simplicial Tensor Modules

This module implements the proper combinatorial Hodge Laplacian for the
diagonal simplicial tensor module construction X(s; A) where s = (n_1, ..., n_k)
is the shape of a tensor of order k.

The key insight: a tensor of shape s has indices (i_1, ..., i_k) where each
i_a ∈ {0, ..., n_a - 1}. The diagonal face map d_j deletes index j from ALL
axes simultaneously. This gives a simplicial structure with dimension
n = min(s) - 1.

The Hodge Laplacian Δ_p = ∂_{p+1} ∂_{p+1}^* + ∂_p^* ∂_p decomposes the chain
space into:
    - Harmonic forms: ker(Δ_p) ≅ H_p (the homology)
    - Exact forms: im(∂_{p+1}) 
    - Coexact forms: im(∂_p^*)

The harmonic representatives are the "irreducible self-referential structures"
that cannot be further simplified.

Copyright (C) 2025 Florian Lengyel
License: GPL-3.0
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from functools import lru_cache
import itertools

# Optional PyTorch support
try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None


# =============================================================================
# Part 1: Index Set Geometry for DSTM
# =============================================================================

@dataclass(frozen=True)
class DSTMStructure:
    """
    Encapsulates the simplicial structure of a diagonal simplicial tensor module.
    
    For shape s = (n_1, ..., n_k):
    - Simplicial dimension n = min(s) - 1
    - Order k = len(s)
    - Index set at degree p: I_p = {(i_1, ..., i_k) : 0 ≤ i_a ≤ M_a(p)}
      where M_a(p) = n_a - 1 - n + p
    
    The diagonal face d_j removes index j from all axes.
    The diagonal degeneracy s_j duplicates index j in all axes.
    """
    shape: Tuple[int, ...]
    
    @property
    def order(self) -> int:
        """k = number of tensor axes"""
        return len(self.shape)
    
    @property
    def dimension(self) -> int:
        """n = simplicial dimension = min(shape) - 1"""
        return min(self.shape) - 1
    
    def M(self, a: int, p: int) -> int:
        """Upper bound for axis a at degree p: M_a(p) = n_a - 1 - n + p"""
        n = self.dimension
        return self.shape[a] - 1 - n + p
    
    def index_set(self, p: int) -> List[Tuple[int, ...]]:
        """
        Returns all valid indices at degree p.
        I_p = {(i_1, ..., i_k) : 0 ≤ i_a ≤ M_a(p) for all a}
        """
        if p < 0 or p > self.dimension:
            return []
        ranges = [range(self.M(a, p) + 1) for a in range(self.order)]
        return list(itertools.product(*ranges))
    
    def index_set_size(self, p: int) -> int:
        """Number of indices at degree p: ∏_a (M_a(p) + 1)"""
        if p < 0 or p > self.dimension:
            return 0
        size = 1
        for a in range(self.order):
            size *= (self.M(a, p) + 1)
        return size
    
    def is_valid_index(self, idx: Tuple[int, ...], p: int) -> bool:
        """Check if idx is valid at degree p"""
        if len(idx) != self.order:
            return False
        for a, i in enumerate(idx):
            if i < 0 or i > self.M(a, p):
                return False
        return True


# =============================================================================
# Part 2: Boundary Matrices
# =============================================================================

def build_boundary_matrix(structure: DSTMStructure, p: int) -> np.ndarray:
    """
    Constructs the boundary matrix ∂_p: C_p → C_{p-1}.
    
    The diagonal boundary is:
        ∂_p = Σ_{j=0}^{p} (-1)^j d_j
    
    where d_j acts by removing index j from each axis of the multi-index.
    
    The matrix B has shape (|I_{p-1}|, |I_p|), where:
    - Columns are indexed by I_p (source)
    - Rows are indexed by I_{p-1} (target)
    
    B[row, col] = (-1)^j if the index at col maps to the index at row via d_j
    """
    if p <= 0 or p > structure.dimension:
        # ∂_0 = 0 (no boundary of 0-simplices)
        # ∂_{n+1} = 0 (nothing above top dimension)
        rows = structure.index_set_size(p - 1)
        cols = structure.index_set_size(p)
        return np.zeros((max(rows, 1), max(cols, 1)), dtype=np.float64)
    
    I_p = structure.index_set(p)
    I_p_minus_1 = structure.index_set(p - 1)
    
    if not I_p or not I_p_minus_1:
        return np.zeros((1, 1), dtype=np.float64)
    
    # Create index-to-row/col mappings
    idx_to_col = {idx: c for c, idx in enumerate(I_p)}
    idx_to_row = {idx: r for r, idx in enumerate(I_p_minus_1)}
    
    B = np.zeros((len(I_p_minus_1), len(I_p)), dtype=np.float64)
    
    for col, source_idx in enumerate(I_p):
        # Apply each face d_j
        for j in range(p + 1):
            # d_j removes index j from each component
            # (i_1, ..., i_k) ↦ (i_1', ..., i_k') where i_a' = i_a if i_a < j, else i_a - 1
            target_idx = tuple(
                (i if i < j else i - 1) if i != j else -999  # -999 marks deletion
                for i in source_idx
            )
            
            # Check if any component was exactly j (those get deleted)
            # In diagonal action, we delete j from ALL indices simultaneously
            # So the target is: for each axis, if i_a >= j, decrement by 1
            # But i_a = j means that position "was" j and is now removed
            
            # Actually, the diagonal face map: d_j removes the j-th "slice"
            # from each axis. An index (i_1, ..., i_k) at degree p becomes
            # (i_1', ..., i_k') at degree p-1 where:
            #   i_a' = i_a      if i_a < j
            #   i_a' = i_a - 1  if i_a > j
            #   undefined       if i_a = j (this index is "deleted")
            
            # For the boundary, we only get a contribution if the source index
            # does NOT have any component equal to j (those are in ker(d_j))
            
            if j in source_idx:
                # This index has a component = j, so d_j(E_{source}) = 0
                continue
            
            # Compute target index
            target_idx = tuple(
                i if i < j else i - 1
                for i in source_idx
            )
            
            if target_idx in idx_to_row:
                row = idx_to_row[target_idx]
                sign = (-1) ** j
                B[row, col] += sign
    
    return B


def build_full_chain_complex(structure: DSTMStructure) -> Dict[int, np.ndarray]:
    """
    Builds all boundary matrices for the chain complex.
    
    Returns dict mapping p → ∂_p matrix.
    """
    n = structure.dimension
    boundaries = {}
    for p in range(n + 2):  # p = 0, 1, ..., n+1
        boundaries[p] = build_boundary_matrix(structure, p)
    return boundaries


# =============================================================================
# Part 3: Hodge Laplacian
# =============================================================================

def hodge_laplacian(structure: DSTMStructure, p: int, 
                    boundaries: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    """
    Computes the Hodge Laplacian at degree p:
    
        Δ_p = ∂_{p+1} ∂_{p+1}^* + ∂_p^* ∂_p
            = ∂_{p+1} ∂_{p+1}^T + ∂_p^T ∂_p
    
    (using the standard inner product where ∂^* = ∂^T)
    
    The "up-Laplacian" ∂_{p+1} ∂_{p+1}^T measures how p-chains participate
    in (p+1)-chains.
    
    The "down-Laplacian" ∂_p^T ∂_p measures the boundary structure of p-chains.
    
    Returns:
        Δ_p as a matrix of shape (|I_p|, |I_p|)
    """
    if boundaries is None:
        boundaries = build_full_chain_complex(structure)
    
    n = structure.dimension
    size_p = structure.index_set_size(p)
    
    if size_p == 0:
        return np.zeros((1, 1), dtype=np.float64)
    
    # Initialize Laplacian
    Delta = np.zeros((size_p, size_p), dtype=np.float64)
    
    # Up-Laplacian: ∂_{p+1} ∂_{p+1}^T
    if p + 1 in boundaries:
        d_up = boundaries[p + 1]
        if d_up.shape[0] == size_p:  # Check dimensions match
            Delta += d_up @ d_up.T
    
    # Down-Laplacian: ∂_p^T ∂_p
    if p in boundaries:
        d_down = boundaries[p]
        if d_down.shape[1] == size_p:  # Check dimensions match
            Delta += d_down.T @ d_down
    
    return Delta


def full_hodge_laplacians(structure: DSTMStructure) -> Dict[int, np.ndarray]:
    """
    Computes all Hodge Laplacians Δ_0, Δ_1, ..., Δ_n.
    """
    boundaries = build_full_chain_complex(structure)
    n = structure.dimension
    laplacians = {}
    for p in range(n + 1):
        laplacians[p] = hodge_laplacian(structure, p, boundaries)
    return laplacians


# =============================================================================
# Part 4: Spectral Analysis
# =============================================================================

@dataclass
class HodgeSpectrum:
    """Results of spectral analysis of a Hodge Laplacian."""
    degree: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    harmonic_dimension: int  # = Betti number β_p
    spectral_gap: float  # smallest nonzero eigenvalue
    
    @property
    def betti_number(self) -> int:
        return self.harmonic_dimension


def analyze_spectrum(Delta: np.ndarray, degree: int, 
                     tol: float = 1e-10) -> HodgeSpectrum:
    """
    Performs spectral analysis of a Hodge Laplacian.
    
    Returns eigenvalues, eigenvectors, harmonic dimension (Betti number),
    and spectral gap.
    """
    # Symmetrize to handle numerical errors
    Delta_sym = (Delta + Delta.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(Delta_sym)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Count zero eigenvalues (harmonic forms)
    harmonic_dim = np.sum(np.abs(eigenvalues) < tol)
    
    # Spectral gap: smallest nonzero eigenvalue
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) >= tol]
    spectral_gap = nonzero_eigs[0] if len(nonzero_eigs) > 0 else np.inf
    
    return HodgeSpectrum(
        degree=degree,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        harmonic_dimension=harmonic_dim,
        spectral_gap=spectral_gap
    )


def compute_betti_numbers(structure: DSTMStructure) -> List[int]:
    """
    Computes all Betti numbers β_0, β_1, ..., β_n via Hodge theory.
    
    β_p = dim(ker(Δ_p)) = dim(H_p)
    """
    laplacians = full_hodge_laplacians(structure)
    betti = []
    for p in range(structure.dimension + 1):
        spectrum = analyze_spectrum(laplacians[p], p)
        betti.append(spectrum.betti_number)
    return betti


# =============================================================================
# Part 5: Effective Resistance and Graph-Theoretic Quantities
# =============================================================================

def effective_resistance(Delta: np.ndarray, i: int, j: int,
                        tol: float = 1e-10) -> float:
    """
    Computes the effective resistance between indices i and j.
    
    R(i, j) = (e_i - e_j)^T Δ^+ (e_i - e_j)
    
    where Δ^+ is the Moore-Penrose pseudoinverse.
    
    This measures the "distance" between two indices in the simplicial
    structure, accounting for all possible paths.
    """
    n = Delta.shape[0]
    if i < 0 or i >= n or j < 0 or j >= n:
        raise ValueError(f"Indices {i}, {j} out of range for matrix of size {n}")
    
    if i == j:
        return 0.0
    
    # Compute pseudoinverse
    Delta_pinv = np.linalg.pinv(Delta, rcond=tol)
    
    # Compute R(i,j)
    e_diff = np.zeros(n)
    e_diff[i] = 1.0
    e_diff[j] = -1.0
    
    return float(e_diff @ Delta_pinv @ e_diff)


def effective_resistance_matrix(Delta: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Computes the full effective resistance matrix R[i,j] for all pairs.
    """
    n = Delta.shape[0]
    Delta_pinv = np.linalg.pinv(Delta, rcond=tol)
    
    # R[i,j] = Δ^+[i,i] + Δ^+[j,j] - 2*Δ^+[i,j]
    diag = np.diag(Delta_pinv)
    R = diag[:, None] + diag[None, :] - 2 * Delta_pinv
    
    return R


# =============================================================================
# Part 6: PyTorch Integration for Neural Networks
# =============================================================================

if HAS_TORCH:
    
    class HodgeLaplacianRegularizer:
        """
        Regularizer that penalizes weight matrices based on the Hodge Laplacian
        of their simplicial structure.
        
        The intuition: weights that respect the simplicial structure (small
        Laplacian action) are preferred. This encourages "topologically coherent"
        weight configurations.
        """
        
        def __init__(self, structure: DSTMStructure, degree: int = 1,
                     precompute: bool = True):
            """
            Args:
                structure: DSTM structure for the weight shape
                degree: which degree Laplacian to use (default: 1 for edges)
                precompute: whether to precompute the Laplacian matrix
            """
            self.structure = structure
            self.degree = degree
            
            if precompute:
                boundaries = build_full_chain_complex(structure)
                self._laplacian_np = hodge_laplacian(structure, degree, boundaries)
                self._laplacian = torch.tensor(
                    self._laplacian_np, dtype=torch.float32
                )
            else:
                self._laplacian = None
        
        @property
        def laplacian(self) -> Tensor:
            if self._laplacian is None:
                boundaries = build_full_chain_complex(self.structure)
                self._laplacian_np = hodge_laplacian(
                    self.structure, self.degree, boundaries
                )
                self._laplacian = torch.tensor(
                    self._laplacian_np, dtype=torch.float32
                )
            return self._laplacian
        
        def __call__(self, W: Tensor) -> Tensor:
            """
            Computes the Laplacian penalty: ||Δ W||^2
            
            W should be flattened to match the index set size.
            """
            L = self.laplacian.to(W.device)
            W_flat = W.flatten()
            
            if W_flat.shape[0] != L.shape[0]:
                raise ValueError(
                    f"Weight size {W_flat.shape[0]} doesn't match "
                    f"Laplacian size {L.shape[0]}"
                )
            
            Lw = L @ W_flat
            return torch.sum(Lw * Lw)
        
        def harmonic_projection(self, W: Tensor, tol: float = 1e-6) -> Tensor:
            """
            Projects W onto the harmonic subspace (kernel of Δ).
            
            This extracts the "topologically essential" part of the weights.
            """
            L_np = self._laplacian_np
            eigenvalues, eigenvectors = np.linalg.eigh(L_np)
            
            # Find harmonic eigenvectors (zero eigenvalues)
            harmonic_mask = np.abs(eigenvalues) < tol
            V_harmonic = eigenvectors[:, harmonic_mask]
            
            if V_harmonic.shape[1] == 0:
                return torch.zeros_like(W)
            
            V_h = torch.tensor(V_harmonic, dtype=W.dtype, device=W.device)
            W_flat = W.flatten()
            
            # Project: P_H(W) = V_H V_H^T W
            coeffs = V_h.T @ W_flat
            projected = V_h @ coeffs
            
            return projected.reshape(W.shape)
        
        def spectral_gap_penalty(self, W: Tensor) -> Tensor:
            """
            Penalty that encourages W to have components in low-eigenvalue
            (but nonzero) modes. These are the "soft" topological modes.
            """
            L = self.laplacian.to(W.device)
            W_flat = W.flatten()
            
            # Rayleigh quotient: (W^T Δ W) / (W^T W)
            numerator = W_flat @ (L @ W_flat)
            denominator = W_flat @ W_flat + 1e-8
            
            return numerator / denominator

    
    class HodgeFeedbackLinear(torch.nn.Linear):
        """
        Linear layer that incorporates Hodge Laplacian feedback.
        
        During forward pass, the weight is modified:
            W' = W + β * Δ^{-1} * ∂ W
        
        where Δ^{-1} is the pseudoinverse of the Laplacian.
        This "smooths" the weights along topological directions.
        """
        
        def __init__(self, in_features: int, out_features: int,
                     beta: float = 0.01, bias: bool = True):
            super().__init__(in_features, out_features, bias)
            self.beta = beta
            
            # Build DSTM structure for this weight shape
            self.structure = DSTMStructure((out_features, in_features))
            
            # Precompute Laplacian and its pseudoinverse
            boundaries = build_full_chain_complex(self.structure)
            self._laplacian_np = hodge_laplacian(self.structure, 1, boundaries)
            self._laplacian_pinv_np = np.linalg.pinv(self._laplacian_np)
            
            self.register_buffer(
                '_laplacian',
                torch.tensor(self._laplacian_np, dtype=torch.float32)
            )
            self.register_buffer(
                '_laplacian_pinv',
                torch.tensor(self._laplacian_pinv_np, dtype=torch.float32)
            )
        
        def forward(self, x: Tensor) -> Tensor:
            if self.beta == 0:
                return torch.nn.functional.linear(x, self.weight, self.bias)
            
            # Compute boundary of weight
            W_flat = self.weight.flatten()
            
            # For 2D weights viewed as 1-chains, the boundary goes to 0-chains
            # We use the Laplacian action as the feedback
            feedback = self._laplacian @ W_flat
            
            # Apply pseudoinverse to get smoothed correction
            correction = self._laplacian_pinv @ feedback
            
            # Modify weight
            W_corrected = W_flat - self.beta * correction
            W_new = W_corrected.reshape(self.weight.shape)
            
            return torch.nn.functional.linear(x, W_new, self.bias)
        
        def get_harmonic_component(self) -> Tensor:
            """Returns the harmonic (topologically essential) part of weights."""
            eigenvalues, eigenvectors = np.linalg.eigh(self._laplacian_np)
            harmonic_mask = np.abs(eigenvalues) < 1e-8
            V_h = torch.tensor(
                eigenvectors[:, harmonic_mask],
                dtype=self.weight.dtype,
                device=self.weight.device
            )
            
            if V_h.shape[1] == 0:
                return torch.zeros_like(self.weight)
            
            W_flat = self.weight.flatten()
            coeffs = V_h.T @ W_flat
            harmonic = V_h @ coeffs
            return harmonic.reshape(self.weight.shape)


# =============================================================================
# Part 7: Verification and Testing
# =============================================================================

def verify_chain_complex(structure: DSTMStructure) -> bool:
    """
    Verifies ∂_{p-1} ∘ ∂_p = 0 for all p.
    """
    boundaries = build_full_chain_complex(structure)
    n = structure.dimension
    
    for p in range(1, n + 1):
        d_p = boundaries[p]
        d_pm1 = boundaries[p - 1]
        
        # Check dimensions are compatible
        if d_pm1.shape[1] != d_p.shape[0]:
            print(f"Dimension mismatch at p={p}: ∂_{p-1} has {d_pm1.shape[1]} cols, "
                  f"∂_{p} has {d_p.shape[0]} rows")
            return False
        
        # Check ∂∂ = 0
        composition = d_pm1 @ d_p
        if not np.allclose(composition, 0, atol=1e-10):
            print(f"∂_{p-1} ∘ ∂_{p} ≠ 0!")
            print(f"Max error: {np.max(np.abs(composition))}")
            return False
    
    print("✓ Chain complex verified: ∂∂ = 0")
    return True


def verify_laplacian_symmetry(structure: DSTMStructure) -> bool:
    """
    Verifies that all Laplacians are symmetric.
    """
    laplacians = full_hodge_laplacians(structure)
    
    for p, Delta in laplacians.items():
        if not np.allclose(Delta, Delta.T, atol=1e-10):
            print(f"Δ_{p} is not symmetric!")
            return False
    
    print("✓ All Laplacians are symmetric")
    return True


def verify_hodge_decomposition(structure: DSTMStructure, p: int) -> bool:
    """
    Verifies the Hodge decomposition:
        C_p = ker(Δ_p) ⊕ im(∂_{p+1}) ⊕ im(∂_p^T)
    
    by checking dimensions sum correctly.
    """
    boundaries = build_full_chain_complex(structure)
    Delta = hodge_laplacian(structure, p, boundaries)
    
    d_p = boundaries[p]
    d_p_plus_1 = boundaries.get(p + 1, np.zeros((1, 1)))
    
    size_Cp = structure.index_set_size(p)
    
    # Dimensions of each component
    rank_d_p = np.linalg.matrix_rank(d_p)
    rank_d_p_plus_1 = np.linalg.matrix_rank(d_p_plus_1)
    
    eigenvalues = np.linalg.eigvalsh(Delta)
    dim_harmonic = np.sum(np.abs(eigenvalues) < 1e-10)
    
    # im(∂_{p+1}) has dimension = rank(∂_{p+1})
    # im(∂_p^T) has dimension = rank(∂_p^T) = rank(∂_p)
    # These should be orthogonal and together with harmonics span C_p
    
    expected_total = dim_harmonic + rank_d_p + rank_d_p_plus_1
    
    print(f"Degree {p}: |C_p| = {size_Cp}")
    print(f"  dim(ker Δ) = {dim_harmonic} (Betti number β_{p})")
    print(f"  dim(im ∂_{p+1}) = {rank_d_p_plus_1}")
    print(f"  dim(im ∂_{p}^T) = {rank_d_p}")
    print(f"  Sum = {expected_total}")
    
    if expected_total == size_Cp:
        print(f"  ✓ Hodge decomposition verified")
        return True
    else:
        print(f"  ✗ Dimension mismatch!")
        return False


# =============================================================================
# Part 7b: Relative Hodge Laplacian for Horn Filling
# =============================================================================

def compute_missing_indices(structure: DSTMStructure, j: int) -> List[Tuple[int, ...]]:
    """
    Computes the set of "missing indices" for horn Λ^n_j.
    
    An index (i_1, ..., i_k) at the top degree n is missing if it contains 
    ALL face indices except j. That is:
        {0, 1, ..., n} \ {j} ⊆ {i_1, ..., i_k}
    
    These are the indices whose values are not determined by the horn.
    """
    n = structure.dimension
    k = structure.order
    
    if k < n:
        return []  # Below threshold: no missing indices
    
    horn_faces = set(range(n + 1)) - {j}  # {0, ..., n} \ {j}
    
    missing = []
    for idx in structure.index_set(n):  # Top degree indices
        if horn_faces.issubset(set(idx)):
            missing.append(idx)
    
    return missing


def build_relative_boundary_matrix(structure: DSTMStructure, j: int, p: int) -> np.ndarray:
    """
    Builds the boundary matrix for the relative chain complex C_*(A, L_j).
    
    The relative complex quotients out the horn:
    - Generators at degree n are only the "missing indices"
    - Lower degrees inherit the quotient structure
    
    For practical purposes, we restrict to the missing indices at top degree.
    """
    if p != structure.dimension:
        # For now, only implement top degree
        return build_boundary_matrix(structure, p)
    
    n = structure.dimension
    missing_n = compute_missing_indices(structure, j)
    I_n_minus_1 = structure.index_set(n - 1)
    
    if not missing_n or not I_n_minus_1:
        return np.zeros((1, 1), dtype=np.float64)
    
    # Map missing indices to columns
    idx_to_col = {idx: c for c, idx in enumerate(missing_n)}
    idx_to_row = {idx: r for r, idx in enumerate(I_n_minus_1)}
    
    B = np.zeros((len(I_n_minus_1), len(missing_n)), dtype=np.float64)
    
    for col, source_idx in enumerate(missing_n):
        for face_idx in range(n + 1):
            if face_idx in source_idx:
                continue  # d_j(E_{source}) = 0 when j is a component
            
            target_idx = tuple(i if i < face_idx else i - 1 for i in source_idx)
            
            if target_idx in idx_to_row:
                row = idx_to_row[target_idx]
                sign = (-1) ** face_idx
                B[row, col] += sign
    
    return B


def relative_hodge_laplacian(structure: DSTMStructure, j: int) -> np.ndarray:
    """
    Computes the Hodge Laplacian on the relative chain complex C_n(A, L_j).
    
    Since this is at top degree n, there's no "up-Laplacian" contribution.
    The Laplacian is just ∂_n^T ∂_n restricted to missing indices.
    
    Returns:
        Δ_n^{rel} as a matrix over the missing indices
    """
    n = structure.dimension
    missing = compute_missing_indices(structure, j)
    
    if not missing:
        return np.zeros((1, 1), dtype=np.float64)
    
    # Build the restricted boundary matrix
    B = build_relative_boundary_matrix(structure, j, n)
    
    # Relative Laplacian: Δ = B^T B (no up-Laplacian at top degree)
    Delta = B.T @ B
    
    return Delta


@dataclass
class RelativeHodgeSpectrum:
    """Results for relative Hodge analysis at top degree."""
    horn_index: int
    missing_indices: List[Tuple[int, ...]]
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    kernel_dimension: int  # Should be len(missing_indices) for acyclic complexes
    spectral_gap: float


def analyze_relative_hodge(structure: DSTMStructure, j: int,
                           tol: float = 1e-10) -> RelativeHodgeSpectrum:
    """
    Performs complete Hodge analysis on the relative complex for horn j.
    """
    missing = compute_missing_indices(structure, j)
    Delta = relative_hodge_laplacian(structure, j)
    
    if Delta.shape[0] <= 1:
        return RelativeHodgeSpectrum(
            horn_index=j,
            missing_indices=missing,
            laplacian=Delta,
            eigenvalues=np.array([0.0]),
            eigenvectors=np.eye(1),
            kernel_dimension=1 if missing else 0,
            spectral_gap=np.inf
        )
    
    # Symmetrize and diagonalize
    Delta_sym = (Delta + Delta.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(Delta_sym)
    
    # Sort
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Kernel dimension
    kernel_dim = np.sum(np.abs(eigenvalues) < tol)
    
    # Spectral gap
    nonzero = eigenvalues[np.abs(eigenvalues) >= tol]
    gap = nonzero[0] if len(nonzero) > 0 else np.inf
    
    return RelativeHodgeSpectrum(
        horn_index=j,
        missing_indices=missing,
        laplacian=Delta,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kernel_dimension=kernel_dim,
        spectral_gap=gap
    )


def relative_effective_resistance(structure: DSTMStructure, j: int,
                                  idx1: Tuple[int, ...], idx2: Tuple[int, ...],
                                  tol: float = 1e-10) -> float:
    """
    Computes effective resistance between two missing indices in the
    relative Hodge structure.
    
    This measures the "topological distance" between two undetermined
    tensor entries when filling the horn.
    """
    missing = compute_missing_indices(structure, j)
    Delta = relative_hodge_laplacian(structure, j)
    
    if idx1 not in missing or idx2 not in missing:
        raise ValueError(f"Indices must be in the missing set")
    
    if idx1 == idx2:
        return 0.0
    
    i = missing.index(idx1)
    k = missing.index(idx2)
    
    Delta_pinv = np.linalg.pinv(Delta, rcond=tol)
    
    e_diff = np.zeros(len(missing))
    e_diff[i] = 1.0
    e_diff[k] = -1.0
    
    return float(e_diff @ Delta_pinv @ e_diff)


# =============================================================================
# Part 7c: Neural Network with Relative Hodge Structure
# =============================================================================

if HAS_TORCH:
    
    class RelativeHodgeRegularizer:
        """
        Regularizer based on the relative Hodge Laplacian.
        
        This penalizes weight configurations where the "missing" (topologically
        undetermined) components have large Laplacian action.
        
        The intuition: the network's weights should be "harmonically coherent"
        with respect to the simplicial structure. Large Laplacian values indicate
        weights that don't respect the topology.
        """
        
        def __init__(self, structure: DSTMStructure, horn_index: int = 0):
            self.structure = structure
            self.horn_index = horn_index
            
            self.missing = compute_missing_indices(structure, horn_index)
            self._laplacian_np = relative_hodge_laplacian(structure, horn_index)
            self._laplacian = torch.tensor(
                self._laplacian_np, dtype=torch.float32
            )
            
            # Precompute missing index positions for fast extraction
            all_indices = structure.index_set(structure.dimension)
            self._missing_positions = [all_indices.index(m) for m in self.missing]
        
        def extract_missing(self, W: Tensor) -> Tensor:
            """Extracts the missing components from a weight tensor."""
            W_flat = W.flatten()
            positions = torch.tensor(self._missing_positions, device=W.device)
            return W_flat[positions]
        
        def __call__(self, W: Tensor) -> Tensor:
            """Computes the relative Laplacian penalty."""
            W_missing = self.extract_missing(W)
            L = self._laplacian.to(W.device)
            Lw = L @ W_missing
            return torch.sum(Lw * Lw)
        
        def harmonic_energy(self, W: Tensor) -> Tensor:
            """
            Computes the harmonic energy: W^T Δ W.
            Lower values indicate more topologically coherent weights.
            """
            W_missing = self.extract_missing(W)
            L = self._laplacian.to(W.device)
            return W_missing @ (L @ W_missing)


# =============================================================================
# Part 8: Example Usage and Demonstrations
# =============================================================================

def demo_basic_usage():
    """Demonstrates basic usage of the Hodge Laplacian."""
    
    print("=" * 60)
    print("HODGE LAPLACIAN FOR DIAGONAL SIMPLICIAL TENSOR MODULES")
    print("=" * 60)
    
    # Example: 3×3 matrices
    shape = (3, 3)
    structure = DSTMStructure(shape)
    
    print(f"\nShape: {shape}")
    print(f"Order k = {structure.order}")
    print(f"Simplicial dimension n = {structure.dimension}")
    
    print(f"\nIndex sets:")
    for p in range(structure.dimension + 1):
        I_p = structure.index_set(p)
        print(f"  I_{p}: {len(I_p)} indices")
        if len(I_p) <= 10:
            print(f"       {I_p}")
    
    # Build and verify chain complex
    print("\n" + "-" * 40)
    verify_chain_complex(structure)
    verify_laplacian_symmetry(structure)
    
    # Compute Hodge Laplacians
    print("\n" + "-" * 40)
    print("Hodge Laplacians:")
    laplacians = full_hodge_laplacians(structure)
    
    for p, Delta in laplacians.items():
        spectrum = analyze_spectrum(Delta, p)
        print(f"\n  Δ_{p}: shape {Delta.shape}")
        print(f"    Eigenvalues: {np.round(spectrum.eigenvalues, 4)}")
        print(f"    β_{p} (Betti number) = {spectrum.betti_number}")
        print(f"    Spectral gap = {spectrum.spectral_gap:.4f}")
    
    # Verify Hodge decomposition
    print("\n" + "-" * 40)
    print("Hodge Decomposition:")
    for p in range(structure.dimension + 1):
        verify_hodge_decomposition(structure, p)
    
    return structure, laplacians


def demo_3d_tensor():
    """Demonstrates the Hodge Laplacian for 3D tensors."""
    
    print("\n" + "=" * 60)
    print("3D TENSOR EXAMPLE")
    print("=" * 60)
    
    shape = (3, 3, 3)
    structure = DSTMStructure(shape)
    
    print(f"\nShape: {shape}")
    print(f"Order k = {structure.order}")
    print(f"Simplicial dimension n = {structure.dimension}")
    
    # Betti numbers
    betti = compute_betti_numbers(structure)
    print(f"\nBetti numbers: {betti}")
    
    # This should give β_0 = 1, β_1 = ?, β_2 = ?
    # depending on the structure
    
    verify_chain_complex(structure)
    
    return structure


def demo_neural_network():
    """Demonstrates PyTorch integration (if available)."""
    
    if not HAS_TORCH:
        print("\nPyTorch not available. Skipping neural network demo.")
        return
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK INTEGRATION")
    print("=" * 60)
    
    # Create a Hodge-regularized linear layer
    layer = HodgeFeedbackLinear(10, 5, beta=0.1)
    
    # Random input
    x = torch.randn(32, 10)
    
    # Forward pass with Hodge feedback
    y = layer(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Get harmonic component of weights
    harmonic_W = layer.get_harmonic_component()
    print(f"Harmonic weight component norm: {harmonic_W.norm().item():.4f}")
    print(f"Full weight norm: {layer.weight.norm().item():.4f}")
    
    # Regularizer example
    structure = DSTMStructure((5, 10))
    regularizer = HodgeLaplacianRegularizer(structure, degree=1)
    
    penalty = regularizer(layer.weight)
    print(f"Laplacian penalty: {penalty.item():.4f}")


def demo_relative_hodge():
    """Demonstrates the relative Hodge structure for horn filling."""
    
    print("\n" + "=" * 60)
    print("RELATIVE HODGE STRUCTURE FOR HORN FILLING")
    print("=" * 60)
    
    shape = (3, 3, 3)
    structure = DSTMStructure(shape)
    n = structure.dimension
    
    print(f"\nShape: {shape}, dimension n = {n}")
    print(f"Testing all horns Λ^{n}_j for j = 0, ..., {n}")
    
    for j in range(n + 1):
        spectrum = analyze_relative_hodge(structure, j)
        
        print(f"\n  Horn Λ^{n}_{j}:")
        print(f"    |Missing indices| = {len(spectrum.missing_indices)}")
        print(f"    Laplacian shape: {spectrum.laplacian.shape}")
        print(f"    Eigenvalues: {np.round(spectrum.eigenvalues[:6], 3)}...")
        print(f"    Kernel dimension: {spectrum.kernel_dimension}")
        print(f"    Spectral gap: {spectrum.spectral_gap:.4f}")
    
    # Effective resistance example
    print("\n  Effective resistance between missing indices (j=1):")
    missing_j1 = compute_missing_indices(structure, 1)
    if len(missing_j1) >= 2:
        idx1, idx2 = missing_j1[0], missing_j1[1]
        R = relative_effective_resistance(structure, 1, idx1, idx2)
        print(f"    R({idx1}, {idx2}) = {R:.4f}")
    
    return structure


if __name__ == "__main__":
    # Run all demos
    demo_basic_usage()
    demo_3d_tensor()
    demo_relative_hodge()
    demo_neural_network()
