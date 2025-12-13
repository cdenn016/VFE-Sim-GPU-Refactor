# -*- coding: utf-8 -*-
"""
Batched Operations for GPU Acceleration
=======================================

Compiled batched operations for efficient N x N pairwise computations.
Uses torch.compile for kernel fusion when available (PyTorch 2.0+).

These operations are critical for achieving the 50-100x speedup target
by replacing Python loops with vectorized GPU kernels.

Note: Core KL and pairwise functions are imported from gradients.torch_energy
to avoid duplication. This module adds compilation and additional utilities.

Author: Claude (refactoring)
Date: December 2024
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Import canonical implementations from torch_energy (avoid duplication)
from gradients.torch_energy import (
    kl_divergence_gaussian as batched_kl_divergence,
    batched_pairwise_kl as compute_all_pairwise_kl,
    transport_gaussian as batched_transport_gaussian,
    compute_transport_operator,
)


# =============================================================================
# Core Batched Operations (unique to this module)
# =============================================================================

def batched_matrix_exp(X: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix exponential.

    Args:
        X: Matrices, shape (..., K, K)

    Returns:
        exp(X), shape (..., K, K)
    """
    return torch.linalg.matrix_exp(X)


def batched_cholesky(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Batched Cholesky decomposition with regularization.

    Args:
        Sigma: SPD matrices, shape (..., K, K)
        eps: Regularization for numerical stability

    Returns:
        Lower Cholesky factor L, shape (..., K, K)
    """
    K = Sigma.shape[-1]
    eye = torch.eye(K, device=Sigma.device, dtype=Sigma.dtype)
    return torch.linalg.cholesky(Sigma + eps * eye)


def batched_cholesky_inverse(L: torch.Tensor) -> torch.Tensor:
    """
    Batched inverse from Cholesky factor: Sigma^{-1} from L where Sigma = L @ L^T.

    Args:
        L: Lower Cholesky factors, shape (..., K, K)

    Returns:
        Sigma^{-1}, shape (..., K, K)
    """
    return torch.cholesky_inverse(L)


def batched_logdet_from_cholesky(L: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute log determinant from Cholesky factor: log|Sigma| = 2 * sum(log(diag(L))).

    Args:
        L: Lower Cholesky factors, shape (..., K, K)
        eps: Minimum value for log

    Returns:
        log|Sigma|, shape (...)
    """
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag.clamp(min=eps)), dim=-1)


# =============================================================================
# Batched Transport Operator (separate phi arrays version)
# =============================================================================

def batched_transport_operator(
    phi_i: torch.Tensor,      # (N, 3) or (..., 3)
    phi_j: torch.Tensor,      # (M, 3) or (..., 3)
    generators: torch.Tensor, # (3, K, K)
) -> torch.Tensor:
    """
    Compute batched transport operators: Omega[i,j] = exp(phi_i . J) @ exp(-phi_j . J).

    Args:
        phi_i: Source gauge fields, shape (N, 3) or broadcasted
        phi_j: Target gauge fields, shape (M, 3) or broadcasted
        generators: Lie algebra generators, shape (3, K, K)

    Returns:
        Transport operators, shape (N, M, K, K) or broadcasted
    """
    # Contract with generators
    X_i = torch.einsum('...a,akl->...kl', phi_i, generators)
    X_j = torch.einsum('...a,akl->...kl', phi_j, generators)

    # Matrix exponentials
    exp_i = torch.linalg.matrix_exp(X_i)
    exp_neg_j = torch.linalg.matrix_exp(-X_j)

    # Compose
    return exp_i @ exp_neg_j


# =============================================================================
# Attention Computation
# =============================================================================

def compute_softmax_attention(
    kl_matrix: torch.Tensor,  # (N, N)
    kappa: float = 1.0,
    mask_diagonal: bool = True,
) -> torch.Tensor:
    """
    Compute softmax attention weights from KL matrix.

    beta[i, j] = softmax(-kappa * KL[i, :])_j

    Args:
        kl_matrix: KL divergences, shape (N, N)
        kappa: Temperature parameter
        mask_diagonal: If True, exclude self-interactions

    Returns:
        Attention weights, shape (N, N)
    """
    N = kl_matrix.shape[0]
    device = kl_matrix.device

    if mask_diagonal:
        # Mask diagonal with large value before softmax
        mask = torch.eye(N, device=device, dtype=torch.bool)
        kl_masked = kl_matrix.clone()
        kl_masked[mask] = float('inf')

        beta = F.softmax(-kappa * kl_masked, dim=-1)
        beta = beta * (~mask).float()
    else:
        beta = F.softmax(-kappa * kl_matrix, dim=-1)

    return beta


# =============================================================================
# Compiled Versions (PyTorch 2.0+)
# =============================================================================

# Try to compile key functions if torch.compile is available
_COMPILED = False

try:
    if hasattr(torch, 'compile'):
        # Compile the most expensive operations
        compute_all_pairwise_kl_compiled = torch.compile(
            compute_all_pairwise_kl,
            mode='reduce-overhead',
            dynamic=False,
        )

        batched_kl_divergence_compiled = torch.compile(
            batched_kl_divergence,
            mode='reduce-overhead',
            dynamic=False,
        )

        _COMPILED = True
except Exception:
    # Fallback to non-compiled versions
    compute_all_pairwise_kl_compiled = compute_all_pairwise_kl
    batched_kl_divergence_compiled = batched_kl_divergence


def is_compiled() -> bool:
    """Check if compiled versions are available."""
    return _COMPILED


# =============================================================================
# Utility Functions
# =============================================================================

def symmetrize(M: torch.Tensor) -> torch.Tensor:
    """Symmetrize a matrix: sym(M) = (M + Mᵀ)/2"""
    return 0.5 * (M + M.transpose(-1, -2))


def ensure_spd(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Project matrix to SPD cone via eigenvalue clipping.

    Handles AMP/mixed precision safely by computing eigendecomposition in FP32.

    Args:
        Sigma: Matrix, shape (..., K, K)
        eps: Minimum eigenvalue

    Returns:
        SPD matrix, shape (..., K, K)
    """
    orig_dtype = Sigma.dtype
    K = Sigma.shape[-1]
    device = Sigma.device

    # Add diagonal regularization before eigendecomposition
    eye = torch.eye(K, device=device, dtype=Sigma.dtype)
    # Broadcast eye to match batch dimensions
    for _ in range(Sigma.ndim - 2):
        eye = eye.unsqueeze(0)
    Sigma_reg = Sigma + eps * eye
    Sigma_reg = symmetrize(Sigma_reg)

    # Eigendecomposition (force FP32 for numerical stability under AMP)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_reg.float())
        # Clip eigenvalues to be positive
        eigenvalues_clipped = torch.clamp(eigenvalues, min=eps)
        # Reconstruct and cast back to original dtype
        Sigma_spd = (eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.transpose(-1, -2)).to(orig_dtype)
    except RuntimeError:
        # Fallback to regularized identity
        Sigma_spd = (eps * eye).expand_as(Sigma).clone()

    return symmetrize(Sigma_spd)


def project_to_principal_ball(phi: torch.Tensor, max_norm: float = 3.1) -> torch.Tensor:
    """
    Project phi to principal ball |phi| < pi.

    Args:
        phi: Gauge fields, shape (..., 3)
        max_norm: Maximum allowed norm (default slightly less than pi)

    Returns:
        Projected phi, shape (..., 3)
    """
    norms = torch.linalg.norm(phi, dim=-1, keepdim=True)
    scale = torch.where(
        norms > max_norm,
        max_norm / (norms + 1e-8),
        torch.ones_like(norms)
    )
    return phi * scale
