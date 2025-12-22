"""
Variational Feed-Forward Networks for Gauge Transformer
========================================================

Integrates with validated gradient_engine.py for theoretically correct active inference!

Three implementations:
1. APPROXIMATE: Omit second-order ∂β_ij/∂μ_i term (legacy, simple)
2. FULL: Include all terms manually (legacy, exact but complex)
3. GRADIENT_ENGINE: Use validated gradient_engine backend (RECOMMENDED!)

The GRADIENT_ENGINE version:
- Updates BOTH means μ AND covariances Σ
- Uses natural gradients via Fisher-Rao metric
- Includes all energy terms (self-coupling, alignment, observations, softmax coupling)
- Proper χ-weighting and gauge transport
- Theoretically principled active inference

Mathematical Foundation:
-----------------------
Free Energy (E-STEP):
    F = α·Σ_i KL(q_i||p_i)                      # Prior consistency
      + λ_β·Σ_{i,j} β_ij·KL(q_i||Ω_{ij}q_j)    # Belief alignment
      + λ_γ·Σ_{i,j} γ_ij·KL(p_i||Ω_{ij}p_j)    # Prior alignment
      + CE(W_out·μ, targets)                    # DISCRETE OBSERVATIONS!

CRITICAL: The cross-entropy term is the SINGLE observation model!
- E-step: Minimize F w.r.t. μ, Σ → compute ∂CE/∂μ with W_out frozen
- M-step: Minimize F w.r.t. W_out, embeddings → compute ∂CE/∂W_out with μ frozen

This is classic EM:
- E-step: "Given model (W_out), what beliefs (μ) explain observations?"
- M-step: "Given beliefs (μ), what model parameters explain observations?"

The SAME cross-entropy appears in both steps, just optimizing different parameters!

Gradient Engine computes:
    ∂F/∂θ for θ = {μ_q, Σ_q, μ_p, Σ_p, φ}

With natural gradient projection:
    Δθ = -η · F⁻¹(θ) · ∇F(θ)

Where F(θ) is the Fisher-Rao metric.

Author: Integrated with validated gradient_engine.py
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import validated gradient engine
import sys
from pathlib import Path
# Add parent directory to path for gradient_engine import
sys.path.insert(0, str(Path(__file__).parent.parent))

from gradients.gradient_engine import (
    _compute_agent_euclidean_gradients,
    project_to_natural_gradients,
)
from config import SystemConfig, AgentConfig
from agent.agents import Agent
from geometry.geometry_base import BaseManifold, TopologyType
from gradients.retraction import retract_spd  # For SPD manifold updates

# Import attention computation for dynamic β
from transformer.attention import compute_attention_weights


# =============================================================================
# Memory-Efficient VFE Gradient Helpers
# =============================================================================

def _compute_vfe_gradients_block_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) full block-diagonal covariances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K, K) prior covariances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: float,
    lambda_belief: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    chunk_size: Optional[int],
    compute_sigma_align_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-diagonal VFE gradient computation with optional chunking.

    Processes each irrep block separately to reduce memory from O(N²K²) to O(N² × max(dᵢ²)).
    When chunk_size is provided, also chunks over query positions to reduce memory further.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Default chunk size to N (no chunking) if not provided
    C = chunk_size if chunk_size is not None else N

    # Initialize output gradients
    grad_mu = torch.zeros(B, N, K, device=device, dtype=dtype)
    grad_sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

    # =================================================================
    # 1. Self-Coupling Gradient (block-wise but simpler)
    # =================================================================
    sigma_p_reg = sigma_p + eps * torch.eye(K, device=device, dtype=dtype)
    sigma_p_inv = torch.linalg.inv(sigma_p_reg)
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

    sigma_q_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
    sigma_q_inv = torch.linalg.inv(sigma_q_reg)
    grad_sigma_self = alpha * 0.5 * (sigma_p_inv - sigma_q_inv)

    grad_mu = grad_mu + grad_mu_self
    grad_sigma = grad_sigma + grad_sigma_self

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + chunked processing)
    # =================================================================
    # Precompute matrix exponentials per block (these are (B, N, d, d))
    block_exp_phi = []
    block_exp_neg_phi = []
    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d
        gen_block = generators[:, block_start:block_end, block_start:block_end]
        phi_matrix_block = torch.einsum('bna,aij->bnij', phi, gen_block)
        block_exp_phi.append(torch.matrix_exp(phi_matrix_block))
        block_exp_neg_phi.append(torch.matrix_exp(-phi_matrix_block))
        block_start = block_end

    # Accumulators for alignment gradients
    grad_mu_align = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # For KL values and gradients - we'll accumulate these in chunks
    # We need full (B, N, N) for final softmax coupling, so accumulate
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)

    # Process in chunks over query positions (i dimension)
    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        C_actual = i_end - i_start

        # Process each irrep block for this chunk of query positions
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d

            # Extract block beliefs
            mu_block = mu_q[:, :, block_start:block_end]  # (B, N, d)
            sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end]  # (B, N, d, d)

            # Get chunked exponentials for query positions
            exp_phi_i = block_exp_phi[block_idx][:, i_start:i_end]  # (B, C, d, d)
            exp_neg_phi_j = block_exp_neg_phi[block_idx]  # (B, N, d, d)

            # Compute Omega for this chunk: (B, C, N, d, d)
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )  # (B, C, N, d, d)

            # Transport means and covariances for this chunk
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_chunk, mu_block)  # (B, C, N, d)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_chunk, sigma_block, Omega_chunk.transpose(-1, -2)
            )  # (B, C, N, d, d)

            del Omega_chunk

            # Regularize and invert
            I_d = torch.eye(d, device=device, dtype=dtype)
            sigma_j_reg = sigma_j_transported + eps * I_d
            sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, C, N, d, d)

            # Delta mu for this block (query chunk)
            mu_block_i = mu_block[:, i_start:i_end]  # (B, C, d)
            delta_mu_block = mu_block_i[:, :, None, :] - mu_j_transported  # (B, C, N, d)

            # ∂KL_ij/∂μ_i for this block
            grad_kl_block = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_block)  # (B, C, N, d)
            grad_kl_per_pair_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block

            # KL terms for this block
            mahal_block = torch.einsum('bijk,bijk->bij', delta_mu_block, grad_kl_block)  # (B, C, N)

            sigma_i_block = sigma_block[:, i_start:i_end, None, :, :].expand(-1, -1, N, -1, -1)  # (B, C, N, d, d)
            trace_block = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_block))

            try:
                L_j = torch.linalg.cholesky(sigma_j_reg)
                logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                logdet_j = torch.zeros(B, C_actual, N, device=device, dtype=dtype)

            sigma_i_block_diag = sigma_block[:, i_start:i_end] + eps * I_d  # (B, C, d, d)
            try:
                L_i = torch.linalg.cholesky(sigma_i_block_diag)
                logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                logdet_i = torch.zeros(B, C_actual, device=device, dtype=dtype)

            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_j - logdet_i[:, :, None])
            kl_values[:, i_start:i_end, :] = kl_values[:, i_start:i_end, :] + kl_block.clamp(min=0.0)

            # Sigma alignment gradient for this block
            if compute_sigma_align_grad:
                sigma_i_inv_block = torch.linalg.inv(sigma_i_block_diag)  # (B, C, d, d)
                sigma_i_inv_exp = sigma_i_inv_block[:, :, None, :, :].expand(-1, -1, N, -1, -1)
                grad_sigma_block = 0.5 * (sigma_j_inv - sigma_i_inv_exp)
                beta_chunk = beta[:, i_start:i_end, :]  # (B, C, N)
                grad_sigma_block_weighted = lambda_belief * torch.einsum('bij,bijkl->bikl', beta_chunk, grad_sigma_block)
                grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] += grad_sigma_block_weighted

            del sigma_j_transported, sigma_j_inv, mu_j_transported
            block_start = block_end

    # Direct term
    grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)

    # Softmax coupling term
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_deviation = grad_kl_per_pair_full - avg_grad.unsqueeze(2)
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa
    grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu + grad_mu_align
    grad_sigma = grad_sigma + grad_sigma_align

    return grad_mu, grad_sigma


def _compute_vfe_gradients_chunked(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: float,
    lambda_belief: float,
    kappa: float,
    eps: float,
    chunk_size: int,
    compute_sigma_align_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked VFE gradient computation for diagonal covariance mode.

    Processes N×N pairs in C×C chunks to reduce peak memory.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # =================================================================
    # 1. Self-Coupling Gradient (simple, no chunking needed)
    # =================================================================
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * delta_mu / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # =================================================================
    # 2. Alignment Gradient (chunked processing)
    # =================================================================
    # Precompute matrix exponentials for all positions
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi = torch.matrix_exp(phi_matrix)
    exp_neg_phi = torch.matrix_exp(-phi_matrix)
    del phi_matrix

    # Expand diagonal to full for transport
    sigma_j_diag = torch.diag_embed(sigma_q_safe)  # (B, N, K, K)

    # Accumulators
    grad_mu_direct = torch.zeros_like(mu_q)
    grad_mu_softmax = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # We need to accumulate these for the softmax coupling term
    beta_grad_kl_sum = torch.zeros_like(mu_q)  # For avg_grad
    kl_dbeta_grad_sum = torch.zeros_like(mu_q)  # For softmax term

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        exp_phi_i = exp_phi[:, i_start:i_end]  # (B, n_i, K, K)
        mu_i = mu_q[:, i_start:i_end]
        sigma_i = sigma_q_safe[:, i_start:i_end]
        beta_i = beta[:, i_start:i_end, :]  # (B, n_i, N)

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end]
            mu_j = mu_q[:, j_start:j_end]
            sigma_j_diag_chunk = sigma_j_diag[:, j_start:j_end]
            beta_chunk = beta_i[:, :, j_start:j_end]  # (B, n_i, n_j)

            # Compute Omega for this chunk
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)

            # Transport means
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_j)

            # Transport covariances (diagonal -> full -> diagonal for grad)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega, sigma_j_diag_chunk, Omega.transpose(-1, -2)
            )

            del Omega

            # Regularize and invert
            sigma_j_reg = sigma_j_transported + eps * torch.eye(K, device=device, dtype=dtype)
            sigma_j_inv = torch.linalg.inv(sigma_j_reg)

            # Delta mu
            delta_mu_ij = mu_i[:, :, None, :] - mu_j_transported

            # ∂KL/∂μ_i
            grad_kl = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

            # Direct term contribution
            grad_mu_direct[:, i_start:i_end] += lambda_belief * torch.einsum(
                'bij,bijk->bik', beta_chunk, grad_kl
            )

            # For softmax coupling, we need full KL values
            mahal = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl)

            sigma_i_diag_exp = torch.diag_embed(sigma_i)[:, :, None, :, :].expand(-1, -1, n_j, -1, -1)
            trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_diag_exp))

            try:
                L_p = torch.linalg.cholesky(sigma_j_reg)
                logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1) + eps), dim=-1)
            except RuntimeError:
                logdet_p = torch.zeros(B, n_i, n_j, device=device, dtype=dtype)

            logdet_q = torch.sum(torch.log(sigma_i), dim=-1)[:, :, None].expand(-1, -1, n_j)

            kl_chunk = 0.5 * (trace_term + mahal - K + logdet_p - logdet_q).clamp(min=0.0)

            # Weighted average gradient for this chunk
            beta_grad_kl_sum[:, i_start:i_end] += torch.einsum('bij,bijk->bik', beta_chunk, grad_kl)

            # Softmax coupling contribution
            avg_grad_i = beta_grad_kl_sum[:, i_start:i_end] / (beta_i.sum(dim=-1, keepdim=True) + eps)
            grad_deviation = grad_kl - avg_grad_i.unsqueeze(2)
            d_beta_d_mu = beta_chunk.unsqueeze(-1) * grad_deviation / kappa
            grad_mu_softmax[:, i_start:i_end] += lambda_belief * torch.einsum(
                'bij,bijk->bik', kl_chunk, d_beta_d_mu
            )

            # Sigma alignment gradient
            if compute_sigma_align_grad:
                sigma_j_inv_diag = torch.diagonal(sigma_j_inv, dim1=-2, dim2=-1)
                sigma_i_inv = 1.0 / sigma_i
                sigma_i_inv_exp = sigma_i_inv[:, :, None, :].expand(-1, -1, n_j, -1)
                grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv_exp)
                grad_sigma_align[:, i_start:i_end] += lambda_belief * torch.einsum(
                    'bij,bijk->bik', beta_chunk, grad_sigma_pair
                )

            del sigma_j_transported, sigma_j_inv, mu_j_transported

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align

    return grad_mu, grad_sigma


# =============================================================================
# GPU-Based Gradient Computation (PyTorch - FAST!)
# =============================================================================

def compute_vfe_gradients_gpu(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances or (B, N, K, K) full
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators (SO(3) or SO(N))
    alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
    lambda_belief: float = 1.0,  # Belief alignment weight
    kappa: float = 1.0,        # Temperature (for normalization)
    eps: float = 1e-6,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
    # Memory-efficient options (NEW!)
    irrep_dims: Optional[List[int]] = None,  # Block dimensions for block-diagonal processing
    chunk_size: Optional[int] = None,  # Chunk size for memory-efficient processing
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VFE gradients entirely on GPU using PyTorch.

    This is the FAST version that replaces the NumPy-based gradient_engine.
    Fully vectorized - no loops over batch or agents!

    Supports both SO(3) and SO(N) gauge groups. The number of generators
    determines the gauge group: 3 for SO(3), N(N-1)/2 for SO(N).

    Gradients computed:
    1. Self-coupling: ∂/∂μ_q [α · KL(q||p)]
    2. Belief alignment: ∂/∂μ_q [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief variances - diagonal (B, N, K) or full (B, N, K, K)
        mu_p: Prior means (B, N, K)
        sigma_p: Prior variances - diagonal (B, N, K) or full (B, N, K, K)
        beta: Attention weights (B, N, N), already normalized
        phi: Gauge frames (B, N, n_gen) where n_gen is # of generators
        generators: Lie algebra generators (n_gen, K, K) - SO(3) has 3, SO(N) has N(N-1)/2
        alpha: Weight for KL(q||p) self-coupling term
        lambda_belief: Weight for belief alignment term
        kappa: Temperature parameter
        eps: Numerical stability
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators().
                         When provided, avoids redundant matrix exponential computations.
        compute_sigma_align_grad: If True (default), compute sigma gradient from belief alignment term.
                                  This is the theoretically correct gradient:
                                    ∂KL/∂Σ_q = 0.5 * (Σ_transported^{-1} - Σ_q^{-1})
                                  Set to False for legacy behavior (zero sigma alignment gradient).
        irrep_dims: Optional list of block dimensions for memory-efficient block-diagonal processing.
                   When provided, processes each irrep block separately to reduce memory from
                   O(N²K²) to O(N² × max(dᵢ²)).
        chunk_size: Optional chunk size for processing N×N pairs in C×C chunks.

    Returns:
        grad_mu: Gradient w.r.t. μ_q, shape (B, N, K)
        grad_sigma: Gradient w.r.t. σ_q, shape (B, N, K) for diagonal
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect diagonal vs full covariance
    is_diagonal = sigma_q.dim() == 3

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal processing
    # =================================================================
    if irrep_dims is not None and not is_diagonal:
        return _compute_vfe_gradients_block_diagonal(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, irrep_dims, chunk_size,
            compute_sigma_align_grad
        )

    # =================================================================
    # MEMORY-EFFICIENT PATH: Chunked processing for diagonal mode
    # =================================================================
    if chunk_size is not None and is_diagonal:
        return _compute_vfe_gradients_chunked(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, chunk_size,
            compute_sigma_align_grad
        )

    # =================================================================
    # 1. Self-Coupling Gradient: ∂/∂μ_q [α · KL(q||p)]
    # =================================================================
    # For diagonal Gaussians:
    #   KL(q||p) = 0.5 * Σ_k [ σ_q[k]/σ_p[k] + (μ_p[k]-μ_q[k])²/σ_p[k] - 1 + log(σ_p[k]/σ_q[k]) ]
    #   ∂KL/∂μ_q = (μ_q - μ_p) / σ_p
    #   ∂KL/∂σ_q = 0.5 * (1/σ_p - 1/σ_q)

    if is_diagonal:
        # Clamp for stability
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)

        # Self-coupling gradient w.r.t. μ
        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * delta_mu / sigma_p_safe  # (B, N, K)

        # Self-coupling gradient w.r.t. σ (diagonal)
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)  # (B, N, K)
    else:
        # Full covariance - use matrix operations
        # ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        sigma_p_reg = sigma_p + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_p_inv = torch.linalg.inv(sigma_p_reg)  # (B, N, K, K)

        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

        # ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        sigma_q_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_q_inv = torch.linalg.inv(sigma_q_reg)
        grad_sigma_self = alpha * 0.5 * (sigma_p_inv - sigma_q_inv)

    # =================================================================
    # 2. Belief Alignment Gradient: ∂/∂μ_i [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]
    # =================================================================
    # Full gradient via product rule:
    #   ∂/∂μ_i [Σ_j β_ij · KL_ij] = Σ_j β_ij · ∂KL_ij/∂μ_i + Σ_j KL_ij · ∂β_ij/∂μ_i
    #                                  ↑ direct term           ↑ SOFTMAX COUPLING (nonlinearity!)
    #
    # The softmax coupling gradient is the KEY nonlinearity (replaces GELU/ReLU):
    #   ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    if is_diagonal:
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            # Compute transport operators (vectorized)
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
            exp_phi = torch.matrix_exp(phi_matrix)       # (B, N, K, K)
            exp_neg_phi = torch.matrix_exp(-phi_matrix)  # (B, N, K, K)

            # Transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
            # For all pairs: (B, N, N, K, K)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport all μ_j to frame i: μ_j_transported[i,j] = Ω_ij @ μ_j
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Difference: μ_i - μ_j_transported (for each pair i,j)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported  # (B, N, N, K)

        # =================================================================
        # PROPER COVARIANCE TRANSPORT: Σ_j_transported = Ω @ diag(σ_j) @ Ω^T
        # =================================================================
        # Even though input is diagonal, transported covariance is FULL!
        # This is crucial for gauge equivariance.

        # Expand diagonal to full: diag(σ_j) -> (B, N, K, K)
        sigma_j_diag = torch.diag_embed(sigma_q.clamp(min=eps))  # (B, N, K, K)

        # Transport covariance: Σ_j_transported = Ω_ij @ Σ_j @ Ω_ij^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_j_diag, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert transported covariance
        sigma_j_reg = sigma_j_transported + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i = Σ_j_transported^{-1} @ (μ_i - μ_j_transported)
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)  # (B, N, N, K)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        # Σ_i is diagonal: diag(σ_q[i]) expanded to all pairs
        sigma_i_diag = torch.diag_embed(sigma_q.clamp(min=eps))  # (B, N, K, K)
        sigma_i_expanded = sigma_i_diag[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K)
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms
        # For transported covariance (full matrix): use Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N, N)
        except RuntimeError:
            # Cholesky failed - use eigenvalue decomposition as fallback
            # This is slower but more robust for ill-conditioned matrices
            eigvals = torch.linalg.eigvalsh(sigma_j_reg)  # (B, N, N, K)
            logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)  # (B, N, N)
        # For source covariance (diagonal): log|diag(σ)| = sum(log(σ))
        logdet_i = torch.sum(torch.log(sigma_q.clamp(min=eps)), dim=-1)  # (B, N)
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N)  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        kl_values = kl_values.clamp(min=0.0)  # (B, N, N)

        # =================================================================
        # 2a. Direct term: Σ_j β_ij · ∂KL_ij/∂μ_i
        # =================================================================
        grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)

        # =================================================================
        # 2b. Softmax coupling term (THE NONLINEARITY!):
        #     ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ
        #     grad_softmax = Σ_j KL_ij · ∂β_ij/∂μ_i
        # =================================================================
        # Weighted average gradient: avg_grad_i = Σ_k β_ik · ∂KL_ik/∂μ_i
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)  # (B, N, K)

        # Deviation from average: ∂KL_ij/∂μ_i - avg_grad_i
        # grad_kl_per_pair: (B, N, N, K), avg_grad: (B, N, K) -> expand to (B, N, 1, K)
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)  # (B, N, N, K)

        # Softmax coupling gradient: ∂β_ij/∂μ_i = β_ij · grad_deviation / κ
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa  # (B, N, N, K)

        # Weight by KL values and sum: Σ_j KL_ij · ∂β_ij/∂μ_i
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        # Total alignment gradient (direct + softmax coupling)
        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # For diagonal mode, we take the diagonal of the transported inverse
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Diagonal of inverse of transported covariance: diag(Σ_j_transported^{-1})
            sigma_j_inv_diag = torch.diagonal(sigma_j_inv, dim1=-2, dim2=-1)  # (B, N, N, K)

            # Inverse of diagonal Σ_i: 1/σ_i expanded for all pairs
            sigma_i_inv = 1.0 / sigma_q.clamp(min=eps)  # (B, N, K)
            sigma_i_inv_expanded = sigma_i_inv[:, :, None, :].expand(-1, -1, N, -1)  # (B, N, N, K)

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1}_kk - 1/σ_i[k])
            grad_sigma_per_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv_expanded)  # (B, N, N, K)

            # Weight by attention and sum: Σ_j β_ij * ∂KL_ij/∂σ_i
            grad_sigma_align = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)  # (B, N, K)
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance belief alignment
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
            exp_phi = torch.matrix_exp(phi_matrix)
            exp_neg_phi = torch.matrix_exp(-phi_matrix)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

        # Transport covariances: Σ_j_transported = Ω @ Σ_j @ Ω^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert
        sigma_j_reg = sigma_j_transported + eps * torch.eye(K, device=device, dtype=dtype)
        sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)

        # ∂KL_ij/∂μ_i
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        sigma_i_expanded = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K)
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms using Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_j_reg)
            logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)

        sigma_i_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        try:
            L_i = torch.linalg.cholesky(sigma_i_reg)  # (B, N, K, K)
            logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1) + eps), dim=-1)  # (B, N)
        except RuntimeError:
            eigvals = torch.linalg.eigvalsh(sigma_i_reg)
            logdet_i = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N)  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        kl_values = kl_values.clamp(min=0.0)  # (B, N, N)

        # Direct term
        grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)

        # Softmax coupling term
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (full covariance case)
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Use Σ_i^{-1} computed earlier in self-coupling section (sigma_q_inv)
            sigma_i_inv_expanded = sigma_q_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1)  # (B, N, N, K, K)

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
            grad_sigma_per_pair = 0.5 * (sigma_j_inv - sigma_i_inv_expanded)  # (B, N, N, K, K)

            # Weight by attention and sum: Σ_j β_ij * ∂KL_ij/∂Σ_i
            grad_sigma_align = lambda_belief * torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)

    # =================================================================
    # 3. Combine Gradients
    # =================================================================
    grad_mu = grad_mu_self + grad_mu_align

    if is_diagonal:
        grad_sigma = grad_sigma_self + grad_sigma_align
    else:
        grad_sigma = grad_sigma_self + grad_sigma_align

    return grad_mu, grad_sigma


def compute_natural_gradient_gpu(
    grad_mu: torch.Tensor,     # (B, N, K) Euclidean gradient
    grad_sigma: torch.Tensor,  # (B, N, K) or (B, N, K, K)
    sigma_q: torch.Tensor,     # (B, N, K) or (B, N, K, K)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Euclidean gradients to natural gradients using Fisher metric.

    For Gaussian distributions, the Fisher information metric is:
        F_μ = Σ^{-1}  →  natural_grad_μ = Σ @ euclidean_grad_μ
        F_σ = 2Σ^{-2} →  natural_grad_σ = 0.5 * Σ² @ euclidean_grad_σ (diagonal approx)

    Args:
        grad_mu: Euclidean gradient w.r.t. μ
        grad_sigma: Euclidean gradient w.r.t. σ
        sigma_q: Current covariance
        eps: Numerical stability

    Returns:
        nat_grad_mu: Natural gradient for μ
        nat_grad_sigma: Natural gradient for σ
    """
    is_diagonal = sigma_q.dim() == 3

    if is_diagonal:
        # Diagonal case: simple element-wise multiplication
        sigma_safe = sigma_q.clamp(min=eps)
        nat_grad_mu = sigma_safe * grad_mu  # (B, N, K)
        nat_grad_sigma = 0.5 * sigma_safe * sigma_safe * grad_sigma  # (B, N, K)
    else:
        # Full covariance: matrix multiplication
        nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
        # For sigma, use diagonal approximation for simplicity
        sigma_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1)
        nat_grad_sigma = 0.5 * sigma_diag.unsqueeze(-1) * sigma_diag.unsqueeze(-2) * grad_sigma

    return nat_grad_mu, nat_grad_sigma


# =============================================================================
# SPD Retraction (PyTorch GPU version)
# =============================================================================

def retract_spd_torch(
    Sigma: torch.Tensor,
    delta_Sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SPD-preserving retraction for covariance matrices (PyTorch GPU).

    Uses exponential map with WHITENED trust region (like retraction.py).
    The whitening normalizes out the σ² scaling in natural gradients,
    preventing runaway growth.

    Args:
        Sigma: SPD matrices, shape (B, N, K, K) or (B*N, K, K)
        delta_Sigma: Symmetric tangent vectors, same shape as Sigma
        step_size: Learning rate τ (already applied to delta_Sigma typically)
        trust_region: Max norm of WHITENED tangent ||Σ^{-1/2} ΔΣ Σ^{-1/2}||_F
        eps: Regularization floor for numerical stability

    Returns:
        Sigma_new: SPD matrices, same shape as Sigma
    """
    # Handle different input shapes
    original_shape = Sigma.shape
    if Sigma.dim() == 4:
        B, N, K, _ = Sigma.shape
        Sigma = Sigma.reshape(B * N, K, K)
        delta_Sigma = delta_Sigma.reshape(B * N, K, K)

    batch_size, K, _ = Sigma.shape
    device = Sigma.device
    dtype = Sigma.dtype

    # Symmetrize inputs (numerical safety)
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    delta_Sigma = 0.5 * (delta_Sigma + delta_Sigma.transpose(-1, -2))

    # Use DIAGONAL whitening (robust to ill-conditioning, avoids eigh)
    # This approximates full whitening B = Σ^{-1/2} ΔΣ Σ^{-1/2}
    # For diagonal-dominant covariances (common case), this is accurate
    # Key insight: whitening normalizes out the σ² scaling in natural gradients
    diag_sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1)  # (batch, K)
    diag_sigma = diag_sigma.clamp(min=eps)
    inv_sqrt_diag = 1.0 / torch.sqrt(diag_sigma)  # (batch, K)

    # Diagonal-whitened tangent: B ≈ D^{-1/2} ΔΣ D^{-1/2} where D = diag(Σ)
    B = (inv_sqrt_diag.unsqueeze(-1) * delta_Sigma) * inv_sqrt_diag.unsqueeze(-2)  # (batch, K, K)

    # Trust region on WHITENED Frobenius norm (this is the key fix!)
    if trust_region is not None and trust_region > 0:
        B_norm = torch.linalg.norm(B, ord='fro', dim=(-2, -1), keepdim=True)  # (batch, 1, 1)
        scale = torch.clamp(trust_region / (B_norm + eps), max=1.0)
        B = B * scale

    # Un-whiten to get scaled delta: ΔΣ_scaled = D^{1/2} B D^{1/2}
    sqrt_diag = torch.sqrt(diag_sigma)  # (batch, K)
    delta_scaled = (sqrt_diag.unsqueeze(-1) * B) * sqrt_diag.unsqueeze(-2)  # (batch, K, K)

    # Linear update with trust-region-scaled delta
    Sigma_new = Sigma + step_size * delta_scaled

    # Symmetrize
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(-1, -2))

    # Ensure SPD via iterative regularization
    # Try Cholesky; if it fails, add regularization and retry
    # This is more robust than eigendecomposition for ill-conditioned matrices
    eye_K = torch.eye(K, device=device, dtype=dtype)

    # Compute scale for adaptive regularization (based on diagonal magnitude)
    diag_mean = torch.diagonal(Sigma_new, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True)  # (batch, 1)
    base_reg = torch.clamp(diag_mean, min=eps).unsqueeze(-1)  # (batch, 1, 1)

    # Try with minimal regularization first
    reg_scales = [eps, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    for reg_scale in reg_scales:
        Sigma_reg = Sigma_new + (reg_scale * base_reg) * eye_K
        try:
            # Cholesky is the gold standard for checking SPD
            _ = torch.linalg.cholesky(Sigma_reg)
            # Success! Use this regularized version
            Sigma_new = Sigma_reg
            break
        except RuntimeError:
            # Cholesky failed, try more regularization
            continue
    else:
        # All attempts failed - fall back to heavily regularized original
        Sigma_new = Sigma + 0.1 * base_reg * eye_K

    # Restore original shape
    if len(original_shape) == 4:
        Sigma_new = Sigma_new.reshape(original_shape)

    return Sigma_new


def retract_spd_diagonal_torch(
    sigma_diag: torch.Tensor,
    delta_sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SPD retraction for diagonal covariances (much simpler).

    For diagonal matrices, the exponential map reduces to:
        σ_new = σ * exp(τ * δσ / σ)

    This ensures positivity: exp(x) > 0 for all x.

    Args:
        sigma_diag: Diagonal variances, shape (B, N, K)
        delta_sigma: Tangent in diagonal form, shape (B, N, K)
        step_size: Learning rate τ
        trust_region: Max absolute value of exponent argument
        eps: Floor for sigma values

    Returns:
        sigma_new: Positive diagonal variances, shape (B, N, K)
    """
    sigma_safe = sigma_diag.clamp(min=eps)

    # Whitened tangent: δσ / σ (element-wise for diagonal)
    whitened = delta_sigma / sigma_safe

    # Trust region on whitened tangent
    if trust_region is not None and trust_region > 0:
        whitened = whitened.clamp(-trust_region, trust_region)

    # Exponential update: σ_new = σ * exp(τ * whitened)
    # Clip exponent to prevent overflow
    exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
    sigma_new = sigma_safe * torch.exp(exp_arg)

    # Floor for safety
    return sigma_new.clamp(min=eps)


# =============================================================================
# Utilities
# =============================================================================

def _sanitize_euclidean_gradients(euc_grads, max_norm: float = 1e3, debug: bool = True):
    """
    Sanitize Euclidean gradients to prevent NaN in natural gradient computation.

    This clips gradients that are too large, which can cause numerical overflow
    when computing natural gradients via Σ^{-1}.

    Args:
        euc_grads: AgentGradients object
        max_norm: Maximum allowed gradient norm per component
        debug: Print warnings if clipping occurs

    Returns:
        Sanitized AgentGradients object
    """
    
    import copy

    grads_sanitized = copy.copy(euc_grads)

    # Sanitize mu gradient
    if euc_grads.grad_mu_q is not None:
        grad_mu = euc_grads.grad_mu_q
        if not np.all(np.isfinite(grad_mu)):
            if debug:
                print(f"⚠️  NaN/Inf in grad_mu_q, setting to zero")
            grads_sanitized.grad_mu_q = np.zeros_like(grad_mu)
        else:
            norm = np.linalg.norm(grad_mu)
            if norm > max_norm:
                if debug:
                    print(f"⚠️  Clipping grad_mu_q: norm {norm:.2e} > {max_norm:.2e}")
                grads_sanitized.grad_mu_q = grad_mu * (max_norm / norm)

    # Sanitize Sigma gradient
    if euc_grads.grad_Sigma_q is not None:
        grad_Sigma = euc_grads.grad_Sigma_q
        if not np.all(np.isfinite(grad_Sigma)):
            if debug:
                print(f"⚠️  NaN/Inf in grad_Sigma_q, setting to zero")
            grads_sanitized.grad_Sigma_q = np.zeros_like(grad_Sigma)
        else:
            norm = np.linalg.norm(grad_Sigma)
            if norm > max_norm:
                if debug:
                    print(f"⚠️  Clipping grad_Sigma_q: norm {norm:.2e} > {max_norm:.2e}")
                grads_sanitized.grad_Sigma_q = grad_Sigma * (max_norm / norm)

    return grads_sanitized


def _compute_cholesky_robust(sigma: np.ndarray, eps: float = 1e-6, debug: bool = False) -> np.ndarray:
    """
    Compute Cholesky factor L such that Σ = L L^T with robust fallback.

    This is the VALIDATED approach from agents.py that works in simulation_suite.

    Args:
        sigma: Covariance matrix (K, K)
        eps: Regularization for numerical stability (default: 1e-6)
        debug: Print diagnostic info when fallbacks are used

    Returns:
        L: Lower triangular Cholesky factor (K, K)
    """
    K = sigma.shape[0]

    # Check for NaN/Inf
    if not np.all(np.isfinite(sigma)):
        if debug:
            print(f"⚠️  Cholesky: NaN/Inf detected in covariance, using diagonal fallback")
        return np.sqrt(eps) * np.eye(K, dtype=np.float32)

    # Symmetrize and regularize
    sigma_sym = 0.5 * (sigma + sigma.T)
    sigma_reg = sigma_sym + eps * np.eye(K)

    try:
        # Try standard Cholesky
        L = np.linalg.cholesky(sigma_reg)
        return L.astype(np.float32)

    except np.linalg.LinAlgError:
        # Cholesky failed - use eigendecomposition fallback
        # This is the VALIDATED approach from agents.py
        if debug:
            print(f"⚠️  Cholesky: Standard decomposition failed, using eigenvalue fallback")

        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_reg)
            # Clamp eigenvalues
            eigvals_clamped = np.maximum(eigvals, eps)
            if debug and np.any(eigvals < eps):
                min_eig = np.min(eigvals)
                print(f"    Clamped {np.sum(eigvals < eps)} eigenvalues (min was {min_eig:.2e})")
            # Compute Cholesky factor directly: L = V @ diag(sqrt(λ))
            L = eigvecs @ np.diag(np.sqrt(eigvals_clamped))
            return L.astype(np.float32)

        except np.linalg.LinAlgError:
            # Even eigendecomposition failed - return diagonal fallback
            if debug:
                print(f"⚠️  Cholesky: Eigendecomposition also failed, using diagonal fallback")
            return np.sqrt(eps) * np.eye(K, dtype=np.float32)


# =============================================================================
# Adapter: PyTorch Transformer → Multi-Agent System
# =============================================================================

class MockMultiAgentSystem:
    """
    Lightweight adapter that converts PyTorch transformer tensors
    to multi-agent system format for gradient_engine.

    This allows us to reuse validated gradient code without full system overhead.
    """

    def __init__(
        self,
        mu_q: np.ndarray,      # (N, K)
        sigma_q: np.ndarray,   # (N, K, K)
        mu_p: np.ndarray,      # (N, K)
        sigma_p: np.ndarray,   # (N, K, K)
        phi: np.ndarray,       # (N, 3)
        generators: np.ndarray,  # (3, K, K)
        config: SystemConfig,
        beta_weights: Optional[np.ndarray] = None,  # (N, N) - precomputed attention
    ):
        """
        Create mock system from transformer state.

        Args:
            mu_q: Belief means (N, K)
            sigma_q: Belief covariances (N, K, K)
            mu_p: Prior means (N, K)
            sigma_p: Prior covariances (N, K, K)
            phi: Gauge frames (N, 3)
            generators: SO(3) generators (3, K, K)
            config: SystemConfig with hyperparameters
            beta_weights: Optional precomputed attention weights (N, N)
        """
        self.config = config
        self.n_agents = mu_q.shape[0]

        # Create mock agents (0D point agents)
        self.agents = []
        for i in range(self.n_agents):
            agent = self._create_mock_agent(
                i, mu_q[i], sigma_q[i], mu_p[i], sigma_p[i], phi[i], generators, config
            )
            self.agents.append(agent)

        # Store precomputed beta for efficiency
        self._beta_cache = beta_weights

    def _create_mock_agent(
        self, agent_id: int, mu_q, sigma_q, mu_p, sigma_p, phi, generators, config
    ):
        """Create a lightweight mock agent without observations."""
        # Create minimal agent config
        K = mu_q.shape[0]
        agent_config = AgentConfig(
            K=K,
            spatial_shape=(),  # 0D point agent
            alpha=config.lambda_self,
        )

        # Create 0D base manifold (single point)
        base_manifold = BaseManifold(
            shape=(),  # 0D
            topology=TopologyType.PERIODIC
        )

        # Create agent (will initialize with defaults)
        
        agent = Agent(agent_id, agent_config, base_manifold=base_manifold)

        # Override with our values
        agent.mu_q = mu_q.copy()
        agent.mu_p = mu_p.copy()

        # Covariance matrices (NOT Cholesky - gauge covariance requires Σ storage)
        agent.Sigma_q = sigma_q.copy()
        agent.Sigma_p = sigma_p.copy()

        # Gauge field
        agent.gauge = type('obj', (object,), {'phi': phi.copy()})()

        # Generators
        agent.generators = generators.copy()

        # No observations - will add discrete observation gradients separately
        agent.observations = {}

        return agent

    def get_neighbors(self, agent_idx: int):
        """Return all other agents as neighbors (fully connected)."""
        return [j for j in range(self.n_agents) if j != agent_idx]

    def compute_transport_ij(self, i: int, j: int):
        """Compute transport operator Ω_ij."""
        from math_utils.transport import compute_transport
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        return compute_transport(
            agent_i.gauge.phi, agent_j.gauge.phi, agent_i.generators
        )


def convert_torch_to_numpy_system(
    mu_q: torch.Tensor,      # (B, N, K)
    sigma_q: torch.Tensor,   # (B, N, K, K) or (B, N, K) if diagonal
    mu_prior: torch.Tensor,  # (B, N, K)
    phi: torch.Tensor,       # (B, N, 3)
    generators: torch.Tensor,  # (3, K, K)
    config: SystemConfig,
    beta: Optional[torch.Tensor] = None,  # (B, N, N) averaged attention
    batch_idx: int = 0,
) -> MockMultiAgentSystem:
    """
    Convert PyTorch transformer tensors to multi-agent system format.

    Assumes priors have same covariance as beliefs (simplified).

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief covariances (B, N, K, K) full or (B, N, K) diagonal
        mu_prior: Prior means (B, N, K)
        phi: Gauge frames (B, N, 3)
        generators: SO(3) generators (3, K, K)
        config: SystemConfig
        beta: Optional attention weights (B, N, N)
        batch_idx: Which batch element to extract

    Returns:
        MockMultiAgentSystem ready for gradient_engine
    """
    # Extract single batch element and convert to numpy
    mu_q_np = mu_q[batch_idx].detach().cpu().numpy()  # (N, K)
    mu_p_np = mu_prior[batch_idx].detach().cpu().numpy()  # (N, K)
    phi_np = phi[batch_idx].detach().cpu().numpy()  # (N, 3)
    gen_np = generators.detach().cpu().numpy()  # (3, K, K)

    # Handle diagonal vs full covariance
    if sigma_q.dim() == 3:
        # Diagonal covariance: (B, N, K) -> expand to (N, K, K)
        sigma_diag = sigma_q[batch_idx].detach().cpu().numpy()  # (N, K)
        N, K = sigma_diag.shape
        sigma_q_np = np.zeros((N, K, K), dtype=sigma_diag.dtype)
        for i in range(N):
            np.fill_diagonal(sigma_q_np[i], sigma_diag[i])
    else:
        # Full covariance: (B, N, K, K)
        sigma_q_np = sigma_q[batch_idx].detach().cpu().numpy()  # (N, K, K)

    # Assume prior covariances same as beliefs (could be different)
    sigma_p_np = sigma_q_np.copy()

    # Extract beta if provided
    beta_np = None
    if beta is not None:
        beta_np = beta[batch_idx].detach().cpu().numpy()  # (N, N)

    return MockMultiAgentSystem(
        mu_q=mu_q_np,
        sigma_q=sigma_q_np,
        mu_p=mu_p_np,
        sigma_p=sigma_p_np,
        phi=phi_np,
        generators=gen_np,
        config=config,
        beta_weights=beta_np,
    )


# =============================================================================
# Gradient Engine FFN (RECOMMENDED!)
# =============================================================================

class VariationalFFNGradientEngine(nn.Module):
    """
    Variational FFN using validated gradient_engine.py backend.

    This is the FULL active inference implementation:
    - Updates both μ AND Σ
    - Uses natural gradients (Fisher-Rao metric)
    - Includes all energy terms
    - Proper gauge transport and χ-weighting

    Complexity: O(N²·K²) for full system
    But: Theoretically correct and validated!

    GPU Mode (use_gpu=True, DEFAULT):
    - Fully vectorized PyTorch operations
    - Runs entirely on GPU - no CPU transfers!
    - ~10-100x faster than CPU mode

    CPU Mode (use_gpu=False):
    - Uses NumPy-based gradient_engine.py
    - Transfers data GPU→CPU→GPU each iteration
    - Slower but matches original validated implementation
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.001,      # Self-coupling weight
        lambda_belief: float = 1.0,  # Belief alignment weight
        lambda_prior: float = 0.0,   # Prior alignment weight (usually off in transformer)
        lambda_phi: float = 0.0,     # Gauge field weight (usually off in transformer)
        kappa_beta: float = 1.0,   # Softmax temperature
        n_iterations: int = 1,     # Number of inference steps
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True,  # Update covariances?
        use_gpu: bool = True,      # Use GPU-accelerated gradients (FAST!)
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
    ):
        """
        Initialize gradient engine FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport
            alpha: Self-coupling weight (KL(q||p) term)
            lambda_belief: Belief alignment weight
            lambda_prior: Prior alignment weight (0 = off)
            lambda_phi: Gauge field evolution weight (0 = off)
            kappa_beta: Softmax temperature for attention
            n_iterations: Number of variational descent iterations
            learnable_lr: Learn step size as parameter?
            update_sigma: Update covariances? (True = full Gaussian inference)
            use_gpu: If True, use GPU-accelerated gradient computation (default)
            compute_sigma_align_grad: If True, compute sigma gradient from alignment term
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)  # (3, K, K)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma
        self.use_gpu = use_gpu
        self.compute_sigma_align_grad = compute_sigma_align_grad

        # Store hyperparameters for GPU path
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa_beta = kappa_beta

        # Create system config (for CPU path)
        self.config = SystemConfig(
            lambda_self=alpha,
            lambda_belief_align=lambda_belief,
            lambda_prior_align=lambda_prior,
            lambda_phi=lambda_phi,
            kappa_beta=kappa_beta,
            kappa_gamma=kappa_beta,  # Use same temperature for priors
            overlap_threshold=0.0,  # No spatial structure in transformer
            cache_transports=False,  # Don't need caching for single forward pass
        )

        # Learnable step size (or fixed)
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))  # Initialize to 0.1
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - attention weights
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) - covariances
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs (observations!)
        W_out: Optional[torch.Tensor] = None,  # (V, K) - output projection for discrete observations
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Variational descent using gradient_engine with discrete observations.

        E-STEP: Minimize full free energy F w.r.t. beliefs (μ, Σ, φ)

        F = α·KL(q||p) + λ_β·Σ β_ij·KL + λ_γ·Σ γ_ij·KL + CE(W_out·μ, targets)
                                                              ↑ DISCRETE OBSERVATIONS!

        The cross-entropy term is the SAME as in M-step, but here we compute ∂CE/∂μ.

        Args:
            mu: Current belief means (B, N, K)
            beta: Attention weights (B, n_heads, N, N)
            mu_prior: Prior means (B, N, K)
            phi: Gauge frames (B, N, 3)
            sigma: Belief covariances (B, N, K, K) - required if update_sigma=True
            targets: Target token IDs (B, N) - discrete observations
            W_out: Output projection matrix (V, K) - for computing CE gradient
            mask: Causal mask (B, N, N) - optional

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (B, N, K, K) if update_sigma=True, else None
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Detect diagonal covariance mode
        is_diagonal_cov = sigma is not None and sigma.dim() == 3

        # Initialize covariances if not provided
        if sigma is None:
            # Use small isotropic covariances
            sigma = 0.1 * torch.eye(K, device=device).unsqueeze(0).unsqueeze(0).expand(
                batch_size, num_agents, -1, -1
            )

        # Average attention over heads
        beta_avg = beta.mean(dim=1)  # (B, N, N)

        # Apply mask if provided
        if mask is not None:
            beta_avg = beta_avg * mask
            # Renormalize
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        # Current state
        mu_current = mu
        sigma_current = sigma

        # =====================================================================
        # Compute discrete observation gradients: ∂CE/∂μ = W_out^T · (p - y)
        # =====================================================================
        # This is the observation term in the free energy!
        # F = ... + CE(W_out·μ, targets)
        # ∂F/∂μ = W_out^T · (softmax(W_out·μ) - one_hot(targets))
        #
        # We'll compute this ONCE and add it to Euclidean gradients in each iteration.
        # Note: We freeze W_out during E-step (no gradient flow back to W_out).
        discrete_obs_grad = None
        if targets is not None and W_out is not None:
            with torch.no_grad():  # Don't backprop through W_out during E-step
                # Compute logits: (B, N, K) @ (K, V)^T = (B, N, V)
                logits = torch.matmul(mu_current, W_out.T)  # (B, N, V)

                # Softmax probabilities
                probs = F.softmax(logits, dim=-1)  # (B, N, V)

                # One-hot targets (handle padding with -1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0  # Temporarily map -1 to 0 for one_hot
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()  # (B, N, V)

                # Mask out padding positions
                mask_obs = (targets != -1).unsqueeze(-1).float()  # (B, N, 1)
                one_hot = one_hot * mask_obs

                # Gradient: W_out^T @ (probs - one_hot)
                # W_out: (V, K), (probs - one_hot): (B, N, V)
                # Result: (B, N, K)
                grad_error = (probs - one_hot) * mask_obs  # (B, N, V)
                discrete_obs_grad = torch.matmul(grad_error, W_out)  # (B, N, K)

        # =====================================================================
        # GPU PATH: Fast vectorized gradient computation (DEFAULT)
        # =====================================================================
        if self.use_gpu:
            # Use sigma_p = sigma_q for now (same covariance for prior and belief)
            sigma_p = sigma_current

            for iteration in range(self.n_iterations):
                # Compute VFE gradients entirely on GPU
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    mu_p=mu_prior,
                    sigma_p=sigma_p,
                    beta=beta_avg,
                    phi=phi,
                    generators=self.generators,
                    alpha=self.alpha,
                    lambda_belief=self.lambda_belief,
                    kappa=self.kappa_beta,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                )

                # Add discrete observation gradient if provided
                if discrete_obs_grad is not None:
                    grad_mu = grad_mu + discrete_obs_grad

                # Clip gradients for stability
                grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
                grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

                # Project to natural gradients (stays on GPU!)
                nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                    grad_mu, grad_sigma, sigma_current
                )

                # Update: descent direction (negative gradient) with WHITENED trust region
                # The natural gradient nat_grad_mu = Σ @ grad scales with σ, so we need
                # whitened trust region: ||δμ / √σ|| instead of ||δμ|| / ||μ||
                delta_mu = -self.lr * nat_grad_mu

                # Whitened trust region for mu (diagonal approximation)
                if is_diagonal_cov:
                    sigma_sqrt = torch.sqrt(sigma_current.clamp(min=1e-6))
                    whitened_delta = delta_mu / sigma_sqrt  # Whiten by √σ
                else:
                    # For full covariance, use diagonal approximation
                    sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=1e-6)
                    whitened_delta = delta_mu / torch.sqrt(sigma_diag)

                whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
                mu_trust_region = 2.0  # Trust region on whitened norm
                scale = torch.clamp(mu_trust_region / (whitened_norm + 1e-6), max=1.0)
                mu_current = mu_current + scale * delta_mu

                # Update sigma using SPD-preserving retraction
                # This is CRITICAL for stability with multiple iterations!
                # Use MUCH smaller step size for sigma (like simulation_runner uses 0.005 vs 0.1 for mu)
                sigma_lr = self.lr * 0.05  # 5% of mu learning rate
                if self.update_sigma and is_diagonal_cov:
                    # Diagonal case: exponential retraction σ_new = σ * exp(τ * δσ / σ)
                    sigma_current = retract_spd_diagonal_torch(
                        sigma_diag=sigma_current,
                        delta_sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.2,  # Max 20% change per iteration
                        eps=1e-6,
                    )
                elif self.update_sigma and not is_diagonal_cov:
                    # Full covariance: SPD-preserving retraction with trust region
                    sigma_current = retract_spd_torch(
                        Sigma=sigma_current,
                        delta_Sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.1,  # Max 10% change per iteration
                        eps=1e-6,
                    )

            # Return updated parameters (detached from computation graph)
            if self.update_sigma:
                return mu_current.detach(), sigma_current.detach()
            else:
                return mu_current.detach(), None

        # =====================================================================
        # CPU PATH: NumPy-based gradient_engine (LEGACY - slower but validated)
        # =====================================================================
        # Perform n_iterations of variational descent
        for iteration in range(self.n_iterations):
            # ==================================================================
            # Compute natural gradients via gradient_engine (per batch element)
            # ==================================================================

            batch_gradients = []

            for b in range(batch_size):
                # Convert to multi-agent system
                system = convert_torch_to_numpy_system(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    mu_prior=mu_prior,
                    phi=phi,
                    generators=self.generators,
                    config=self.config,
                    beta=beta_avg,
                    batch_idx=b,
                )

                # Compute natural gradients for all agents
                # NOTE: Sequential version is faster for transformers due to:
                # - Small per-agent computation time
                # - Large system object pickling overhead in parallel
                # - PyTorch + joblib interaction issues
                natural_grads = []
                for agent_idx in range(num_agents):
                    # Compute Euclidean gradients from active inference terms
                    # (prior, belief alignment, etc.)
                    euc_grads = _compute_agent_euclidean_gradients(system, agent_idx)

                    # ===========================================================
                    # ADD DISCRETE OBSERVATION GRADIENT (if provided)
                    # ===========================================================
                    # This is the key fix! The cross-entropy observation term.
                    # ∂F/∂μ = ... + W_out^T · (softmax(W_out·μ) - one_hot(target))
                    if discrete_obs_grad is not None:
                        # Add observation gradient to Euclidean gradient
                        # discrete_obs_grad: (B, N, K), extract [b, agent_idx, :]
                        obs_grad_np = discrete_obs_grad[b, agent_idx].cpu().numpy()  # (K,)

                        if euc_grads.grad_mu_q is not None:
                            euc_grads.grad_mu_q = euc_grads.grad_mu_q + obs_grad_np
                        else:
                            euc_grads.grad_mu_q = obs_grad_np

                    # Sanitize gradients before projection (prevent NaN)
                    euc_grads = _sanitize_euclidean_gradients(euc_grads, max_norm=1e3, debug=False)

                    # Project to natural gradients
                    nat_grads = project_to_natural_gradients(
                        system.agents[agent_idx], euc_grads
                    )
                    natural_grads.append(nat_grads)

                batch_gradients.append(natural_grads)

            # ==================================================================
            # Update parameters using natural gradients (VALIDATED APPROACH)
            # ==================================================================

            # Mean update: Simple Euclidean update
            # μ_new = μ + τ·Δμ
            delta_mu = torch.zeros_like(mu_current)
            for b in range(batch_size):
                for i, nat_grads in enumerate(batch_gradients[b]):
                    if nat_grads.delta_mu_q is not None:
                        delta_mu[b, i] = torch.from_numpy(nat_grads.delta_mu_q).to(device)

            mu_current = mu_current + self.lr * delta_mu

            # Covariance update: Use validated SPD retraction
            # Σ_new = retract_spd(Σ, τ·ΔΣ)
            if self.update_sigma and is_diagonal_cov:
                # Diagonal mode: gradient_engine computes full K×K gradients
                # We extract the diagonal and apply diagonal retraction
                for b in range(batch_size):
                    for i, nat_grads in enumerate(batch_gradients[b]):
                        if nat_grads.delta_Sigma_q is not None:
                            # Extract diagonal from full gradient
                            delta_sigma_diag = np.diag(nat_grads.delta_Sigma_q)
                            sigma_diag = sigma_current[b, i].detach().cpu().numpy()

                            # Diagonal exponential retraction: σ_new = σ * exp(τ * δσ / σ)
                            lr_scalar = self.lr.item() if isinstance(self.lr, torch.Tensor) else float(self.lr)
                            relative_update = lr_scalar * delta_sigma_diag / np.clip(sigma_diag, 1e-6, None)
                            relative_update = np.clip(relative_update, -0.2, 0.2)  # Trust region
                            sigma_new_diag = sigma_diag * np.exp(relative_update)
                            sigma_new_diag = np.clip(sigma_new_diag, 1e-6, 1e6)  # Clamp for stability

                            sigma_current[b, i] = torch.from_numpy(sigma_new_diag).to(device)
            elif self.update_sigma and not is_diagonal_cov:
                # Convert learning rate to float (may be torch tensor)
                lr_scalar = self.lr.item() if isinstance(self.lr, torch.Tensor) else float(self.lr)

                for b in range(batch_size):
                    for i, nat_grads in enumerate(batch_gradients[b]):
                        if nat_grads.delta_Sigma_q is not None:
                            # Convert to numpy for retraction (detach from computation graph)
                            Sigma_current = sigma_current[b, i].detach().cpu().numpy()
                            delta_Sigma = nat_grads.delta_Sigma_q

                            # Use validated SPD retraction (handles manifold geometry)
                            Sigma_new = retract_spd(
                                Sigma_current,
                                delta_Sigma,
                                step_size=lr_scalar,
                                trust_region=None,  # Could add trust region if needed
                                max_condition=None,  # Could add condition number limit
                                eps=1e-6,
                            )

                            # Convert back to PyTorch
                            sigma_current[b, i] = torch.from_numpy(Sigma_new).to(device)

        # Return updated parameters
        # CRITICAL: Detach from computation graph!
        # The natural gradients are already correct - don't let PyTorch backprop fight them
        if self.update_sigma:
            return mu_current.detach(), sigma_current.detach()
        else:
            # Return original sigma if update_sigma=False
            return mu_current.detach(), sigma.detach()


# =============================================================================
# Dynamic-β VFE: Full Active Inference with Evolving Attention (RECOMMENDED!)
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """
    Dynamic-β Variational FFN: Recomputes attention at each VFE step.

    This is the theoretically correct implementation where beliefs and attention
    co-evolve. At each integration step:

        1. Compute β from current beliefs: β_ij = softmax(-KL(q_i||Ω_ij[q_j])/κ)
        2. Compute full VFE gradient: ∂F/∂θ (includes ∂β/∂θ nonlinearity)
        3. Update beliefs via natural gradient descent
        4. (Optional) M-step: update priors toward beliefs
        5. Repeat

    Key difference from VariationalFFNGradientEngine:
        - GradientEngine: β computed once, held fixed during descent
        - Dynamic: β recomputed at each step → attention-belief co-evolution

    This enables emergent block structure in β as beliefs cluster.

    The ∂β/∂μ term is the principled nonlinearity (replaces GELU):
        ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    With dynamic β, this creates positive feedback:
        - Tokens with similar beliefs → higher β between them
        - Higher β → beliefs pulled closer together
        - → Cluster formation (meta-agents!)

    Complexity: O(n_steps × N² × K) - more expensive but theoretically sound
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
        lambda_belief: float = 1.0,  # Belief alignment weight
        kappa: float = 1.0,        # Attention temperature
        n_iterations: int = 10,    # VFE descent steps (more steps = deeper equilibration)
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True, # Update covariances?
        m_step_interval: int = 0,  # M-step every N steps (0 = no M-step)
        m_step_rate: float = 0.01, # Prior update rate toward beliefs
        diagonal_covariance: bool = False,  # Use diagonal Σ for efficiency
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
        # Pure FEP mode: learning via prior evolution (no backprop)
        max_seq_len: int = 512,    # Max sequence length for persistent priors
        pure_fep_mode: bool = False,  # Enable backprop-free learning
        prior_lr: float = 0.01,    # Learning rate for prior updates
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
    ):
        """
        Initialize dynamic-β VFE FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport (3, K, K)
            alpha: Weight for KL(q||p) self-coupling (prior anchoring)
            lambda_belief: Weight for belief alignment term Σ β_ij KL(q_i||q_j)
            kappa: Temperature for attention softmax (higher = softer)
            n_iterations: Number of VFE descent iterations per forward pass
            learnable_lr: If True, step size η is a learnable parameter
            update_sigma: If True, also update covariance matrices Σ
            m_step_interval: Run M-step every N iterations (0 = disabled)
            m_step_rate: How fast priors move toward beliefs in M-step
            diagonal_covariance: Use diagonal Σ for O(K) instead of O(K²)
            compute_sigma_align_grad: If True, compute sigma gradient from alignment term
            max_seq_len: Maximum sequence length for persistent priors (pure FEP mode)
            pure_fep_mode: If True, use persistent priors that evolve via prediction error
            prior_lr: Learning rate for prior updates in pure FEP mode
            irrep_dims: Block dimensions [d₁, d₂, ...] for memory-efficient block-diagonal KL.
                       When provided, exploits O(N² × Σᵢdᵢ²) vs O(N² × K²) - massive savings!
            chunk_size: Chunk size for memory-efficient processing. Processes N×N in C×C chunks.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.compute_sigma_align_grad = compute_sigma_align_grad

        # Memory-efficient options
        self.irrep_dims = irrep_dims
        self.chunk_size = chunk_size

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa

        # M-step configuration
        self.m_step_interval = m_step_interval
        self.m_step_rate = m_step_rate

        # Pure FEP mode: learning via prior evolution
        self.pure_fep_mode = pure_fep_mode
        self.max_seq_len = max_seq_len
        self.prior_lr = prior_lr

        if pure_fep_mode:
            # Position-dependent persistent priors (the LEARNING happens here!)
            # These evolve based on prediction-error-weighted beliefs
            self.register_buffer('prior_mu', torch.zeros(max_seq_len, embed_dim))
            self.register_buffer('prior_sigma', torch.ones(max_seq_len, embed_dim))
            self.register_buffer('prior_update_count', torch.zeros(max_seq_len))
            self.register_buffer('prior_initialized', torch.tensor(False))

        # Learnable step size
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - INITIAL attention (will be recomputed)
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) if diagonal
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs
        W_out: Optional[torch.Tensor] = None,    # (V, K) - output projection
        return_beta_history: bool = False,  # Return β evolution for analysis
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Dynamic VFE descent with β recomputation at each step.

        Flow at each iteration:
            1. β = softmax(-KL(q||Ω[q])/κ)  [RECOMPUTE from current beliefs]
            2. ∂F/∂μ = α(μ-μ_p)/σ_p + λΣβ(∂KL/∂μ) + Σ KL(∂β/∂μ) + ∂CE/∂μ
            3. μ ← μ - η·F⁻¹·∂F/∂μ  [Natural gradient descent]
            4. (Optional) σ ← retract_spd(σ, -η·∂F/∂σ)
            5. (Optional M-step) μ_p ← μ_p + rate·(μ - μ_p)

        Args:
            mu: Current belief means (B, N, K)
            beta: Initial attention weights (B, n_heads, N, N) - used only for first step
            mu_prior: Prior means from embeddings (B, N, K)
            phi: Gauge frames (B, N, 3)
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal
            mask: Causal mask (B, N, N) where 0 = cannot attend
            targets: Target tokens for observation term (B, N)
            W_out: Output projection for ∂CE/∂μ computation
            return_beta_history: If True, return list of β at each step

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (same shape as input) or None
            beta_history: List of β tensors if return_beta_history, else None
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # Initialize sigma if not provided
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        is_diagonal = sigma.dim() == 3

        # =====================================================================
        # PURE FEP MODE: Use persistent priors instead of embedding priors
        # =====================================================================
        if self.pure_fep_mode:
            # Initialize persistent priors from embeddings on first call
            self.initialize_priors_from_embeddings(mu_prior)

            # Get persistent priors (these evolve via prediction-error updates)
            # Note: persistent priors are always stored as diagonal (B, N, K)
            persistent_mu, persistent_sigma = self.get_persistent_priors(N, B, device)

            # Use persistent priors for VFE dynamics
            mu_p_current = persistent_mu.clone()

            # Convert diagonal persistent_sigma to full covariance if needed
            if persistent_sigma is not None:
                if is_diagonal:
                    # Both are diagonal - use directly
                    sigma_p = persistent_sigma.clone()
                else:
                    # Need full covariance (B, N, K, K) from diagonal (B, N, K)
                    sigma_p = torch.diag_embed(persistent_sigma)  # (B, N, K) -> (B, N, K, K)
            else:
                sigma_p = sigma.clone()
        else:
            # Standard mode: use embedding priors
            mu_p_current = mu_prior.clone()
            sigma_p = sigma.clone()

        # Current state (will evolve)
        mu_current = mu.clone()
        sigma_current = sigma.clone()

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None

        # Store observation info for fresh gradient computation
        has_observations = targets is not None and W_out is not None

        # =====================================================================
        # VFE Descent Loop with Dynamic β
        # =====================================================================
        for iteration in range(self.n_iterations):
            # =================================================================
            # STEP 1: Recompute attention β from current beliefs
            # =================================================================
            # β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)
            beta_current = compute_attention_weights(
                mu_q=mu_current,
                sigma_q=sigma_current,
                phi=phi,
                generators=self.generators,
                kappa=self.kappa,
                epsilon=eps,
                mask=mask,
                use_numba=False,  # Always use PyTorch for GPU
                return_kl=False,
                diagonal_covariance=is_diagonal,
                # Memory-efficient options
                irrep_dims=self.irrep_dims,
                chunk_size=self.chunk_size,
            )  # (B, N, N)

            if return_beta_history:
                beta_history.append(beta_current.detach().clone())

            # =================================================================
            # STEP 2: Compute VFE gradients with current β
            # =================================================================
            grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                mu_q=mu_current,
                sigma_q=sigma_current,
                mu_p=mu_p_current,
                sigma_p=sigma_p,
                beta=beta_current,  # USE RECOMPUTED β!
                phi=phi,
                generators=self.generators,
                alpha=self.alpha,
                lambda_belief=self.lambda_belief,
                kappa=self.kappa,
                eps=eps,
                compute_sigma_align_grad=self.compute_sigma_align_grad,
                # Memory-efficient options
                irrep_dims=self.irrep_dims,
                chunk_size=self.chunk_size,
            )

            # Add FRESH observation gradient (recomputed from current beliefs)
            if has_observations:
                with torch.no_grad():
                    logits = torch.matmul(mu_current, W_out.T)
                    probs = F.softmax(logits, dim=-1)
                    targets_valid = targets.clone()
                    targets_valid[targets == -1] = 0
                    one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()
                    mask_obs = (targets != -1).unsqueeze(-1).float()
                    one_hot = one_hot * mask_obs
                    grad_error = (probs - one_hot) * mask_obs
                    discrete_obs_grad = torch.matmul(grad_error, W_out)
                grad_mu = grad_mu + discrete_obs_grad

            # Clip for stability
            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            # =================================================================
            # STEP 3: Natural gradient projection
            # =================================================================
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_current, eps=eps
            )

            # =================================================================
            # STEP 4: Update beliefs (E-step) with WHITENED trust region
            # =================================================================
            # The natural gradient nat_grad_mu = Σ @ grad scales with σ
            # Use whitened trust region: ||δμ / √σ|| instead of raw norm
            delta_mu = -self.lr * nat_grad_mu

            # Whitened trust region for mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_current.clamp(min=eps))
                whitened_delta = delta_mu / sigma_sqrt
            else:
                sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=eps)
                whitened_delta = delta_mu / torch.sqrt(sigma_diag)

            whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
            mu_trust_region = 2.0  # Trust region on whitened norm
            scale = torch.clamp(mu_trust_region / (whitened_norm + eps), max=1.0)
            mu_current = mu_current + scale * delta_mu

            if self.update_sigma:
                # Use SPD-preserving retraction for stability with multiple iterations
                # Much smaller lr for sigma (matches simulation_runner: 0.005 vs 0.1)
                sigma_lr = self.lr * 0.05
                if is_diagonal:
                    sigma_current = retract_spd_diagonal_torch(
                        sigma_diag=sigma_current,
                        delta_sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.2,  # Max 20% change per iteration
                        eps=eps,
                    )
                else:
                    sigma_current = retract_spd_torch(
                        Sigma=sigma_current,
                        delta_Sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.1,  # Max 10% change per iteration
                        eps=eps,
                    )

            # =================================================================
            # STEP 5: Optional M-step (prior update)
            # =================================================================
            if self.m_step_interval > 0 and (iteration + 1) % self.m_step_interval == 0:
                # Move priors toward beliefs
                mu_p_current = mu_p_current + self.m_step_rate * (mu_current.detach() - mu_p_current)

        # Return results
        # NOTE: Previously returned .detach() which BREAKS gradient flow!
        # The VFE descent is an "inner loop" optimization, but we still need
        # gradients to flow through the final result to train the embeddings.
        # The detach was likely added to prevent backprop through all iterations,
        # but it completely breaks learning. If memory is an issue, consider
        # gradient checkpointing instead.
        if self.update_sigma:
            return mu_current, sigma_current, beta_history
        else:
            return mu_current, None, beta_history

    # =========================================================================
    # Pure FEP Mode: Backprop-free Learning via Prior Evolution
    # =========================================================================

    def initialize_priors_from_embeddings(self, mu_embed: torch.Tensor):
        """
        Initialize persistent priors from embedding priors (first batch only).

        In pure FEP mode, priors start at embedding values and then evolve.
        This provides a warm start for prior learning.

        Args:
            mu_embed: (B, N, K) embedding means - we use mean across batch
        """
        if not self.pure_fep_mode:
            return

        if self.prior_initialized:
            return

        B, N, K = mu_embed.shape
        N_update = min(N, self.max_seq_len)

        # Initialize from mean of embedding priors
        with torch.no_grad():
            self.prior_mu[:N_update] = mu_embed[:, :N_update].mean(dim=0)
            self.prior_initialized.fill_(True)

    def get_persistent_priors(self, seq_len: int, batch_size: int, device: torch.device):
        """
        Get persistent priors for the current sequence, expanded across batch.

        Args:
            seq_len: Current sequence length N
            batch_size: Batch size B
            device: Target device

        Returns:
            mu_prior: (B, N, K) persistent prior means
            sigma_prior: (B, N, K) persistent prior variances (diagonal)
        """
        if not self.pure_fep_mode:
            return None, None

        N = min(seq_len, self.max_seq_len)

        # Expand priors across batch
        mu_prior = self.prior_mu[:N].unsqueeze(0).expand(batch_size, -1, -1)
        sigma_prior = self.prior_sigma[:N].unsqueeze(0).expand(batch_size, -1, -1)

        # Handle sequences longer than max_seq_len (pad with defaults)
        if seq_len > self.max_seq_len:
            pad_len = seq_len - self.max_seq_len
            mu_pad = torch.zeros(batch_size, pad_len, self.embed_dim, device=device)
            sigma_pad = torch.ones(batch_size, pad_len, self.embed_dim, device=device)
            mu_prior = torch.cat([mu_prior, mu_pad], dim=1)
            sigma_prior = torch.cat([sigma_prior, sigma_pad], dim=1)

        return mu_prior, sigma_prior

    def update_priors_from_beliefs(
        self,
        mu_beliefs: torch.Tensor,       # (B, N, K) final beliefs after VFE
        sigma_beliefs: torch.Tensor,    # (B, N, K) belief variances
        prediction_errors: torch.Tensor,  # (B, N) per-position CE loss
        lr: Optional[float] = None,
    ):
        """
        Update persistent priors toward beliefs that successfully predicted.

        This is the LEARNING mechanism in pure FEP mode (replaces backprop):
        - Beliefs with low prediction error are "good" - priors should move toward them
        - Beliefs with high prediction error are "bad" - priors should ignore them
        - Weighting by softmax(-error) gives soft EM-like updates

        The key insight: CE is INSIDE the VFE during forward pass, so beliefs
        have already adjusted to minimize prediction error. We now consolidate
        successful beliefs into priors for future use.

        Args:
            mu_beliefs: (B, N, K) evolved belief means
            sigma_beliefs: (B, N, K) evolved belief variances
            prediction_errors: (B, N) per-position cross-entropy loss
            lr: Learning rate (uses self.prior_lr if None)
        """
        if not self.pure_fep_mode:
            return

        lr = lr if lr is not None else self.prior_lr
        B, N, K = mu_beliefs.shape
        N_update = min(N, self.max_seq_len)
        eps = 1e-6

        with torch.no_grad():
            # Compute position-wise weights from prediction errors
            # Low error = high weight (successful predictions should update priors)
            # Use softmax over batch dimension for each position
            errors_clamped = prediction_errors[:, :N_update].clamp(min=eps, max=20.0)
            weights = F.softmax(-errors_clamped, dim=0)  # (B, N_update)

            # Weighted mean of beliefs across batch for each position
            # Shape: (B, N_update, K) * (B, N_update, 1) -> sum over B -> (N_update, K)
            weighted_mu = (mu_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)
            weighted_sigma = (sigma_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)

            # Compute confidence: inverse mean error per position
            # Higher confidence = larger update
            mean_error = errors_clamped.mean(dim=0)  # (N_update,)
            confidence = 1.0 / (1.0 + mean_error)  # (N_update,) in [0, 0.5]

            # Adaptive learning rate: scale by confidence
            effective_lr = lr * confidence.unsqueeze(-1)  # (N_update, 1)

            # EMA update toward weighted beliefs
            # prior <- (1 - lr) * prior + lr * belief
            self.prior_mu[:N_update] = (
                (1.0 - effective_lr) * self.prior_mu[:N_update] +
                effective_lr * weighted_mu
            )

            # Update sigma with smaller learning rate (more stable)
            sigma_lr = effective_lr * 0.1
            self.prior_sigma[:N_update] = (
                (1.0 - sigma_lr) * self.prior_sigma[:N_update] +
                sigma_lr * weighted_sigma
            ).clamp(min=eps)

            # Track update counts
            self.prior_update_count[:N_update] += 1

    def get_prior_stats(self) -> Dict[str, float]:
        """Get statistics about persistent priors for logging."""
        if not self.pure_fep_mode:
            return {}

        with torch.no_grad():
            active_mask = self.prior_update_count > 0
            n_active = active_mask.sum().item()

            if n_active == 0:
                return {'prior_active_positions': 0}

            active_mu = self.prior_mu[active_mask]
            active_sigma = self.prior_sigma[active_mask]

            return {
                'prior_active_positions': n_active,
                'prior_mu_mean': active_mu.mean().item(),
                'prior_mu_std': active_mu.std().item(),
                'prior_sigma_mean': active_sigma.mean().item(),
                'prior_update_count_mean': self.prior_update_count[active_mask].mean().item(),
            }

    def extra_repr(self) -> str:
        base = (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}, "
            f"m_step_interval={self.m_step_interval}"
        )
        if self.pure_fep_mode:
            base += f", pure_fep_mode=True, prior_lr={self.prior_lr}"
        return base


# =============================================================================
# Dynamic-β VFE with Optional Stabilizations (default: principled only)
# =============================================================================

class VariationalFFNDynamicStable(nn.Module):
    """
    Dynamic-β VFE FFN with optional (ad-hoc) stabilization features.

    By default, this is identical to VariationalFFNDynamic (first-principles).
    Enable stabilization options only if training is unstable.

    PRINCIPLED (always on):
    - Fresh observation gradients: ∂CE/∂μ recomputed from current beliefs

    AD-HOC (opt-in, default OFF):
    - Temperature annealing: κ_start→κ during iterations (NOT in theory)
    - Gradient norm balancing: Scale gradients to equal magnitude (ad-hoc)
    - Entropy penalty: Extra term penalizing uniform β (adds to VFE)
    - Self-attention damping: Suppress diagonal β (artificial)

    Use the ad-hoc features for debugging unstable training, but the
    first-principles version (defaults) should be preferred.
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
        lambda_belief: float = 1.0,  # Belief alignment weight
        kappa: float = 1.0,        # Attention temperature
        kappa_start: float = None, # AD-HOC: Initial temp for annealing (None = no annealing)
        n_iterations: int = 10,    # VFE descent steps
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True, # Update covariances?
        m_step_interval: int = 0,  # M-step every N steps (0 = disabled)
        m_step_rate: float = 0.01, # Prior update rate
        diagonal_covariance: bool = False,
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
        # AD-HOC stabilization (all default OFF for first-principles)
        balance_gradients: bool = False,  # AD-HOC: Balance gradient norms
        obs_grad_weight: float = 1.0,     # Relative weight of observation gradient
        entropy_penalty: float = 0.0,     # AD-HOC: Penalty for uniform β
        self_attn_damping: float = 0.0,   # AD-HOC: Reduce diagonal β (0-1)
        grad_clip: float = 1e3,           # Numerical stability (not ad-hoc, just overflow prevention)
    ):
        """
        Initialize stabilized dynamic-β VFE FFN.

        Args:
            embed_dim: K - belief vector dimension
            generators: SO(3) generators (3, K, K)
            alpha: Prior anchoring weight
            lambda_belief: Belief alignment weight
            kappa: Target temperature (reached at final iteration)
            kappa_start: Initial temperature (higher for soft start)
            n_iterations: Number of VFE iterations
            learnable_lr: Learn step size as parameter
            update_sigma: Also update covariance Σ
            m_step_interval: Run M-step every N iterations (0=off)
            m_step_rate: Prior update rate in M-step
            diagonal_covariance: Use diagonal Σ for efficiency
            compute_sigma_align_grad: If True, compute sigma gradient from alignment term
            balance_gradients: Auto-balance gradient magnitudes
            obs_grad_weight: Relative importance of observation gradient
            entropy_penalty: Penalty coefficient for uniform β
            self_attn_damping: Factor to reduce self-attention (0-1)
            grad_clip: Maximum gradient magnitude per component
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = n_iterations
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.compute_sigma_align_grad = compute_sigma_align_grad

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa
        # kappa_start = None means no annealing (use kappa as constant)
        self.kappa_start = kappa_start if kappa_start is not None else kappa

        # M-step
        self.m_step_interval = m_step_interval
        self.m_step_rate = m_step_rate

        # Stabilization
        self.balance_gradients = balance_gradients
        self.obs_grad_weight = obs_grad_weight
        self.entropy_penalty = entropy_penalty
        self.self_attn_damping = self_attn_damping
        self.grad_clip = grad_clip

        # Learnable step size
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def _compute_obs_gradient(
        self,
        mu: torch.Tensor,      # (B, N, K) current beliefs
        targets: torch.Tensor,  # (B, N) target tokens
        W_out: torch.Tensor,   # (V, K) output projection
    ) -> torch.Tensor:
        """
        Compute fresh observation gradient: ∂CE/∂μ = W^T(softmax(Wμ) - one_hot(y))

        This is called at each iteration, not frozen at step 0!
        """
        with torch.no_grad():
            logits = torch.matmul(mu, W_out.T)  # (B, N, V)
            probs = F.softmax(logits, dim=-1)

            targets_valid = targets.clone()
            targets_valid[targets == -1] = 0
            one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()

            mask_obs = (targets != -1).unsqueeze(-1).float()
            one_hot = one_hot * mask_obs

            grad_error = (probs - one_hot) * mask_obs
            obs_grad = torch.matmul(grad_error, W_out)  # (B, N, K)

        return obs_grad

    def _apply_entropy_penalty(
        self,
        beta: torch.Tensor,  # (B, N, N)
        grad_mu: torch.Tensor,  # (B, N, K)
        mu: torch.Tensor,
        phi: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Add gradient term that penalizes high-entropy (uniform) attention.

        When β is uniform, entropy H(β_i) = log(N).
        We add a penalty that pushes beliefs to be more distinct,
        which increases KL variance and peaks the attention.

        Penalty: λ_ent * mean_i[H(β_i)]
        Gradient: pushes μ to increase KL variance
        """
        if self.entropy_penalty <= 0:
            return grad_mu

        B, N, N_key = beta.shape

        # Compute row-wise entropy: H(β_i) = -Σ_j β_ij log(β_ij)
        log_beta = torch.log(beta + eps)
        entropy_per_query = -torch.sum(beta * log_beta, dim=-1)  # (B, N)

        # Max entropy is log(N) - penalize when close to max
        max_entropy = np.log(N_key)
        entropy_ratio = entropy_per_query / max_entropy  # (B, N) in [0, 1]

        # Only penalize when entropy is high (>0.8 of max)
        penalty_mask = (entropy_ratio > 0.8).float()

        # Simple penalty: push μ away from mean (increase variance)
        mu_mean = mu.mean(dim=1, keepdim=True)  # (B, 1, K)
        anti_collapse_grad = self.entropy_penalty * penalty_mask.unsqueeze(-1) * (mu_mean - mu)

        return grad_mu + anti_collapse_grad

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K)
        beta: torch.Tensor,        # (B, n_heads, N, N) - initial, will be recomputed
        mu_prior: torch.Tensor,    # (B, N, K)
        phi: torch.Tensor,         # (B, N, 3)
        sigma: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        W_out: Optional[torch.Tensor] = None,
        return_beta_history: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Stabilized dynamic VFE descent.

        Flow at each iteration t:
            1. κ(t) = kappa_start + (kappa - kappa_start) * t / n_iterations  [anneal]
            2. β = softmax(-KL/κ(t))  [recompute with current κ]
            3. Apply self-attention damping: β_ii *= (1 - damping)
            4. ∂F_align/∂μ = VFE gradient (alignment + prior)
            5. ∂F_obs/∂μ = observation gradient (FRESH, using current μ)
            6. Balance gradients if enabled
            7. Apply entropy penalty if β too uniform
            8. Natural gradient projection + update
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # Initialize sigma
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        is_diagonal = sigma.dim() == 3
        sigma_p = sigma.clone()

        # Current state
        mu_current = mu.clone()
        sigma_current = sigma.clone()
        mu_p_current = mu_prior.clone()

        beta_history = [] if return_beta_history else None

        # =====================================================================
        # VFE Descent with Stabilization
        # =====================================================================
        for iteration in range(self.n_iterations):
            # =================================================================
            # STEP 1: Compute annealed temperature
            # =================================================================
            progress = iteration / max(1, self.n_iterations - 1)
            kappa_t = self.kappa_start + (self.kappa - self.kappa_start) * progress

            # =================================================================
            # STEP 2: Recompute β with current temperature
            # =================================================================
            beta_current = compute_attention_weights(
                mu_q=mu_current,
                sigma_q=sigma_current,
                phi=phi,
                generators=self.generators,
                kappa=kappa_t,  # Use annealed temperature!
                epsilon=eps,
                mask=mask,
                use_numba=False,
                return_kl=False,
                diagonal_covariance=is_diagonal,
            )

            # =================================================================
            # STEP 3: Apply self-attention damping
            # =================================================================
            if self.self_attn_damping > 0:
                # Reduce diagonal (self-attention) to encourage cross-token comm
                diag_mask = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)
                beta_current = beta_current * (1 - self.self_attn_damping * diag_mask)
                # Renormalize
                beta_current = beta_current / (beta_current.sum(dim=-1, keepdim=True) + eps)

            if return_beta_history:
                beta_history.append(beta_current.detach().clone())

            # =================================================================
            # STEP 4: Compute alignment gradient (VFE terms)
            # =================================================================
            grad_mu_align, grad_sigma = compute_vfe_gradients_gpu(
                mu_q=mu_current,
                sigma_q=sigma_current,
                mu_p=mu_p_current,
                sigma_p=sigma_p,
                beta=beta_current,
                phi=phi,
                generators=self.generators,
                alpha=self.alpha,
                lambda_belief=self.lambda_belief,
                kappa=kappa_t,
                eps=eps,
                compute_sigma_align_grad=self.compute_sigma_align_grad,
            )

            # =================================================================
            # STEP 5: Compute FRESH observation gradient
            # =================================================================
            if targets is not None and W_out is not None:
                grad_mu_obs = self._compute_obs_gradient(mu_current, targets, W_out)
            else:
                grad_mu_obs = torch.zeros_like(mu_current)

            # =================================================================
            # STEP 6: Balance gradient magnitudes
            # =================================================================
            if self.balance_gradients:
                # Compute norms
                norm_align = grad_mu_align.norm() + eps
                norm_obs = grad_mu_obs.norm() + eps

                # Scale so both have similar magnitude
                if norm_obs > eps:
                    # Scale obs gradient to match alignment, then weight
                    scale = (norm_align / norm_obs) * self.obs_grad_weight
                    grad_mu_obs = grad_mu_obs * scale
            else:
                grad_mu_obs = grad_mu_obs * self.obs_grad_weight

            # Combine gradients
            grad_mu = grad_mu_align + grad_mu_obs

            # =================================================================
            # STEP 7: Apply entropy penalty if β too uniform
            # =================================================================
            grad_mu = self._apply_entropy_penalty(
                beta_current, grad_mu, mu_current, phi, eps
            )

            # =================================================================
            # STEP 8: Clip and project to natural gradient
            # =================================================================
            grad_mu = torch.clamp(grad_mu, min=-self.grad_clip, max=self.grad_clip)
            grad_sigma = torch.clamp(grad_sigma, min=-self.grad_clip, max=self.grad_clip)

            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_current, eps=eps
            )

            # =================================================================
            # STEP 9: Update beliefs with WHITENED trust region
            # =================================================================
            # The natural gradient nat_grad_mu = Σ @ grad scales with σ
            # Use whitened trust region: ||δμ / √σ|| instead of raw norm
            delta_mu = -self.lr * nat_grad_mu

            # Whitened trust region for mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_current.clamp(min=eps))
                whitened_delta = delta_mu / sigma_sqrt
            else:
                sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clamp(min=eps)
                whitened_delta = delta_mu / torch.sqrt(sigma_diag)

            whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
            mu_trust_region = 2.0  # Trust region on whitened norm
            scale = torch.clamp(mu_trust_region / (whitened_norm + eps), max=1.0)
            mu_current = mu_current + scale * delta_mu

            if self.update_sigma:
                # Use SPD-preserving retraction for stability with multiple iterations
                # Much smaller lr for sigma (matches simulation_runner: 0.005 vs 0.1)
                sigma_lr = self.lr * 0.05
                if is_diagonal:
                    sigma_current = retract_spd_diagonal_torch(
                        sigma_diag=sigma_current,
                        delta_sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.2,  # Max 20% change per iteration
                        eps=eps,
                    )
                else:
                    sigma_current = retract_spd_torch(
                        Sigma=sigma_current,
                        delta_Sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.1,  # Max 10% change per iteration
                        eps=eps,
                    )

            # =================================================================
            # STEP 10: Optional M-step
            # =================================================================
            if self.m_step_interval > 0 and (iteration + 1) % self.m_step_interval == 0:
                mu_p_current = mu_p_current + self.m_step_rate * (mu_current.detach() - mu_p_current)

        # Return results
        # NOTE: Previously returned .detach() which BREAKS gradient flow!
        # See comment in VariationalFFNDynamic for explanation.
        if self.update_sigma:
            return mu_current, sigma_current, beta_history
        else:
            return mu_current, None, beta_history

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, "
            f"kappa={self.kappa}, kappa_start={self.kappa_start}, "
            f"balance_gradients={self.balance_gradients}, "
            f"entropy_penalty={self.entropy_penalty}"
        )


# =============================================================================
# Legacy Implementations (for backward compatibility)
# =============================================================================

class VariationalFFNApproximate(nn.Module):
    """
    Approximate variational descent FFN (omit ∂β_ij/∂μ_i term).

    LEGACY: Kept for backward compatibility.
    RECOMMENDATION: Use VariationalFFNGradientEngine instead!

    Gradient:
        ∂F/∂μ_i ≈ α·(μ_i - μ_p) + Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)

    Update:
        μ_new = μ - η · ∂F/∂μ_i

    This is a good first-order approximation that captures:
    - Prior pull toward μ_p
    - Weighted neighbor alignment

    But misses:
    - Covariance updates
    - How attention changes with μ_i (second-order)
    - Natural gradient projection

    Complexity: O(N²·K) - same as standard attention
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.001,      # Prior weight
        tau_eff: float = 1.0,      # Effective temperature
        n_iterations: int = 1,     # Number of inference steps
        learnable_lr: bool = True, # Learn step size?
    ):
        """
        Initialize approximate variational FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport
            alpha: Weight for prior term
            tau_eff: Effective temperature (higher = weaker coupling)
            n_iterations: Number of descent iterations per forward pass
            learnable_lr: If True, learning rate η is a learnable parameter
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)  # (3, K, K)
        self.alpha = alpha
        self.tau_eff = tau_eff
        self.n_iterations = n_iterations

        # Learnable step size (or fixed)
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))  # Initialize to 0.1
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - attention weights
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        mask: Optional[torch.Tensor] = None,  # (B, N, N) causal mask
    ) -> torch.Tensor:
        """
        One step of approximate variational descent.

        Args:
            mu: Current belief means (B, N, K)
            beta: Attention weights β_ij (B, n_heads, N, N)
            mu_prior: Prior means p_i (B, N, K)
            phi: Gauge frames φ_i (B, N, 3)
            mask: Causal mask (B, N, N) or None

        Returns:
            mu_new: Updated beliefs after variational descent (B, N, K)
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Average attention weights over heads for simplicity
        # Could also compute separate gradients per head
        beta_avg = beta.mean(dim=1)  # (B, N, N)

        # Apply mask if provided
        if mask is not None:
            beta_avg = beta_avg * mask
            # Renormalize
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        # Perform n_iterations of variational descent
        mu_current = mu

        for _ in range(self.n_iterations):
            # ===========================================================
            # Compute gradient ∂F/∂μ_i (approximate)
            # ===========================================================

            # 1. Prior gradient: α·(μ_i - μ_p)
            grad_prior = self.alpha * (mu_current - mu_prior)  # (B, N, K)

            # 2. Coupling gradient: Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)
            # Need to compute Ω_ij μ_j for all pairs (i,j)

            # Transport all μ_j by Ω_ij
            mu_transported = self._transport_beliefs(mu_current, phi)  # (B, N, N, K)

            # Compute differences: μ_i - Ω_ij μ_j
            # mu_i: (B, N, 1, K), mu_transported: (B, N, N, K)
            delta_mu = mu_current.unsqueeze(2) - mu_transported  # (B, N, N, K)

            # Weight by β_ij/τ: (B, N, N, 1) * (B, N, N, K)
            weighted_delta = (beta_avg.unsqueeze(-1) / self.tau_eff) * delta_mu  # (B, N, N, K)

            # Sum over neighbors j
            grad_coupling = weighted_delta.sum(dim=2)  # (B, N, K)

            # ===========================================================
            # Total gradient (approximate)
            # ===========================================================
            grad_total = grad_prior + grad_coupling  # (B, N, K)

            # ===========================================================
            # Variational descent update
            # ===========================================================
            mu_current = mu_current - self.lr * grad_total

        return mu_current

    def _transport_beliefs(
        self,
        mu: torch.Tensor,  # (B, N, K)
        phi: torch.Tensor,  # (B, N, 3)
    ) -> torch.Tensor:
        """
        Compute transported beliefs Ω_ij μ_j for all pairs (i,j).

        Ω_ij = exp(φ_i) · exp(-φ_j)

        Args:
            mu: Belief means (B, N, K)
            phi: Gauge frames (B, N, 3)

        Returns:
            mu_transported: (B, N, N, K) where [b,i,j,:] = Ω_ij μ_j
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Compute exp(φ_i) and exp(-φ_j) for all agents
        # Using matrix exponential on SO(3)

        # φ_i → Lie algebra element: φ · generators
        # phi: (B, N, 3), generators: (3, K, K)

        # Expand for broadcasting
        phi_expanded = phi.unsqueeze(-1).unsqueeze(-1)  # (B, N, 3, 1, 1)
        gen_expanded = self.generators.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, K, K)

        # φ·G = Σ_a φ^a G_a
        phi_algebra = (phi_expanded * gen_expanded).sum(dim=2)  # (B, N, K, K)

        # Matrix exponential: exp(φ·G)
        Omega_i = torch.matrix_exp(phi_algebra)  # (B, N, K, K)
        Omega_j_inv = torch.matrix_exp(-phi_algebra)  # (B, N, K, K)

        # Compute Ω_ij = Omega_i @ Omega_j_inv for all pairs
        # Omega_i: (B, N, 1, K, K), Omega_j_inv: (B, 1, N, K, K)
        Omega_ij = torch.matmul(
            Omega_i.unsqueeze(2),      # (B, N, 1, K, K)
            Omega_j_inv.unsqueeze(1)    # (B, 1, N, K, K)
        )  # (B, N, N, K, K)

        # Transport beliefs: Ω_ij μ_j
        # mu: (B, 1, N, K, 1), Omega_ij: (B, N, N, K, K)
        mu_transported = torch.matmul(
            Omega_ij,                          # (B, N, N, K, K)
            mu.unsqueeze(1).unsqueeze(-1)      # (B, 1, N, K, 1)
        ).squeeze(-1)  # (B, N, N, K)

        return mu_transported


class VariationalFFNFull(nn.Module):
    """
    FULL variational descent FFN (includes ∂β_ij/∂μ_i term).

    LEGACY: Kept for backward compatibility.
    RECOMMENDATION: Use VariationalFFNGradientEngine instead!

    Complete gradient from active inference:
        ∂F/∂μ_i = α·(μ_i - μ_p)
                + Σ_j (β_ij/τ)·(μ_i - Ω_ij μ_j)                [TERM 1]
                + Σ_j Σ_k (∂β_ij/∂KL_ik)·KL_ij·∂KL_ik/∂μ_i  [TERM 2]

    TERM 2 accounts for how attention weights change as beliefs change.
    This is the FULL gauge-invariant gradient from validated code!

    More complex but theoretically exact.

    Complexity: O(N³·K) due to triple sum over (i,j,k)
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,
        alpha: float = 0.001,
        tau_eff: float = 1.0,
        kappa: float = 1.0,        # Softmax temperature
        n_iterations: int = 1,
        learnable_lr: bool = True,
    ):
        """
        Initialize full variational FFN.

        Args:
            embed_dim: K
            generators: SO(3) generators
            alpha: Prior weight
            tau_eff: Coupling temperature
            kappa: Softmax temperature (for β_ij)
            n_iterations: Descent iterations
            learnable_lr: Learn step size?
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.alpha = alpha
        self.tau_eff = tau_eff
        self.kappa = kappa
        self.n_iterations = n_iterations

        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.05))  # Smaller for stability
        else:
            self.register_buffer('lr', torch.tensor(0.05))

    def forward(
        self,
        mu: torch.Tensor,
        beta: torch.Tensor,
        mu_prior: torch.Tensor,
        phi: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) covariances
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full variational descent with second-order terms.

        NOTE: This is more expensive but theoretically correct!
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Average attention over heads
        beta_avg = beta.mean(dim=1)  # (B, N, N)
        if mask is not None:
            beta_avg = beta_avg * mask
            beta_sum = beta_avg.sum(dim=-1, keepdim=True) + 1e-8
            beta_avg = beta_avg / beta_sum

        mu_current = mu

        for _ in range(self.n_iterations):
            # ===========================================================
            # First-order terms (same as approximate)
            # ===========================================================
            grad_prior = self.alpha * (mu_current - mu_prior)

            mu_transported = self._transport_beliefs(mu_current, phi)
            delta_mu = mu_current.unsqueeze(2) - mu_transported
            weighted_delta = (beta_avg.unsqueeze(-1) / self.tau_eff) * delta_mu
            grad_coupling_first = weighted_delta.sum(dim=2)

            # ===========================================================
            # Second-order term: Σ_j Σ_k (∂β_ij/∂KL_ik)·KL_ij·∂KL_ik/∂μ_i
            # ===========================================================
            grad_coupling_second = self._compute_second_order_gradient(
                mu_current, phi, beta_avg, sigma
            )

            # ===========================================================
            # Total gradient (FULL)
            # ===========================================================
            grad_total = grad_prior + grad_coupling_first + grad_coupling_second

            # ===========================================================
            # Update
            # ===========================================================
            mu_current = mu_current - self.lr * grad_total

        return mu_current

    def _compute_second_order_gradient(
        self,
        mu: torch.Tensor,  # (B, N, K)
        phi: torch.Tensor,  # (B, N, 3)
        beta: torch.Tensor,  # (B, N, N)
        sigma: Optional[torch.Tensor],  # (B, N, K, K) or None
    ) -> torch.Tensor:
        """
        Compute second-order softmax coupling gradient.

        For each agent i:
            ∂F/∂μ_i += Σ_j Σ_k (∂β_ij/∂KL_ik) · KL_ij · ∂KL_ik/∂μ_i

        Where:
            ∂β_ij/∂KL_ik = (β_ij/κ) · [δ_jk - β_ik]

        This is the key term that accounts for how attention changes!

        Returns:
            grad: (B, N, K) second-order gradient contribution
        """
        batch_size, num_agents, K = mu.shape
        device = mu.device

        # Transport all beliefs
        mu_transported = self._transport_beliefs(mu, phi)  # (B, N, N, K)

        # Compute all KL divergences KL(μ_i || Ω_ij μ_j)
        # Simplified: using || · ||² distance as proxy
        # (Full version would use proper Gaussian KL with covariances)

        kl_ij = torch.sum(
            (mu.unsqueeze(2) - mu_transported) ** 2,
            dim=-1
        ) / (2.0 * self.tau_eff)  # (B, N, N)

        # Compute ∂KL_ik/∂μ_i for all k
        # Gradient of KL w.r.t. source: ∂KL(μ_i||Ω_ik μ_k)/∂μ_i = (μ_i - Ω_ik μ_k) / τ
        grad_kl_wrt_mu = (mu.unsqueeze(2) - mu_transported) / self.tau_eff  # (B, N, N, K)

        # Compute ∂β_ij/∂KL_ik = (β_ij/κ) · [δ_jk - β_ik]
        # This is the Jacobian of softmax

        # For each (i,j,k) triple, compute contribution:
        # (β_ij/κ) · [δ_jk - β_ik] · KL_ij · ∂KL_ik/∂μ_i

        grad_second = torch.zeros_like(mu)  # (B, N, K)

        for i in range(num_agents):
            for j in range(num_agents):
                for k in range(num_agents):
                    # Kronecker delta
                    delta_jk = 1.0 if j == k else 0.0

                    # ∂β_ij/∂KL_ik
                    d_beta_ij_d_kl_ik = (beta[:, i, j] / self.kappa) * (delta_jk - beta[:, i, k])
                    # Shape: (B,)

                    # KL_ij
                    kl_ij_val = kl_ij[:, i, j]  # (B,)

                    # ∂KL_ik/∂μ_i
                    grad_kl_ik = grad_kl_wrt_mu[:, i, k, :]  # (B, K)

                    # Product: (B,) * (B,) * (B, K)
                    contrib = (d_beta_ij_d_kl_ik * kl_ij_val).unsqueeze(-1) * grad_kl_ik

                    grad_second[:, i, :] += contrib

        return grad_second

    def _transport_beliefs(self, mu, phi):
        """Same as approximate version."""
        batch_size, num_agents, K = mu.shape
        device = mu.device

        phi_expanded = phi.unsqueeze(-1).unsqueeze(-1)
        gen_expanded = self.generators.unsqueeze(0).unsqueeze(0)
        phi_algebra = (phi_expanded * gen_expanded).sum(dim=2)

        Omega_i = torch.matrix_exp(phi_algebra)
        Omega_j_inv = torch.matrix_exp(-phi_algebra)

        Omega_ij = torch.matmul(
            Omega_i.unsqueeze(2),
            Omega_j_inv.unsqueeze(1)
        )

        mu_transported = torch.matmul(
            Omega_ij,
            mu.unsqueeze(1).unsqueeze(-1)
        ).squeeze(-1)

        return mu_transported


# =============================================================================
# Helper: Replace FFN in Transformer Block
# =============================================================================

def replace_ffn_with_variational(
    transformer_block,
    variant: str = 'gradient_engine',  # 'gradient_engine', 'approximate', or 'full'
    generators: torch.Tensor = None,
    **kwargs
):
    """
    Replace learned FFN with variational descent.

    Args:
        transformer_block: TransformerBlock instance
        variant: 'gradient_engine' (recommended), 'approximate', or 'full' (legacy)
        generators: SO(3) generators (3, K, K)
        **kwargs: Additional parameters for VariationalFFN

    Returns:
        Modified transformer block
    """
    if generators is None:
        raise ValueError("Must provide SO(3) generators")

    embed_dim = transformer_block.ffn.net[0].in_features

    if variant == 'gradient_engine':
        variational_ffn = VariationalFFNGradientEngine(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    elif variant == 'approximate':
        variational_ffn = VariationalFFNApproximate(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    elif variant == 'full':
        variational_ffn = VariationalFFNFull(
            embed_dim=embed_dim,
            generators=generators,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Store original FFN (for comparison/ablation)
    transformer_block.ffn_original = transformer_block.ffn

    # Replace with variational version
    transformer_block.ffn_variational = variational_ffn
    transformer_block.use_variational = True

    return transformer_block
