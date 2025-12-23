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
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

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
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

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
        diagonal_covariance: bool = False,  # Use diagonal Σ for efficiency
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
        # Phi (gauge frame) evolution via VFE gradients
        update_phi: bool = False,  # If True, update phi via ∂F/∂φ
        phi_lr: float = 0.05,      # Learning rate for phi updates
        phi_max_norm: float = 3.14159,  # Max norm for phi (π = 180° rotation)
        # Pure FEP mode: learning via prior evolution (no backprop)
        max_seq_len: int = 512,    # Max sequence length for persistent priors
        pure_fep_mode: bool = False,  # Enable backprop-free learning
        prior_lr: float = 0.01,    # Learning rate for prior updates
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
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
            diagonal_covariance: Use diagonal Σ for O(K) instead of O(K²)
            compute_sigma_align_grad: If True, compute sigma gradient from alignment term
            max_seq_len: Maximum sequence length for persistent priors (pure FEP mode)
            pure_fep_mode: If True, use persistent priors that evolve via prediction error
            prior_lr: Learning rate for prior updates in pure FEP mode
            irrep_dims: Block dimensions [d₁, d₂, ...] for memory-efficient block-diagonal KL.
                       When provided, exploits O(N² × Σᵢdᵢ²) vs O(N² × K²) - massive savings!
            chunk_size: Chunk size for memory-efficient processing. Processes N×N in C×C chunks.
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0 always.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = n_iterations
        self.mask_self_attention = mask_self_attention
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.compute_sigma_align_grad = compute_sigma_align_grad

        # Phi evolution via VFE gradients (principled approach)
        self.update_phi = update_phi
        self.phi_lr = phi_lr
        self.phi_max_norm = phi_max_norm

        # Memory-efficient options
        self.irrep_dims = irrep_dims
        self.chunk_size = chunk_size

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[list]]:
        """
        Dynamic VFE descent with β recomputation at each step.

        Flow at each iteration:
            1. β = softmax(-KL(q||Ω[q])/κ)  [RECOMPUTE from current beliefs]
            2. ∂F/∂μ = α(μ-μ_p)/σ_p + λΣβ(∂KL/∂μ) + Σ KL(∂β/∂μ) + ∂CE/∂μ
            3. μ ← μ - η·F⁻¹·∂F/∂μ  [Natural gradient descent]
            4. (Optional) σ ← retract_spd(σ, -η·∂F/∂σ)
            5. (Optional) φ ← φ - η_φ·∂F/∂φ  [VFE gradient descent on gauge frames]
            6. (Optional M-step) μ_p ← μ_p + rate·(μ - μ_p)

        Args:
            mu: Current belief means (B, N, K)
            beta: Initial attention weights (B, n_heads, N, N) - used only for first step
            mu_prior: Prior means from embeddings (B, N, K)
            phi: Gauge frames (B, N, phi_dim)
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal
            mask: Causal mask (B, N, N) where 0 = cannot attend
            targets: Target tokens for observation term (B, N)
            W_out: Output projection for ∂CE/∂μ computation
            return_beta_history: If True, return list of β at each step

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (same shape as input) or None
            phi_new: Updated gauge frames (B, N, phi_dim)
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

        # Squeeze trailing singleton dimensions for robustness
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

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
                # Self-attention masking
                mask_self_attention=self.mask_self_attention,
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
            # NOTE: Previously used torch.no_grad() which BLOCKED gradient flow to embeddings!
            # The observation gradient guides VFE descent AND must train embeddings.
            # Removing no_grad() allows embeddings to learn from VFE dynamics.
            if has_observations:
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
        # STEP 5: Optional Phi Evolution via VFE Gradient
        # =================================================================
        # This is the PRINCIPLED approach: φ evolves via ∂F/∂φ, not a neural net.
        # The belief alignment term F_align = λ·Σ β_ij KL(q_i || Ω_ij[q_j])
        # depends on φ through the transport operator Ω_ij = exp(φ_i)·exp(-φ_j).
        phi_current = phi
        # Only update phi during training (when gradients are enabled)
        if self.update_phi and torch.is_grad_enabled():
            # Enable gradients for phi
            phi_for_grad = phi.clone().requires_grad_(True)

            # Recompute attention with gradient-enabled phi
            beta_for_phi = compute_attention_weights(
                mu_q=mu_current.detach(),  # Detach mu to isolate phi gradient
                sigma_q=sigma_current.detach() if sigma_current is not None else None,
                phi=phi_for_grad,
                generators=self.generators,
                kappa=self.kappa,
                epsilon=eps,
                mask=mask,
                use_numba=False,
                return_kl=True,  # Need KL for the loss
                diagonal_covariance=is_diagonal,
                irrep_dims=self.irrep_dims,
                chunk_size=self.chunk_size,
                mask_self_attention=self.mask_self_attention,
            )

            # beta_for_phi is (beta, kl_matrix) when return_kl=True
            if isinstance(beta_for_phi, tuple):
                beta_phi, kl_matrix = beta_for_phi
            else:
                beta_phi = beta_for_phi
                # Recompute KL for loss
                kl_matrix = compute_attention_weights(
                    mu_q=mu_current.detach(),
                    sigma_q=sigma_current.detach() if sigma_current is not None else None,
                    phi=phi_for_grad,
                    generators=self.generators,
                    kappa=self.kappa,
                    epsilon=eps,
                    mask=mask,
                    use_numba=False,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    mask_self_attention=self.mask_self_attention,
                )[1]

            # Belief alignment loss: F_align = λ·Σ_ij β_ij · KL_ij
            # This is the term that depends on φ
            alignment_loss = self.lambda_belief * (beta_phi * kl_matrix).sum()

            # Compute ∂F/∂φ
            grad_phi = torch.autograd.grad(
                alignment_loss,
                phi_for_grad,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Gradient descent on phi
            phi_current = phi - self.phi_lr * grad_phi

            # Clamp to max norm (retraction to ball)
            phi_norm = torch.norm(phi_current, dim=-1, keepdim=True)
            phi_current = torch.where(
                phi_norm > self.phi_max_norm,
                phi_current * (self.phi_max_norm / phi_norm),
                phi_current
            )

        # Return results
        # NOTE: Previously returned .detach() which BREAKS gradient flow!
        # The VFE descent is an "inner loop" optimization, but we still need
        # gradients to flow through the final result to train the embeddings.
        # The detach was likely added to prevent backprop through all iterations,
        # but it completely breaks learning. If memory is an issue, consider
        # gradient checkpointing instead.
        if self.update_sigma:
            return mu_current, sigma_current, phi_current, beta_history
        else:
            return mu_current, None, phi_current, beta_history

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
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}"
        )
        if self.pure_fep_mode:
            base += f", pure_fep_mode=True, prior_lr={self.prior_lr}"
        return base

