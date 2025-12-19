"""
Feed-Forward Networks for Gauge Transformer
===========================================

VFE_dynamic mode: Dynamic attention-belief co-evolution
- Recomputes β at each VFE step
- Updates both μ AND Σ
- Enables emergent block structure through attention-belief coupling

Author: Extended architecture with VFE integration
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple, Union

from transformer.variational_ffn import (
    VariationalFFNDynamic,  # Dynamic-β VFE with attention-belief co-evolution
)


class GaugeFFN(nn.Module):
    """
    FFN module for Gauge Transformer using VFE_dynamic mode.

    VFE_dynamic: Dynamic-β VFE with attention-belief co-evolution
    - Recomputes β at each VFE step
    - Updates both μ AND Σ
    - Enables emergent block structure

    Note: For experimental Hamiltonian dynamics, see transformer.experimental.hamiltonian_ffn
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        dropout: float = 0.1,
        mode: Literal['VFE_dynamic'] = 'VFE_dynamic',
        # Dynamic VFE specific parameters
        vfe_dynamic_m_step_interval: int = 0,  # M-step every N steps (0 = disabled)
        vfe_dynamic_m_step_rate: float = 0.01,  # Prior update rate
        # Variational parameters
        alpha: float = 0.001,
        kappa: float = 1.0,
        n_iterations: int = 1,
        learnable_lr: bool = True,
        lambda_belief: float = 1.0,
        update_sigma: bool = True,
        compute_sigma_align_grad: bool = True,  # Sigma gradient from alignment term
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Legacy parameters (ignored, kept for API compatibility)
        **kwargs,
    ):
        """
        Initialize VFE FFN module.

        Args:
            embed_dim: K - latent dimension
            hidden_dim: Hidden layer size (unused, kept for API compatibility)
            generators: SO(3) generators (3, K, K) - required
            dropout: Dropout rate (unused, kept for API compatibility)
            mode: FFN mode - only 'VFE_dynamic' is supported
            vfe_dynamic_m_step_interval: M-step every N steps (0 = disabled)
            vfe_dynamic_m_step_rate: Prior update rate for M-step
            alpha: Prior weight
            kappa: Softmax temperature for attention
            n_iterations: VFE inference steps per forward pass
            learnable_lr: Learn step size for variational descent
            lambda_belief: Belief alignment weight
            update_sigma: Update covariances during inference
            compute_sigma_align_grad: Compute sigma gradient from alignment term
            diagonal_covariance: Use diagonal covariance for memory efficiency
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = 'VFE_dynamic'

        if generators is None:
            raise ValueError("generators required for VFE_dynamic mode")

        # Initialize VFE_dynamic FFN
        self.variational_ffn = VariationalFFNDynamic(
            embed_dim=embed_dim,
            generators=generators,
            alpha=alpha,
            lambda_belief=lambda_belief,
            kappa=kappa,
            n_iterations=n_iterations,
            learnable_lr=learnable_lr,
            update_sigma=update_sigma,
            m_step_interval=vfe_dynamic_m_step_interval,
            m_step_rate=vfe_dynamic_m_step_rate,
            diagonal_covariance=diagonal_covariance,
            compute_sigma_align_grad=compute_sigma_align_grad,
        )

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - always required
        beta: Optional[torch.Tensor] = None,      # (B, n_heads, N, N) or (B, N, N)
        mu_prior: Optional[torch.Tensor] = None,  # (B, N, K)
        phi: Optional[torch.Tensor] = None,       # (B, N, 3)
        sigma: Optional[torch.Tensor] = None,     # (B, N, K, K)
        sigma_prior: Optional[torch.Tensor] = None,  # (B, N, K, K) - unused
        mask: Optional[torch.Tensor] = None,      # (B, N, N)
        targets: Optional[torch.Tensor] = None,   # (B, N) - target tokens
        W_out: Optional[torch.Tensor] = None,     # (V, K) - output projection
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VFE_dynamic FFN.

        Args:
            mu: Current beliefs (B, N, K)
            beta: Initial attention weights (will be recomputed each step)
            mu_prior: Embedding priors (B, N, K)
            phi: Gauge frames (B, N, 3)
            sigma: Covariances (B, N, K, K)
            sigma_prior: Prior covariances (unused)
            mask: Causal mask (B, N, N)
            targets: Target token IDs (B, N)
            W_out: Output projection matrix (V, K)

        Returns:
            (mu_out, sigma_out): Updated beliefs and covariances
        """
        # Check required inputs
        if mu_prior is None or phi is None:
            raise ValueError("VFE_dynamic requires mu_prior, phi")

        # Dynamic VFE returns (mu, sigma, beta_history)
        # beta_history is None unless return_beta_history=True is passed
        mu_out, sigma_out, beta_history = self.variational_ffn(
            mu=mu,
            beta=beta,          # Initial β (will be recomputed each step)
            mu_prior=mu_prior,
            phi=phi,
            sigma=sigma,
            mask=mask,
            targets=targets,
            W_out=W_out,
            return_beta_history=False,
        )
        return (mu_out, sigma_out)

    def get_mode(self) -> str:
        """Get current FFN mode."""
        return self.mode


# =============================================================================
# Convenience functions
# =============================================================================

def create_ffn(
    embed_dim: int,
    hidden_dim: int,
    generators: Optional[torch.Tensor] = None,
    mode: str = 'VFE_dynamic',
    **kwargs
) -> GaugeFFN:
    """
    Factory function for creating FFN.

    Example:
        >>> ffn = create_ffn(
        ...     embed_dim=11, hidden_dim=44,
        ...     generators=generators, alpha=0.001
        ... )
    """
    return GaugeFFN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        generators=generators,
        mode=mode,
        **kwargs
    )
