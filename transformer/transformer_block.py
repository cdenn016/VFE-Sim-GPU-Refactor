"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (KL-based, no W_Q/W_K!)
2. Feedforward network (prior evolution)
3. Layer normalization
4. Residual connections

Standard Architecture, Gauge Mechanism:
    x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

But with gauge-theoretic attention:
    (μ, Σ, φ) → Attention(via KL + transport) → (μ', Σ', φ')

Author: Implementation from plan.py
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Import our gauge attention
from transformer.attention import IrrepMultiHeadAttention

# Import VFE FFN module
from transformer.ffn import GaugeFFN

# Trajectory tracking (optional)
try:
    from transformer.trajectory_tracking import get_global_recorder
    TRAJECTORY_TRACKING_AVAILABLE = True
except ImportError:
    TRAJECTORY_TRACKING_AVAILABLE = False
    def get_global_recorder():
        return None


class GaugeTransformerBlock(nn.Module):
    """
    Single transformer block with gauge-theoretic attention.

    Architecture:
        1. Self-attention sublayer:
           - LayerNorm on means
           - IrrepMultiHeadAttention (KL-based)
           - Residual connection
           - Dropout

        2. Feedforward sublayer:
           - LayerNorm on means
           - FFN (two linear layers + GELU)
           - Residual connection
           - Dropout

    Note: We primarily evolve means (μ), while covariances (Σ) and
          gauge frames (φ) can be evolved or kept fixed depending on mode.

    0D Structure:
        - All agents at single point c*
        - Attention computed via KL divergence
        - No spatial convolutions or position-dependent operations
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        ffn_mode: str = 'VFE_dynamic',  # VFE_dynamic is the only supported mode
        ffn_alpha: float = 0.001,
        # Dynamic VFE specific parameters
        ffn_vfe_dynamic_m_step_interval: int = 0,  # M-step every N steps (0 = disabled)
        ffn_vfe_dynamic_m_step_rate: float = 0.01,  # Prior update rate
        # AD-HOC stabilization (default OFF for first-principles)
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        ffn_lambda_belief: float = 1.0,
        ffn_update_sigma: bool = True,
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Sparse attention
        attention_pattern: str = 'full',
        attention_window: int = 64,
        # Gauge frame dimension
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        # Pure FEP mode: learning via prior evolution (no backprop)
        ffn_pure_fep_mode: bool = False,
        ffn_max_seq_len: int = 512,
        ffn_prior_lr: float = 0.01,
        # Memory-efficient options
        ffn_irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        ffn_chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
    ):
        """
        Initialize gauge transformer block.

        Args:
            embed_dim: Embedding dimension K
            irrep_spec: Irrep structure [(label, mult, dim), ...]
            hidden_dim: FFN hidden dimension (typically 4 × embed_dim)
            kappa_beta: Temperature for attention
            dropout: Dropout probability
            evolve_sigma: If True, update covariances via attention and FFN
            evolve_phi: If True, update gauge frames via FFN
            attention_pattern: 'full', 'local', or 'sparse' for efficient attention
            attention_window: Window size for local attention pattern
            generators: Lie algebra generators (required for VFE mode)
            ffn_mode: 'VFE_dynamic' - dynamic-β VFE with attention-belief co-evolution
            ffn_alpha: Prior weight for VFE
            ffn_kappa: Softmax temperature for attention
            ffn_n_iterations: VFE inference iterations per forward pass
            ffn_learnable_lr: Learn step size for variational descent
            ffn_lambda_belief: Belief alignment weight
            ffn_update_sigma: Update covariances in FFN
            ffn_pure_fep_mode: If True, use persistent priors for backprop-free learning
            ffn_max_seq_len: Max sequence length for persistent priors (pure FEP mode)
            ffn_prior_lr: Learning rate for prior updates (pure FEP mode)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.evolve_sigma = evolve_sigma
        self.evolve_phi = evolve_phi
        self.ffn_mode = ffn_mode
        self.generators = generators  # Store for variational FFN
        self.diagonal_covariance = diagonal_covariance

        # =====================================================================
        # Attention Sublayer
        # =====================================================================
        # Determine gauge group from generators shape
        if generators is not None:
            n_gen = generators.shape[0]
            if n_gen == 3:
                gauge_group = 'SO3'
                gauge_dim_inferred = 3
            else:
                # n_gen = N*(N-1)/2 => N = (1 + sqrt(1 + 8*n_gen)) / 2
                import math
                gauge_dim_inferred = int((1 + math.sqrt(1 + 8 * n_gen)) / 2)
                gauge_group = 'SON'
        else:
            gauge_group = 'SO3'
            gauge_dim_inferred = 3

        self.attention = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            kappa_beta=kappa_beta,
            epsilon=1e-8,
            aggregate_mode='full_distribution' if evolve_sigma else 'mean_only',
            diagonal_covariance=diagonal_covariance,
            attention_pattern=attention_pattern,
            attention_window=attention_window,
            gauge_group=gauge_group,
            gauge_dim=gauge_dim_inferred,
            global_generators=generators,  # Pass for SO(N) mode
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # =====================================================================
        # VFE_dynamic FFN Sublayer
        # =====================================================================
        self.ffn = GaugeFFN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            generators=generators,  # Required for VFE mode
            dropout=dropout,
            mode=ffn_mode,
            # VFE parameters
            alpha=ffn_alpha,
            kappa=ffn_kappa,
            n_iterations=ffn_n_iterations,
            learnable_lr=ffn_learnable_lr,
            lambda_belief=ffn_lambda_belief,
            update_sigma=ffn_update_sigma,
            # Dynamic VFE parameters
            vfe_dynamic_m_step_interval=ffn_vfe_dynamic_m_step_interval,
            vfe_dynamic_m_step_rate=ffn_vfe_dynamic_m_step_rate,
            # Diagonal covariance mode
            diagonal_covariance=diagonal_covariance,
            # Pure FEP mode parameters
            pure_fep_mode=ffn_pure_fep_mode,
            max_seq_len=ffn_max_seq_len,
            prior_lr=ffn_prior_lr,
            # Memory-efficient options
            irrep_dims=ffn_irrep_dims,
            chunk_size=ffn_chunk_size,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        # =====================================================================
        # Optional: Gauge Frame Evolution
        # =====================================================================
        self.phi_dim = phi_dim
        if evolve_phi:
            # Small FFN for φ evolution (phi_dim output in so(n))
            self.phi_ffn = nn.Sequential(
                nn.Linear(embed_dim, 16),
                nn.Tanh(),
                nn.Linear(16, phi_dim),
            )
            # Initialize to near-zero to start with identity frames
            nn.init.zeros_(self.phi_ffn[2].weight)
            nn.init.zeros_(self.phi_ffn[2].bias)
        else:
            self.phi_ffn = None

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        targets: Optional[torch.Tensor] = None,   # For E-step observations
        W_out: Optional[torch.Tensor] = None,     # Output projection for discrete observations
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K, K)
            phi: Gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask (B, N, N)
            mu_prior: Embedding priors (B, N, K) - required for variational FFN
            targets: Target token IDs (B, N) - for E-step discrete observations
            W_out: Output projection (V, K) - for computing CE gradient in E-step
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers.

        Returns:
            mu_q_out: Updated means (B, N, K)
            sigma_q_out: Updated covariances (B, N, K, K)
            phi_out: Updated gauge frames (B, N, 3)
        """
        # =====================================================================
        # 1. Attention Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization on means
        mu_normalized = self.norm1(mu_q)

        # Multi-head attention (gauge-theoretic!)
        # Capture beta if needed for variational/hamiltonian FFN or trajectory recording
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_attention = recorder is not None and recorder.enabled and recorder.record_attention
        need_beta = self.ffn_mode == 'VFE_dynamic'
        need_attention_output = need_beta or recording_attention

        mu_attn, sigma_attn, beta, kl_matrix = self.attention(
            mu_normalized,
            sigma_q,
            phi,
            generators,
            mask=mask,
            return_attention=need_attention_output,  # Compute if needed for FFN or recording
            cached_head_transports=cached_head_transports,  # Cross-layer cache
        )

        # Record attention for trajectory tracking
        if recording_attention and beta is not None:
            recorder.record_attention(beta, kl_matrix)

        # Residual connection + dropout on means
        mu_q = mu_q + self.dropout1(mu_attn)

        # Update covariances if evolving
        if self.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn
        # Otherwise sigma_q stays unchanged

        # =====================================================================
        # 2. Feedforward Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization
        mu_normalized = self.norm2(mu_q)

        # VFE_dynamic FFN: β recomputed at each VFE step
        # Returns (mu, sigma) tuple
        if mu_prior is None:
            raise ValueError("VFE_dynamic mode requires mu_prior argument")

        mu_ffn, sigma_ffn = self.ffn(
            mu=mu_normalized,
            beta=beta,          # Initial β (will be recomputed each step inside FFN)
            mu_prior=mu_prior,  # From embeddings
            phi=phi,            # Current gauge frames
            sigma=sigma_q,      # Current covariances
            mask=mask,          # Causal mask
            targets=targets,    # Target tokens (discrete observations)
            W_out=W_out,        # Output projection for ∂CE/∂μ
        )

        # Update covariances from FFN if evolving
        if self.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        # Residual connection
        mu_q = mu_q + mu_ffn

        # =====================================================================
        # 3. Optional: Gauge Frame Evolution
        # =====================================================================
        if self.evolve_phi and self.phi_ffn is not None:
            # Evolve gauge frames based on current means
            # φ_new = φ + Δφ(μ)
            delta_phi = self.phi_ffn(mu_q)  # (B, N, 3)
            phi = phi + delta_phi

            # Optional: retract to principal ball (keep ||φ|| small)
            phi_norm = torch.norm(phi, dim=-1, keepdim=True)
            max_phi_norm = 2.0  # Keep gauge frames moderate
            phi = torch.where(
                phi_norm > max_phi_norm,
                phi * (max_phi_norm / phi_norm),
                phi
            )
        return mu_q, sigma_q, phi

    def get_hamiltonian_diagnostics(self) -> None:
        """Stub for backward compatibility with trajectory recording."""
        return None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"evolve_sigma={self.evolve_sigma}, "
            f"evolve_phi={self.evolve_phi}"
        )


# =============================================================================
# Stack of Transformer Blocks
# =============================================================================

class GaugeTransformerStack(nn.Module):
    """
    Stack of N gauge transformer blocks.

    This is the main "encoder" of the model, transforming initial
    embeddings through multiple layers of gauge-theoretic attention.
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,
        ffn_mode: str = 'VFE_dynamic',
        ffn_alpha: float = 0.001,
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        ffn_lambda_belief: float = 1.0,
        ffn_update_sigma: bool = True,
        # Dynamic VFE specific parameters
        ffn_vfe_dynamic_m_step_interval: int = 0,  # M-step every N steps (0 = disabled)
        ffn_vfe_dynamic_m_step_rate: float = 0.01,  # Prior update rate
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Sparse attention
        attention_pattern: str = 'full',
        attention_window: int = 64,
        # Gauge frame dimension
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        # Pure FEP mode: learning via prior evolution (no backprop)
        ffn_pure_fep_mode: bool = False,
        ffn_max_seq_len: int = 512,
        ffn_prior_lr: float = 0.01,
        # Memory-efficient options
        ffn_irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        ffn_chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
    ):
        """
        Initialize stack of transformer blocks.

        Args:
            n_layers: Number of transformer blocks
            embed_dim: Embedding dimension
            irrep_spec: Irrep structure
            hidden_dim: FFN hidden dimension
            kappa_beta: Attention temperature
            dropout: Dropout probability
            evolve_sigma: If True, covariances evolve through layers
            evolve_phi: If True, gauge frames evolve through layers
            generators: Lie algebra generators (required for VFE FFN)
            phi_dim: Dimension of gauge frame (3 for SO(3), N(N-1)/2 for SO(N))
            ffn_mode: 'VFE_dynamic' (only supported mode)
            ffn_alpha: Prior weight
            ffn_kappa: Softmax temperature for attention
            ffn_n_iterations: VFE inference iterations
            ffn_learnable_lr: Learn step size for variational descent
            ffn_lambda_belief: Belief alignment weight
            ffn_update_sigma: Update covariances in FFN
            ffn_vfe_dynamic_m_step_interval: M-step every N iterations (0=disabled)
            ffn_vfe_dynamic_m_step_rate: Prior update rate in M-step
            attention_pattern: 'full', 'local', or 'sparse' for efficient attention
            attention_window: Window size for local attention pattern
            ffn_pure_fep_mode: If True, use persistent priors for backprop-free learning
            ffn_max_seq_len: Max sequence length for persistent priors (pure FEP mode)
            ffn_prior_lr: Learning rate for prior updates (pure FEP mode)
        """
        super().__init__()
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            GaugeTransformerBlock(
                embed_dim=embed_dim,
                irrep_spec=irrep_spec,
                hidden_dim=hidden_dim,
                kappa_beta=kappa_beta,
                dropout=dropout,
                evolve_sigma=evolve_sigma,
                evolve_phi=evolve_phi,
                # VFE FFN
                generators=generators,
                ffn_mode=ffn_mode,
                ffn_alpha=ffn_alpha,
                ffn_kappa=ffn_kappa,
                ffn_n_iterations=ffn_n_iterations,
                ffn_learnable_lr=ffn_learnable_lr,
                ffn_lambda_belief=ffn_lambda_belief,
                ffn_update_sigma=ffn_update_sigma,
                # Dynamic VFE
                ffn_vfe_dynamic_m_step_interval=ffn_vfe_dynamic_m_step_interval,
                ffn_vfe_dynamic_m_step_rate=ffn_vfe_dynamic_m_step_rate,
                # Diagonal covariance mode
                diagonal_covariance=diagonal_covariance,
                # Sparse attention
                attention_pattern=attention_pattern,
                attention_window=attention_window,
                # Gauge frame dimension
                phi_dim=phi_dim,
                # Pure FEP mode
                ffn_pure_fep_mode=ffn_pure_fep_mode,
                ffn_max_seq_len=ffn_max_seq_len,
                ffn_prior_lr=ffn_prior_lr,
                # Memory-efficient options
                ffn_irrep_dims=ffn_irrep_dims,
                ffn_chunk_size=ffn_chunk_size,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        return_intermediates: bool = False,
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward through all transformer blocks.

        Args:
            mu_q: Initial means (B, N, K)
            sigma_q: Initial covariances (B, N, K, K)
            phi: Initial gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask
            mu_prior: Embedding priors (B, N, K) - for variational FFN
            return_intermediates: If True, return states after each layer
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers (6× speedup).

        Returns:
            mu_q: Final means (B, N, K)
            sigma_q: Final covariances (B, N, K, K)
            phi: Final gauge frames (B, N, 3)
            intermediates: Optional list of intermediate states
        """
        intermediates = [] if return_intermediates else None

        # Get trajectory recorder
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_enabled = recorder is not None and recorder.enabled

        for layer_idx, block in enumerate(self.blocks):
            # Trajectory recording: start layer
            if recording_enabled:
                recorder.start_layer(layer_idx)
                recorder.record_layer_input(mu_q, sigma_q, phi)

            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, generators, mask, mu_prior,
                cached_head_transports=cached_head_transports,
            )

            # Trajectory recording: record output and diagnostics
            if recording_enabled:
                # Get Hamiltonian diagnostics if available
                diagnostics = block.get_hamiltonian_diagnostics()
                recorder.record_layer_output(mu_q, sigma_q, phi, diagnostics)
                recorder.end_layer()

            if return_intermediates:
                intermediates.append({
                    'layer': layer_idx,
                    'mu': mu_q.detach(),
                    'sigma': sigma_q.detach() if sigma_q is not None else None,
                    'phi': phi.detach(),
                })

        # Final normalization
        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates

    def get_hamiltonian_diagnostics(self) -> List[Optional[dict]]:
        """Get Hamiltonian diagnostics from all layers."""
        return [block.get_hamiltonian_diagnostics() for block in self.blocks]


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TRANSFORMER BLOCK TEST")
    print("="*70)

    # Test configuration
    B, N, K = 2, 8, 16
    n_layers = 3
    hidden_dim = 64

    print(f"\n[1] Configuration:")
    print(f"    Batch size: {B}")
    print(f"    Num agents: {N}")
    print(f"    Embed dim:  {K}")
    print(f"    Layers:     {n_layers}")
    print(f"    Hidden dim: {hidden_dim}")

    # Create test data
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
    phi = torch.randn(B, N, 3) * 0.1

    # Random generators
    G = torch.randn(3, K, K)
    G = 0.5 * (G - G.transpose(-1, -2))

    # Irrep spec
    irrep_spec = [
        ('ℓ0', 4, 1),
        ('ℓ1', 2, 3),
        ('ℓ2', 1, 5),
    ]

    # Test single block
    print(f"\n[2] Testing single transformer block...")
    block = GaugeTransformerBlock(
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
        generators=G,  # Required for VFE_dynamic mode
    )
    print(f"    {block}")

    # Create mu_prior for VFE_dynamic mode
    mu_prior = mu_q.clone() * 0.5

    mu_out, sigma_out, phi_out = block(mu_q, sigma_q, phi, G, mu_prior=mu_prior)
    print(f"    Output μ shape: {mu_out.shape}")
    print(f"    Output Σ shape: {sigma_out.shape}")
    print(f"    Output φ shape: {phi_out.shape}")
    print(f"    ✓ Single block forward pass complete")

    # Test stack
    print(f"\n[3] Testing transformer stack ({n_layers} layers)...")
    stack = GaugeTransformerStack(
        n_layers=n_layers,
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
        generators=G,  # Required for VFE_dynamic mode
    )

    mu_final, sigma_final, phi_final, intermediates = stack(
        mu_q, sigma_q, phi, G, mu_prior=mu_prior, return_intermediates=True
    )

    print(f"    Final μ shape: {mu_final.shape}")
    print(f"    Intermediate states: {len(intermediates)}")
    print(f"    ✓ Stack forward pass complete")

    # Test with causal mask
    print(f"\n[4] Testing with causal mask...")
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
    mu_causal, _, _, _ = stack(mu_q, sigma_q, phi, G, mask=mask, mu_prior=mu_prior)
    print(f"    Causal output shape: {mu_causal.shape}")
    print(f"    ✓ Causal masking works")

    # Parameter count
    total_params = sum(p.numel() for p in stack.parameters())
    per_layer = total_params // n_layers

    print(f"\n[5] Parameter count:")
    print(f"    Total:     {total_params:,} parameters")
    print(f"    Per layer: {per_layer:,} parameters")
    print(f"    Attention: ~{per_layer // 3:,} params (1/3 of layer)")
    print(f"    FFN:       ~{2 * per_layer // 3:,} params (2/3 of layer)")

    # Compare to standard transformer
    standard_params = 4 * K * K + 2 * K * hidden_dim + 4 * K  # Q,K,V,O + FFN + LN
    standard_total = standard_params * n_layers
    reduction = standard_total / total_params

    print(f"\n[6] Comparison to standard transformer:")
    print(f"    Standard total: {standard_total:,} parameters")
    print(f"    Gauge total:    {total_params:,} parameters")
    print(f"    Reduction:      {reduction:.2f}x fewer parameters!")

    print("\n" + "="*70)
    print("All transformer block tests passed!")
    print("="*70)