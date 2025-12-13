# -*- coding: utf-8 -*-
"""
Pure Free Energy Principle Transformer
========================================

A transformer architecture that learns ENTIRELY through VFE minimization,
WITHOUT backpropagation or external optimizers (Adam, SGD, etc.).

Key Principles:
---------------
1. BELIEF UPDATE (fast timescale - perception):
   μ_q ← μ_q - η_μ · ∂F/∂μ_q

   Where F = α·KL(q||p) + λ·Σ_j β_ij·KL(q_i||Ω_ij·q_j) + CE(output, target)

2. PRIOR UPDATE (slow timescale - learning):
   p_child ← Ω · q_parent    (hierarchical top-down flow)

   This IS the learning mechanism! Meta-agents at scale ζ+1 form beliefs,
   which flow down as priors to scale ζ. Credit assignment happens through
   the hierarchy - no backprop needed.

3. TWO-TIMESCALE DYNAMICS:
   - Fast: VFE gradient descent on beliefs (perception)
   - Slow: Hierarchical prior updates (learning)

   The timescale separation emerges naturally from information accumulation:
   τ_ζ ∝ 10^ζ (higher scales update less frequently)

Theory:
-------
This implements "predictive coding" in the FEP sense:
- Each layer maintains beliefs about the layer below
- Prediction errors (KL divergences) drive belief updates
- Top-down priors constrain bottom-up inference
- Learning = adjusting priors to minimize long-term VFE

The key insight: In standard transformers, backprop computes ∂Loss/∂W.
In pure FEP, we don't have explicit weights - instead:
- "Weights" are implicit in the prior structure
- Learning = evolution of priors under VFE pressure
- Credit assignment = hierarchical message passing

Author: Chris & Claude
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
import math

# Import existing VFE components
from transformer.attention import (
    compute_attention_weights,
    aggregate_messages,
    compute_transport_operators,
)
from transformer.variational_ffn import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)
from math_utils.generators import generate_so3_generators


@dataclass
class PureFEPConfig:
    """Configuration for Pure FEP Transformer."""

    # Architecture
    embed_dim: int = 127          # K - embedding dimension (MUST be odd for SO(3))
    num_layers: int = 2           # Number of hierarchical scales
    seq_length: int = 128         # N - sequence length (agents)
    vocab_size: int = 10000       # For language modeling

    # Irrep structure for SO(3) decomposition
    # Each tuple: (label, multiplicity, dim) where dim must be odd (1,3,5,7,...)
    # Total dims must equal embed_dim
    # Example for K=127: 32×1 + 15×3 + 10×5 = 32 + 45 + 50 = 127
    irrep_spec: List[Tuple[str, int, int]] = None  # Will be auto-generated if None

    # VFE parameters
    alpha: float = 0.1            # Self-coupling: KL(q||p) - increased for stronger prior influence
    lambda_belief: float = 1.0    # Belief alignment weight
    lambda_obs: float = 1.0       # Observation likelihood weight (CE in VFE)
    kappa: float = 0.1            # Attention temperature (lower = sharper attention)

    # Learning rates (natural gradient allows larger steps)
    mu_lr: float = 0.1            # Belief mean learning rate
    sigma_lr: float = 0.025       # Belief variance learning rate
    prior_lr: float = 0.01        # Prior update rate (SLOWER for stability)
    phi_lr: float = 0.05          # Gauge frame learning rate

    # Timescale separation - CRITICAL for FEP!
    # Fast timescale: VFE steps (perception)
    # Slow timescale: Prior updates (learning)
    belief_steps: int = 20        # VFE steps per forward (MORE for convergence)
    prior_update_interval: int = 1   # Update priors every batch (learning happens via VFE)

    # Covariance mode
    diagonal_covariance: bool = True  # Use diagonal Σ (faster, less memory)

    # Numerical stability
    eps: float = 1e-6
    grad_clip: float = 1.0

    # PURE FEP MODE: No backprop, all learning via prior evolution
    pure_fep_mode: bool = True    # When True: NO backprop, ONLY VFE dynamics

    # VFE differentiability mode (only used when pure_fep_mode=False)
    differentiable_vfe: bool = True  # Compute VFE via autograd (for gradient flow)

    # =========================================================================
    # ADVANCED FEP FEATURES (toggled off by default)
    # =========================================================================

    # Prior coupling: λ_γ term for priors learning from each other
    # Implements F_prior = λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)
    # This allows priors to form consistent world model across positions
    prior_coupling_enabled: bool = False
    lambda_prior: float = 0.1         # Weight for prior-prior coupling

    # Gradient-based prior updates: use VFE gradient to update priors
    # Instead of simple EMA, update priors via: p ← p - η_p · ∂F/∂p
    gradient_prior_updates: bool = False
    prior_grad_lr: float = 0.01       # Learning rate for gradient-based prior updates

    # Gauge field evolution: evolve gauge frames φ over time
    # Updates φ via: φ ← φ - η_φ · ∂F/∂φ
    gauge_evolution_enabled: bool = False
    gauge_lr: float = 0.01            # Learning rate for gauge frame evolution

    # Dynamic layer emergence: allow layers to spawn/merge based on VFE
    # When enabled, monitors VFE gradients to detect when new layers needed
    dynamic_layers_enabled: bool = False
    layer_spawn_threshold: float = 0.5   # VFE gradient threshold for spawning
    max_layers: int = 8                   # Maximum allowed layers

    def __post_init__(self):
        """Validate configuration."""
        if self.embed_dim % 2 == 0:
            raise ValueError(
                f"embed_dim must be ODD for SO(3) irreps (got {self.embed_dim}). "
                f"Try {self.embed_dim - 1} or {self.embed_dim + 1}."
            )

        # Auto-generate irrep_spec if not provided
        if self.irrep_spec is None:
            self.irrep_spec = self._generate_irrep_spec(self.embed_dim)

        # Validate irrep_spec sums to embed_dim
        total_dim = sum(mult * dim for _, mult, dim in self.irrep_spec)
        if total_dim != self.embed_dim:
            raise ValueError(
                f"irrep_spec dimensions ({total_dim}) must equal embed_dim ({self.embed_dim}). "
                f"Current spec: {self.irrep_spec}"
            )

    @staticmethod
    def _generate_irrep_spec(K: int) -> List[Tuple[str, int, int]]:
        """
        Auto-generate a reasonable irrep decomposition for dimension K.

        Strategy: Mix of scalars (ℓ=0), vectors (ℓ=1), and rank-2 tensors (ℓ=2)
        with roughly equal representation of each type.
        """
        # Target: ~40% scalars, ~35% vectors, ~25% rank-2 tensors
        n_ℓ2 = K // 15         # Each ℓ=2 irrep is 5-dim
        n_ℓ1 = K // 9          # Each ℓ=1 irrep is 3-dim
        remaining = K - (n_ℓ2 * 5 + n_ℓ1 * 3)
        n_ℓ0 = remaining       # Rest as scalars

        # Adjust to hit exact K
        current = n_ℓ0 * 1 + n_ℓ1 * 3 + n_ℓ2 * 5
        while current < K:
            n_ℓ0 += 1
            current += 1
        while current > K:
            if n_ℓ0 > 0:
                n_ℓ0 -= 1
                current -= 1
            elif n_ℓ1 > 0:
                n_ℓ1 -= 1
                current -= 3
            elif n_ℓ2 > 0:
                n_ℓ2 -= 1
                current -= 5

        spec = []
        if n_ℓ0 > 0:
            spec.append(('ℓ0', n_ℓ0, 1))
        if n_ℓ1 > 0:
            spec.append(('ℓ1', n_ℓ1, 3))
        if n_ℓ2 > 0:
            spec.append(('ℓ2', n_ℓ2, 5))

        return spec


class PureFEPLayer(nn.Module):
    """
    Single layer/scale in the Pure FEP hierarchy.

    Each layer maintains:
    - Beliefs q_i = N(μ_q, Σ_q) for each token/agent
    - Priors p_i = N(μ_p, Σ_p) constraining beliefs - NOW POSITION-DEPENDENT!
    - Gauge frames φ_i for parallel transport

    The layer performs VFE minimization:
    1. Compute attention β_ij from KL divergences (no W_Q, W_K!)
    2. Compute VFE gradients ∂F/∂μ, ∂F/∂σ INCLUDING observation term
    3. Update beliefs via natural gradient descent
    4. Optionally receive priors from parent layer

    CRITICAL FIX: Priors are now POSITION-DEPENDENT (N, K) not global (K,)
    This allows the model to learn position-specific patterns.
    """

    def __init__(
        self,
        embed_dim: int,
        scale: int,  # Hierarchical scale ζ
        config: PureFEPConfig,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = embed_dim
        self.N = config.seq_length  # Store sequence length for priors
        self.scale = scale
        self.config = config

        # SO(3) generators for gauge transport
        gen_np = generate_so3_generators(embed_dim)
        self.register_buffer('generators', torch.from_numpy(gen_np).float())

        # Output projection - maps beliefs to logits for observation likelihood
        # In pure FEP mode, this is updated via VFE pressure on priors, not backprop
        self.output_proj = nn.Linear(embed_dim, config.vocab_size, bias=False)

        # =====================================================================
        # POSITION-DEPENDENT PRIORS - the key to learning structure!
        # =====================================================================
        # Shape (N, K) - each position has its own prior
        # This allows the model to learn:
        #   - Position-specific patterns (e.g., "The" often starts sentences)
        #   - Sequential dependencies via prior evolution
        #   - Context-dependent expectations
        self.register_buffer('prior_mu', torch.zeros(config.seq_length, embed_dim))
        self.register_buffer('prior_sigma', torch.ones(config.seq_length, embed_dim))

        # Prior update statistics (for adaptive learning rate)
        self.register_buffer('prior_update_count', torch.zeros(config.seq_length))
        self.register_buffer('prior_prediction_error', torch.zeros(config.seq_length))

        # Timescale tracking
        self.info_accumulator = 0.0
        self.timescale_threshold = 10.0 ** scale

        # Statistics tracking
        self.total_vfe_steps = 0
        self.total_prior_updates = 0

    def init_beliefs(
        self,
        x: torch.Tensor,  # (B, N, K) input embeddings
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize beliefs, priors, and gauge frames from input.

        Beliefs are initialized from input embeddings.
        Priors are loaded from PERSISTENT POSITION-DEPENDENT state.
        Gauge frames start at identity (zero angle).

        CRITICAL: Priors are now (N, K) - position-dependent!

        Returns:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances (diagonal)
            mu_p: (B, N, K) prior means (position-dependent!)
            sigma_p: (B, N, K) prior variances (position-dependent!)
            phi: (B, N, 3) gauge frames in so(3)
        """
        B, N, K = x.shape

        # Beliefs initialized from input
        # In pure FEP mode, we DON'T need gradient connection - learning is via priors
        if self.config.pure_fep_mode:
            mu_q = x.detach().clone()  # Detach in pure FEP mode
        else:
            mu_q = x.clone()
            if self.config.differentiable_vfe and not mu_q.requires_grad:
                mu_q.requires_grad_(True)

        sigma_q = torch.ones(B, N, K, device=device) * 0.1  # Small initial variance

        # =====================================================================
        # POSITION-DEPENDENT PRIORS - broadcast across batch only
        # =====================================================================
        # Handle sequence length mismatch (input may be shorter than max seq_length)
        N_prior = min(N, self.prior_mu.shape[0])

        # Get position-specific priors and expand across batch
        mu_p = self.prior_mu[:N_prior, :].unsqueeze(0).expand(B, -1, -1).clone()
        sigma_p = self.prior_sigma[:N_prior, :].unsqueeze(0).expand(B, -1, -1).clone()

        # Pad if input sequence is longer than stored priors
        if N > N_prior:
            # Use zero mean, unit variance for new positions
            mu_p_pad = torch.zeros(B, N - N_prior, K, device=device)
            sigma_p_pad = torch.ones(B, N - N_prior, K, device=device)
            mu_p = torch.cat([mu_p, mu_p_pad], dim=1)
            sigma_p = torch.cat([sigma_p, sigma_p_pad], dim=1)

        # Gauge frames start at identity
        phi = torch.zeros(B, N, 3, device=device)
        # Enable gradients for phi if gauge evolution is enabled
        if self.config.gauge_evolution_enabled:
            phi.requires_grad_(True)

        return mu_q, sigma_q, mu_p, sigma_p, phi

    def vfe_step(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        phi: torch.Tensor,
        targets: Optional[torch.Tensor] = None,  # (B, N) target tokens
        mask: Optional[torch.Tensor] = None,     # (B, N, N) causal mask
        is_final_step: bool = False,             # Only create_graph on final step!
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Single VFE gradient descent step on beliefs.

        This is the PERCEPTION step - updating beliefs to minimize VFE
        given current priors and observations.

        CRITICAL FIX: Observation term (CE) is now INSIDE the VFE!
        F = α·KL(q||p) + λ·alignment + λ_obs·E_q[-log p(y|x)]

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances
            mu_p: (B, N, K) prior means
            sigma_p: (B, N, K) prior variances
            phi: (B, N, 3) gauge frames
            targets: (B, N) target tokens for cross-entropy
            mask: (B, N, N) causal attention mask

        Returns:
            mu_q_new: Updated belief means
            sigma_q_new: Updated belief variances
            phi_new: Updated gauge frames
            metrics: Dict of loss components
        """
        B, N, K = mu_q.shape
        device = mu_q.device

        # ==================================================================
        # 1. Compute dynamic attention from KL divergences
        # ==================================================================
        beta, kl_matrix = compute_attention_weights(
            mu_q, sigma_q, phi, self.generators,
            kappa=self.config.kappa,
            mask=mask,
            return_kl=True,
            diagonal_covariance=True,
        )

        # ==================================================================
        # 2. Compute FULL VFE loss INCLUDING observations
        # ==================================================================
        # F = α·KL(q||p) + λ·alignment + λ_obs·CE
        # This is the TRUE variational free energy!

        # Self-coupling: α·KL(q||p)
        sigma_q_safe = sigma_q.clamp(min=self.config.eps)
        sigma_p_safe = sigma_p.clamp(min=self.config.eps)
        kl_self = 0.5 * (
            sigma_q_safe / sigma_p_safe
            + (mu_q - mu_p)**2 / sigma_p_safe
            - 1.0
            + torch.log(sigma_p_safe / sigma_q_safe)
        ).sum(dim=-1).mean()

        # Alignment: λ·Σ β_ij·KL_ij (use precomputed)
        alignment = (beta * kl_matrix).sum(dim=-1).mean()

        # Prior coupling: λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)
        prior_coupling = self.compute_prior_coupling_loss(mu_p, sigma_p, phi, mask)

        # ==================================================================
        # OBSERVATION TERM: E_q[-log p(y|x)] ≈ CE(W·μ_q, targets)
        # ==================================================================
        # This is the CRITICAL addition - observations are INSIDE the VFE
        if targets is not None:
            logits = self.output_proj(mu_q)
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                reduction='mean'
            )
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # FULL VFE = KL terms + observation term
        # ==================================================================
        lambda_obs = getattr(self.config, 'lambda_obs', 1.0)
        vfe_loss = (self.config.alpha * kl_self +
                   self.config.lambda_belief * alignment +
                   prior_coupling +
                   lambda_obs * ce_loss)  # Observations IN the VFE!

        # ==================================================================
        # 3. Compute gradients (analytical mode for pure FEP)
        # ==================================================================
        if self.config.pure_fep_mode:
            # PURE FEP MODE: Use analytical gradients, no autograd graph
            grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                mu_q, sigma_q, mu_p, sigma_p,
                beta, phi, self.generators,
                alpha=self.config.alpha,
                lambda_belief=self.config.lambda_belief,
                kappa=self.config.kappa,
                eps=self.config.eps,
            )

            # Add observation gradient (CE) analytically
            if targets is not None:
                with torch.enable_grad():
                    mu_q_grad = mu_q.detach().requires_grad_(True)
                    logits_grad = self.output_proj(mu_q_grad)
                    ce_for_grad = F.cross_entropy(
                        logits_grad.view(-1, self.config.vocab_size),
                        targets.view(-1),
                        reduction='sum'
                    )
                    grad_mu_ce = torch.autograd.grad(ce_for_grad, mu_q_grad)[0]
                # Add CE gradient with lambda_obs weight
                grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)
        else:
            # Differentiable mode (for hybrid training)
            grad_enabled = torch.is_grad_enabled() and mu_q.requires_grad
            if grad_enabled and mu_q.requires_grad:
                if is_final_step:
                    grad_mu = torch.autograd.grad(
                        vfe_loss, mu_q,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                else:
                    grad_mu = torch.autograd.grad(
                        vfe_loss, mu_q,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
            else:
                grad_mu = torch.zeros_like(mu_q)
            grad_sigma = torch.zeros_like(sigma_q)

        # ==================================================================
        # 4. Project to natural gradients (Fisher-Rao metric)
        # ==================================================================
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_q, eps=self.config.eps
        )

        # ==================================================================
        # 5. Gradient clipping for stability
        # ==================================================================
        grad_norm = torch.norm(nat_grad_mu)
        if grad_norm > self.config.grad_clip:
            nat_grad_mu = nat_grad_mu * self.config.grad_clip / grad_norm

        sigma_grad_norm = torch.norm(nat_grad_sigma)
        if sigma_grad_norm > self.config.grad_clip:
            nat_grad_sigma = nat_grad_sigma * self.config.grad_clip / sigma_grad_norm

        # ==================================================================
        # 6. Update beliefs (gradient DESCENT, so subtract)
        # ==================================================================
        mu_q_new = mu_q - self.config.mu_lr * nat_grad_mu
        sigma_q_new = sigma_q - self.config.sigma_lr * nat_grad_sigma

        # Ensure sigma stays positive
        sigma_q_new = sigma_q_new.clamp(min=self.config.eps)

        # ==================================================================
        # 7. Update gauge frames (if gauge evolution enabled)
        # ==================================================================
        # Only update gauge frames during training with gradients enabled
        if self.config.gauge_evolution_enabled and phi.requires_grad and grad_enabled:
            # Compute gradient of VFE w.r.t. gauge frames
            # The gauge frames affect attention via transport operators
            # Use vfe_loss if we computed it in differentiable mode
            loss_for_phi = vfe_loss if (self.config.differentiable_vfe and grad_enabled) else (kl_self + alignment)
            try:
                grad_phi = torch.autograd.grad(
                    loss_for_phi,
                    phi,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad_phi is not None:
                    # Gradient descent on gauge frames
                    phi_new = phi - self.config.gauge_lr * grad_phi
                else:
                    phi_new = phi
            except RuntimeError:
                # If grad computation fails, keep phi unchanged
                phi_new = phi
        else:
            phi_new = phi

        # ==================================================================
        # 8. Compute metrics
        # ==================================================================
        # Recompute metrics if we didn't run differentiable path
        # (either analytical mode or eval mode with grad disabled)
        if not (self.config.differentiable_vfe and grad_enabled):
            # Recompute for metrics (needed in analytical mode or eval mode)
            kl_self = 0.5 * (
                sigma_q / sigma_p.clamp(min=self.config.eps)
                + (mu_q - mu_p)**2 / sigma_p.clamp(min=self.config.eps)
                - 1.0
                + torch.log(sigma_p.clamp(min=self.config.eps) / sigma_q.clamp(min=self.config.eps))
            ).sum(dim=-1).mean()
            alignment = (beta * kl_matrix).sum(dim=-1).mean()
            prior_coupling = self.compute_prior_coupling_loss(mu_p, sigma_p, phi, mask)

        # Detach for metrics to avoid graph issues
        prior_coupling_val = prior_coupling.detach().item() if isinstance(prior_coupling, torch.Tensor) else 0.0
        metrics = {
            'vfe_total': (self.config.alpha * kl_self.detach() +
                         self.config.lambda_belief * alignment.detach() +
                         prior_coupling_val +
                         ce_loss.detach()).item(),
            'kl_self': kl_self.detach().item(),
            'alignment': alignment.detach().item(),
            'prior_coupling': prior_coupling_val,
            'ce_loss': ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            'grad_norm_mu': grad_norm.item(),
            'grad_norm_sigma': sigma_grad_norm.item(),
        }

        self.total_vfe_steps += 1

        return mu_q_new, sigma_q_new, phi_new, metrics

    def update_prior_from_parent(
        self,
        mu_q: torch.Tensor,      # Child beliefs
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,      # Child priors (to be updated)
        sigma_p: torch.Tensor,
        phi: torch.Tensor,       # Child gauge frames
        parent_mu_q: torch.Tensor,    # Parent beliefs
        parent_sigma_q: torch.Tensor,
        parent_phi: torch.Tensor,     # Parent gauge frames
        parent_generators: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical prior update: p_child ← Ω · q_parent

        This is the LEARNING mechanism in pure FEP:
        - Parent layer forms beliefs about what children should believe
        - These flow down as priors, constraining child inference
        - Over time, this shapes the "knowledge" of the network

        Args:
            mu_q, sigma_q: Child belief distribution
            mu_p, sigma_p: Child prior (to be updated)
            phi: Child gauge frames
            parent_mu_q, parent_sigma_q: Parent beliefs
            parent_phi: Parent gauge frames
            parent_generators: Parent's SO(3) generators

        Returns:
            mu_p_new: Updated prior means
            sigma_p_new: Updated prior variances
        """
        B, N, K = mu_q.shape
        device = mu_q.device

        # Compute transport from parent frame to child frame
        # Ω_ij = exp(φ_child) @ exp(-φ_parent)
        phi_child_matrix = torch.einsum('bna,aij->bnij', phi, self.generators)
        phi_parent_matrix = torch.einsum('bna,aij->bnij', parent_phi, parent_generators)

        exp_child = torch.matrix_exp(phi_child_matrix)
        exp_neg_parent = torch.matrix_exp(-phi_parent_matrix)

        # Transport operator (per token)
        Omega = torch.einsum('bnik,bnjk->bnij', exp_child, exp_neg_parent)

        # Transport parent beliefs to child frame
        # μ_p_new = Ω @ μ_parent
        parent_mu_transported = torch.einsum('bnij,bnj->bni', Omega, parent_mu_q)

        # For diagonal covariance: variance doesn't change under rotation
        # (This is an approximation - full transport would rotate the covariance)
        parent_sigma_transported = parent_sigma_q.clone()

        # Soft update: blend old prior with new (for stability)
        blend_factor = self.config.prior_lr
        mu_p_new = (1 - blend_factor) * mu_p + blend_factor * parent_mu_transported
        sigma_p_new = (1 - blend_factor) * sigma_p + blend_factor * parent_sigma_transported

        # Ensure sigma stays positive
        sigma_p_new = sigma_p_new.clamp(min=self.config.eps)

        self.total_prior_updates += 1

        return mu_p_new, sigma_p_new

    def compute_prior_coupling_loss(
        self,
        mu_p: torch.Tensor,      # (B, N, K) prior means
        sigma_p: torch.Tensor,   # (B, N, K) prior variances
        phi: torch.Tensor,       # (B, N, 3) gauge frames
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prior-prior coupling loss: F_prior = λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)

        This encourages priors at different positions to form a consistent
        world model, where information is coherent across the sequence.

        Args:
            mu_p: Prior means (B, N, K)
            sigma_p: Prior variances (B, N, K)
            phi: Gauge frames (B, N, 3)
            mask: Optional causal mask (B, N, N)

        Returns:
            prior_coupling_loss: Scalar loss value
        """
        if not self.config.prior_coupling_enabled:
            return torch.tensor(0.0, device=mu_p.device)

        B, N, K = mu_p.shape
        device = mu_p.device

        # Compute transport operators between positions
        transport_cache = compute_transport_operators(phi, self.generators)
        Omega = transport_cache['Omega']  # (B, N, N, K, K)

        # For diagonal covariance, we approximate transport as identity for variance
        # and only transport the mean
        # Transported prior: Ω_ij · p_j
        mu_p_transported = torch.einsum('bnmij,bmj->bnmi', Omega, mu_p)  # (B, N, N, K)

        # KL divergence: KL(p_i || Ω_ij·p_j)
        # For diagonal Gaussians: KL = 0.5 * (σ_i/σ_j + (μ_i-μ_j)²/σ_j - 1 + log(σ_j/σ_i))
        sigma_p_safe = sigma_p.clamp(min=self.config.eps)

        # Expand for pairwise computation
        mu_i = mu_p.unsqueeze(2)  # (B, N, 1, K)
        sigma_i = sigma_p_safe.unsqueeze(2)  # (B, N, 1, K)
        sigma_j = sigma_p_safe.unsqueeze(1)  # (B, 1, N, K)

        kl_prior = 0.5 * (
            sigma_i / sigma_j
            + (mu_i - mu_p_transported)**2 / sigma_j
            - 1.0
            + torch.log(sigma_j / sigma_i)
        ).sum(dim=-1)  # (B, N, N)

        # Apply causal mask if provided
        if mask is not None:
            kl_prior = kl_prior * mask

        # Average over all pairs
        prior_coupling_loss = self.config.lambda_prior * kl_prior.mean()

        return prior_coupling_loss

    def update_persistent_prior(
        self,
        mu_q_batch: torch.Tensor,     # (B, N, K) batch belief means (after VFE)
        sigma_q_batch: torch.Tensor,  # (B, N, K) batch belief variances
        prediction_error: Optional[torch.Tensor] = None,  # (B,) or (B, N) CE loss
        per_position_error: Optional[torch.Tensor] = None,  # (B, N) per-position CE
    ):
        """
        Update POSITION-DEPENDENT persistent priors using prediction-error-weighted learning.

        This is where LEARNING is stored! Each position's prior moves towards beliefs
        that successfully minimize prediction error at THAT position.

        CRITICAL FIX: Priors are now (N, K) - position-specific!
        Each position learns its own prior based on:
        1. Beliefs at that position across the batch
        2. Prediction error weighting (beliefs with lower error have more influence)

        Args:
            mu_q_batch: Beliefs after VFE convergence (B, N, K)
            sigma_q_batch: Belief variances (B, N, K)
            prediction_error: Per-sample CE loss (B,) - for batch-level weighting
            per_position_error: Per-position CE loss (B, N) - for position-level weighting
        """
        B, N, K = mu_q_batch.shape

        # Detach beliefs to prevent graph accumulation
        mu_q_batch = mu_q_batch.detach()
        sigma_q_batch = sigma_q_batch.detach()

        # =====================================================================
        # POSITION-SPECIFIC PRIOR UPDATE
        # =====================================================================
        # Update each position's prior based on beliefs at that position

        # Handle sequence length mismatch
        N_prior = min(N, self.prior_mu.shape[0])

        if per_position_error is not None and per_position_error.numel() > 0:
            # POSITION-LEVEL PREDICTION-ERROR WEIGHTING
            # Each position is weighted by how well beliefs at that position predicted
            # Lower error = higher weight

            # per_position_error: (B, N) -> weights: (B, N)
            temperature = 1.0
            # Softmax over batch dimension for each position
            weights = F.softmax(-per_position_error / temperature, dim=0)  # (B, N)

            # Weighted average across batch for each position
            # weights: (B, N) -> (B, N, 1) for broadcasting
            weights_expanded = weights.unsqueeze(-1)  # (B, N, 1)
            mu_p_new = (mu_q_batch[:, :N_prior, :] * weights_expanded[:, :N_prior, :]).sum(dim=0)  # (N, K)
            sigma_p_new = (sigma_q_batch[:, :N_prior, :] * weights_expanded[:, :N_prior, :]).sum(dim=0)  # (N, K)

        elif prediction_error is not None and prediction_error.numel() > 0:
            # BATCH-LEVEL PREDICTION-ERROR WEIGHTING
            # Use per-sample error (same weight for all positions in a sample)
            temperature = 1.0
            weights = F.softmax(-prediction_error / temperature, dim=0)  # (B,)

            # Weighted average across batch for each position
            weights_expanded = weights.view(B, 1, 1)  # (B, 1, 1)
            mu_p_new = (mu_q_batch[:, :N_prior, :] * weights_expanded).sum(dim=0)  # (N, K)
            sigma_p_new = (sigma_q_batch[:, :N_prior, :] * weights_expanded).sum(dim=0)  # (N, K)

        else:
            # Fallback: simple average across batch for each position
            mu_p_new = mu_q_batch[:, :N_prior, :].mean(dim=0)  # (N, K)
            sigma_p_new = sigma_q_batch[:, :N_prior, :].mean(dim=0)  # (N, K)

        # =====================================================================
        # Apply update to persistent priors
        # =====================================================================
        blend = self.config.prior_lr

        if self.config.gradient_prior_updates:
            # GRADIENT-BASED PRIOR UPDATE
            # ∂F/∂μ_p ∝ (μ_p - μ_q) / σ_p²
            grad_mu_p = (self.prior_mu[:N_prior, :] - mu_p_new) / self.prior_sigma[:N_prior, :].clamp(min=self.config.eps)**2

            # Gradient descent on prior
            self.prior_mu[:N_prior, :].sub_(self.config.prior_grad_lr * grad_mu_p)

            # EMA for sigma
            self.prior_sigma[:N_prior, :].lerp_(sigma_p_new, blend)
        else:
            # EXPONENTIAL MOVING AVERAGE
            self.prior_mu[:N_prior, :].lerp_(mu_p_new, blend)
            self.prior_sigma[:N_prior, :].lerp_(sigma_p_new, blend)

        # Track update statistics
        self.prior_update_count[:N_prior] += 1

        # Ensure sigma stays positive
        self.prior_sigma.clamp_(min=self.config.eps)

        self.total_prior_updates += 1

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        parent_beliefs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        n_vfe_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through layer: initialize beliefs and run VFE steps.

        Args:
            x: (B, N, K) input embeddings
            targets: (B, N) target tokens
            mask: (B, N, N) causal mask
            parent_beliefs: Optional (mu_q, sigma_q, phi) from parent layer
            n_vfe_steps: Number of VFE gradient descent steps

        Returns:
            mu_q: Final belief means (B, N, K)
            info: Dict with metrics and intermediate states
        """
        B, N, K = x.shape
        device = x.device

        # KEEP gradient connection to embeddings!
        # VFE dynamics should influence embedding learning through the belief updates.
        # Previously this was detached, breaking the learning signal.

        # Initialize beliefs from input
        mu_q, sigma_q, mu_p, sigma_p, phi = self.init_beliefs(x, device)

        # Update priors from parent if available
        if parent_beliefs is not None:
            parent_mu_q, parent_sigma_q, parent_phi = parent_beliefs
            mu_p, sigma_p = self.update_prior_from_parent(
                mu_q, sigma_q, mu_p, sigma_p, phi,
                parent_mu_q, parent_sigma_q, parent_phi,
                self.generators,
            )

        # Run VFE gradient descent steps
        # CRITICAL: Only use create_graph=True on FINAL step to avoid
        # "backward through graph twice" errors from nested autograd.grad calls
        all_metrics = []
        for step in range(n_vfe_steps):
            is_final = (step == n_vfe_steps - 1)
            mu_q, sigma_q, phi, metrics = self.vfe_step(
                mu_q, sigma_q, mu_p, sigma_p, phi,
                targets=targets, mask=mask,
                is_final_step=is_final,
            )
            all_metrics.append(metrics)

        # Aggregate metrics
        final_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                        for k in all_metrics[0].keys()}

        info = {
            'metrics': final_metrics,
            'beliefs': (mu_q, sigma_q),
            'priors': (mu_p, sigma_p),
            'phi': phi,
        }

        return mu_q, info


class PureFEPTransformer(nn.Module):
    """
    Complete Pure FEP Transformer for language modeling.

    Architecture:
    - Token embedding layer (learned)
    - Multiple PureFEPLayers in hierarchy
    - Each layer has its own beliefs, priors, gauge frames
    - Priors flow TOP-DOWN from higher to lower layers
    - Beliefs flow BOTTOM-UP through VFE minimization

    Training:
    - NO Adam/SGD - purely VFE gradient descent
    - NO backprop through computational graph
    - Learning via hierarchical prior updates
    """

    def __init__(self, config: PureFEPConfig):
        super().__init__()
        self.config = config

        # Token embeddings (the only truly learned parameters besides output_proj)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Hierarchical layers
        self.layers = nn.ModuleList([
            PureFEPLayer(config.embed_dim, scale=i, config=config)
            for i in range(config.num_layers)
        ])

        # Output head (shared with layers, or separate)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Position encoding (simple sinusoidal)
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(config.seq_length, config.embed_dim)
        )

        # Training state
        self.step_count = 0

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal position encoding (handles odd dimensions)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Handle odd d_model: sin gets ceil(d/2), cos gets floor(d/2)
        n_sin = (d_model + 1) // 2  # ceil
        n_cos = d_model // 2        # floor

        div_term_sin = torch.exp(torch.arange(0, n_sin).float() * (-math.log(10000.0) / d_model) * 2)
        div_term_cos = torch.exp(torch.arange(0, n_cos).float() * (-math.log(10000.0) / d_model) * 2)

        pe[:, 0::2] = torch.sin(position * div_term_sin)
        pe[:, 1::2] = torch.cos(position * div_term_cos)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, N) token IDs
        targets: Optional[torch.Tensor] = None,  # (B, N) target token IDs
        n_vfe_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with pure VFE learning.

        Args:
            input_ids: (B, N) input token IDs
            targets: (B, N) target token IDs (shifted by 1 for LM)
            n_vfe_steps: VFE steps per layer

        Returns:
            logits: (B, N, vocab_size) output logits
            info: Dict with all metrics and states
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.embedding(input_ids)  # (B, N, K)

        # Add position encoding
        x = x + self.pos_encoding[:, :N, :]

        # Causal mask
        mask = torch.tril(torch.ones(N, N, device=device)).unsqueeze(0).expand(B, -1, -1)

        # Process through hierarchical layers
        layer_infos = []
        parent_beliefs = None

        # BOTTOM-UP: process layers 0 → L-1
        for layer_idx, layer in enumerate(self.layers):
            x, info = layer(
                x, targets=targets, mask=mask,
                parent_beliefs=parent_beliefs,
                n_vfe_steps=n_vfe_steps,
            )
            layer_infos.append(info)

            # Current layer's beliefs become parent for next layer
            parent_beliefs = (info['beliefs'][0], info['beliefs'][1], info['phi'])

        # Output logits from final layer beliefs
        logits = self.output_proj(x)  # (B, N, vocab_size)

        # NOTE: Prior updates now happen in train_step() where we have prediction errors
        # This allows prediction-error-weighted learning instead of simple averaging.
        self.step_count += 1

        # Aggregate info
        all_metrics = {}
        for i, info in enumerate(layer_infos):
            for k, v in info['metrics'].items():
                all_metrics[f'layer_{i}/{k}'] = v

        info = {
            'metrics': all_metrics,
            'layer_infos': layer_infos,
        }

        return logits, info

    def _hierarchical_prior_update(
        self,
        layer_infos: List[Dict],
        prediction_errors: Optional[torch.Tensor] = None,
        per_position_errors: Optional[torch.Tensor] = None,
    ):
        """
        Top-down prior update: propagate beliefs down the hierarchy.

        This is where LEARNING happens in pure FEP:
        - Top layer forms beliefs about the world
        - These flow down as priors to lower layers
        - Lower layers must now explain observations under these priors
        - This shapes what the network "knows"

        CRITICAL: Updates are persisted to layer.prior_mu/prior_sigma buffers!

        Args:
            layer_infos: List of layer outputs from forward pass
            prediction_errors: (B,) per-sample CE loss for weighted prior updates
            per_position_errors: (B, N) per-position CE for fine-grained learning
        """
        # Process top-down: layer L-1 → layer 0
        for i in range(len(self.layers) - 1, 0, -1):
            parent_info = layer_infos[i]
            child_layer = self.layers[i - 1]
            child_info = layer_infos[i - 1]

            # Get parent beliefs and child priors
            parent_mu_q, parent_sigma_q = parent_info['beliefs']
            parent_phi = parent_info['phi']

            child_mu_q, child_sigma_q = child_info['beliefs']
            child_mu_p, child_sigma_p = child_info['priors']
            child_phi = child_info['phi']

            # Update child priors from parent beliefs
            new_mu_p, new_sigma_p = child_layer.update_prior_from_parent(
                child_mu_q, child_sigma_q,
                child_mu_p, child_sigma_p,
                child_phi,
                parent_mu_q, parent_sigma_q,
                parent_phi,
                self.layers[i].generators,
            )

            # PERSIST the learning! Update the layer's stored priors.
            # Use prediction-error-weighted beliefs for the update
            child_layer.update_persistent_prior(
                child_mu_q, child_sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )

            # Store updated priors in info dict too (for current pass)
            child_info['priors'] = (new_mu_p, new_sigma_p)

        # Also update top layer's prior from its own beliefs (self-supervision)
        if len(self.layers) > 0:
            top_layer = self.layers[-1]
            top_info = layer_infos[-1]
            top_mu_q, top_sigma_q = top_info['beliefs']
            # Top layer learns from its own beliefs (no parent)
            top_layer.update_persistent_prior(
                top_mu_q, top_sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        n_vfe_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Single training step using pure VFE learning.

        PURE FEP MODE (config.pure_fep_mode=True):
        - NO backprop on embeddings or output projections
        - ALL learning happens via prior evolution under VFE pressure
        - Perception: VFE gradient descent on beliefs
        - Learning: Position-dependent prior updates weighted by prediction error

        HYBRID MODE (config.pure_fep_mode=False):
        - Backprop updates embeddings and projections
        - Prior updates provide additional learning signal

        Args:
            input_ids: (B, N) input tokens
            targets: (B, N) target tokens
            n_vfe_steps: VFE steps per layer

        Returns:
            Dict of training metrics
        """
        self.train()
        B, N = input_ids.shape

        if self.config.pure_fep_mode:
            # ==================================================================
            # PURE FEP MODE: No backprop, all learning via prior evolution
            # ==================================================================
            with torch.no_grad():
                # Forward pass with VFE
                logits, info = self(input_ids, targets=targets, n_vfe_steps=n_vfe_steps)

                # Compute overall CE loss for metrics
                ce_loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                )

                # Compute PER-POSITION prediction error for prior learning
                # This is CRITICAL: each position learns from its own errors
                logits_reshaped = logits.view(B, N, -1)
                targets_reshaped = targets.view(B, N)
                per_position_loss = F.cross_entropy(
                    logits_reshaped.permute(0, 2, 1),  # (B, vocab, N)
                    targets_reshaped,                   # (B, N)
                    reduction='none'
                )  # (B, N) - loss at each position

                # Per-sample loss (for batch-level weighting)
                per_sample_loss = per_position_loss.mean(dim=1)  # (B,)

                # Store both for prior updates
                info['prediction_errors'] = per_sample_loss
                info['per_position_errors'] = per_position_loss

            # Update persistent priors - THIS IS WHERE LEARNING HAPPENS!
            self._update_priors_with_prediction_error(info)

        else:
            # ==================================================================
            # HYBRID MODE: Backprop + prior evolution
            # ==================================================================
            # Zero gradients
            self.zero_grad()

            # Forward pass with VFE - gradients flow through
            logits, info = self(input_ids, targets=targets, n_vfe_steps=n_vfe_steps)

            # CE loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

            # Compute per-position errors
            with torch.no_grad():
                logits_reshaped = logits.view(B, N, -1)
                targets_reshaped = targets.view(B, N)
                per_position_loss = F.cross_entropy(
                    logits_reshaped.permute(0, 2, 1),
                    targets_reshaped,
                    reduction='none'
                )
                per_sample_loss = per_position_loss.mean(dim=1)

            info['prediction_errors'] = per_sample_loss
            info['per_position_errors'] = per_position_loss

            # Backprop
            ce_loss.backward()

            # Apply gradients (manual SGD)
            with torch.no_grad():
                for param in [self.embedding.weight, self.output_proj.weight]:
                    if param.grad is not None:
                        grad_norm = param.grad.norm()
                        if grad_norm > self.config.grad_clip:
                            param.grad.mul_(self.config.grad_clip / grad_norm)
                        param.sub_(self.config.mu_lr * param.grad)
                        param.grad.zero_()

                for layer in self.layers:
                    if layer.output_proj.weight.grad is not None:
                        grad_norm = layer.output_proj.weight.grad.norm()
                        if grad_norm > self.config.grad_clip:
                            layer.output_proj.weight.grad.mul_(self.config.grad_clip / grad_norm)
                        layer.output_proj.weight.sub_(self.config.mu_lr * layer.output_proj.weight.grad)
                        layer.output_proj.weight.grad.zero_()

            # Update priors
            self._update_priors_with_prediction_error(info)

        # Perplexity
        ppl = torch.exp(ce_loss.detach()).item()

        metrics = {
            'ce_loss': ce_loss.item(),
            'perplexity': ppl,
            **info['metrics'],
        }

        return metrics

    def _update_priors_with_prediction_error(self, info: Dict):
        """
        Update POSITION-DEPENDENT priors using prediction-error-weighted beliefs.

        This is the LEARNING step in pure FEP:
        - Each position's prior evolves based on beliefs at that position
        - Beliefs with lower prediction error have more influence
        - Per-position errors allow fine-grained learning

        Called from train_step() where prediction errors are available.
        """
        # Only update priors at specified intervals
        if self.step_count % self.config.prior_update_interval != 0:
            return

        prediction_errors = info.get('prediction_errors')  # (B,)
        per_position_errors = info.get('per_position_errors')  # (B, N)
        layer_infos = info.get('layer_infos', [])

        # Update each layer's persistent prior with position-specific errors
        for i, layer_info in enumerate(layer_infos):
            mu_q, sigma_q = layer_info['beliefs']
            self.layers[i].update_persistent_prior(
                mu_q, sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )


class PureFEPTrainer:
    """
    Trainer for Pure FEP Transformer.

    Implements the training loop without external optimizers.
    """

    def __init__(self, model: PureFEPTransformer, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)

        # Training statistics
        self.total_steps = 0
        self.best_ppl = float('inf')
        self.history = []

    def train_epoch(
        self,
        dataloader,
        n_vfe_steps: int = 1,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ppl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            metrics = self.model.train_step(input_ids, targets, n_vfe_steps)

            total_loss += metrics['ce_loss']
            total_ppl += metrics['perplexity']
            n_batches += 1
            self.total_steps += 1

            if batch_idx % log_interval == 0:
                avg_ppl = total_ppl / n_batches
                print(f"Step {self.total_steps} | Loss: {metrics['ce_loss']:.4f} | PPL: {avg_ppl:.2f}")

        epoch_metrics = {
            'loss': total_loss / n_batches,
            'perplexity': total_ppl / n_batches,
        }

        self.history.append(epoch_metrics)

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)

        return {'loss': avg_loss, 'perplexity': ppl}


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PURE FEP TRANSFORMER TEST")
    print("=" * 70)

    # Config (embed_dim must be ODD for SO(3) irreps)
    config = PureFEPConfig(
        embed_dim=63,  # Must be odd!
        num_layers=2,
        seq_length=32,
        vocab_size=1000,
        mu_lr=0.1,
        sigma_lr=0.025,
        prior_lr=0.05,
    )

    print(f"\n[1] Creating model...")
    model = PureFEPTransformer(config)
    print(f"    Config: K={config.embed_dim}, L={config.num_layers}, V={config.vocab_size}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")

    # Test forward pass
    print(f"\n[2] Testing forward pass...")
    B, N = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, N))
    targets = torch.randint(0, config.vocab_size, (B, N))

    logits, info = model(input_ids, targets=targets, n_vfe_steps=1)
    print(f"    Input shape: {input_ids.shape}")
    print(f"    Output shape: {logits.shape}")
    print(f"    VFE metrics: {list(info['metrics'].keys())[:5]}...")

    # Test training step
    print(f"\n[3] Testing training step...")
    metrics = model.train_step(input_ids, targets, n_vfe_steps=1)
    print(f"    CE Loss: {metrics['ce_loss']:.4f}")
    print(f"    Perplexity: {metrics['perplexity']:.2f}")

    # Test multiple VFE steps
    print(f"\n[4] Testing multiple VFE steps...")
    for n_steps in [1, 2, 5]:
        metrics = model.train_step(input_ids, targets, n_vfe_steps=n_steps)
        print(f"    n_vfe_steps={n_steps}: PPL={metrics['perplexity']:.2f}")

    print("\n" + "=" * 70)
    print("PURE FEP TRANSFORMER TEST COMPLETE")
    print("=" * 70)
