# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:24:37 2025

@author: chris and christine
"""

"""
Publication Proof-of-Principle Training Script
===============================================

Language modeling on WikiText-2 with byte-level encoding for minimal publishable claim.

Demonstrates:
1. Variational FFN works - inference comparable to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference holds
4. Hamiltonian dynamics - energy-conserving symplectic integration (NEW!)

Five FFN Modes for Ablation Study:
    - learned: Standard MLP baseline (GELU activation)
    - variational_approx: First-order active inference (O(N²K), legacy)
    - variational_full: Complete gauge-invariant with second-order terms (O(N³K), legacy)
    - variational_gradient_engine: Full active inference via gradient_engine.py
    - hamiltonian: Symplectic Hamiltonian dynamics on belief space (NEW!)
      * Energy-conserving leapfrog integration
      * Full faithful SPD geometry with curvature corrections
      * NO learned weights - pure physics!

Comprehensive Metrics Tracking:
    - Free energy components (α, β, γ terms)
    - Gradient norms (total, μ, FFN)
    - All learning rates (μ, σ, φ, FFN)
    - Bits-per-character (BPC)
    - Attention statistics (β_mean, KL_mean)
    - Performance (step time, tokens/sec)
    - Hamiltonian diagnostics (H_init, H_final, ΔH) for hamiltonian mode

Output Files:
    - checkpoints_publication/ffn_{mode}/metrics.csv - comprehensive training metrics
    - checkpoints_publication/ffn_{mode}/best_model.pt - best model checkpoint
    - checkpoints_publication/result_{mode}.json - final summary (if single mode)
    - checkpoints_publication/ablation_results.json - comparison (if --run_ablation)

Usage:
    # Just click Run (edit defaults below)
    python transformer/train_publication.py

    # Or use command-line args:
    python transformer/train_publication.py --ffn_mode learned
    python transformer/train_publication.py --ffn_mode hamiltonian

Author: Designed for minimal publishable claim
Date: December 2025
"""

import torch
import argparse
import json
import csv
import time
import math
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


from transformer.model import GaugeTransformerLM
from transformer.standard_transformer import StandardTransformerLM
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import (
    compute_free_energy_loss,
    compute_rg_metrics_from_attention,
    pure_fep_train_step,
    pure_fep_validate,
    PureFEPConfig,
    PureFEPTrainer,
)
from transformer.train_fast import FastTrainer, FastTrainingConfig
from transformer.publication_metrics import PublicationMetrics, ExperimentResult


def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        dirty = len(status) > 0

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}


def get_system_info() -> Dict[str, Any]:
    """Get system/hardware information."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def save_experiment_config(
    config: Dict[str, Any],
    ffn_mode: str,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
) -> Path:
    """
    Save complete experiment configuration to JSON.

    Args:
        config: Model/training configuration dictionary
        ffn_mode: FFN mode being used
        checkpoint_dir: Directory to save config
        args: Command-line arguments (if available)

    Returns:
        Path to saved config file
    """
    experiment_config = {
        # Metadata
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'ffn_mode': ffn_mode,

        # Full model/training config
        'config': config,

        # Command-line args (if available)
        'args': vars(args) if args else None,

        # Git info for reproducibility
        'git': get_git_info(),

        # System info
        'system': get_system_info(),
    }

    # Save to checkpoint directory
    config_path = checkpoint_dir / 'experiment_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)

    print(f"📋 Saved experiment config: {config_path}")

    return config_path


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS
# ============================================================================
DEFAULT_FFN_MODE = 'VFE_dynamic'  # 'learned', 'VFE_dynamic', 'variational_gradient_engine', 'hamiltonian', or None
DEFAULT_RUN_ABLATION = False  # Set True to run all three modes
DEFAULT_ENABLE_SIGMA_PHI = True   # Set True to enable learning Σ and φ (required for hamiltonian!)
DEFAULT_USE_GPU_OPTIMIZED = True  # Set True for RTX 5090 / high-end GPU settings
# ============================================================================



# =============================================================================
# STANDARD TRANSFORMER BASELINE CONFIG
# =============================================================================
# Standard dot-product attention + learned MLP for fair comparison.
# Uses same batch size, seq length, and embedding dim as gauge VFE.
#
STANDARD_TRANSFORMER_CONFIG = {
    # Model architecture - match gauge VFE dimensions
    'vocab_size': 2000000,        # Will be overridden by tokenizer
    'embed_dim': 25,              # Same K as gauge VFE
    'n_layers': 1,                # Same depth
    'hidden_dim': 100,            # 4×embed_dim standard ratio
    'max_seq_len': 94,            # Same context length
    'n_heads': 5,                 # embed_dim / head_dim (5 heads of dim 5)

    # GPU Training - same as gauge VFE
    'batch_size': 24,
    'use_amp': False,
    'num_workers': 4,

    # Standard transformer settings
    'ffn_mode': 'learned',        # Standard learned MLP
    'attention_type': 'standard', # Use dot-product attention
    'pos_encoding_mode': 'learned',
    'tie_embeddings': True,       # Standard practice

    # Disable gauge-specific features
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,  # Not used but safer
    'use_positional_embedding': True,

    # Standard attention parameters
    'kappa_beta': 1.0,            # Not used in standard attention
    'epsilon': 1e-8,
    'attention_pattern': 'full',
    'attention_window': 24,

    # Training
    'max_steps': 20000,
    'warmup_steps': 25,

    # Learning rates - standard Adam rates
    'mu_lr': 0.001,               # Standard LR for embeddings
    'sigma_lr': 0.001,
    'phi_lr': 0.001,
    'ffn_lr': 0.001,              # Standard LR for MLP

    # Free energy weights (not used in standard mode)
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization - standard
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging
    'log_interval': 100,
    'eval_interval': 500,
    'checkpoint_interval': 5000,
    'patience': 5,

    # Disable gauge group (not used)
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'use_multi_irrep': False,
    'gauge_fixed_priors': True,

    # Irrep spec (not used in standard mode)
    'irrep_spec': [('ℓ0', 5, 1), ('ℓ1', 0, 3)],  # 5 scalars only

    # RG metrics disabled for standard
    'compute_rg_metrics': False,
}

# =============================================================================
# GPU-OPTIMIZED CONFIG (RTX 5090 / 32GB VRAM)
# =============================================================================
# MEMORY REALITY CHECK:
#   Gauge transformer has O(N² × K²) memory for attention KL matrices!
#   Standard transformer: O(N² × d) for attention
#   Ours: O(N² × K²) because KL divergence uses full covariance matrices
#
#   Memory for KL computation: B × N × N × K² × 4 bytes (FP32)
#   Example: B=32, N=256, K=127 → 32 × 256 × 256 × 127² × 4 = ~134GB (!)
#
#   Realistic for 32GB: B=16, N=64, K=63 → ~2GB for KL matrices
#
GPU_OPTIMIZED_CONFIG = {
    # Model architecture - WITH diagonal_covariance=True, can scale up!
    # Diagonal mode: O(N²×K) memory instead of O(N²×K²)
    # Model architecture (realistic for 32GB VRAM)
    # Can't match Vaswani d=512 due to K² memory cost!
    
    'vocab_size': 2000000,        # Full byte-level vocab
    'embed_dim': 25,          # K=63 (ODD for SO(3)) - realistic for memory
    'n_layers': 1,            # Fewer layers to save memory
    'hidden_dim': 508,        # 4×embed_dim Only for 'learned'
    'max_seq_len': 94,        # N=64 - attention is O(N²×K²)!

    # GPU Training - fits in 32GB
    'batch_size': 24,         # Conservative for memory
    'use_amp': False,         # Disabled - Hamiltonian dynamics needs FP32 precision
    'num_workers': 4,         # Parallel data loading

    # Gauge transformer parameters
    # =========================================================================
    # TEMPERATURE SCALING (κ ∝ K)
    # Theory: E[D_KL] = Kρ²/σ², so κ must scale with K for stable attention.
    # When kappa_beta_auto_scale=True, κ is computed as:
    #   κ = kappa_beta_base × (K / K_ref)
    # K_ref=11 is the reference dimension where kappa_beta_base=1 works well.
    # =========================================================================
    'kappa_beta_auto_scale': True,   # Enable automatic κ scaling with K
    'kappa_beta_base': 1.0,          # Base temperature at K=K_ref
    'kappa_beta_k_ref': 11,          # Reference dimension (K=11 works with κ=1)
    'epsilon': 1e-8,
    'pos_encoding_mode': 'learned',   #'learned' or 'sinusoidal'
    'evolve_sigma': True,     # Full geometric learning
    'evolve_phi': True,       # Full geometric learning
    'tie_embeddings': False,
  
    # Attention pattern
    'attention_pattern': 'full',   #'full', 'local', 'sparse' 
    'attention_window': 24,
    
    # =========================================================================
    # SIGMA GRADIENT FROM ALIGNMENT (theoretical correctness vs legacy)
    # True (default): Compute ∂KL/∂Σ = 0.5*(Σ_transported^{-1} - Σ^{-1})
    # False:  zero sigma gradient from alignment term
    # Setting True enables proper uncertainty propagation through gauge transport.
    # =========================================================================
    'compute_sigma_align_grad': True,
    
    # =========================================================================
    # FAST MATRIX EXPONENTIAL (speed optimization)
    # True:  Use Taylor series approximation for exp(φ·G) - faster but approximate
    # False: Use torch.matrix_exp - accurate but slower
    # Taylor is accurate for small angles |φ| < 0.5, use with phi_scale < 0.3
    # =========================================================================
    'use_fast_exp': True,
    'exp_order': 4,  # Taylor series order when use_fast_exp=True

    # =========================================================================
    # DIAGONAL COVARIANCE MODE (memory optimization)
    # True:  Σ is (B,N,K) diagonal - O(N²×K) memory - can scale to Vaswani size?
    # False: Σ is (B,N,K,K) full   - O(N²×K²) memory - limited to small K,N
    # Diagonal loses off-diagonal correlations but keeps per-dim uncertainty.
    # =========================================================================
    'diagonal_covariance': False,
    'use_positional_embedding': True,
    
    # Variational FFN parameters
    'ffn_mode': 'VFE_dynamic',
    'ffn_alpha': 1,
    'ffn_tau_eff': 1.0,
    'ffn_kappa': 1,
    'ffn_n_iterations': 1,
    'ffn_learnable_lr': True,
    'ffn_pattern': 'full',
    'ffn_window': 64,

    # =========================================================================
    # BLOCK-DIAGONAL KL COMPUTATION (Principled Memory Optimization)
    # When use_block_diagonal_kl=True, exploits the irrep structure:
    # - Generators are block-diagonal → Omega is block-diagonal
    # - KL decomposes additively across blocks
    # - Memory: O(N² × Σᵢdᵢ²) instead of O(N² × K²)
    # For K=255 with 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂: ~82× memory savings!
    # This is the PRINCIPLED approach that respects gauge structure.
    # =========================================================================
    'use_block_diagonal_kl': True,  # Enable block-diagonal KL (recommended!)

    # =========================================================================
    # CHUNKED KL COMPUTATION (Additional Memory Optimization)
    # Processes N×N attention matrix in C×C chunks to reduce peak memory.
    # Set to None for no chunking, or a small value (32-64) for memory savings.
    # Combines with block-diagonal for maximum efficiency.
    # =========================================================================
    'ffn_chunk_size': 64,  # Chunk size for memory-efficient attention (None = no chunking)

    # =========================================================================
    # PURE FEP MODE (Backprop-Free Learning)
    # When enabled, learning happens through prior evolution, not backprop.
    # - CE (cross-entropy) is INSIDE the VFE during forward pass
    # - Beliefs adjust to minimize prediction error
    # - Priors update toward successful (low-error) beliefs
    # This is the BELIEF paradigm: Backprop-free Evolving Local Inference via Free Energy
    # =========================================================================
    'ffn_pure_fep_mode': False,   # Enable backprop-free learning
    'ffn_prior_lr': 0.01,         # Learning rate for prior updates

    'gauge_fixed_priors': True,

    # Training (scaled for GPU)
    'max_steps': 20000 ,         # More steps for convergence

    # Learning rates (same natural gradient rates)
    'mu_lr': 0.2,
    'sigma_lr': 0.01,
    'phi_lr': 0.05,
    'ffn_lr': 0.2,
    'warmup_steps': 25,

    # Free energy weights
    'alpha': 1,
    'beta': 1,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # Logging (less frequent for speed)
    'log_interval': 100,
    'eval_interval': 500,
    'checkpoint_interval': 5000,
    'patience': 5,

    # =================================================================
    # GAUGE GROUP SELECTION
    # =================================================================
    # SO3: Standard SO(3) gauge group with 3 generators
    #      Requires embed_dim = sum(mult * dim) for irrep_spec or odd embed_dim
    # SON: SO(N) gauge group with N(N-1)/2 generators
    #      More flexible - can use N-dimensional fundamental representation
    #      embed_dim = mult * N for direct sums of fundamental
    # =================================================================
    'gauge_group': 'SO3',  # 'SO3' or 'SON'
    'gauge_dim': 3,        # N for SO(N) - only used when gauge_group='SON'
    'use_multi_irrep': True,  # Use block-diagonal generators from irrep_spec


    # Irrep structure (for K=255)
    # 75×1 + 30×3 + 18×5 = 75 + 90 + 90 = 255 ✓
    'irrep_spec': [
        #('ℓ0', 2, 1),   # 75 dimensions (scalars)
      # ('ℓ1', 2, 3),   # 90 dimensions (vectors)
       ('ℓ2', 1, 5),   # 90 dimensions (rank-2 tensors)
     #  ('ℓ3', 1, 7),
       ('ℓ4', 1, 9),
       ('ℓ5', 1, 11),
      # ('ℓ6', 1, 13),
      # ('ℓ7', 1, 15),
      # ('ℓ8', 1, 17),
    ],

    # RG Metrics Configuration (meta-agent emergence detection)
    'compute_rg_metrics': False,           # Enable RG metrics computation
    'rg_metrics_interval': 25,            # Compute RG metrics every N steps
    'rg_auto_cluster': True,              # Auto-detect clusters via spectral clustering
    'rg_n_clusters': None,                # Fixed number of clusters (None = auto)
}

class PublicationMetricsTracker:
    """Track ALL metrics needed for publication."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []

        # Create CSV with comprehensive headers
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_align',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats (crucial for interpretability!)
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',
            'attention_entropy', 'attention_concentration',

            # RG Metrics (meta-agent emergence!)
            'rg_modularity', 'rg_effective_rank', 'rg_n_clusters',
            'rg_kl_within_mean', 'rg_kl_within_std',
            'rg_kl_between_mean', 'rg_kl_between_std',
            'rg_beta_entropy',

            # Learning rates
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Performance
            'step_time', 'tokens_per_sec',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, metrics: Dict, lrs: Dict, grad_norms: Dict,
                 step_time: float, batch_size: int, seq_len: int):
        """Log training step with full metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Bits per character (convert from nats)
        train_bpc = metrics.get('train_loss_ce', 0) / math.log(2)

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses
            'train_loss_total': metrics.get('train_loss_total'),
            'train_loss_ce': metrics.get('train_loss_ce'),
            'train_loss_belief_align': metrics.get('train_loss_belief_align', 0),
            'train_loss_self_consistency': metrics.get('train_loss_self_consistency', 0),
            'train_loss_model_align': metrics.get('train_loss_model_align', 0),
            'val_loss': None,
            'val_ce': None,

            # Metrics
            'train_ppl': metrics.get('train_ppl'),
            'train_bpc': train_bpc,
            'val_ppl': None,
            'val_bpc': None,

            # Attention (crucial for interpretability!)
            'beta_mean': metrics.get('beta_mean'),
            'beta_std': metrics.get('beta_std'),
            'kl_mean': metrics.get('kl_mean'),
            'kl_std': metrics.get('kl_std'),
            'attention_entropy': metrics.get('attention_entropy'),
            'attention_concentration': metrics.get('attention_concentration'),

            # RG Metrics (meta-agent emergence!)
            'rg_modularity': metrics.get('rg/modularity'),
            'rg_effective_rank': metrics.get('rg/effective_rank'),
            'rg_n_clusters': metrics.get('rg/n_clusters'),
            'rg_kl_within_mean': metrics.get('rg/kl_within_mean'),
            'rg_kl_within_std': metrics.get('rg/kl_within_std'),
            'rg_kl_between_mean': metrics.get('rg/kl_between_mean'),
            'rg_kl_between_std': metrics.get('rg/kl_between_std'),
            'rg_beta_entropy': metrics.get('rg/beta_entropy'),

            # Learning rates
            'mu_lr': lrs.get('mu_embed', 0),
            'sigma_lr': lrs.get('sigma_embed', 0),
            'phi_lr': lrs.get('phi_embed', 0),
            'ffn_lr': lrs.get('ffn', 0),

            # Gradients
            'grad_norm_total': grad_norms.get('total', 0),
            'grad_norm_mu': grad_norms.get('mu', 0),
            'grad_norm_ffn': grad_norms.get('ffn', 0),

            # Performance
            'step_time': step_time,
            'tokens_per_sec': tokens_per_sec,
        }

        self.history.append(entry)

    def log_val(self, step: int, val_metrics: Dict):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_metrics.get('loss')
                entry['val_ce'] = val_metrics.get('ce_loss', val_metrics.get('loss'))
                entry['val_ppl'] = val_metrics.get('perplexity')
                entry['val_bpc'] = entry['val_ce'] / math.log(2) if entry['val_ce'] else None
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


class PublicationTrainer(FastTrainer):
    """Enhanced trainer with publication-quality metrics."""

    def __init__(self, *args, publication_metrics: PublicationMetrics = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Basic CSV metrics tracker
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        print(f"[INFO] Logging publication metrics to: {metrics_path}")

        # Comprehensive publication metrics (optional)
        self.pub_metrics = publication_metrics
        if self.pub_metrics:
            print(f"[INFO] Comprehensive metrics enabled: {self.pub_metrics.experiment_dir}")

        # Track attention visualization count
        self._attention_viz_count = 0

    def save_attention_visualization(self, step: int, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Save attention pattern visualization for interpretability analysis.

        Generates attention heatmap from a forward pass through the model.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return  # Skip if matplotlib unavailable

        self.model.eval()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)

        # Get attention from forward pass
        with torch.no_grad():
            if hasattr(self.model, 'forward_with_attention'):
                _, attn_info = self.model.forward_with_attention(input_ids, targets=None)
                beta = attn_info.get('beta')

                if beta is not None:
                    # Average over heads: (B, H, N, N) -> (B, N, N)
                    if beta.dim() == 4:
                        attn = beta[0].mean(dim=0).cpu().numpy()
                    else:
                        attn = beta[0].cpu().numpy()

                    N = attn.shape[0]

                    # Mask diagonal (self-attention dominates) and use log scale
                    import numpy as np
                    attn_plot = attn.copy()
                    np.fill_diagonal(attn_plot, np.nan)
                    attn_plot = np.log10(np.maximum(attn_plot, 1e-6))

                    # Create visualization with focused colorbar range
                    # vmin=-3 (0.001), vmax=0 (1.0) to see medium-weight connections
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(attn_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)

                    ax.set_xlabel('Key Position (j)')
                    ax.set_ylabel('Query Position (i)')
                    ax.set_title(f'Attention Weights (Step {step}) [log₁₀, diag masked]')
                    plt.colorbar(im, ax=ax, label='log₁₀(β)')

                    # Save to checkpoint directory
                    save_dir = self.config.checkpoint_dir / 'attention_patterns'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(save_dir / f'attention_step_{step:06d}.png', dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    self._attention_viz_count += 1
                    if self._attention_viz_count == 1:
                        print(f"[INFO] Attention patterns saved to: {save_dir}/")

        self.model.train()

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics and AMP support."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Check if we should compute RG metrics this step
        compute_rg = (
            getattr(self.config, 'compute_rg_metrics', False) and
            self.global_step % getattr(self.config, 'rg_metrics_interval', 100) == 0
        )

        # Forward pass with full metrics (with optional AMP)
        if self.scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                loss, full_metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,
                    
                )
            # Scaled backward
            self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            loss, full_metrics = compute_free_energy_loss(
                self.model,
                input_ids,
                target_ids,
                alpha=self.config.alpha,
                lambda_beta=self.config.beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                
            )
            loss.backward()

        # Compute gradient norms BEFORE clipping
        # Check if this is a log step (need to check global_step here)
        is_log_step = (self.global_step + 1) % self.config.log_interval == 0
        grad_norms = self._compute_gradient_norms() if is_log_step else None

        # Clip and step (with scaler if AMP enabled)
        if self.scaler is not None:
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        # Format comprehensive metrics
        metrics = {
            'train_loss_total': full_metrics['loss/total'],
            'train_loss_ce': full_metrics['loss/ce'],
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_align': full_metrics.get('loss/model_align', 0),
            'train_ppl': math.exp(full_metrics['loss/ce']),
            'beta_mean': full_metrics.get('attention/beta_mean', 0),
            'beta_std': 0,  # Could compute if needed
            'kl_mean': full_metrics.get('attention/kl_mean', 0),
            'kl_std': 0,
            # Crucial attention interpretability metrics
            'attention_entropy': full_metrics.get('attention/entropy', 0),
            'attention_concentration': full_metrics.get('attention/concentration', 0),
        }

        # Carry over Hamiltonian diagnostics for physics metrics
        if 'hamiltonian_diagnostics' in full_metrics:
            metrics['hamiltonian_diagnostics'] = full_metrics['hamiltonian_diagnostics']

        # Compute RG metrics if enabled and attention info was returned
        if compute_rg and 'attention_info' in full_metrics:
            rg_metrics = compute_rg_metrics_from_attention(
                attn_info=full_metrics['attention_info'],
                step=self.global_step,
                auto_cluster=getattr(self.config, 'rg_auto_cluster', True),
                n_clusters=getattr(self.config, 'rg_n_clusters', None),
            )
            # Add RG metrics with proper key mapping for CSV
            metrics['rg/modularity'] = rg_metrics.get('rg/modularity')
            metrics['rg/effective_rank'] = rg_metrics.get('rg/effective_rank')
            metrics['rg/n_clusters'] = rg_metrics.get('rg/n_clusters')
            metrics['rg/kl_within_mean'] = rg_metrics.get('rg/kl_within_mean')
            metrics['rg/kl_within_std'] = rg_metrics.get('rg/kl_within_std')
            metrics['rg/kl_between_mean'] = rg_metrics.get('rg/kl_between_mean')
            metrics['rg/kl_between_std'] = rg_metrics.get('rg/kl_between_std')
            metrics['rg/beta_entropy'] = rg_metrics.get('rg/beta_entropy')

        return metrics, grad_norms

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        norms = {'total': 0, 'mu': 0, 'sigma': 0, 'phi': 0, 'ffn': 0}

        total_norm = 0
        mu_norm = 0
        sigma_norm = 0
        phi_norm = 0
        ffn_norm = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'mu_embed' in name or 'mu' in name.lower():
                    mu_norm += param_norm ** 2
                elif 'sigma_embed' in name or 'sigma' in name.lower() or 'L_embed' in name:
                    sigma_norm += param_norm ** 2
                elif 'phi_embed' in name or 'phi' in name.lower():
                    phi_norm += param_norm ** 2
                elif 'ffn' in name:
                    ffn_norm += param_norm ** 2

        norms['total'] = math.sqrt(total_norm)
        norms['mu'] = math.sqrt(mu_norm)
        norms['sigma'] = math.sqrt(sigma_norm)
        norms['phi'] = math.sqrt(phi_norm)
        norms['ffn'] = math.sqrt(ffn_norm)

        return norms

    def sample_text(
        self,
        prompt: str = "The",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        """
        Generate text to verify the model is learning.

        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling

        Returns:
            Generated text string
        """
        self.model.eval()

        # Get dataset which has encode/decode methods
        dataset = self.train_loader.dataset

        # Encode prompt using dataset's method
        prompt_ids = dataset.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode using dataset's method
        generated_text = dataset.decode(generated[0])

        self.model.train()
        return generated_text

    def train(self):
        """Training loop with publication metrics."""
        print(f"{'='*70}")
        print("PUBLICATION-QUALITY TRAINING")
        print(f"{'='*70}\n")

        start_time = time.time()
        train_iterator = iter(self.train_loader)

        try:
            from tqdm import tqdm
            pbar = tqdm(range(self.config.max_steps), desc="Training")
            use_tqdm = True
        except ImportError:
            pbar = range(self.config.max_steps)
            use_tqdm = False

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step with full metrics (grad_norms computed inside before zero_grad)
            metrics, grad_norms = self.train_step(batch)

            step_time = time.time() - step_start

            is_log_step = (step + 1) % self.config.log_interval == 0

            # Get learning rates
            lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

            # Log to basic tracker (only at log intervals)
            if is_log_step:
                batch_size = batch[0].shape[0]
                seq_len = batch[0].shape[1]
                self.metrics_tracker.log_step(
                    step + 1, metrics, lrs, grad_norms, step_time, batch_size, seq_len
                )

                # Log to comprehensive publication metrics (if enabled)
                if self.pub_metrics:
                    diagnostics = metrics.get('hamiltonian_diagnostics', None)
                    self.pub_metrics.record_training_step(
                        step=step + 1,
                        epoch=(step + 1) / len(self.train_loader),
                        train_metrics={
                            'loss': metrics['train_loss_total'],
                            'ce_loss': metrics['train_loss_ce'],
                        },
                        diagnostics=diagnostics,
                        grad_norms=grad_norms,
                        lrs=lrs,
                        step_time=step_time,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )

            # Console logging
            if is_log_step:
                log_msg = (
                    f"Step {step+1}/{self.config.max_steps} | "
                    f"Loss: {metrics['train_loss_total']:.4f} | "
                    f"CE: {metrics['train_loss_ce']:.4f} | "
                    f"β: {metrics['train_loss_belief_align']:.4f} | "
                    f"PPL: {metrics['train_ppl']:.1f}"
                )

                if use_tqdm:
                    pbar.set_description(log_msg)
                    # Print gradient norms using tqdm.write for proper display
                    if grad_norms:
                        tqdm.write(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                                   f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                                   f"phi: {grad_norms['phi']:.3e}")
                else:
                    print(log_msg)
                    if grad_norms:
                        print(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                              f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                              f"phi: {grad_norms['phi']:.3e}")

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.metrics_tracker.log_val(step + 1, val_metrics)

                # Log to comprehensive metrics
                if self.pub_metrics:
                    self.pub_metrics.record_validation(step + 1, val_metrics)

                # Log attention entropy/concentration for interpretability
                attn_entropy = metrics.get('attention_entropy', 0)
                attn_concentration = metrics.get('attention_concentration', 0)

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    CE: {val_metrics['ce_loss']:.4f}")
                print(f"    PPL: {val_metrics['perplexity']:.2f}")
                print(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}")
                print(f"    Attn entropy: {attn_entropy:.3f} | concentration: {attn_concentration:.3f}")

                # Log RG metrics if available (meta-agent emergence!)
                if metrics.get('rg/modularity') is not None:
                    print(f"    RG Metrics (meta-agent emergence):")
                    print(f"      Modularity Q: {metrics['rg/modularity']:.4f} (higher = more structure)")
                    print(f"      Effective rank: {metrics['rg/effective_rank']:.2f} (lower = concentrated)")
                    print(f"      Clusters (meta-agents): {metrics['rg/n_clusters']}")
                    print(f"      KL within: {metrics['rg/kl_within_mean']:.4f} (lower = tighter)")
                    print(f"      KL between: {metrics['rg/kl_between_mean']:.4f}")

                # Generate sample text to verify learning (varied prompts for diversity)
                try:
                    import random
                    prompts = ["The", "In", "A", "It", "This", "As", "One", "When", "For",
                               "After", "Before", "During", "While", "Although", "However"]
                    prompt = random.choice(prompts)
                    # Use temperature 0.9 and lower top_k for more diversity
                    sample = self.sample_text(prompt=prompt, max_new_tokens=30, temperature=0.9, top_k=30)
                    print(f"    Sample: {sample[:100]}...")
                except Exception as e:
                    import traceback
                    print(f"    Sample generation failed: {e}")
                    traceback.print_exc()
                print()

                # Save attention visualization periodically
                try:
                    sample_batch = next(iter(self.val_loader))
                    self.save_attention_visualization(step + 1, sample_batch)
                except StopIteration:
                    pass

                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    if self.config.patience > 0 and self.patience_counter >= self.config.patience:
                        print(f"\n[WARNING] Early stopping!")
                        break

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
                self.metrics_tracker.save()

        # Save final metrics
        self.metrics_tracker.save()
        print(f"\n[INFO] Final metrics saved to: {self.metrics_tracker.save_path}")

        # Save comprehensive publication metrics
        if self.pub_metrics:
            self.pub_metrics.save_all()
            self.pub_metrics.generate_all_figures()

            # Generate interpretability outputs using a sample batch from validation
            try:
                sample_batch = next(iter(self.val_loader))
                self.pub_metrics.generate_interpretability_outputs(
                    model=self.model,
                    sample_batch=sample_batch,
                    tokenizer=None,  # Byte-level, no tokenizer needed
                    device=self.device,
                )
            except Exception as e:
                import traceback
                print(f"[WARNING] Could not generate interpretability outputs: {e}")
                print(f"  Traceback: {traceback.format_exc()}")

            self.pub_metrics.print_summary()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Time: {elapsed/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        # Note: best_val_loss includes free energy terms, not just CE
        # PPL calculation is done in final eval using CE loss only
        print(f"{'='*70}\n")


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
    pure_fep: bool = False,
    prior_lr: float = 0.01,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('VFE_dynamic')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics
        pure_fep: If True, use backprop-free learning via prior evolution
        prior_lr: Learning rate for prior updates in pure FEP mode

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    if pure_fep:
        print(f"EXPERIMENT: PURE FEP MODE (Backprop-Free)")
    else:
        print(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    print("="*70)

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration at the START
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (BPE tokenization using GPT-2 tokenizer)
    # =================================================================

    dataset_name = config.get('dataset', 'wikitext-2')
    print("\n" + "="*70)
    print(f"LOADING {dataset_name.upper()} DATA")
    print("="*70)

    # Tokenizer selection: 'char', 'bpe', or 'auto' (default)
    # 'auto' uses char for vocab_size <= 256, bpe otherwise
    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config['vocab_size'] <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    if use_char:
        print(f"Using CHARACTER-LEVEL tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
        )
    else:
        print(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, actual_vocab_size = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config['vocab_size'],  # Top K BPE tokens
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
        )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Pure FEP Mode Configuration
    # =================================================================
    if pure_fep:
        config['ffn_pure_fep_mode'] = True
        config['ffn_prior_lr'] = prior_lr
        # CRITICAL: Disable gauge_fixed_priors for pure FEP!
        # We need per-token embeddings (mu_embed) that can be updated individually.
        # With gauge_fixed_priors=True, there's only one shared base_mu for all tokens,
        # which doesn't have enough capacity for token-specific learning.
        config['gauge_fixed_priors'] = False
        # CRITICAL: More VFE iterations for pure FEP!
        # With only 1 iteration, beliefs don't have time to minimize CE.
        # Need 5-10 iterations for beliefs to actually converge.
        config['ffn_n_iterations'] = 10
        print("\n" + "="*70)
        print("PURE FEP MODE ENABLED (Backprop-Free Learning)")
        print("="*70)
        print("  Learning via prior evolution - NO backprop!")
        print(f"  Prior learning rate: {prior_lr}")
        print("  gauge_fixed_priors: DISABLED (need per-token embeddings)")
        print("  ffn_n_iterations: 10 (beliefs need time to minimize CE)")
        print("  CE (cross-entropy) is INSIDE the VFE")
        print("  Beliefs adjust to minimize prediction error")
        print("  Priors update toward successful beliefs")
        print("  Embeddings update toward successful beliefs")
        print("="*70)

    # =================================================================
    # Model Creation
    # =================================================================

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"  FFN mode: {ffn_mode}")
    if pure_fep:
        print(f"  Pure FEP: ENABLED (backprop-free)")
    print(f"  N (seq len): {config['max_seq_len']}")
    print(f"  K (embed): {config['embed_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Vocab: {actual_vocab_size} ({'char' if use_char else 'BPE'})")

    # Create appropriate model based on mode
    if ffn_mode == 'standard':
        # Standard transformer baseline
        print("  Model type: STANDARD TRANSFORMER (dot-product attention)")
        model = StandardTransformerLM(
            vocab_size=actual_vocab_size,
            embed_dim=config['embed_dim'],
            n_layers=config['n_layers'],
            n_heads=config.get('n_heads', 1),
            hidden_dim=config.get('hidden_dim', config['embed_dim'] * 4),
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1),
        )
    else:
        # Gauge VFE transformer
        print("  Model type: GAUGE VFE TRANSFORMER (KL-divergence attention)")

        # Compute kappa_beta: either auto-scale with K or use fixed value
        K = config['embed_dim']
        if config.get('kappa_beta_auto_scale', False):
            kappa_base = config.get('kappa_beta_base', 1.0)
            K_ref = config.get('kappa_beta_k_ref', 11)
            config['kappa_beta'] = kappa_base * (K / K_ref)
            print(f"  kappa_beta: {kappa_base} × ({K}/{K_ref}) = {config['kappa_beta']:.3f} (auto-scaled)")
        else:
            # Backward compatibility: use kappa_beta directly if set, else compute from base
            if 'kappa_beta' not in config:
                config['kappa_beta'] = config.get('kappa_beta_base', 1.0)
            print(f"  kappa_beta: {config['kappa_beta']} (fixed)")

        model = GaugeTransformerLM(config)

    model = model.to(device)

    # Get parameter counts
    if hasattr(model, 'get_num_params'):
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        non_embed_params = sum(p.numel() for p in model.parameters() if 'embed' not in str(p))

    print(f"\nModel Parameters:")
    print(f"  Total:         {total_params:,}")
    print(f"  Non-embedding: {non_embed_params:,}")
    print(f"  Embedding:     {total_params - non_embed_params:,}")

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = FastTrainingConfig(
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Natural gradient learning rates
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        phi_lr=config['phi_lr'],
        attention_lr=config['phi_lr'],
        ffn_lr=config['ffn_lr'],
        output_lr=config['ffn_lr'],

        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],

        alpha=config['alpha'],
        beta=config['beta'],
        lambda_gamma=config['lambda_gamma'],

        log_interval=config['log_interval'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval'],

        use_wandb=use_wandb,
        checkpoint_dir=exp_checkpoint_dir,

        # GPU optimizations
        use_amp=config.get('use_amp', False),

       
    )

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Max steps:      {train_config.max_steps}")
    print(f"  Warmup:         {train_config.warmup_steps}")
    print(f"  Batch size:     {config['batch_size']}")
    print(f"  Seq length:     {config['max_seq_len']}")
    print(f"  Use AMP:        {train_config.use_amp}")
    print(f"  Num workers:    {config.get('num_workers', 0)}")
    print(f"\nFree Energy Weights:")
    print(f"  α (self-consistency): {train_config.alpha}")
    print(f"  β (belief align):     {train_config.beta}")
    print(f"  γ (model align):      {train_config.lambda_gamma}")

    

    # =================================================================
    # Create Trainer (Pure FEP or Standard)
    # =================================================================

    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)

    if pure_fep:
        # =========================================================
        # PURE FEP MODE: Backprop-free learning via prior evolution
        # =========================================================
        print("Mode: PURE FEP (Backprop-Free)")
        print(f"Prior learning rate: {prior_lr}")

        pure_fep_config = PureFEPConfig(
            prior_lr=prior_lr,
            max_seq_len=config['max_seq_len'],
            max_steps=config['max_steps'],
            log_every=config['log_interval'],
            eval_every=config['eval_interval'],
            save_every=config['checkpoint_interval'],
            checkpoint_dir=exp_checkpoint_dir,
            device=str(device),
        )

        trainer = PureFEPTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=pure_fep_config,
        )

        # =================================================================
        # Training (Pure FEP)
        # =================================================================

        print("\n" + "="*70)
        print("STARTING PURE FEP TRAINING (No Backprop!)")
        print("="*70)
        print(f"Device: {device}")
        print(f"Total steps: {pure_fep_config.max_steps:,}")
        print("\nLearning via prior evolution - beliefs update priors!")
        print("="*70 + "\n")

        try:
            trainer.train()

            print("\n" + "="*70)
            print("✓ PURE FEP TRAINING COMPLETE!")
            print("="*70)

            # Final evaluation
            final_metrics = pure_fep_validate(model, val_loader, device)

            print(f"\nFinal Validation Metrics:")
            print(f"  Loss:       {final_metrics['val/loss']:.4f}")
            print(f"  Perplexity: {final_metrics['val/perplexity']:.2f}")

            # vs random baseline
            random_ppl = actual_vocab_size
            improvement = random_ppl / final_metrics['val/perplexity']
            print(f"\nImprovement over random:")
            print(f"  Random:     {random_ppl:.0f}")
            print(f"  Model:      {final_metrics['val/perplexity']:.2f}")
            print(f"  Factor:     {improvement:.1f}x better!")

            # Return metrics
            return {
                'ffn_mode': ffn_mode,
                'pure_fep': True,
                'final_loss': final_metrics['val/loss'],
                'final_ppl': final_metrics['val/perplexity'],
                'random_ppl': random_ppl,
                'improvement': improvement,
                'total_params': total_params,
                'vocab_size': actual_vocab_size,
                'checkpoint': str(exp_checkpoint_dir / 'final_model.pt'),
            }

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("⚠ Training interrupted by user")
            print("="*70)
            return None

        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            raise

    else:
        # =========================================================
        # STANDARD MODE: Backprop-based training
        # =========================================================
        # Safety check: warn if model has pure_fep_mode but we're using standard training
        for block in model.transformer.blocks:
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'pure_fep_mode'):
                if block.ffn.pure_fep_mode:
                    print("\n⚠ WARNING: Model has pure_fep_mode=True but using standard training!")
                    print("  This may cause issues. Use --pure_fep flag for backprop-free training.")
                    print("  Or set ffn_pure_fep_mode=False in config.\n")
                break

        # Create comprehensive publication metrics tracker
        pub_metrics = None
        if enable_publication_metrics:
            experiment_name = f"{ffn_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
            pub_metrics = PublicationMetrics(
                experiment_name=experiment_name,
                base_dir=exp_checkpoint_dir / "publication_outputs"
            )

        trainer = PublicationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            publication_metrics=pub_metrics,
        )

        # =================================================================
        # Training (Standard Backprop)
        # =================================================================

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Device: {device}")
        print(f"FFN mode: {ffn_mode}")
        print(f"Total steps: {train_config.max_steps:,}")
        print("\nNOTE: First few batches may be slow (JIT compilation)")
        print("="*70 + "\n")

        try:
            trainer.train()

            print("\n" + "="*70)
            print("✓ TRAINING COMPLETE!")
            print("="*70)

            # Final evaluation
            final_metrics = trainer.validate()

            print(f"\nFinal Validation Metrics:")
            print(f"  Loss:       {final_metrics['loss']:.4f}")
            print(f"  Perplexity: {final_metrics['perplexity']:.2f}")

            # vs random baseline
            random_ppl = actual_vocab_size
            improvement = random_ppl / final_metrics['perplexity']
            print(f"\nImprovement over random:")
            print(f"  Random:     {random_ppl:.0f}")
            print(f"  Model:      {final_metrics['perplexity']:.2f}")
            print(f"  Factor:     {improvement:.1f}x better!")

            # Save final checkpoint
            final_ckpt = trainer.save_checkpoint(is_best=True)
            print(f"\n✓ Saved: {final_ckpt}")

            # Return metrics
            return {
                'ffn_mode': ffn_mode,
                'pure_fep': False,
                'final_loss': final_metrics['loss'],
                'final_ppl': final_metrics['perplexity'],
                'random_ppl': random_ppl,
                'improvement': improvement,
                'total_params': total_params,
                'vocab_size': actual_vocab_size,
                'checkpoint': str(final_ckpt),
            }

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("TRAINING INTERRUPTED")
            print("="*70)
            ckpt = trainer.save_checkpoint(is_best=False)
            print(f"✓ Saved: {ckpt}")
            return None

        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Publication Training Script')

    # FFN mode (uses defaults from top of file)
    parser.add_argument('--ffn_mode', type=str, default=DEFAULT_FFN_MODE,
                        choices=['VFE_dynamic', 'standard'],
                        help='FFN mode: VFE_dynamic (gauge VFE) or standard (baseline transformer)')

    # Enable full geometric learning (Σ and φ)
    parser.add_argument('--enable_sigma_phi', action='store_true', default=DEFAULT_ENABLE_SIGMA_PHI,
                        help='Enable learning covariances (Σ) and gauge frames (φ) - full geometric learning!')

    # Pure FEP mode (backprop-free learning)
    parser.add_argument('--pure_fep', action='store_true', default=False,
                        help='Enable pure FEP mode: learning via prior evolution, NO backprop!')
    parser.add_argument('--prior_lr', type=float, default=0.1,
                        help='Learning rate for prior updates in pure FEP mode (default: 0.1)')
    parser.add_argument('--embed_lr', type=float, default=0.1,
                        help='Learning rate for embedding updates in pure FEP mode (default: 0.1)')

    # GPU optimization
    parser.add_argument('--gpu_optimized', action='store_true', default=DEFAULT_USE_GPU_OPTIMIZED,
                        help='Use GPU-optimized config for high-end GPUs')
    parser.add_argument('--no_gpu_optimized', action='store_true',
                        help='Force use of smaller config even on GPU')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_publication')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None = random)')
    parser.add_argument('--dataset', type=str, default='wikitext-2',
                        choices=['wikitext-2', 'wikitext-103'],
                        help='Dataset to use: wikitext-2 (~2M tokens) or wikitext-103 (~103M tokens)')

    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("PUBLICATION PROOF-OF-PRINCIPLE TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Run single mode
    if args.ffn_mode is None:
        print("\nError: Must specify --ffn_mode")
        print("Edit DEFAULT_FFN_MODE at top of train_publication.py or use command-line args")
        return

    # Select config based on mode
    if args.ffn_mode == 'standard':
        print("\n" + "="*70)
        print("STANDARD TRANSFORMER BASELINE")
        print("="*70)
        print("   Using dot-product attention + learned MLP")
        print("   This is the comparison baseline!")
        print("="*70 + "\n")
        base_config = STANDARD_TRANSFORMER_CONFIG.copy()
    else:
        base_config = GPU_OPTIMIZED_CONFIG.copy()

    config = base_config.copy()
    config['ffn_mode'] = args.ffn_mode
    config['dataset'] = args.dataset

    # Enable full geometric learning if requested
    if args.enable_sigma_phi:
        print("\n" + "="*70)
        print("FULL GEOMETRIC LEARNING ENABLED")
        print("="*70)
        print("   Learning: μ (means), Σ (covariances), φ (gauge frames)")
        print("   This tests the FULL natural gradient framework!")
        print("="*70 + "\n")
        config['evolve_sigma'] = True
        config['evolve_phi'] = True

    # Pure FEP mode announcement
    if args.pure_fep:
        print("\n" + "="*70)
        print("PURE FEP MODE (BELIEF: Backprop-free Evolving Local Inference)")
        print("="*70)
        print("   Learning via prior evolution - NO backprop!")
        print(f"   Prior learning rate: {args.prior_lr}")
        print("="*70 + "\n")

    result = run_single_experiment(
        config=config,
        ffn_mode=args.ffn_mode,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb,
        args=args,
        pure_fep=args.pure_fep,
        prior_lr=args.prior_lr,
    )

    if result is not None:
        # Save result
        mode_suffix = "pure_fep" if args.pure_fep else args.ffn_mode
        result_file = checkpoint_dir / f"result_{mode_suffix}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result: {result_file}")

    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)


if __name__ == '__main__':

    main()
