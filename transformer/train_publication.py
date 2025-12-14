# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:24:37 2025

@author: chris and christine
"""

"""
Gauge Transformer Training Script
=================================

Language modeling on WikiText-2 with gauge-theoretic attention.

Two FFN Modes:
    - learned: Standard MLP baseline (GELU activation)
    - VFE_dynamic: Variational Free Energy with natural gradient descent

Usage:
    # Just click Run (edit defaults below)
    python transformer/train_publication.py

    # Or use command-line args:
    python transformer/train_publication.py --ffn_mode learned
    python transformer/train_publication.py --ffn_mode VFE_dynamic
    python transformer/train_publication.py --run_ablation  # Compare both

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
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import compute_free_energy_loss, compute_rg_metrics_from_attention
from transformer.train_fast import FastTrainer, FastTrainingConfig
from transformer.publication_metrics import PublicationMetrics, AblationConfig, AblationResult


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

    print(f"Saved experiment config: {config_path}")

    return config_path


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS
# ============================================================================
DEFAULT_FFN_MODE = 'VFE_dynamic'  # 'learned' or 'VFE_dynamic'
DEFAULT_RUN_ABLATION = False      # Set True to compare learned vs VFE_dynamic
# ============================================================================


# =============================================================================
# SIMPLIFIED CONFIG
# =============================================================================
# Mode 1: 'learned'      - Gauge transformer with KL attention + learned MLP FFN
# Mode 2: 'VFE_dynamic'  - Gauge transformer with KL attention + VFE dynamic FFN
#
# Hamiltonian mode moved to experimental/hamiltonian_transformer.py
# =============================================================================

BASE_CONFIG = {
    # === MODEL ARCHITECTURE ===
    'vocab_size': 50000,      # BPE vocabulary size
    'embed_dim': 67,          # Embedding dimension K (ODD for SO(3))
    'n_layers': 1,            # Number of transformer layers
    'hidden_dim': 508,        # FFN hidden dimension (only for 'learned')
    'max_seq_len': 28,        # Maximum sequence length

    # === TRAINING ===
    'batch_size': 20,
    'max_steps': 5000,
    'warmup_steps': 25,
    'use_amp': False,         # Mixed precision (disable for stability)
    'num_workers': 4,

    # === LEARNING RATES ===
    'mu_lr': 0.2,             # Embedding means
    'sigma_lr': 0.01,         # Embedding covariances
    'phi_lr': 0.05,           # Gauge frames
    'ffn_lr': 0.2,            # FFN parameters

    # === REGULARIZATION ===
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,

    # === FREE ENERGY WEIGHTS ===
    'alpha': 1.0,             # Self-consistency (KL to prior)
    'beta': 1.0,              # Belief alignment
    'lambda_gamma': 0.0,      # Model alignment (usually 0)
    'kappa_gamma': 1.0,

    # === GAUGE ATTENTION ===
    'kappa_beta': 0.5,        # Attention temperature
    'pos_encoding_mode': 'learned',
    'pos_encoding_scale': 0.1,
    'use_positional_embedding': True,  # Add position to μ (standard approach)

    # === GEOMETRY ===
    'evolve_sigma': True,     # Learn covariances
    'evolve_phi': True,       # Learn gauge frames
    'diagonal_covariance': False,
    'tie_embeddings': False,

    # === FFN MODE (set by mode selection) ===
    'ffn_mode': 'VFE_dynamic',
    'ffn_n_iterations': 1,
    'ffn_learnable_lr': True,
    'ffn_kappa': 0.5,

    # === LOGGING ===
    'log_interval': 10,
    'eval_interval': 200,
    'checkpoint_interval': 1000,
    'patience': 5,

    # === IRREP STRUCTURE (for embed_dim=67) ===
    'irrep_spec': [
        ('ℓ0', 12, 1),   # 12 scalars
        ('ℓ1', 10, 3),   # 30 vectors
        ('ℓ2', 5, 5),    # 25 tensors
        # Total: 12 + 30 + 25 = 67
    ],
}


def get_mode_config(mode: str) -> dict:
    """
    Get configuration for a specific mode.

    Modes:
        'learned':     Gauge transformer + learned MLP FFN
        'VFE_dynamic': Gauge transformer + VFE dynamic FFN

    Returns complete config dict ready for training.
    """
    config = BASE_CONFIG.copy()

    if mode == 'learned':
        # Standard MLP FFN (baseline)
        config['ffn_mode'] = 'learned'
        config['ffn_learnable_lr'] = False  # Not used for learned mode

    elif mode == 'VFE_dynamic':
        # Variational Free Energy dynamic FFN
        config['ffn_mode'] = 'VFE_dynamic'
        config['ffn_n_iterations'] = 1
        config['ffn_learnable_lr'] = True

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'learned' or 'VFE_dynamic'")

    return config


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
                    ax.set_title(f'Attention Weights (Step {step}) [log10, diag masked]')
                    plt.colorbar(im, ax=ax, label='log10(beta)')

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
        print("GAUGE TRANSFORMER TRAINING")
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
                    f"beta: {metrics['train_loss_belief_align']:.4f} | "
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

                # Generate sample text to verify learning
                try:
                    sample = self.sample_text(prompt="The", max_new_tokens=30, temperature=0.8)
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
        print(f"{'='*70}\n")


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('learned' or 'VFE_dynamic')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
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

    print("\n" + "="*70)
    print("LOADING WIKITEXT-2 DATA")
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
        )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation
    # =================================================================

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"  FFN mode: {ffn_mode}")
    print(f"  N (seq len): {config['max_seq_len']}")
    print(f"  K (embed): {config['embed_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Vocab: {actual_vocab_size} ({'char' if use_char else 'BPE'})")

    model = GaugeTransformerLM(config)
    model = model.to(device)

    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)

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
    print(f"  alpha (self-consistency): {train_config.alpha}")
    print(f"  beta (belief align):     {train_config.beta}")
    print(f"  gamma (model align):      {train_config.lambda_gamma}")



    # =================================================================
    # Create Trainer
    # =================================================================

    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)

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
    # Training
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
        print("TRAINING COMPLETE!")
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
        print(f"\nSaved: {final_ckpt}")

        # Return metrics
        return {
            'ffn_mode': ffn_mode,
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
        print(f"Saved: {ckpt}")
        return None

    except Exception as e:
        print(f"\n\nError: {e}")
        raise


def run_ablation_study(
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
) -> List[Dict]:
    """
    Run ablation study comparing learned vs VFE_dynamic FFN modes.

    Args:
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging

    Returns:
        List of result dictionaries for each FFN mode
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: LEARNED vs VFE_DYNAMIC")
    print("="*70)

    print("\nWill run:")
    print("  1. learned     (baseline - standard MLP with GELU)")
    print("  2. VFE_dynamic (variational free energy - natural gradient)")
    print("="*70)

    modes = ['learned', 'VFE_dynamic']
    results = []

    for i, mode in enumerate(modes):
        print(f"\n\n{'='*70}")
        print(f"EXPERIMENT {i+1}/{len(modes)}: {mode}")
        print("="*70)

        # Get mode-specific config from simplified system
        config = get_mode_config(mode)

        result = run_single_experiment(
            config=config,
            ffn_mode=mode,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb,
            args=args,
        )

        if result is not None:
            results.append(result)

    # Save combined results
    results_file = checkpoint_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)

    # Print comparison
    print("\nResults Comparison:")
    print("-"*70)
    print(f"{'Mode':<20} {'PPL':>10} {'vs Random':>12} {'vs Learned':>12}")
    print("-"*70)

    learned_ppl = None
    for result in results:
        mode = result['ffn_mode']
        ppl = result['final_ppl']
        improvement = result['improvement']

        if mode == 'learned':
            learned_ppl = ppl
            vs_learned = "baseline"
        elif learned_ppl is not None:
            diff_pct = ((ppl - learned_ppl) / learned_ppl) * 100
            vs_learned = f"{diff_pct:+.1f}%"
        else:
            vs_learned = "N/A"

        print(f"{mode:<20} {ppl:>10.2f} {improvement:>11.1f}x {vs_learned:>12}")

    print("-"*70)
    print(f"\nSaved: {results_file}")

    # Check comparison
    print("\n" + "="*70)
    print("COMPARISON: LEARNED vs VFE_DYNAMIC")
    print("="*70)

    if learned_ppl is not None:
        for result in results:
            if result['ffn_mode'] == 'VFE_dynamic':
                mode = result['ffn_mode']
                ppl = result['final_ppl']
                diff_pct = ((ppl - learned_ppl) / learned_ppl) * 100

                print(f"\n{mode}:")
                print(f"  Learned PPL:     {learned_ppl:.2f}")
                print(f"  VFE_dynamic PPL: {ppl:.2f}")
                print(f"  Difference:      {diff_pct:+.1f}%")

                if abs(diff_pct) < 20:
                    print(f"  Comparable performance!")
                elif diff_pct < 0:
                    print(f"  VFE_dynamic outperforms learned!")
                else:
                    print(f"  VFE_dynamic underperforms - may need tuning")

    return results


def main():
    parser = argparse.ArgumentParser(description='Gauge Transformer Training')

    # FFN mode - simplified to 2 main modes
    parser.add_argument('--ffn_mode', type=str, default=DEFAULT_FFN_MODE,
                        choices=['learned', 'VFE_dynamic'],
                        help='FFN mode: learned (MLP baseline) or VFE_dynamic (variational)')

    # Ablation study
    parser.add_argument('--run_ablation', action='store_true', default=DEFAULT_RUN_ABLATION,
                        help='Run both learned and VFE_dynamic modes for comparison')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_publication')
    parser.add_argument('--use_wandb', action='store_true')

    # Config overrides (optional)
    parser.add_argument('--embed_dim', type=int, default=None, help='Override embed_dim')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max_steps')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch_size')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("GAUGE TRANSFORMER TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Run experiments
    if args.run_ablation:
        # Run both modes for comparison
        results = run_ablation_study(
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            args=args,
        )

    else:
        # Run single mode using simplified config
        if args.ffn_mode is None:
            print("\nError: Must specify --ffn_mode or --run_ablation")
            print("Edit DEFAULT_FFN_MODE at top of train_publication.py or use command-line args")
            return

        # Get mode-specific config
        config = get_mode_config(args.ffn_mode)

        # Apply any command-line overrides
        if args.embed_dim is not None:
            config['embed_dim'] = args.embed_dim
            config['hidden_dim'] = args.embed_dim * 4
        if args.max_steps is not None:
            config['max_steps'] = args.max_steps
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size

        result = run_single_experiment(
            config=config,
            ffn_mode=args.ffn_mode,
            device=device,
            checkpoint_dir=checkpoint_dir,
            use_wandb=args.use_wandb,
            args=args,
        )

        if result is not None:
            # Save result
            result_file = checkpoint_dir / f"result_{args.ffn_mode}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved result: {result_file}")

    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
