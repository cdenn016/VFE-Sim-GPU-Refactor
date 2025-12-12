#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure FEP Transformer Training Script
=====================================

Language modeling with PURE Free Energy Principle learning.
NO Adam, NO backprop through VFE - just hierarchical belief/prior dynamics.

Usage:
    # Just edit the config below and run:
    python transformer/train_pure_fep.py

Author: Chris & Claude
Date: December 2025
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
import math
from datetime import datetime

from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer


# =============================================================================
# EDIT THIS CONFIG - Just click Run!
# =============================================================================

CONFIG = {
    # Architecture (embed_dim MUST be ODD for SO(3) irreps!)
    'embed_dim': 127,             # K - embedding dimension
    'num_layers': 2,              # Hierarchical scales
    'seq_length': 128,            # Max sequence length
    'vocab_size': 10000,          # Will be overwritten if using dataset

    # VFE parameters
    'alpha': 0.01,                # Self-coupling weight KL(q||p)
    'lambda_belief': 1.0,         # Belief alignment weight
    'kappa': 1.0,                 # Attention temperature

    # Learning rates (natural gradient allows larger steps!)
    'mu_lr': 0.1,                 # Belief mean update
    'sigma_lr': 0.025,            # Belief variance update
    'prior_lr': 0.05,             # Prior update (slower = learning timescale)
    'phi_lr': 0.05,               # Gauge frame update

    # Timescales
    'n_vfe_steps': 1,             # VFE iterations per forward pass
    'prior_update_interval': 10,  # Steps between hierarchical prior updates

    # Stability
    'grad_clip': 1.0,

    # Training
    'batch_size': 32,
    'epochs': 10,
    'log_interval': 100,

    # Data
    'dataset': 'synthetic',       # 'synthetic' or 'wikitext2'
    'data_dir': './data',

    # Output
    'save_dir': './checkpoints/pure_fep',

    # Device
    'device': 'auto',             # 'auto', 'cuda', 'cpu'

    # Reproducibility
    'seed': 42,
}

# Debug mode - uncomment to use smaller config for testing
# CONFIG.update({
#     'embed_dim': 63,
#     'num_layers': 1,
#     'seq_length': 32,
#     'batch_size': 4,
#     'epochs': 2,
#     'log_interval': 10,
# })


# =============================================================================
# DATA LOADING
# =============================================================================

class SyntheticDataLoader:
    """Simple synthetic data for testing."""

    def __init__(self, n_samples, seq_length, vocab_size, batch_size):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data = torch.randint(0, vocab_size, (n_samples, seq_length + 1))

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = torch.randperm(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_data = self.data[batch_idx]
            yield {
                'input_ids': batch_data[:, :-1],
                'targets': batch_data[:, 1:],
            }


def load_data(config):
    """Load training and validation data."""
    dataset = config['dataset']
    batch_size = config['batch_size']
    seq_length = config['seq_length']

    if dataset == 'wikitext2':
        try:
            from transformer.data import create_dataloaders
            train_loader, val_loader = create_dataloaders(
                batch_size=batch_size,
                seq_length=seq_length,
                data_dir=config['data_dir'],
            )
            vocab_size = len(train_loader.dataset.vocab_mapping)
            return train_loader, val_loader, vocab_size
        except Exception as e:
            print(f"  Warning: Could not load WikiText-2: {e}")
            print(f"  Falling back to synthetic data")

    # Synthetic data
    vocab_size = config['vocab_size']
    train_loader = SyntheticDataLoader(5000, seq_length, vocab_size, batch_size)
    val_loader = SyntheticDataLoader(500, seq_length, vocab_size, batch_size)
    return train_loader, val_loader, vocab_size


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        metrics = model.train_step(
            input_ids, targets,
            n_vfe_steps=config['n_vfe_steps']
        )

        total_loss += metrics['ce_loss']
        total_ppl += metrics['perplexity']
        n_batches += 1

        if batch_idx % config['log_interval'] == 0 and batch_idx > 0:
            avg_loss = total_loss / n_batches
            avg_ppl = total_ppl / n_batches
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")

    return {
        'loss': total_loss / n_batches,
        'perplexity': total_ppl / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, vocab_size):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow

    return {'loss': avg_loss, 'perplexity': ppl}


def save_checkpoint(model, config, history, save_dir, name):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_dir / f'{name}.pt')

    with open(save_dir / f'{name}_config.json', 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = CONFIG.copy()

    # Validate embed_dim is odd
    if config['embed_dim'] % 2 == 0:
        raise ValueError(
            f"embed_dim must be ODD for SO(3) irreps (got {config['embed_dim']}). "
            f"Try {config['embed_dim'] - 1} or {config['embed_dim'] + 1}."
        )

    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"\n{'='*60}")
    print("PURE FEP TRANSFORMER TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dataset: {config['dataset']}")

    # Load data
    print(f"\nLoading data...")
    train_loader, val_loader, vocab_size = load_data(config)
    config['vocab_size'] = vocab_size

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Vocab size: {vocab_size}")

    # Create model config
    model_config = PureFEPConfig(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        seq_length=config['seq_length'],
        vocab_size=vocab_size,
        alpha=config['alpha'],
        lambda_belief=config['lambda_belief'],
        kappa=config['kappa'],
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        prior_lr=config['prior_lr'],
        phi_lr=config['phi_lr'],
        belief_steps=config['n_vfe_steps'],
        prior_update_interval=config['prior_update_interval'],
        grad_clip=config['grad_clip'],
    )

    print(f"\nModel config:")
    print(f"  embed_dim (K): {model_config.embed_dim}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  mu_lr: {model_config.mu_lr}")
    print(f"  prior_lr: {model_config.prior_lr}")
    print(f"  alpha: {model_config.alpha}")
    print(f"  n_vfe_steps: {config['n_vfe_steps']}")

    # Create model
    print(f"\nCreating model...")
    model = PureFEPTransformer(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_val_ppl = float('inf')
    history = []

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, device, config, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, device, vocab_size)

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")

        # Track history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_ppl': train_metrics['perplexity'],
            'val_loss': val_metrics['loss'],
            'val_ppl': val_metrics['perplexity'],
            'time': epoch_time,
        })

        # Save best model
        if val_metrics['perplexity'] < best_val_ppl:
            best_val_ppl = val_metrics['perplexity']
            save_checkpoint(model, config, history, config['save_dir'], 'best')
            print(f"  [NEW BEST] Saved checkpoint")

    # Save final model
    save_checkpoint(model, config, history, config['save_dir'], 'final')

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val PPL: {best_val_ppl:.2f}")
    print(f"Checkpoints saved to: {config['save_dir']}")


if __name__ == '__main__':
    main()
