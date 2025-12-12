#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure FEP Transformer Training Script
=====================================

Entry point for training the Pure FEP Transformer.

Usage:
    python -m transformer.train_pure_fep --help
    python -m transformer.train_pure_fep --epochs 10 --embed-dim 128
    python -m transformer.train_pure_fep --dataset wikitext2 --batch-size 32

Author: Chris & Claude
Date: December 2025
"""

import click
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
from datetime import datetime

from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer, PureFEPTrainer


@click.command()
@click.option('--embed-dim', default=127, help='Embedding dimension K (must be ODD for SO(3) irreps)')
@click.option('--num-layers', default=2, help='Number of hierarchical layers')
@click.option('--seq-length', default=128, help='Sequence length')
@click.option('--vocab-size', default=10000, help='Vocabulary size (ignored if using dataset)')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--mu-lr', default=0.1, help='Belief mean learning rate')
@click.option('--sigma-lr', default=0.025, help='Belief variance learning rate')
@click.option('--prior-lr', default=0.05, help='Prior update learning rate')
@click.option('--phi-lr', default=0.05, help='Gauge frame learning rate')
@click.option('--alpha', default=0.01, help='Self-coupling weight KL(q||p)')
@click.option('--lambda-belief', default=1.0, help='Belief alignment weight')
@click.option('--kappa', default=1.0, help='Attention temperature')
@click.option('--n-vfe-steps', default=1, help='VFE iterations per forward pass')
@click.option('--prior-update-interval', default=10, help='Steps between prior updates')
@click.option('--grad-clip', default=1.0, help='Gradient clipping norm')
@click.option('--dataset', default='synthetic', type=click.Choice(['synthetic', 'wikitext2']), help='Dataset to use')
@click.option('--data-dir', default='./data', help='Data directory')
@click.option('--log-interval', default=100, help='Log every N steps')
@click.option('--save-dir', default='./checkpoints/pure_fep', help='Checkpoint save directory')
@click.option('--device', default='auto', help='Device: auto, cuda, cpu')
@click.option('--seed', default=42, help='Random seed')
@click.option('--debug', is_flag=True, help='Debug mode (small dataset, verbose)')
def main(
    embed_dim, num_layers, seq_length, vocab_size, batch_size, epochs,
    mu_lr, sigma_lr, prior_lr, phi_lr, alpha, lambda_belief, kappa,
    n_vfe_steps, prior_update_interval, grad_clip, dataset, data_dir,
    log_interval, save_dir, device, seed, debug
):
    """Train Pure FEP Transformer - learning via VFE minimization, not backprop."""

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    click.echo(f"\n{'='*60}")
    click.echo("PURE FEP TRANSFORMER TRAINING")
    click.echo(f"{'='*60}")
    click.echo(f"Device: {device}")
    click.echo(f"Dataset: {dataset}")

    # Validate embed_dim is odd (required for SO(3) irreps)
    if embed_dim % 2 == 0:
        click.echo(f"\nERROR: embed_dim must be ODD for SO(3) irreps (got {embed_dim})")
        click.echo(f"Suggestion: use --embed-dim {embed_dim + 1} or --embed-dim {embed_dim - 1}")
        raise click.Abort()

    # Debug mode adjustments
    if debug:
        click.echo("\n[DEBUG MODE] Using small config")
        embed_dim = 63  # Must be odd!
        num_layers = 1
        seq_length = 32
        batch_size = 4
        epochs = 2
        log_interval = 10

    # Load data
    click.echo(f"\nLoading data...")
    train_loader, val_loader, actual_vocab_size = load_data(
        dataset, data_dir, batch_size, seq_length, debug
    )

    # Use actual vocab size from dataset
    if actual_vocab_size is not None:
        vocab_size = actual_vocab_size

    click.echo(f"  Train batches: {len(train_loader)}")
    click.echo(f"  Val batches: {len(val_loader)}")
    click.echo(f"  Vocab size: {vocab_size}")

    # Create config
    config = PureFEPConfig(
        embed_dim=embed_dim,
        num_layers=num_layers,
        seq_length=seq_length,
        vocab_size=vocab_size,
        alpha=alpha,
        lambda_belief=lambda_belief,
        kappa=kappa,
        mu_lr=mu_lr,
        sigma_lr=sigma_lr,
        prior_lr=prior_lr,
        phi_lr=phi_lr,
        belief_steps=n_vfe_steps,
        prior_update_interval=prior_update_interval,
        diagonal_covariance=True,
        grad_clip=grad_clip,
    )

    click.echo(f"\nModel config:")
    click.echo(f"  embed_dim (K): {config.embed_dim}")
    click.echo(f"  num_layers: {config.num_layers}")
    click.echo(f"  mu_lr: {config.mu_lr}")
    click.echo(f"  prior_lr: {config.prior_lr}")
    click.echo(f"  alpha: {config.alpha}")
    click.echo(f"  lambda_belief: {config.lambda_belief}")
    click.echo(f"  n_vfe_steps: {n_vfe_steps}")

    # Create model
    click.echo(f"\nCreating model...")
    model = PureFEPTransformer(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"  Total parameters: {total_params:,}")
    click.echo(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = PureFEPTrainer(model, device)

    # Training loop
    click.echo(f"\n{'='*60}")
    click.echo("TRAINING")
    click.echo(f"{'='*60}")

    best_val_ppl = float('inf')
    history = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, device, n_vfe_steps, log_interval, epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        # Log
        click.echo(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        click.echo(f"  Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
        click.echo(f"  Val Loss: {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")

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
            save_checkpoint(model, config, history, save_dir, 'best')
            click.echo(f"  [NEW BEST] Saved checkpoint")

    # Save final model
    save_checkpoint(model, config, history, save_dir, 'final')

    click.echo(f"\n{'='*60}")
    click.echo("TRAINING COMPLETE")
    click.echo(f"{'='*60}")
    click.echo(f"Best Val PPL: {best_val_ppl:.2f}")
    click.echo(f"Checkpoints saved to: {save_dir}")


def load_data(dataset, data_dir, batch_size, seq_length, debug=False):
    """Load training and validation data."""

    if dataset == 'wikitext2':
        try:
            from transformer.data import WikiText2Dataset, create_dataloaders

            train_loader, val_loader = create_dataloaders(
                batch_size=batch_size,
                seq_length=seq_length,
                data_dir=data_dir,
            )

            vocab_size = len(train_loader.dataset.vocab_mapping)
            return train_loader, val_loader, vocab_size

        except Exception as e:
            click.echo(f"  Warning: Could not load WikiText-2: {e}")
            click.echo(f"  Falling back to synthetic data")
            dataset = 'synthetic'

    if dataset == 'synthetic':
        # Synthetic data for testing
        vocab_size = 1000 if debug else 10000
        n_train = 100 if debug else 5000
        n_val = 20 if debug else 500

        train_loader = SyntheticDataLoader(
            n_samples=n_train,
            seq_length=seq_length,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )
        val_loader = SyntheticDataLoader(
            n_samples=n_val,
            seq_length=seq_length,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )

        return train_loader, val_loader, vocab_size

    raise ValueError(f"Unknown dataset: {dataset}")


class SyntheticDataLoader:
    """Simple synthetic data loader for testing."""

    def __init__(self, n_samples, seq_length, vocab_size, batch_size):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        # Generate random data
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


def train_epoch(model, dataloader, device, n_vfe_steps, log_interval, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        # Training step
        metrics = model.train_step(input_ids, targets, n_vfe_steps=n_vfe_steps)

        total_loss += metrics['ce_loss']
        total_ppl += metrics['perplexity']
        n_batches += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / n_batches
            avg_ppl = total_ppl / n_batches
            click.echo(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")

    return {
        'loss': total_loss / n_batches,
        'perplexity': total_ppl / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            targets.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return {'loss': avg_loss, 'perplexity': ppl}


def save_checkpoint(model, config, history, save_dir, name):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'seq_length': config.seq_length,
            'vocab_size': config.vocab_size,
            'alpha': config.alpha,
            'lambda_belief': config.lambda_belief,
            'kappa': config.kappa,
            'mu_lr': config.mu_lr,
            'sigma_lr': config.sigma_lr,
            'prior_lr': config.prior_lr,
            'phi_lr': config.phi_lr,
        },
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_dir / f'{name}.pt')

    # Also save config as JSON for easy inspection
    with open(save_dir / f'{name}_config.json', 'w') as f:
        json.dump(checkpoint['config'], f, indent=2)


if __name__ == '__main__':
    main()
