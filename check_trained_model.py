#!/usr/bin/env python
"""
Check your trained model to find why attention is uniform.

Run this with your actual trained model!
"""

import torch
import numpy as np

# =============================================================================
# STEP 1: Load your trained model
# =============================================================================
print("="*80)
print("CHECKING TRAINED MODEL FOR UNIFORM ATTENTION")
print("="*80)

# TODO: Replace with your actual model loading code
print("\n[LOADING MODEL]")
print("TODO: Load your trained model here!")
print("Example:")
print("  from transformer.model import GaugeTransformerLM")
print("  model = torch.load('checkpoints/your_checkpoint.pt')")
print("  OR")
print("  model = GaugeTransformerLM(config)")
print("  model.load_state_dict(torch.load('checkpoints/state_dict.pt'))")

# For testing, create a fresh model (YOU SHOULD REPLACE THIS!)
from transformer.model import GaugeTransformerLM

# =============================================================================
# OPTION 1: Load from checkpoint (RECOMMENDED)
# =============================================================================
# Uncomment this if you have a saved checkpoint:
# checkpoint_path = 'path/to/your/checkpoint.pt'
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# model = checkpoint['model']  # Or however your checkpoint is structured
# config = checkpoint.get('config', {})  # Get config from checkpoint

# =============================================================================
# OPTION 2: Recreate model with YOUR config
# =============================================================================
# You MUST provide the EXACT config you used for training!
# Copy the config from your training script here:

config = {
    'vocab_size': 50257,  # Your vocab size
    'embed_dim': 255,  # CRITICAL: What is your actual embed_dim?
    'n_layers': 4,
    'hidden_dim': 1024,
    'max_seq_len': 128,
    'kappa_beta': 1.0,  # CRITICAL: Is this your actual kappa?
    'mask_self_attention': True,
    'evolve_sigma': True,

    # CRITICAL: You MUST provide your irrep_spec!
    # You said "spin-0 ...19 heads" - what is the actual spec?
    # Examples:
    # - 19 spin-0 heads: [('l0', 19, 1)]  (total dim = 19)
    # - Mixed: [('l0', 10, 1), ('l1', 3, 3), ...]
    # - Spin-0 through spin-9: [('l0', 1, 1), ('l1', 1, 3), ('l2', 1, 5), ...]

    'irrep_spec': [
        # REPLACE THIS with your actual irrep_spec!
        # For now, assuming 19 spin-0 heads:
        ('l0', 19, 1),
        # If total embed_dim = 255, remaining dims need other irreps:
        # 255 - 19 = 236 remaining dimensions
        # You might have something like:
        # ('l1', 10, 3),  # 10 spin-1 heads × 3 dims = 30
        # ('l2', 8, 5),   # 8 spin-2 heads × 5 dims = 40
        # etc.
        # Total must equal embed_dim!
    ],
}

print("\n⚠️  WARNING: Using hardcoded config!")
print("   Modify the irrep_spec above to match your training config.")
print("\nCurrent config:")
for key in ['embed_dim', 'n_layers', 'kappa_beta', 'mask_self_attention', 'irrep_spec']:
    print(f"  {key}: {config.get(key)}")

# Verify irrep_spec dimensions add up
if 'irrep_spec' in config:
    total_dim = sum(num * dim for name, num, dim in config['irrep_spec'])
    print(f"\nIrrep spec dimensions: {total_dim}")
    if total_dim != config['embed_dim']:
        print(f"  ❌ ERROR: irrep_spec total ({total_dim}) != embed_dim ({config['embed_dim']})")
        print(f"     Fix your irrep_spec to match embed_dim!")
        raise ValueError("Irrep dimensions mismatch")
    else:
        print(f"  ✓ Dimensions match!")

# Create model
model = GaugeTransformerLM(config)
model.eval()

print("\n⚠️  NOTE: This is a FRESH model with random weights!")
print("   To diagnose your TRAINED model, load it from checkpoint above.")

# =============================================================================
# STEP 2: Get a test sequence
# =============================================================================
print("\n[TEST SEQUENCE]")
token_ids = torch.randint(0, 100, (1, 16))
print(f"  Shape: {token_ids.shape}")
print(f"  Tokens: {token_ids[0].tolist()}")

# =============================================================================
# STEP 3: Check embeddings
# =============================================================================
print("\n[CHECKING EMBEDDINGS]")

with torch.no_grad():
    mu, sigma, phi = model.token_embed(token_ids)

mu_np = mu[0].cpu().numpy()  # (N, K)
phi_np = phi[0].cpu().numpy()  # (N, 3)

# Check μ diversity
mu_dists = torch.cdist(mu[0], mu[0], p=2)
mu_dist_mean = mu_dists.mean().item()
mu_dist_std = mu_dists.std().item()

print(f"  μ diversity:")
print(f"    Mean pairwise distance: {mu_dist_mean:.4f}")
print(f"    Std of distances: {mu_dist_std:.4f}")

if mu_dist_mean < 0.5:
    print(f"  ❌ PROBLEM: All beliefs are too similar!")
else:
    print(f"  ✓ Beliefs are diverse")

# Check φ diversity
phi_std = phi[0].std(dim=0)
phi_mean_std = phi_std.mean().item()

print(f"  φ diversity:")
print(f"    Std per component: {phi_std.tolist()}")
print(f"    Mean std: {phi_mean_std:.4f}")

if phi_mean_std < 0.01:
    print(f"  ❌ PROBLEM: All gauge frames are too similar!")
else:
    print(f"  ✓ Gauge frames are diverse")

# =============================================================================
# STEP 4: Check KL divergence matrix
# =============================================================================
print("\n[CHECKING KL DIVERGENCES]")

with torch.no_grad():
    _, attn_info = model.forward_with_attention(token_ids)
    beta = attn_info['beta']  # (B, n_heads, N, N)
    kl_matrix = attn_info.get('kl_matrix')  # (B, N, N)

if kl_matrix is not None:
    kl_np = kl_matrix[0].cpu().numpy()

    # Exclude diagonal
    kl_safe = kl_np.copy()
    np.fill_diagonal(kl_safe, np.nan)

    kl_mean = np.nanmean(kl_safe)
    kl_std = np.nanstd(kl_safe)

    print(f"  KL divergence statistics:")
    print(f"    Mean: {kl_mean:.4f}")
    print(f"    Std:  {kl_std:.4f}")

    if kl_std < 0.5:
        print(f"  ❌ PROBLEM: KL divergences are too uniform!")
        print(f"     This causes uniform attention!")
    else:
        print(f"  ✓ KL divergences have good variation")

    # Show example row
    print(f"\n  Example KL row (position 8):")
    print(f"    {kl_np[8, :].round(3)}")

# =============================================================================
# STEP 5: Check attention patterns
# =============================================================================
print("\n[CHECKING ATTENTION PATTERNS]")

beta_np = beta[0].cpu().numpy()  # (n_heads, N, N)
n_heads, N, _ = beta_np.shape

print(f"  Number of heads: {n_heads}")

for head_idx in range(min(5, n_heads)):  # Show first 5 heads
    attn_head = beta_np[head_idx]

    # Exclude diagonal
    attn_safe = attn_head.copy()
    np.fill_diagonal(attn_safe, np.nan)

    row_std = np.nanstd(attn_safe, axis=1).mean()
    status = "✓SHARP" if row_std > 0.1 else "⚠MEDIUM" if row_std > 0.05 else "❌UNIFORM"

    print(f"  Head {head_idx}: row_std={row_std:.4f} [{status}]")

    # Show example row
    example_row = attn_head[8, :8]
    print(f"    Example (pos 8→0:7): {example_row.round(4)}")

if n_heads > 5:
    print(f"  ... ({n_heads - 5} more heads)")

# =============================================================================
# STEP 6: Test different kappa values
# =============================================================================
print("\n[TESTING TEMPERATURE SENSITIVITY]")

test_kappas = [0.1, 0.3, 0.5, 1.0, 2.0]
print(f"  Current kappa: {config.get('kappa_beta', 'unknown')}")
print(f"\n  Testing different kappa values:")

for kappa_test in test_kappas:
    # Change kappa temporarily
    for layer in model.layers:
        layer.attention.kappa = kappa_test

    with torch.no_grad():
        _, attn_test = model.forward_with_attention(token_ids)
        beta_test = attn_test['beta'][0, 0].cpu().numpy()

    # Compute row std
    beta_safe = beta_test.copy()
    np.fill_diagonal(beta_safe, np.nan)
    row_std = np.nanstd(beta_safe, axis=1).mean()

    status = "✓" if row_std > 0.1 else "⚠" if row_std > 0.05 else "❌"
    print(f"    κ={kappa_test:4.1f}: row_std={row_std:.4f} [{status}]")

# =============================================================================
# DIAGNOSIS
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

issues = []

if mu_dist_mean < 0.5:
    issues.append(("Similar beliefs (μ)", "All tokens have nearly identical embeddings"))

if phi_mean_std < 0.01:
    issues.append(("Similar gauge frames (φ)", "No positional variation in gauge frames"))

if kl_matrix is not None and kl_std < 0.5:
    issues.append(("Uniform KL divergences", "Root cause of uniform attention"))

if config.get('kappa_beta', 1.0) > 1.5:
    issues.append(("High temperature (κ)", "Softmax is too flat"))

if issues:
    print("\n❌ PROBLEMS FOUND:\n")
    for i, (problem, explanation) in enumerate(issues, 1):
        print(f"{i}. {problem}")
        print(f"   → {explanation}")

    print("\n📋 RECOMMENDED FIXES:\n")

    if mu_dist_mean < 0.5:
        print("  1. Check embedding initialization")
        print("     - Are different tokens getting different embeddings?")
        print("     - Is the embedding layer being updated during training?")

    if phi_mean_std < 0.01:
        print("  2. Check positional encoding")
        print("     - Is use_positional_embedding=True?")
        print("     - Is φ varying across positions?")

    if kl_matrix is not None and kl_std < 0.5:
        print("  3. KL divergences are uniform - check:")
        print("     - Are beliefs (μ) diverse? (see above)")
        print("     - Is transport Ω_ij working? (check φ)")
        print("     - Is the KL computation correct?")

    if config.get('kappa_beta', 1.0) > 1.5:
        print("  4. Lower kappa_beta to 0.5 or 0.3")
else:
    print("\n✓ No obvious problems detected.")
    print("  Model may just need more training.")

print("\n" + "="*80)
