#!/usr/bin/env python
"""
Diagnose why attention is uniform.

This script checks all possible causes:
1. Belief diversity (μ variation)
2. Gauge frame diversity (φ variation)
3. KL divergence variation
4. Temperature (kappa) effects
5. Model training state

Run: python diagnose_uniform_attention.py
"""

import torch
import numpy as np
from transformer.model import GaugeTransformerLM

def diagnose_uniform_attention():
    """Comprehensive diagnostic for uniform attention."""

    print("="*80)
    print("DIAGNOSING UNIFORM ATTENTION")
    print("="*80)

    # Create model
    config = {
        'vocab_size': 100,
        'embed_dim': 64,
        'n_layers': 1,
        'hidden_dim': 128,
        'max_seq_len': 128,
        'kappa_beta': 1.0,  # Check if this is too high!
        'mask_self_attention': True,
        'evolve_sigma': True,
        'irrep_spec': [('l0', 32, 1), ('l1', 8, 3), ('l2', 2, 4)],
    }

    print(f"\n[CONFIG]")
    print(f"  kappa_beta: {config['kappa_beta']} (lower = sharper attention)")
    print(f"  mask_self_attention: {config['mask_self_attention']}")
    print(f"  evolve_sigma: {config['evolve_sigma']}")

    model = GaugeTransformerLM(config)
    model.eval()

    # Test sequence
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    B, N = token_ids.shape

    print(f"\n[SEQUENCE] {token_ids.shape}")
    print(f"  Tokens: {token_ids[0].tolist()}")

    # =========================================================================
    # STEP 1: Check embedding diversity
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Checking Belief (μ) Diversity")
    print("="*80)

    mu, sigma, phi = model.token_embed(token_ids)
    mu_np = mu[0].detach().numpy()  # (N, K)

    # Check pairwise distances
    mu_dists = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(mu_np[i] - mu_np[j])
            mu_dists.append(dist)

    mu_dist_mean = np.mean(mu_dists)
    mu_dist_std = np.std(mu_dists)

    print(f"  μ pairwise distances:")
    print(f"    Mean: {mu_dist_mean:.4f}")
    print(f"    Std:  {mu_dist_std:.4f}")

    if mu_dist_mean < 0.5:
        print(f"  ❌ PROBLEM: Beliefs are too similar!")
        print(f"     All tokens have nearly identical μ → uniform KL → uniform attention")
        print(f"     Solution: Check embedding initialization, ensure different tokens get different embeddings")
    else:
        print(f"  ✓ Beliefs are diverse enough")

    # =========================================================================
    # STEP 2: Check gauge frame diversity
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Checking Gauge Frame (φ) Diversity")
    print("="*80)

    phi_np = phi[0].detach().numpy()  # (N, 3)

    # Check pairwise differences
    phi_diffs = []
    for i in range(N):
        for j in range(i+1, N):
            diff = np.linalg.norm(phi_np[i] - phi_np[j])
            phi_diffs.append(diff)

    phi_diff_mean = np.mean(phi_diffs)
    phi_diff_std = np.std(phi_diffs)

    print(f"  φ pairwise differences:")
    print(f"    Mean: {phi_diff_mean:.4f}")
    print(f"    Std:  {phi_diff_std:.4f}")

    if phi_diff_mean < 0.01:
        print(f"  ❌ PROBLEM: Gauge frames are too similar!")
        print(f"     All tokens have nearly identical φ → Ω_ij ≈ I → no transport → uniform attention")
        print(f"     Solution: Ensure positional encoding varies φ across positions")
    else:
        print(f"  ✓ Gauge frames are diverse enough")

    # =========================================================================
    # STEP 3: Compute and check KL divergence matrix
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Checking KL Divergence Variation")
    print("="*80)

    with torch.no_grad():
        _, attn_info = model.forward_with_attention(token_ids)
        beta = attn_info['beta']  # (B, n_heads, N, N)
        kl_matrix = attn_info.get('kl_matrix')  # (B, N, N)

    if kl_matrix is not None:
        kl_np = kl_matrix[0].detach().numpy()  # (N, N)

        # Exclude diagonal (KL(q_i||q_i)=0 always)
        kl_safe = kl_np.copy()
        np.fill_diagonal(kl_safe, np.nan)

        kl_row_stds = np.nanstd(kl_safe, axis=1)  # Std per row
        kl_mean = np.nanmean(kl_safe)
        kl_std = np.nanstd(kl_safe)

        print(f"  KL divergence statistics:")
        print(f"    Overall mean: {kl_mean:.4f}")
        print(f"    Overall std:  {kl_std:.4f}")
        print(f"    Row std (avg): {kl_row_stds.mean():.4f}")

        if kl_std < 0.5:
            print(f"  ❌ PROBLEM: KL divergences are too uniform!")
            print(f"     All KL(q_i||Ω_ij[q_j]) values are similar → softmax is flat → uniform attention")
            print(f"     Root causes:")
            print(f"       - Similar beliefs (μ_i ≈ μ_j for all i,j)")
            print(f"       - Similar transports (Ω_ij ≈ I for all i,j)")
            print(f"       - Model hasn't learned meaningful representations yet")
        else:
            print(f"  ✓ KL divergences vary enough")

        # Show example row
        print(f"\n  Example: KL divergences from position 8 to all others:")
        print(f"    {kl_np[8, :].round(3)}")

    # =========================================================================
    # STEP 4: Check attention patterns
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: Analyzing Attention Patterns")
    print("="*80)

    beta_np = beta[0].detach().numpy()  # (n_heads, N, N)
    n_heads = beta_np.shape[0]

    for head_idx in range(n_heads):
        attn_head = beta_np[head_idx]

        # Exclude diagonal
        attn_safe = attn_head.copy()
        np.fill_diagonal(attn_safe, np.nan)

        row_std = np.nanstd(attn_safe, axis=1).mean()
        status = "✓SHARP" if row_std > 0.1 else "⚠MEDIUM" if row_std > 0.05 else "❌UNIFORM"

        print(f"  Head {head_idx}: row_std={row_std:.4f} [{status}]")

        # Show example row
        example_row = attn_head[8, :8]  # Position 8 attending to positions 0-7
        print(f"    Example (pos 8→0:7): {example_row.round(4)}")

    # Averaged
    attn_avg = beta_np.mean(axis=0)
    attn_safe = attn_avg.copy()
    np.fill_diagonal(attn_safe, np.nan)
    row_std = np.nanstd(attn_safe, axis=1).mean()
    status = "✓" if row_std > 0.1 else "⚠" if row_std > 0.05 else "❌"
    print(f"\n  AVERAGED: row_std={row_std:.4f} [{status}]")
    print(f"    (Averaging can make sharp patterns look uniform)")

    # =========================================================================
    # STEP 5: Temperature (kappa) sensitivity test
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: Temperature (κ) Sensitivity Test")
    print("="*80)

    print(f"  Current kappa: {config['kappa_beta']}")

    # Test different kappa values
    test_kappas = [0.3, 0.5, 1.0, 2.0, 5.0]
    print(f"\n  Testing attention sharpness with different κ values:")

    for kappa_test in test_kappas:
        # Temporarily change kappa
        model.layers[0].attention.kappa = kappa_test

        with torch.no_grad():
            _, attn_info_test = model.forward_with_attention(token_ids)
            beta_test = attn_info_test['beta'][0, 0].detach().numpy()  # First head

        # Compute row std
        beta_safe = beta_test.copy()
        np.fill_diagonal(beta_safe, np.nan)
        row_std_test = np.nanstd(beta_safe, axis=1).mean()

        status = "✓SHARP" if row_std_test > 0.1 else "⚠MEDIUM" if row_std_test > 0.05 else "❌UNIFORM"
        print(f"    κ={kappa_test:.1f}: row_std={row_std_test:.4f} [{status}]")

    print(f"\n  Recommendation:")
    if config['kappa_beta'] > 1.0:
        print(f"    Try lowering kappa to 0.5 or 0.3 for sharper attention")
    else:
        print(f"    Kappa is reasonable. Problem is likely belief/KL uniformity.")

    # =========================================================================
    # FINAL DIAGNOSIS
    # =========================================================================
    print(f"\n{'='*80}")
    print("FINAL DIAGNOSIS")
    print("="*80)

    issues = []

    if mu_dist_mean < 0.5:
        issues.append("❌ Beliefs (μ) are too similar")

    if phi_diff_mean < 0.01:
        issues.append("❌ Gauge frames (φ) are too similar")

    if kl_matrix is not None and kl_std < 0.5:
        issues.append("❌ KL divergences are too uniform")

    if config['kappa_beta'] > 1.0:
        issues.append("⚠️ Temperature (κ) might be too high")

    if not issues:
        print("  ✓ No obvious problems detected!")
        print("  Model may just need more training to learn sharp attention patterns.")
    else:
        print("  Issues found:")
        for issue in issues:
            print(f"    {issue}")

        print(f"\n  Recommended fixes:")
        if mu_dist_mean < 0.5:
            print(f"    1. Check token embedding initialization")
            print(f"       Ensure different tokens get different embeddings")
        if phi_diff_mean < 0.01:
            print(f"    2. Check positional encoding for φ")
            print(f"       Ensure φ varies across positions")
        if kl_matrix is not None and kl_std < 0.5:
            print(f"    3. Train the model longer")
            print(f"       Random init often gives uniform KL → train to differentiate")
        if config['kappa_beta'] > 1.0:
            print(f"    4. Lower kappa_beta to 0.3-0.5")

    print("="*80)


if __name__ == '__main__':
    diagnose_uniform_attention()
