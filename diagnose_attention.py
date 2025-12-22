#!/usr/bin/env python
"""
Diagnostic script to identify sources of positional attention in VFE-dynamic.

Run with: python diagnose_attention.py

This tests whether attention is content-based or position-based by:
1. Checking config options are properly applied
2. Comparing attention patterns for same vs different content at different positions
3. Identifying any remaining positional bias sources
"""

import torch
import numpy as np
torch.set_grad_enabled(False)

from transformer.model import GaugeTransformerLM  # Correct class name!


def create_diagnostic_model():
    """Create model with all positional sources disabled."""
    config = {
        'vocab_size': 1000,
        'embed_dim': 64,
        'n_layers': 2,
        'hidden_dim': 128,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'irrep_spec': [('ℓ0', 16, 1), ('ℓ1', 8, 3), ('ℓ2', 4, 5)],  # Total: 16+24+20=60 -> pad to 64

        # Disable ALL positional sources
        'pos_encoding_mode': 'none',          # No gauge positional encoding
        'use_positional_embedding': False,     # No μ positional embedding
        'use_identity_transport': True,        # Ω_ij = I (no gauge transport)
        'alibi_slope': None,                   # No ALiBi

        # Other settings
        'evolve_sigma': False,
        'evolve_phi': False,
        'tie_embeddings': True,
    }

    print("=" * 70)
    print("DIAGNOSTIC MODEL CONFIGURATION")
    print("=" * 70)
    print(f"  pos_encoding_mode:        {config['pos_encoding_mode']}")
    print(f"  use_positional_embedding: {config['use_positional_embedding']}")
    print(f"  use_identity_transport:   {config['use_identity_transport']}")
    print(f"  alibi_slope:              {config['alibi_slope']}")
    print("=" * 70)

    model = GaugeTransformerLM(config)
    model.eval()
    return model, config


def verify_config_applied(model):
    """Verify that config options are actually applied in the model."""
    print("\n[1] VERIFYING CONFIG OPTIONS ARE APPLIED")
    print("-" * 50)

    # Check positional encoding mode
    pos_mode = model.pos_encoding.mode
    print(f"  pos_encoding.mode = '{pos_mode}'",
          "✓" if pos_mode == 'none' else "✗ WRONG!")

    # Check if pos_phi is learnable (should be buffer, not parameter)
    pos_phi_is_buffer = not isinstance(model.pos_encoding.pos_phi, torch.nn.Parameter)
    print(f"  pos_phi is buffer (not learnable) = {pos_phi_is_buffer}",
          "✓" if pos_phi_is_buffer else "✗ STILL LEARNABLE!")

    # Check pos_phi values (should be all zeros for mode='none')
    pos_phi_is_zero = torch.allclose(model.pos_encoding.pos_phi,
                                      torch.zeros_like(model.pos_encoding.pos_phi))
    print(f"  pos_phi is all zeros = {pos_phi_is_zero}",
          "✓" if pos_phi_is_zero else "✗ NON-ZERO!")

    # Check positional embedding in token embeddings
    use_pos_embed = model.token_embed.use_positional_embedding
    print(f"  token_embed.use_positional_embedding = {use_pos_embed}",
          "✓" if not use_pos_embed else "✗ SHOULD BE FALSE!")

    # Check identity transport in attention layers
    first_block = model.transformer.blocks[0]
    use_identity = first_block.attention.use_identity_transport
    print(f"  attention.use_identity_transport = {use_identity}",
          "✓" if use_identity else "✗ SHOULD BE TRUE!")

    # Check ALiBi
    alibi = first_block.attention.alibi_slope
    print(f"  attention.alibi_slope = {alibi}",
          "✓" if alibi is None else "✗ SHOULD BE None!")

    return all([
        pos_mode == 'none',
        pos_phi_is_buffer,
        pos_phi_is_zero,
        not use_pos_embed,
        use_identity,
        alibi is None
    ])


def test_position_invariance(model):
    """Test if attention is truly position-invariant."""
    print("\n[2] TESTING POSITION INVARIANCE")
    print("-" * 50)

    # Create test sequences:
    # Sequence 1: [A, B, C, D, E, F, G, H]  (tokens 1-8)
    # Sequence 2: [A, B, C, D, E, F, G, H]  (same tokens, same positions)
    # If position-invariant, attention patterns should be identical

    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # (1, 8)

    # Get attention weights from forward_with_attention
    logits, attn_info = model.forward_with_attention(token_ids)
    beta = attn_info['beta']  # (B, n_heads, N, N)

    print(f"  Attention shape: {beta.shape}")

    # Average across heads
    beta_avg = beta.mean(dim=1)[0]  # (N, N)

    print(f"\n  Attention pattern (averaged across heads):")
    print(f"  Rows = query positions, Cols = key positions")
    print(f"  (Only lower triangle valid due to causal mask)\n")

    # Print attention matrix
    np.set_printoptions(precision=3, suppress=True)
    print(beta_avg.numpy())

    return beta_avg


def test_content_vs_position(model):
    """Compare attention for same content at different positions."""
    print("\n[3] CONTENT vs POSITION TEST")
    print("-" * 50)
    print("  If position-invariant: same token pairs should have same attention")
    print("  regardless of their absolute positions.\n")

    # Sequence with repeated tokens
    # [A, B, A, B, A, B, A, B]
    token_ids = torch.tensor([[10, 20, 10, 20, 10, 20, 10, 20]])

    logits, attn_info = model.forward_with_attention(token_ids)
    beta_avg = attn_info['beta'].mean(dim=1)[0]  # (N, N)

    print("  Sequence: [A, B, A, B, A, B, A, B]  (A=10, B=20)")
    print(f"\n  Attention pattern:\n")
    print(beta_avg.numpy())

    # Check: For position-invariant attention,
    # - All A tokens attending to earlier A tokens should have similar weights
    # - All A tokens attending to earlier B tokens should have similar weights

    # A positions: 0, 2, 4, 6
    # B positions: 1, 3, 5, 7

    # Check A→A attention (content match)
    a_to_a = [beta_avg[2, 0].item(), beta_avg[4, 0].item(), beta_avg[4, 2].item(),
              beta_avg[6, 0].item(), beta_avg[6, 2].item(), beta_avg[6, 4].item()]

    # Check A→B attention (content mismatch)
    a_to_b = [beta_avg[2, 1].item(), beta_avg[4, 1].item(), beta_avg[4, 3].item(),
              beta_avg[6, 1].item(), beta_avg[6, 3].item(), beta_avg[6, 5].item()]

    print(f"\n  A→A attention weights (should be similar if position-invariant):")
    print(f"    {[f'{x:.3f}' for x in a_to_a]}")
    print(f"    Mean: {np.mean(a_to_a):.3f}, Std: {np.std(a_to_a):.3f}")

    print(f"\n  A→B attention weights (should be similar if position-invariant):")
    print(f"    {[f'{x:.3f}' for x in a_to_b]}")
    print(f"    Mean: {np.mean(a_to_b):.3f}, Std: {np.std(a_to_b):.3f}")

    # If std is low, attention is position-invariant
    position_invariant = np.std(a_to_a) < 0.1 and np.std(a_to_b) < 0.1

    if position_invariant:
        print(f"\n  ✓ Attention appears POSITION-INVARIANT (low variance)")
    else:
        print(f"\n  ✗ Attention appears POSITION-DEPENDENT (high variance)")
        print(f"    Same-content pairs at different positions have different attention!")

    return position_invariant


def test_raw_embeddings(model):
    """Check if embeddings themselves are position-dependent."""
    print("\n[4] RAW EMBEDDING TEST")
    print("-" * 50)

    # Get embeddings for same token at different "positions"
    # (Note: with use_positional_embedding=False, positions shouldn't matter)

    token_ids = torch.tensor([[42, 42, 42, 42]])  # Same token 4 times

    mu, sigma, phi = model.token_embed(token_ids)

    # After positional encoding composition
    phi_after = model.pos_encoding.compose(phi, 4, device=mu.device)

    print(f"  Token 42 repeated 4 times:")
    print(f"  μ embeddings (should be identical for same token):")
    for i in range(4):
        print(f"    pos {i}: μ[:5] = {mu[0, i, :5].numpy()}")

    mu_identical = torch.allclose(mu[0, 0], mu[0, 1]) and torch.allclose(mu[0, 0], mu[0, 2])
    print(f"\n  μ identical across positions: {mu_identical}",
          "✓" if mu_identical else "✗ DIFFERENT!")

    print(f"\n  φ embeddings after pos_encoding.compose:")
    for i in range(4):
        print(f"    pos {i}: φ = {phi_after[0, i].numpy()}")

    phi_identical = torch.allclose(phi_after[0, 0], phi_after[0, 1])
    print(f"\n  φ identical across positions: {phi_identical}",
          "✓" if phi_identical else "✗ DIFFERENT (mode='none' not working!)")


def main():
    print("\n" + "=" * 70)
    print("VFE-DYNAMIC POSITIONAL ATTENTION DIAGNOSTIC")
    print("=" * 70)

    # Create model
    model, config = create_diagnostic_model()

    # Run diagnostics
    config_ok = verify_config_applied(model)

    if not config_ok:
        print("\n⚠️  CONFIG NOT PROPERLY APPLIED - check model initialization!")

    test_raw_embeddings(model)
    test_position_invariance(model)
    is_invariant = test_content_vs_position(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if config_ok and is_invariant:
        print("  ✓ All positional sources disabled and attention is position-invariant!")
        print("\n  If you're still seeing triangular patterns during training,")
        print("  this is likely due to the CAUSAL MASK, not positional encoding.")
        print("  The causal mask creates lower-triangular attention by design.")
        print("\n  Later tokens have more context to attend to, which can create")
        print("  patterns that LOOK positional but are actually structural.")
    else:
        print("  ✗ Positional bias detected!")
        print("\n  Possible causes:")
        print("  1. Config options not taking effect (check wiring)")
        print("  2. Other sources of positional bias in the code")
        print("  3. Learned embeddings correlating with position")

    print("=" * 70)


if __name__ == '__main__':
    main()
