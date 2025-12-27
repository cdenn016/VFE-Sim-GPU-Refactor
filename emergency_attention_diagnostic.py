#!/usr/bin/env python
"""
EMERGENCY DIAGNOSTIC: Why is attention perfectly uniform after full training?

Run this on your trained model to find the root cause.
"""

import torch
import numpy as np

# Load your trained model
print("="*80)
print("EMERGENCY DIAGNOSTIC: UNIFORM ATTENTION AFTER TRAINING")
print("="*80)

# YOU NEED TO LOAD YOUR ACTUAL TRAINED MODEL HERE
# Example:
# from transformer.model import GaugeTransformerLM
# model = GaugeTransformerLM.from_checkpoint('path/to/checkpoint')

# For now, let's check what the config should be
print("\nPlease provide your model checkpoint path and run this diagnostic.")
print("This script needs your TRAINED model, not a fresh one.")
print("\nWhat to check:\n")

print("1. LOAD YOUR TRAINED MODEL:")
print("   from transformer.model import GaugeTransformerLM")
print("   model = torch.load('checkpoints/your_model.pt')")
print("   model.eval()")

print("\n2. GET A BATCH FROM VALIDATION DATA:")
print("   from transformer.data import create_dataloaders")
print("   _, val_loader, _ = create_dataloaders(...)")
print("   input_ids, _ = next(iter(val_loader))")

print("\n3. FORWARD PASS WITH ATTENTION:")
print("   _, attn_info = model.forward_with_attention(input_ids)")
print("   beta = attn_info['beta']  # (B, n_heads, N, N)")
print("   kl_matrix = attn_info['kl_matrix']  # (B, N, N)")

print("\n4. CHECK EMBEDDINGS:")
print("   mu, sigma, phi = model.token_embed(input_ids)")
print("   print('μ diversity:', torch.cdist(mu[0], mu[0]).std().item())")
print("   print('φ diversity:', phi[0].std(dim=0))")

print("\n5. CHECK KL MATRIX:")
print("   kl_np = kl_matrix[0].cpu().numpy()")
print("   np.fill_diagonal(kl_np, np.nan)")
print("   print('KL std:', np.nanstd(kl_np))")
print("   print('KL matrix:\\n', kl_np)")

print("\n6. CHECK BELIEFS AT DIFFERENT POSITIONS:")
print("   for i in [0, 5, 10]:")
print("       print(f'μ[{i}] norm:', torch.norm(mu[0, i]).item())")
print("       print(f'φ[{i}]:', phi[0, i].tolist())")

print("\n" + "="*80)
print("EXPECTED CAUSES OF UNIFORM ATTENTION:")
print("="*80)

causes = [
    ("Similar embeddings", "All μ_i are nearly identical → KL divergences all similar → uniform softmax"),
    ("Identical gauge frames", "All φ_i are the same → Ω_ij = I for all pairs → no transport effect"),
    ("High temperature", "kappa_beta too large → softmax flattens everything"),
    ("Broken KL computation", "KL divergence function returning constant values"),
    ("No positional encoding", "Tokens at different positions get identical embeddings"),
    ("Gradient issues", "Embeddings not updating during training"),
]

for i, (cause, explanation) in enumerate(causes, 1):
    print(f"\n{i}. {cause}:")
    print(f"   {explanation}")

print("\n" + "="*80)
print("IMMEDIATE ACTION ITEMS:")
print("="*80)

print("\n1. CHECK YOUR CONFIG:")
print("   - What is kappa_beta? (should be 0.3-1.0)")
print("   - What is use_positional_embedding? (should be True)")
print("   - What is evolve_sigma? (should be True)")
print("   - What is mask_self_attention? (should be True)")

print("\n2. CHECK TRAINING LOGS:")
print("   - Is loss decreasing? (should go from ~10 to <5)")
print("   - Is perplexity decreasing? (should be <100 for good model)")
print("   - Are embeddings being updated? (check gradient norms)")

print("\n3. PRINT ACTUAL VALUES:")
print("   Run the code above on your trained model and paste results here.")

print("\n" + "="*80)
