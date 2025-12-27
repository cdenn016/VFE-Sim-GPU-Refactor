"""
Suggested config fix for uniform attention issue.

PROBLEM: Single spin-9 irrep (dim=19) causes uniform KL divergences
SOLUTION: Use multiple lower-dimensional irreps instead

This gives better attention patterns while keeping total embed_dim small.
"""

# =============================================================================
# OPTION 1: Mixed low-dimensional irreps (RECOMMENDED)
# =============================================================================
# Total: 7×1 + 4×3 = 7 + 12 = 19 dimensions
recommended_config_1 = {
    'embed_dim': 19,
    'irrep_spec': [
        ('ℓ0', 7, 1),   # 7 scalar heads
        ('ℓ1', 4, 3),   # 4 vector heads
    ],
    # Now you have 11 attention heads (7+4) instead of 1!
    # Lower dimensionality → less uniform KL → sharper attention
}

# =============================================================================
# OPTION 2: More heads with scalars + vectors
# =============================================================================
# Total: 9×1 + 2×3 + 1×4 = 9 + 6 + 4 = 19 dimensions
recommended_config_2 = {
    'embed_dim': 19,
    'irrep_spec': [
        ('ℓ0', 9, 1),   # 9 scalar heads
        ('ℓ1', 2, 3),   # 2 vector heads
        ('ℓ2', 1, 4),   # 1 rank-2 tensor head (dim = 2ℓ+1 = 5, but using 4)
    ],
    # Error: ℓ2 should have dim=5, not 4. Let me fix:
}

# Corrected Option 2:
# Total: 8×1 + 2×3 + 1×5 = 8 + 6 + 5 = 19 dimensions
recommended_config_2_fixed = {
    'embed_dim': 19,
    'irrep_spec': [
        ('ℓ0', 8, 1),   # 8 scalar heads
        ('ℓ1', 2, 3),   # 2 vector heads
        ('ℓ2', 1, 5),   # 1 rank-2 tensor head
    ],
    # Total: 11 heads (8+2+1)
}

# =============================================================================
# OPTION 3: Scale up embed_dim for better representational capacity
# =============================================================================
# If memory allows, use larger embed_dim with mixed irreps
# Total: 32×1 + 8×3 + 4×5 = 32 + 24 + 20 = 76 dimensions
recommended_config_3 = {
    'embed_dim': 76,
    'irrep_spec': [
        ('ℓ0', 32, 1),   # 32 scalar heads
        ('ℓ1', 8, 3),    # 8 vector heads
        ('ℓ2', 4, 5),    # 4 rank-2 tensor heads
    ],
    # Total: 44 heads!
    # Better representational capacity
    # Still computationally feasible with diagonal_covariance=True
}

# =============================================================================
# WHY THIS FIXES UNIFORM ATTENTION
# =============================================================================
print("""
Why high-dimensional irreps cause uniform attention:
====================================================

PROBLEM (your current config):
  irrep_spec = [('ℓ9', 1, 19)]
  - Single 19-dimensional attention mechanism
  - KL divergence: sum over 19 dimensions
  - Law of large numbers → sums become similar
  - Result: KL(q_i||q_j) ≈ constant for all pairs → uniform softmax

SOLUTION (recommended configs):
  irrep_spec = [('ℓ0', 7, 1), ('ℓ1', 4, 3)]
  - 11 separate attention heads (7 scalar + 4 vector)
  - Each head has lower dimensionality (1 or 3)
  - Lower dim → more variation in KL divergences
  - Different heads can learn different patterns
  - Result: Sharp, diverse attention!

ANALOGY:
  Current:  One 19D sensor averaging everything → sees "average" everywhere
  Fixed:    11 lower-D sensors, each specialized → sees distinct patterns

EXPECTED IMPROVEMENT:
  Before: row_std = 0.0217 (all heads uniform)
  After:  row_std = 0.15-0.30 (heads learn sharp patterns)
""")

# =============================================================================
# OTHER RECOMMENDED CHANGES
# =============================================================================
other_fixes = {
    # Lower μ initialization (currently 7.0 is very high for K=19)
    'mu_init_std': 1.0,  # Or 1/√K ≈ 0.23 for K=19

    # Optionally normalize embeddings to unit sphere
    'mu_normalize': True,  # Projects μ to unit sphere

    # Keep other settings
    'diagonal_covariance': True,  # Good for memory
    'kappa_beta_auto_scale': True,  # Good for stability
}

print("\n" + "="*80)
print("RECOMMENDED ACTION")
print("="*80)
print("""
1. Change irrep_spec to Option 1 (recommended):
   'irrep_spec': [('ℓ0', 7, 1), ('ℓ1', 4, 3)]

2. Lower mu_init_std:
   'mu_init_std': 1.0  (from 7.0)

3. Retrain and check attention diagnostics:
   - Should see row_std > 0.10 in many heads
   - Different heads will have different patterns
   - PPL should improve (currently 144 → target <100)

4. If still uniform, try Option 3 (larger embed_dim=76)
""")
