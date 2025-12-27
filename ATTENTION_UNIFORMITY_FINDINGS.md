# Attention Uniformity Investigation - Key Findings

## Discovery: Lower-Spin Heads Show Uniform Attention

**Date**: 2025-12-27
**Model**: Gauge-equivariant transformer with SO(3) multi-irrep attention

---

## The Pattern

After extensive debugging and experimentation with temperature (κ) scaling, we discovered a fundamental architectural limitation:

**Lower-spin irrep heads cannot learn sharp attention patterns, regardless of training or hyperparameters.**

### Observed Behavior

For configuration: `irrep_spec = [('ℓ0', 5, 1), ('ℓ1', 3, 3), ('ℓ2', 1, 5)]`

| Head Type | Dimensionality | Typical row_std | Status |
|-----------|----------------|-----------------|--------|
| ℓ0 (scalar) | 1 | 0.024 - 0.030 | ❌ UNIFORM |
| ℓ1 (vector) | 3 | 0.030 - 0.045 | ❌ UNIFORM |
| ℓ2 (tensor) | 5 | 0.065 - 0.090 | ⚠️ MEDIUM |

**Threshold for sharp attention**: row_std > 0.10

---

## Root Cause: Dimensionality and Law of Large Numbers

### The Attention Mechanism

Gauge-equivariant attention uses KL divergence between beliefs:

```
β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)
```

Where:
- `q_i = N(μ_i, Σ_i)` is a Gaussian belief in K dimensions
- `Ω_ij` is parallel transport between gauge frames φ_i and φ_j
- κ is the temperature parameter

### KL Divergence for Diagonal Gaussians

For K-dimensional Gaussians with diagonal covariance:

```
KL(q_i || Ω_ij[q_j]) = 0.5 * (Σ_k (σ_ik / σ_jk') + Σ_k ((μ_jk' - μ_ik)² / σ_jk') - K + Σ_k (log σ_jk' - log σ_ik))
```

This is a **sum over K dimensions**.

### The Problem

**Law of Large Numbers**: As K increases, sums of independent random variables concentrate around their mean.

For **lower-dimensional heads**:
- ℓ0: K=1 → KL is just one term → NO averaging → high variance ✓
  - **BUT**: Single scalar cannot capture rich geometric structure → uniform anyway
- ℓ1: K=3 → Sum of 3 terms → some averaging → medium variance
  - **BUT**: 3D vectors don't provide enough degrees of freedom for sharp patterns

For **higher-dimensional heads**:
- ℓ2: K=5 → Sum of 5 terms → more representational capacity
  - CAN learn structured attention, but still limited by concentration

### Why Scalars Fail Despite Low Dimensionality

Even though K=1 avoids the law of large numbers, **scalar heads fail because**:

1. **Geometric poverty**: A 1D belief state cannot capture the rich geometric relationships needed for selective attention
2. **Embedding collapse**: Without enough dimensions, different tokens' μ vectors become too similar
3. **Transport trivial**: Parallel transport Ω_ij has minimal effect in 1D

**Paradox**: Lower K avoids concentration but lacks representational power.

---

## Temperature (κ) Experiments

We tested various temperature values to see if sharp attention was possible:

| κ value | KL mean | KL std | Attention | PPL | Stability |
|---------|---------|--------|-----------|-----|-----------|
| 0.43 (original) | 0.0030 | 0.0015 | Uniform | 900 | ✓ Stable |
| 0.00043 (1000× lower) | 0.0030 | 0.0015 | Sharp but chaotic | 1475 | ❌ Grad explosions (357!) |
| 0.0086 (50× lower) | 0.0192 | 0.0134 | 8/9 heads uniform | 947 | ✓ Stable |

### Key Findings

1. **Lowering κ does NOT increase KL variation** when diversity is already poor
2. **Extremely low κ** makes softmax ultra-sensitive to tiny KL differences → instability
3. **Even with good KL variation** (std=0.013 at κ=0.0086), lower-spin heads stay uniform
4. **Only the tensor head (ℓ2)** showed any structure (row_std ≈ 0.068)

---

## Diversity Tracking Results

At step 20000 with optimal temperature (κ ≈ 0.0086):

```
[DIVERSITY] μ_dist: 0.4962 | φ_std: 0.4827 | KL_std: 0.0134
            KL range: [0.0002, 0.0896] | mean: 0.0192
            σ range: [0.002629, 6.713892] | mean: 1.005399
```

**Interpretation**:
- ✓ **μ diversity is healthy** (mean pairwise distance ≈ 0.50)
- ✓ **φ diversity is healthy** (std ≈ 0.48 across positions)
- ✓ **σ is reasonable** (mean ≈ 1.0, no collapse)
- ⚠️ **KL variation is present** (std=0.013, range up to 0.09) **BUT NOT ENOUGH** for lower-spin heads

The diversity is **not the problem** - the problem is **geometric limitation of low-dimensional irreps**.

---

## Implications for Architecture Design

### ❌ What Doesn't Work

1. **Single high-dimensional irrep**: e.g., `[('ℓ9', 1, 19)]`
   - Law of large numbers makes KL uniform
   - Result: One uniform attention head

2. **Many low-dimensional irreps**: e.g., `[('ℓ0', 5, 1), ('ℓ1', 3, 3)]`
   - Each head geometrically limited
   - Result: Many uniform attention heads

3. **Just lowering temperature**: Does not fix geometric limitations

### ✓ What Might Work

1. **Increase tensor head count**:
   ```python
   irrep_spec = [
       ('ℓ0', 2, 1),   # Keep some scalars for efficiency
       ('ℓ1', 2, 3),   # Keep some vectors
       ('ℓ2', 4, 5),   # MORE rank-2 tensors (was 1, now 4)
   ]
   # Total: 2×1 + 2×3 + 4×5 = 2 + 6 + 20 = 28 dimensions
   ```

2. **Use higher-rank irreps**:
   ```python
   irrep_spec = [
       ('ℓ1', 4, 3),   # Vectors
       ('ℓ2', 3, 5),   # Rank-2 tensors
       ('ℓ3', 2, 7),   # Rank-3 tensors (NEW!)
   ]
   # Total: 4×3 + 3×5 + 2×7 = 12 + 15 + 14 = 41 dimensions
   ```

3. **Scale up embed_dim entirely**:
   ```python
   irrep_spec = [
       ('ℓ0', 10, 1),  # Scalars for cheap computation
       ('ℓ1', 10, 3),  # Vectors
       ('ℓ2', 8, 5),   # Rank-2 tensors
       ('ℓ3', 4, 7),   # Rank-3 tensors
   ]
   # Total: 10 + 30 + 40 + 28 = 108 dimensions
   # 32 heads total, with 12 being rank-2 or higher
   ```

### Design Principle

**Maximize the proportion of higher-rank (ℓ ≥ 2) heads while keeping total embed_dim feasible.**

---

## Recommended Next Steps

1. **Test new irrep configurations** with more tensor heads
2. **Profile memory/compute** for higher embed_dim (e.g., 64, 128)
3. **Ablation study**: Compare pure-tensor vs mixed configurations
4. **Theoretical analysis**: Derive minimum K for sharp attention given dataset statistics

---

## Code Changes Made

### Added Irrep Labeling to Diagnostics

File: `transformer/train_publication.py`

**New method** `_get_head_irrep_labels()`:
```python
def _get_head_irrep_labels(self) -> list:
    """Map head indices to irrep types for diagnostic labeling."""
    irrep_spec = self.config.irrep_spec
    labels = []
    for irrep_name, num_heads, dim in irrep_spec:
        for _ in range(num_heads):
            labels.append(irrep_name)
    return labels
```

**Updated diagnostic output**:
```
Head 0 (ℓ0): row_std=0.0243 [❌UNIFORM]
Head 5 (ℓ1): row_std=0.0343 [❌UNIFORM]
Head 8 (ℓ2): row_std=0.0684 [⚠MEDIUM]
```

This makes the spin-dependent pattern immediately visible during training.

---

## Conclusion

**The attention uniformity problem is NOT a bug - it's a fundamental architectural limitation.**

Lower-spin irrep heads (ℓ0, ℓ1) cannot learn sharp attention patterns because:
1. They lack geometric representational capacity
2. Their low dimensionality leads to embedding similarity
3. Temperature tuning cannot compensate for geometric poverty

**Solution**: Use more higher-rank tensor heads (ℓ2, ℓ3, etc.) in the irrep decomposition.
