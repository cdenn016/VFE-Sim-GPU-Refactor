# Attention Pattern Validation Guide

## Purpose

This script helps you determine if attention patterns are:
- **Mathematically correct** (normalization, causality, KL properties)
- **Interpretable** (positional bias, token patterns)
- **Statistically meaningful** (vs uniform baseline)
- **Actually helping** (ablation study)

**Critical insight**: If your model generates complex words like "valkyrie" and "demonstrate" correctly, but attention *looks* uniform (row_std ≈ 0.024), this tool tells you whether:
1. Those subtle differences actually matter
2. The FFN is doing most of the work
3. You're measuring the wrong thing

## Usage

### Basic Usage

```bash
python validate_attention.py \
    --checkpoint checkpoints_publication/ffn_hamiltonian/best_model.pt \
    --seq_len 128
```

### With Custom Config

```bash
python validate_attention.py \
    --checkpoint checkpoints_publication/ffn_hamiltonian/best_model.pt \
    --config transformer/config.py \
    --seq_len 128
```

### Output Example

```
======================================================================
ATTENTION VALIDATION REPORT
======================================================================

[1/9] Sanity Checks (Correctness)
----------------------------------------------------------------------
✓ All rows sum to 1.0 (max error: 2.38e-07)
ℹ Non-causal attention detected (max future attn: 0.0078)
✓ KL ≥ 0 (min: -1.23e-08), diag ≈ 0 (mean: 0.0003)

[2/9] Positional Patterns
----------------------------------------------------------------------
⚠️  No local bias (nearby: 0.0082, far: 0.0079, ratio: 1.04x)

[3/9] Token-Level Patterns
----------------------------------------------------------------------
✓ Repeated tokens attract (same: 0.0085, diff: 0.0078, ratio: 1.09x)
ℹ Special token attention: 0.0090 vs regular: 0.0078 (ratio: 1.15x)

[4/9] Statistical Tests
----------------------------------------------------------------------
❌ Nearly uniform (mean dev: 0.0243, max: 0.0456)
❌ Nearly uniform (H/H_max: 0.983)

[5/9] Per-Head Analysis
----------------------------------------------------------------------
  Head  0 (ℓ0): row_std=0.0243 [❌UNIFORM]
  Head  1 (ℓ0): row_std=0.0251 [❌UNIFORM]
  Head  2 (ℓ0): row_std=0.0239 [❌UNIFORM]
  Head  3 (ℓ0): row_std=0.0247 [❌UNIFORM]
  Head  4 (ℓ0): row_std=0.0255 [❌UNIFORM]
  Head  5 (ℓ1): row_std=0.0343 [❌UNIFORM]
  Head  6 (ℓ1): row_std=0.0337 [❌UNIFORM]
  Head  7 (ℓ1): row_std=0.0351 [❌UNIFORM]
  Head  8 (ℓ2): row_std=0.0684 [⚠MEDIUM]

  Summary: 0/9 sharp (0%), 1/9 medium (11%), 8/9 uniform (89%)

[6/9] Query-Side Variation (Do different tokens attend differently?)
----------------------------------------------------------------------
  Average query variation (L2 between rows): 0.0523
  Max query variation: 0.0712

✓ Different queries attend differently (L2=0.0523)

  Positional pattern analysis:
    Adjacent queries (Δpos=1): L2=0.0312
    Distant queries (Δpos=10): L2=0.0498
    → Positional structure detected (distant > adjacent)

[7/9] Diversity Correlation
----------------------------------------------------------------------
  Belief diversity (μ_dist): 0.4962
  Attention diversity (row_std): 0.0345

❌ High belief diversity BUT uniform attention (architectural issue!)

[8/9] Gradient Flow Check
----------------------------------------------------------------------
  Found 12 attention parameter groups (45,056 params total)
✓ Attention parameters found

[9/9] Ablation Study (Does Attention Help?)
----------------------------------------------------------------------
  Learned attention: loss=6.8012, PPL=898.45
  ⚠️  Ablation test skipped (model API incompatible)

======================================================================
SUMMARY
======================================================================

✓ Passed: 8/15 tests
❌ Critical issues: 3
   - vs Uniform: ❌ Nearly uniform (mean dev: 0.0243)
   - Entropy: ❌ Nearly uniform (H/H_max: 0.983)
   - Diversity Correlation: ❌ High belief diversity BUT uniform attention

----------------------------------------------------------------------
CONCLUSION:
----------------------------------------------------------------------
⚠️  Attention is correct but may not be learning meaningful patterns.
   This could be due to:
   - Architectural limitations (low-dimensional heads)
   - Insufficient training
   - Hyperparameter issues (temperature, learning rate)
======================================================================
```

## What The Tests Mean

### 1. Sanity Checks
- **Normalization**: Each attention row must sum to 1.0
- **Causality**: No attending to future positions (for autoregressive)
- **KL Properties**: KL(q||q) ≈ 0, all KL ≥ 0

### 2. Positional Patterns
- Checks if nearby tokens get more attention than distant ones
- Common in language models (local context matters)

### 3. Token-Level Patterns
- **Repetition**: Do repeated tokens attend to each other?
- **Special tokens**: Do BOS/EOS/punctuation receive more attention?

### 4. Statistical Tests
- **vs Uniform**: Is attention significantly different from 1/N everywhere?
- **Entropy**: H=0 (sharp), H=1 (uniform)

### 5. Per-Head Analysis
- Shows which heads learn structure
- **Key finding**: Lower-spin irrep heads (ℓ0, ℓ1) tend to be uniform

### 6. Query-Side Variation ⭐ **NEW**
- **row_std** measures: "does each query spread uniformly?" (KEY-side)
- **This test** measures: "do different queries attend differently?" (QUERY-side)
- **Critical insight**: Model can work with uniform KEY distribution if QUERY variation is high!

### 7. Diversity Correlation
- If beliefs are diverse (μ_dist > 0.3) but attention is uniform → architectural problem
- If beliefs are similar → uniform attention is expected

### 8. Gradient Flow
- Checks if attention parameters exist and could receive gradients

### 9. Ablation Study
- **The ultimate test**: Does learned attention beat uniform/local baselines?
- Requires model modification (see below)

## Interpreting Your Results

### Scenario 1: Model works, attention looks uniform
**Symptoms**:
- PPL is good (< 1000)
- Generates complex words correctly
- row_std < 0.05 (appears uniform)
- Query variation > 0.05 ✓

**Interpretation**:
- Attention IS working, just subtly!
- The variation is in WHICH tokens attend (query-side), not HOW MUCH to each key
- This is actually fine - small differences × softmax = meaningful signal

**Action**: Nothing! Model is working as intended.

### Scenario 2: Model works, attention truly uniform
**Symptoms**:
- PPL is good
- row_std < 0.03
- Query variation < 0.02 ❌

**Interpretation**:
- FFN/belief propagation doing all the work
- Attention is just uniform mixing

**Action**: Try ablation study (replace with uniform, see if PPL changes)

### Scenario 3: Architectural limitation
**Symptoms**:
- High belief diversity (μ_dist > 0.4)
- Uniform attention (row_std < 0.03)
- Lower-spin heads dominate (ℓ0, ℓ1)

**Interpretation**:
- Model CAN'T learn sharp attention with current architecture
- Low-dimensional irreps lack geometric capacity

**Action**: Increase proportion of tensor heads (ℓ2, ℓ3)

## Enabling Ablation Study

To enable the ablation test, modify your model's forward method:

```python
def forward(self, input_ids, targets=None, beta_override=None):
    """
    Args:
        beta_override: Optional (B, H, N, N) tensor to replace learned attention
    """
    # ... existing code ...

    # In attention computation:
    if beta_override is not None:
        beta = beta_override
    else:
        # Compute beta normally
        beta = self._compute_attention(...)

    # ... rest of forward pass ...
```

Then the validation script will compare:
- Learned attention
- Uniform attention (β_ij = 1/N)
- Local attention (β_ij = 0 if |i-j| > 5)

## Common Questions

### Q: My row_std is 0.024, is that bad?

**A**: Not necessarily! If:
- Query variation > 0.05 (different tokens attend differently)
- Model perplexity is good
- Positional structure exists

Then 0.024 variation might be sufficient. Remember: small differences get amplified by softmax.

### Q: All my scalar heads (ℓ0) are uniform, is that a bug?

**A**: No, it's a **feature** (or limitation) of low-dimensional irreps. Scalar heads project beliefs to 1D, which loses geometric structure. This is why tensor heads (ℓ2, dim=5) show more structure.

### Q: Should I mask self-attention?

**A**: Try it! If diagonal KL is exactly 0 (KL(q_i || q_i) = 0), softmax will heavily favor self-attention, suppressing all other values uniformly. Masking the diagonal forces the model to differentiate among context tokens.

### Q: How do I know if attention is actually helping?

**A**: Run the ablation study (requires model modification). If learned attention gives significantly better PPL than uniform, it's helping.

## Related Files

- `ATTENTION_UNIFORMITY_FINDINGS.md` - Detailed analysis of spin-dependent attention behavior
- `transformer/train_publication.py` - Training script with built-in attention diagnostics
- `transformer/attention.py` - Attention implementation

## Citation

If this tool helps you discover insights about your model, please cite both:
1. Your model's attention mechanism
2. The specific tests that revealed the insight

Example:
> "Query-side variation analysis revealed that while key-side attention appeared uniform (row_std=0.024), different query tokens exhibited distinct attention patterns (L2=0.052), suggesting the model learned token-specific context selection despite superficial uniformity."
