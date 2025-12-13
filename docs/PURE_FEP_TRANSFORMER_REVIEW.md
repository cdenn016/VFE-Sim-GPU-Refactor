# Pure FEP Transformer Code Review

**Reviewer:** Claude (Opus 4.5)
**Date:** December 2025
**Files Reviewed:**
- `transformer/pure_fep_transformer.py`
- `transformer/train_pure_fep.py`
- `transformer/attention.py`
- `transformer/variational_ffn.py`
- `transformer/embeddings.py`
- `math_utils/generators.py`

---

## Executive Summary

The Pure FEP Transformer is an ambitious implementation of a transformer architecture that learns entirely through Variational Free Energy (VFE) minimization, embodying the Free Energy Principle. The theoretical foundation is sound and the code is well-documented. However, several critical runtime bugs must be fixed before the code can run successfully in all modes.

---

## Critical Bugs (Will Cause Runtime Errors)

### 1. Invalid Function Call Arguments

**Location:** `pure_fep_transformer.py:1291-1295`

```python
cached_transport = compute_transport_operators(
    phi, self.generators,
    use_fast_exp=self.config.use_fast_matrix_exp,  # NOT A VALID PARAMETER!
    exp_order=self.config.matrix_exp_order,         # NOT A VALID PARAMETER!
)
```

**Issue:** The function `compute_transport_operators` in `attention.py:129-132` only accepts `phi` and `generators`:

```python
def compute_transport_operators(
    phi: torch.Tensor,
    generators: torch.Tensor,
) -> dict:
```

**Impact:** `TypeError: compute_transport_operators() got an unexpected keyword argument 'use_fast_exp'`

**Fix:** Either:
1. Remove the invalid keyword arguments, or
2. Update `compute_transport_operators` to accept these parameters and implement fast matrix exponential

---

### 2. AttributeError in Hybrid Mode (Embedding)

**Location:** `pure_fep_transformer.py:1787`

```python
for param in [self.embedding.weight, self.output_proj.weight]:
```

**Issue:** When `embedding_mode='prior_bank'`, `self.embedding` is set to `None` (line 1381).

**Impact:** `AttributeError: 'NoneType' object has no attribute 'weight'`

**Fix:** Check for None before accessing:

```python
params = []
if self.embedding is not None:
    params.append(self.embedding.weight)
if self.output_proj is not None:
    params.append(self.output_proj.weight)
for param in params:
    ...
```

---

### 3. AttributeError in Hybrid Mode (Layer Output Projection)

**Location:** `pure_fep_transformer.py:1796`

```python
if layer.output_proj.weight.grad is not None:
```

**Issue:** When `output_mode='kl_to_prior'`, `layer.output_proj` is `None` (line 608).

**Impact:** `AttributeError: 'NoneType' object has no attribute 'weight'`

**Fix:** Add None check:

```python
for layer in self.layers:
    if layer.output_proj is not None and layer.output_proj.weight.grad is not None:
        ...
```

---

## Mathematical/Theoretical Issues

### 4. Unused `irrep_spec` Configuration

**Location:** `PureFEPConfig` class

**Issue:** The `irrep_spec` parameter allows specifying a multi-irrep decomposition like:
```python
irrep_spec = [
    ('ℓ0', 32, 1),  # 32 scalars
    ('ℓ1', 15, 3),  # 45 dims (vectors)
    ('ℓ2', 10, 5),  # 50 dims (rank-2 tensors)
]
```

However, this is only used for validation and printing. The actual generator construction always uses:
```python
gen_np = generate_so3_generators(embed_dim)  # Single irrep!
```

**Impact:** Misleading documentation; users may expect block-diagonal structure that doesn't exist.

**Recommendation:** Either:
1. Implement proper multi-irrep block-diagonal generators, or
2. Remove `irrep_spec` and clarify that only single irreps are supported

---

### 5. Overly Restrictive Dimension Constraint

**Location:** `PureFEPConfig.__post_init__`

```python
if self.embed_dim % 2 == 0:
    raise ValueError("embed_dim must be ODD for SO(3) irreps...")
```

**Issue:** This is only true for *single* SO(3) irreps. A sum of irreps can have any dimension (e.g., 1+3=4, which is even).

**Impact:** Unnecessarily restricts valid configurations if multi-irrep support were added.

---

### 6. Variance Transport Approximation

**Locations:**
- `pure_fep_transformer.py:1069-1070`
- `pure_fep_transformer.py:1116-1117`

```python
# For diagonal covariance: variance doesn't change under rotation
# (This is an approximation - full transport would rotate the covariance)
parent_sigma_transported = parent_sigma_q.clone()
```

**Issue:** Covariance *does* change under rotation unless isotropic (σ²I). The correct transport is:
```
Σ_transported = Ω @ Σ @ Ω^T
```

Even for diagonal Σ, the result is generally full (non-diagonal).

**Impact:** Theoretical correctness of prior updates is compromised. The approximation may work empirically but violates gauge equivariance.

---

### 7. KL Divergence in Prior Coupling Uses Wrong Covariance

**Location:** `compute_prior_coupling_loss` method

The method transports means but uses untransported variance for KL computation, which is mathematically inconsistent.

---

## Numerical Stability Issues

### 8. Potential Overflow in Perplexity Calculation

**Location:** `train_pure_fep.py:247`

```python
ppl = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow
```

**Issue:** `exp(100) ≈ 2.69 × 10^43` which is still astronomically large and meaningless as a metric.

**Recommendation:** Use `min(avg_loss, 20)` or report log-perplexity directly.

---

### 9. Cholesky Decomposition Failure Risk

**Location:** `variational_ffn.py:220`

```python
L_j_t = torch.linalg.cholesky(sigma_j_reg)
```

**Issue:** Even with regularization, transported covariances can be ill-conditioned. No error handling exists.

**Recommendation:** Add try/catch with fallback:

```python
try:
    L_j_t = torch.linalg.cholesky(sigma_j_reg)
except RuntimeError:
    # Fallback to eigenvalue-based logdet
    eigvals = torch.linalg.eigvalsh(sigma_j_reg)
    logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
```

---

### 10. Division by Very Small Numbers

**Location:** `pure_fep_transformer.py:1224`

```python
grad_mu_p = (self.prior_mu - mu_p_new) / self.prior_sigma.clamp(min=self.config.eps)**2
```

**Issue:** `eps = 1e-6`, so `eps² = 1e-12`. Dividing by this creates huge gradients.

**Recommendation:** Use larger minimum: `.clamp(min=1e-4)**2` or `.clamp(min=1e-8)` after squaring.

---

## Design/Code Quality Issues

### 11. Dead Code: `cached_transport` Parameter in VFE Gradients

**Location:** `variational_ffn.py:78-90`

The `cached_transport` parameter exists in the function signature but is never used. The function recomputes transport operators internally, defeating the purpose of caching.

---

### 12. Inconsistent Attribute Access Pattern

The code mixes direct access and `getattr` with defaults:
```python
self.config.lambda_obs              # Direct
getattr(self.config, 'lambda_obs', 1.0)  # With default
```

**Recommendation:** Be consistent. If attributes have defaults in the dataclass, use direct access.

---

### 13. Momentum Buffers Not Registered

```python
self._momentum_mu = None
self._momentum_sigma = None
```

**Issue:** These aren't registered with `register_buffer`, so:
1. They won't be saved/loaded with model state
2. They may cause issues with multi-GPU training (not on correct device)

**Fix:** Use `register_buffer` or initialize as None and move to device on first use.

---

### 14. `maybe_compile()` Never Called

The config has `use_torch_compile` flag and `maybe_compile()` method is defined, but it's never invoked anywhere.

---

## Performance Issues

### 15. O(N²×K²) Memory for Full Covariance Transport

**Location:** `variational_ffn.py:194-197`

```python
sigma_j_transported = torch.einsum(
    'bijkl,bjlm,bijmn->bijkn',
    Omega, sigma_j_diag, Omega.transpose(-1, -2)
)  # (B, N, N, K, K)
```

**Memory Calculation:** For B=24, N=128, K=127:
```
24 × 128 × 128 × 127 × 127 × 4 bytes ≈ 25 GB
```

**Impact:** Out-of-memory on most GPUs.

**Recommendation:**
1. Process in chunks
2. Use diagonal approximation more aggressively
3. Reduce sequence length for large K

---

### 16. Redundant KL Recomputation

**Location:** `pure_fep_transformer.py:988-996`

KL terms are recomputed for metrics after already being computed in the forward pass. Store values instead of recomputing.

---

## Test Coverage Recommendations

The test suite at the end of the file (`if __name__ == '__main__'`) only tests:
1. Pure FEP config forward pass
2. Ad hoc config forward pass

**Missing tests:**
- Hybrid mode training step
- `embedding_mode='hybrid'`
- Gradient flow verification
- Prior update correctness
- Gauge frame evolution

---

## Summary Table

| Category | Count | Severity |
|----------|-------|----------|
| Critical Bugs | 3 | High |
| Mathematical Issues | 4 | Medium-High |
| Numerical Stability | 3 | Medium |
| Design Issues | 4 | Low-Medium |
| Performance Issues | 2 | Medium |

---

## Recommended Fix Priority

1. **Immediate:** Fix the three critical bugs (runtime errors)
2. **High:** Address numerical stability issues (training failure risk)
3. **Medium:** Resolve mathematical approximations or document limitations
4. **Low:** Code cleanup and optimization

---

## Conclusion

The Pure FEP Transformer represents innovative research combining information geometry, gauge theory, and the Free Energy Principle. The core algorithms are theoretically motivated and well-documented. However, several implementation bugs prevent the code from running in all configurations. With the fixes outlined above, the implementation should be functional and ready for experimentation.
