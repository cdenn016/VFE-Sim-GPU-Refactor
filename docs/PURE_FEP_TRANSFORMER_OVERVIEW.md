# Pure FEP Transformer: Complete Architecture Overview

This document explains how the Pure FEP (Free Energy Principle) Transformer works - a transformer architecture that learns **entirely through Variational Free Energy (VFE) minimization**, without backpropagation or external optimizers.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [The Complete VFE Formula](#the-complete-vfe-formula)
3. [Architecture Components](#architecture-components)
4. [Two-Timescale Learning Dynamics](#two-timescale-learning-dynamics)
5. [How Attention Emerges](#how-attention-emerges)
6. [Gauge-Equivariant Position Encoding](#gauge-equivariant-position-encoding)
7. [Advanced Features](#advanced-features)
8. [Data Flow Summary](#data-flow-summary)

---

## Core Philosophy

Standard transformers use backpropagation to adjust weights. The Pure FEP Transformer instead follows a biological principle: **minimize surprise**.

The brain is theorized to continuously minimize its "free energy" - a bound on surprise. The Pure FEP Transformer implements this:

1. **Beliefs** represent what the model thinks about each token
2. **Priors** represent what the model expects before seeing data
3. **Learning** = adjusting priors to reduce the gap between beliefs and observations

No `optimizer.step()`. No `.backward()`. Just VFE gradient descent on beliefs and slow prior evolution.

---

## The Complete VFE Formula

The transformer minimizes this objective:

```
F = α·Σ_i KL(q_i||p_i)                       [Self-coupling: belief-to-prior]
  + λ_β·Σ_i Σ_j β_ij·KL(q_i||Ω_ij·q_j)      [Belief alignment: social inference]
  + Σ_i E_{q_i}[-log p(y_i|z_i)]             [Observation likelihood]
  + λ_γ·Σ_i Σ_j KL(p_i||Ω_ij·p_j)           [Prior coupling: world model coherence]
  + Σ_i Σ_d decay^d·KL(p_i||h_i^d)          [Ouroboros Tower: non-Markovian memory]
```

Where:
- `i, j` index positions (tokens)
- `d` indexes ancestor depth in the hierarchy
- `q_i` = belief distribution at position i
- `p_i` = prior distribution at position i
- `β_ij` = attention weights (emerge from KL divergences!)
- `Ω_ij` = transport operator from position j to i
- `h_i^d` = hyperprior from ancestor at depth d

### What Each Term Does

| Term | Purpose | Effect |
|------|---------|--------|
| **Self-coupling** | Keeps beliefs close to priors | Regularization, prevents belief drift |
| **Belief alignment** | Positions attend to each other | Creates the attention mechanism |
| **Observation likelihood** | Match predictions to actual tokens | The cross-entropy loss |
| **Prior coupling** | Priors learn from each other | World model consistency |
| **Ouroboros Tower** | Non-Markovian memory | Long-range hierarchical influence |

---

## Architecture Components

### 1. PriorBank: Unified Embedding and Output

**Location:** `transformer/pure_fep_transformer.py:107-295`

Traditional transformers have separate embedding and output layers. PriorBank unifies them:

```
Each token v has a prior belief: π_v = N(μ_v, Σ_v)
```

**Encoding (replaces nn.Embedding):**
```python
q(z_t) ← π_{token}  # Initialize belief from token's prior
```

**Decoding (replaces linear output projection):**
```python
p(y = v | q) ∝ exp(-KL(q || π_v) / τ)  # Probability via KL to all token priors
```

This creates beautiful symmetry - the same prior bank handles both directions.

### 2. GaugePositionEncoder: Position in φ, Not μ

**Location:** `transformer/pure_fep_transformer.py:301-398`

Standard transformers add position to the embedding. This is ad hoc - it conflates content and position.

The Pure FEP Transformer separates them:
- **Content** lives in the belief mean μ
- **Position** lives in the gauge frame φ ∈ so(3)

```python
# Same token at different positions:
position_5: μ = [semantic content], φ = [0.1, 0.2, 0.3]
position_9: μ = [semantic content], φ = [0.5, 0.6, 0.7]  # SAME μ, DIFFERENT φ
```

Position information affects **how beliefs interact** (through transport operators), not **what they contain**.

### 3. PureFEPLayer: The Core Processing Unit

**Location:** `transformer/pure_fep_transformer.py:621-1752`

Each layer maintains:
- **Beliefs** `q_i = N(μ_q, Σ_q)` - what the model currently thinks
- **Priors** `p_i = N(μ_p, Σ_p)` - what the model expects (position-dependent!)
- **Gauge frames** `φ_i` - for parallel transport between positions

The forward pass:
1. Initialize beliefs from input
2. Receive priors from parent layer (if any)
3. Run N VFE gradient descent steps
4. Return refined beliefs

### 4. PureFEPTransformer: The Complete Model

**Location:** `transformer/pure_fep_transformer.py:1755-2199`

Combines all components with configurable modes:

| Component | Options | Pure FEP Choice |
|-----------|---------|-----------------|
| Embedding | 'learned', 'prior_bank', 'hybrid' | `prior_bank` |
| Position | 'sinusoidal_mu', 'gauge_frame', 'both' | `gauge_frame` |
| Output | 'linear', 'kl_to_prior', 'both' | `kl_to_prior` |

---

## Two-Timescale Learning Dynamics

Learning happens at two timescales:

### Fast Timescale: Perception (VFE Steps)

Within a single forward pass, beliefs are updated multiple times:

```python
for step in range(n_vfe_steps):  # Default: 20 steps
    # Natural gradient descent on beliefs
    μ_q ← μ_q - η_μ · Σ_q · ∂F/∂μ_q
    σ_q ← σ_q - η_σ · ∂F/∂σ_q
```

This is **perception** - making sense of the current input given current priors.

### Slow Timescale: Learning (Prior Evolution)

After perception, priors slowly evolve:

```python
# Priors move toward beliefs (what the model learned)
μ_p ← μ_p + λ · (μ_q - μ_p)  # EMA with learning rate λ

# Or gradient-based:
μ_p ← μ_p - η_p · ∂F/∂μ_p
```

This is **learning** - updating what the model expects based on what it perceived.

### Why Two Timescales?

| Timescale | Rate | Purpose |
|-----------|------|---------|
| Fast (beliefs) | η_μ ≈ 0.1 | Quickly fit current observation |
| Slow (priors) | η_p ≈ 0.01 | Gradually accumulate knowledge |

Without this separation, the model would either:
- Overfit to each sample (if priors update too fast)
- Never learn patterns (if priors don't update at all)

---

## How Attention Emerges

Standard attention uses learned Q, K, V projections. Pure FEP attention **emerges from information geometry**:

### Step 1: Compute KL Divergences

For each pair of positions (i, j), compute how "surprising" j's belief is from i's perspective:

```python
KL_ij = KL(q_i || Ω_ij · q_j)  # KL after transporting j to i's frame
```

### Step 2: Convert to Attention Weights

```python
β_ij = softmax(-KL_ij / κ)  # Temperature κ controls sharpness
```

Positions with similar beliefs (low KL) get high attention.

### Step 3: Aggregate Information

```python
# Message from j to i, weighted by attention
μ_aggregate = Σ_j β_ij · Ω_ij · μ_j
```

**No learned W_Q, W_K, W_V matrices!** Attention weights emerge from the geometry of belief distributions.

---

## Gauge-Equivariant Position Encoding

### The Problem with Covariance Transport

When computing `KL(q_i || Ω_ij · q_j)`, we need to transport j's distribution to i's frame:

```
μ_transported = Ω_ij · μ_j        # Mean transport: simple matrix multiply
Σ_transported = Ω_ij · Σ_j · Ω_ij^T  # Covariance transport: full tensor contraction
```

The full covariance transport requires O(N²K²) memory - prohibitive for large models.

### Efficient Diagonal Transport

For diagonal covariances, we use an efficient formula:

```python
# Transport diagonal of Ω @ diag(σ) @ Ω^T without materializing full tensor
# Formula: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² · σ[l]

Omega_sq = Omega ** 2  # (B, N, N, K, K)
sigma_transported = torch.einsum('bijkl,bjl->bijk', Omega_sq, sigma_j)
```

This maintains gauge equivariance while using only O(N²K) memory.

### What is Gauge Equivariance?

The transport operator `Ω_ij = exp(φ_i) · exp(-φ_j)` depends on gauge frames:

```
If we shift all frames: φ_i → φ_i + δ
Then: Ω_ij → Ω_ij (unchanged!)
```

This gives **translation invariance**: the same relative positions produce the same attention pattern, regardless of absolute position.

---

## Advanced Features

### Prior Coupling (λ_γ term)

When enabled (`prior_coupling_enabled=True`), priors learn from each other:

```
F_prior = λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)
```

This encourages priors to form a **coherent world model** - position 5's prior should be consistent with position 6's prior.

### Ouroboros Tower (Non-Markovian Memory)

Standard hierarchies are Markovian: layer L only receives priors from layer L+1.

The Ouroboros Tower breaks this:

```
Layer 0 gets priors from:
  - Layer 1 (parent)
  - Layer 2 (grandparent) with decay^1
  - Layer 3 (great-grandparent) with decay^2
  - ...
```

This creates long-range memory where top-level abstractions directly influence low-level processing.

### Dynamic Layer Emergence

When enabled, the model can spawn new layers if VFE gradients exceed a threshold:

```python
if ∇F > layer_spawn_threshold:
    spawn_new_layer()
```

This allows the hierarchy depth to adapt to task complexity.

---

## Data Flow Summary

```
Input: token_ids (B, N)
         │
         ▼
┌─────────────────────────────────────────┐
│  PriorBank.encode()                     │
│  μ_q, σ_q ← π_{token_ids}               │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  GaugePositionEncoder()                 │
│  φ = compose_bch(φ_token, φ_position)   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  For each layer:                        │
│  ┌───────────────────────────────────┐  │
│  │ 1. Init beliefs from input        │  │
│  │ 2. Receive priors from parent     │  │
│  │ 3. For step in range(n_vfe_steps):│  │
│  │    a. Compute β_ij from KL        │  │
│  │    b. Compute VFE gradients       │  │
│  │    c. Update beliefs (natural grad)│ │
│  │ 4. Update persistent priors       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  PriorBank.decode()                     │
│  logits = -KL(q || π_v) / τ             │
└─────────────────────────────────────────┘
         │
         ▼
Output: logits (B, N, vocab_size)
```

---

## Configuration Quick Reference

Key `PureFEPConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 127 | Must be ODD for SO(3) irreps |
| `num_layers` | 2 | Hierarchical depth |
| `belief_steps` | 20 | VFE iterations per forward |
| `alpha` | 0.1 | Self-coupling weight |
| `kappa` | 0.1 | Attention temperature |
| `mu_lr` | 0.1 | Belief mean learning rate |
| `prior_lr` | 0.01 | Prior update rate (slower!) |
| `pure_fep_mode` | True | Disable backprop entirely |
| `embedding_mode` | 'prior_bank' | Use unified prior bank |
| `position_mode` | 'gauge_frame' | Position in gauge frames |
| `output_mode` | 'kl_to_prior' | Output via KL to priors |

---

## Summary

The Pure FEP Transformer eliminates ad hoc components by deriving everything from VFE minimization:

| Traditional | Pure FEP |
|-------------|----------|
| nn.Embedding | PriorBank.encode() |
| W_Q, W_K, W_V | KL-based attention |
| Position sinusoids | Gauge frame φ |
| Linear output | KL to token priors |
| Backprop | Two-timescale VFE dynamics |
| Adam/SGD | Natural gradient descent |

The result is a transformer that learns like a brain is theorized to: by minimizing surprise through belief-prior dynamics.
