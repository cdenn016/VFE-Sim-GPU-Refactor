# Pure FEP Transformer: Complete Architecture Overview

This document explains how the Pure FEP (Free Energy Principle) Transformer works - a transformer architecture that learns **entirely through Variational Free Energy (VFE) minimization**, without backpropagation or external optimizers.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [The Complete VFE Formula](#the-complete-vfe-formula)
3. [What Beliefs and Priors Represent](#what-beliefs-and-priors-represent)
4. [Architecture Components](#architecture-components)
5. [Two-Timescale Learning Dynamics](#two-timescale-learning-dynamics)
6. [How Attention Emerges](#how-attention-emerges)
7. [Meta-Agents and Hierarchical Structure](#meta-agents-and-hierarchical-structure)
8. [Gauge-Equivariant Position Encoding](#gauge-equivariant-position-encoding)
9. [Untied Embeddings](#untied-embeddings)
10. [Data Flow Summary](#data-flow-summary)

---

## Core Philosophy

Standard transformers use backpropagation to adjust weights. The Pure FEP Transformer instead follows a biological principle: **minimize surprise**.

The brain is theorized to continuously minimize its "free energy" - a bound on surprise. The Pure FEP Transformer implements this:

1. **Beliefs (q)** represent what the model INFERS given observations
2. **Priors (p)** represent what the model EXPECTS before seeing observations
3. **Learning** = priors flowing toward posteriors (beliefs)

No `optimizer.step()`. No `.backward()`. Just VFE gradient descent on beliefs and slow prior evolution.

---

## The Complete VFE Formula

The transformer minimizes this objective:

```
F = α·Σ_i KL(q_i||p_i)                       [Self-coupling: belief-to-prior]
  + λ_β·Σ_i Σ_j β_ij·KL(q_i||Ω_ij·q_j)      [Belief alignment: attention]
  + Σ_i E_{q_i}[-log p(y_i|μ_i)]             [Observation likelihood: CE]
  + λ_γ·Σ_i Σ_j γ_ij·KL(p_i||Ω_ij·p_j)      [Prior alignment: structure]
  + Σ_i Σ_d decay^d·KL(p_i||h_i^d)          [Ouroboros Tower: hyperpriors]
```

Where:
- `i, j` index positions (tokens/agents)
- `q_i` = belief distribution at position i (posterior)
- `p_i` = prior distribution at position i
- `β_ij` = belief attention weights (dynamic, from current beliefs)
- `γ_ij` = prior attention weights (structural, learned)
- `Ω_ij` = gauge transport operator from position j to i
- `h_i^d` = hyperprior at depth d (static or slow)
- `y_i` = observation (target token)

### What Each Term Does

| Term | Purpose | Timescale |
|------|---------|-----------|
| **KL(q\|\|p)** | Beliefs stay close to priors | Fast (inference) |
| **β·KL(q\|\|Ωq)** | Beliefs align with neighbors | Fast (inference) |
| **CE(μ, y)** | Beliefs explain observations | Fast (inference) |
| **γ·KL(p\|\|Ωp)** | Priors of related tokens align | Slow (learning) |
| **KL(p\|\|h)** | Priors anchored to hyperpriors | Slowest (structure) |

---

## What Beliefs and Priors Represent

### The Key Insight

```
μ_p (prior mean)    → encodes WHAT token (content/identity)
φ   (gauge frame)   → encodes WHERE in sequence (position)
```

The token embedding IS the prior. Position is encoded separately in the gauge frame.

### In Language Modeling

| Component | Symbol | Meaning |
|-----------|--------|---------|
| **Prior** | p_i = embed[x_i] | "What I expect given input token x_i" |
| **Belief** | q_i (after VFE) | "What I infer given context + observations" |
| **Observation** | y_i = target | "The actual next token" |
| **Likelihood** | p(y\|μ) = softmax(W·μ) | "How well μ predicts y" |

### The Belief is a BALANCE

During VFE dynamics, the belief μ_q finds equilibrium between competing forces:

```
        μ_p (prior)
           \
            \  KL(q||p) pulls toward prior
             \
              ★ ← μ_q lands HERE (balanced point)
             / \
   β·align /   \ CE gradient
          /     \
   neighbors   embed[y] (observation)
```

The posterior is NOT the observation - it's a compromise that:
- Respects the prior (doesn't forget token identity)
- Aligns with context (attention to neighbors)
- Explains observations (predicts target)

---

## Architecture Components

### 1. Token Embeddings as Priors

Each token v has a prior belief encoded in its embedding:
```
p_v = N(μ_v, Σ_v) where μ_v = embed[v]
```

**Input:** Initialize belief from input token's prior
```python
q(z_i) ← p_{input_token[i]}  # Belief starts at prior
```

**Output:** Compare belief to all token embeddings
```python
logits = μ_q @ W_out.T  # Similarity to output embeddings
```

### 2. Untied Embeddings (Critical for Pure FEP)

With tied embeddings, the same matrix serves as both:
- Input priors (what to expect from this token)
- Output targets (what to aim for when predicting)

**Problem:** This creates circular learning dynamics.

**Solution:** Untie embeddings:
```python
W_in  = input_embed.weight   # (V, K) - PRIORS, updated via p-flow
W_out = output_embed.weight  # (V, K) - OBSERVATIONS, fixed anchors
```

Now:
- W_in learns "what beliefs should look like when this token is input"
- W_out defines "what the observation targets look like"
- No collapse because they're separate matrices

### 3. Gauge Position Encoding

Position lives in the gauge frame φ, NOT in the belief mean μ:

```python
position_5: μ = [semantic content], φ = [0.1, 0.2, 0.3]
position_9: μ = [semantic content], φ = [0.5, 0.6, 0.7]
```

Same μ (token identity), different φ (position).

Position affects HOW beliefs interact through transport:
```
Ω_ij = exp(φ_i - φ_j)  # Relative position determines transport
```

---

## Two-Timescale Learning Dynamics

### Fast Timescale: Q-Flow (Inference)

Within a single forward pass, beliefs evolve to minimize F:

```python
for step in range(n_vfe_steps):
    # Compute attention from current beliefs
    β_ij = softmax(-KL(q_i || Ω_ij·q_j) / κ)

    # Compute gradients
    grad_q = ∂F/∂q = ∂KL(q||p)/∂q + ∂(β·alignment)/∂q + ∂CE/∂q

    # Natural gradient descent
    μ_q ← μ_q - η · Σ_q · grad_q
```

This is **perception** - inferring the current situation.

### Slow Timescale: P-Flow (Learning)

After inference, priors slowly evolve toward posteriors:

```python
# P-flow: prior moves toward posterior
μ_p ← (1 - lr) · μ_p + lr · μ_q

# Equivalently:
μ_p ← μ_p + lr · (μ_q - μ_p)
```

This is **learning** - updating expectations based on experience.

### Why Two Timescales?

| Flow | What Updates | Rate | Purpose |
|------|--------------|------|---------|
| Q-flow | Beliefs q | Fast (η ≈ 0.1) | Fit current observation |
| P-flow | Priors p | Slow (lr ≈ 0.01) | Accumulate knowledge |

The prior accumulates "where beliefs typically end up" across many examples.

---

## How Attention Emerges

### Attention from KL Divergence

No learned W_Q, W_K, W_V matrices. Attention emerges from belief geometry:

```python
# Step 1: Compute divergence between transported beliefs
KL_ij = KL(q_i || Ω_ij · q_j)

# Step 2: Convert to attention weights
β_ij = softmax(-KL_ij / κ)  # Low KL = high attention

# Step 3: Aggregate information
μ_aggregate = Σ_j β_ij · Ω_ij · μ_j
```

Tokens with aligned beliefs (low KL after transport) attend to each other.

### Two Types of Attention

| Type | Formula | Meaning |
|------|---------|---------|
| **β_ij (belief attention)** | softmax(-KL(q_i\|\|Ω·q_j)/κ) | "These align NOW" (dynamic) |
| **γ_ij (prior attention)** | From prior similarity | "These SHOULD align" (structural) |

β is computed fresh each forward pass from current beliefs.
γ is learned structure about which tokens are related.

---

## Meta-Agents and Hierarchical Structure

### What Are Meta-Agents?

When tokens share aligned beliefs/priors, they form a **meta-agent**:

```
β matrix:
        cat  dog  sat  mat
cat   [  1   .8   .1   .1  ]
dog   [ .8    1   .1   .1  ]     ← "cat-dog" meta-agent
sat   [ .1   .1    1   .7  ]
mat   [ .1   .1   .7    1  ]     ← "sat-mat" meta-agent
```

The block structure in β reveals meta-agents.

### Meta-Agents at Different Levels

| Level | What Aligns | Timescale | Example |
|-------|-------------|-----------|---------|
| Beliefs | High β_ij now | Fast | "cat" and "dog" in this sentence |
| Priors | High γ_ij learned | Slow | "cat" and "dog" are both animals |
| Hyperpriors | Shared h | Static | Abstract categories |

### The Human Analogy

```
Shared generative model (priors)?  + Shared beliefs?  = Meta-agent?

All humans:      ✅ Similar priors       ❌ Different beliefs  → Same species
Cult/clique:     ✅ Similar priors       ✅ Aligned beliefs    → Meta-agent!
Human + dog:     ❌ Incompatible priors  ❌ Can't align        → Never meta-agent
```

Meta-agent formation requires:
1. **Compatible priors** (gauge transport can align them)
2. **Aligned beliefs** (actually synchronize in context)

### Gauge Orbits as Categories

Meta-agents are **gauge orbits** in prior space:

```
"animal" = { p : p ≈ Ω·p_cat ≈ Ω'·p_dog ≈ Ω''·p_bird ... }
```

Tokens whose priors are related by gauge transport form an equivalence class.

---

## Gauge-Equivariant Position Encoding

### Transport Operators

The gauge transport Ω_ij relates beliefs at different positions:

```
Ω_ij = exp(φ_i · G) · exp(-φ_j · G)
```

Where G are the SO(3) or SO(N) generators.

### What Transport Does

```
μ_transported = Ω_ij · μ_j           # Rotate belief j into frame i
Σ_transported = Ω_ij · Σ_j · Ω_ij^T  # Rotate covariance accordingly
```

### Translation Invariance

If we shift all frames by δ:
```
φ_i → φ_i + δ  for all i
Ω_ij = exp(φ_i + δ) · exp(-φ_j - δ) = exp(φ_i) · exp(-φ_j)  # Unchanged!
```

Relative positions determine attention, not absolute positions.

---

## Untied Embeddings

### Why Untying is Necessary

With tied embeddings (W_in = W_out):

```
Input "cat" → embed["cat"] → VFE → μ_q → predict "sat"
                                    ↓
                              μ_q ≈ embed["sat"]

If we update embed["cat"] ← μ_q ≈ embed["sat"]
→ "cat" embedding becomes like "sat" embedding
→ Embeddings collapse!
```

### The Correct Setup

```python
W_in  = input_embed.weight   # Priors - "what to expect from this token"
W_out = output_embed.weight  # Targets - "what observations look like"
```

**P-flow updates W_in only:**
```python
W_in[x_i] ← (1-lr) · W_in[x_i] + lr · μ_q[i]
```

**W_out stays fixed** as the observation reference.

---

## Data Flow Summary

```
Input: token_ids (B, N)
         │
         ▼
┌─────────────────────────────────────────┐
│  Input Embedding (W_in)                 │
│  μ_p, σ_p ← W_in[token_ids]  (PRIORS)   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Position Encoding                       │
│  φ = position_encode(positions)          │
│  (Position in gauge frame, NOT in μ)     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  VFE Dynamics (Q-FLOW)                   │
│  For step in range(n_vfe_steps):         │
│    1. β_ij = softmax(-KL(q_i||Ω·q_j)/κ) │
│    2. grad = ∂F/∂q (prior + align + CE) │
│    3. μ_q ← μ_q - η·Σ·grad              │
│  Output: μ_q (balanced posterior)        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Output Projection (W_out)               │
│  logits = μ_q @ W_out.T  (FIXED)         │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Learning (P-FLOW)                       │
│  W_in[x_i] ← (1-lr)·W_in[x_i] + lr·μ_q  │
│  (Priors move toward posteriors)         │
└─────────────────────────────────────────┘
         │
         ▼
Output: logits (B, N, vocab_size)
```

---

## The Complete Learning Picture

```
                    HYPERPRIORS (h)
                         │
                         │ KL(p||h) - anchor
                         ▼
    ┌─────────────────────────────────────┐
    │           PRIORS (p = W_in)          │
    │                                      │
    │  ← γ·KL(p||Ωp) - structural align   │
    │  ← p-flow: p → q (slow learning)    │
    └─────────────────────────────────────┘
                         │
                         │ KL(q||p) - initialization
                         ▼
    ┌─────────────────────────────────────┐
    │          BELIEFS (q)                 │
    │                                      │
    │  ← β·KL(q||Ωq) - attention align    │
    │  ← CE(Wμ,y) - observation gradient  │
    │  ← VFE descent (fast inference)     │
    └─────────────────────────────────────┘
                         │
                         │ logits = q @ W_out.T
                         ▼
                   OBSERVATIONS (y)
```

**Three timescales:**
- **Beliefs (q):** Fastest - within forward pass
- **Priors (p):** Slow - across training steps
- **Hyperpriors (h):** Slowest/static - fixed structure

---

## Configuration Quick Reference

Key parameters for Pure FEP mode:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ffn_pure_fep_mode` | True | Enable pure FEP (no backprop) |
| `ffn_n_iterations` | 10 | VFE steps per forward pass |
| `ffn_prior_lr` | 0.1 | P-flow learning rate |
| `tie_embeddings` | False | Must be False for pure FEP |
| `gauge_fixed_priors` | False | Must be False for per-token learning |
| `kappa` | 1.0 | Attention temperature |
| `alpha` | 0.01 | Prior anchoring weight |

---

## Summary

The Pure FEP Transformer derives everything from VFE minimization:

| Traditional | Pure FEP |
|-------------|----------|
| nn.Embedding | Input priors (W_in) |
| Output projection | Observation anchors (W_out) |
| W_Q, W_K, W_V | KL-based attention |
| Position sinusoids | Gauge frame φ |
| Backprop | P-flow (priors → posteriors) |
| Adam/SGD | Natural gradient descent |
| Attention heads | Emergent meta-agents |

The result is a transformer that learns like a brain is theorized to: by minimizing surprise through belief-prior dynamics, with meta-agents emerging naturally from attention structure.
