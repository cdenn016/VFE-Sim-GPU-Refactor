Communication Style
Be direct:

State errors and concerns plainly without excessive hedging
"This is wrong because X" not "This might potentially be slightly off"
Push back:

Challenge gaps in derivations, ask for justification
If a claim needs proof, ask for it
Skip praise preambles:

No "Great question!" openers—just answer
No "Excellent point!"—just engage with the substance
Flag simpler alternatives:

Call out over-engineering
Ask what complexity buys if something seems unnecessarily elaborate
Maintain position under pushback:

Don't fold immediately when disagreeing
Ask "What am I missing?" rather than capitulating
Honest uncertainty:

"I'm not sure this is right" beats confident speculation
Acknowledge when something needs verification

## Transformer VFE Implementation - Bug Fixes & Architecture Notes

### Critical Bugs Fixed (Dec 2025)

| Commit | Bug | Impact | Fix |
|--------|-----|--------|-----|
| `0bef864` | OOV tokens in val but not train | Val sees unknown tokens | Pass train vocab_mapping to val dataset |
| `7e9f0ab` | Padding not masked in loss | Loss computed on pad tokens | Add `pad_token_id` param, use as `ignore_index` |
| `85ee4df` | Fallback URLs downloaded PTB instead of WikiText-2 | Wrong dataset distribution | Update URLs to WikiText-2 sources |
| `e3b7376` | **CRITICAL**: `.detach()` on VFE outputs | **All gradients broken** | Remove `.detach()` from `VFE_dynamic` returns |

### Architecture: FEP, Not EM

The VFE transformer does **NOT** use traditional E-M steps. It implements **Free Energy Principle (FEP)** variational inference:

```
Algorithm: Iterative VFE Minimization with Fixed Priors
─────────────────────────────────────────────────────────
For each forward pass:
  1. Initialize beliefs from embeddings: μ = embed(tokens), σ = σ_init
  2. For n_iterations VFE steps:
     a. Recompute attention β from current beliefs (dynamic-β)
     b. Compute VFE gradients: ∂F/∂μ, ∂F/∂σ
     c. Apply Fisher preconditioning (natural gradient)
     d. Update beliefs: μ ← μ - lr · ∇_nat F
  3. Return final beliefs μ_final, σ_final
  4. Backprop through μ_final updates embeddings (learning)
```

**Key insight**: This is principled FEP with two timescales:
- **Fast (perception)**: VFE iterations minimize F w.r.t. beliefs q(z)
- **Slow (learning)**: Backprop minimizes F w.r.t. generative model θ (embeddings)

The M-step option (`m_step_interval > 0`) is **disabled by default** and is experimental online prior adaptation, not core FEP.

### Known Issues (Not Yet Fixed)

| Issue | Location | Status |
|-------|----------|--------|
| Position encoding disabled | `model.py:335` (commented out) | Needs decision |
| Double `token_embed()` call | `model.py:433,566` | Potential redundancy |
| Observation gradient uses `no_grad()` | `variational_ffn.py` | By design (observation is fixed) |

### Additional Bug Fixed

| Commit | Bug | Impact | Fix |
|--------|-----|--------|-----|
| (pending) | KL in VFE gradient used only Mahalanobis term | Softmax coupling gradient was incorrect | Added full KL with trace + logdet terms |

The VFE gradient computation in `compute_vfe_gradients_gpu()` was computing KL values for the softmax coupling term using only the Mahalanobis (quadratic) term:
```python
# BEFORE (incomplete):
kl_values = 0.5 * δμᵀ Σ⁻¹ δμ

# AFTER (full KL):
kl_values = 0.5 * (tr(Σ_p⁻¹ Σ_q) + δμᵀ Σ_p⁻¹ δμ - K + log|Σ_p| - log|Σ_q|)
```


---

## Pure FEP Transformer (Dec 2025)

A new module `transformer/pure_fep_transformer.py` implements a transformer that learns **entirely through VFE minimization**, without backpropagation or external optimizers (Adam, SGD, etc.).

### Core Dynamics

Both beliefs and priors evolve via gradient descent on the Variational Free Energy:

```
dq/dt = -η_q · Σ_q · ∂F/∂μ_q     (fast timescale - perception)
dp/dt = -η_p · ∂F/∂μ_p           (slow timescale - learning)
```

Where the full VFE is:

```
F = α·KL(q||p)                              [Self-coupling]
  + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij·q_j)         [Belief alignment]
  + λ_γ·Σ_ij γ_ij·KL(p_i||Ω_ij·p_j)         [Prior alignment]
  + Σ_d decay^d·KL(p||h^d)                   [Ouroboros Tower]
  + E[-log p(y|z)]                           [Observation likelihood]
```

### Prior Gradient Components

The prior gradient `∂F/∂μ_p` includes three terms:

1. **Self-coupling**: `α·(μ_p - μ_q)/σ_p²` — pulls priors toward beliefs
2. **Prior alignment**: `λ_γ·Σ_j(μ_p - Ω_ij·μ_p[j])/σ_j` — inter-position consistency
3. **Hyperprior (Ouroboros)**: `Σ_d decay^d·(μ_p - μ_h^d)/σ_h^d` — alignment with ancestors

### Hierarchical Prior Flow

Priors also receive **top-down flow** from parent layer beliefs:

```
p_child^(ζ) ← EMA(p_child^(ζ), Ω · q_parent^(ζ+1))
```

This is NOT gradient descent — it's direct assignment of parent beliefs (transported via gauge) as child priors.

### Architecture

```
PureFEPTransformer
├── PriorBank (unified embedding + output via token priors)
├── GaugePositionEncoder (position in φ ∈ so(3))
├── PureFEPLayer[0] (scale ζ=0)
│   ├── Beliefs q_i = N(μ_q, Σ_q)
│   ├── Priors p_i = N(μ_p, Σ_p)  ← from parent + gradient descent
│   ├── Hyperpriors h_i^d         ← Ouroboros Tower (grandparent, great-grandparent, ...)
│   └── Gauge frames φ_i
├── PureFEPLayer[1] (scale ζ=1)
│   └── ...
└── Dynamic layer spawning/merging based on VFE pressure
```

### Advanced Features

| Feature | Config Flag | Description |
|---------|-------------|-------------|
| **Prior Coupling** | `prior_coupling_enabled` | λ_γ term for prior-prior KL alignment |
| **Gradient Prior Updates** | `gradient_prior_updates` | dp/dt = -∂F/∂p instead of EMA |
| **Ouroboros Tower** | `enable_ouroboros_tower` | Non-Markovian hyperpriors from ALL ancestors |
| **Dynamic Layers** | `dynamic_layers_enabled` | Spawn/merge layers based on VFE gradient |
| **Exact Covariance Transport** | `exact_covariance_transport` | Σ_t = Ω·Σ·Ω^T (vs approximate) |
| **Multi-Irrep** | `use_multi_irrep` | Block-diagonal SO(3) generators |

### Ouroboros Tower (Non-Markovian Memory)

Instead of just parent → child prior flow (Markovian), collect hyperpriors from ALL ancestors:

```
p_i^(ζ)     ← q^(ζ+1)      parent (immediate prior)
h_i^(ζ,0)   ← q^(ζ+2)      grandparent (1st hyperprior)
h_i^(ζ,1)   ← q^(ζ+3)      great-grandparent (2nd hyperprior)
```

Each hyperprior contributes with decaying weight: `F += Σ_d decay^d · KL(p || h^d)`

This creates **long-range memory** where top-layer abstract beliefs directly influence bottom layers.

### Dynamic Layer Emergence

Layers can spawn or merge based on VFE pressure:

- **SPAWN**: When VFE gradient norm > `layer_spawn_threshold`
  - New layer inserted with priors interpolated from neighbors
- **MERGE**: When adjacent layers have >0.99 belief similarity
  - Redundant layers combined to reduce computation

### Usage

```python
from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer

config = PureFEPConfig(
    embed_dim=127,              # Must be ODD for SO(3) irreps
    num_layers=3,
    vocab_size=10000,
    # Dynamics
    mu_lr=0.1,                  # Belief learning rate
    prior_lr=0.01,              # Prior learning rate (slower!)
    # Advanced features
    gradient_prior_updates=True,     # Full dp/dt = -∂F/∂p
    prior_coupling_enabled=True,     # Prior alignment term
    enable_ouroboros_tower=True,     # Non-Markovian hyperpriors
    tower_max_depth=3,
    tower_decay=0.3,
    dynamic_layers_enabled=True,     # Adaptive architecture
    max_layers=8,
)

model = PureFEPTransformer(config)
metrics = model.train_step(input_ids, targets, n_vfe_steps=20)
```

### Embedding Modes

| Mode | Description |
|------|-------------|
| `prior_bank` | **Principled**: Token priors serve as both embedding AND output |
| `learned` | Standard nn.Embedding (ad hoc but fast) |
| `hybrid` | Learned embedding + PriorBank output |

### Output Modes

| Mode | Description |
|------|-------------|
| `kl_to_prior` | **Principled**: p(y\|q) ∝ exp(-KL(q\|\|π_y)/τ) |
| `linear` | Standard W·μ projection (ad hoc) |

### Position Modes

| Mode | Description |
|------|-------------|
| `gauge_frame` | **Principled**: Position encoded in φ ∈ so(3) — affects transport! |
| `sinusoidal_mu` | Standard sinusoidal added to μ |

### The Full Learning Loop

```
Forward Pass (perception):
  1. Initialize beliefs from token priors: q ← π_token
  2. For n_vfe_steps:
     - Compute attention β_ij = softmax(-KL_ij/κ)
     - Compute VFE gradients ∂F/∂μ_q, ∂F/∂σ_q
     - Natural gradient descent: μ ← μ - η·Σ·∇F
  3. Output via KL to token priors: logits = -KL(q||π_v)/τ

Backward Pass (learning):
  1. Collect hyperpriors from ancestors (Ouroboros Tower)
  2. Top-down prior flow: p_child ← Ω·q_parent
  3. Gradient-based prior update: p ← p - η_p·∂F/∂p
  4. Check dynamic emergence conditions (spawn/merge layers)
```

This implements **predictive coding** in the FEP sense with proper two-timescale dynamics!

---

## References

- PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
- torch.compile: https://pytorch.org/docs/stable/torch.compiler.html
- Symplectic Integrators: Hairer, Lubich, Wanner - "Geometric Numerical Integration"
- Information Geometry: Amari - "Information Geometry and Its Applications"
- Free Energy Principle: Friston - "The free-energy principle: a unified brain theory?"

