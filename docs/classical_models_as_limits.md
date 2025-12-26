# Classical Social Influence Models as Limiting Cases of Hamiltonian VFE

## Core Framework Equations

### 1. Variational Free Energy (Potential)

```
F = Σ_i α ∫ χ_i KL(q_i||p_i)                        [Self-coupling]
  + Σ_ij λ_β ∫ χ_ij β_ij(c) KL(q_i||Ω_ij[q_j])     [Belief alignment]
  + Σ_ij λ_γ ∫ χ_ij γ_ij(c) KL(p_i||Ω_ij[p_j])     [Prior alignment]
  - Σ_i λ_obs ∫ χ_i E_q[log p(o|x)]                 [Observations]
```

Where:
- `q_i = N(μ_i, Σ_i)`: Agent i's belief distribution
- `p_i = N(μ_p,i, Σ_p,i)`: Agent i's prior distribution
- `β_ij(c) = softmax_j[-KL(q_i || Ω_ij q_j) / κ_β]`: Attention weights (belief-based)
- `γ_ij(c) = softmax_j[-KL(p_i || Ω_ij p_j) / κ_γ]`: Attention weights (prior-based)
- `Ω_ij`: Transport operator (gauge transformation from j to i's frame)
- `χ_i, χ_ij`: Spatial support overlap weights

### 2. Mass Matrix (Epistemic Inertia)

```
M_i = Σ_p,i^{-1} + Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
      ─────────   ────────────────────────────────
      Bare mass   Relational/social mass
```

### 3. Hamilton's Equations of Motion

**For belief means μ:**
```
dμ_i/dt = M_i^{-1} π_μ,i                                    [Velocity]
dπ_μ,i/dt = -∂F/∂μ_i - (1/2) π^T (∂M_i^{-1}/∂μ_i) π - γ π  [Force + geodesic correction + friction]
```

**For covariances Σ (hyperbolic SPD manifold):**
```
dΣ_i/dt = Σ_i Π_Σ,i Σ_i                                     [Geodesic flow]
dΠ_Σ,i/dt = -∂F/∂Σ_i - γ Π_Σ,i                              [Force + friction]
```

---

## Derivation of Classical Models

### Model 1: DeGroot Social Learning

**Classical form:**
```
x_i(t+1) = Σ_j w_ij x_j(t)
```

**Derivation from Hamiltonian VFE:**

Take the following limits:
1. **Overdamped limit**: `γ → ∞` (high friction, dissipative dynamics)
2. **No uncertainty**: `Σ_i → 0` (Dirac delta beliefs)
3. **Flat manifold**: No gauge structure, `Ω_ij = I`
4. **No self-coupling**: `α = 0`
5. **No prior alignment**: `λ_γ = 0`
6. **No observations**: `λ_obs = 0`
7. **Fixed attention**: `β_ij = w_ij` (constant, not softmax)

**Limiting dynamics:**

In overdamped limit: `dμ_i/dt ≈ -M_i^{-1} ∂F/∂μ_i`

With no uncertainty, the belief alignment energy becomes:
```
F_belief = λ_β Σ_ij w_ij ||μ_i - μ_j||^2 / 2
```

Gradient: `∂F/∂μ_i = λ_β Σ_j w_ij (μ_i - μ_j)`

With `M_i → I` (no mass structure when Σ → 0):
```
dμ_i/dt = -λ_β Σ_j w_ij (μ_i - μ_j)
        = λ_β [Σ_j w_ij μ_j - μ_i Σ_j w_ij]
```

If `Σ_j w_ij = 1` (row-stochastic), discrete-time update gives DeGroot:
```
μ_i(t+1) = Σ_j w_ij μ_j(t)
```

**What Hamiltonian VFE adds:**
- **Underdamped regime** (γ → 0): Beliefs can overshoot, oscillate
- **Uncertainty dynamics**: Full distribution evolution, not just means
- **Dynamic attention**: `β_ij` depends on current belief disagreement
- **Epistemic inertia**: Mass grows with attention received

---

### Model 2: Friedkin-Johnsen Opinion Dynamics

**Classical form:**
```
x_i(t+1) = α_i x_i(0) + (1-α_i) Σ_j w_ij x_j(t)
```

Where `α_i ∈ [0,1]` is stubbornness (attachment to initial opinion).

**Derivation from Hamiltonian VFE:**

Take DeGroot limits PLUS:
8. **Non-zero self-coupling**: `α > 0`
9. **Fixed priors**: `p_i(c) = N(μ_i(0), Σ_p)` (initial beliefs)

**Limiting dynamics:**

Free energy now includes:
```
F = α Σ_i ||μ_i - μ_i(0)||^2 / (2 Σ_p)              [Self-coupling to initial belief]
  + λ_β Σ_ij w_ij ||μ_i - μ_j||^2 / 2                [Social influence]
```

Gradient:
```
∂F/∂μ_i = (α/Σ_p)(μ_i - μ_i(0)) + λ_β Σ_j w_ij (μ_i - μ_j)
```

Overdamped dynamics with mass `M_i = Σ_p^{-1}`:
```
dμ_i/dt = -Σ_p ∂F/∂μ_i
        = -α(μ_i - μ_i(0)) - λ_β Σ_p Σ_j w_ij (μ_i - μ_j)
```

In equilibrium or discrete-time update:
```
μ_i^* = α_i' μ_i(0) + (1-α_i') Σ_j w_ij μ_j
```

Where `α_i' = α/(α + λ_β Σ_p Σ_j w_ij)` depends on:
- Prior precision `Σ_p^{-1}` (stubbornness from uncertainty)
- Social coupling strength `λ_β`

**What Hamiltonian VFE adds:**
- **Mechanistic stubbornness**: `α_i` emerges from prior precision, not ad-hoc parameter
- **Dynamic stubbornness**: `α_i' = α_i'(t)` changes as beliefs/attention evolve
- **Inertial effects**: High-mass agents (many followers) become MORE stubborn over time

---

### Model 3: Bounded Confidence (Hegselmann-Krause)

**Classical form:**
```
x_i(t+1) = average{ x_j(t) : |x_j(t) - x_i(t)| < ε }
```

Agents only influenced by similar others within threshold `ε`.

**Derivation from Hamiltonian VFE:**

Take DeGroot limits PLUS:
10. **Softmax attention with low temperature**: `κ_β → 0`

**Key insight:** Softmax attention creates natural threshold:
```
β_ij = exp(-KL(q_i || q_j) / κ_β) / Z_i

When κ_β → 0:
  β_ij → { 1/|N_i|  if KL(q_i || q_j) ≈ min_k KL(q_i || q_k)
         { 0        otherwise
```

For Gaussian beliefs with `Σ_i = Σ_j = σ^2 I`:
```
KL(q_i || q_j) ≈ ||μ_i - μ_j||^2 / (2σ^2)
```

So low-temperature softmax creates hard threshold at:
```
ε ≈ √(2σ^2 κ_β log(N))
```

**Limiting dynamics:**

Overdamped + low-κ gives:
```
dμ_i/dt ∝ Σ_j β_ij (μ_j - μ_i)
        ≈ average{μ_j - μ_i : ||μ_j - μ_i|| < ε}
```

**What Hamiltonian VFE adds:**
- **Continuous threshold**: Soft cutoff, not hard boundary
- **Adaptive threshold**: `ε = ε(σ, κ_β, N)` depends on uncertainty and temperature
- **Threshold asymmetry**: `β_ij ≠ β_ji` (directed influence)
- **Beyond mean-field**: Full distribution dynamics, not just means

---

### Model 4: Confirmation Bias / Biased Assimilation

**Psychological phenomenon:**
- People update less from counter-attitudinal evidence
- Prior beliefs resistant to change
- "Motivated reasoning"

**Derivation from Hamiltonian VFE:**

**NOT** an ad-hoc cognitive bias, but emergent from mass structure:

Recall mass matrix:
```
M_i = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
```

Velocity equation:
```
dμ_i/dt = M_i^{-1} π_μ,i
```

**Key insight:** High prior precision `Σ_p^{-1}` → high mass → slow updates

When agent receives evidence opposing their prior:
- Force `π_μ` points toward new evidence
- Large mass `M_i` resists acceleration
- Result: Small belief update `dμ_i/dt = M_i^{-1} π_μ << π_μ`

**Quantitative prediction:**

Update magnitude for evidence `o` contradicting prior:
```
Δμ_i ∝ M_i^{-1} ∂F_obs/∂μ_i
      = [Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T]^{-1} · (evidence force)
```

Agents with:
- Strong priors (`Σ_p^{-1}` large) → small updates (confirmation bias)
- Many followers (`Σ_j β_ij` large) → small updates (social confirmation bias)

**What Hamiltonian VFE adds:**
- **Mechanistic explanation**: Bias emerges from information geometry, not irrationality
- **Testable prediction**: Bias strength ∝ prior precision + social attention received
- **Individual differences**: Variation in bias from variation in Σ_p and β network

---

### Model 5: Backfire Effect

**Psychological phenomenon:**
- Weak counter-evidence INCREASES confidence in original belief
- "Boomerang effect" in persuasion

**Derivation from Hamiltonian VFE:**

**Requires underdamped dynamics** (low friction γ):

In underdamped regime, system has inertia:
```
dμ_i/dt = M_i^{-1} π_μ,i
dπ_μ,i/dt = -∂F/∂μ_i - γ π_μ,i
```

When γ << 1 (low friction), momentum π can cause overshoot:

**Scenario:**
1. Agent starts with belief `μ_i(0)` and momentum `π_μ = 0`
2. Weak counter-evidence creates force `F_evidence` pointing toward `μ_new`
3. Agent accelerates: `π_μ` increases
4. Agent's belief moves toward evidence
5. **But:** High mass `M_i` means slow deceleration
6. Agent overshoots `μ_new`, ends up on opposite side
7. Appears as "strengthening" original belief direction

**Quantitative condition for backfire:**

Backfire occurs when:
```
Δμ_final · (μ_i(0) - μ_new) > 0
```

(Final belief change same sign as original deviation)

This requires:
```
γ < γ_critical ≈ 2√(k/M_i)
```

Where `k` is the "stiffness" of the potential well.

**Prediction:** Backfire most likely for:
- **High-mass agents** (influencers, many followers)
- **Low friction** (underdamped social dynamics)
- **Weak evidence** (small perturbation)

**What Hamiltonian VFE adds:**
- **Mechanistic prediction**: Backfire emerges from underdamped dynamics, NOT irrationality
- **Who shows backfire**: Influencers (high `Σ_j β_ij`) more prone
- **When it occurs**: Weak evidence in low-friction regimes
- **Testable**: Backfire should show oscillations if measured over time

---

### Model 6: Social Impact Theory (Latané)

**Classical form:**
```
Impact = f × (Strength × Immediacy × Number)
```

Where:
- Strength: Expertise, status, power of source
- Immediacy: Proximity in space/time
- Number: How many sources

**Derivation from Hamiltonian VFE:**

Recall mass matrix:
```
M_i = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
      ─────────   ──────────────────────────────
      baseline    social mass from j's influence
```

**Mapping:**

**Strength** → `Σ_q,j^{-1}` (precision of source j's belief)
- Experts have high precision → high Σ^{-1} → large contribution to mass

**Immediacy** → Transport penalty `||Ω_ij - I||`
- Distant agents: `Ω_ij` has large rotation → KL penalty → low β_ij
- Close agents: `Ω_ij ≈ I` → small KL → high β_ij

**Number** → `Σ_j` (sum over all influencing agents)
- More sources → more terms in mass matrix

**Quantitative prediction:**

Social mass contribution from source j:
```
ΔM_i^{(j)} = β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
           ≈ (Immediacy) × (Strength) × (Attention)
```

Total epistemic inertia:
```
M_i ≈ Σ_p^{-1} + Σ_j (Strength_j × Immediacy_ij × β_ij)
                      ──────────────────────────────────
                      Latané's multiplicative impact
```

**What Hamiltonian VFE adds:**
- **Quantitative predictions**: Exact formula for impact, not qualitative
- **Testable mechanics**: Measure Σ_q,j, Ω_ij, β_ij independently
- **Dynamics**: How impact changes over time as beliefs evolve
- **Reciprocity**: Impact is NOT symmetric (β_ij ≠ β_ji in general)

---

### Model 7: Echo Chambers & Polarization

**Phenomenon:**
- Homophily → in-group clustering
- Out-group avoidance → polarization
- Belief divergence despite shared information

**Derivation from Hamiltonian VFE:**

Softmax attention creates **homophily automatically**:
```
β_ij ∝ exp(-KL(q_i || q_j) / κ_β)
```

Agents with similar beliefs → low KL → high attention
Agents with distant beliefs → high KL → low attention (ignore out-group)

**Dynamics:**

Two groups with initial beliefs `μ_A ≈ a`, `μ_B ≈ b` where `||a - b||` large.

Within-group attention:
```
β_ij^{(A)} ≈ 1/|A|  for i,j in group A  (high, uniform)
β_ij^{(cross)} ≈ 0  for i in A, j in B  (low, ignored)
```

Equations of motion:
```
dμ_i/dt ≈ Σ_{j ∈ group(i)} β_ij (μ_j - μ_i)  + (external forces)
```

**Result:** Two decoupled subsystems → beliefs converge within group, diverge between groups

**Stability analysis:**

Polarized state `{μ_A, μ_B}` is stable when:
```
KL(N(μ_A, Σ) || N(μ_B, Σ)) > κ_β log(N)
```

(Threshold for softmax to effectively zero out cross-group attention)

**What Hamiltonian VFE adds:**
- **Emergent homophily**: Not assumed, derived from attention mechanism
- **Phase transition**: Critical κ_β where polarization emerges
- **Stability analysis**: When polarized states are attractors vs. transient
- **Escape dynamics**: Underdamped regime allows tunneling between basins

---

## Summary Table

| Classical Model | Key Assumptions | Emerges from VFE when... | Novel Predictions from Full VFE |
|-----------------|-----------------|--------------------------|----------------------------------|
| **DeGroot** | Linear averaging, fixed weights | γ→∞, Σ→0, flat manifold, β fixed | Underdamped oscillations, dynamic β, epistemic inertia |
| **Friedkin-Johnsen** | Stubbornness parameter α | γ→∞, Σ→0, α>0 | α emerges from Σ_p, changes dynamically with attention |
| **Bounded Confidence** | Hard threshold ε | κ_β→0 (low temperature) | Soft threshold, adaptive, asymmetric influence |
| **Confirmation Bias** | Cognitive bias | High M from Σ_p^{-1} or social mass | Bias ∝ followers, mechanistic not irrational |
| **Backfire Effect** | Paradoxical strengthening | γ<γ_c (underdamped), high M | Predicts who/when: influencers, weak evidence |
| **Social Impact Theory** | Multiplicative impact | Direct readoff from M formula | Quantitative, time-evolving, asymmetric |
| **Echo Chambers** | Homophily assumption | Softmax β with moderate κ | Homophily emerges, phase transition, escape paths |

---

## Phase Diagram of Dynamics Regimes

```
            Friction γ
                │
     High       │  Gradient Flow Regime
      (γ→∞)     │  • DeGroot
                │  • Friedkin-Johnsen
                │  • Dissipative
   ─────────────┼─────────────────────  Critical Damping
                │
                │  Hamiltonian Regime
                │  • Backfire possible
     Low        │  • Oscillations
      (γ→0)     │  • Nearly conservative
                │
                └────────────────────> Social Coupling (Σ_j β_ij)
                         Low                       High


      Low Coupling:              High Coupling:
      • DeGroot-like            • Emergent rigidity
      • Fast convergence        • Epistemic inertia
      • No backfire             • Influencer effects
```

---

## Critical Parameters

| Parameter | Physical Meaning | Controls Transition Between... |
|-----------|------------------|-------------------------------|
| **γ** (friction) | Dissipation rate | Gradient flow ↔ Hamiltonian |
| **κ_β** (temperature) | Attention sharpness | Soft ↔ Hard bounded confidence |
| **Σ_p** (prior covariance) | Prior uncertainty | Flexible ↔ Stubborn agents |
| **Σ_j β_ij** (social coupling) | Followers/attention | Isolated ↔ Rigid (epistemic inertia) |
| **α** (self-coupling) | Prior weight | Social ↔ Individual-focused |

---

## Next Steps: Empirical Validation

To validate this unifying framework, test predictions that distinguish it from classical models:

1. **Underdamped dynamics** (backfire in influencers)
2. **Dynamic attention** (β_ij changes over time, not fixed)
3. **Epistemic inertia** (followers → rigidity correlation)
4. **Uncertainty propagation** (not just means, full distributions)

See `experiments/` for simulation demonstrations.
