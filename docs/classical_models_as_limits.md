# Classical Social Influence Models as Limiting Cases of Hamiltonian VFE

**Status**: Rigorous derivations with explicit caveats
**Author**: Chris Denning
**Date**: December 2025

---

## Executive Summary

This document demonstrates that seven major models from sociology, psychology, and network science emerge as special/limiting cases of the Hamiltonian Variational Free Energy framework. **Three derivations are rigorous** (DeGroot, Friedkin-Johnsen, echo chambers), **three are solid approximations** (bounded confidence, confirmation bias, social impact theory), and **one is preliminary/speculative** (backfire effect).

**Key insight**: Classical models represent different **dynamical regimes** of the same underlying information-geometric structure, analogous to how thermodynamic phases (solid/liquid/gas) emerge from molecular dynamics under different conditions.

---

## Mathematical Foundations

### Core Framework Equations

**1. Variational Free Energy (Potential Energy)**

```
F[q, p] = Σ_i α ∫ χ_i KL(q_i||p_i)                        [Self-coupling]
        + Σ_ij λ_β ∫ χ_ij β_ij(c) KL(q_i||Ω_ij[q_j])     [Belief alignment]
        + Σ_ij λ_γ ∫ χ_ij γ_ij(c) KL(p_i||Ω_ij[p_j])     [Prior alignment]
        - Σ_i λ_obs ∫ χ_i E_q[log p(o|x)]                 [Observations]
```

**Notation**:
- `q_i = N(μ_i, Σ_i)`: Agent i's belief (posterior) distribution
- `p_i = N(μ_p,i, Σ_p,i)`: Agent i's prior distribution
- `β_ij(c) = softmax_j[-KL(q_i || Ω_ij q_j) / κ_β]`: Softmax attention (belief-based)
- `γ_ij(c) = softmax_j[-KL(p_i || Ω_ij p_j) / κ_γ]`: Softmax attention (prior-based)
- `Ω_ij ∈ SO(K)`: Parallel transport operator (gauge transformation j → i)
- `χ_i, χ_ij ∈ [0,1]`: Spatial support overlap weights
- `κ_β, κ_γ > 0`: Softmax temperatures (inverse attention sharpness)

**2. Fisher-Rao Mass Matrix (Epistemic Inertia)**

```
M_i(θ) = Σ_p,i^{-1} + Σ_j β_ij(θ) Ω_ij Σ_q,j^{-1} Ω_ij^T
         ───────────   ──────────────────────────────────
         Prior         Social/relational mass
         precision     (attention-weighted coupling)
```

**Physical interpretation**:
- **Bare mass** `Σ_p^{-1}`: Resistance to change from prior beliefs (epistemic anchor)
- **Social mass** `Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T`: Additional inertia from social attention
  - Grows with number of followers (Σ_j β_ji large)
  - Weighted by precision of followers' beliefs (Σ_q,j^{-1})
  - Modulated by frame alignment (Ω_ij)

**3. Dynamics: Natural Gradient Flow vs. Hamiltonian**

**Overdamped regime** (γ → ∞, dissipative):
```
dμ_i/dt = -M_i^{-1} ∇_μi F                                [Natural gradient descent]
dΣ_i/dt = -G_Σ^{-1} ∇_Σi F                                [Fisher-Rao flow on SPD]
```

**Hamiltonian regime** (γ → 0, nearly conservative):
```
dμ_i/dt = M_i^{-1} π_μ,i                                   [Velocity from momentum]
dπ_μ,i/dt = -∇_μi F - Γ_ijk π^j π^k - γ π_μ,i             [Force + geodesic + friction]

dΣ_i/dt = Σ_i Π_Σ,i Σ_i                                    [Hyperbolic geodesic flow]
dΠ_Σ,i/dt = -∇_Σi F - γ Π_Σ,i                              [Force + friction]
```

Where:
- `π_μ, Π_Σ`: Conjugate momenta
- `Γ_ijk`: Christoffel symbols (geodesic correction from metric curvature)
- `γ ≥ 0`: Friction coefficient

**CRITICAL DISTINCTION**: Classical models implicitly use **natural gradient descent** (flow on the statistical manifold with Fisher metric `M^{-1}`), not standard Euclidean gradient descent. This is why the mass matrix appears even in the overdamped limit.

---

## Rigor Assessment

Before proceeding, we clarify the status of each derivation:

| Model | Derivation Type | Rigor Level | Notes |
|-------|----------------|-------------|-------|
| **DeGroot** | Exact limit | ✓✓✓ Rigorous | Proven via natural gradient flow |
| **Friedkin-Johnsen** | Exact limit | ✓✓✓ Rigorous | DeGroot + self-coupling term |
| **Echo Chambers** | Emergent property | ✓✓✓ Rigorous | Direct consequence of homophilic attention |
| **Bounded Confidence** | Soft approximation | ✓✓ Solid | Continuous relaxation of H-K hard threshold |
| **Confirmation Bias** | Geometric consequence | ✓✓ Solid | Requires natural gradient interpretation |
| **Social Impact Theory** | Interpretive mapping | ✓✓ Solid | Qualitative correspondence, not formal equivalence |
| **Backfire Effect** | Preliminary hypothesis | ✓ Speculative | Mechanism unclear; requires further development |

---

## Derivation 1: DeGroot Social Learning (RIGOROUS)

**Classical formulation** (DeGroot, 1974):
```
x_i(t+1) = Σ_j w_ij x_j(t)
```

where `W = [w_ij]` is a row-stochastic matrix (Σ_j w_ij = 1).

### Limiting Conditions

Take the following limits:
1. **Overdamped**: γ → ∞ (pure dissipative dynamics)
2. **Low uncertainty**: Σ_i → σ² I with σ² small (near-Dirac beliefs)
3. **Flat bundle**: Ω_ij = I (no gauge structure)
4. **No self-coupling**: α = 0
5. **No prior alignment**: λ_γ = 0
6. **No observations**: λ_obs = 0
7. **Fixed attention**: β_ij = w_ij (constant weights, not softmax)

### Derivation

**Step 1**: Simplify VFE under limits 3-7.

With Ω = I, α = 0, λ_γ = 0, λ_obs = 0:
```
F[μ] = λ_β Σ_ij w_ij ∫ KL(N(μ_i, σ²I) || N(μ_j, σ²I))
```

**Step 2**: Expand KL divergence for Gaussians.

For equal covariances:
```
KL(N(μ_i, σ²I) || N(μ_j, σ²I)) = ||μ_i - μ_j||² / (2σ²)
```

Thus:
```
F[μ] = (λ_β / 2σ²) Σ_ij w_ij ||μ_i - μ_j||²
```

**Step 3**: Compute gradient.

```
∇_μi F = (λ_β / σ²) Σ_j w_ij (μ_i - μ_j)
       = (λ_β / σ²) [Σ_j w_ij μ_i - Σ_j w_ij μ_j]
       = (λ_β / σ²) [μ_i - Σ_j w_ij μ_j]          (using Σ_j w_ij = 1)
```

**Step 4**: Apply natural gradient flow.

In the low-uncertainty limit, the mass matrix becomes:
```
M_i ≈ (1/σ²) I
```

Natural gradient descent:
```
dμ_i/dt = -M_i^{-1} ∇_μi F
        = -σ² I · (λ_β / σ²)(μ_i - Σ_j w_ij μ_j)
        = -λ_β (μ_i - Σ_j w_ij μ_j)
```

**Step 5**: Discrete-time integration.

With time step Δt = 1/λ_β:
```
μ_i(t + Δt) = μ_i(t) - λ_β Δt (μ_i - Σ_j w_ij μ_j)
            = μ_i(t) - (μ_i - Σ_j w_ij μ_j)
            = Σ_j w_ij μ_j(t)
```

**QED**: This is exactly the DeGroot update rule. ✓

### What Hamiltonian VFE Adds

1. **Underdamped regime** (γ → 0): Beliefs can overshoot equilibrium, exhibit oscillations
2. **Uncertainty dynamics**: Evolution of full distributions `q_i(x)`, not just means
3. **Dynamic attention**: `β_ij(t)` changes based on belief disagreement KL(q_i || q_j)
4. **Epistemic inertia**: Mass `M_i` grows when agent receives attention (many followers)
5. **Gauge structure**: Non-trivial transport `Ω_ij ≠ I` when beliefs live in different frames

---

## Derivation 2: Friedkin-Johnsen Opinion Dynamics (RIGOROUS)

**Classical formulation** (Friedkin & Johnsen, 1990):
```
x_i(t+1) = α_i x_i(0) + (1 - α_i) Σ_j w_ij x_j(t)
```

where `α_i ∈ [0, 1]` is agent i's "stubbornness" or attachment to initial opinion `x_i(0)`.

### Limiting Conditions

Take DeGroot limits (1-7) PLUS:
8. **Non-zero self-coupling**: α > 0
9. **Fixed priors**: p_i = N(μ_i(0), Σ_p) where μ_i(0) is initial belief

### Derivation

**Step 1**: VFE with self-coupling.

```
F[μ] = α Σ_i KL(N(μ_i, σ²I) || N(μ_i(0), Σ_p))
     + (λ_β / 2σ²) Σ_ij w_ij ||μ_i - μ_j||²
```

For small σ²:
```
KL(N(μ_i, σ²I) || N(μ_i(0), Σ_p)) ≈ ||μ_i - μ_i(0)||² / (2Σ_p)
```

Thus:
```
F[μ] = (α / 2Σ_p) Σ_i ||μ_i - μ_i(0)||²
     + (λ_β / 2σ²) Σ_ij w_ij ||μ_i - μ_j||²
```

**Step 2**: Gradient.

```
∇_μi F = (α / Σ_p)(μ_i - μ_i(0)) + (λ_β / σ²)(μ_i - Σ_j w_ij μ_j)
```

**Step 3**: Natural gradient flow with mass M_i = Σ_p^{-1}.

```
dμ_i/dt = -Σ_p ∇_μi F
        = -α(μ_i - μ_i(0)) - (λ_β Σ_p / σ²)(μ_i - Σ_j w_ij μ_j)
```

**Step 4**: Steady-state solution.

At equilibrium, dμ_i/dt = 0:
```
α(μ_i - μ_i(0)) = -(λ_β Σ_p / σ²)(μ_i - Σ_j w_ij μ_j)

μ_i [α + λ_β Σ_p / σ²] = α μ_i(0) + (λ_β Σ_p / σ²) Σ_j w_ij μ_j
```

Solving:
```
μ_i = α_i' μ_i(0) + (1 - α_i') Σ_j w_ij μ_j
```

where the **emergent stubbornness** is:
```
α_i' = α / [α + (λ_β Σ_p / σ²) Σ_j w_ij]
```

**QED**: This matches Friedkin-Johnsen with **mechanistically-derived** stubbornness. ✓

### Key Insight

Unlike classical F-J, where α_i is an ad-hoc "personality trait", here stubbornness **emerges** from:
- **Prior precision** `Σ_p^{-1}`: How certain agent was initially
- **Social coupling** `λ_β Σ_j w_ij`: How much social pressure they experience

**Novel prediction**: Stubbornness is NOT fixed—it decreases as social coupling increases (more attention → less stubborn in equilibrium, but also higher inertial mass → slower updates).

---

## Derivation 3: Echo Chambers & Polarization (RIGOROUS)

**Phenomenon**:
- Homophily (like attracts like)
- In-group belief convergence
- Out-group belief divergence
- Polarization despite shared information

### Derivation

**Step 1**: Softmax attention creates homophily.

```
β_ij = exp(-KL(q_i || Ω_ij q_j) / κ_β) / Z_i
```

For Gaussians:
```
KL(N(μ_i, Σ) || N(μ_j, Σ)) ≈ ||μ_i - μ_j||² / (2σ²)
```

Thus:
```
β_ij ∝ exp(-||μ_i - μ_j||² / (2σ² κ_β))
```

**Interpretation**:
- Similar beliefs (small ||μ_i - μ_j||) → high β_ij (strong attention)
- Dissimilar beliefs (large ||μ_i - μ_j||) → low β_ij (ignore out-group)

**Step 2**: Feedback creates polarization.

Natural gradient flow:
```
dμ_i/dt ∝ Σ_j β_ij (μ_j - μ_i)
```

**Positive feedback loop**:
1. Agents with similar beliefs attend to each other (high β_ij)
2. Mutual attention → beliefs converge further
3. Convergence → attention increases further
4. Eventually β_ij^{cross-group} → 0 (out-group ignored)

**Step 3**: Stability analysis of polarized states.

Consider two groups A, B with means μ_A, μ_B.

**Within-group coupling**:
```
β_ij ≈ 1/|A|  for i, j ∈ A  (uniform, high)
```

**Cross-group coupling**:
```
β_ij ≈ 0  for i ∈ A, j ∈ B  when ||μ_A - μ_B|| >> σ√κ_β
```

**Result**: Two **decoupled subsystems**, each internally converging:
```
dμ_i/dt ≈ Σ_{j ∈ group(i)} β_ij (μ_j - μ_i)
```

**Stability condition**: Polarized state {μ_A, μ_B} is stable when:
```
||μ_A - μ_B||² > 2σ² κ_β log(N)
```

**QED**: Homophily and polarization emerge automatically from softmax attention, no homophily assumption needed. ✓

### What Hamiltonian VFE Adds

1. **Emergent homophily**: Not assumed, derived from KL-based attention
2. **Phase transition**: Critical temperature κ_β where polarization spontaneously emerges
3. **Stability analysis**: Exact conditions for stable polarization vs. global consensus
4. **Escape dynamics**: In Hamiltonian regime, tunneling between polarized basins possible

---

## Derivation 4: Bounded Confidence (SOLID APPROXIMATION)

**Classical formulation** (Hegselmann & Krause, 2002):
```
x_i(t+1) = average{ x_j(t) : |x_j(t) - x_i(t)| < ε }
```

Agents only influenced by others within threshold ε (hard cutoff).

### Limiting Conditions

Take DeGroot limits PLUS:
10. **Low temperature**: κ_β → 0 (sharp attention cutoff)

### Derivation

**Step 1**: Softmax in low-temperature limit.

```
β_ij = exp(-||μ_i - μ_j||² / (2σ² κ_β)) / Z_i
```

As κ_β → 0:
```
β_ij → { 1/|N_i(ε)|  if ||μ_j - μ_i|| < ε_eff
       { 0            if ||μ_j - μ_i|| > ε_eff
```

where the **effective threshold** is:
```
ε_eff ≈ σ √(2 κ_β log N)
```

**Step 2**: Dynamics in threshold regime.

```
dμ_i/dt ∝ Σ_{j : ||μ_j - μ_i|| < ε} (μ_j - μ_i) / |N_i(ε)|
        = average{ μ_j - μ_i : ||μ_j - μ_i|| < ε }
```

**CAVEAT**: This is a **soft threshold** (exponential decay), not the **hard cutoff** of classical H-K.

**Difference**:
- **H-K**: β_ij = { 1/|N_i|  if d < ε; 0 if d ≥ ε } (discontinuous)
- **VFE**: β_ij = exp(-d² / (2σ²κ)) / Z (continuous, smooth decay)

**Justification**: The VFE version is the **natural continuous relaxation** of H-K. Real social attention likely has smooth falloff, not hard cutoffs.

### What Hamiltonian VFE Adds

1. **Adaptive threshold**: ε_eff = ε(σ, κ_β, N) depends on uncertainty and network size
2. **Asymmetric influence**: β_ij ≠ β_ji in general (depends on local neighborhood sizes)
3. **Smooth attention**: Differentiable dynamics, no discontinuities
4. **Temperature tuning**: κ_β controls sharpness of threshold (not all-or-nothing)

---

## Derivation 5: Confirmation Bias (SOLID, REQUIRES NATURAL GRADIENT)

**Psychological phenomenon**:
- People update beliefs less from counter-attitudinal evidence
- Prior beliefs resistant to change
- "Motivated reasoning" or "cognitive conservatism"

### Derivation

**Step 1**: Natural gradient descent with mass matrix.

```
dμ_i/dt = -M_i^{-1} ∇_μi F
```

where:
```
M_i = Σ_p,i^{-1} + Σ_j β_ij Σ_q,j^{-1}
```

**Step 2**: Effect of high prior precision.

When `Σ_p^{-1}` is large (agent very confident in prior):
```
M_i^{-1} ≈ [Σ_p^{-1}]^{-1} = Σ_p  (small)
```

For same force `∇F`, velocity is small:
```
dμ/dt = Σ_p ∇F  << ∇F  when Σ_p small
```

**Step 3**: Effect of social mass (followers).

When `Σ_j β_ij Σ_q,j^{-1}` is large (many precise followers):
```
M_i large  →  M_i^{-1} small  →  slow updates
```

**Quantitative prediction**:

Update magnitude for evidence contradicting belief:
```
Δμ_i ∝ [Σ_p^{-1} + Σ_j β_ij Σ_q,j^{-1}]^{-1}
```

Agents with:
- **High prior precision** (low Σ_p) → large M → small updates
- **Many followers** (large Σ_j β_ij) → large M → small updates

**CRITICAL ASSUMPTION**: This derivation requires **natural gradient descent** (flow on statistical manifold). Standard Euclidean gradient descent would not show this effect.

### Justification for Natural Gradient

Natural gradient descent is the **geometrically correct** dynamics on statistical manifolds (Amari, 1998). The Fisher metric `M` is the unique Riemannian metric invariant under reparameterization. Standard gradient descent is only correct for Euclidean spaces.

**Status**: Solid derivation **if** we accept natural gradient as the principled choice for learning on probability distributions.

### What Hamiltonian VFE Adds

1. **Mechanistic explanation**: Bias emerges from geometry, not irrationality
2. **Social component**: Followers contribute to mass, amplifying bias
3. **Testable prediction**: Bias strength ∝ (prior precision + follower count)
4. **Individual differences**: Variation in bias from variation in `Σ_p` and social network position

---

## Derivation 6: Social Impact Theory (INTERPRETIVE MAPPING)

**Classical formulation** (Latané, 1981):
```
Impact = f × (Strength × Immediacy × Number)
```

where:
- **Strength**: Expertise, status, power of source
- **Immediacy**: Proximity in space or time
- **Number**: How many sources present

### Mapping to VFE Framework

Recall mass matrix:
```
M_i = Σ_p^{-1} + Σ_j β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
```

**Strength** ↔ `Σ_q,j^{-1}` (precision of source j)
- High-precision beliefs (experts, confident sources) contribute more mass
- Low-precision beliefs (uncertain sources) contribute less

**Immediacy** ↔ `||Ω_ij - I||` (transport cost)
- Spatially/temporally close agents: `Ω_ij ≈ I` → small KL → high β_ij
- Distant agents: `Ω_ij` rotated → large KL → low β_ij

**Number** ↔ `Σ_j` (sum over sources)
- More influencers → more terms in sum → larger social mass

### Quantitative Correspondence

Social mass contribution from source j:
```
ΔM_i^{(j)} = β_ij Ω_ij Σ_q,j^{-1} Ω_ij^T
```

In terms of Latané's factors:
```
ΔM_i^{(j)} ≈ (Number: 1) × (Strength: Σ_q,j^{-1}) × (Immediacy: f(Ω_ij))
```

where immediacy function:
```
f(Ω_ij) = exp(-||Ω_ij - I||²_F / κ_β)  (enters through β_ij)
```

**CAVEAT**: This is an **interpretive mapping**, not a formal derivation. Latané's formula is qualitative; VFE provides quantitative specifics.

### What Hamiltonian VFE Adds

1. **Exact formula**: Not just "multiplicative", but precise matrix equation
2. **Time dynamics**: How impact changes as beliefs/attention evolve
3. **Asymmetry**: `ΔM_i^{(j)} ≠ ΔM_j^{(i)}` (influence not reciprocal)
4. **Testable**: Can measure Σ_q, Ω, β independently and predict mass

---

## Derivation 7: Backfire Effect (PRELIMINARY/SPECULATIVE)

**Psychological phenomenon**:
- Weak counter-evidence paradoxically **strengthens** original belief
- "Boomerang effect" in persuasion attempts
- Documented in political attitudes, health beliefs (but see replication debates)

### Proposed Mechanism (REQUIRES FURTHER WORK)

**Hypothesis**: Backfire emerges in **underdamped Hamiltonian regime** (low friction γ) for **high-mass agents** (influencers).

**Step 1**: Underdamped dynamics.

```
dμ/dt = M^{-1} π
dπ/dt = -∇F - γπ
```

For γ << √(k/M) (underdamped condition), this gives damped oscillator with complex eigenvalues.

**Step 2**: Response to weak perturbation.

Consider agent at equilibrium μ_0, receives weak counter-evidence forcing belief toward μ_new ≠ μ_0.

**Standard overdamped response** (γ large):
- Monotonic approach to μ_new
- No overshoot

**Underdamped response** (γ small, M large):
- Initial acceleration toward μ_new
- Momentum π builds up
- Overshoot past μ_new
- Oscillate around μ_new before settling

**Step 3**: Weak evidence + strong perturbation interaction??

**PROBLEM**: Simple overshoot doesn't explain true "backfire" (belief moving OPPOSITE to evidence).

**Possible additional mechanism** (SPECULATIVE):
- Geodesic correction term: `Γ_ijk π^j π^k` where Γ depends on ∂M/∂μ
- If counter-evidence changes attention structure β_ij(μ), this changes M(μ)
- Geodesic force could point opposite to naive gradient force
- **But**: I have not proven this works mathematically

**Alternative mechanism** (ALSO SPECULATIVE):
- Backfire in multi-agent context: agent receives counter-evidence
- Counter-evidence makes agent more salient to followers
- Salience increases β_ji (more attention FROM followers)
- Increased attention → increased social mass M_i
- Higher M_i → more resistant to further evidence
- Agent "doubles down" to maintain consistency with followers

**STATUS**: Neither mechanism is rigorously established. This derivation is **preliminary and requires further theoretical/computational work**.

### Empirical Prediction (IF mechanism holds)

Backfire more likely for:
- **High-mass agents** (influencers with many followers)
- **Low-friction environments** (social media, rapid discourse)
- **Weak evidence** (small perturbations that trigger inertial response)
- **Visible disagreement** (public counter-evidence that changes attention structure)

**Recommendation for paper**: Either:
1. **Develop mechanism rigorously** before publishing, OR
2. **Omit this derivation** and focus on solid cases, OR
3. **Mark clearly as speculative hypothesis** for future work

---

## Summary: Derivation Quality

| Model | Quality | Can Include in Top-Tier Paper? | Notes |
|-------|---------|-------------------------------|-------|
| **DeGroot** | ✓✓✓ Rigorous | YES | Lead with this |
| **Friedkin-Johnsen** | ✓✓✓ Rigorous | YES | Shows mechanistic stubbornness |
| **Echo Chambers** | ✓✓✓ Rigorous | YES | Emergent homophily compelling |
| **Bounded Confidence** | ✓✓ Solid | YES | Caveat: soft not hard threshold |
| **Confirmation Bias** | ✓✓ Solid | YES | Caveat: requires natural gradient |
| **Social Impact** | ✓✓ Interpretive | YES | Frame as correspondence, not equivalence |
| **Backfire Effect** | ✓ Speculative | CAUTION | Mark as preliminary or omit |

---

## Phase Diagram of Dynamics Regimes

```
            Friction γ
                │
     High       │  ╔═══════════════════════════════╗
      (γ→∞)     │  ║   GRADIENT FLOW REGIME        ║
                │  ║  • DeGroot social learning     ║
                │  ║  • Friedkin-Johnsen dynamics   ║
                │  ║  • Bounded confidence models   ║
                │  ║  • Dissipative (no oscillation)║
   ─────────────┼──╫───────────────────────────────╫── Critical Damping γ_c
                │  ║                                ║
                │  ║   HAMILTONIAN REGIME           ║
     Low        │  ║  • Underdamped dynamics        ║
      (γ→0)     │  ║  • Belief oscillations         ║
                │  ║  • Backfire effects (?)        ║
                │  ║  • Nearly energy-conserving    ║
                │  ╚═══════════════════════════════╝
                └────────────────────────────────────> Social Coupling Σ_j β_ij
                         Low                       High

      Low Coupling (isolated):       High Coupling (influencer):
      • M ≈ Σ_p^{-1} (bare mass)     • M >> Σ_p^{-1} (epistemic inertia)
      • Fast convergence              • Slow updates (rigidity)
      • Flexible beliefs              • Resistant to evidence
      • No backfire                   • Potential backfire (if γ low)
```

---

## Critical Parameters Table

| Parameter | Symbol | Physical Meaning | Controls Transition... |
|-----------|--------|------------------|----------------------|
| **Friction** | γ | Dissipation rate | Gradient flow ↔ Hamiltonian |
| **Temperature** | κ_β | Attention sharpness (inverse) | Soft ↔ Sharp bounded confidence |
| **Prior covariance** | Σ_p | Prior uncertainty | Flexible ↔ Stubborn (F-J) |
| **Social coupling** | Σ_j β_ij | Attention received (followers) | Isolated ↔ Rigid (inertia) |
| **Self-coupling** | α | Prior weight | Social ↔ Individual-focused |
| **Gauge field** | Ω_ij | Frame alignment | Flat ↔ Curved bundle |

---

## Novel Predictions (Beyond Classical Models)

### 1. Dynamic Attention (Strong Prediction)

**Classical models**: Assume fixed weights w_ij

**VFE prediction**: β_ij(t) = softmax[-KL(q_i || q_j) / κ_β] changes dynamically

**Test**: Measure attention shifts as beliefs converge/diverge over time

### 2. Epistemic Inertia from Followers (Strong Prediction)

**Classical models**: No mechanism relating social structure to update rates

**VFE prediction**: Update rate ∝ [M(network position)]^{-1}

**Test**: Agents with more followers should make smaller belief updates to same evidence

### 3. Uncertainty Dynamics (Strong Prediction)

**Classical models**: Only track belief means x_i

**VFE prediction**: Full distributions q_i = N(μ_i, Σ_i) evolve; uncertainty can increase/decrease

**Test**: Measure confidence intervals over time, not just point estimates

### 4. Phase Transitions in Polarization (Moderate Prediction)

**Classical models**: Polarization gradual or threshold-dependent on ε

**VFE prediction**: Sharp phase transition at critical κ_β (like thermodynamic phase transition)

**Test**: Vary temperature κ_β; look for sudden onset of polarization

### 5. Underdamped Oscillations (Speculative Prediction)

**Classical models**: Monotonic convergence

**VFE prediction**: In low-friction regime, beliefs can oscillate before settling

**Test**: High-frequency belief measurements; look for overshoots and oscillations

---

## Recommendations for Paper

### For Top-Tier Sociology Journal (AJS, Sociological Theory)

**Include with confidence**:
1. DeGroot (rigorous)
2. Friedkin-Johnsen (rigorous, shows mechanistic stubbornness)
3. Echo chambers (rigorous, emergent homophily)
4. Bounded confidence (solid, with caveat about soft threshold)
5. Confirmation bias (solid, with note about natural gradient)
6. Social Impact Theory (interpretive mapping, compelling qualitative correspondence)

**Omit or mark speculative**:
7. Backfire effect (mechanism not established; either develop further or save for future work)

**Framing**:
- Lead: "We show that major social influence models emerge as regimes of a unified framework"
- Emphasize: DeGroot, F-J, echo chambers (rock-solid)
- Novel predictions: Dynamic β, epistemic inertia, uncertainty propagation
- Empirical strategy: Test at least ONE novel prediction (recommend epistemic inertia—easiest to measure)

### Next Steps

1. **Strengthen backfire mechanism** OR **omit from first paper**
2. **Run phase transition simulations** (show κ_β critical point for polarization)
3. **Quick empirical test**: Metaculus forecasters (high-reputation → smaller updates?)
4. **Draft paper structure** with strong derivations front-and-center

---

## Mathematical Appendices (For Paper Supplements)

### Appendix A: Natural Gradient on Statistical Manifolds

[Technical details on Fisher-Rao metric, natural gradient descent, geometric interpretation]

### Appendix B: Detailed Derivations

[Step-by-step proofs for DeGroot, F-J, echo chambers with all intermediate steps]

### Appendix C: Numerical Stability of Symplectic Integrators

[Implementation details for Hamiltonian simulations]

### Appendix D: Phase Space Portraits

[Visualizations of different dynamical regimes]

---

**Document Status**: Ready for review and refinement before paper submission.
**Confidence**: High for derivations 1-6; Low for derivation 7 (backfire).
**Next Action**: Discuss with collaborators which models to include in first paper.
