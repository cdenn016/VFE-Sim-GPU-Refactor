# Reddit Post Draft: Inertia of Belief Paradigm

## Suggested Subreddits
- r/sociology
- r/psychology
- r/AcademicPsychology
- r/cogsci (cognitive science)

---

## Title Options

**Option A (Psychology focus):**
> A mathematical framework showing beliefs have genuine inertia—explaining confirmation bias, belief perseverance, and oscillation as mechanical phenomena rather than cognitive "flaws"

**Option B (Sociology focus):**
> New framework derives DeGroot, Friedkin-Johnsen, and bounded confidence models as limiting cases of a single equation—and predicts why influential people become resistant to change

**Option C (Unified/Provocative):**
> What if confirmation bias isn't irrational? A Hamiltonian mechanics approach to belief dynamics

---

## Post Body

I've been developing a theoretical framework that reframes how we think about belief change, and I'd love feedback from this community (and potential collaborators with relevant data).

### The Core Idea

Beliefs possess **genuine inertia**—resistance to change proportional to their precision/confidence—analogous to how physical mass resists acceleration. This isn't just a metaphor. The mathematics of information geometry shows that the Fisher Information Metric (which measures statistical distinguishability between beliefs) is *exactly* equivalent to an inertial mass tensor.

**The key equation:**

```
M = Λ_prior + Λ_observation + Σ(attention_out × neighbor_precision) + Σ(attention_in × own_precision)
```

Where epistemic mass M comes from four sources:
1. **Prior precision** — How confident you already are
2. **Observation precision** — How grounded in direct experience
3. **Outgoing attention** — Beliefs you attend to in others
4. **Incoming attention** — How many people attend to YOUR beliefs

### Why This Matters for Psychology

Instead of treating cognitive biases as separate phenomena requiring separate explanations, they emerge naturally from a single dynamical equation:

**M·μ̈ + γ·μ̇ + ∇F = 0**

(Mass × acceleration + damping × velocity + gradient of free energy = 0)

This is a damped harmonic oscillator. Depending on parameters, you get three regimes:

| Regime | Behavior | Psychology |
|--------|----------|-----------|
| **Overdamped** | Smooth approach to equilibrium | Standard Bayesian updating |
| **Critically damped** | Fastest equilibration | Optimal learning |
| **Underdamped** | Oscillation, overshoot | Belief oscillation, attitude swings |

**Specific predictions:**

- **Confirmation bias** = stopping distance. A belief with momentum v₀ traveling against force f stops after distance d = Mv₀²/2f. Twice the precision → twice the stopping distance. Not irrationality—mechanics.

- **Belief perseverance** = decay time. False beliefs persist with characteristic time τ = M/γ. High-confidence misinformation takes *proportionally longer* to correct, even with identical evidence exposure.

- **The backfire effect** = oscillatory overshoot. Strong counter-evidence on high-confidence beliefs can push them *past* equilibrium and back toward the original position.

- **Attitude oscillation** (Kaplowitz & Fink's work) emerges naturally in the underdamped regime—something first-order Bayesian models cannot produce.

### Why This Matters for Sociology

The framework derives classical opinion dynamics models as limiting cases:

- **DeGroot (1974)** — Fixed attention weights, no self-coupling
- **Friedkin-Johnsen (1990)** — Add stubbornness (prior precision)
- **Bounded confidence models** — Uncertainty bounds create natural attention cutoffs
- **Echo chambers** — Softmax attention over KL-divergence creates endogenous homophily

But here's the interesting part—the **incoming attention term**:

```
Social mass contribution = Σ_j (attention from j) × (your precision)
```

**The more people attend to your beliefs, the more epistemic mass YOU accumulate, making you harder to persuade.**

This predicts that influential people become cognitively isolated not through moral failure but through *geometric necessity*. Power literally weighs down belief updating. Leaders become less responsive to evidence as their following grows. (Echoes of "power corrupts," but with a mathematical mechanism.)

### Falsifiable Predictions

This framework makes predictions that differ from standard first-order models:

1. **Belief trajectories should oscillate** (not monotonically converge) when high-confidence beliefs encounter strong counter-evidence

2. **Decay times should scale with initial precision** — measure how long false beliefs persist as a function of initial confidence; should be proportional

3. **Resonant persuasion** — Periodic messaging at frequency ω = √(K/M) should produce maximum belief change. Off-resonance messaging is wasted.

4. **Asymmetric social updating** — In deliberation, low-precision agents should shift more than high-precision agents, even with symmetric information exchange

5. **Attention-induced rigidity** — Experimentally manipulating how much attention an agent receives should change their update magnitude (more attention → smaller updates)

### Connection to Existing Work

This builds on and connects:
- Friston's Free Energy Principle / Active Inference (overdamped limit)
- Information geometry (Amari, Ay)
- Predictive processing (Clark)
- Social impact theory (Latané)
- Opinion dynamics (Hegselmann, Deffuant, Friedkin)

The novelty is showing these are all limiting cases of the same Hamiltonian system, and that the *underdamped* regime—largely unexplored—may explain phenomena that don't fit first-order models.

---

## Looking for Data / Collaboration

I'm particularly interested in:

1. **Longitudinal belief tracking data** — Studies that measured belief trajectories over time (not just before/after). Ideal: multiple timepoints to detect oscillation vs. monotonic convergence.

2. **Social network + belief data** — Any datasets combining network position (especially attention/influence asymmetries) with belief updating behavior.

3. **Deliberation/negotiation studies** — Especially any that tracked belief changes at multiple points during discussion.

4. **Forecasting/prediction data** — Platforms like Metaculus, PredictIt, Manifold where we can observe how "epistemic mass" (reputation, followers, track record) correlates with update magnitude.

5. **Misinformation correction studies** — Especially any with multiple follow-up measurements that might reveal continued influence timing.

If you have access to relevant datasets or are interested in collaboration, I'd love to connect. The framework makes specific quantitative predictions (decay times proportional to precision, oscillation frequencies, resonance effects) that could be tested with the right data.

---

## TL;DR

Beliefs resist change like mass resists acceleration—and the math isn't metaphorical, it's exact (Fisher information = inertial mass). This single framework:
- Derives confirmation bias as "stopping distance"
- Explains belief perseverance via decay time τ = M/γ
- Predicts belief oscillation (not in standard Bayesian models)
- Shows why influential people become resistant to updating
- Unifies 6+ classical opinion dynamics models as limiting cases

Looking for collaborators with longitudinal belief data, social network studies, or interest in testing these predictions.

---

## Notes for Posting

**For r/psychology / r/AcademicPsychology:**
- Emphasize the cognitive bias angle
- Lead with "confirmation bias isn't irrational"
- Reference Kaplowitz & Fink on attitude oscillation, Lewandowsky on continued influence

**For r/sociology:**
- Emphasize the opinion dynamics unification
- Lead with "derives DeGroot and Friedkin-Johnsen as limits"
- Reference network position → rigidity prediction

**For r/cogsci:**
- Can be more technical
- Emphasize connection to predictive processing and Free Energy Principle
- Note this extends Friston to underdamped regime

---

## Potential Questions to Anticipate

**Q: How is this different from just saying "confident people are stubborn"?**
A: It's quantitative. The framework predicts *exactly* how much more stubborn (τ scales linearly with precision), predicts oscillation (not just resistance), and predicts that social attention independently increases mass.

**Q: Isn't this just curve-fitting?**
A: No—the mass formula is derived from first principles (variational free energy minimization on statistical manifolds). The four-term decomposition isn't fit to data; it falls out of the math.

**Q: What about motivated reasoning?**
A: Could be modeled as anisotropic damping (direction-dependent γ) or asymmetric evidence weighting. The framework accommodates this as a special case.

**Q: How do you measure "epistemic mass" empirically?**
A: Through proxies: stated confidence, behavioral consistency, prediction track record, social influence metrics. The framework predicts these should all correlate with update resistance.
