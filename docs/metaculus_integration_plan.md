# Integration Plan: Metaculus Empirics + Theory Papers

## Overview

This document outlines the strategy for integrating Metaculus epistemic inertia results with theoretical manuscripts for publication in top-tier journals.

## Current Status

### Completed Work

1. **Psychology Paper (`papers/psych/belief_inertia.tex`)**
   - ✓ Complete 4-term mass matrix derivation (Eq. 266-268)
   - ✓ Physical interpretations for all components
   - ✓ Testable predictions formulated
   - ✓ Nassar helicopter task validation (overdamped regime)
   - ✗ **Desk rejected** by PPS, Entropy, JMP

2. **Sociology Paper (`docs/derivations_sociology_manuscript.tex`)**
   - ✓ Shows 6 classical models as limiting cases
   - ✓ Rigorous Proposition/Proof structure
   - ✓ Detailed LaTeX derivations
   - ⏳ **Not yet submitted**

3. **Metaculus Experiment (`experiments/metaculus_epistemic_inertia/`)**
   - ✓ Complete analysis pipeline (fetch, detect, analyze, visualize)
   - ✓ Tests epistemic inertia directly
   - ✓ Addresses empirical gap from psych paper
   - ⏳ **Data collection not yet run**

4. **Code Implementation (`agent/hamiltonian_trainer.py`)**
   - ✓ Complete 4-term mass matrix (Dec 2024 update)
   - ✓ Observation precision Λ_o added
   - ✓ Incoming attention term added

## The Empirical Gap

### Why Psychology Paper Was Rejected

From limitations section (line 977):

> "Our primary empirical test involved a task designed to elicit **overdamped dynamics**. Direct observation of underdamped belief oscillation, precision-scaled relaxation, or resonant persuasion **remains for future experimental work**."

**The problem**: Nassar task correctly showed γ >> 2√(KM) → overdamped → reduces to gradient descent.

**What reviewers wanted**:
- Evidence of unique predictions (oscillations, inertia, overshooting)
- Data showing second-order effects, not first-order limits
- Regime where social mass dominates

## How Metaculus Fills the Gap

### Direct Test of Epistemic Inertia

**Hypothesis**: High track_record → High mass → Smaller |Δp|

**Why this works**:
1. **Social setting**: β_ik, β_ji ≠ 0 (unlike Nassar task)
2. **High reputation ~ high mass** from ALL 4 components:
   - Λ_p ↑: Experience → sharper priors
   - Λ_o ↑: Activity → more observations
   - Σ_k β_ik ↑: Engagement → following many sources
   - Σ_j β_ji ↑: Influence → being followed
3. **Regime where theory matters**: Social mass can exceed Λ_p + Λ_o

### Testable Predictions

| Prediction | Metaculus Test | Expected Result |
|------------|----------------|-----------------|
| High mass → small updates | track_record ~ \|Δp\| | β < 0, p < 0.05 |
| Effect scales with info content | track_record × info ~ \|Δp\| | interaction β < 0 |
| Larger for experienced users | Subsample (>10 predictions) | Stronger effect |
| Larger for big events | Subsample (>10 forecasters) | Stronger effect |

## Publication Strategies

### Strategy 1: Resubmit Psychology Paper

**Target**: Cognitive Science, Psychological Review, TICS

**Changes needed**:
1. Add Metaculus empirical section (new Section 4.2)
2. Update abstract to highlight social validation
3. Revise limitations: Remove "future work" caveat
4. Add 3 main Metaculus figures to manuscript

**Structure**:
```
1. Introduction (unchanged)
2. Mathematical Framework (unchanged)
3. Results
   3.1. Cognitive Phenomena (unchanged)
   3.2. Nassar Task (overdamped validation)
   3.3. **[NEW] Metaculus Forecasting (epistemic inertia)**
4. Discussion (updated with Metaculus interpretation)
```

**Pros**:
- Addresses reviewer concerns directly
- Strong theory + now has empirical support
- Natural fit for cognitive science

**Cons**:
- Already desk rejected by 3 journals
- May be stigmatized in psychology community
- Still might want "lab" data rather than observational

### Strategy 2: Submit to Sociology Journals

**Target**: American Journal of Sociology, Sociological Theory

**Approach**: Combine sociology derivations + Metaculus empirics

**Structure**:
```
1. Introduction
   - Why belief dynamics matter for sociology
   - Classical models lack inertia

2. Theoretical Framework
   - VFE as unifying principle
   - Mass matrix derivation
   - Classical models as limits (from sociology paper)

3. Empirical Validation
   - Metaculus experiment design
   - Results: Epistemic inertia confirmed
   - Figures 1-3

4. Implications
   - Status → rigidity
   - Influence → reduced flexibility
   - Echo chambers as high-mass clusters

5. Discussion
```

**Pros**:
- Fresh audience, no rejection history
- Unifying framework narrative (like Coleman's Foundations)
- Empirics grounded in social process (reputation, influence)
- Classical models as limits = strong contribution

**Cons**:
- Sociology journals want deep engagement with canon
- May find math intimidating
- Journal of Mathematical Sociology safer but lower impact

### Strategy 3: Two-Paper Strategy

**Paper A (Theory)**: Network Science, Journal of Complex Networks
- Complete mass formula
- Classical models as limits
- Simulation results from codebase
- Target: Methods/theory audience

**Paper B (Empirics)**: Computational Social Science journal
- Metaculus analysis
- Focus on forecasting applications
- Lighter on theory, heavy on data
- Target: Applied audience

**Pros**:
- Separates theory from empirics
- Each paper focused and digestible
- Increases total publication count

**Cons**:
- Neither paper has full story
- Theory without empirics is weak
- Empirics without theory is atheoretical

## Recommended Path

### Phase 1: Run Metaculus Experiment (Weeks 1-2)

1. **Data collection** (`python fetch_data.py`)
   - ~100 questions, 50+ predictions each
   - Expected runtime: 30-60 minutes
   - Output: ~5,000-10,000 update events

2. **Event detection** (`python detect_events.py`)
   - DBSCAN clustering (6hr windows, 5+ forecasters)
   - Expected: 50-100 significant events
   - Output: Regression-ready panel

3. **Statistical analysis** (`python analyze_data.py`)
   - 4 regression models
   - Robustness checks
   - Effect size calculations
   - Output: LaTeX table + JSON summary

4. **Visualization** (`python visualize_results.py`)
   - 5 publication figures (PDFs)
   - Main effect, distributions, interaction, examples, quantiles

### Phase 2: Interpretation (Week 3)

**Key questions to answer**:

1. **Did we find epistemic inertia?**
   - Is β_1 < 0 and significant (p < 0.05)?
   - What's the effect size? (% change per SD track_record)
   - Does it survive robustness checks?

2. **What's the mechanism?**
   - Is it interaction with info content (β_3 < 0)?
   - Stronger for experienced forecasters?
   - Stronger for large events?

3. **What do null results mean?**
   - If β_1 ≈ 0: Track record ≠ mass (proxy failure)
   - If β_1 > 0: Sophistication effect dominates inertia
   - Still informative! Constrains theory

### Phase 3: Manuscript Integration (Week 4)

**If results positive (β_1 < 0, p < 0.05)**:

**Option A**: Resubmit psychology paper
- Add Section 4.2 "Empirical Validation: Metaculus Forecasting"
- Update abstract: "...validated using public forecasting data"
- Add Figures: Main effect + distribution + interaction
- Target: Psychological Review (prestigious, interdisciplinary)

**Option B**: Submit to sociology
- Merge sociology derivations + Metaculus empirics
- Frame as unifying framework (Coleman-style)
- Emphasize: Status → inertia, influence → rigidity
- Target: AJS or Sociological Theory

**Option C**: Split into two papers
- Theory → Journal of Mathematical Sociology
- Empirics → Computational Social Science

**If results null or opposite (β_1 ≥ 0)**:

- Still publishable! Negative results matter
- Discuss: Proxy validity, regime boundaries, sophistication effects
- Target: PLOS ONE, Royal Society Open Science (null-friendly)
- Frame as: "Testing boundary conditions of epistemic inertia theory"

## Timeline

```
Week 1: Run Metaculus pipeline, get results
Week 2: Interpret findings, compute effect sizes
Week 3: Draft empirical section for target journal
Week 4: Integrate with theory, submit manuscript

Total: 4 weeks to submission-ready
```

## Critical Success Factors

### For Positive Results

1. **Effect size must be meaningful**
   - Statistical significance (p < 0.05) necessary but not sufficient
   - Need practical significance: >5% change per SD track_record
   - Robustness across subsamples

2. **Mechanism must be clear**
   - Not just "high reputation → smaller updates"
   - Show it's driven by ALL 4 mass components
   - Interaction with event characteristics

3. **Alternative explanations addressed**
   - Sophistication: High-reputation = better calibrated?
   - Selection: Do high-reputation users attend to different events?
   - Endogeneity: Does mass cause inertia or vice versa?

### For Null Results

1. **Proxy validity check**
   - Does track_record correlate with activity, engagement?
   - Is there variance in update magnitudes to explain?
   - Are events actually "news shocks"?

2. **Regime analysis**
   - Is Metaculus also in overdamped regime?
   - Compute estimated γ/√(KM) from data
   - Maybe need even MORE social coupling

3. **Theory refinement**
   - If not track_record, what DOES predict inertia?
   - Can we directly measure β_ij from platform data?
   - Alternative operationalizations

## Next Steps

### Immediate (Today)

1. ✓ Complete mass matrix implementation (DONE)
2. ✓ Update Metaculus README (DONE)
3. ✓ Create integration plan (THIS DOCUMENT)

### Short-term (This Week)

1. **Run Metaculus pipeline**
   ```bash
   cd experiments/metaculus_epistemic_inertia
   python run_pipeline.py
   ```

2. **Analyze results**
   - Check regression tables
   - Examine figures
   - Compute effect sizes

3. **Decide on publication strategy**
   - Based on results strength
   - Based on effect interpretation
   - Based on target audience preference

### Medium-term (Next Month)

1. **Draft empirical section**
   - Methods: Metaculus platform, data, measures
   - Results: Regression tables + figures
   - Interpretation: Epistemic inertia confirmed/refined/rejected

2. **Integrate with theory manuscript**
   - Choose target journal
   - Adapt framing and emphasis
   - Write cohesive narrative

3. **Submit for publication**
   - Psychology: Psych Review, Cog Sci
   - Sociology: AJS, Soc Theory, JMS
   - Interdisciplinary: Proc Nat Acad Sci, Sci Adv

## Contingency Plans

### If Metaculus Shows Weak/Null Effect

**Plan A: Refine proxy**
- Get comment/discussion data → infer β_ij directly
- Use peer_score instead of track_record
- Construct composite mass index

**Plan B: Find different data**
- Twitter/X: Follower count + belief updates (polls, predictions)
- Reddit: Karma + opinion changes in CMV (Change My View)
- Economic forecasters: Survey of Professional Forecasters

**Plan C: Simulation paper**
- Use codebase to demonstrate regimes
- Show when epistemic inertia dominates
- Propose experimental designs for lab validation

### If Reviewers Still Skeptical

**Strengthening moves**:
1. **Robustness**: Add quantile regression, instrumental variables
2. **Mechanism**: Mediation analysis for 4 mass components
3. **Prediction**: Out-of-sample forecasting of update magnitudes
4. **Comparison**: Benchmark against first-order models (AIC/BIC)

## Conclusion

The Metaculus experiment represents a critical test of the epistemic inertia framework. Success provides the missing empirical validation needed for publication in top-tier journals. Null results constrain theory and motivate refinement. Either outcome advances the research program.

**Key insight**: We now have COMPLETE theory (4-term mass) + TARGETED empirics (Metaculus) + SOLID code (updated hamiltonian_trainer.py). The pieces are in place for impactful publication.

**Decision point**: Week 1 results will determine optimal publication strategy. Be ready to adapt based on findings.
