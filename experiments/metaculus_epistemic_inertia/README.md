# Metaculus Epistemic Inertia Experiment

Empirical test of the **epistemic inertia** prediction from Hamiltonian Variational Free Energy (VFE) using public forecasting data from Metaculus.

## Theoretical Background

### Complete Epistemic Mass Formula

The Hamiltonian VFE framework derives belief dynamics from the VFE Hessian. The **effective mass matrix** has FOUR components:

```
M_i = Λ_p,i + Λ_o,i + Σ_k β_ik Λ̃_q,k + Σ_j β_ji Λ_q,i
```

Expanding explicitly:
```
M_i = Σ_p,i^{-1}                              [1. Prior precision]
    + R_obs,i^{-1}                            [2. Observation precision]
    + Σ_k β_ik Ω_ik Σ_q,k^{-1} Ω_ik^T         [3. Outgoing attention: i→k]
    + (Σ_j β_ji) Σ_q,i^{-1}                   [4. Incoming attention: j→i]
```

**Physical Interpretations**:

1. **Prior precision (Λ_p)**: Confident priors resist change
2. **Observation precision (Λ_o)**: High-quality data anchors beliefs (counterintuitive!)
3. **Outgoing attention**: Attending to confident neighbors → inheriting their rigidity
4. **Incoming attention**: Being watched by others → influence costs flexibility

### Core Prediction

**Epistemic mass increases through multiple channels**

High-reputation agents develop inertia via:
1. ✓ Sharper priors from experience (Λ_p ↑)
2. ✓ More observations accumulated (Λ_o ↑)
3. ✓ Following many sources (Σ_k β_ik ↑)
4. ✓ Being followed by others (Σ_j β_ji ↑)

This predicts:
- Smaller updates in response to evidence
- Delayed responses to information shocks
- Belief rigidity proportional to reputation

**Key insight**: Track record proxies for TOTAL mass, not just one component!

## Empirical Strategy

### Data Source: Metaculus

[Metaculus](https://www.metaculus.com) is a forecasting platform where users make probabilistic predictions about future events.

**Why Metaculus?**
- ✓ Sequential probability updates (clean belief dynamics)
- ✓ Timestamped predictions (temporal resolution)
- ✓ Track record scores (proxy for attention/reputation)
- ✓ Public API (reproducible data collection)
- ✓ Resolved questions (ground truth outcomes)

### Operationalization

| Theory | Empirical Proxy |
|--------|----------------|
| Attention/Followers (β_ji) | Track record score |
| Belief update (Δμ) | Change in probability |
| Information shock | Temporal cluster of updates (news event) |
| Epistemic inertia | Negative correlation: Track record → Update magnitude |

### Hypothesis

**H1 (Main Effect)**: High track record forecasters make smaller updates
```
|Δp_it| = β_0 + β_1 · track_record_i + controls + ε_it
Prediction: β_1 < 0
```

**H2 (Interaction)**: Epistemic inertia increases with information content
```
|Δp_it| = β_0 + β_1 · track_record_i + β_2 · info_content_t
         + β_3 · (track_record_i × info_content_t) + controls + ε_it
Prediction: β_3 < 0
```

## Pipeline Architecture

### 1. Data Collection (`fetch_data.py`)

**Inputs**: Metaculus API
**Outputs**:
- `questions_TIMESTAMP.csv`: Question metadata
- `predictions_TIMESTAMP.csv`: All prediction updates
- `users_TIMESTAMP.csv`: Forecaster statistics
- `updates_TIMESTAMP.csv`: Computed update magnitudes

**Key Functions**:
- `fetch_questions()`: Get resolved binary questions
- `fetch_prediction_timeseries()`: Get all updates for each question
- `fetch_user_stats()`: Get forecaster track records
- `compute_updates()`: Calculate |Δp_t - p_{t-1}|

### 2. Event Detection (`detect_events.py`)

**Inputs**: Updates time series
**Outputs**:
- `labeled_updates_TIMESTAMP.csv`: Updates with event labels
- `event_summary_TIMESTAMP.csv`: Event metadata
- `significant_events_TIMESTAMP.csv`: Filtered events
- `regression_panel_TIMESTAMP.csv`: Analysis-ready dataset

**Methods**:
- **DBSCAN clustering**: Temporal density-based event detection
  - `eps_hours=6.0`: Updates within 6 hours belong to same event
  - `min_samples=5`: Require ≥5 forecasters for an event
- **Sliding window**: Alternative density-based approach

**Event Criteria**:
- ≥5 forecasters updated
- Median update magnitude ≥0.01
- Temporal density > 2× baseline

### 3. Statistical Analysis (`analyze_data.py`)

**Inputs**: Regression panel
**Outputs**:
- `regression_table.tex`: LaTeX table for manuscript
- `analysis_summary_TIMESTAMP.json`: Full results

**Models**:

```python
# Model 1: Basic
Δp ~ track_record

# Model 2: Controls
Δp ~ track_record + prev_belief + time_delta + experience

# Model 3: Interaction
Δp ~ track_record × information_content + controls

# Model 4: Fixed Effects
Δp ~ track_record + controls + question_FE
```

All models use:
- **Cluster-robust standard errors** (clustered by question)
- **Question fixed effects** (Model 4)
- **Log transformations** (skewed distributions)

**Robustness Checks**:
- Log-log specification
- Quantile regression (effects across distribution)
- Experienced forecasters subsample
- Large events subsample

### 4. Visualization (`visualize_results.py`)

**Inputs**: Panel data + events
**Outputs**: Publication-ready PDFs

**Figures**:

1. **`fig1_main_effect.pdf`**: Main epistemic inertia result
   - Binned scatter: Track record vs update magnitude
   - OLS regression line with 95% CI
   - Effect size annotation

2. **`fig2_distributions.pdf`**: Distribution comparison
   - Panel A: Violin plots (high vs low reputation)
   - Panel B: CDFs (cumulative distributions)
   - Mann-Whitney U test

3. **`fig3_interaction.pdf`**: Information content interaction
   - Separate regression lines for high/low info events
   - Shows epistemic inertia increases with shock size

4. **`fig4_examples.pdf`**: Belief trajectories
   - 4 example events showing temporal dynamics
   - Individual trajectories + group averages
   - High vs low reputation forecasters

5. **`fig5_quantiles.pdf`**: Quantile regression
   - Effect across update magnitude distribution
   - Tests if epistemic inertia varies by update size

## Usage

### Quick Start

Run complete pipeline:
```bash
python run_pipeline.py
```

This will:
1. Fetch ~100 questions from Metaculus (takes ~30 min)
2. Detect news events
3. Run statistical analysis
4. Generate all figures

Output saved to `results_TIMESTAMP/`

### Custom Parameters

```bash
python run_pipeline.py \
    --max-questions 200 \
    --min-predictions 100 \
    --output-dir my_analysis
```

### Skip Data Fetching (Use Cached Data)

```bash
python run_pipeline.py --skip-fetch
```

### Individual Steps

Run components separately:

```bash
# 1. Fetch data
python fetch_data.py

# 2. Detect events
python detect_events.py

# 3. Run analysis
python analyze_data.py

# 4. Generate figures
python visualize_results.py
```

## Dependencies

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn requests tqdm scikit-learn
```

## Expected Results

If epistemic inertia exists, we expect:

✓ **β_1 < 0**: Negative coefficient on track record
✓ **p < 0.05**: Statistically significant
✓ **Effect size**: ~10-20% reduction in updates per SD increase in track record
✓ **Robustness**: Effect holds across specifications

### Null Results Interpretation

If **β_1 ≈ 0** (no effect):
- Metaculus track record ≠ social attention (proxy failure)
- Individual inference dominates (no social coupling)
- Epistemic inertia too small to detect with this sample
- Alternative: High-skill forecasters update optimally (sophistication ≠ rigidity)

## Manuscript Integration

### For Sociology Journals (AJS, Sociological Theory)

**Narrative Arc**:
1. **Problem**: Classical models (DeGroot, Friedkin-Johnsen) are first-order → no inertia
2. **Theory**: Hamiltonian VFE unifies existing models + adds epistemic mass
3. **Derivations**: Show 6 classical models as limits (see `docs/derivations_sociology_manuscript.tex`)
4. **Novel Prediction**: Epistemic inertia from social coupling
5. **Empirical Test**: Metaculus analysis
6. **Results**: [Insert findings]
7. **Implications**: Status → belief rigidity, echo chamber dynamics, etc.

**Table 1**: Classical Models as Limits of Hamiltonian VFE
**Table 2**: Regression Results (Epistemic Inertia)
**Figure 1**: Main effect (track record → updates)
**Figure 2**: Distributions comparison
**Figure 3**: Information content interaction

### Citation

If you use this analysis:

```bibtex
@article{dennison2025epistemic,
  title={Epistemic Inertia in Social Belief Dynamics:
         A Hamiltonian Variational Free Energy Framework},
  author={Dennison, C.},
  journal={Manuscript in preparation},
  year={2025}
}
```

## Limitations

1. **Proxy validity**: Track record may not equal social attention/followers
2. **Selection bias**: Metaculus users are sophisticated forecasters (not general population)
3. **Observational data**: Cannot establish causation
4. **No network data**: Cannot measure β_ji directly
5. **Binary questions only**: Limited to probability updates on {0,1} outcomes

## Extensions

Possible follow-ups:

1. **Heterogeneity**: Does epistemic inertia vary by topic domain?
2. **Temporal dynamics**: Test for delayed responses (phase lag)
3. **Network effects**: Use comment/discussion data to infer β_ji
4. **Prediction accuracy**: Does inertia harm or help performance?
5. **Community polarization**: Epistemic inertia → echo chambers?

## References

### Theoretical Foundation

This experiment provides empirical validation for:

**Dennis, R.C. (2025). The Inertia of Belief.**
- Full mass matrix derivation (Equation 266-268)
- Physical interpretations of all 4 components
- Testable predictions for epistemic inertia
- Location: `papers/psych/belief_inertia.tex`

**Critical Context**: The psychology paper was desk rejected by:
- Perspectives in Psychological Science
- Entropy
- Journal of Mathematical Psychology

**Reason for rejection**: Beautiful theory but insufficient empirical evidence. The Nassar helicopter task validation showed overdamped dynamics (as predicted), but this meant the framework reduced to standard gradient descent in that regime. Reviewers wanted evidence of the UNIQUE predictions (oscillations, inertia, overshooting).

**This Metaculus experiment addresses the gap**: Tests epistemic inertia directly in a regime where social mass dominates, providing the missing empirical validation.

### Related Documentation

- **Sociology manuscript**: `docs/derivations_sociology_manuscript.tex`
  - Shows 6 classical models (DeGroot, Friedkin-Johnsen, etc.) as limits
  - Full mathematical derivations with Proposition/Proof structure
  - Intended for AJS, Sociological Theory submission

- **Code implementation**: `agent/hamiltonian_trainer.py`
  - Complete 4-term mass matrix (lines 445-606)
  - Observation precision + incoming attention added Dec 2024

## Contact

For questions about this analysis:
- Theory: See `papers/psych/belief_inertia.tex` (psychology paper)
- Sociology integration: See `docs/derivations_sociology_manuscript.tex`
- Implementation: See code comments in each module
- Issues: Open GitHub issue

## License

MIT License - Free to use with attribution

---

**Status**: ✓ Pipeline complete, ready to run
**Purpose**: Empirical validation for rejected psychology paper
**Target journals**: AJS, Sociological Theory (theory + empirics)
**Last updated**: December 2024
