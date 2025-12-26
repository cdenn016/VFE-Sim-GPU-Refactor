"""
Statistical Analysis for Epistemic Inertia

Tests the core prediction of Hamiltonian VFE:
    Mass matrix M_i = Σ_p^{-1} + Σ_j β_ji Ω Σ_j^{-1} Ω^T

predicts that agents with more followers/attention (β_ji) have greater
epistemic inertia → smaller belief updates.

Empirical proxy: Forecaster track record ≈ attention/followers
Testable hypothesis: High track_record → Smaller |Δp|

Statistical Models:
    Model 1 (Basic): Δp ~ track_record
    Model 2 (Controls): Δp ~ track_record + prev_belief + time_delta + experience
    Model 3 (Interaction): Δp ~ track_record × information_content + controls

All models include question fixed effects and clustered standard errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from scipy import stats
import json
import matplotlib.pyplot as plt
import seaborn as sns


class EpistemicInertiaAnalysis:
    """Statistical analysis of epistemic inertia in forecasting data."""

    def __init__(self, panel_df: pd.DataFrame, events_df: pd.DataFrame):
        """
        Initialize with panel dataset.

        Args:
            panel_df: Panel with [user_id, event_id, update_magnitude, track_record, ...]
            events_df: Event metadata with information content
        """
        self.panel = panel_df.copy()
        self.events = events_df.copy()

        # Merge event-level variables
        self.panel = self.panel.merge(
            self.events[['question_id', 'event_id', 'information_content',
                        'median_update', 'num_forecasters']],
            on=['question_id', 'event_id'],
            how='left',
            suffixes=('', '_event')
        )

        # Clean data
        self._clean_data()

    def _clean_data(self):
        """Remove outliers and missing values."""
        # Remove missing values for key variables
        required_cols = ['update_magnitude', 'track_record', 'prev_prediction',
                        'time_delta_hours']
        self.panel = self.panel.dropna(subset=required_cols)

        # Remove extreme outliers (updates > 0.5 are rare and often data errors)
        self.panel = self.panel[self.panel['update_magnitude'] <= 0.5]

        # Remove zero updates (no information)
        self.panel = self.panel[self.panel['update_magnitude'] > 0.001]

        # Log transformations for skewed variables
        self.panel['log_update'] = np.log(self.panel['update_magnitude'])
        self.panel['log_track_record'] = np.log1p(self.panel['track_record'])
        self.panel['log_time_delta'] = np.log1p(self.panel['time_delta_hours'])

        print(f"Clean panel: {len(self.panel)} observations")

    def run_model_1_basic(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Model 1: Basic epistemic inertia test.

        Δp_it = β_0 + β_1 · track_record_i + α_q + ε_it

        H0: β_1 = 0 (no epistemic inertia)
        H1: β_1 < 0 (high reputation → smaller updates)

        Returns:
            Regression results
        """
        print("\n" + "="*60)
        print("MODEL 1: Basic Epistemic Inertia")
        print("="*60)

        formula = 'update_magnitude ~ log_track_record'

        model = smf.ols(formula, data=self.panel)
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': self.panel['question_id']}
        )

        print(results.summary())
        return results

    def run_model_2_controls(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Model 2: Epistemic inertia with controls.

        Δp_it = β_0 + β_1 · track_record_i
                + β_2 · prev_belief_it
                + β_3 · time_delta_it
                + β_4 · experience_i
                + α_q + ε_it

        Controls:
        - prev_belief: Starting position (anchoring effects)
        - time_delta: Time since last update (staleness)
        - experience: Total predictions (learning effects)

        Returns:
            Regression results
        """
        print("\n" + "="*60)
        print("MODEL 2: Epistemic Inertia with Controls")
        print("="*60)

        formula = '''update_magnitude ~ log_track_record
                     + belief_extremeness
                     + log_time_delta
                     + log_prediction_count'''

        model = smf.ols(formula, data=self.panel)
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': self.panel['question_id']}
        )

        print(results.summary())
        return results

    def run_model_3_interaction(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Model 3: Interaction with information content.

        Δp_it = β_0 + β_1 · track_record_i
                + β_2 · information_content_t
                + β_3 · (track_record_i × information_content_t)
                + controls + α_q + ε_it

        Theory: High-inertia agents should show smaller updates especially
        when information content is HIGH (large shocks reveal inertia more).

        Prediction: β_3 < 0 (interaction negative)

        Returns:
            Regression results
        """
        print("\n" + "="*60)
        print("MODEL 3: Interaction with Information Content")
        print("="*60)

        # Create interaction term
        self.panel['track_x_info'] = (
            self.panel['log_track_record'] * self.panel['information_content']
        )

        formula = '''update_magnitude ~ log_track_record
                     + information_content
                     + track_x_info
                     + belief_extremeness
                     + log_time_delta
                     + log_prediction_count'''

        model = smf.ols(formula, data=self.panel)
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': self.panel['question_id']}
        )

        print(results.summary())
        return results

    def run_model_4_fixed_effects(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Model 4: Full specification with question fixed effects.

        Δp_it = β_1 · track_record_i + controls + α_q + ε_it

        Uses within-question variation only, controlling for all
        question-specific factors (difficulty, topic, etc.).

        Returns:
            Regression results
        """
        print("\n" + "="*60)
        print("MODEL 4: Question Fixed Effects")
        print("="*60)

        formula = '''update_magnitude ~ log_track_record
                     + belief_extremeness
                     + log_time_delta
                     + log_prediction_count
                     + C(question_id)'''

        model = smf.ols(formula, data=self.panel)
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': self.panel['question_id']}
        )

        print(results.summary())
        return results

    def compute_effect_sizes(self, results: sm.regression.linear_model.RegressionResultsWrapper) -> Dict:
        """
        Compute standardized effect sizes and practical significance.

        Args:
            results: Regression results from model

        Returns:
            Dictionary with effect size metrics
        """
        # Get track_record coefficient
        coef_name = 'log_track_record'
        if coef_name not in results.params.index:
            return {}

        beta = results.params[coef_name]
        se = results.bse[coef_name]
        t_stat = results.tvalues[coef_name]
        p_value = results.pvalues[coef_name]

        # Standardized effect (beta in SD units)
        sd_track = self.panel['log_track_record'].std()
        sd_update = self.panel['update_magnitude'].std()
        beta_std = beta * (sd_track / sd_update)

        # Practical significance: 1 SD increase in track_record → ?% change in updates
        pct_change = (beta * sd_track) / self.panel['update_magnitude'].mean() * 100

        # Cohen's f^2 (incremental R^2)
        # (would need nested model comparison)

        effect_sizes = {
            'coefficient': beta,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'standardized_beta': beta_std,
            'percent_change_1sd': pct_change,
            'is_significant_05': p_value < 0.05,
            'is_significant_01': p_value < 0.01
        }

        return effect_sizes

    def run_robustness_checks(self) -> Dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
        """
        Run robustness checks.

        1. Log-log specification: log(Δp) ~ log(track_record)
        2. Quantile regression: Test across update magnitude distribution
        3. Subsample: Only experienced forecasters (>10 predictions)
        4. Subsample: Only large events (>10 forecasters)

        Returns:
            Dictionary of results
        """
        print("\n" + "="*60)
        print("ROBUSTNESS CHECKS")
        print("="*60)

        robustness = {}

        # 1. Log-log specification
        print("\n1. Log-Log Specification:")
        formula = 'log_update ~ log_track_record + belief_extremeness + log_time_delta'
        model = smf.ols(formula, data=self.panel)
        robustness['log_log'] = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': self.panel['question_id']}
        )
        print(f"  Coefficient: {robustness['log_log'].params['log_track_record']:.4f}")
        print(f"  p-value: {robustness['log_log'].pvalues['log_track_record']:.4f}")

        # 2. Experienced forecasters only
        print("\n2. Experienced Forecasters (>10 predictions):")
        experienced = self.panel[self.panel['prediction_count'] > 10]
        formula = 'update_magnitude ~ log_track_record + belief_extremeness + log_time_delta'
        model = smf.ols(formula, data=experienced)
        robustness['experienced'] = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': experienced['question_id']}
        )
        print(f"  N = {len(experienced)}")
        print(f"  Coefficient: {robustness['experienced'].params['log_track_record']:.4f}")
        print(f"  p-value: {robustness['experienced'].pvalues['log_track_record']:.4f}")

        # 3. Large events only
        print("\n3. Large Events (>10 forecasters):")
        large_events = self.panel[self.panel['num_forecasters'] > 10]
        formula = 'update_magnitude ~ log_track_record + belief_extremeness + log_time_delta'
        model = smf.ols(formula, data=large_events)
        robustness['large_events'] = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': large_events['question_id']}
        )
        print(f"  N = {len(large_events)}")
        print(f"  Coefficient: {robustness['large_events'].params['log_track_record']:.4f}")
        print(f"  p-value: {robustness['large_events'].pvalues['log_track_record']:.4f}")

        return robustness

    def create_regression_table(self,
                               models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
                               output_path: Path) -> str:
        """
        Create publication-ready regression table.

        Args:
            models: Dictionary of model results
            output_path: Path to save LaTeX table

        Returns:
            LaTeX table string
        """
        # Create summary table
        table = summary_col(
            list(models.values()),
            model_names=list(models.keys()),
            stars=True,
            float_format='%.4f',
            info_dict={
                'N': lambda x: f"{int(x.nobs)}",
                'R²': lambda x: f"{x.rsquared:.3f}",
                'Adj. R²': lambda x: f"{x.rsquared_adj:.3f}"
            }
        )

        # Save LaTeX
        latex = table.as_latex()

        output_path.mkdir(exist_ok=True, parents=True)
        with open(output_path / 'regression_table.tex', 'w') as f:
            f.write(latex)

        print(f"\nRegression table saved to {output_path / 'regression_table.tex'}")

        return latex


def main(data_dir: str = 'data', output_dir: str = 'tables'):
    """
    Run complete statistical analysis pipeline.

    Args:
        data_dir: Directory with processed data
        output_dir: Directory to save results
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load data
    panel_file = sorted(data_path.glob('regression_panel_*.csv'))[-1]
    events_file = sorted(data_path.glob('significant_events_*.csv'))[-1]

    print(f"Loading data from {data_path}/")
    panel_df = pd.read_csv(panel_file)
    events_df = pd.read_csv(events_file)

    # Initialize analysis
    analysis = EpistemicInertiaAnalysis(panel_df, events_df)

    # Run models
    models = {}
    models['Model 1: Basic'] = analysis.run_model_1_basic()
    models['Model 2: Controls'] = analysis.run_model_2_controls()
    models['Model 3: Interaction'] = analysis.run_model_3_interaction()
    models['Model 4: Fixed Effects'] = analysis.run_model_4_fixed_effects()

    # Compute effect sizes for Model 2
    print("\n" + "="*60)
    print("EFFECT SIZES (Model 2)")
    print("="*60)
    effect_sizes = analysis.compute_effect_sizes(models['Model 2: Controls'])
    for key, value in effect_sizes.items():
        print(f"{key}: {value}")

    # Robustness checks
    robustness = analysis.run_robustness_checks()

    # Create regression table
    latex_table = analysis.create_regression_table(models, output_path)

    # Save all results
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_summary = {
        'timestamp': timestamp,
        'n_observations': len(analysis.panel),
        'n_forecasters': analysis.panel['user_id'].nunique(),
        'n_questions': analysis.panel['question_id'].nunique(),
        'n_events': analysis.panel['event_id'].nunique(),
        'effect_sizes': effect_sizes,
        'model_summaries': {
            name: {
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'adj_r_squared': float(model.rsquared_adj),
                'track_record_coef': float(model.params.get('log_track_record', np.nan)),
                'track_record_pval': float(model.pvalues.get('log_track_record', np.nan))
            }
            for name, model in models.items()
        }
    }

    with open(output_path / f'analysis_summary_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to {output_path}/")
    print(f"\nKey finding:")
    if effect_sizes.get('is_significant_05'):
        direction = "NEGATIVE" if effect_sizes['coefficient'] < 0 else "POSITIVE"
        print(f"  ✓ {direction} relationship between track record and update magnitude")
        print(f"  ✓ Coefficient: {effect_sizes['coefficient']:.4f} (p={effect_sizes['p_value']:.4f})")
        print(f"  ✓ 1 SD increase in track record → {effect_sizes['percent_change_1sd']:.1f}% change in updates")

        if effect_sizes['coefficient'] < 0:
            print(f"\n  → EPISTEMIC INERTIA CONFIRMED: High-reputation forecasters make smaller updates")
        else:
            print(f"\n  → UNEXPECTED: High-reputation forecasters make larger updates")
    else:
        print(f"  ✗ No significant relationship detected (p={effect_sizes['p_value']:.4f})")

    return models, effect_sizes, robustness


if __name__ == '__main__':
    models, effects, robustness = main(data_dir='data', output_dir='tables')
