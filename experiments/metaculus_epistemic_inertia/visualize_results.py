"""
Publication-Quality Visualizations for Epistemic Inertia

Creates figures for sociology manuscript demonstrating:
1. Main effect: Track record → Update magnitude
2. Distributions: High vs low reputation forecasters
3. Event analysis: Information content interactions
4. Temporal dynamics: Example belief trajectories

Style: Publication-ready for AJS, Sociological Theory, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# Publication style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

sns.set_palette("colorblind")


class EpistemicInertiaVisualizer:
    """Create publication-ready visualizations."""

    def __init__(self, panel_df: pd.DataFrame, events_df: pd.DataFrame):
        """
        Initialize with analysis data.

        Args:
            panel_df: Panel dataset with update observations
            events_df: Event metadata
        """
        self.panel = panel_df.copy()
        self.events = events_df.copy()

    def plot_main_effect(self, output_path: Path, filename: str = 'fig1_main_effect.pdf'):
        """
        Figure 1: Main epistemic inertia effect.

        Binned scatter plot showing track record vs update magnitude.
        Includes regression line and 95% CI.

        Args:
            output_path: Directory to save figure
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # Create bins for track record
        self.panel['track_record_bin'] = pd.qcut(
            self.panel['log_track_record'],
            q=20,
            labels=False,
            duplicates='drop'
        )

        # Compute bin statistics
        bin_stats = self.panel.groupby('track_record_bin').agg({
            'log_track_record': 'mean',
            'update_magnitude': ['mean', 'sem', 'count']
        }).reset_index()

        bin_stats.columns = ['bin', 'track_record', 'update_mean', 'update_sem', 'count']

        # Scatter plot with error bars
        ax.errorbar(
            bin_stats['track_record'],
            bin_stats['update_mean'],
            yerr=1.96 * bin_stats['update_sem'],  # 95% CI
            fmt='o',
            markersize=6,
            capsize=4,
            alpha=0.7,
            label='Binned means (95% CI)'
        )

        # Regression line
        z = np.polyfit(
            self.panel['log_track_record'],
            self.panel['update_magnitude'],
            1
        )
        p = np.poly1d(z)
        x_line = np.linspace(
            self.panel['log_track_record'].min(),
            self.panel['log_track_record'].max(),
            100
        )
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='OLS fit', alpha=0.8)

        # Formatting
        ax.set_xlabel('Log Track Record (Reputation)', fontsize=12)
        ax.set_ylabel('Update Magnitude (|Δp|)', fontsize=12)
        ax.set_title('Epistemic Inertia: High-Reputation Forecasters Make Smaller Updates',
                    fontsize=13, pad=15)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')

        # Add equation and stats
        r, p_val = stats.pearsonr(
            self.panel['log_track_record'],
            self.panel['update_magnitude']
        )
        textstr = f'β = {z[0]:.4f}\nr = {r:.3f}\np < 0.001' if p_val < 0.001 else f'β = {z[0]:.4f}\nr = {r:.3f}\np = {p_val:.3f}'
        ax.text(0.05, 0.95, textstr,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path / filename)
        print(f"Figure 1 saved: {output_path / filename}")
        plt.close()

    def plot_distributions(self, output_path: Path, filename: str = 'fig2_distributions.pdf'):
        """
        Figure 2: Distribution comparison.

        Violin plots or histograms comparing update magnitudes for
        high vs low reputation forecasters.

        Args:
            output_path: Directory to save figure
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Split into high/low reputation
        median_track = self.panel['log_track_record'].median()
        self.panel['reputation_group'] = self.panel['log_track_record'].apply(
            lambda x: 'High Reputation' if x >= median_track else 'Low Reputation'
        )

        # Panel A: Violin plots
        ax = axes[0]
        parts = ax.violinplot(
            [
                self.panel[self.panel['reputation_group'] == 'Low Reputation']['update_magnitude'],
                self.panel[self.panel['reputation_group'] == 'High Reputation']['update_magnitude']
            ],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Low\nReputation', 'High\nReputation'])
        ax.set_ylabel('Update Magnitude (|Δp|)', fontsize=12)
        ax.set_title('(A) Distribution of Update Magnitudes', fontsize=12, pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add means
        low_mean = self.panel[self.panel['reputation_group'] == 'Low Reputation']['update_magnitude'].mean()
        high_mean = self.panel[self.panel['reputation_group'] == 'High Reputation']['update_magnitude'].mean()
        ax.text(1, ax.get_ylim()[1] * 0.95, f'μ = {low_mean:.3f}',
               ha='center', fontsize=9)
        ax.text(2, ax.get_ylim()[1] * 0.95, f'μ = {high_mean:.3f}',
               ha='center', fontsize=9)

        # Panel B: CDFs
        ax = axes[1]

        for group in ['Low Reputation', 'High Reputation']:
            data = self.panel[self.panel['reputation_group'] == group]['update_magnitude']
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, linewidth=2, label=group, alpha=0.8)

        ax.set_xlabel('Update Magnitude (|Δp|)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('(B) Cumulative Distribution Functions', fontsize=12, pad=10)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3, linestyle='--')

        # Statistical test
        low_updates = self.panel[self.panel['reputation_group'] == 'Low Reputation']['update_magnitude']
        high_updates = self.panel[self.panel['reputation_group'] == 'High Reputation']['update_magnitude']
        statistic, p_value = stats.mannwhitneyu(low_updates, high_updates, alternative='greater')

        fig.text(0.5, 0.02,
                f'Mann-Whitney U test: U = {statistic:.0f}, p < 0.001' if p_value < 0.001 else f'Mann-Whitney U test: U = {statistic:.0f}, p = {p_value:.4f}',
                ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path / filename)
        print(f"Figure 2 saved: {output_path / filename}")
        plt.close()

    def plot_information_interaction(self, output_path: Path,
                                     filename: str = 'fig3_interaction.pdf'):
        """
        Figure 3: Interaction with information content.

        Shows that epistemic inertia is more pronounced when information
        content is high (large shocks reveal mass more clearly).

        Args:
            output_path: Directory to save figure
            filename: Output filename
        """
        # Merge event information
        panel_with_info = self.panel.merge(
            self.events[['question_id', 'event_id', 'information_content']],
            on=['question_id', 'event_id'],
            how='left',
            suffixes=('', '_event')
        )

        # Split by information content
        median_info = panel_with_info['information_content'].median()
        panel_with_info['info_group'] = panel_with_info['information_content'].apply(
            lambda x: 'High Info' if x >= median_info else 'Low Info'
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        # Separate regression lines for each group
        for group, color, marker in [('Low Info', 'blue', 'o'), ('High Info', 'red', 's')]:
            data = panel_with_info[panel_with_info['info_group'] == group]

            # Bin and plot
            data['track_bin'] = pd.qcut(data['log_track_record'], q=10,
                                       labels=False, duplicates='drop')
            bin_stats = data.groupby('track_bin').agg({
                'log_track_record': 'mean',
                'update_magnitude': ['mean', 'sem']
            }).reset_index()

            bin_stats.columns = ['bin', 'track_record', 'update_mean', 'update_sem']

            ax.errorbar(
                bin_stats['track_record'],
                bin_stats['update_mean'],
                yerr=1.96 * bin_stats['update_sem'],
                fmt=marker,
                color=color,
                markersize=6,
                capsize=3,
                alpha=0.7,
                label=f'{group} Events'
            )

            # Regression line
            z = np.polyfit(data['log_track_record'], data['update_magnitude'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data['log_track_record'].min(),
                               data['log_track_record'].max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8,
                   linestyle='--', label=f'{group} fit (β={z[0]:.4f})')

        ax.set_xlabel('Log Track Record (Reputation)', fontsize=12)
        ax.set_ylabel('Update Magnitude (|Δp|)', fontsize=12)
        ax.set_title('Epistemic Inertia Increases with Information Content\n' +
                    '(High-reputation forecasters show smaller updates especially for large shocks)',
                    fontsize=12, pad=15)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path / filename)
        print(f"Figure 3 saved: {output_path / filename}")
        plt.close()

    def plot_event_examples(self, output_path: Path,
                           predictions_df: pd.DataFrame,
                           users_df: pd.DataFrame,
                           n_events: int = 4,
                           filename: str = 'fig4_examples.pdf'):
        """
        Figure 4: Example belief trajectories for specific events.

        Shows how high vs low reputation forecasters respond to
        the same news shock.

        Args:
            output_path: Directory to save figure
            predictions_df: Raw prediction time series
            users_df: User metadata
            n_events: Number of example events to show
            filename: Output filename
        """
        # Get top events by number of forecasters
        top_events = self.events.nlargest(n_events, 'num_forecasters')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (_, event) in enumerate(top_events.iterrows()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get predictions for this event
            event_preds = predictions_df[
                predictions_df['question_id'] == event['question_id']
            ].copy()

            # Merge user stats
            event_preds = event_preds.merge(users_df, on='user_id', how='left')

            # Split by reputation
            median_track = event_preds['track_record'].median()
            event_preds['group'] = event_preds['track_record'].apply(
                lambda x: 'high' if x >= median_track else 'low'
            )

            # Convert time to hours
            event_preds['time'] = pd.to_datetime(event_preds['time'])
            event_preds['hours'] = (
                event_preds['time'] - event_preds['time'].min()
            ).dt.total_seconds() / 3600

            # Plot individual trajectories (subsample for clarity)
            for group, color, alpha in [('low', 'blue', 0.3), ('high', 'red', 0.3)]:
                group_data = event_preds[event_preds['group'] == group]

                # Subsample users
                sampled_users = group_data['user_id'].unique()[:20]
                for user in sampled_users:
                    user_data = group_data[group_data['user_id'] == user].sort_values('hours')
                    ax.plot(user_data['hours'], user_data['prediction'],
                           color=color, alpha=alpha, linewidth=0.5)

            # Plot group averages
            for group, color, label in [('low', 'blue', 'Low Reputation'),
                                       ('high', 'red', 'High Reputation')]:
                group_data = event_preds[event_preds['group'] == group]

                # Time bins
                time_bins = np.linspace(0, group_data['hours'].max(), 50)
                bin_means = []

                for i in range(len(time_bins) - 1):
                    in_bin = group_data[
                        (group_data['hours'] >= time_bins[i]) &
                        (group_data['hours'] < time_bins[i+1])
                    ]['prediction']

                    if len(in_bin) > 0:
                        bin_means.append(in_bin.mean())
                    else:
                        bin_means.append(np.nan)

                # Smooth
                valid_means = np.array([m for m in bin_means if not np.isnan(m)])
                if len(valid_means) > 5:
                    smoothed = gaussian_filter1d(valid_means, sigma=2)
                    time_valid = time_bins[:-1][~np.isnan(bin_means)]
                    ax.plot(time_valid, smoothed, color=color, linewidth=3,
                           label=label, alpha=0.9)

            # Formatting
            ax.set_xlabel('Hours Since First Update', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title(f"Event {idx+1}: {event['question_id']}\n" +
                        f"({int(event['num_forecasters'])} forecasters)",
                        fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3, linestyle='--')
            if idx == 0:
                ax.legend(loc='best', fontsize=9)

        plt.suptitle('Example Belief Trajectories: High vs Low Reputation Forecasters',
                    fontsize=13, y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / filename)
        print(f"Figure 4 saved: {output_path / filename}")
        plt.close()

    def plot_quantile_effects(self, output_path: Path,
                             filename: str = 'fig5_quantiles.pdf'):
        """
        Figure 5: Epistemic inertia across update magnitude distribution.

        Quantile regression showing effect at different parts of distribution.

        Args:
            output_path: Directory to save figure
            filename: Output filename
        """
        from statsmodels.regression.quantile_regression import QuantReg

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        coefficients = []
        ci_lower = []
        ci_upper = []

        print("\nRunning quantile regressions...")

        for q in quantiles:
            # Prepare data
            X = self.panel[['log_track_record', 'belief_extremeness', 'log_time_delta']]
            X = sm.add_constant(X)
            y = self.panel['update_magnitude']

            # Fit quantile regression
            model = QuantReg(y, X)
            results = model.fit(q=q)

            coef = results.params['log_track_record']
            ci = results.conf_int().loc['log_track_record']

            coefficients.append(coef)
            ci_lower.append(ci[0])
            ci_upper.append(ci[1])

            print(f"  τ={q}: β={coef:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(quantiles, coefficients, 'o-', linewidth=2, markersize=8,
               color='darkblue', label='Coefficient')
        ax.fill_between(quantiles, ci_lower, ci_upper, alpha=0.3,
                        color='lightblue', label='95% CI')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7,
                  label='Null effect')

        ax.set_xlabel('Quantile (τ) of Update Magnitude Distribution', fontsize=12)
        ax.set_ylabel('Coefficient on Log Track Record', fontsize=12)
        ax.set_title('Epistemic Inertia Effect Across Update Distribution\n' +
                    '(Quantile Regression)',
                    fontsize=12, pad=15)
        ax.legend(loc='best')
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path / filename)
        print(f"Figure 5 saved: {output_path / filename}")
        plt.close()


def main(data_dir: str = 'data', output_dir: str = 'figures'):
    """
    Generate all publication figures.

    Args:
        data_dir: Directory with processed data
        output_dir: Directory to save figures
    """
    import statsmodels.api as sm  # Import here to avoid issues

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("Loading data...")
    panel_file = sorted(data_path.glob('regression_panel_*.csv'))[-1]
    events_file = sorted(data_path.glob('significant_events_*.csv'))[-1]
    predictions_file = sorted(data_path.glob('predictions_*.csv'))[-1]
    users_file = sorted(data_path.glob('users_*.csv'))[-1]

    panel_df = pd.read_csv(panel_file)
    events_df = pd.read_csv(events_file)
    predictions_df = pd.read_csv(predictions_file)
    users_df = pd.read_csv(users_file)

    # Initialize visualizer
    viz = EpistemicInertiaVisualizer(panel_df, events_df)

    # Generate all figures
    print("\nGenerating figures...")
    viz.plot_main_effect(output_path)
    viz.plot_distributions(output_path)
    viz.plot_information_interaction(output_path)
    viz.plot_event_examples(output_path, predictions_df, users_df)
    viz.plot_quantile_effects(output_path)

    print(f"\n{'='*60}")
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"Saved to {output_path}/")
    print("  - fig1_main_effect.pdf")
    print("  - fig2_distributions.pdf")
    print("  - fig3_interaction.pdf")
    print("  - fig4_examples.pdf")
    print("  - fig5_quantiles.pdf")


if __name__ == '__main__':
    main(data_dir='data', output_dir='figures')
