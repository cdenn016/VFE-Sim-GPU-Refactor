"""
Epistemic Inertia Analysis for Manifold Markets Data

Tests the hypothesis that traders with higher "epistemic mass" exhibit
greater belief rigidity (smaller updates, less frequent trading).

Mass proxies:
- follower_count: Direct measure of social influence (β_ji terms)
- total_profit: Track record (proxy for confidence/reputation)
- trader_count: Number of people trading on their markets (influence)

Predictions from mass matrix theory:
  M_i ∝ follower_count + total_profit + experience

  High M_i → Smaller |Δp| per bet
           → Lower bet frequency
           → Slower response to new information
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class EpistemicInertiaAnalyzer:
    """Analyze epistemic inertia in prediction market data."""

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)

    def load_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Load most recent data files."""
        updates_file = sorted(self.data_dir.glob('updates_*.csv'))[-1]
        bets_file = sorted(self.data_dir.glob('bets_*.csv'))[-1]
        users_file = sorted(self.data_dir.glob('users_*.csv'))[-1]
        markets_file = sorted(self.data_dir.glob('markets_*.csv'))[-1]

        print(f"Loading data from {self.data_dir}/")
        return {
            'updates': pd.read_csv(updates_file),
            'bets': pd.read_csv(bets_file),
            'users': pd.read_csv(users_file),
            'markets': pd.read_csv(markets_file)
        }

    def compute_mass_proxies(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute epistemic mass proxies from user statistics.

        Mass components:
        1. follower_count: Social influence (Σ_j β_ji)
        2. total_profit: Track record (proxy for Λ_p precision)
        3. trader_count: Influence breadth
        4. experience: Time on platform

        Returns:
            DataFrame with mass_score column
        """
        users = users_df.copy()

        # Normalize each component to [0, 1]
        users['follower_norm'] = (users['follower_count'] - users['follower_count'].min()) / \
                                 (users['follower_count'].max() - users['follower_count'].min() + 1e-10)

        users['profit_norm'] = (users['total_profit'] - users['total_profit'].min()) / \
                              (users['total_profit'].max() - users['total_profit'].min() + 1e-10)

        users['trader_norm'] = (users['trader_count'] - users['trader_count'].min()) / \
                               (users['trader_count'].max() - users['trader_count'].min() + 1e-10)

        # Compute experience (days since account creation)
        users['created_time'] = pd.to_datetime(users['created_time'])
        now = pd.Timestamp.now()
        users['experience_days'] = (now - users['created_time']).dt.total_seconds() / 86400

        users['experience_norm'] = (users['experience_days'] - users['experience_days'].min()) / \
                                   (users['experience_days'].max() - users['experience_days'].min() + 1e-10)

        # Composite mass score (weighted average)
        users['mass_score'] = (
            0.4 * users['follower_norm'] +      # Social influence (highest weight)
            0.3 * users['profit_norm'] +         # Track record
            0.2 * users['trader_norm'] +         # Influence breadth
            0.1 * users['experience_norm']       # Experience
        )

        return users

    def test_inertia_hypothesis(self,
                                updates_df: pd.DataFrame,
                                users_df: pd.DataFrame) -> Dict:
        """
        Test epistemic inertia hypothesis.

        H1: High mass → Smaller average |Δp|
        H2: High mass → Lower bet frequency
        H3: High mass → Larger bets (higher commitment threshold)

        Returns:
            Dictionary with test results and statistics
        """
        # Merge updates with user mass scores
        users_with_mass = self.compute_mass_proxies(users_df)
        updates = updates_df.merge(
            users_with_mass[['user_id', 'mass_score', 'follower_count', 'total_profit']],
            on='user_id',
            how='left'
        )

        # Remove users with no mass data
        updates = updates[updates['mass_score'].notna()]

        print("=" * 70)
        print("EPISTEMIC INERTIA ANALYSIS")
        print("=" * 70)
        print(f"\nAnalyzing {len(updates)} updates from {updates['user_id'].nunique()} users")

        # Split into high-mass and low-mass groups
        mass_median = updates['mass_score'].median()
        high_mass = updates[updates['mass_score'] >= mass_median]
        low_mass = updates[updates['mass_score'] < mass_median]

        results = {}

        # Test 1: Update magnitude
        print("\n" + "=" * 70)
        print("TEST 1: Update Magnitude (|Δp|)")
        print("=" * 70)

        high_mass_updates = high_mass['update_magnitude'].dropna()
        low_mass_updates = low_mass['update_magnitude'].dropna()

        print(f"\nHigh mass (N={len(high_mass_updates)}):")
        print(f"  Mean |Δp|: {high_mass_updates.mean():.4f}")
        print(f"  Median |Δp|: {high_mass_updates.median():.4f}")
        print(f"  Std |Δp|: {high_mass_updates.std():.4f}")

        print(f"\nLow mass (N={len(low_mass_updates)}):")
        print(f"  Mean |Δp|: {low_mass_updates.mean():.4f}")
        print(f"  Median |Δp|: {low_mass_updates.median():.4f}")
        print(f"  Std |Δp|: {low_mass_updates.std():.4f}")

        # Statistical test
        t_stat, p_value = stats.mannwhitneyu(high_mass_updates, low_mass_updates, alternative='less')
        print(f"\nMann-Whitney U test (H: high_mass < low_mass):")
        print(f"  U-statistic: {t_stat:.2f}")
        print(f"  p-value: {p_value:.4e}")

        if p_value < 0.05:
            print("  ✓ SIGNIFICANT: High mass traders make SMALLER updates")
        else:
            print("  ✗ NOT SIGNIFICANT: No difference in update magnitudes")

        results['update_magnitude'] = {
            'high_mass_mean': high_mass_updates.mean(),
            'low_mass_mean': low_mass_updates.mean(),
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        # Test 2: Correlation between mass and update size
        print("\n" + "=" * 70)
        print("TEST 2: Mass-Update Correlation")
        print("=" * 70)

        corr, corr_p = stats.spearmanr(updates['mass_score'], updates['update_magnitude'])
        print(f"\nSpearman correlation (mass_score vs |Δp|):")
        print(f"  ρ = {corr:.4f}")
        print(f"  p-value: {corr_p:.4e}")

        if corr_p < 0.05:
            if corr < 0:
                print("  ✓ SIGNIFICANT NEGATIVE: Higher mass → Smaller updates")
            else:
                print("  ✗ SIGNIFICANT POSITIVE: Higher mass → LARGER updates (unexpected!)")
        else:
            print("  ✗ NOT SIGNIFICANT: No correlation")

        results['correlation'] = {
            'rho': corr,
            'p_value': corr_p,
            'significant': corr_p < 0.05
        }

        # Test 3: Bet frequency by mass
        print("\n" + "=" * 70)
        print("TEST 3: Trading Frequency")
        print("=" * 70)

        # Count bets per user
        bets_per_user = updates.groupby('user_id').agg({
            'bet_id': 'count',
            'mass_score': 'first'
        }).rename(columns={'bet_id': 'num_bets'})

        high_mass_users = bets_per_user[bets_per_user['mass_score'] >= mass_median]
        low_mass_users = bets_per_user[bets_per_user['mass_score'] < mass_median]

        print(f"\nHigh mass (N={len(high_mass_users)} users):")
        print(f"  Mean bets per user: {high_mass_users['num_bets'].mean():.2f}")
        print(f"  Median bets per user: {high_mass_users['num_bets'].median():.2f}")

        print(f"\nLow mass (N={len(low_mass_users)} users):")
        print(f"  Mean bets per user: {low_mass_users['num_bets'].mean():.2f}")
        print(f"  Median bets per user: {low_mass_users['num_bets'].median():.2f}")

        t_stat_freq, p_value_freq = stats.mannwhitneyu(
            high_mass_users['num_bets'],
            low_mass_users['num_bets'],
            alternative='less'
        )
        print(f"\nMann-Whitney U test (H: high_mass < low_mass):")
        print(f"  p-value: {p_value_freq:.4e}")

        if p_value_freq < 0.05:
            print("  ✓ SIGNIFICANT: High mass traders bet LESS frequently")
        else:
            print("  ✗ NOT SIGNIFICANT")

        results['frequency'] = {
            'high_mass_mean': high_mass_users['num_bets'].mean(),
            'low_mass_mean': low_mass_users['num_bets'].mean(),
            'p_value': p_value_freq,
            'significant': p_value_freq < 0.05
        }

        return results

    def plot_results(self, updates_df: pd.DataFrame, users_df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots."""
        users_with_mass = self.compute_mass_proxies(users_df)
        updates = updates_df.merge(
            users_with_mass[['user_id', 'mass_score', 'follower_count']],
            on='user_id',
            how='left'
        )
        updates = updates[updates['mass_score'].notna()]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Mass vs Update Magnitude (scatter)
        ax1 = axes[0, 0]
        ax1.scatter(updates['mass_score'], updates['update_magnitude'],
                   alpha=0.3, s=20)
        ax1.set_xlabel('Epistemic Mass Score')
        ax1.set_ylabel('Update Magnitude |Δp|')
        ax1.set_title('Epistemic Inertia: Mass vs Update Size')
        ax1.grid(alpha=0.3)

        # Add trend line
        z = np.polyfit(updates['mass_score'], updates['update_magnitude'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(updates['mass_score'].min(), updates['mass_score'].max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8, label='Trend')
        ax1.legend()

        # 2. Distribution of updates by mass group
        ax2 = axes[0, 1]
        mass_median = updates['mass_score'].median()
        high_mass = updates[updates['mass_score'] >= mass_median]['update_magnitude']
        low_mass = updates[updates['mass_score'] < mass_median]['update_magnitude']

        ax2.hist([low_mass, high_mass], bins=50, alpha=0.6,
                label=['Low Mass', 'High Mass'], density=True)
        ax2.set_xlabel('Update Magnitude |Δp|')
        ax2.set_ylabel('Density')
        ax2.set_title('Update Distribution by Mass Group')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Follower count vs update magnitude
        ax3 = axes[1, 0]
        ax3.scatter(updates['follower_count'], updates['update_magnitude'],
                   alpha=0.3, s=20)
        ax3.set_xlabel('Follower Count')
        ax3.set_ylabel('Update Magnitude |Δp|')
        ax3.set_title('Social Influence vs Update Size')
        ax3.set_xscale('log')
        ax3.grid(alpha=0.3)

        # 4. Boxplot by mass quartiles
        ax4 = axes[1, 1]
        updates['mass_quartile'] = pd.qcut(updates['mass_score'], q=4,
                                           labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        updates.boxplot(column='update_magnitude', by='mass_quartile', ax=ax4)
        ax4.set_xlabel('Mass Quartile')
        ax4.set_ylabel('Update Magnitude |Δp|')
        ax4.set_title('Update Size by Mass Quartile')
        plt.sca(ax4)
        plt.xticks(rotation=45)

        plt.tight_layout()
        output_path = output_dir / 'epistemic_inertia_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved plots to {output_path}")


def main():
    """Run epistemic inertia analysis on Manifold data."""
    analyzer = EpistemicInertiaAnalyzer(data_dir='data')

    # Load data
    data = analyzer.load_latest_data()

    # Run tests
    results = analyzer.test_inertia_hypothesis(
        data['updates'],
        data['users']
    )

    # Generate plots
    analyzer.plot_results(data['updates'], data['users'], Path('data'))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nEpistemic Inertia Hypothesis:")
    print("  High mass → Smaller updates, less frequent trading")

    if results['update_magnitude']['significant']:
        print("\n✓ UPDATE MAGNITUDE: CONFIRMED")
        print(f"  High mass: {results['update_magnitude']['high_mass_mean']:.4f}")
        print(f"  Low mass: {results['update_magnitude']['low_mass_mean']:.4f}")
        print(f"  p = {results['update_magnitude']['p_value']:.4e}")
    else:
        print("\n✗ UPDATE MAGNITUDE: NOT CONFIRMED")

    if results['correlation']['significant'] and results['correlation']['rho'] < 0:
        print("\n✓ CORRELATION: CONFIRMED (negative)")
        print(f"  ρ = {results['correlation']['rho']:.4f}")
        print(f"  p = {results['correlation']['p_value']:.4e}")
    else:
        print("\n✗ CORRELATION: NOT CONFIRMED")

    if results['frequency']['significant']:
        print("\n✓ BET FREQUENCY: CONFIRMED")
        print(f"  High mass: {results['frequency']['high_mass_mean']:.2f} bets/user")
        print(f"  Low mass: {results['frequency']['low_mass_mean']:.2f} bets/user")
        print(f"  p = {results['frequency']['p_value']:.4e}")
    else:
        print("\n✗ BET FREQUENCY: NOT CONFIRMED")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
