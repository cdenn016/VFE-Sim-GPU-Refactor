#!/usr/bin/env python3
"""
Complete Manifold Markets Epistemic Inertia Pipeline

Runs data collection and analysis in sequence.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fetch_data import ManifoldDataFetcher
from analyze_inertia import EpistemicInertiaAnalyzer


def run_full_pipeline(max_markets: int = 100, min_volume: int = 100):
    """
    Run complete epistemic inertia analysis pipeline.

    Args:
        max_markets: Number of markets to analyze
        min_volume: Minimum market volume to include
    """
    print("=" * 70)
    print("MANIFOLD MARKETS EPISTEMIC INERTIA ANALYSIS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Data collection
    print("\n" + "=" * 70)
    print("STEP 1: FETCHING DATA FROM MANIFOLD MARKETS API")
    print("=" * 70)

    fetcher = ManifoldDataFetcher(output_dir='data')
    data = fetcher.run_full_pipeline(
        max_markets=max_markets,
        min_volume=min_volume
    )

    print("\n✓ Data collection complete")
    print(f"  - {len(data['markets'])} markets")
    print(f"  - {len(data['bets'])} bets")
    print(f"  - {len(data['users'])} users")
    print(f"  - {len(data['updates'])} updates")

    # Check if we have enough data
    if len(data['updates']) < 100:
        print("\n⚠️  WARNING: Very few updates collected (<100)")
        print("Statistical tests may lack power. Consider:")
        print("  - Increasing max_markets")
        print("  - Decreasing min_volume threshold")
        response = input("\nContinue with analysis anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return None

    # Step 2: Analysis
    print("\n" + "=" * 70)
    print("STEP 2: TESTING EPISTEMIC INERTIA HYPOTHESIS")
    print("=" * 70)

    analyzer = EpistemicInertiaAnalyzer(data_dir='data')
    results = analyzer.test_inertia_hypothesis(
        data['updates'],
        data['users']
    )

    # Step 3: Visualization
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 70)

    analyzer.plot_results(data['updates'], data['users'], Path('data'))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    significant_tests = sum([
        results['update_magnitude']['significant'],
        results['correlation']['significant'] and results['correlation']['rho'] < 0,
        results['frequency']['significant']
    ])

    print(f"\n{significant_tests}/3 tests support epistemic inertia hypothesis")

    if significant_tests >= 2:
        print("\n✓✓✓ STRONG EVIDENCE FOR EPISTEMIC INERTIA ✓✓✓")
        print("\nHigh-mass traders show:")
        if results['update_magnitude']['significant']:
            print("  ✓ Smaller belief updates")
        if results['correlation']['significant'] and results['correlation']['rho'] < 0:
            print("  ✓ Negative mass-update correlation")
        if results['frequency']['significant']:
            print("  ✓ Lower trading frequency")

        print("\nThis validates the VFE Hamiltonian mass matrix theory!")
        print("Results are ready for publication.")

    elif significant_tests == 1:
        print("\n⚠️  MIXED EVIDENCE")
        print("Some tests significant, but not all.")
        print("Consider collecting more data or refining mass proxies.")

    else:
        print("\n✗ NO SIGNIFICANT EVIDENCE")
        print("Epistemic inertia not detected in this dataset.")
        print("\nPossible reasons:")
        print("  1. Insufficient statistical power (too few users/bets)")
        print("  2. Mass proxies don't capture true epistemic mass")
        print("  3. Prediction markets work differently than forecasting")
        print("  4. Theory needs refinement")

    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


def main():
    """Parse arguments and run pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test epistemic inertia using Manifold Markets data"
    )
    parser.add_argument(
        '--max-markets',
        type=int,
        default=100,
        help='Maximum number of markets to fetch (default: 100)'
    )
    parser.add_argument(
        '--min-volume',
        type=int,
        default=100,
        help='Minimum market volume to include (default: 100)'
    )

    args = parser.parse_args()

    results = run_full_pipeline(
        max_markets=args.max_markets,
        min_volume=args.min_volume
    )

    if results is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
