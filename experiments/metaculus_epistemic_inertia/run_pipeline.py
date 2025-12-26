"""
Master Pipeline for Metaculus Epistemic Inertia Analysis

Runs complete analysis from data collection to publication figures:
1. Fetch data from Metaculus API
2. Detect news events (temporal clusters)
3. Run statistical analysis (epistemic inertia regressions)
4. Generate publication figures

Theory tested:
    M_i = Σ_p^{-1} + Σ_j β_ji Ω Σ_j^{-1} Ω^T
    ⟹ High attention/followers → Greater mass → Smaller updates

Empirical prediction:
    High track_record → Smaller |Δp| in response to news events
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse


def run_full_pipeline(max_questions: int = 100,
                     min_predictions: int = 50,
                     skip_fetch: bool = False,
                     output_dir: str = None):
    """
    Run complete analysis pipeline.

    Args:
        max_questions: Maximum number of questions to fetch
        min_predictions: Minimum predictions per question
        skip_fetch: Skip data fetching (use existing data)
        output_dir: Base directory for outputs (default: timestamped)
    """
    print("="*70)
    print("METACULUS EPISTEMIC INERTIA ANALYSIS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup directories
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"results_{timestamp}"

    base_path = Path(output_dir)
    data_dir = base_path / "data"
    tables_dir = base_path / "tables"
    figures_dir = base_path / "figures"
    logs_dir = base_path / "logs"

    for d in [data_dir, tables_dir, figures_dir, logs_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Step 1: Fetch data
    if not skip_fetch:
        print("\n" + "="*70)
        print("STEP 1: FETCHING DATA FROM METACULUS API")
        print("="*70)

        from fetch_data import MetaculusDataFetcher

        fetcher = MetaculusDataFetcher(output_dir=str(data_dir))
        data = fetcher.run_full_pipeline(
            max_questions=max_questions,
            min_predictions=min_predictions
        )

        print(f"\n✓ Data collection complete")
        print(f"  - {len(data['questions'])} questions")
        print(f"  - {len(data['predictions'])} predictions")
        print(f"  - {len(data['users'])} users")
        print(f"  - {len(data['updates'])} updates")
    else:
        print("\n" + "="*70)
        print("STEP 1: SKIPPED (using existing data)")
        print("="*70)

    # Step 2: Detect events
    print("\n" + "="*70)
    print("STEP 2: DETECTING NEWS EVENTS")
    print("="*70)

    from detect_events import main as detect_events_main

    panel, events = detect_events_main(
        data_dir=str(data_dir),
        output_dir=str(data_dir)
    )

    print(f"\n✓ Event detection complete")
    print(f"  - {len(events)} significant events")
    print(f"  - {len(panel)} panel observations")

    # Step 3: Statistical analysis
    print("\n" + "="*70)
    print("STEP 3: STATISTICAL ANALYSIS")
    print("="*70)

    from analyze_data import main as analyze_main

    models, effect_sizes, robustness = analyze_main(
        data_dir=str(data_dir),
        output_dir=str(tables_dir)
    )

    print(f"\n✓ Statistical analysis complete")

    # Step 4: Generate figures
    print("\n" + "="*70)
    print("STEP 4: GENERATING PUBLICATION FIGURES")
    print("="*70)

    from visualize_results import main as visualize_main

    visualize_main(
        data_dir=str(data_dir),
        output_dir=str(figures_dir)
    )

    print(f"\n✓ Visualization complete")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {base_path}/")
    print(f"\nData:")
    print(f"  - {data_dir}/")
    print(f"\nTables:")
    print(f"  - {tables_dir}/regression_table.tex")
    print(f"\nFigures:")
    print(f"  - {figures_dir}/fig1_main_effect.pdf")
    print(f"  - {figures_dir}/fig2_distributions.pdf")
    print(f"  - {figures_dir}/fig3_interaction.pdf")
    print(f"  - {figures_dir}/fig4_examples.pdf")
    print(f"  - {figures_dir}/fig5_quantiles.pdf")

    print(f"\nKey Finding:")
    if effect_sizes.get('is_significant_05'):
        direction = "NEGATIVE" if effect_sizes['coefficient'] < 0 else "POSITIVE"
        print(f"  ✓ {direction} epistemic inertia effect detected")
        print(f"  ✓ Coefficient: {effect_sizes['coefficient']:.4f}")
        print(f"  ✓ p-value: {effect_sizes['p_value']:.4f}")
        print(f"  ✓ Effect size: {effect_sizes['percent_change_1sd']:.1f}% per SD")

        if effect_sizes['coefficient'] < 0:
            print(f"\n  → High-reputation forecasters make SMALLER updates")
            print(f"  → Consistent with Hamiltonian VFE prediction!")
        else:
            print(f"\n  → High-reputation forecasters make LARGER updates")
            print(f"  → Inconsistent with mass matrix theory")
    else:
        print(f"  ✗ No significant effect detected")
        print(f"  ✗ p-value: {effect_sizes['p_value']:.4f}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'models': models,
        'effect_sizes': effect_sizes,
        'robustness': robustness,
        'output_dir': str(base_path)
    }


def main():
    """Parse arguments and run pipeline."""
    parser = argparse.ArgumentParser(
        description='Run Metaculus epistemic inertia analysis pipeline'
    )

    parser.add_argument(
        '--max-questions',
        type=int,
        default=100,
        help='Maximum number of questions to fetch (default: 100)'
    )

    parser.add_argument(
        '--min-predictions',
        type=int,
        default=50,
        help='Minimum predictions per question (default: 50)'
    )

    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip data fetching (use existing data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: timestamped results_YYYYMMDD_HHMMSS)'
    )

    args = parser.parse_args()

    results = run_full_pipeline(
        max_questions=args.max_questions,
        min_predictions=args.min_predictions,
        skip_fetch=args.skip_fetch,
        output_dir=args.output_dir
    )

    return results


if __name__ == '__main__':
    results = main()
