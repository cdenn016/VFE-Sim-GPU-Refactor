#!/usr/bin/env python3
"""
Check what prediction/forecast data is actually available in Metaculus questions.
"""

import requests
import json
from pprint import pprint


def analyze_question_data():
    """Analyze what forecast/prediction data exists in question objects."""
    print("=" * 70)
    print("ANALYZING AVAILABLE DATA IN QUESTION OBJECTS")
    print("=" * 70)

    # Fetch a sample question
    url = "https://www.metaculus.com/api/posts/"
    params = {
        'has_group': 'false',
        'forecast_type': 'binary',
        'status': 'resolved',
        'limit': 1
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    post = data['results'][0]
    question = post['question']

    print(f"\nQuestion ID: {question['id']}")
    print(f"Title: {question.get('title', '')[:60]}...")
    print(f"Status: {post['status']}")
    print(f"Forecasters: {post['nr_forecasters']}")
    print(f"Forecast count: {post['forecasts_count']}")

    print("\n" + "=" * 70)
    print("QUESTION OBJECT STRUCTURE (35 keys)")
    print("=" * 70)

    # Print all keys in question object
    for key in sorted(question.keys()):
        value = question[key]
        if isinstance(value, dict):
            print(f"\n{key}: (dict with {len(value)} keys)")
            for subkey in list(value.keys())[:5]:
                print(f"    - {subkey}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more")
        elif isinstance(value, list):
            print(f"\n{key}: (list with {len(value)} items)")
            if value and isinstance(value[0], dict):
                print(f"    First item keys: {list(value[0].keys())[:5]}")
        else:
            val_str = str(value)
            if len(val_str) > 60:
                val_str = val_str[:60] + "..."
            print(f"\n{key}: {val_str}")

    print("\n" + "=" * 70)
    print("LOOKING FOR AGGREGATE FORECAST DATA")
    print("=" * 70)

    # Check for forecast/prediction related fields
    forecast_keys = [
        'aggregations', 'community_prediction', 'my_predictions',
        'prediction', 'latest_community_prediction',
        'forecaster_count', 'forecast_values', 'resolution_criteria'
    ]

    found_data = False
    for key in forecast_keys:
        if key in question:
            print(f"\n✓ Found '{key}':")
            value = question[key]
            if isinstance(value, dict):
                print(f"  Keys: {list(value.keys())}")
                pprint(value, indent=4, depth=2)
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if value:
                    print(f"  First item: {value[0]}")
            else:
                print(f"  Value: {value}")
            found_data = True

    if not found_data:
        print("\n⚠️  No aggregate forecast data found in question object")

    print("\n" + "=" * 70)
    print("POST-LEVEL DATA (outside question object)")
    print("=" * 70)

    # Check post-level fields
    post_forecast_keys = ['forecasts_count', 'nr_forecasters', 'vote']

    for key in post_forecast_keys:
        if key in post:
            print(f"\n✓ Found '{key}' in post:")
            value = post[key]
            if isinstance(value, dict):
                pprint(value, indent=4)
            else:
                print(f"  Value: {value}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print("\nAvailable data points per question:")
    print("  ✓ nr_forecasters: Number of unique forecasters")
    print("  ✓ forecasts_count: Total number of forecasts made")
    print("  ✓ resolution: Final outcome (0 or 1 for binary)")
    print("  ✓ scheduled_close_time: When question closed")
    print("  ✓ scheduled_resolve_time: When it resolved")

    print("\nNOT available without authentication:")
    print("  ✗ Individual user predictions over time")
    print("  ✗ Prediction timestamps")
    print("  ✗ User-level forecast data")

    print("\n" + "=" * 70)
    print("ALTERNATIVE ANALYSIS STRATEGIES")
    print("=" * 70)

    print("\nOption 1: Cross-sectional analysis")
    print("  Use nr_forecasters as proxy for question 'mass'")
    print("  Hypothesis: Questions with more forecasters → ???")
    print("  (This doesn't test individual epistemic inertia)")

    print("\nOption 2: Get Metaculus API credentials")
    print("  Contact Metaculus or check their documentation")
    print("  May require researcher/academic access")

    print("\nOption 3: Use different data source")
    print("  Good Judgment Open, Manifold Markets, etc.")
    print("  These may have more accessible APIs")

    print("\nOption 4: Simplified hypothesis test")
    print("  Instead of testing individual inertia,")
    print("  test whether high-forecaster questions behave differently")
    print("  (Aggregate-level effect instead of individual-level)")

    print("=" * 70)


if __name__ == '__main__':
    analyze_question_data()
