#!/usr/bin/env python3
"""
Check if RESOLVED questions have aggregate prediction history.

This is the key question: Can we get community prediction timeseries
to test aggregate-level epistemic inertia?
"""

import requests
import json
from pprint import pprint


def check_resolved_question_aggregations():
    """Check if resolved questions have aggregations.history data."""
    print("=" * 70)
    print("CHECKING RESOLVED QUESTIONS FOR AGGREGATE PREDICTION HISTORY")
    print("=" * 70)

    # Fetch a resolved binary question
    url = "https://www.metaculus.com/api/posts/"
    params = {
        'has_group': 'false',
        'forecast_type': 'binary',
        'status': 'resolved',  # RESOLVED not open
        'limit': 5
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    results = data.get('results', [])

    if not results:
        print("No resolved questions found")
        return

    print(f"\nFound {len(results)} resolved questions")
    print("\nChecking each for aggregate prediction history...\n")

    found_history = False

    for i, post in enumerate(results[:5]):
        question = post['question']
        q_id = question['id']
        title = question.get('title', '')[:50]

        print(f"\n{i+1}. Question {q_id}: {title}...")
        print(f"   Status: {post['status']}")
        print(f"   Forecasters: {post.get('nr_forecasters', 0)}")
        print(f"   Forecasts: {post.get('forecasts_count', 0)}")

        # Check aggregations
        aggregations = question.get('aggregations', {})

        if not aggregations:
            print("   ✗ No aggregations field")
            continue

        unweighted = aggregations.get('unweighted', {})

        if not unweighted:
            print("   ✗ No unweighted aggregation")
            continue

        history = unweighted.get('history', [])
        latest = unweighted.get('latest')

        print(f"   Aggregations.unweighted.history: {len(history)} entries")
        print(f"   Aggregations.unweighted.latest: {latest}")

        if history:
            found_history = True
            print(f"   ✓✓✓ FOUND PREDICTION HISTORY! ✓✓✓")
            print(f"\n   First few entries:")
            for j, entry in enumerate(history[:3]):
                print(f"   [{j}] {entry}")

            # Save full data for this question
            filename = f'resolved_question_{q_id}_aggregations.json'
            with open(filename, 'w') as f:
                json.dump({
                    'question_id': q_id,
                    'title': question.get('title'),
                    'status': post['status'],
                    'nr_forecasters': post.get('nr_forecasters'),
                    'forecasts_count': post.get('forecasts_count'),
                    'aggregations': aggregations
                }, f, indent=2)
            print(f"   ✓ Saved full data to {filename}")

        else:
            print("   ✗ History is empty")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if found_history:
        print("\n✓✓✓ SUCCESS! ✓✓✓")
        print("\nResolved questions DO have aggregate prediction history!")
        print("This means you CAN get community prediction timeseries data.")
        print("\nWhat this enables:")
        print("  1. Aggregate community prediction over time")
        print("  2. Timestamps for each aggregate update")
        print("  3. Can compute aggregate belief updates |Δp_agg|")
        print("\nMODIFIED HYPOTHESIS:")
        print("  Instead of: 'High-rep forecasters make smaller updates'")
        print("  Test: 'Questions with more forecasters show different")
        print("         aggregate update patterns (smaller updates? less volatile?)'")
        print("\nThis is NOT the same as individual epistemic inertia,")
        print("but it's a testable aggregate-level prediction from your theory!")

    else:
        print("\n⚠️  No aggregate prediction history found")
        print("Even resolved questions don't have history data")
        print("This suggests aggregations.history is not populated,")
        print("or requires authentication to access.")

    print("=" * 70)


if __name__ == '__main__':
    check_resolved_question_aggregations()
