#!/usr/bin/env python3
"""
Metaculus API Explorer

This script explores the Metaculus API to find where prediction data actually lives.
The /api/questions/{id}/predict/ endpoint appears to not exist or requires auth.

Let's systematically explore the API structure.
"""

import requests
import json
from pprint import pprint


def explore_question_structure():
    """Fetch a single question and examine its full structure."""
    print("=" * 70)
    print("EXPLORING QUESTION STRUCTURE")
    print("=" * 70)

    # Get a single resolved binary question
    url = "https://www.metaculus.com/api/posts/"
    params = {
        'has_group': 'false',
        'forecast_type': 'binary',
        'status': 'resolved',
        'limit': 1
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print(f"Failed to fetch questions: {response.status_code}")
        return None

    data = response.json()
    results = data.get('results', [])

    if not results:
        print("No questions found")
        return None

    first_post = results[0]
    question = first_post.get('question', {})
    question_id = question.get('id')

    print(f"\nQuestion ID: {question_id}")
    print(f"Title: {question.get('title', 'N/A')[:60]}...")
    print(f"\nFull structure of 'post' object:")
    print("-" * 70)

    # Print top-level keys
    print("\nTop-level keys in post:")
    for key in sorted(first_post.keys()):
        value = first_post[key]
        if isinstance(value, dict):
            print(f"  {key}: {{dict with {len(value)} keys}}")
        elif isinstance(value, list):
            print(f"  {key}: [list with {len(value)} items]")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Check for prediction-related fields
    print("\n" + "=" * 70)
    print("LOOKING FOR PREDICTION DATA IN POST OBJECT")
    print("=" * 70)

    # Common places prediction data might be
    prediction_keys = [
        'predictions', 'forecasts', 'community_prediction',
        'my_predictions', 'prediction_timeseries', 'votes',
        'nr_forecasters', 'prediction_histogram'
    ]

    for key in prediction_keys:
        if key in first_post:
            print(f"\n✓ Found '{key}':")
            value = first_post[key]
            if isinstance(value, dict):
                print(f"  Type: dict")
                print(f"  Keys: {list(value.keys())[:10]}")
            elif isinstance(value, list):
                print(f"  Type: list")
                print(f"  Length: {len(value)}")
                if value:
                    print(f"  First item: {value[0]}")
            else:
                print(f"  Value: {value}")

    # Save full JSON for inspection
    with open('question_structure.json', 'w') as f:
        json.dump(first_post, f, indent=2)
    print(f"\n✓ Saved full structure to question_structure.json")

    return question_id


def try_prediction_endpoints(question_id):
    """Try various endpoint patterns to find predictions."""
    print("\n" + "=" * 70)
    print("TRYING DIFFERENT PREDICTION ENDPOINTS")
    print("=" * 70)

    endpoints = [
        f"https://www.metaculus.com/api/questions/{question_id}/predictions/",
        f"https://www.metaculus.com/api/questions/{question_id}/predict/",
        f"https://www.metaculus.com/api/questions/{question_id}/forecasts/",
        f"https://www.metaculus.com/api/questions/{question_id}/prediction_timeseries/",
        f"https://www.metaculus.com/api/questions/{question_id}/community_prediction/",
        f"https://www.metaculus.com/api/posts/{question_id}/predictions/",
        f"https://www.metaculus.com/api/posts/{question_id}/forecasts/",
    ]

    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                print(f"\n✓ SUCCESS: {endpoint}")
                data = response.json()
                print(f"  Response type: {type(data)}")
                if isinstance(data, list):
                    print(f"  Items: {len(data)}")
                    if data:
                        print(f"  First item keys: {list(data[0].keys())[:10]}")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")

                # Save successful response
                filename = endpoint.split('/')[-2] + '_response.json'
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved to: {filename}")
                return endpoint
            else:
                print(f"✗ {response.status_code}: {endpoint.split('/')[-2]}")
        except Exception as e:
            print(f"✗ ERROR: {endpoint.split('/')[-2]} - {e}")

    return None


def check_authenticated_access():
    """Check if predictions require authentication."""
    print("\n" + "=" * 70)
    print("AUTHENTICATION CHECK")
    print("=" * 70)

    print("\nPrediction data may require authentication.")
    print("To access individual prediction timeseries, you may need:")
    print("  1. Create a Metaculus account")
    print("  2. Get an API key from your account settings")
    print("  3. Pass the API key in request headers")
    print("\nAlternatively, prediction data may only be available:")
    print("  - In aggregate (community prediction)")
    print("  - For logged-in users viewing their own predictions")
    print("  - Through web scraping (not recommended)")


def main():
    """Run API exploration."""
    print("\n" * 2)
    print("*" * 70)
    print(" METACULUS API EXPLORATION - FINDING PREDICTION DATA")
    print("*" * 70)

    # Step 1: Get a question and explore its structure
    question_id = explore_question_structure()

    if question_id:
        # Step 2: Try different endpoints
        working_endpoint = try_prediction_endpoints(question_id)

        if not working_endpoint:
            # Step 3: Check authentication requirements
            check_authenticated_access()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nIf no prediction endpoints worked:")
    print("  1. Check question_structure.json for aggregate data")
    print("  2. Individual prediction timeseries may not be public")
    print("  3. May need Metaculus API key/authentication")
    print("\nNext steps:")
    print("  - Look for 'community_prediction' or aggregate forecasts")
    print("  - Consider using only aggregate data for analysis")
    print("  - Or get API credentials from Metaculus")
    print("=" * 70)


if __name__ == '__main__':
    main()
