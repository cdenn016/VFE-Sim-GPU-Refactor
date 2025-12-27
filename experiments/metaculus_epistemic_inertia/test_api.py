#!/usr/bin/env python3
"""
Quick test script to verify Metaculus API connection.

This script tests the updated API endpoints without fetching large amounts of data.
"""

import requests
import json


def test_questions_endpoint():
    """Test if we can fetch questions from the new API."""
    print("Testing questions endpoint...")

    url = "https://www.metaculus.com/api/posts/"
    params = {
        'has_group': 'false',
        'forecast_type': 'binary',
        'status': 'resolved',
        'limit': 5,
        'offset': 0,
        'order_by': '-hotness'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"  Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"  ✓ Successfully fetched {len(results)} questions")

            if results:
                first_q = results[0].get('question', {})
                print(f"  Example question: '{first_q.get('title', 'N/A')[:60]}...'")
                print(f"  Question ID: {first_q.get('id')}")
                print(f"  Type: {first_q.get('type')}")
                return True
        else:
            print(f"  ✗ Failed with status {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_prediction_endpoint(question_id: int = None):
    """Test if we can fetch predictions for a question."""
    if question_id is None:
        print("\nSkipping prediction test (no question_id provided)")
        return None

    print(f"\nTesting prediction endpoint for question {question_id}...")

    url = f"https://www.metaculus.com/api/questions/{question_id}/predict/"

    try:
        response = requests.get(url, timeout=30)
        print(f"  Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Successfully fetched prediction data")
            print(f"  Data type: {type(data)}")

            if isinstance(data, list):
                print(f"  Number of predictions: {len(data)}")
            elif isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:5]}")

            return True
        else:
            print(f"  ✗ Failed with status {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_user_endpoint(user_id: int = 1):
    """Test if we can fetch user stats."""
    print(f"\nTesting user endpoint for user {user_id}...")

    url = f"https://www.metaculus.com/api/users/{user_id}/"

    try:
        response = requests.get(url, timeout=30)
        print(f"  Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Successfully fetched user data")
            print(f"  Username: {data.get('username')}")

            if 'scores' in data:
                print(f"  Scores: {data.get('scores')}")
            elif 'track_record' in data:
                print(f"  Track record: {data.get('track_record')}")

            return True
        else:
            print(f"  ✗ Failed with status {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 70)
    print("METACULUS API CONNECTION TEST")
    print("=" * 70)

    # Test questions endpoint
    questions_ok = test_questions_endpoint()

    # If questions work, try to get a question ID for further testing
    question_id = None
    if questions_ok:
        try:
            url = "https://www.metaculus.com/api/posts/"
            params = {'has_group': 'false', 'forecast_type': 'binary',
                     'status': 'resolved', 'limit': 1}
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            results = data.get('results', [])
            if results:
                question_id = results[0].get('question', {}).get('id')
        except:
            pass

    # Test predictions endpoint
    predictions_ok = test_prediction_endpoint(question_id)

    # Test user endpoint
    user_ok = test_user_endpoint(user_id=1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Questions endpoint: {'✓ PASS' if questions_ok else '✗ FAIL'}")
    print(f"  Predictions endpoint: {'✓ PASS' if predictions_ok else '✗ FAIL' if predictions_ok is False else '- SKIP'}")
    print(f"  User endpoint: {'✓ PASS' if user_ok else '✗ FAIL'}")

    if questions_ok and user_ok:
        print("\n✓ API connection successful! Ready to run data pipeline.")
    else:
        print("\n⚠️  Some endpoints failed. API may have changed further.")
        print("   Check Metaculus API documentation for updates.")

    print("=" * 70)


if __name__ == '__main__':
    main()
