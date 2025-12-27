#!/usr/bin/env python3
"""
Quick test to see if authentication unlocks Metaculus prediction data.

Usage:
    python test_auth.py
"""

import requests

# Your API token
API_TOKEN = "0cf240b19ca7a6477de3b6dc73a2dabfa9e9e355"

def test_authenticated_access():
    """Test if authentication unlocks prediction endpoints."""
    print("=" * 70)
    print("TESTING METACULUS AUTHENTICATED API ACCESS")
    print("=" * 70)

    session = requests.Session()
    session.headers.update({
        'Authorization': f'Token {API_TOKEN}'
    })

    # Step 1: Get a sample question
    print("\nStep 1: Fetching a sample resolved question...")
    url = "https://www.metaculus.com/api/posts/"
    params = {
        'has_group': 'false',
        'forecast_type': 'binary',
        'status': 'resolved',
        'limit': 1
    }

    response = session.get(url, params=params, timeout=30)
    data = response.json()

    if not data.get('results'):
        print("✗ No questions found")
        return

    question = data['results'][0]['question']
    question_id = question['id']
    title = question.get('title', '')[:60]

    print(f"✓ Found question {question_id}: {title}...")

    # Step 2: Try prediction endpoints with authentication
    print("\nStep 2: Testing prediction endpoints WITH authentication...")

    endpoints_to_test = [
        f"https://www.metaculus.com/api/questions/{question_id}/predict/",
        f"https://www.metaculus.com/api/questions/{question_id}/predictions/",
        f"https://www.metaculus.com/api/questions/{question_id}/prediction-timeseries/",
        f"https://www.metaculus.com/api/questions/{question_id}/forecast-history/",
        f"https://www.metaculus.com/api/posts/{question['id']}/predictions/",
    ]

    for endpoint in endpoints_to_test:
        endpoint_name = endpoint.split('/')[-2]
        try:
            resp = session.get(endpoint, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n✓✓✓ SUCCESS: {endpoint_name}")
                print(f"  Status: {resp.status_code}")
                print(f"  Data type: {type(data)}")
                if isinstance(data, list):
                    print(f"  Items: {len(data)}")
                    if data:
                        print(f"  First item keys: {list(data[0].keys())[:10]}")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")

                # Save successful response
                import json
                filename = f"auth_test_{endpoint_name}_response.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  ✓ Saved to: {filename}")
                return True  # Found working endpoint!
            else:
                print(f"✗ {resp.status_code}: {endpoint_name}")
        except Exception as e:
            print(f"✗ ERROR: {endpoint_name} - {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("⚠️  Authentication did NOT unlock prediction endpoints")
    print("All endpoints still return 404/403")
    print("\nNext steps:")
    print("  1. Email Metaculus (api-requests@metaculus.com)")
    print("  2. Ask specifically about prediction timeseries access")
    print("  3. Mention you have an API token but can't access predictions")
    return False


if __name__ == '__main__':
    test_authenticated_access()
