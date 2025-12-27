"""
Metaculus Data Fetcher for Epistemic Inertia Analysis

Pulls forecasting data from Metaculus API to test the hypothesis that
high-reputation forecasters (proxy for followers/attention) exhibit
greater epistemic inertia (smaller belief updates in response to new evidence).

Theory: Mass matrix M_i = Σ_p^{-1} + Σ_j β_ji Ω Σ_j^{-1} Ω^T predicts
that agents with many followers become more rigid in beliefs.

Empirical test: Do high track-record forecasters make smaller updates?
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm


class MetaculusDataFetcher:
    """Fetch prediction data from Metaculus API."""

    BASE_URL = "https://www.metaculus.com/api/posts"

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()

    def fetch_questions(self,
                       status: str = "resolved",
                       min_predictions: int = 50,
                       max_questions: int = 100) -> List[Dict]:
        """
        Fetch resolved binary questions with sufficient predictions.

        Args:
            status: Question status ('resolved', 'closed', 'open')
            min_predictions: Minimum number of predictions required
            max_questions: Maximum number of questions to fetch

        Returns:
            List of question metadata dictionaries
        """
        print(f"Fetching {status} questions with ≥{min_predictions} predictions...")

        questions = []
        offset = 0
        limit = 100

        while len(questions) < max_questions:
            # Fetch batch of questions - new API structure
            url = f"{self.BASE_URL}/"
            params = {
                'has_group': 'false',
                'forecast_type': 'binary',
                'status': 'resolved' if status == 'resolved' else 'open',
                'limit': limit,
                'offset': offset,
                'order_by': '-hotness'
            }

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Error fetching questions at offset {offset}: {e}")
                print(f"  Response status: {response.status_code if 'response' in locals() else 'N/A'}")
                print(f"  URL: {url}")
                print(f"  Params: {params}")
                break

            results = data.get('results', [])
            if not results:
                print(f"No more results at offset {offset}")
                break

            # Extract question data from new API format
            for item in results:
                if len(questions) >= max_questions:
                    break

                # New API wraps question in 'question' field
                q = item.get('question')
                if not q:
                    continue

                # Check if binary
                question_type = q.get('type', '')
                if question_type != 'binary':
                    continue

                # Check prediction count
                pred_count = item.get('nr_forecasters', 0)
                if pred_count < min_predictions:
                    continue

                questions.append({
                    'id': q['id'],
                    'title': q.get('title', ''),
                    'url': item.get('url', ''),
                    'created_time': q.get('created_at'),
                    'publish_time': q.get('published_at'),
                    'resolve_time': q.get('scheduled_resolve_time'),
                    'resolution': q.get('resolution'),
                    'prediction_count': pred_count,
                    'forecaster_count': pred_count
                })

            offset += limit
            time.sleep(1)  # Rate limiting

        print(f"Found {len(questions)} questions meeting criteria")
        return questions

    def fetch_prediction_timeseries(self, question_id: int) -> List[Dict]:
        """
        Fetch all predictions for a specific question.

        Args:
            question_id: Metaculus question ID

        Returns:
            List of prediction dictionaries with timestamps
        """
        # New API endpoint for predictions
        url = f"https://www.metaculus.com/api/questions/{question_id}/predict/"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            predictions = []

            # The new API may return predictions in different formats
            # Try multiple possible structures
            if isinstance(data, list):
                # List of predictions
                for entry in data:
                    predictions.append({
                        'question_id': question_id,
                        'user_id': entry.get('user_id') or entry.get('user', {}).get('id'),
                        'time': entry.get('t') or entry.get('created_at'),
                        'prediction': entry.get('x') or entry.get('prediction', {}).get('full', {}).get('q2'),
                        'created_time': entry.get('created_time') or entry.get('created_at')
                    })
            elif isinstance(data, dict):
                # Single prediction or wrapped format
                if 'results' in data:
                    for entry in data['results']:
                        predictions.append({
                            'question_id': question_id,
                            'user_id': entry.get('user_id') or entry.get('user', {}).get('id'),
                            'time': entry.get('t') or entry.get('created_at'),
                            'prediction': entry.get('x') or entry.get('prediction', {}).get('full', {}).get('q2'),
                            'created_time': entry.get('created_time') or entry.get('created_at')
                        })

            return predictions

        except Exception as e:
            print(f"Error fetching predictions for question {question_id}: {e}")
            return []

    def fetch_user_stats(self, user_id: int) -> Optional[Dict]:
        """
        Fetch forecaster statistics (track record, reputation).

        Args:
            user_id: Metaculus user ID

        Returns:
            User statistics dictionary
        """
        # New API endpoint for users
        url = f"https://www.metaculus.com/api/users/{user_id}/"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract score from new API structure
            score = 0
            if 'scores' in data:
                # New API structure
                scores = data.get('scores', {})
                score = scores.get('baseline_score') or scores.get('peer_score') or 0
            elif 'track_record' in data:
                # Old API structure
                score = data.get('track_record', {}).get('score', 0)

            return {
                'user_id': user_id,
                'username': data.get('username'),
                'track_record': score,
                'question_count': data.get('question_count') or data.get('num_questions', 0),
                'prediction_count': data.get('prediction_count') or data.get('num_predictions', 0),
                'medal_count': data.get('medal_count', 0),
                'peer_score': data.get('peer_score') or data.get('scores', {}).get('peer_score'),
                'coverage': data.get('coverage')
            }

        except Exception as e:
            # User data may not be public or user doesn't exist
            return None

    def compute_updates(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute belief updates (Δp) from prediction time series.

        For each forecaster, compute sequential updates:
        Δp_t = |p_t - p_{t-1}|

        Args:
            predictions: DataFrame with columns [question_id, user_id, time, prediction]

        Returns:
            DataFrame with update magnitudes
        """
        # Sort by user and time
        predictions = predictions.sort_values(['user_id', 'question_id', 'time'])

        # Compute update magnitudes
        predictions['prev_prediction'] = predictions.groupby(
            ['user_id', 'question_id']
        )['prediction'].shift(1)

        predictions['update_magnitude'] = abs(
            predictions['prediction'] - predictions['prev_prediction']
        )

        # Filter to actual updates (drop first prediction per user-question)
        updates = predictions[predictions['prev_prediction'].notna()].copy()

        # Compute time since last update
        predictions['prev_time'] = predictions.groupby(
            ['user_id', 'question_id']
        )['time'].shift(1)

        updates['time_delta'] = pd.to_datetime(updates['time']) - pd.to_datetime(
            updates['prev_time']
        )
        updates['time_delta_hours'] = updates['time_delta'].dt.total_seconds() / 3600

        return updates

    def run_full_pipeline(self,
                         max_questions: int = 100,
                         min_predictions: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Run complete data collection pipeline.

        1. Fetch resolved questions
        2. Fetch prediction time series for each question
        3. Fetch user statistics for active forecasters
        4. Compute update magnitudes
        5. Save to CSV files

        Args:
            max_questions: Maximum number of questions to fetch
            min_predictions: Minimum predictions per question

        Returns:
            Dictionary with DataFrames: questions, predictions, updates, users
        """
        # Step 1: Fetch questions
        questions = self.fetch_questions(
            status='resolved',
            min_predictions=min_predictions,
            max_questions=max_questions
        )
        questions_df = pd.DataFrame(questions)

        # Step 2: Fetch predictions for each question
        all_predictions = []
        print(f"\nFetching prediction time series for {len(questions)} questions...")

        for q in tqdm(questions):
            preds = self.fetch_prediction_timeseries(q['id'])
            all_predictions.extend(preds)
            time.sleep(0.5)  # Rate limiting

        predictions_df = pd.DataFrame(all_predictions)
        print(f"Collected {len(predictions_df)} total predictions")

        # Check if we have any predictions
        if len(predictions_df) == 0:
            print("\n⚠️  WARNING: No predictions collected!")
            print("This could be due to:")
            print("  1. API changes - the endpoint structure may have changed")
            print("  2. No questions matched the criteria")
            print("  3. Rate limiting or authentication issues")
            print("\nReturning empty datasets.")

            return {
                'questions': questions_df,
                'predictions': pd.DataFrame(),
                'users': pd.DataFrame(),
                'updates': pd.DataFrame(),
                'metadata': {
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'num_questions': len(questions_df),
                    'num_predictions': 0,
                    'num_users': 0,
                    'num_updates': 0,
                    'min_predictions': min_predictions,
                    'max_questions': max_questions
                }
            }

        # Step 3: Fetch user statistics
        unique_users = predictions_df['user_id'].dropna().unique()
        print(f"\nFetching statistics for {len(unique_users)} forecasters...")

        user_stats = []
        for user_id in tqdm(unique_users):
            stats = self.fetch_user_stats(int(user_id))
            if stats:
                user_stats.append(stats)
            time.sleep(0.3)  # Rate limiting

        users_df = pd.DataFrame(user_stats)
        print(f"Collected stats for {len(users_df)} users")

        # Step 4: Compute updates
        print("\nComputing belief updates...")
        updates_df = self.compute_updates(predictions_df)
        print(f"Computed {len(updates_df)} update events")

        # Step 5: Save to files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        questions_df.to_csv(
            self.output_dir / f'questions_{timestamp}.csv',
            index=False
        )
        predictions_df.to_csv(
            self.output_dir / f'predictions_{timestamp}.csv',
            index=False
        )
        users_df.to_csv(
            self.output_dir / f'users_{timestamp}.csv',
            index=False
        )
        updates_df.to_csv(
            self.output_dir / f'updates_{timestamp}.csv',
            index=False
        )

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_questions': len(questions_df),
            'num_predictions': len(predictions_df),
            'num_users': len(users_df),
            'num_updates': len(updates_df),
            'min_predictions': min_predictions,
            'max_questions': max_questions
        }

        with open(self.output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nData saved to {self.output_dir}/")
        print(f"  - {len(questions_df)} questions")
        print(f"  - {len(predictions_df)} predictions")
        print(f"  - {len(users_df)} users")
        print(f"  - {len(updates_df)} updates")

        return {
            'questions': questions_df,
            'predictions': predictions_df,
            'users': users_df,
            'updates': updates_df,
            'metadata': metadata
        }


if __name__ == '__main__':
    # Run data collection
    fetcher = MetaculusDataFetcher(output_dir='data')

    data = fetcher.run_full_pipeline(
        max_questions=100,
        min_predictions=50
    )

    print("\nData collection complete!")
    print(f"Ready for analysis in analyze_data.py")
