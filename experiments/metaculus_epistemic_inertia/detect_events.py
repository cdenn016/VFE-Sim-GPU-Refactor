"""
Event Detection for Epistemic Inertia Analysis

Identifies "news shocks" from temporal clusters of prediction updates.

Rationale: When new information becomes available (news event, report release, etc.),
many forecasters update their beliefs simultaneously. By identifying these clusters,
we can compare how high vs low reputation forecasters respond to the SAME stimulus.

This controls for information content and tests pure epistemic inertia:
    High reputation → Greater mass → Smaller updates
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import timedelta
from sklearn.cluster import DBSCAN
from scipy import stats
import json


class EventDetector:
    """Detect news events from temporal clusters of updates."""

    def __init__(self, updates_df: pd.DataFrame, questions_df: pd.DataFrame):
        """
        Initialize detector with update and question data.

        Args:
            updates_df: DataFrame with columns [question_id, user_id, time, update_magnitude]
            questions_df: DataFrame with question metadata
        """
        self.updates = updates_df.copy()
        self.questions = questions_df.copy()

        # Convert time to datetime
        self.updates['time'] = pd.to_datetime(self.updates['time'])

    def detect_events_dbscan(self,
                             question_id: int,
                             eps_hours: float = 6.0,
                             min_samples: int = 5) -> pd.DataFrame:
        """
        Detect update clusters using DBSCAN on temporal dimension.

        Args:
            question_id: Question to analyze
            eps_hours: Maximum time gap within cluster (hours)
            min_samples: Minimum updates to form a cluster

        Returns:
            DataFrame with cluster labels added
        """
        # Get updates for this question
        q_updates = self.updates[
            self.updates['question_id'] == question_id
        ].copy()

        if len(q_updates) < min_samples:
            q_updates['event_id'] = -1
            return q_updates

        # Convert time to hours since first update
        min_time = q_updates['time'].min()
        q_updates['hours_since_start'] = (
            q_updates['time'] - min_time
        ).dt.total_seconds() / 3600

        # Run DBSCAN on temporal dimension
        X = q_updates[['hours_since_start']].values
        clustering = DBSCAN(eps=eps_hours, min_samples=min_samples)
        q_updates['event_id'] = clustering.fit_predict(X)

        return q_updates

    def detect_events_sliding_window(self,
                                     question_id: int,
                                     window_hours: float = 12.0,
                                     min_updates: int = 5,
                                     density_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect events using sliding window density estimation.

        An event is a time window with update density > threshold × baseline.

        Args:
            question_id: Question to analyze
            window_hours: Window size for density estimation
            min_updates: Minimum updates in window to be an event
            density_threshold: Multiplier over baseline density

        Returns:
            DataFrame with event labels
        """
        # Get updates for this question
        q_updates = self.updates[
            self.updates['question_id'] == question_id
        ].copy().sort_values('time')

        if len(q_updates) < min_updates:
            q_updates['event_id'] = -1
            return q_updates

        # Compute baseline update density
        time_span = (q_updates['time'].max() - q_updates['time'].min()).total_seconds() / 3600
        baseline_density = len(q_updates) / time_span  # updates per hour

        # Sliding window
        window_delta = timedelta(hours=window_hours)
        event_id = 0
        q_updates['event_id'] = -1

        i = 0
        while i < len(q_updates):
            current_time = q_updates.iloc[i]['time']
            window_end = current_time + window_delta

            # Count updates in window
            in_window = (q_updates['time'] >= current_time) & (q_updates['time'] < window_end)
            window_updates = q_updates[in_window]

            # Check if this is an event
            window_density = len(window_updates) / window_hours
            is_event = (
                len(window_updates) >= min_updates and
                window_density >= density_threshold * baseline_density
            )

            if is_event:
                # Mark all updates in window
                q_updates.loc[in_window, 'event_id'] = event_id
                event_id += 1
                # Skip past this window
                i += len(window_updates)
            else:
                i += 1

        return q_updates

    def detect_all_events(self,
                         method: str = 'dbscan',
                         **kwargs) -> pd.DataFrame:
        """
        Detect events across all questions.

        Args:
            method: 'dbscan' or 'sliding_window'
            **kwargs: Parameters for detection method

        Returns:
            DataFrame with all updates labeled by event
        """
        print(f"Detecting events using {method} method...")

        all_labeled = []

        for qid in self.updates['question_id'].unique():
            if method == 'dbscan':
                labeled = self.detect_events_dbscan(qid, **kwargs)
            elif method == 'sliding_window':
                labeled = self.detect_events_sliding_window(qid, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            all_labeled.append(labeled)

        result = pd.concat(all_labeled, ignore_index=True)
        print(f"Detected {len(result[result['event_id'] >= 0])} updates in events")

        return result

    def summarize_events(self, labeled_updates: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize detected events.

        For each event, compute:
        - Number of forecasters who updated
        - Median update magnitude
        - Time span
        - Information content (variance in updates)

        Args:
            labeled_updates: Updates with event_id labels

        Returns:
            DataFrame with event summaries
        """
        # Filter to actual events (not noise)
        events = labeled_updates[labeled_updates['event_id'] >= 0].copy()

        if len(events) == 0:
            return pd.DataFrame()

        # Group by question and event
        event_summary = events.groupby(['question_id', 'event_id']).agg({
            'user_id': 'nunique',  # Number of forecasters
            'update_magnitude': ['median', 'std', 'mean', 'count'],
            'time': ['min', 'max']
        }).reset_index()

        # Flatten column names
        event_summary.columns = [
            'question_id', 'event_id',
            'num_forecasters', 'median_update', 'std_update', 'mean_update',
            'num_updates', 'start_time', 'end_time'
        ]

        # Compute time span
        event_summary['duration_hours'] = (
            pd.to_datetime(event_summary['end_time']) -
            pd.to_datetime(event_summary['start_time'])
        ).dt.total_seconds() / 3600

        # Information content (variance in beliefs)
        event_summary['information_content'] = event_summary['std_update']

        return event_summary

    def filter_significant_events(self,
                                  event_summary: pd.DataFrame,
                                  min_forecasters: int = 5,
                                  min_median_update: float = 0.01) -> pd.DataFrame:
        """
        Filter to events with sufficient participation and magnitude.

        Args:
            event_summary: Event summary DataFrame
            min_forecasters: Minimum number of forecasters
            min_median_update: Minimum median update magnitude

        Returns:
            Filtered event summary
        """
        filtered = event_summary[
            (event_summary['num_forecasters'] >= min_forecasters) &
            (event_summary['median_update'] >= min_median_update)
        ].copy()

        print(f"Filtered to {len(filtered)} significant events")
        return filtered

    def create_event_panel(self,
                          labeled_updates: pd.DataFrame,
                          users_df: pd.DataFrame,
                          significant_events: pd.DataFrame) -> pd.DataFrame:
        """
        Create panel dataset for regression analysis.

        Each row = one forecaster's response to one event.

        Columns:
        - event_id, question_id: Event identifier
        - user_id: Forecaster identifier
        - update_magnitude: Dependent variable (Δp)
        - track_record: Independent variable (proxy for followers)
        - prev_prediction: Control (starting belief)
        - time_delta: Control (time since last update)
        - Other user stats: Controls

        Args:
            labeled_updates: Updates with event labels
            users_df: User statistics
            significant_events: Filtered event summary

        Returns:
            Panel dataset ready for regression
        """
        # Merge to get only significant events
        event_keys = significant_events[['question_id', 'event_id']]

        panel = labeled_updates.merge(
            event_keys,
            on=['question_id', 'event_id'],
            how='inner'
        )

        # Merge user statistics
        panel = panel.merge(
            users_df,
            on='user_id',
            how='left'
        )

        # Create analysis variables
        panel['log_track_record'] = np.log1p(panel['track_record'])
        panel['log_prediction_count'] = np.log1p(panel['prediction_count'])
        panel['has_peer_score'] = panel['peer_score'].notna().astype(int)

        # Distance from 0.5 (extremeness of prior belief)
        panel['belief_extremeness'] = abs(panel['prev_prediction'] - 0.5)

        # Question-level fixed effects
        panel['question_fe'] = pd.Categorical(panel['question_id'])

        print(f"Created panel with {len(panel)} observations")
        print(f"  - {panel['user_id'].nunique()} unique forecasters")
        print(f"  - {panel['event_id'].nunique()} unique events")
        print(f"  - {panel['question_id'].nunique()} unique questions")

        return panel


def main(data_dir: str = 'data', output_dir: str = 'data'):
    """
    Run event detection pipeline.

    Args:
        data_dir: Directory with fetched data (CSVs)
        output_dir: Directory to save results
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load latest data files - check if they exist
    updates_files = sorted(data_path.glob('updates_*.csv'))
    questions_files = sorted(data_path.glob('questions_*.csv'))
    users_files = sorted(data_path.glob('users_*.csv'))

    if not updates_files or not questions_files or not users_files:
        print("\n⚠️  ERROR: No data files found!")
        print(f"Looking in: {data_path.absolute()}")
        print(f"  Updates files: {len(updates_files)}")
        print(f"  Questions files: {len(questions_files)}")
        print(f"  Users files: {len(users_files)}")
        print("\nThis means no predictions were collected from the API.")
        print("The prediction endpoint may not be publicly accessible.")
        raise FileNotFoundError("No data files found - prediction collection failed")

    updates_file = updates_files[-1]
    questions_file = questions_files[-1]
    users_file = users_files[-1]

    print(f"Loading data from {data_path}/")
    updates_df = pd.read_csv(updates_file)
    questions_df = pd.read_csv(questions_file)
    users_df = pd.read_csv(users_file)

    # Initialize detector
    detector = EventDetector(updates_df, questions_df)

    # Detect events using DBSCAN
    labeled_updates = detector.detect_all_events(
        method='dbscan',
        eps_hours=6.0,
        min_samples=5
    )

    # Summarize events
    event_summary = detector.summarize_events(labeled_updates)

    # Filter to significant events
    significant_events = detector.filter_significant_events(
        event_summary,
        min_forecasters=5,
        min_median_update=0.01
    )

    # Create regression panel
    panel = detector.create_event_panel(
        labeled_updates,
        users_df,
        significant_events
    )

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    labeled_updates.to_csv(
        output_path / f'labeled_updates_{timestamp}.csv',
        index=False
    )
    event_summary.to_csv(
        output_path / f'event_summary_{timestamp}.csv',
        index=False
    )
    significant_events.to_csv(
        output_path / f'significant_events_{timestamp}.csv',
        index=False
    )
    panel.to_csv(
        output_path / f'regression_panel_{timestamp}.csv',
        index=False
    )

    print(f"\nEvent detection complete!")
    print(f"Results saved to {output_path}/")
    print(f"  - {len(labeled_updates)} labeled updates")
    print(f"  - {len(event_summary)} total events")
    print(f"  - {len(significant_events)} significant events")
    print(f"  - {len(panel)} panel observations for regression")

    return panel, significant_events


if __name__ == '__main__':
    panel, events = main(data_dir='data', output_dir='data')
