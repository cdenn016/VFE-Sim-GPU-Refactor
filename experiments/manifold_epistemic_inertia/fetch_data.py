"""
Manifold Markets Data Fetcher for Epistemic Inertia Analysis

Fetches betting/trading data from Manifold Markets to test the hypothesis that
users with high follower counts (proxy for social influence/mass) exhibit
greater epistemic inertia (smaller belief updates, less frequent trading).

Theory: Mass matrix M_i ∝ followerCount + totalProfit + experience
        High M_i → Smaller |Δp| per trade, lower trading frequency

API Docs: https://docs.manifold.markets/api
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm


class ManifoldDataFetcher:
    """Fetch prediction market data from Manifold Markets API."""

    BASE_URL = "https://api.manifold.markets/v0"

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()

    def fetch_markets(self,
                     limit: int = 100,
                     resolved: bool = True) -> List[Dict]:
        """
        Fetch resolved binary markets with sufficient activity.

        Args:
            limit: Maximum number of markets to fetch
            resolved: If True, fetch only resolved markets

        Returns:
            List of market metadata dictionaries
        """
        print(f"Fetching {'resolved' if resolved else 'active'} markets...")

        markets = []
        before_id = None  # Pagination cursor

        while len(markets) < limit:
            url = f"{self.BASE_URL}/markets"
            params = {
                'limit': min(100, limit - len(markets)),
            }

            if before_id:
                params['before'] = before_id

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                batch = response.json()
            except Exception as e:
                print(f"Error fetching markets: {e}")
                break

            if not batch:
                break

            for market in batch:
                # Filter for binary markets
                if market.get('outcomeType') != 'BINARY':
                    continue

                # Filter by resolution status
                if resolved and not market.get('isResolved'):
                    continue

                # Filter for markets with enough activity
                if market.get('volume', 0) < 100:  # At least $100 volume
                    continue

                markets.append({
                    'id': market['id'],
                    'question': market['question'],
                    'created_time': market['createdTime'],
                    'close_time': market.get('closeTime'),
                    'resolution_time': market.get('resolutionTime'),
                    'resolution': market.get('resolution'),
                    'volume': market.get('volume', 0),
                    'unique_bettors': market.get('uniqueBettorCount', 0),
                    'total_liquidity': market.get('totalLiquidity', 0),
                    'creator_username': market.get('creatorUsername'),
                    'url': market.get('url')
                })

                if len(markets) >= limit:
                    break

            # Set pagination cursor
            if batch:
                before_id = batch[-1]['id']

            time.sleep(0.5)  # Rate limiting

        print(f"Found {len(markets)} markets meeting criteria")
        return markets

    def fetch_bets_for_market(self, market_id: str) -> List[Dict]:
        """
        Fetch all bets for a specific market.

        Args:
            market_id: Manifold market ID

        Returns:
            List of bet dictionaries with userId, amount, probability changes
        """
        url = f"{self.BASE_URL}/bets"
        params = {'contractId': market_id}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            bets = response.json()

            formatted_bets = []
            for bet in bets:
                formatted_bets.append({
                    'market_id': market_id,
                    'bet_id': bet['id'],
                    'user_id': bet['userId'],
                    'created_time': bet['createdTime'],
                    'amount': bet.get('amount', 0),
                    'shares': bet.get('shares', 0),
                    'prob_before': bet.get('probBefore'),
                    'prob_after': bet.get('probAfter'),
                    'is_filled': bet.get('isFilled', True),
                    'is_cancelled': bet.get('isCancelled', False)
                })

            return formatted_bets

        except Exception as e:
            print(f"Error fetching bets for market {market_id}: {e}")
            return []

    def fetch_user_stats(self, user_id: str) -> Optional[Dict]:
        """
        Fetch user statistics and profile.

        Args:
            user_id: Manifold user ID

        Returns:
            User statistics dictionary including follower count, profit
        """
        url = f"{self.BASE_URL}/user/by-id/{user_id}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            user = response.json()

            # Handle creatorTraders which might be dict, int, or missing
            creator_traders = user.get('creatorTraders', 0)
            if isinstance(creator_traders, dict):
                trader_count = len(creator_traders)
            elif isinstance(creator_traders, (int, float)):
                trader_count = int(creator_traders)
            else:
                trader_count = 0

            return {
                'user_id': user_id,
                'username': user.get('username'),
                'name': user.get('name'),
                'created_time': user.get('createdTime'),
                'total_profit': user.get('profitCached', {}).get('allTime', 0),
                'follower_count': user.get('followerCountCached', 0),
                'following_count': user.get('followingCount', 0),
                'trader_count': trader_count,  # People who trade on their markets
                'balance': user.get('balance', 0),
                'total_deposits': user.get('totalDeposits', 0)
            }

        except Exception as e:
            # User may not exist or be private
            return None

    def compute_updates(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute belief updates (Δp) from betting time series.

        For each user on each market, compute sequential probability updates:
        Δp_t = |prob_after - prob_before|

        Args:
            bets_df: DataFrame with columns [market_id, user_id, created_time, prob_before, prob_after]

        Returns:
            DataFrame with update magnitudes and metadata
        """
        # Sort by user, market, and time
        bets_df = bets_df.sort_values(['user_id', 'market_id', 'created_time'])

        # Compute update magnitude for each bet
        bets_df['update_magnitude'] = abs(
            bets_df['prob_after'] - bets_df['prob_before']
        )

        # Compute time since last bet on this market
        bets_df['prev_bet_time'] = bets_df.groupby(
            ['user_id', 'market_id']
        )['created_time'].shift(1)

        bets_df['time_since_last_bet'] = pd.to_datetime(bets_df['created_time']) - pd.to_datetime(
            bets_df['prev_bet_time']
        )
        bets_df['hours_since_last_bet'] = bets_df['time_since_last_bet'].dt.total_seconds() / 3600

        # Filter to valid updates (non-zero, non-null)
        updates = bets_df[
            (bets_df['update_magnitude'].notna()) &
            (bets_df['update_magnitude'] > 0)
        ].copy()

        return updates

    def run_full_pipeline(self,
                         max_markets: int = 100,
                         min_volume: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Run complete data collection pipeline.

        1. Fetch resolved markets
        2. Fetch bets for each market
        3. Fetch user statistics for active traders
        4. Compute update magnitudes
        5. Save to CSV files

        Args:
            max_markets: Maximum number of markets to fetch
            min_volume: Minimum market volume to include

        Returns:
            Dictionary with DataFrames: markets, bets, updates, users
        """
        # Step 1: Fetch markets
        markets = self.fetch_markets(limit=max_markets, resolved=True)
        markets_df = pd.DataFrame(markets)

        # Step 2: Fetch bets for each market
        all_bets = []
        print(f"\nFetching bets for {len(markets)} markets...")

        for market in tqdm(markets):
            bets = self.fetch_bets_for_market(market['id'])
            all_bets.extend(bets)
            time.sleep(0.5)  # Rate limiting

        bets_df = pd.DataFrame(all_bets)
        print(f"Collected {len(bets_df)} total bets")

        if len(bets_df) == 0:
            print("\n⚠️  WARNING: No bets collected!")
            return {
                'markets': markets_df,
                'bets': pd.DataFrame(),
                'users': pd.DataFrame(),
                'updates': pd.DataFrame(),
                'metadata': {'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')}
            }

        # Step 3: Fetch user statistics
        unique_users = bets_df['user_id'].dropna().unique()
        print(f"\nFetching statistics for {len(unique_users)} traders...")

        user_stats = []
        for user_id in tqdm(unique_users):
            stats = self.fetch_user_stats(user_id)
            if stats:
                user_stats.append(stats)
            time.sleep(0.3)  # Rate limiting

        users_df = pd.DataFrame(user_stats)
        print(f"Collected stats for {len(users_df)} users")

        # Step 4: Compute updates
        print("\nComputing belief updates...")
        updates_df = self.compute_updates(bets_df)
        print(f"Computed {len(updates_df)} update events")

        # Step 5: Save to files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        markets_df.to_csv(
            self.output_dir / f'markets_{timestamp}.csv',
            index=False
        )
        bets_df.to_csv(
            self.output_dir / f'bets_{timestamp}.csv',
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
            'num_markets': len(markets_df),
            'num_bets': len(bets_df),
            'num_users': len(users_df),
            'num_updates': len(updates_df),
            'total_volume': markets_df['volume'].sum(),
        }

        with open(self.output_dir / f'metadata_{timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nData saved to {self.output_dir}/")
        print(f"  - {len(markets_df)} markets")
        print(f"  - {len(bets_df)} bets")
        print(f"  - {len(users_df)} users")
        print(f"  - {len(updates_df)} updates")

        return {
            'markets': markets_df,
            'bets': bets_df,
            'users': users_df,
            'updates': updates_df,
            'metadata': metadata
        }


if __name__ == '__main__':
    # Run data collection
    fetcher = ManifoldDataFetcher(output_dir='data')

    data = fetcher.run_full_pipeline(
        max_markets=100,
        min_volume=100
    )

    print("\nData collection complete!")
    print(f"Ready for epistemic inertia analysis")
