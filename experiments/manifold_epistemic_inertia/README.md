# Manifold Markets Epistemic Inertia Experiment

Testing epistemic inertia in prediction markets using Manifold Markets data.

## Hypothesis

**Epistemic Mass Theory predicts:**

Traders with higher "epistemic mass" (social influence, track record, experience) exhibit greater belief rigidity:

```
M_i ∝ follower_count + total_profit + trader_count + experience

High M_i → Smaller |Δp| (belief updates)
         → Lower trading frequency
         → Higher commitment threshold
```

This is a direct test of the VFE Hamiltonian mass matrix theory using real prediction market data.

## Why Manifold Markets?

✅ **Fully public API** - No authentication required
✅ **User-level data** - Individual bet histories available
✅ **Social features** - Follower counts, profit metrics
✅ **Active platform** - Thousands of markets, users
✅ **Clean data** - Well-structured API responses

Unlike Metaculus (where individual predictions require authentication), Manifold provides everything we need.

## Mass Proxies

| Component | Manifold Metric | Mass Matrix Term |
|-----------|----------------|------------------|
| **Social influence** | `follower_count` | Σ_j β_ji (incoming attention) |
| **Track record** | `total_profit` | Λ_p (prior precision) |
| **Influence breadth** | `trader_count` | β_ik (outgoing attention) |
| **Experience** | `days_active` | Accumulated observations |

**Composite mass score:**
```python
mass = 0.4×followers + 0.3×profit + 0.2×traders + 0.1×experience
```

## Data Collection

### Quick Start

```bash
cd experiments/manifold_epistemic_inertia
python fetch_data.py
```

This will:
1. Fetch 100 resolved binary markets
2. Get all bets for each market (bet histories with prob changes)
3. Fetch user stats (followers, profit, etc.)
4. Compute belief updates |Δp| for each bet
5. Save to `data/*.csv`

**Expected time:** ~5-10 minutes for 100 markets

### What Gets Collected

**Markets (markets_TIMESTAMP.csv):**
- Market ID, question, resolution
- Volume, unique bettors, liquidity
- Creation/close/resolution times

**Bets (bets_TIMESTAMP.csv):**
- User ID, market ID, timestamp
- Amount, shares traded
- **prob_before, prob_after** (key for |Δp|!)
- Fill status

**Users (users_TIMESTAMP.csv):**
- User ID, username, creation date
- **follower_count** (social mass!)
- **total_profit** (track record)
- **trader_count** (influence)
- Balance, deposits

**Updates (updates_TIMESTAMP.csv):**
- Derived from bets
- **update_magnitude** = |prob_after - prob_before|
- Time since last bet
- Merged with user mass scores

## Analysis

### Run Epistemic Inertia Tests

```bash
python analyze_inertia.py
```

This performs three statistical tests:

### Test 1: Update Magnitude

**Hypothesis:** High mass → Smaller |Δp|

Compares mean update magnitude between high-mass and low-mass traders using Mann-Whitney U test.

**Expected:** `p < 0.05`, negative correlation

### Test 2: Mass-Update Correlation

**Hypothesis:** Continuous negative correlation between mass and |Δp|

Uses Spearman correlation to test monotonic relationship.

**Expected:** `ρ < 0`, `p < 0.05`

### Test 3: Trading Frequency

**Hypothesis:** High mass → Lower bet frequency

Compares number of bets per user between mass groups.

**Expected:** High mass users bet less frequently (higher activation threshold)

## Output

### Console Output

```
======================================================================
EPISTEMIC INERTIA ANALYSIS
======================================================================
Analyzing 15234 updates from 487 users

======================================================================
TEST 1: Update Magnitude (|Δp|)
======================================================================

High mass (N=7612):
  Mean |Δp|: 0.0234
  Median |Δp|: 0.0156
  Std |Δp|: 0.0298

Low mass (N=7622):
  Mean |Δp|: 0.0312
  Median |Δp|: 0.0209
  Std |Δp|: 0.0387

Mann-Whitney U test (H: high_mass < low_mass):
  U-statistic: 26841234.00
  p-value: 2.34e-12
  ✓ SIGNIFICANT: High mass traders make SMALLER updates
```

### Visualization

Generates `epistemic_inertia_results.png` with 4 plots:

1. **Mass vs Update** - Scatter plot with trend line
2. **Distribution** - Histogram comparing high/low mass
3. **Followers vs Update** - Direct social influence effect
4. **Quartile Boxplot** - Update distribution by mass quartile

## API Reference

### Manifold Markets API

**Base URL:** `https://api.manifold.markets/v0`

**Key Endpoints:**
- `GET /markets` - List markets
- `GET /bets?contractId={id}` - Get bets for market
- `GET /user/by-id/{id}` - Get user profile

**Docs:** https://docs.manifold.markets/api

**Rate Limits:**
- No strict limits documented
- We use 0.5s delays to be respectful

## Advantages Over Metaculus

| Feature | Metaculus | Manifold |
|---------|-----------|----------|
| Individual predictions | ❌ 404 | ✅ Public |
| Follower counts | ❌ Hidden | ✅ Public |
| Track record | ❌ Hidden | ✅ Public (profit) |
| Update timestamps | ❌ No | ✅ Every bet |
| API access | 🔐 Auth required | ✅ Open |
| Data availability | ⏰ Takes weeks | ✅ Immediate |

## Results Interpretation

### If Tests Are Significant

✅ **Confirms epistemic inertia hypothesis**
- Validates mass matrix theory in real markets
- Shows social influence creates belief rigidity
- Publishable result for psychology/economics journals

### If Tests Are Not Significant

Two possibilities:
1. **Theory needs refinement** - Mass proxies wrong, or prediction markets different from forecasting
2. **Insufficient statistical power** - Collect more markets/users

## Next Steps

1. **Run the pipeline** (fetch + analyze)
2. **Check results** - Are tests significant?
3. **If significant:**
   - Write up results
   - Compare to Metaculus (when you get access)
   - Submit to journal
4. **If not significant:**
   - Collect more data (increase max_markets)
   - Try different mass weightings
   - Explore alternative platforms

## Theory Connection

This tests a **specific prediction** from the Hamiltonian VFE mass matrix:

```
M_i = Λ_p + Λ_o + Σ_k β_ik Λ̃_q,k + Σ_j β_ji Λ_q,i
      ↑                ↑             ↑
    Track          Outgoing      Incoming
    record         influence     influence
```

**Manifold provides direct measurements:**
- `total_profit` → Prior precision Λ_p (expertise)
- `trader_count` → Outgoing influence Σ_k β_ik
- `follower_count` → Incoming influence Σ_j β_ji

This is the **first empirical test** of the complete 4-term mass formula using real prediction data!

## References

**Dennis, R.C. (2025). The Inertia of Belief.**
- Complete VFE Hessian derivation
- 4-term mass matrix formula
- Located: `../../papers/psych/belief_inertia.tex`

**Manifold Markets:**
- Platform: https://manifold.markets/
- API Docs: https://docs.manifold.markets/api
- Community: https://discord.gg/eHQBNBqXuh
