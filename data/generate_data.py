"""
Synthetic Salon Booking Data Generator v2.0
Generates 50,000 realistic salon booking records with:
  - Customer-level reliability traits (stable personality)
  - Temporal patterns (seasonal, day-of-week, time-of-day)
  - Branch × Service interaction effects
  - Correlated payment/lead-time preferences
  - Realistic service/branch distributions
Target: ~19% no-show rate, achievable model AUC ~0.88-0.93
"""

import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def generate_salon_data(n_records: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic salon booking records with realistic no-show correlations."""
    np.random.seed(seed)
    random.seed(seed)

    print(f"[INFO] Generating {n_records} synthetic salon booking records (v2.0)...")

    # ==================================================================
    # Constants & Configuration
    # ==================================================================
    services = [
        'Haircut', 'Color', 'Keratin', 'Facial',
        'Manicure', 'Pedicure', 'Waxing', 'Bridal',
    ]

    # Realistic service distribution for an Indian premium salon
    # Haircuts dominate, bridal is rare
    service_weights = np.array([0.32, 0.14, 0.05, 0.13, 0.12, 0.10, 0.12, 0.02])
    service_weights /= service_weights.sum()

    branches = [
        'Science City', 'Memnagar', 'Sindhu Bhavan Road',
        'Sabarmati', 'Chandkheda',
    ]

    # Flagship branches get more traffic
    branch_weights = np.array([0.28, 0.25, 0.22, 0.14, 0.11])
    branch_weights /= branch_weights.sum()

    payment_methods = ['Online Prepaid', 'Card on Arrival', 'Cash', 'UPI']

    staff_ids = [f"S{str(i).zfill(2)}" for i in range(1, 21)]

    service_duration_map = {
        'Haircut': (30, 60),    'Color': (90, 150),
        'Keratin': (120, 180),  'Facial': (45, 75),
        'Manicure': (30, 50),   'Pedicure': (40, 60),
        'Waxing': (20, 45),     'Bridal': (180, 300),
    }

    # Pricing reference (for summary stats only — not stored in CSV)
    service_price_map = {
        'Haircut': 1500, 'Color': 4000,   'Keratin': 6000,
        'Facial': 2500,  'Manicure': 1000, 'Pedicure': 1200,
        'Waxing': 800,   'Bridal': 20000,
    }

    # Salon operating hours distribution (9 AM – 8 PM last slot)
    # Peak hours: 10-12 AM and 4-7 PM
    hour_options = list(range(9, 21))
    hour_weights = np.array([
        0.04,   # 9 AM  — opening, light
        0.08,   # 10 AM — picking up
        0.10,   # 11 AM — peak morning
        0.10,   # 12 PM — peak
        0.06,   # 1 PM  — lunch lull
        0.05,   # 2 PM  — quiet
        0.07,   # 3 PM  — picking up
        0.11,   # 4 PM  — evening rush
        0.12,   # 5 PM  — peak evening
        0.11,   # 6 PM  — peak evening
        0.09,   # 7 PM  — winding down
        0.07,   # 8 PM  — last slots
    ])
    hour_weights /= hour_weights.sum()

    # Day of week weights (weekends heavier)
    dow_weights = np.array([
        0.12,   # Monday
        0.11,   # Tuesday
        0.12,   # Wednesday
        0.13,   # Thursday
        0.15,   # Friday
        0.20,   # Saturday — busiest
        0.17,   # Sunday
    ])
    dow_weights /= dow_weights.sum()

    # Date range: last 12 months (clean for monthly reporting)
    end_date = datetime(2026, 3, 16)
    start_date = end_date - timedelta(days=365)

    # ==================================================================
    # No-Show Effect Maps (centralized for clarity)
    # ==================================================================

    # Payment method effects on no-show probability
    payment_noshow_effect = {
        'Cash':             +0.10,
        'Card on Arrival':  +0.05,
        'UPI':              -0.04,
        'Online Prepaid':   -0.10,
    }

    # Branch-level base effects
    branch_noshow_effect = {
        'Science City':         +0.03,
        'Memnagar':             -0.04,
        'Sindhu Bhavan Road':    0.00,
        'Sabarmati':            +0.02,
        'Chandkheda':           +0.05,
    }

    # Service-level base effects
    service_noshow_effect = {
        'Bridal':   -0.15,
        'Keratin':  -0.06,
        'Color':    -0.04,
        'Facial':    0.00,
        'Haircut':  +0.03,
        'Manicure': +0.05,
        'Pedicure': +0.05,
        'Waxing':   +0.02,
    }

    # Branch × Service interaction effects
    # Realistic: some branches are worse for certain services
    branch_service_effect = {
        # Chandkheda: worst for low-commit services (walk-in area)
        ('Chandkheda', 'Manicure'):         +0.06,
        ('Chandkheda', 'Pedicure'):         +0.06,
        ('Chandkheda', 'Waxing'):           +0.05,
        ('Chandkheda', 'Haircut'):          +0.03,
        ('Chandkheda', 'Facial'):           +0.02,

        # Memnagar: premium area, even haircuts are reliable
        ('Memnagar', 'Haircut'):            -0.03,
        ('Memnagar', 'Color'):              -0.03,
        ('Memnagar', 'Keratin'):            -0.02,
        ('Memnagar', 'Bridal'):             -0.03,
        ('Memnagar', 'Facial'):             -0.02,

        # Science City: mixed area, facials often skipped
        ('Science City', 'Facial'):         +0.04,
        ('Science City', 'Manicure'):       +0.03,
        ('Science City', 'Haircut'):        +0.02,
        ('Science City', 'Pedicure'):       +0.02,

        # Sabarmati: budget-conscious, low-commit services risky
        ('Sabarmati', 'Waxing'):            +0.04,
        ('Sabarmati', 'Manicure'):          +0.05,
        ('Sabarmati', 'Pedicure'):          +0.04,
        ('Sabarmati', 'Haircut'):           +0.02,

        # Sindhu Bhavan Road: corporate area, evening services skip
        ('Sindhu Bhavan Road', 'Facial'):    +0.02,
        ('Sindhu Bhavan Road', 'Manicure'):  +0.02,
        ('Sindhu Bhavan Road', 'Keratin'):   -0.02,
        ('Sindhu Bhavan Road', 'Bridal'):    -0.02,
        ('Sindhu Bhavan Road', 'Color'):     -0.01,
    }

    # ==================================================================
    # Customer-level traits (stable personality)
    # ==================================================================
    n_customers = 5000
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)]

    print(f"[INFO] Creating {n_customers} customer profiles with reliability traits...")

    customer_traits = {}
    for cid in customer_ids:
        # Beta distribution: skewed toward reliable (low values)
        # ~70% have trait < 0.2 (reliable)
        # ~15% have trait 0.2–0.5 (moderate risk)
        # ~10% have trait 0.5–0.8 (high risk)
        # ~5% have trait > 0.8 (chronic no-showers)
        reliability_trait = float(np.random.beta(2, 8))

        # Payment preference correlated with reliability
        if reliability_trait < 0.15:
            pay_probs = [0.45, 0.15, 0.10, 0.30]   # prefer prepaid/UPI
        elif reliability_trait < 0.40:
            pay_probs = [0.25, 0.25, 0.25, 0.25]   # mixed
        else:
            pay_probs = [0.10, 0.20, 0.50, 0.20]   # prefer cash

        # Lead time style correlated with reliability
        if reliability_trait < 0.2:
            lead_time_scale = 96    # plans ahead
        elif reliability_trait < 0.5:
            lead_time_scale = 48    # moderate
        else:
            lead_time_scale = 18    # last minute or forgets

        # Visit frequency tendency
        visit_tendency = np.random.choice(
            ['frequent', 'regular', 'occasional', 'rare'],
            p=[0.15, 0.30, 0.35, 0.20],
        )

        # Branch preference (some customers prefer specific branches)
        preferred_branch = np.random.choice(branches, p=branch_weights)

        customer_traits[cid] = {
            'reliability_trait': reliability_trait,
            'preferred_pay_probs': pay_probs,
            'lead_time_scale': lead_time_scale,
            'visit_tendency': visit_tendency,
            'preferred_branch': preferred_branch,
        }

    # Track customer history chronologically
    customer_history = defaultdict(lambda: {
        'visits': 0, 'noshows': 0, 'cancellations': 0
    })

    # ==================================================================
    # Pre-generate booking times (sorted chronologically)
    # ==================================================================
    print(f"[INFO] Generating booking timestamps with DOW weighting...")

    booking_times = []
    for _ in range(n_records):
        while True:
            random_seconds = random.randint(
                0, int((end_date - start_date).total_seconds())
            )
            candidate_dt = start_date + timedelta(seconds=random_seconds)
            dow = candidate_dt.weekday()
            # Accept/reject based on DOW weight
            if random.random() < dow_weights[dow] / dow_weights.max():
                break
        booking_times.append(candidate_dt)

    # Sort chronologically for realistic cumulative history
    booking_times.sort()

    # ==================================================================
    # Generate bookings
    # ==================================================================
    print(f"[INFO] Generating booking records...")

    records = []

    for i, base_dt in enumerate(booking_times):
        if (i + 1) % 10000 == 0:
            print(f"  ... generated {i + 1}/{n_records} records")

        booking_id = str(uuid.uuid4())

        # ── Customer selection ───────────────────────────────────
        customer_id = random.choice(customer_ids)
        traits = customer_traits[customer_id]
        hist = customer_history[customer_id]

        # ── Service selection (weighted) ─────────────────────────
        service_type = np.random.choice(services, p=service_weights)

        # ── Branch: 70% preferred, 30% random ───────────────────
        if random.random() < 0.70:
            branch = traits['preferred_branch']
        else:
            branch = np.random.choice(branches, p=branch_weights)

        # ── Payment correlated with customer trait ───────────────
        payment_method = np.random.choice(
            payment_methods, p=traits['preferred_pay_probs']
        )

        # ── Time: weighted salon hours ───────────────────────────
        hour_of_day = int(np.random.choice(hour_options, p=hour_weights))
        minute = random.choice([0, 15, 30, 45])
        booking_datetime = base_dt.replace(
            hour=hour_of_day, minute=minute, second=0, microsecond=0
        )
        day_of_week = booking_datetime.weekday()

        # ── Lead time correlated with customer trait ─────────────
        lead_time = int(np.clip(
            np.random.exponential(scale=traits['lead_time_scale']),
            0, 720
        ))

        # Bridal bookings have much longer lead times
        if service_type == 'Bridal':
            lead_time = int(np.clip(
                np.random.normal(480, 120), 168, 720
            ))

        # ── Past history from accumulated records ────────────────
        past_visit_count = hist['visits']
        past_noshow_count = hist['noshows']
        past_cancellation_count = hist['cancellations']
        is_repeat_customer = past_visit_count > 0

        # ── Service duration ─────────────────────────────────────
        dur_low, dur_high = service_duration_map[service_type]
        service_duration_mins = random.randint(dur_low, dur_high)

        # ── Staff assignment ─────────────────────────────────────
        staff_id = random.choice(staff_ids)

        # ==============================================================
        # No-show probability — multi-factor with customer trait
        # ==============================================================

        # ── 1. Customer trait (strongest base signal) ────────────
        # trait=0.0 → +0.00, trait=0.5 → +0.14, trait=1.0 → +0.40
        trait_effect = traits['reliability_trait'] ** 1.5 * 0.40
        base_prob = 0.06 + trait_effect

        # ── 2. Historical no-show rate ───────────────────────────
        if past_visit_count >= 3:
            noshow_rate_hist = past_noshow_count / past_visit_count
            base_prob += noshow_rate_hist * 0.25
        elif past_visit_count == 0:
            base_prob += 0.08   # unknown = higher risk

        # ── 3. Lead time (non-linear) ────────────────────────────
        if lead_time < 2:
            base_prob += 0.12
        elif lead_time < 6:
            base_prob += 0.07
        elif lead_time > 336:       # > 2 weeks
            base_prob += 0.09
        elif lead_time > 168:       # > 1 week
            base_prob += 0.05

        # ── 4. Payment method ────────────────────────────────────
        base_prob += payment_noshow_effect[payment_method]

        # ── 5. Time-of-day effects ───────────────────────────────
        if hour_of_day >= 20:
            base_prob += 0.08
        elif hour_of_day >= 18:
            base_prob += 0.04
        elif hour_of_day <= 10 and day_of_week < 5:
            base_prob += 0.03       # early morning weekday reluctance

        # ── 6. Day-of-week ───────────────────────────────────────
        if day_of_week == 0:        # Monday blues
            base_prob += 0.04
        elif day_of_week in [5, 6]: # Weekend — planned, lower risk
            base_prob -= 0.03

        # ── 7. Seasonal effects ──────────────────────────────────
        month = booking_datetime.month
        if month in [7, 8, 9]:      # Monsoon — harder to travel
            base_prob += 0.04
        elif month in [11, 12]:     # Wedding/festival season
            base_prob -= 0.02

        # ── 8. Branch effect ─────────────────────────────────────
        base_prob += branch_noshow_effect[branch]

        # ── 9. Service effect ────────────────────────────────────
        base_prob += service_noshow_effect[service_type]

        # ── 10. Branch × Service interaction ─────────────────────
        base_prob += branch_service_effect.get(
            (branch, service_type), 0.0
        )

        # ── 11. Loyalty tiers ────────────────────────────────────
        if past_visit_count >= 20:
            base_prob -= 0.12
        elif past_visit_count >= 10:
            base_prob -= 0.07
        elif past_visit_count >= 5:
            base_prob -= 0.03

        # ── 12. Cancellation rate signal ─────────────────────────
        if past_visit_count >= 2:
            cancel_rate = past_cancellation_count / past_visit_count
            base_prob += cancel_rate * 0.20

        # ── 13. Two-way interaction effects ──────────────────────

        # Cash + New customer = very high risk
        if payment_method == 'Cash' and past_visit_count == 0:
            base_prob += 0.10

        # Prepaid + Bridal = near guaranteed show
        if payment_method == 'Online Prepaid' and service_type == 'Bridal':
            base_prob -= 0.08

        # Late evening + low-commitment service
        if (hour_of_day >= 19 and
                service_type in ['Manicure', 'Pedicure', 'Waxing']):
            base_prob += 0.07

        # Monday morning + new customer
        if (day_of_week == 0 and hour_of_day <= 11 and
                past_visit_count == 0):
            base_prob += 0.06

        # ── 14. Three-way interaction effects ────────────────────

        # Chronic no-shower + cash + evening (triple risk)
        if (traits['reliability_trait'] > 0.6 and
                payment_method == 'Cash' and hour_of_day >= 18):
            base_prob += 0.08

        # Weekend + premium service + prepaid (very safe combo)
        if (day_of_week in [5, 6] and
                service_type in ['Bridal', 'Keratin', 'Color'] and
                payment_method == 'Online Prepaid'):
            base_prob -= 0.06

        # Monsoon + evening + low-commit (weather + time + service)
        if (month in [7, 8, 9] and hour_of_day >= 18 and
                service_type in ['Manicure', 'Pedicure', 'Waxing']):
            base_prob += 0.04

        # ── 15. Final probability with noise ─────────────────────
        noise = np.random.normal(0, 0.015)
        final_prob = np.clip(base_prob + noise, 0.01, 0.95)

        outcome = 'No-Show' if np.random.random() < final_prob else 'Show'

        # ── Update customer history AFTER outcome ────────────────
        hist['visits'] += 1
        if outcome == 'No-Show':
            hist['noshows'] += 1
        # Random cancellation (separate from no-show, correlated with trait)
        if np.random.random() < traits['reliability_trait'] * 0.3:
            hist['cancellations'] += 1

        # ── Append record ────────────────────────────────────────
        records.append({
            'booking_id': booking_id,
            'customer_id': customer_id,
            'service_type': service_type,
            'branch': branch,
            'booking_datetime': booking_datetime,
            'booking_lead_time_hours': lead_time,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'payment_method': payment_method,
            'past_visit_count': past_visit_count,
            'past_cancellation_count': past_cancellation_count,
            'past_noshow_count': past_noshow_count,
            'is_repeat_customer': is_repeat_customer,
            'service_duration_mins': service_duration_mins,
            'staff_id': staff_id,
            'outcome': outcome,
        })

    df = pd.DataFrame(records)

    # ==================================================================
    # Validation & Summary Statistics
    # ==================================================================
    noshow_rate = (df['outcome'] == 'No-Show').mean()

    noshow_df = df[df['outcome'] == 'No-Show']
    gross_lost = noshow_df['service_type'].map(service_price_map).sum()
    avg_price = df['service_type'].map(service_price_map).mean()

    # Customer trait stats
    trait_values = [
        customer_traits[cid]['reliability_trait'] for cid in customer_ids
    ]

    print(f"\n{'='*65}")
    print(f"  DATASET SUMMARY (v2.0)")
    print(f"{'='*65}")
    print(f"  Total records:       {len(df):,}")
    print(f"  No-Show rate:        {noshow_rate:.2%}")
    print(f"  Unique customers:    {df['customer_id'].nunique():,}")
    print(f"  Date range:          {df['booking_datetime'].min().date()} → "
          f"{df['booking_datetime'].max().date()}")
    print(f"  Avg service price:   ₹{avg_price:,.0f}")
    print(f"  Gross revenue lost:  ₹{gross_lost:,.0f} "
          f"({gross_lost / 1e5:,.1f}L)")

    # ── Customer trait distribution ──────────────────────────────
    print(f"\n  👤 Customer Reliability Traits:")
    print(f"     Mean:              {np.mean(trait_values):.3f}")
    print(f"     Median:            {np.median(trait_values):.3f}")
    reliable_pct = sum(
        1 for t in trait_values if t < 0.2
    ) / len(trait_values)
    moderate_pct = sum(
        1 for t in trait_values if 0.2 <= t < 0.5
    ) / len(trait_values)
    high_risk_pct = sum(
        1 for t in trait_values if t >= 0.5
    ) / len(trait_values)
    print(f"     Reliable (<0.2):   {reliable_pct:.1%}")
    print(f"     Moderate (0.2-0.5): {moderate_pct:.1%}")
    print(f"     High risk (>0.5):  {high_risk_pct:.1%}")

    # ── Service distribution ─────────────────────────────────────
    print(f"\n  📋 Service Distribution:")
    for svc in services:
        subset = df[df['service_type'] == svc]
        rate = (subset['outcome'] == 'No-Show').mean()
        pct = len(subset) / len(df) * 100
        print(f"    {svc:12s}: {pct:5.1f}% | "
              f"No-Show: {rate:.2%} | n={len(subset):,}")

    # ── Payment method breakdown ─────────────────────────────────
    print(f"\n  💳 No-Show by Payment Method:")
    for pm in payment_methods:
        subset = df[df['payment_method'] == pm]
        rate = (subset['outcome'] == 'No-Show').mean()
        pct = len(subset) / len(df) * 100
        print(f"    {pm:20s}: {rate:.2%} "
              f"({len(subset):,} bookings, {pct:.1f}%)")

    # ── Branch breakdown ─────────────────────────────────────────
    print(f"\n  🏢 No-Show by Branch:")
    for br in branches:
        subset = df[df['branch'] == br]
        rate = (subset['outcome'] == 'No-Show').mean()
        pct = len(subset) / len(df) * 100
        print(f"    {br:25s}: {rate:.2%} | {pct:.1f}% of traffic")

    # ── Branch × Service heatmap preview ─────────────────────────
    print(f"\n  🗺️ Branch × Service No-Show Rates (top 5 / bottom 5):")
    branch_svc = df.groupby(['branch', 'service_type']).agg(
        noshow_rate=('outcome', lambda x: (x == 'No-Show').mean()),
        count=('outcome', 'count'),
    ).reset_index()
    branch_svc = branch_svc[branch_svc['count'] >= 20]

    top5 = branch_svc.nlargest(5, 'noshow_rate')
    bottom5 = branch_svc.nsmallest(5, 'noshow_rate')

    print(f"    🔴 Highest no-show combos:")
    for _, row in top5.iterrows():
        print(f"      {row['branch']:20s} × {row['service_type']:10s}: "
              f"{row['noshow_rate']:.2%} (n={row['count']})")

    print(f"    🟢 Lowest no-show combos:")
    for _, row in bottom5.iterrows():
        print(f"      {row['branch']:20s} × {row['service_type']:10s}: "
              f"{row['noshow_rate']:.2%} (n={row['count']})")

    # ── Signal strength validation ───────────────────────────────
    print(f"\n  📊 Signal Strength Validation:")

    new_cust = df[df['past_visit_count'] == 0]
    loyal_cust = df[df['past_visit_count'] >= 10]
    cash_new = df[
        (df['payment_method'] == 'Cash') &
        (df['past_visit_count'] == 0)
    ]
    prepaid_loyal = df[
        (df['payment_method'] == 'Online Prepaid') &
        (df['past_visit_count'] >= 10)
    ]
    evening_low_commit = df[
        (df['hour_of_day'] >= 19) &
        (df['service_type'].isin(['Manicure', 'Pedicure', 'Waxing']))
    ]
    weekend_prepaid_premium = df[
        (df['day_of_week'].isin([5, 6])) &
        (df['payment_method'] == 'Online Prepaid') &
        (df['service_type'].isin(['Bridal', 'Keratin', 'Color']))
    ]
    monday_morning = df[
        (df['day_of_week'] == 0) &
        (df['hour_of_day'] <= 11)
    ]
    monsoon = df[df['booking_datetime'].dt.month.isin([7, 8, 9])]

    new_ns = (new_cust['outcome'] == 'No-Show').mean()
    loyal_ns = (loyal_cust['outcome'] == 'No-Show').mean()
    cash_new_ns = (cash_new['outcome'] == 'No-Show').mean()
    prepaid_loyal_ns = (prepaid_loyal['outcome'] == 'No-Show').mean()
    evening_lc_ns = (evening_low_commit['outcome'] == 'No-Show').mean()
    wpp_ns = (weekend_prepaid_premium['outcome'] == 'No-Show').mean()
    monday_ns = (monday_morning['outcome'] == 'No-Show').mean()
    monsoon_ns = (monsoon['outcome'] == 'No-Show').mean()

    print(f"    Overall no-show rate:       {noshow_rate:.2%}")
    print(f"    New customers:              {new_ns:.2%}")
    print(f"    Loyal (10+ visits):         {loyal_ns:.2%}")
    print(f"    Cash + New:                 {cash_new_ns:.2%}")
    print(f"    Prepaid + Loyal:            {prepaid_loyal_ns:.2%}")
    print(f"    Evening + Low-commit:       {evening_lc_ns:.2%}")
    print(f"    Weekend + Prepaid + Premium: {wpp_ns:.2%}")
    print(f"    Monday morning:             {monday_ns:.2%}")
    print(f"    Monsoon months:             {monsoon_ns:.2%}")

    spread = cash_new_ns - prepaid_loyal_ns
    print(f"\n    Key spread (Cash+New vs Prepaid+Loyal): {spread:.2%}")
    if spread > 0.15:
        print(f"    ✅ Strong signal separation — model should achieve AUC > 0.85")
    elif spread > 0.08:
        print(f"    ⚠️ Moderate signal — model AUC likely 0.75-0.85")
    else:
        print(f"    ❌ Weak signal — consider increasing effect sizes")

    # ── Temporal stats ───────────────────────────────────────────
    print(f"\n  🕐 Temporal Stats:")
    print(f"     Booking hours:   {df['hour_of_day'].min()} – "
          f"{df['hour_of_day'].max()}")
    print(f"     Peak hour:       "
          f"{df['hour_of_day'].mode().iloc[0]}:00")
    print(f"     Peak day:        "
          f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][df['day_of_week'].mode().iloc[0]]}")

    # ── Day-of-week no-show rates ────────────────────────────────
    print(f"\n  📅 No-Show by Day of Week:")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for d in range(7):
        subset = df[df['day_of_week'] == d]
        rate = (subset['outcome'] == 'No-Show').mean()
        count = len(subset)
        bar = "█" * int(rate * 100)
        print(f"    {day_names[d]}: {rate:.2%} (n={count:,}) {bar}")

    print(f"{'='*65}")

    return df


def main():
    """Generate data and save to CSV."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bookings.csv"

    df = generate_salon_data(n_records=50000)
    df.to_csv(output_path, index=False)

    print(f"\n[SAVED] Data written to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n  Next steps:")
    print(f"    1. python src/model_trainer.py")
    print(f"    2. streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()