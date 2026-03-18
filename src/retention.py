"""
Customer Retention Intelligence Module v2.0
Segments customers using RFM framework, calculates churn risk
with multi-signal scoring, and generates data-backed retention strategies.

Designed to work with generate_data.py v2.0 (chronological customer history).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class CustomerRetentionAnalyzer:
    """
    Analyzes customer behavior for churn risk and retention strategies.

    Uses RFM (Recency-Frequency-Monetary) framework enhanced with
    behavioral signals (no-show rate, cancellation rate, visit trend).
    """

    # ── Service pricing (must match generate_data.py / app.py) ──
    SERVICE_PRICE_MAP = {
        'Haircut': 1500, 'Color': 4000,   'Keratin': 6000,
        'Facial': 2500,  'Manicure': 1000, 'Pedicure': 1200,
        'Waxing': 800,   'Bridal': 20000,
    }

    DEFAULT_AVG_PRICE = 1800  # fallback

    # ── Segment thresholds ──────────────────────────────────────
    VIP_MIN_VISITS = 15
    VIP_MAX_NOSHOW_RATE = 0.10
    LOYAL_MIN_VISITS = 6
    AT_RISK_NOSHOW_THRESHOLD = 0.30
    AT_RISK_DORMANT_DAYS = 60

    # ── Churn risk weights ──────────────────────────────────────
    CHURN_WEIGHTS = {
        'recency':          0.25,
        'frequency_trend':  0.20,
        'noshow_rate':      0.20,
        'cancel_rate':      0.10,
        'payment_risk':     0.10,
        'tenure_ratio':     0.15,
    }

    def __init__(self, reference_date: Optional[datetime] = None):
        self.reference_date = reference_date or datetime(2026, 3, 16)
        self.customer_df: Optional[pd.DataFrame] = None
        self.strategies: List[Dict[str, str]] = []
        self._segment_summary: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Customer Profiles
    # ------------------------------------------------------------------
    def build_customer_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-customer summary profiles from booking data.

        Returns DataFrame with one row per customer including:
        - Visit stats, revenue, no-show/cancellation rates
        - RFM scores, segment, churn risk score & tier
        - Visit trend (accelerating vs declining)
        """
        df = df.copy()
        if 'booking_datetime' in df.columns:
            df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])

        # ── Map service prices ───────────────────────────────────
        df['estimated_price'] = (
            df['service_type']
            .map(self.SERVICE_PRICE_MAP)
            .fillna(self.DEFAULT_AVG_PRICE)
        )

        # ── Only count "Show" visits for revenue ─────────────────
        df['revenue'] = np.where(
            df['outcome'] == 'Show',
            df['estimated_price'],
            0,
        )

        # ── Payment risk flag ────────────────────────────────────
        df['is_risky_payment'] = df['payment_method'].isin(
            ['Cash', 'Card on Arrival']
        ).astype(int)

        # ── Core aggregation ─────────────────────────────────────
        customer_agg = df.groupby('customer_id').agg(
            # Volume
            total_bookings=('booking_id', 'count'),
            show_count=('outcome', lambda x: (x == 'Show').sum()),
            noshow_count=('outcome', lambda x: (x == 'No-Show').sum()),

            # Revenue
            total_revenue=('revenue', 'sum'),
            avg_ticket=('estimated_price', 'mean'),

            # Timing
            last_visit=('booking_datetime', 'max'),
            first_visit=('booking_datetime', 'min'),
            avg_lead_time=('booking_lead_time_hours', 'mean'),

            # Preferences
            primary_branch=(
                'branch',
                lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
            ),
            primary_service=(
                'service_type',
                lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
            ),
            primary_payment=(
                'payment_method',
                lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
            ),
            pct_risky_payment=('is_risky_payment', 'mean'),

            # Services
            unique_services=('service_type', 'nunique'),
            total_duration_mins=('service_duration_mins', 'sum'),
        ).reset_index()

        # ── Derived metrics ──────────────────────────────────────
        cdf = customer_agg
        safe_bookings = cdf['total_bookings'].clip(lower=1)

        cdf['noshow_rate'] = cdf['noshow_count'] / safe_bookings
        cdf['show_rate'] = cdf['show_count'] / safe_bookings

        cdf['days_since_last_visit'] = (
            (self.reference_date - cdf['last_visit']).dt.days
        )
        cdf['customer_tenure_days'] = (
            (cdf['last_visit'] - cdf['first_visit']).dt.days
        ).clip(lower=1)

        # ✅ Visit frequency (visits per 30 days of tenure)
        cdf['visit_frequency'] = (
            cdf['total_bookings'] / (cdf['customer_tenure_days'] / 30).clip(lower=1)
        )

        # ✅ Estimated LTV (actual revenue-based, not flat ₹850)
        cdf['estimated_ltv'] = cdf['total_revenue']

        # ✅ Projected annual value
        cdf['projected_annual_value'] = (
            cdf['visit_frequency'] * 12 * cdf['avg_ticket'] * cdf['show_rate']
        )

        # ── Visit trend analysis ─────────────────────────────────
        cdf['visit_trend'] = self._compute_visit_trends(df, cdf)

        # ── Cancellation rate (from actual booking-level data) ───
        cancel_agg = df.groupby('customer_id').agg(
            max_cancel_count=('past_cancellation_count', 'max'),
        ).reset_index()
        cdf = cdf.merge(cancel_agg, on='customer_id', how='left')
        cdf['cancellation_rate'] = (
            cdf['max_cancel_count'].fillna(0) / safe_bookings
        )

        # ── RFM Scoring ──────────────────────────────────────────
        cdf = self._compute_rfm_scores(cdf)

        # ── Segmentation ─────────────────────────────────────────
        cdf['segment'] = cdf.apply(self._assign_segment, axis=1)

        # ── Churn Risk ───────────────────────────────────────────
        cdf['churn_score'] = cdf.apply(self._compute_churn_score, axis=1)
        cdf['churn_risk'] = cdf['churn_score'].apply(self._churn_tier)

        self.customer_df = cdf
        self._segment_summary = None  # reset cache

        self._print_profile_summary(cdf)

        return cdf

    # ------------------------------------------------------------------
    # Visit Trend Analysis
    # ------------------------------------------------------------------
    def _compute_visit_trends(
        self, raw_df: pd.DataFrame, cdf: pd.DataFrame
    ) -> pd.Series:
        """
        Compute visit trend: compare first-half vs second-half frequency.

        Returns:
            Series with values: 'accelerating', 'stable', 'declining', 'new'
        """
        trends = {}

        for cid in cdf['customer_id']:
            cust_bookings = raw_df[raw_df['customer_id'] == cid]

            if len(cust_bookings) < 4:
                trends[cid] = 'new'
                continue

            sorted_dates = cust_bookings['booking_datetime'].sort_values()
            midpoint = len(sorted_dates) // 2

            first_half = sorted_dates.iloc[:midpoint]
            second_half = sorted_dates.iloc[midpoint:]

            # Compute visits per 30 days in each half
            first_span = max(
                (first_half.max() - first_half.min()).days, 1
            )
            second_span = max(
                (second_half.max() - second_half.min()).days, 1
            )

            first_freq = len(first_half) / (first_span / 30)
            second_freq = len(second_half) / (second_span / 30)

            if second_freq > first_freq * 1.2:
                trends[cid] = 'accelerating'
            elif second_freq < first_freq * 0.7:
                trends[cid] = 'declining'
            else:
                trends[cid] = 'stable'

        return cdf['customer_id'].map(trends).fillna('new')

    # ------------------------------------------------------------------
    # RFM Scoring
    # ------------------------------------------------------------------
    def _compute_rfm_scores(self, cdf: pd.DataFrame) -> pd.DataFrame:
        """Compute RFM quintile scores (1–5, 5 is best)."""
        cdf = cdf.copy()

        # Recency: lower days_since = better = higher score
        cdf['r_score'] = pd.qcut(
            cdf['days_since_last_visit'].rank(method='first'),
            q=5, labels=[5, 4, 3, 2, 1],
        ).astype(int)

        # Frequency: more visits = higher score
        cdf['f_score'] = pd.qcut(
            cdf['total_bookings'].rank(method='first'),
            q=5, labels=[1, 2, 3, 4, 5],
        ).astype(int)

        # Monetary: higher revenue = higher score
        cdf['m_score'] = pd.qcut(
            cdf['total_revenue'].rank(method='first'),
            q=5, labels=[1, 2, 3, 4, 5],
        ).astype(int)

        # Combined RFM score (weighted)
        cdf['rfm_score'] = (
            cdf['r_score'] * 0.35 +
            cdf['f_score'] * 0.35 +
            cdf['m_score'] * 0.30
        )

        return cdf

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _assign_segment(self, row) -> str:
        """
        Assign customer segment using multi-signal approach.

        Segments: VIP, Loyal, Promising, Occasional, At-Risk,
                  Hibernating, New
        """
        visits = row['total_bookings']
        noshow_rate = row['noshow_rate']
        days_since = row['days_since_last_visit']
        rfm = row.get('rfm_score', 3)
        trend = row.get('visit_trend', 'stable')
        revenue = row.get('total_revenue', 0)

        # ── VIP: High visits, very reliable, recent ──────────────
        if (visits >= self.VIP_MIN_VISITS
                and noshow_rate < self.VIP_MAX_NOSHOW_RATE
                and days_since < 45):
            return 'VIP'

        # ── Loyal: Regular visitors, decent reliability ──────────
        if visits >= self.LOYAL_MIN_VISITS and noshow_rate < 0.2:
            if days_since > self.AT_RISK_DORMANT_DAYS:
                return 'At-Risk'    # was loyal but going dormant
            return 'Loyal'

        # ── Promising: Newer but accelerating ────────────────────
        if (visits >= 3 and trend == 'accelerating'
                and noshow_rate < 0.2):
            return 'Promising'

        # ── At-Risk: High no-show OR declining OR dormant loyal ──
        if (noshow_rate > self.AT_RISK_NOSHOW_THRESHOLD
                or (visits >= 4 and trend == 'declining')
                or (visits >= self.LOYAL_MIN_VISITS
                    and days_since > self.AT_RISK_DORMANT_DAYS)):
            return 'At-Risk'

        # ── Hibernating: Haven't visited in a long time ──────────
        if days_since > 90 and visits >= 2:
            return 'Hibernating'

        # ── Occasional: Sporadic visitors ────────────────────────
        if visits >= 2:
            return 'Occasional'

        # ── New ──────────────────────────────────────────────────
        return 'New'

    # ------------------------------------------------------------------
    # Churn Risk Scoring
    # ------------------------------------------------------------------
    def _compute_churn_score(self, row) -> float:
        """
        Multi-signal churn risk score (0.0 = no risk, 1.0 = certain churn).

        Combines: recency, frequency trend, no-show rate,
                  cancellation rate, payment risk, tenure utilization.
        """
        W = self.CHURN_WEIGHTS

        # ── Recency signal (0–1) ─────────────────────────────────
        days_since = row['days_since_last_visit']
        recency_score = np.clip(days_since / 120, 0, 1)

        # ── Frequency trend signal ───────────────────────────────
        trend = row.get('visit_trend', 'stable')
        trend_scores = {
            'accelerating': 0.0,
            'stable':       0.2,
            'new':          0.4,
            'declining':    0.8,
        }
        trend_score = trend_scores.get(trend, 0.4)

        # ── No-show rate signal ──────────────────────────────────
        noshow_score = np.clip(row['noshow_rate'] / 0.5, 0, 1)

        # ── Cancellation rate signal ─────────────────────────────
        cancel_score = np.clip(row['cancellation_rate'] / 0.5, 0, 1)

        # ── Payment risk signal ──────────────────────────────────
        payment_score = row.get('pct_risky_payment', 0.5)

        # ── Tenure utilization ───────────────────────────────────
        # How much of their tenure is "active" vs dormant at end
        tenure = max(row['customer_tenure_days'], 1)
        active_pct = 1 - (days_since / max(tenure + days_since, 1))
        tenure_score = 1 - np.clip(active_pct, 0, 1)

        # ── Weighted combination ─────────────────────────────────
        churn_score = (
            recency_score   * W['recency'] +
            trend_score     * W['frequency_trend'] +
            noshow_score    * W['noshow_rate'] +
            cancel_score    * W['cancel_rate'] +
            payment_score   * W['payment_risk'] +
            tenure_score    * W['tenure_ratio']
        )

        return round(np.clip(churn_score, 0, 1), 4)

    @staticmethod
    def _churn_tier(score: float) -> str:
        if score >= 0.6:
            return 'HIGH'
        elif score >= 0.35:
            return 'MEDIUM'
        return 'LOW'

    # ------------------------------------------------------------------
    # Retention Strategies
    # ------------------------------------------------------------------
    def generate_retention_strategies(self) -> List[Dict[str, str]]:
        """
        Generate data-backed retention strategies computed from
        actual customer data patterns.
        """
        if self.customer_df is None:
            raise ValueError("Call build_customer_profiles() first")

        cdf = self.customer_df
        strategies = []

        # ── Strategy 1: At-Risk Re-engagement ────────────────────
        at_risk = cdf[cdf['segment'] == 'At-Risk']
        at_risk_high_value = at_risk[
            at_risk['estimated_ltv'] > at_risk['estimated_ltv'].median()
        ]
        dormant = at_risk[at_risk['days_since_last_visit'] > 45]

        at_risk_revenue = at_risk['estimated_ltv'].sum()
        dormant_pct = (
            len(dormant) / max(len(at_risk), 1) * 100
        )
        avg_ltv = at_risk['estimated_ltv'].mean() if len(at_risk) > 0 else 0

        # Find their most common payment method
        risky_pmt_pct = (
            at_risk['pct_risky_payment'].mean() * 100
            if len(at_risk) > 0 else 0
        )

        strategies.append({
            'strategy_name': 'At-Risk High-Value Re-engagement',
            'target_segment': f'At-Risk ({len(at_risk):,} customers)',
            'rationale': (
                f"{dormant_pct:.0f}% haven't visited in 45+ days. "
                f"Combined LTV at risk: ₹{at_risk_revenue:,.0f}. "
                f"Avg LTV per customer: ₹{avg_ltv:,.0f}. "
                f"{risky_pmt_pct:.0f}% use cash/card-on-arrival "
                f"(conversion to prepaid could reduce future no-shows)."
            ),
            'suggested_action': (
                "Tiered re-engagement: "
                "High-LTV (top 50%): Personal call from salon manager + "
                "complimentary add-on service. "
                "Others: WhatsApp campaign with 15% comeback discount + "
                "direct booking link. "
                "All: migrate to online prepayment with 5% incentive."
            ),
            'projected_impact': (
                f"Recovering 25% of dormant At-Risk customers could "
                f"generate ₹{int(at_risk_revenue * 0.25):,} in "
                f"recovered revenue."
            ),
        })

        # ── Strategy 2: New Customer Conversion ──────────────────
        new_custs = cdf[cdf['segment'] == 'New']
        total_new = len(new_custs)
        new_noshow_rate = (
            new_custs['noshow_rate'].mean() if total_new > 0 else 0
        )
        new_cash_pct = (
            new_custs['pct_risky_payment'].mean() * 100
            if total_new > 0 else 0
        )

        # Compare: new customers who became loyal
        promising = cdf[cdf['segment'] == 'Promising']
        promising_avg_visits = (
            promising['total_bookings'].mean() if len(promising) > 0 else 0
        )

        # Average revenue of a loyal customer
        loyal = cdf[cdf['segment'].isin(['Loyal', 'VIP'])]
        loyal_avg_ltv = loyal['estimated_ltv'].mean() if len(loyal) > 0 else 0

        strategies.append({
            'strategy_name': 'New Customer → Loyal Conversion',
            'target_segment': f'New ({total_new:,} customers)',
            'rationale': (
                f"No-show rate for new customers: {new_noshow_rate:.0%} "
                f"(vs {cdf['noshow_rate'].mean():.0%} overall). "
                f"{new_cash_pct:.0f}% pay with cash. "
                f"Average Loyal/VIP customer LTV: ₹{loyal_avg_ltv:,.0f}. "
                f"Converting 30% of new customers could add "
                f"₹{int(total_new * 0.3 * loyal_avg_ltv * 0.3):,} "
                f"in projected value."
            ),
            'suggested_action': (
                "First 3 Visits loyalty ladder: "
                "Visit 2 → 10% off, Visit 3 → free add-on service. "
                "Require online prepayment (reduces no-show by ~15pp). "
                "Personalized follow-up WhatsApp 5 days after first visit. "
                "Auto-suggest rebooking at checkout."
            ),
            'projected_impact': (
                f"If 30% of {total_new} new customers convert to "
                f"Occasional+ with avg 4 visits/year: "
                f"₹{int(total_new * 0.3 * 4 * cdf['avg_ticket'].mean()):,}/year "
                f"in additional revenue."
            ),
        })

        # ── Strategy 3: VIP/Loyal Retention Shield ───────────────
        vip_loyal = cdf[cdf['segment'].isin(['VIP', 'Loyal'])]
        vl_at_churn = vip_loyal[
            vip_loyal['churn_risk'].isin(['MEDIUM', 'HIGH'])
        ]
        declining_vl = vip_loyal[vip_loyal['visit_trend'] == 'declining']

        vl_total_ltv = vip_loyal['estimated_ltv'].sum()
        vl_avg_ltv = (
            vip_loyal['estimated_ltv'].mean() if len(vip_loyal) > 0 else 0
        )
        at_churn_ltv = vl_at_churn['estimated_ltv'].sum()

        strategies.append({
            'strategy_name': 'VIP & Loyal Retention Shield',
            'target_segment': (
                f"VIP & Loyal ({len(vip_loyal):,} customers, "
                f"₹{vl_total_ltv:,.0f} total LTV)"
            ),
            'rationale': (
                f"{len(vl_at_churn)} high-value customers show churn signals "
                f"(₹{at_churn_ltv:,.0f} LTV at risk). "
                f"{len(declining_vl)} show declining visit frequency. "
                f"Avg VIP/Loyal LTV: ₹{vl_avg_ltv:,.0f} — "
                f"each lost customer costs {vl_avg_ltv / self.DEFAULT_AVG_PRICE:.0f}x "
                f"a single service."
            ),
            'suggested_action': (
                "VIP Priority Program: "
                "• Priority booking slots & dedicated stylist. "
                "• Birthday month 20% discount + surprise gift. "
                "• Quarterly exclusive preview of new services. "
                "• Manager outreach at 21+ days without visit. "
                "• Annual loyalty gift at 20+ visits/year. "
                "For declining-frequency VIPs: personal call + "
                "complimentary premium service to re-engage."
            ),
            'projected_impact': (
                f"Preventing 50% of at-risk VIP/Loyal churn saves "
                f"₹{int(at_churn_ltv * 0.5):,} in lifetime value."
            ),
        })

        # ── Strategy 4: Hibernating Win-Back ─────────────────────
        hibernating = cdf[cdf['segment'] == 'Hibernating']
        if len(hibernating) > 0:
            hib_avg_visits = hibernating['total_bookings'].mean()
            hib_avg_ltv = hibernating['estimated_ltv'].mean()
            hib_total_ltv = hibernating['estimated_ltv'].sum()

            strategies.append({
                'strategy_name': 'Hibernating Customer Win-Back',
                'target_segment': (
                    f"Hibernating ({len(hibernating):,} customers)"
                ),
                'rationale': (
                    f"These customers averaged {hib_avg_visits:.1f} visits "
                    f"but haven't returned in 90+ days. "
                    f"Combined historical LTV: ₹{hib_total_ltv:,.0f}. "
                    f"Win-back campaigns typically recover 10–15% of "
                    f"hibernating customers."
                ),
                'suggested_action': (
                    "Staged win-back: "
                    "Week 1: 'We miss you' SMS with 20% discount. "
                    "Week 3: WhatsApp with before/after portfolio. "
                    "Week 5: Final offer — 30% off premium service. "
                    "Non-responders: mark as churned after 6 months."
                ),
                'projected_impact': (
                    f"Recovering 12% of {len(hibernating)} hibernating "
                    f"customers: ₹{int(hib_total_ltv * 0.12):,} in "
                    f"recovered revenue."
                ),
            })

        self.strategies = strategies
        return strategies

    # ------------------------------------------------------------------
    # Summary & At-Risk Customers
    # ------------------------------------------------------------------
    def get_segment_summary(self) -> pd.DataFrame:
        """Get segment summary with revenue and churn metrics."""
        if self.customer_df is None:
            raise ValueError("Call build_customer_profiles() first")

        if self._segment_summary is not None:
            return self._segment_summary

        summary = self.customer_df.groupby('segment').agg(
            count=('customer_id', 'count'),
            avg_visits=('total_bookings', 'mean'),
            avg_noshow_rate=('noshow_rate', 'mean'),
            avg_ltv=('estimated_ltv', 'mean'),
            total_ltv=('estimated_ltv', 'sum'),
            avg_days_since_visit=('days_since_last_visit', 'mean'),
            avg_churn_score=('churn_score', 'mean'),
            pct_high_churn=(
                'churn_risk', lambda x: (x == 'HIGH').mean()
            ),
            avg_visit_frequency=('visit_frequency', 'mean'),
            avg_ticket=('avg_ticket', 'mean'),
        ).round(2).reset_index()

        # Add percentage of total
        summary['pct_of_customers'] = (
            summary['count'] / summary['count'].sum() * 100
        ).round(1)
        summary['pct_of_revenue'] = (
            summary['total_ltv'] / summary['total_ltv'].sum() * 100
        ).round(1)

        self._segment_summary = summary
        return summary

    def get_at_risk_customers(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N at-risk customers ranked by recoverable value.
        Prioritizes: high LTV + high churn risk.
        """
        if self.customer_df is None:
            raise ValueError("Call build_customer_profiles() first")

        cdf = self.customer_df

        # ✅ Score = churn_risk_weight × LTV (prioritize high-value churners)
        candidates = cdf[
            cdf['churn_risk'].isin(['HIGH', 'MEDIUM'])
        ].copy()

        churn_weight = candidates['churn_score']
        max_ltv = max(candidates['estimated_ltv'].max(), 1)
        ltv_weight = candidates['estimated_ltv'] / max_ltv
        candidates['recovery_priority'] = (
            churn_weight * 0.5 + ltv_weight * 0.5
        )

        candidates = candidates.sort_values(
            'recovery_priority', ascending=False
        ).head(top_n)

        # ✅ Add contextual suggested action
        candidates['suggested_action'] = candidates.apply(
            self._suggest_action, axis=1
        )

        return candidates

    def _suggest_action(self, row) -> str:
        """Generate context-aware retention action per customer."""
        segment = row['segment']
        churn = row['churn_risk']
        noshow_rate = row['noshow_rate']
        days_since = row['days_since_last_visit']
        trend = row.get('visit_trend', 'stable')
        payment_risk = row.get('pct_risky_payment', 0.5)

        # ── VIP/Loyal at risk ────────────────────────────────────
        if churn == 'HIGH' and segment in ['VIP', 'Loyal']:
            return (
                "🚨 URGENT: Personal call from salon manager. "
                "Offer complimentary premium service + priority booking."
            )

        # ── Declining frequency ──────────────────────────────────
        if trend == 'declining' and segment in ['VIP', 'Loyal', 'Promising']:
            return (
                "📉 Declining visits. Send personalized 'exclusive offer' "
                "WhatsApp + schedule preferred stylist."
            )

        # ── High no-show rate ────────────────────────────────────
        if noshow_rate > 0.4:
            return (
                "⚠️ Chronic no-shower. Require prepayment for "
                "future bookings + automated reminder chain (48h, 24h, 2h)."
            )

        # ── Dormant ──────────────────────────────────────────────
        if days_since > 60:
            return (
                "💤 Dormant customer. 'We miss you' campaign: "
                "20% comeback discount + free consultation."
            )

        # ── Cash payment customers ───────────────────────────────
        if payment_risk > 0.6:
            return (
                "💳 Migrate to prepaid: Offer 5% discount for "
                "online booking + prepayment."
            )

        # ── General medium risk ──────────────────────────────────
        if churn == 'MEDIUM':
            return (
                "📱 Re-engagement: Loyalty program enrollment + "
                "15% discount on next visit."
            )

        return "Standard loyalty program check-in."

    # ------------------------------------------------------------------
    # Advanced Analytics
    # ------------------------------------------------------------------
    def get_revenue_at_risk(self) -> Dict[str, Any]:
        """Calculate total revenue at risk from churn."""
        if self.customer_df is None:
            raise ValueError("Call build_customer_profiles() first")

        cdf = self.customer_df

        high_churn = cdf[cdf['churn_risk'] == 'HIGH']
        medium_churn = cdf[cdf['churn_risk'] == 'MEDIUM']

        return {
            'high_risk_customers': len(high_churn),
            'high_risk_ltv': float(high_churn['estimated_ltv'].sum()),
            'high_risk_annual_value': float(
                high_churn['projected_annual_value'].sum()
            ),
            'medium_risk_customers': len(medium_churn),
            'medium_risk_ltv': float(medium_churn['estimated_ltv'].sum()),
            'total_at_risk_ltv': float(
                high_churn['estimated_ltv'].sum()
                + medium_churn['estimated_ltv'].sum()
            ),
        }

    def get_segment_migration_opportunities(self) -> pd.DataFrame:
        """
        Identify customers who are close to upgrading segments.
        E.g., Occasional with 5 visits → 1 more visit = Loyal.
        """
        if self.customer_df is None:
            raise ValueError("Call build_customer_profiles() first")

        cdf = self.customer_df
        opportunities = []

        # Occasional → Loyal (need 6 visits, low no-show)
        near_loyal = cdf[
            (cdf['segment'] == 'Occasional') &
            (cdf['total_bookings'] >= 4) &
            (cdf['noshow_rate'] < 0.2)
        ].copy()
        near_loyal['upgrade_target'] = 'Loyal'
        near_loyal['visits_needed'] = (
            self.LOYAL_MIN_VISITS - near_loyal['total_bookings']
        ).clip(lower=1)
        opportunities.append(near_loyal)

        # Loyal → VIP (need 15 visits, <10% no-show)
        near_vip = cdf[
            (cdf['segment'] == 'Loyal') &
            (cdf['total_bookings'] >= 12) &
            (cdf['noshow_rate'] < 0.15)
        ].copy()
        near_vip['upgrade_target'] = 'VIP'
        near_vip['visits_needed'] = (
            self.VIP_MIN_VISITS - near_vip['total_bookings']
        ).clip(lower=1)
        opportunities.append(near_vip)

        if opportunities:
            result = pd.concat(opportunities, ignore_index=True)
            return result.sort_values('visits_needed')
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _print_profile_summary(self, cdf: pd.DataFrame):
        """Print a rich summary of customer profiles."""
        print(f"\n{'='*60}")
        print(f"  CUSTOMER RETENTION ANALYSIS")
        print(f"{'='*60}")
        print(f"  Total customers: {len(cdf):,}")
        print(f"  Avg LTV:         ₹{cdf['estimated_ltv'].mean():,.0f}")
        print(f"  Total LTV:       ₹{cdf['estimated_ltv'].sum():,.0f}")
        print(f"  Avg no-show:     {cdf['noshow_rate'].mean():.2%}")

        print(f"\n  📊 Segment Breakdown:")
        for seg in ['VIP', 'Loyal', 'Promising', 'Occasional',
                     'At-Risk', 'Hibernating', 'New']:
            subset = cdf[cdf['segment'] == seg]
            if len(subset) > 0:
                pct = len(subset) / len(cdf) * 100
                ltv = subset['estimated_ltv'].mean()
                print(f"    {seg:15s}: {len(subset):>5,} "
                      f"({pct:5.1f}%) | Avg LTV ₹{ltv:>8,.0f}")

        print(f"\n  ⚠️ Churn Risk:")
        for risk in ['LOW', 'MEDIUM', 'HIGH']:
            subset = cdf[cdf['churn_risk'] == risk]
            pct = len(subset) / len(cdf) * 100
            print(f"    {risk:8s}: {len(subset):>5,} ({pct:5.1f}%)")

        print(f"\n  📈 Visit Trends:")
        for trend in ['accelerating', 'stable', 'declining', 'new']:
            subset = cdf[cdf['visit_trend'] == trend]
            pct = len(subset) / len(cdf) * 100
            print(f"    {trend:15s}: {len(subset):>5,} ({pct:5.1f}%)")

        print(f"{'='*60}")


def main():
    """Test retention analyzer with sample data."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "bookings.csv"

    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path, parse_dates=['booking_datetime'])

    analyzer = CustomerRetentionAnalyzer()
    cdf = analyzer.build_customer_profiles(df)

    strategies = analyzer.generate_retention_strategies()
    print(f"\n{'='*60}")
    print(f"  RETENTION STRATEGIES")
    print(f"{'='*60}")
    for i, s in enumerate(strategies, 1):
        print(f"\n  Strategy {i}: {s['strategy_name']}")
        print(f"  Target:     {s['target_segment']}")
        print(f"  Rationale:  {s['rationale']}")
        print(f"  Action:     {s['suggested_action']}")
        print(f"  Impact:     {s.get('projected_impact', 'N/A')}")

    # Revenue at risk
    rev_risk = analyzer.get_revenue_at_risk()
    print(f"\n  💰 Revenue at Risk:")
    print(f"    High-risk:   ₹{rev_risk['high_risk_ltv']:,.0f} "
          f"({rev_risk['high_risk_customers']} customers)")
    print(f"    Medium-risk: ₹{rev_risk['medium_risk_ltv']:,.0f} "
          f"({rev_risk['medium_risk_customers']} customers)")
    print(f"    Total:       ₹{rev_risk['total_at_risk_ltv']:,.0f}")

    # Migration opportunities
    migrations = analyzer.get_segment_migration_opportunities()
    if len(migrations) > 0:
        print(f"\n  🎯 Segment Migration Opportunities:")
        for target in migrations['upgrade_target'].unique():
            subset = migrations[migrations['upgrade_target'] == target]
            print(f"    → {target}: {len(subset)} customers "
                  f"({subset['visits_needed'].mean():.1f} avg visits needed)")


if __name__ == "__main__":
    main()