
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import LabelEncoder


class DataPipeline:
    """Full data preprocessing and feature engineering pipeline v2.0."""

    # Columns from raw CSV
    CATEGORICAL_COLS = ['service_type', 'branch', 'payment_method', 'staff_id']

    RAW_NUMERICAL_COLS = [
        'booking_lead_time_hours', 'day_of_week', 'hour_of_day',
        'past_visit_count', 'past_cancellation_count', 'past_noshow_count',
        'service_duration_mins',
    ]

    TARGET_COL = 'outcome'

    # ✅ Domain-knowledge encodings (not learned — consistent across train/predict)
    PAYMENT_RISK_MAP = {
        'Online Prepaid':  0,    # lowest risk
        'UPI':             1,
        'Card on Arrival': 2,
        'Cash':            3,    # highest risk
    }

    SERVICE_COMMITMENT_MAP = {
        'Bridal':   6,   # highest commitment
        'Keratin':  5,
        'Color':    4,
        'Facial':   3,
        'Haircut':  2,
        'Pedicure': 2,
        'Waxing':   1,
        'Manicure': 1,   # lowest commitment
    }

    BRANCH_TIER_MAP = {
        'Memnagar':           0,   # best area — lowest no-show
        'Sindhu Bhavan Road':  1,
        'Science City':       2,
        'Sabarmati':          3,
        'Chandkheda':         4,   # highest no-show
    }

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = (
            Path(models_dir) if models_dir
            else Path(__file__).parent.parent / "models"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self._staff_noshow_rates: Dict[str, float] = {}
        self._global_noshow_rate: float = 0.19

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Add all engineered features to the DataFrame."""
        df = df.copy()

        # ── 1. Historical rates ──────────────────────────────────
        safe_visits = df['past_visit_count'].clip(lower=1)

        df['noshow_rate_historical'] = df['past_noshow_count'] / safe_visits
        df['cancellation_rate'] = df['past_cancellation_count'] / safe_visits

        # ✅ Combined unreliability score
        df['unreliability_rate'] = (
            (df['past_noshow_count'] + df['past_cancellation_count']) / safe_visits
        )

        # ── 2. Binary flags ─────────────────────────────────────
        df['is_new_customer'] = (df['past_visit_count'] == 0).astype(int)
        df['is_repeat_customer'] = (df['past_visit_count'] > 0).astype(int)

        # ✅ Loyalty tiers (gradient, not binary)
        df['is_loyal'] = (df['past_visit_count'] >= 8).astype(int)
        df['is_very_loyal'] = (df['past_visit_count'] >= 15).astype(int)
        df['is_vip'] = (
            (df['past_visit_count'] >= 15) &
            (df['noshow_rate_historical'] < 0.1)
        ).astype(int)

        # ✅ Historical risk flags
        df['is_chronic_nosho'] = (
            (df['past_visit_count'] >= 3) &
            (df['noshow_rate_historical'] > 0.3)
        ).astype(int)

        df['has_noshow_history'] = (df['past_noshow_count'] > 0).astype(int)
        df['has_cancel_history'] = (df['past_cancellation_count'] > 0).astype(int)

        # ── 3. Lead time features ───────────────────────────────
        df['is_last_minute'] = (df['booking_lead_time_hours'] < 3).astype(int)
        df['is_short_notice'] = (df['booking_lead_time_hours'] < 6).astype(int)
        df['is_same_day'] = (df['booking_lead_time_hours'] < 24).astype(int)
        df['is_far_advance'] = (df['booking_lead_time_hours'] > 168).astype(int)
        df['is_very_far_advance'] = (df['booking_lead_time_hours'] > 336).astype(int)

        # ✅ Lead time buckets (captures non-linear relationship)
        df['lead_time_bucket'] = pd.cut(
            df['booking_lead_time_hours'],
            bins=[-1, 2, 6, 24, 72, 168, 336, 720],
            labels=[0, 1, 2, 3, 4, 5, 6],
        ).astype(float).fillna(3).astype(int)

        # ✅ Log transform of lead time (diminishing returns)
        df['lead_time_log'] = np.log1p(df['booking_lead_time_hours'])

        # ── 4. Temporal features ────────────────────────────────
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        df['is_evening'] = (df['hour_of_day'] >= 18).astype(int)
        df['is_late_evening'] = (df['hour_of_day'] >= 20).astype(int)
        df['is_morning'] = (df['hour_of_day'] <= 10).astype(int)
        df['is_peak_hours'] = df['hour_of_day'].isin([11, 12, 17, 18]).astype(int)

        # ✅ Month and season
        if 'booking_datetime' in df.columns:
            df['month'] = pd.to_datetime(df['booking_datetime']).dt.month
        else:
            df['month'] = 6  # default for prediction

        df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
        df['is_wedding_season'] = df['month'].isin([11, 12, 1, 2]).astype(int)
        df['is_summer'] = df['month'].isin([4, 5, 6]).astype(int)

        # ── 5. Domain-knowledge encodings ───────────────────────
        # ✅ These encode ordinal business meaning, not arbitrary labels
        df['payment_risk_encoded'] = (
            df['payment_method'].map(self.PAYMENT_RISK_MAP).fillna(2)
        )
        df['service_commitment'] = (
            df['service_type'].map(self.SERVICE_COMMITMENT_MAP).fillna(2)
        )
        df['branch_tier'] = (
            df['branch'].map(self.BRANCH_TIER_MAP).fillna(2)
        )

        # ── 6. Interaction features ─────────────────────────────
        # ✅ These capture the key non-linear patterns from data generation

        # Payment × Customer type
        df['cash_x_new'] = (
            (df['payment_method'] == 'Cash') &
            (df['past_visit_count'] == 0)
        ).astype(int)

        df['prepaid_x_loyal'] = (
            (df['payment_method'] == 'Online Prepaid') &
            (df['past_visit_count'] >= 8)
        ).astype(int)

        df['cash_x_chronic'] = (
            (df['payment_method'] == 'Cash') &
            (df['is_chronic_nosho'] == 1)
        ).astype(int)

        # Time × Service
        df['evening_x_low_commit'] = (
            (df['hour_of_day'] >= 18) &
            (df['service_type'].isin(['Manicure', 'Pedicure', 'Waxing']))
        ).astype(int)

        df['evening_x_cash'] = (
            (df['hour_of_day'] >= 18) &
            (df['payment_method'] == 'Cash')
        ).astype(int)

        # Day × Customer type
        df['monday_morning_new'] = (
            (df['day_of_week'] == 0) &
            (df['hour_of_day'] <= 11) &
            (df['past_visit_count'] == 0)
        ).astype(int)

        # ✅ Triple interaction: evening + cash + high history
        df['evening_cash_chronic'] = (
            (df['hour_of_day'] >= 18) &
            (df['payment_method'] == 'Cash') &
            (df['noshow_rate_historical'] > 0.3)
        ).astype(int)

        # ✅ Safe combo: weekend + premium service + prepaid
        df['weekend_premium_prepaid'] = (
            (df['is_weekend'] == 1) &
            (df['service_type'].isin(['Bridal', 'Keratin', 'Color'])) &
            (df['payment_method'] == 'Online Prepaid')
        ).astype(int)

        # ── 7. Ratio & composite features ───────────────────────
        # ✅ Noshow count relative to visit count (different from rate for low counts)
        df['noshow_intensity'] = np.where(
            df['past_visit_count'] >= 3,
            df['noshow_rate_historical'],
            df['past_noshow_count'] * 0.3   # uncertain estimate for new-ish
        )

        # ✅ Visit recency proxy (higher visit count with low noshows = reliable)
        df['reliability_score'] = (
            np.log1p(df['past_visit_count']) *
            (1 - df['noshow_rate_historical'])
        )

        # ── 8. Staff-level features ─────────────────────────────
        if is_training and self.TARGET_COL in df.columns:
            # ✅ Compute staff no-show rates (smoothed)
            staff_stats = df.groupby('staff_id').agg(
                staff_total=('outcome', 'count'),
                staff_noshows=(
                    'outcome',
                    lambda x: (x == 'No-Show').sum()
                ),
            ).reset_index()
            # Bayesian smoothing: blend with global rate
            global_rate = (df[self.TARGET_COL] == 'No-Show').mean()
            smoothing = 50  # prior weight
            staff_stats['staff_noshow_rate'] = (
                (staff_stats['staff_noshows'] + smoothing * global_rate) /
                (staff_stats['staff_total'] + smoothing)
            )
            self._staff_noshow_rates = dict(
                zip(staff_stats['staff_id'], staff_stats['staff_noshow_rate'])
            )
            self._global_noshow_rate = global_rate

        df['staff_noshow_rate'] = (
            df['staff_id']
            .map(self._staff_noshow_rates)
            .fillna(self._global_noshow_rate)
        )

        # ── 9. Weakened risk score (one of many, not dominant) ──
        # ✅ Simpler version — the model should learn its own weighting
        df['risk_score_v2'] = (
            df['noshow_rate_historical'] * 0.20 +
            df['cancellation_rate'] * 0.10 +
            df['is_new_customer'] * 0.10 +
            df['payment_risk_encoded'] / 3 * 0.10 +
            df['is_evening'] * 0.05 +
            df['cash_x_new'] * 0.10
        )

        return df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def fit_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit label encoders on categorical columns."""
        df = df.copy()

        # ✅ Only LabelEncode columns that need it
        # service_type, branch, payment_method already have
        # domain encodings above — but we keep label encoding
        # for the model to also learn its own representation
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col + '_label'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Encode target
        le_target = LabelEncoder()
        df['target'] = le_target.fit_transform(df[self.TARGET_COL])
        self.label_encoders[self.TARGET_COL] = le_target

        # ✅ Build feature list (organized by category)
        self.feature_names = self._build_feature_list()

        # ✅ No StandardScaler — trees don't need it
        # Keeps SHAP values interpretable and avoids hurting tree models

        self._save_artifacts()

        return df

    def _build_feature_list(self) -> List[str]:
        """Construct the ordered list of all model features."""
        features = []

        # Raw numerical
        features += self.RAW_NUMERICAL_COLS

        # Historical rates
        features += [
            'noshow_rate_historical', 'cancellation_rate',
            'unreliability_rate',
        ]

        # Binary flags — customer type
        features += [
            'is_new_customer', 'is_repeat_customer',
            'is_loyal', 'is_very_loyal', 'is_vip',
            'is_chronic_nosho', 'has_noshow_history',
            'has_cancel_history',
        ]

        # Lead time features
        features += [
            'is_last_minute', 'is_short_notice', 'is_same_day',
            'is_far_advance', 'is_very_far_advance',
            'lead_time_bucket', 'lead_time_log',
        ]

        # Temporal
        features += [
            'is_weekend', 'is_monday', 'is_friday',
            'is_evening', 'is_late_evening', 'is_morning',
            'is_peak_hours',
            'month', 'is_monsoon', 'is_wedding_season', 'is_summer',
        ]

        # Domain-knowledge encodings
        features += [
            'payment_risk_encoded', 'service_commitment',
            'branch_tier',
        ]

        # Interactions
        features += [
            'cash_x_new', 'prepaid_x_loyal', 'cash_x_chronic',
            'evening_x_low_commit', 'evening_x_cash',
            'monday_morning_new', 'evening_cash_chronic',
            'weekend_premium_prepaid',
        ]

        # Composite
        features += [
            'noshow_intensity', 'reliability_score',
            'staff_noshow_rate', 'risk_score_v2',
        ]

        # Label-encoded categoricals
        features += [col + '_label' for col in self.CATEGORICAL_COLS]

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders (for prediction)."""
        df = df.copy()
        df = self.engineer_features(df, is_training=False)

        for col in self.CATEGORICAL_COLS:
            le = self.label_encoders[col]
            known = set(le.classes_)
            df[col + '_label'] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

        if self.TARGET_COL in df.columns:
            le_target = self.label_encoders[self.TARGET_COL]
            known_targets = set(le_target.classes_)
            df['target'] = df[self.TARGET_COL].apply(
                lambda x: le_target.transform([x])[0]
                if x in known_targets else -1
            )

        return df

    def get_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract X, y arrays from a processed DataFrame."""
        # ✅ Validate all features exist
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing features in DataFrame: {missing}"
            )

        X = df[self.feature_names].values.astype(np.float32)
        y = df['target'].values if 'target' in df.columns else None

        # ✅ Check for NaN/Inf
        if np.any(np.isnan(X)):
            nan_cols = [
                self.feature_names[i]
                for i in range(X.shape[1])
                if np.any(np.isnan(X[:, i]))
            ]
            print(f"  [WARN] NaN detected in features: {nan_cols}")
            X = np.nan_to_num(X, nan=0.0)

        if np.any(np.isinf(X)):
            print(f"  [WARN] Inf values detected, clipping...")
            X = np.clip(X, -1e6, 1e6)

        return X, y

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_artifacts(self):
        """Save fitted encoders and feature names."""
        joblib.dump(
            self.label_encoders,
            self.models_dir / "label_encoders.joblib"
        )
        joblib.dump(
            self._staff_noshow_rates,
            self.models_dir / "staff_noshow_rates.joblib"
        )
        with open(self.models_dir / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)
        with open(self.models_dir / "pipeline_config.json", "w") as f:
            json.dump({
                'payment_risk_map': self.PAYMENT_RISK_MAP,
                'service_commitment_map': self.SERVICE_COMMITMENT_MAP,
                'branch_tier_map': self.BRANCH_TIER_MAP,
                'global_noshow_rate': float(self._global_noshow_rate),
                'n_features': len(self.feature_names),
            }, f, indent=2)
        print(f"[SAVED] Pipeline artifacts ({len(self.feature_names)} features) "
              f"→ {self.models_dir}")

    def load_artifacts(self):
        """Load fitted encoders and feature names."""
        self.label_encoders = joblib.load(
            self.models_dir / "label_encoders.joblib"
        )
        with open(self.models_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        # Load staff rates if available
        staff_path = self.models_dir / "staff_noshow_rates.joblib"
        if staff_path.exists():
            self._staff_noshow_rates = joblib.load(staff_path)

        # Load config
        config_path = self.models_dir / "pipeline_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self._global_noshow_rate = config.get(
                'global_noshow_rate', 0.19
            )

        print(f"[LOADED] Pipeline artifacts ({len(self.feature_names)} features) "
              f"← {self.models_dir}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self, csv_path: str
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load CSV → engineer features → encode → return (df, X, y)."""
        print(f"\n{'='*60}")
        print(f"  DATA PIPELINE v2.0")
        print(f"{'='*60}")

        print(f"[1/4] Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=['booking_datetime'])
        print(f"       Rows: {len(df):,} | Columns: {len(df.columns)}")

        print(f"[2/4] Engineering features...")
        df = self.engineer_features(df, is_training=True)
        print(f"       Columns after engineering: {len(df.columns)}")

        print(f"[3/4] Fitting encoders...")
        df = self.fit_encoders(df)

        print(f"[4/4] Building feature matrix...")
        X, y = self.get_feature_matrix(df)

        # ✅ Summary
        noshow_rate = y.mean()
        print(f"\n{'─'*60}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'─'*60}")
        print(f"  Feature matrix: X = {X.shape}")
        print(f"  Target:         y = {y.shape} "
              f"(No-Show rate: {noshow_rate:.2%})")
        print(f"  Total features: {len(self.feature_names)}")
        print(f"")
        print(f"  Feature groups:")
        print(f"    Raw numerical:      {len(self.RAW_NUMERICAL_COLS)}")
        print(f"    Historical rates:   3")
        print(f"    Binary flags:       8")
        print(f"    Lead time:          7")
        print(f"    Temporal:           11")
        print(f"    Domain encodings:   3")
        print(f"    Interactions:       8")
        print(f"    Composite:          4")
        print(f"    Label-encoded cats: {len(self.CATEGORICAL_COLS)}")
        print(f"{'─'*60}")
        print(f"  Features:\n  {self.feature_names}")
        print(f"{'='*60}\n")

        return df, X, y


def main():
    """Standalone test of the pipeline."""
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "bookings.csv"

    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run generate_data.py first.")
        return

    pipeline = DataPipeline()
    df, X, y = pipeline.run_full_pipeline(str(csv_path))

    # Quick validation
    print("\n[VALIDATION]")
    print(f"  Any NaN in X: {np.any(np.isnan(X))}")
    print(f"  Any Inf in X: {np.any(np.isinf(X))}")
    print(f"  X dtype: {X.dtype}")
    print(f"  y unique: {np.unique(y, return_counts=True)}")

    # Feature value ranges
    print(f"\n  Feature ranges:")
    for i, name in enumerate(pipeline.feature_names):
        col = X[:, i]
        print(f"    {name:40s}  min={col.min():.3f}  "
              f"max={col.max():.3f}  mean={col.mean():.3f}")


if __name__ == "__main__":
    main()