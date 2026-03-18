
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class NoShowPredictor:
    """Real-time inference engine for no-show prediction v2.0."""

    RISK_TIERS = {
        'LOW':      (0.00, 0.25),
        'MEDIUM':   (0.25, 0.50),
        'HIGH':     (0.50, 0.70),
        'CRITICAL': (0.70, 1.01),
    }

    RECOMMENDED_ACTIONS = {
        'LOW':      "Standard confirmation SMS 24hrs before appointment.",
        'MEDIUM':   "Send reminder SMS + WhatsApp. Offer 10% discount for prepayment.",
        'HIGH':     "Call customer directly + request deposit or reschedule to prepaid.",
        'CRITICAL': "Require full prepayment OR double-book the slot. Flag for manager review.",
    }

    # ── Domain maps (must match DataPipeline v2.0) ──────────────
    PAYMENT_RISK_MAP = {
        'Online Prepaid':  0,
        'UPI':             1,
        'Card on Arrival': 2,
        'Cash':            3,
    }

    SERVICE_COMMITMENT_MAP = {
        'Bridal':   6, 'Keratin':  5, 'Color':    4,
        'Facial':   3, 'Haircut':  2, 'Pedicure': 2,
        'Waxing':   1, 'Manicure': 1,
    }

    BRANCH_TIER_MAP = {
        'Memnagar':           0,
        'Sindhu Bhavan Road':  1,
        'Science City':       2,
        'Sabarmati':          3,
        'Chandkheda':         4,
    }

    LOW_COMMIT_SERVICES = {'Manicure', 'Pedicure', 'Waxing'}
    PREMIUM_SERVICES = {'Bridal', 'Keratin', 'Color'}

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = (
            Path(models_dir) if models_dir
            else Path(__file__).parent.parent / "models"
        )
        self.model = None
        self.label_encoders: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.model_metadata: Dict[str, Any] = {}
        self.staff_noshow_rates: Dict[str, float] = {}
        self.global_noshow_rate: float = 0.19
        self.optimal_threshold: float = 0.5
        self.noshow_class_index: int = 1   # determined at load time
        self._loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load model, encoders, staff rates, and metadata."""
        print(f"[PREDICTOR] Loading model from {self.models_dir}...")

        # ── Core artifacts ───────────────────────────────────────
        self.model = joblib.load(self.models_dir / "best_model.joblib")
        self.label_encoders = joblib.load(
            self.models_dir / "label_encoders.joblib"
        )

        with open(self.models_dir / "feature_names.json", "r") as f:
            self.feature_names = json.load(f)

        # ── Staff no-show rates ──────────────────────────────────
        staff_path = self.models_dir / "staff_noshow_rates.joblib"
        if staff_path.exists():
            self.staff_noshow_rates = joblib.load(staff_path)

        # ── Pipeline config ──────────────────────────────────────
        config_path = self.models_dir / "pipeline_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.global_noshow_rate = config.get(
                'global_noshow_rate', 0.19
            )

        # ── Model metadata ───────────────────────────────────────
        meta_path = self.models_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.model_metadata = json.load(f)
            self.optimal_threshold = self.model_metadata.get(
                'optimal_threshold', 0.5
            )

        # ── Determine which class index = No-Show ───────────────
        if 'outcome' in self.label_encoders:
            le_target = self.label_encoders['outcome']
            classes = list(le_target.classes_)
            if 'No-Show' in classes:
                self.noshow_class_index = classes.index('No-Show')
            else:
                # Fallback: assume 1 = No-Show
                self.noshow_class_index = 1

        self._loaded = True
        model_name = self.model_metadata.get('best_model_name', 'Unknown')
        print(f"[PREDICTOR] ✅ Loaded: {model_name}")
        print(f"  Features:  {len(self.feature_names)}")
        print(f"  Threshold: {self.optimal_threshold:.4f}")
        print(f"  No-Show class index: {self.noshow_class_index}")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()

    # ------------------------------------------------------------------
    # Risk tier assignment
    # ------------------------------------------------------------------
    def _get_risk_tier(self, probability: float) -> str:
        """Map probability to risk tier."""
        for tier, (low, high) in self.RISK_TIERS.items():
            if low <= probability < high:
                return tier
        return 'CRITICAL'

    # ------------------------------------------------------------------
    # Human-readable risk factors
    # ------------------------------------------------------------------
    def _get_risk_factors(self, booking: Dict[str, Any]) -> List[str]:
        """Generate ranked, human-readable risk factors."""
        factors: List[tuple] = []   # (priority, message)

        past_visits = booking.get('past_visit_count', 0)
        past_noshows = booking.get('past_noshow_count', 0)
        past_cancels = booking.get('past_cancellation_count', 0)
        payment = booking.get('payment_method', '')
        lead_time = booking.get('booking_lead_time_hours', 24)
        hour = booking.get('hour_of_day', 12)
        day = booking.get('day_of_week', 0)
        service = booking.get('service_type', '')

        # ── No-show history (strongest signal) ───────────────────
        if past_visits >= 3:
            noshow_rate = past_noshows / past_visits
            if noshow_rate > 0.4:
                factors.append((10, f"Chronic no-show history ({noshow_rate:.0%} of past visits)"))
            elif noshow_rate > 0.2:
                factors.append((8, f"Elevated no-show history ({noshow_rate:.0%} of past visits)"))
        elif past_visits == 0:
            factors.append((6, "New customer — no booking history to assess reliability"))

        # ── Cancellation history ─────────────────────────────────
        if past_visits >= 2:
            cancel_rate = past_cancels / past_visits
            if cancel_rate > 0.3:
                factors.append((7, f"High cancellation rate ({cancel_rate:.0%})"))

        # ── Payment method ───────────────────────────────────────
        if payment == 'Cash':
            factors.append((6, "Cash payment — no financial commitment to attend"))
        elif payment == 'Card on Arrival':
            factors.append((4, "Card on arrival — no prepayment commitment"))

        # ── Lead time ────────────────────────────────────────────
        if lead_time < 2:
            factors.append((7, f"Extremely last-minute booking ({lead_time}h lead time)"))
        elif lead_time < 6:
            factors.append((5, f"Short-notice booking ({lead_time}h lead time)"))
        elif lead_time > 336:
            factors.append((5, f"Very far advance booking ({lead_time // 24} days ahead — may forget)"))
        elif lead_time > 168:
            factors.append((3, f"Far advance booking ({lead_time // 24} days ahead)"))

        # ── Time of day ──────────────────────────────────────────
        if hour >= 20:
            factors.append((5, f"Late evening slot ({hour}:00) — higher no-show tendency"))
        elif hour >= 18:
            factors.append((3, f"Evening appointment ({hour}:00)"))

        # ── Day of week ──────────────────────────────────────────
        if day == 0:
            factors.append((2, "Monday booking — higher no-show tendency"))

        # ── Service type ─────────────────────────────────────────
        if service in self.LOW_COMMIT_SERVICES:
            factors.append((3, f"Low-commitment service ({service}) — easier to skip"))

        # ── Dangerous combinations ───────────────────────────────
        if payment == 'Cash' and past_visits == 0:
            factors.append((9, "⚠️ Cash payment + new customer — highest risk combination"))

        if hour >= 19 and service in self.LOW_COMMIT_SERVICES:
            factors.append((6, f"⚠️ Late evening + {service} — high skip probability"))

        if (past_visits >= 3 and past_noshows / max(past_visits, 1) > 0.3
                and payment == 'Cash' and hour >= 18):
            factors.append((10, "⚠️ Triple risk: chronic no-shower + cash + evening"))

        # ── Positive signals (reduce alarm) ──────────────────────
        if payment == 'Online Prepaid' and service in self.PREMIUM_SERVICES:
            factors.append((-2, "✅ Prepaid + premium service — low risk combination"))
        elif payment == 'Online Prepaid':
            factors.append((-1, "✅ Online prepaid — customer financially committed"))

        if past_visits >= 15 and past_noshows / max(past_visits, 1) < 0.1:
            factors.append((-3, "✅ VIP customer — highly reliable history"))
        elif past_visits >= 8:
            factors.append((-1, "✅ Loyal customer — established relationship"))

        # Sort by priority (descending) and return top 4
        factors.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in factors[:4]] or [
            "No significant risk factors identified"
        ]

    # ------------------------------------------------------------------
    # Feature engineering (must match DataPipeline v2.0 EXACTLY)
    # ------------------------------------------------------------------
    def _prepare_features(self, booking: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single booking dict into the full feature vector.
        Must produce features in EXACT same order as DataPipeline v2.0.
        """
        # ── Extract raw values with defaults ─────────────────────
        lead_time = booking.get('booking_lead_time_hours', 24)
        day_of_week = booking.get('day_of_week', 0)
        hour_of_day = booking.get('hour_of_day', 12)
        past_visits = booking.get('past_visit_count', 0)
        past_cancels = booking.get('past_cancellation_count', 0)
        past_noshows = booking.get('past_noshow_count', 0)
        duration = booking.get('service_duration_mins', 60)
        service = booking.get('service_type', 'Haircut')
        branch = booking.get('branch', 'Science City')
        payment = booking.get('payment_method', 'Cash')
        staff_id = booking.get('staff_id', 'S01')
        month = booking.get('month', datetime.now().month)

        safe_visits = max(past_visits, 1)

        # ── Historical rates ─────────────────────────────────────
        noshow_rate_hist = past_noshows / safe_visits
        cancel_rate = past_cancels / safe_visits
        unreliability_rate = (past_noshows + past_cancels) / safe_visits

        # ── Binary flags — customer type ─────────────────────────
        is_new_customer = int(past_visits == 0)
        is_repeat_customer = int(past_visits > 0)
        is_loyal = int(past_visits >= 8)
        is_very_loyal = int(past_visits >= 15)
        is_vip = int(past_visits >= 15 and noshow_rate_hist < 0.1)
        is_chronic_nosho = int(past_visits >= 3 and noshow_rate_hist > 0.3)
        has_noshow_history = int(past_noshows > 0)
        has_cancel_history = int(past_cancels > 0)

        # ── Lead time features ───────────────────────────────────
        is_last_minute = int(lead_time < 3)
        is_short_notice = int(lead_time < 6)
        is_same_day = int(lead_time < 24)
        is_far_advance = int(lead_time > 168)
        is_very_far_advance = int(lead_time > 336)

        # Lead time bucket
        if lead_time <= 2:
            lead_time_bucket = 0
        elif lead_time <= 6:
            lead_time_bucket = 1
        elif lead_time <= 24:
            lead_time_bucket = 2
        elif lead_time <= 72:
            lead_time_bucket = 3
        elif lead_time <= 168:
            lead_time_bucket = 4
        elif lead_time <= 336:
            lead_time_bucket = 5
        else:
            lead_time_bucket = 6

        lead_time_log = np.log1p(lead_time)

        # ── Temporal features ────────────────────────────────────
        is_weekend = int(day_of_week in [5, 6])
        is_monday = int(day_of_week == 0)
        is_friday = int(day_of_week == 4)
        is_evening = int(hour_of_day >= 18)
        is_late_evening = int(hour_of_day >= 20)
        is_morning = int(hour_of_day <= 10)
        is_peak_hours = int(hour_of_day in [11, 12, 17, 18])
        is_monsoon = int(month in [7, 8, 9])
        is_wedding_season = int(month in [11, 12, 1, 2])
        is_summer = int(month in [4, 5, 6])

        # ── Domain-knowledge encodings ───────────────────────────
        payment_risk = self.PAYMENT_RISK_MAP.get(payment, 2)
        service_commit = self.SERVICE_COMMITMENT_MAP.get(service, 2)
        branch_tier = self.BRANCH_TIER_MAP.get(branch, 2)

        # ── Interaction features ─────────────────────────────────
        cash_x_new = int(payment == 'Cash' and past_visits == 0)
        prepaid_x_loyal = int(
            payment == 'Online Prepaid' and past_visits >= 8
        )
        cash_x_chronic = int(
            payment == 'Cash' and is_chronic_nosho == 1
        )
        evening_x_low_commit = int(
            hour_of_day >= 18 and service in self.LOW_COMMIT_SERVICES
        )
        evening_x_cash = int(hour_of_day >= 18 and payment == 'Cash')
        monday_morning_new = int(
            day_of_week == 0 and hour_of_day <= 11 and past_visits == 0
        )
        evening_cash_chronic = int(
            hour_of_day >= 18 and payment == 'Cash'
            and noshow_rate_hist > 0.3
        )
        weekend_premium_prepaid = int(
            is_weekend == 1
            and service in self.PREMIUM_SERVICES
            and payment == 'Online Prepaid'
        )

        # ── Composite features ───────────────────────────────────
        noshow_intensity = (
            noshow_rate_hist if past_visits >= 3
            else past_noshows * 0.3
        )
        reliability_score = np.log1p(past_visits) * (1 - noshow_rate_hist)

        staff_noshow_rate = self.staff_noshow_rates.get(
            staff_id, self.global_noshow_rate
        )

        risk_score_v2 = (
            noshow_rate_hist * 0.20 +
            cancel_rate * 0.10 +
            is_new_customer * 0.10 +
            payment_risk / 3 * 0.10 +
            is_evening * 0.05 +
            cash_x_new * 0.10
        )

        # ── Label-encode categoricals ────────────────────────────
        def safe_encode(col_name: str, value: str) -> int:
            le = self.label_encoders.get(col_name)
            if le is None:
                return -1
            if value in le.classes_:
                return le.transform([value])[0]
            return -1

        service_label = safe_encode('service_type', service)
        branch_label = safe_encode('branch', branch)
        payment_label = safe_encode('payment_method', payment)
        staff_label = safe_encode('staff_id', staff_id)

        # ── Build feature vector (MUST match pipeline order) ─────
        feature_vector = [
            # Raw numerical (7)
            lead_time, day_of_week, hour_of_day,
            past_visits, past_cancels, past_noshows,
            duration,

            # Historical rates (3)
            noshow_rate_hist, cancel_rate, unreliability_rate,

            # Customer flags (8)
            is_new_customer, is_repeat_customer,
            is_loyal, is_very_loyal, is_vip,
            is_chronic_nosho, has_noshow_history, has_cancel_history,

            # Lead time (7)
            is_last_minute, is_short_notice, is_same_day,
            is_far_advance, is_very_far_advance,
            lead_time_bucket, lead_time_log,

            # Temporal (11)
            is_weekend, is_monday, is_friday,
            is_evening, is_late_evening, is_morning, is_peak_hours,
            month, is_monsoon, is_wedding_season, is_summer,

            # Domain encodings (3)
            payment_risk, service_commit, branch_tier,

            # Interactions (8)
            cash_x_new, prepaid_x_loyal, cash_x_chronic,
            evening_x_low_commit, evening_x_cash,
            monday_morning_new, evening_cash_chronic,
            weekend_premium_prepaid,

            # Composite (4)
            noshow_intensity, reliability_score,
            staff_noshow_rate, risk_score_v2,

            # Label-encoded categoricals (4)
            service_label, branch_label, payment_label, staff_label,
        ]

        # ── Validate length ──────────────────────────────────────
        if len(feature_vector) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: built {len(feature_vector)}, "
                f"expected {len(self.feature_names)}.\n"
                f"Expected: {self.feature_names}\n"
                f"Got {len(feature_vector)} values."
            )

        return np.array([feature_vector], dtype=np.float32)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, booking: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict no-show probability for a single booking.

        Args:
            booking: dict with keys like service_type, branch,
                     payment_method, past_visit_count, etc.

        Returns:
            {
                'noshow_probability': float,
                'risk_tier': str,
                'risk_factors': list[str],
                'recommended_action': str,
                'would_flag': bool,
            }
        """
        self._ensure_loaded()

        features = self._prepare_features(booking)

        # ✅ Get probability for the No-Show class specifically
        proba_all = self.model.predict_proba(features)[0]
        noshow_prob = float(proba_all[self.noshow_class_index])

        risk_tier = self._get_risk_tier(noshow_prob)
        risk_factors = self._get_risk_factors(booking)
        action = self.RECOMMENDED_ACTIONS[risk_tier]

        # ✅ Whether this booking would be flagged at optimal threshold
        would_flag = noshow_prob >= self.optimal_threshold

        return {
            'noshow_probability': round(noshow_prob, 4),
            'risk_tier': risk_tier,
            'risk_factors': risk_factors,
            'recommended_action': action,
            'would_flag': would_flag,
        }

    def predict_batch(
        self, bookings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict no-show for a batch of bookings."""
        self._ensure_loaded()
        return [self.predict(b) for b in bookings]

    def predict_batch_fast(
        self, bookings: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Fast batch prediction returning a DataFrame.
        More efficient for large batches.
        """
        self._ensure_loaded()

        results = []
        for b in bookings:
            features = self._prepare_features(b)
            proba = float(
                self.model.predict_proba(features)[0][self.noshow_class_index]
            )
            results.append({
                **b,
                'noshow_probability': round(proba, 4),
                'risk_tier': self._get_risk_tier(proba),
                'would_flag': proba >= self.optimal_threshold,
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata and configuration."""
        self._ensure_loaded()
        return {
            'model_name': self.model_metadata.get(
                'best_model_name', 'Unknown'
            ),
            'optimal_threshold': self.optimal_threshold,
            'roc_auc': self.model_metadata.get('roc_auc', 0),
            'f1': self.model_metadata.get('f1', 0),
            'precision': self.model_metadata.get('precision', 0),
            'recall': self.model_metadata.get('recall', 0),
            'accuracy': self.model_metadata.get('accuracy', 0),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'noshow_class_index': self.noshow_class_index,
        }

    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that the predictor is correctly configured.
        Run this after loading to catch issues early.
        """
        self._ensure_loaded()

        issues = []

        # Check feature count
        expected = len(self.feature_names)
        test_booking = {
            'service_type': 'Haircut',
            'branch': 'Science City',
            'payment_method': 'Cash',
            'staff_id': 'S01',
            'booking_lead_time_hours': 24,
            'day_of_week': 0,
            'hour_of_day': 12,
            'past_visit_count': 0,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'month': 6,
        }

        try:
            features = self._prepare_features(test_booking)
            actual = features.shape[1]
            if actual != expected:
                issues.append(
                    f"Feature count: expected {expected}, got {actual}"
                )
        except Exception as e:
            issues.append(f"Feature preparation failed: {e}")

        try:
            result = self.predict(test_booking)
            prob = result['noshow_probability']
            if not 0 <= prob <= 1:
                issues.append(f"Invalid probability: {prob}")
        except Exception as e:
            issues.append(f"Prediction failed: {e}")

        # Check label encoders
        required_encoders = [
            'service_type', 'branch', 'payment_method',
            'staff_id', 'outcome',
        ]
        for enc in required_encoders:
            if enc not in self.label_encoders:
                issues.append(f"Missing label encoder: {enc}")

        status = "✅ PASS" if not issues else "❌ FAIL"

        return {
            'status': status,
            'issues': issues,
            'n_features': expected,
            'model_name': self.model_metadata.get(
                'best_model_name', 'Unknown'
            ),
            'test_prediction': (
                result if not issues else None
            ),
        }


def main():
    """Test the predictor with sample bookings."""
    predictor = NoShowPredictor()
    predictor.load_model()

    # ── Validate setup ───────────────────────────────────────
    print("\n[VALIDATION]")
    validation = predictor.validate_setup()
    print(f"  Status: {validation['status']}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    else:
        print(f"  Test prediction: {validation['test_prediction']}")

    # ── Test scenarios ───────────────────────────────────────
    test_bookings = [
        {
            'name': '🟢 Low Risk: VIP + Prepaid + Keratin',
            'service_type': 'Keratin',
            'branch': 'Memnagar',
            'payment_method': 'Online Prepaid',
            'staff_id': 'S03',
            'booking_lead_time_hours': 72,
            'day_of_week': 5,
            'hour_of_day': 14,
            'past_visit_count': 20,
            'past_cancellation_count': 0,
            'past_noshow_count': 1,
            'service_duration_mins': 150,
            'month': 11,
        },
        {
            'name': '🟡 Medium Risk: Occasional + UPI',
            'service_type': 'Haircut',
            'branch': 'Science City',
            'payment_method': 'UPI',
            'staff_id': 'S07',
            'booking_lead_time_hours': 48,
            'day_of_week': 2,
            'hour_of_day': 17,
            'past_visit_count': 3,
            'past_cancellation_count': 1,
            'past_noshow_count': 1,
            'service_duration_mins': 45,
            'month': 6,
        },
        {
            'name': '🔴 High Risk: New + Cash + Evening',
            'service_type': 'Manicure',
            'branch': 'Chandkheda',
            'payment_method': 'Cash',
            'staff_id': 'S15',
            'booking_lead_time_hours': 2,
            'day_of_week': 0,
            'hour_of_day': 20,
            'past_visit_count': 0,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 40,
            'month': 8,
        },
        {
            'name': '🔴 Critical: Chronic + Cash + Late + Monday',
            'service_type': 'Pedicure',
            'branch': 'Chandkheda',
            'payment_method': 'Cash',
            'staff_id': 'S19',
            'booking_lead_time_hours': 1,
            'day_of_week': 0,
            'hour_of_day': 20,
            'past_visit_count': 8,
            'past_cancellation_count': 4,
            'past_noshow_count': 5,
            'service_duration_mins': 50,
            'month': 8,
        },
    ]

    print(f"\n{'='*65}")
    print(f"  PREDICTION TEST SCENARIOS")
    print(f"{'='*65}")

    for booking in test_bookings:
        name = booking.pop('name')
        result = predictor.predict(booking)

        print(f"\n  {name}")
        print(f"    Probability:  {result['noshow_probability']:.1%}")
        print(f"    Risk Tier:    {result['risk_tier']}")
        print(f"    Would Flag:   {result['would_flag']}")
        print(f"    Action:       {result['recommended_action']}")
        print(f"    Risk Factors:")
        for factor in result['risk_factors']:
            print(f"      • {factor}")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()