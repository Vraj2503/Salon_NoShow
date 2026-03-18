"""
Tests for the NoShowPredictor v2.0 inference engine.
Covers: field validation, probability ranges, risk ordering,
        edge cases, batch prediction, and setup validation.
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictor import NoShowPredictor


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def predictor():
    """Load the predictor once for all tests."""
    p = NoShowPredictor()
    models_dir = Path(__file__).parent.parent / "models"
    if not (models_dir / "best_model.joblib").exists():
        pytest.skip("Model artifacts not found. Run model_trainer.py first.")
    p.load_model()
    return p


@pytest.fixture
def low_risk_booking():
    """VIP + Prepaid + Bridal + Weekend morning."""
    return {
        'service_type': 'Bridal',
        'branch': 'Memnagar',
        'booking_lead_time_hours': 48,
        'day_of_week': 6,
        'hour_of_day': 10,
        'payment_method': 'Online Prepaid',
        'past_visit_count': 20,
        'past_cancellation_count': 0,
        'past_noshow_count': 0,
        'service_duration_mins': 240,
        'staff_id': 'S02',
        'month': 11,
    }


@pytest.fixture
def high_risk_booking():
    """New + Cash + Last-minute + Late evening + Low-commit service."""
    return {
        'service_type': 'Manicure',
        'branch': 'Chandkheda',
        'booking_lead_time_hours': 1,
        'day_of_week': 0,
        'hour_of_day': 20,
        'payment_method': 'Cash',
        'past_visit_count': 0,
        'past_cancellation_count': 0,
        'past_noshow_count': 0,
        'service_duration_mins': 40,
        'staff_id': 'S15',
        'month': 8,
    }


@pytest.fixture
def chronic_nosho_booking():
    """Chronic no-shower + Cash + Late evening + Monday."""
    return {
        'service_type': 'Pedicure',
        'branch': 'Chandkheda',
        'booking_lead_time_hours': 1,
        'day_of_week': 0,
        'hour_of_day': 20,
        'payment_method': 'Cash',
        'past_visit_count': 8,
        'past_cancellation_count': 4,
        'past_noshow_count': 5,
        'service_duration_mins': 50,
        'staff_id': 'S19',
        'month': 8,
    }


@pytest.fixture
def medium_risk_booking():
    """Occasional customer + UPI + weekday afternoon."""
    return {
        'service_type': 'Haircut',
        'branch': 'Science City',
        'booking_lead_time_hours': 48,
        'day_of_week': 2,
        'hour_of_day': 14,
        'payment_method': 'UPI',
        'past_visit_count': 3,
        'past_cancellation_count': 1,
        'past_noshow_count': 1,
        'service_duration_mins': 45,
        'staff_id': 'S07',
        'month': 6,
    }


# ------------------------------------------------------------------
# Test: Setup & Validation
# ------------------------------------------------------------------
class TestPredictorSetup:
    """Tests for model loading and validation."""

    def test_model_loads_successfully(self, predictor):
        """Predictor should load without errors."""
        assert predictor._loaded is True
        assert predictor.model is not None

    def test_feature_names_loaded(self, predictor):
        """Feature names should be a non-empty list."""
        assert isinstance(predictor.feature_names, list)
        assert len(predictor.feature_names) > 30  # v2.0 has 51 features

    def test_label_encoders_loaded(self, predictor):
        """All required label encoders should be present."""
        required = ['service_type', 'branch', 'payment_method',
                     'staff_id', 'outcome']
        for enc in required:
            assert enc in predictor.label_encoders, (
                f"Missing encoder: {enc}"
            )

    def test_noshow_class_index(self, predictor):
        """No-show class index should be 0 or 1."""
        assert predictor.noshow_class_index in [0, 1]

    def test_optimal_threshold_range(self, predictor):
        """Optimal threshold should be between 0 and 1."""
        assert 0.0 < predictor.optimal_threshold < 1.0

    def test_validate_setup_passes(self, predictor):
        """Full setup validation should pass."""
        result = predictor.validate_setup()
        assert result['status'] == "✅ PASS", (
            f"Validation failed: {result['issues']}"
        )
        assert len(result['issues']) == 0

    def test_model_info_contains_metrics(self, predictor):
        """Model info should contain all performance metrics."""
        info = predictor.get_model_info()
        assert 'model_name' in info
        assert 'roc_auc' in info
        assert 'optimal_threshold' in info
        assert 'n_features' in info
        assert info['roc_auc'] > 0.5  # better than random
        assert info['n_features'] == len(predictor.feature_names)


# ------------------------------------------------------------------
# Test: Output Format
# ------------------------------------------------------------------
class TestOutputFormat:
    """Tests for prediction output structure and types."""

    def test_predict_returns_all_required_fields(
        self, predictor, low_risk_booking
    ):
        """Prediction must contain all v2.0 output fields."""
        result = predictor.predict(low_risk_booking)
        required_fields = [
            'noshow_probability', 'risk_tier',
            'risk_factors', 'recommended_action', 'would_flag',
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_probability_is_float(self, predictor, low_risk_booking):
        """Probability should be a Python float."""
        result = predictor.predict(low_risk_booking)
        assert isinstance(result['noshow_probability'], float)

    def test_risk_tier_is_string(self, predictor, low_risk_booking):
        """Risk tier should be a string."""
        result = predictor.predict(low_risk_booking)
        assert isinstance(result['risk_tier'], str)

    def test_risk_factors_is_list_of_strings(
        self, predictor, high_risk_booking
    ):
        """Risk factors should be a non-empty list of strings."""
        result = predictor.predict(high_risk_booking)
        assert isinstance(result['risk_factors'], list)
        assert len(result['risk_factors']) >= 1
        assert all(isinstance(f, str) for f in result['risk_factors'])

    def test_would_flag_is_bool(self, predictor, low_risk_booking):
        """would_flag should be a boolean."""
        result = predictor.predict(low_risk_booking)
        assert isinstance(result['would_flag'], bool)

    def test_recommended_action_is_nonempty_string(
        self, predictor, low_risk_booking
    ):
        """Recommended action should be a non-empty string."""
        result = predictor.predict(low_risk_booking)
        assert isinstance(result['recommended_action'], str)
        assert len(result['recommended_action']) > 10


# ------------------------------------------------------------------
# Test: Probability Range
# ------------------------------------------------------------------
class TestProbabilityRange:
    """Tests that probabilities are always valid."""

    def test_probability_between_0_and_1(
        self, predictor, low_risk_booking
    ):
        result = predictor.predict(low_risk_booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_high_risk_probability_range(
        self, predictor, high_risk_booking
    ):
        result = predictor.predict(high_risk_booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_chronic_nosho_probability_range(
        self, predictor, chronic_nosho_booking
    ):
        result = predictor.predict(chronic_nosho_booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    @pytest.mark.parametrize("past_visits", [0, 1, 5, 10, 20, 50, 100])
    def test_probability_valid_across_visit_counts(
        self, predictor, past_visits
    ):
        """Probability should be valid for any visit count."""
        booking = {
            'service_type': 'Haircut',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'UPI',
            'past_visit_count': past_visits,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S05',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    @pytest.mark.parametrize("lead_time", [0, 1, 3, 24, 72, 168, 336, 720])
    def test_probability_valid_across_lead_times(
        self, predictor, lead_time
    ):
        """Probability should be valid for any lead time."""
        booking = {
            'service_type': 'Facial',
            'branch': 'Science City',
            'booking_lead_time_hours': lead_time,
            'day_of_week': 2,
            'hour_of_day': 15,
            'payment_method': 'Cash',
            'past_visit_count': 3,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 60,
            'staff_id': 'S01',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0


# ------------------------------------------------------------------
# Test: Risk Tier Validity
# ------------------------------------------------------------------
class TestRiskTier:
    """Tests for risk tier assignment."""

    VALID_TIERS = {'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'}

    def test_tier_is_valid(self, predictor, low_risk_booking):
        result = predictor.predict(low_risk_booking)
        assert result['risk_tier'] in self.VALID_TIERS

    @pytest.mark.parametrize("service", [
        'Haircut', 'Color', 'Keratin', 'Facial',
        'Manicure', 'Pedicure', 'Waxing', 'Bridal',
    ])
    def test_valid_tier_for_all_services(self, predictor, service):
        booking = {
            'service_type': service,
            'branch': 'Memnagar',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'UPI',
            'past_visit_count': 5,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 60,
            'staff_id': 'S01',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert result['risk_tier'] in self.VALID_TIERS

    @pytest.mark.parametrize("branch", [
        'Science City', 'Memnagar', 'Sindhu Bhavan Road',
        'Sabarmati', 'Chandkheda',
    ])
    def test_valid_tier_for_all_branches(self, predictor, branch):
        booking = {
            'service_type': 'Haircut',
            'branch': branch,
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'Cash',
            'past_visit_count': 3,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S01',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert result['risk_tier'] in self.VALID_TIERS


# ------------------------------------------------------------------
# Test: Risk Ordering (most important business logic test)
# ------------------------------------------------------------------
class TestRiskOrdering:
    """Tests that risk predictions follow logical ordering."""

    def test_high_risk_greater_than_low_risk(
        self, predictor, high_risk_booking, low_risk_booking
    ):
        """High-risk scenario should produce higher probability."""
        high = predictor.predict(high_risk_booking)
        low = predictor.predict(low_risk_booking)
        assert high['noshow_probability'] > low['noshow_probability'], (
            f"High risk ({high['noshow_probability']:.3f}) should be > "
            f"Low risk ({low['noshow_probability']:.3f})"
        )

    def test_chronic_nosho_higher_than_new_customer(
        self, predictor, chronic_nosho_booking, high_risk_booking
    ):
        """Chronic no-shower should be riskier than new customer."""
        chronic = predictor.predict(chronic_nosho_booking)
        new = predictor.predict(high_risk_booking)
        assert chronic['noshow_probability'] >= new['noshow_probability'], (
            f"Chronic ({chronic['noshow_probability']:.3f}) should be >= "
            f"New ({new['noshow_probability']:.3f})"
        )

    def test_prepaid_lower_than_cash(self, predictor):
        """Same customer, prepaid should be lower risk than cash."""
        base = {
            'service_type': 'Haircut',
            'branch': 'Science City',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'past_visit_count': 3,
            'past_cancellation_count': 1,
            'past_noshow_count': 1,
            'service_duration_mins': 45,
            'staff_id': 'S05',
            'month': 6,
        }
        cash_booking = {**base, 'payment_method': 'Cash'}
        prepaid_booking = {**base, 'payment_method': 'Online Prepaid'}

        cash_result = predictor.predict(cash_booking)
        prepaid_result = predictor.predict(prepaid_booking)

        assert cash_result['noshow_probability'] > prepaid_result['noshow_probability'], (
            f"Cash ({cash_result['noshow_probability']:.3f}) should be > "
            f"Prepaid ({prepaid_result['noshow_probability']:.3f})"
        )

    def test_noshow_history_increases_risk(self, predictor):
        """More past no-shows should increase predicted risk."""
        base = {
            'service_type': 'Haircut',
            'branch': 'Science City',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'Cash',
            'past_visit_count': 10,
            'past_cancellation_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S05',
            'month': 6,
        }
        low_history = {**base, 'past_noshow_count': 0}
        high_history = {**base, 'past_noshow_count': 7}

        low_result = predictor.predict(low_history)
        high_result = predictor.predict(high_history)

        assert high_result['noshow_probability'] > low_result['noshow_probability'], (
            f"7 no-shows ({high_result['noshow_probability']:.3f}) "
            f"should be > 0 no-shows ({low_result['noshow_probability']:.3f})"
        )

    def test_last_minute_increases_risk(self, predictor):
        """Last-minute booking should be higher risk than planned."""
        base = {
            'service_type': 'Haircut',
            'branch': 'Science City',
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'Cash',
            'past_visit_count': 3,
            'past_cancellation_count': 0,
            'past_noshow_count': 1,
            'service_duration_mins': 45,
            'staff_id': 'S05',
            'month': 6,
        }
        last_minute = {**base, 'booking_lead_time_hours': 1}
        planned = {**base, 'booking_lead_time_hours': 72}

        lm_result = predictor.predict(last_minute)
        p_result = predictor.predict(planned)

        assert lm_result['noshow_probability'] > p_result['noshow_probability'], (
            f"Last-minute ({lm_result['noshow_probability']:.3f}) "
            f"should be > Planned ({p_result['noshow_probability']:.3f})"
        )

    def test_loyal_customer_lower_than_new(self, predictor):
        """Loyal VIP should be lower risk than brand new customer."""
        loyal = {
            'service_type': 'Haircut',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 48,
            'day_of_week': 5,
            'hour_of_day': 14,
            'payment_method': 'Online Prepaid',
            'past_visit_count': 20,
            'past_cancellation_count': 0,
            'past_noshow_count': 1,
            'service_duration_mins': 45,
            'staff_id': 'S03',
            'month': 6,
        }
        new = {
            'service_type': 'Haircut',
            'branch': 'Chandkheda',
            'booking_lead_time_hours': 6,
            'day_of_week': 0,
            'hour_of_day': 19,
            'payment_method': 'Cash',
            'past_visit_count': 0,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S15',
            'month': 8,
        }
        loyal_result = predictor.predict(loyal)
        new_result = predictor.predict(new)

        assert loyal_result['noshow_probability'] < new_result['noshow_probability'], (
            f"Loyal ({loyal_result['noshow_probability']:.3f}) "
            f"should be < New ({new_result['noshow_probability']:.3f})"
        )


# ------------------------------------------------------------------
# Test: Edge Cases
# ------------------------------------------------------------------
class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_zero_past_visits(self, predictor):
        """Brand new customer should not crash."""
        booking = {
            'service_type': 'Manicure',
            'branch': 'Sabarmati',
            'booking_lead_time_hours': 5,
            'day_of_week': 1,
            'hour_of_day': 15,
            'payment_method': 'Card on Arrival',
            'past_visit_count': 0,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 35,
            'staff_id': 'S15',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert result['noshow_probability'] is not None
        assert len(result['risk_factors']) > 0

    def test_extreme_visit_count(self, predictor):
        """Very high visit count should not crash."""
        booking = {
            'service_type': 'Haircut',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 12,
            'payment_method': 'Online Prepaid',
            'past_visit_count': 500,
            'past_cancellation_count': 2,
            'past_noshow_count': 3,
            'service_duration_mins': 45,
            'staff_id': 'S01',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_zero_lead_time(self, predictor):
        """Zero lead time (walk-in) should be handled."""
        booking = {
            'service_type': 'Waxing',
            'branch': 'Chandkheda',
            'booking_lead_time_hours': 0,
            'day_of_week': 3,
            'hour_of_day': 15,
            'payment_method': 'Cash',
            'past_visit_count': 1,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 30,
            'staff_id': 'S07',
            'month': 6,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_max_lead_time(self, predictor):
        """Maximum lead time should be handled."""
        booking = {
            'service_type': 'Bridal',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 720,
            'day_of_week': 6,
            'hour_of_day': 10,
            'payment_method': 'Online Prepaid',
            'past_visit_count': 5,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 240,
            'staff_id': 'S02',
            'month': 12,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_all_noshows_in_history(self, predictor):
        """100% no-show history should not crash."""
        booking = {
            'service_type': 'Haircut',
            'branch': 'Chandkheda',
            'booking_lead_time_hours': 2,
            'day_of_week': 0,
            'hour_of_day': 20,
            'payment_method': 'Cash',
            'past_visit_count': 5,
            'past_cancellation_count': 5,
            'past_noshow_count': 5,
            'service_duration_mins': 40,
            'staff_id': 'S01',
            'month': 8,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0
        # Should be high risk
        assert result['risk_tier'] in ['HIGH', 'CRITICAL']

    def test_missing_month_uses_default(self, predictor):
        """Missing month field should use default (not crash)."""
        booking = {
            'service_type': 'Haircut',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'UPI',
            'past_visit_count': 5,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S01',
            # month intentionally missing
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0

    def test_unknown_staff_id(self, predictor):
        """Unknown staff ID should fall back gracefully."""
        booking = {
            'service_type': 'Haircut',
            'branch': 'Memnagar',
            'booking_lead_time_hours': 24,
            'day_of_week': 3,
            'hour_of_day': 14,
            'payment_method': 'UPI',
            'past_visit_count': 5,
            'past_cancellation_count': 0,
            'past_noshow_count': 0,
            'service_duration_mins': 45,
            'staff_id': 'S99',   # doesn't exist
            'month': 6,
        }
        result = predictor.predict(booking)
        assert 0.0 <= result['noshow_probability'] <= 1.0


# ------------------------------------------------------------------
# Test: Batch Prediction
# ------------------------------------------------------------------
class TestBatchPrediction:
    """Tests for batch prediction methods."""

    def test_predict_batch_returns_correct_count(self, predictor):
        """Batch should return one result per input."""
        bookings = [
            {
                'service_type': 'Haircut', 'branch': 'Science City',
                'booking_lead_time_hours': 24, 'day_of_week': 0,
                'hour_of_day': 10, 'payment_method': 'Cash',
                'past_visit_count': 3, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 45,
                'staff_id': 'S01', 'month': 6,
            },
            {
                'service_type': 'Color', 'branch': 'Memnagar',
                'booking_lead_time_hours': 72, 'day_of_week': 4,
                'hour_of_day': 14, 'payment_method': 'UPI',
                'past_visit_count': 8, 'past_cancellation_count': 1,
                'past_noshow_count': 1, 'service_duration_mins': 120,
                'staff_id': 'S10', 'month': 6,
            },
            {
                'service_type': 'Bridal', 'branch': 'Memnagar',
                'booking_lead_time_hours': 480, 'day_of_week': 6,
                'hour_of_day': 10, 'payment_method': 'Online Prepaid',
                'past_visit_count': 15, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 240,
                'staff_id': 'S02', 'month': 12,
            },
        ]
        results = predictor.predict_batch(bookings)
        assert len(results) == 3
        for r in results:
            assert 'noshow_probability' in r
            assert 'risk_tier' in r
            assert 'would_flag' in r

    def test_predict_batch_fast_returns_dataframe(self, predictor):
        """Fast batch should return a DataFrame."""
        bookings = [
            {
                'service_type': 'Haircut', 'branch': 'Science City',
                'booking_lead_time_hours': 24, 'day_of_week': 0,
                'hour_of_day': 10, 'payment_method': 'Cash',
                'past_visit_count': 3, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 45,
                'staff_id': 'S01', 'month': 6,
            },
            {
                'service_type': 'Facial', 'branch': 'Memnagar',
                'booking_lead_time_hours': 48, 'day_of_week': 3,
                'hour_of_day': 15, 'payment_method': 'UPI',
                'past_visit_count': 5, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 60,
                'staff_id': 'S05', 'month': 6,
            },
        ]
        result_df = predictor.predict_batch_fast(bookings)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert 'noshow_probability' in result_df.columns
        assert 'risk_tier' in result_df.columns
        assert 'would_flag' in result_df.columns

    def test_empty_batch(self, predictor):
        """Empty batch should return empty list."""
        results = predictor.predict_batch([])
        assert results == []


# ------------------------------------------------------------------
# Test: Recommended Action Consistency
# ------------------------------------------------------------------
class TestRecommendedAction:
    """Tests that recommended actions match assigned risk tiers."""

    def test_action_matches_tier(self, predictor):
        """Action text should correspond to the risk tier."""
        bookings = [
            {
                'service_type': 'Keratin',
                'branch': 'Sindhu Bhavan Road',
                'booking_lead_time_hours': 36,
                'day_of_week': 2, 'hour_of_day': 11,
                'payment_method': 'Online Prepaid',
                'past_visit_count': 12,
                'past_cancellation_count': 0,
                'past_noshow_count': 0,
                'service_duration_mins': 150,
                'staff_id': 'S04', 'month': 6,
            },
            {
                'service_type': 'Manicure',
                'branch': 'Chandkheda',
                'booking_lead_time_hours': 1,
                'day_of_week': 0, 'hour_of_day': 20,
                'payment_method': 'Cash',
                'past_visit_count': 5,
                'past_cancellation_count': 3,
                'past_noshow_count': 4,
                'service_duration_mins': 40,
                'staff_id': 'S01', 'month': 8,
            },
        ]
        for booking in bookings:
            result = predictor.predict(booking)
            tier = result['risk_tier']
            action = result['recommended_action']
            expected = predictor.RECOMMENDED_ACTIONS[tier]
            assert action == expected, (
                f"Tier {tier}: expected '{expected}', got '{action}'"
            )

    def test_would_flag_consistent_with_threshold(self, predictor):
        """would_flag should be True iff probability >= threshold."""
        bookings = [
            {
                'service_type': 'Haircut', 'branch': 'Memnagar',
                'booking_lead_time_hours': 48, 'day_of_week': 5,
                'hour_of_day': 14, 'payment_method': 'Online Prepaid',
                'past_visit_count': 20, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 45,
                'staff_id': 'S03', 'month': 6,
            },
            {
                'service_type': 'Manicure', 'branch': 'Chandkheda',
                'booking_lead_time_hours': 1, 'day_of_week': 0,
                'hour_of_day': 20, 'payment_method': 'Cash',
                'past_visit_count': 0, 'past_cancellation_count': 0,
                'past_noshow_count': 0, 'service_duration_mins': 40,
                'staff_id': 'S15', 'month': 8,
            },
        ]
        for booking in bookings:
            result = predictor.predict(booking)
            expected_flag = (
                result['noshow_probability'] >= predictor.optimal_threshold
            )
            assert result['would_flag'] == expected_flag, (
                f"Prob={result['noshow_probability']:.3f}, "
                f"threshold={predictor.optimal_threshold:.3f}, "
                f"expected would_flag={expected_flag}"
            )


# ------------------------------------------------------------------
# Test: Feature Count Consistency
# ------------------------------------------------------------------
class TestFeatureConsistency:
    """Tests that feature engineering produces correct dimensions."""

    def test_feature_vector_length(self, predictor, low_risk_booking):
        """Feature vector should match expected length."""
        features = predictor._prepare_features(low_risk_booking)
        assert features.shape == (1, len(predictor.feature_names)), (
            f"Expected (1, {len(predictor.feature_names)}), "
            f"got {features.shape}"
        )

    def test_feature_vector_no_nan(self, predictor, low_risk_booking):
        """Feature vector should not contain NaN."""
        features = predictor._prepare_features(low_risk_booking)
        assert not np.any(np.isnan(features)), "NaN found in features"

    def test_feature_vector_no_inf(self, predictor, high_risk_booking):
        """Feature vector should not contain infinity."""
        features = predictor._prepare_features(high_risk_booking)
        assert not np.any(np.isinf(features)), "Inf found in features"

    def test_feature_vector_dtype(self, predictor, low_risk_booking):
        """Feature vector should be float32."""
        features = predictor._prepare_features(low_risk_booking)
        assert features.dtype == np.float32


# ------------------------------------------------------------------
# Test: Determinism
# ------------------------------------------------------------------
class TestDeterminism:
    """Tests that predictions are deterministic."""

    def test_same_input_same_output(self, predictor, medium_risk_booking):
        """Same input should always produce same output."""
        r1 = predictor.predict(medium_risk_booking)
        r2 = predictor.predict(medium_risk_booking)
        assert r1['noshow_probability'] == r2['noshow_probability']
        assert r1['risk_tier'] == r2['risk_tier']

    def test_deterministic_across_10_calls(self, predictor, low_risk_booking):
        """10 identical calls should produce identical results."""
        probs = [
            predictor.predict(low_risk_booking)['noshow_probability']
            for _ in range(10)
        ]
        assert len(set(probs)) == 1, (
            f"Non-deterministic: {set(probs)}"
        )