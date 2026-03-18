"""
Data Drift Detection Module
Compares feature distributions between reference and current data using
KS test (numerical) and Chi-square test (categorical).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from scipy import stats


class DriftDetector:
    """Detect data drift between reference and current datasets."""

    NUMERICAL_FEATURES = [
        'booking_lead_time_hours', 'day_of_week', 'hour_of_day',
        'past_visit_count', 'past_cancellation_count', 'past_noshow_count',
        'service_duration_mins',
    ]
    CATEGORICAL_FEATURES = [
        'service_type', 'branch', 'payment_method',
    ]

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else Path(__file__).parent.parent / "models" / "drift_log.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _ks_test(self, ref: pd.Series, curr: pd.Series) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for numerical features."""
        stat, p_value = stats.ks_2samp(ref.dropna(), curr.dropna())
        return float(stat), float(p_value)

    def _chi2_test(self, ref: pd.Series, curr: pd.Series) -> Tuple[float, float]:
        """Chi-square test for categorical features."""
        # Build contingency table from both distributions
        all_categories = set(ref.unique()) | set(curr.unique())

        ref_counts = ref.value_counts()
        curr_counts = curr.value_counts()

        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

        # Normalize to same total for fair comparison
        ref_total = sum(ref_freq)
        curr_total = sum(curr_freq)

        if ref_total == 0 or curr_total == 0:
            return 0.0, 1.0

        # Expected frequencies based on combined proportions
        combined = [r + c for r, c in zip(ref_freq, curr_freq)]
        combined_total = sum(combined)

        expected_ref = [c * ref_total / combined_total for c in combined]
        expected_curr = [c * curr_total / combined_total for c in combined]

        # Chi-square for current vs expected
        try:
            stat, p_value = stats.chisquare(curr_freq, f_exp=expected_curr)
            return float(stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    def compare_distributions(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Compare feature distributions between reference and current data.

        Returns:
            {
                'drift_report': {feature: {test, statistic, p_value, drifted}},
                'drift_score': float (% of features drifted),
                'drifted_features': list[str],
                'total_features_tested': int,
                'timestamp': str,
            }
        """
        drift_report: Dict[str, Dict[str, Any]] = {}
        drifted_features: List[str] = []
        total_tested = 0

        # Test numerical features
        for feature in self.NUMERICAL_FEATURES:
            if feature in reference_df.columns and feature in current_df.columns:
                stat, p_value = self._ks_test(reference_df[feature], current_df[feature])
                is_drifted = p_value < significance_level
                drift_report[feature] = {
                    'test': 'KS',
                    'statistic': round(stat, 4),
                    'p_value': round(p_value, 6),
                    'drifted': is_drifted,
                }
                if is_drifted:
                    drifted_features.append(feature)
                total_tested += 1

        # Test categorical features
        for feature in self.CATEGORICAL_FEATURES:
            if feature in reference_df.columns and feature in current_df.columns:
                stat, p_value = self._chi2_test(reference_df[feature], current_df[feature])
                is_drifted = p_value < significance_level
                drift_report[feature] = {
                    'test': 'Chi-Square',
                    'statistic': round(stat, 4),
                    'p_value': round(p_value, 6),
                    'drifted': is_drifted,
                }
                if is_drifted:
                    drifted_features.append(feature)
                total_tested += 1

        drift_score = len(drifted_features) / max(total_tested, 1)

        result = {
            'drift_report': drift_report,
            'drift_score': round(drift_score, 4),
            'drifted_features': drifted_features,
            'total_features_tested': total_tested,
            'timestamp': datetime.now().isoformat(),
        }

        # Log the drift event
        self._log_drift_event(result)

        return result

    def should_retrain(self, drift_score: float) -> bool:
        """Determine if model retraining is needed based on drift score."""
        return drift_score > 0.3

    def _log_drift_event(self, result: Dict[str, Any]) -> None:
        """Append drift result to the drift log JSON file."""
        log = []
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    log = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                log = []

        log_entry = {
            'timestamp': result['timestamp'],
            'drift_score': result['drift_score'],
            'drifted_features': result['drifted_features'],
            'total_features_tested': result['total_features_tested'],
            'should_retrain': self.should_retrain(result['drift_score']),
        }
        log.append(log_entry)

        with open(self.log_path, 'w') as f:
            json.dump(log, f, indent=2)

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Read drift log history."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return []
