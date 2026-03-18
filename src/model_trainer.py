"""
Model Trainer v2.0 — Train, compare, and select the best no-show prediction model.
Supports: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.
Uses Optuna for XGBoost hyperparameter tuning + threshold optimization.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import shap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline import DataPipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Train, compare, and persist no-show prediction models."""

    def __init__(self, models_dir: str = None):
        self.models_dir = (
            Path(models_dir) if models_dir
            else Path(__file__).parent.parent / "models"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.best_model = None
        self.best_model_name = ""
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        return {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
        }

    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray, y_proba: np.ndarray
    ) -> float:
        """Find threshold that maximizes F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_proba
        )
        f1_scores = (
            2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        )
        print(
            f"    Optimal threshold: {best_threshold:.4f} "
            f"(F1: {f1_scores[best_idx]:.4f})"
        )
        return float(best_threshold)

    # ------------------------------------------------------------------
    # Individual trainers
    # ------------------------------------------------------------------
    def _train_logistic_regression(self, X, y, class_weight):
        print("\n[1/5] Training Logistic Regression...")
        model = LogisticRegression(
            max_iter=1000, class_weight=class_weight,
            solver='lbfgs', C=0.5, random_state=42
        )
        model.fit(X, y)
        return model

    def _train_random_forest(self, X, y, class_weight):
        print("\n[2/5] Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=500, max_depth=14, min_samples_split=8,
            min_samples_leaf=4, class_weight=class_weight,
            random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        return model

    def _train_xgboost_optuna(self, X, y, scale_pos_weight):
        print("\n[3/5] Training XGBoost with Optuna (80 trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.005, 0.2, log=True
                ),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', 0.5, 0.95
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', 1, 15
                ),
                'gamma': trial.suggest_float('gamma', 0, 8),
                'reg_alpha': trial.suggest_float(
                    'reg_alpha', 1e-8, 10, log=True
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', 1e-8, 10, log=True
                ),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'eval_metric': 'auc',
                'verbosity': 0,
                'tree_method': 'hist',
            }
            model = xgb.XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr_idx, val_idx in skf.split(X, y):
                model.fit(
                    X[tr_idx], y[tr_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False,
                )
                proba = model.predict_proba(X[val_idx])[:, 1]
                scores.append(roc_auc_score(y[val_idx], proba))
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=80, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'auc',
            'verbosity': 0,
            'tree_method': 'hist',
        })

        print(f"  ✅ Best Optuna AUC: {study.best_value:.4f}")
        print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_params.items()}, indent=2)}")

        model = xgb.XGBClassifier(**best_params)
        model.fit(X, y)
        return model

    def _train_lightgbm(self, X, y, scale_pos_weight):
        print("\n[4/5] Training LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20,
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X, y)
        return model

    def _train_catboost(self, X, y, scale_pos_weight):
        print("\n[5/5] Training CatBoost...")
        model = CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.03,
            l2_leaf_reg=5, auto_class_weights='Balanced',
            random_seed=42, verbose=0
        )
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Cross-validated evaluation
    # ------------------------------------------------------------------
    def _evaluate_model_cv(
        self, model, X, y, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model using 5-fold stratified CV with threshold tuning."""
        print(f"  Evaluating {model_name} with 5-fold CV...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        y_pred_cv = cross_val_predict(model, X, y, cv=skf, method='predict')
        y_proba_cv = cross_val_predict(
            model, X, y, cv=skf, method='predict_proba'
        )[:, 1]

        # Find optimal threshold
        optimal_thresh = self.find_optimal_threshold(y, y_proba_cv)
        y_pred_tuned = (y_proba_cv >= optimal_thresh).astype(int)

        # Metrics at default threshold
        metrics_default = self.evaluate(y, y_pred_cv, y_proba_cv)

        # Metrics at optimal threshold
        metrics_tuned = self.evaluate(y, y_pred_tuned, y_proba_cv)

        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_proba_cv, n_bins=10
        )

        cm = confusion_matrix(y, y_pred_tuned)

        result = {
            'model_name': model_name,
            **{f"{k}": v for k, v in metrics_tuned.items()},
            **{f"{k}_default_thresh": v for k, v in metrics_default.items()},
            'optimal_threshold': optimal_thresh,
            'confusion_matrix': cm.tolist(),
            'calibration_fraction_positives': fraction_of_positives.tolist(),
            'calibration_mean_predicted': mean_predicted_value.tolist(),
        }

        print(
            f"    [Default 0.5]  ROC-AUC: {metrics_default['roc_auc']:.4f} | "
            f"F1: {metrics_default['f1']:.4f} | "
            f"P: {metrics_default['precision']:.4f} | "
            f"R: {metrics_default['recall']:.4f}"
        )
        print(
            f"    [Tuned {optimal_thresh:.3f}]  ROC-AUC: {metrics_tuned['roc_auc']:.4f} | "
            f"F1: {metrics_tuned['f1']:.4f} | "
            f"P: {metrics_tuned['precision']:.4f} | "
            f"R: {metrics_tuned['recall']:.4f}"
        )

        return result

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def _compute_shap(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        model_name: str,
        is_primary: bool = False,
    ):
        """Compute and save SHAP values for tree-based models."""
        print(f"  Computing SHAP values for {model_name}...")
        try:
            explainer = shap.TreeExplainer(model)
            sample_size = min(5000, X.shape[0])
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_idx]
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Save with model-specific filenames
            safe_name = model_name.lower().replace(" ", "_")
            np.save(
                self.models_dir / f"shap_values_{safe_name}.npy",
                shap_values,
            )

            # Primary model also saves to default filenames
            # (for dashboard compatibility)
            if is_primary:
                np.save(
                    self.models_dir / "shap_values.npy", shap_values
                )
                with open(
                    self.models_dir / "shap_feature_names.json", "w"
                ) as f:
                    json.dump(feature_names, f)

            # Feature importance from SHAP
            shap_importance = np.abs(shap_values).mean(axis=0)
            fi_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': shap_importance,
            }).sort_values('shap_importance', ascending=False)

            # Save model-specific importance
            fi_df.to_csv(
                self.models_dir / f"feature_importance_{safe_name}.csv",
                index=False,
            )

            # Primary model saves to default filename
            if is_primary:
                fi_df.to_csv(
                    self.models_dir / "feature_importance.csv",
                    index=False,
                )

            tag = "  [PRIMARY]" if is_primary else ""
            print(f"    ✅ SHAP saved{tag}. Top 10 features:")
            for _, row in fi_df.head(10).iterrows():
                bar = "█" * int(
                    row['shap_importance']
                    / fi_df['shap_importance'].max()
                    * 30
                )
                print(
                    f"      {row['feature']:35s} "
                    f"{row['shap_importance']:.4f} {bar}"
                )

        except Exception as e:
            print(f"  [WARN] SHAP computation failed for {model_name}: {e}")

    # ------------------------------------------------------------------
    # Ensemble builder
    # ------------------------------------------------------------------
    def _build_ensemble(
        self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray
    ) -> VotingClassifier:
        """Build a soft-voting ensemble of top 3 models."""
        print(
            "\n[ENSEMBLE] Building soft-voting ensemble of top 3 models..."
        )

        # Sort by actual CV results
        sorted_results = sorted(
            self.results, key=lambda r: r['roc_auc'], reverse=True
        )
        top3_names = [r['model_name'] for r in sorted_results[:3]]

        estimators = [(name, models[name]) for name in top3_names]
        print(f"  Members: {top3_names}")
        print(
            f"  ⚠️  Ensemble CV will refit all 3 models × 5 folds "
            f"(may take a few minutes)..."
        )

        ensemble = VotingClassifier(
            estimators=estimators, voting='soft', n_jobs=-1
        )
        ensemble.fit(X, y)
        return ensemble

    # ------------------------------------------------------------------
    # Main training pipeline
    # ------------------------------------------------------------------
    def train_all(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Tuple[Any, str]:
        """Train all models, evaluate, select best, and save artifacts."""
        self.feature_names = feature_names

        # Class imbalance handling
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        scale_pos_weight = n_neg / n_pos
        class_weight = {0: 1.0, 1: scale_pos_weight}

        print(f"\n{'='*70}")
        print(f"  CLASS BALANCE")
        print(f"  Show (0): {n_neg:,} | No-Show (1): {n_pos:,}")
        print(f"  No-Show rate: {n_pos / (n_pos + n_neg):.2%}")
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"{'='*70}")

        models: Dict[str, Any] = {}

        # ── Train all models ─────────────────────────────────────
        models['Logistic Regression'] = self._train_logistic_regression(
            X, y, class_weight
        )
        models['Random Forest'] = self._train_random_forest(
            X, y, class_weight
        )
        models['XGBoost'] = self._train_xgboost_optuna(
            X, y, scale_pos_weight
        )
        models['LightGBM'] = self._train_lightgbm(
            X, y, scale_pos_weight
        )
        models['CatBoost'] = self._train_catboost(
            X, y, scale_pos_weight
        )

        # ── Evaluate all models with CV ──────────────────────────
        print(f"\n{'='*70}")
        print("  MODEL COMPARISON (5-Fold Stratified CV + Threshold Tuning)")
        print(f"{'='*70}")

        for name, model in models.items():
            result = self._evaluate_model_cv(model, X, y, name)
            result['model'] = model
            self.results.append(result)

        # ── Build ensemble ───────────────────────────────────────
        ensemble = self._build_ensemble(models, X, y)
        ensemble_result = self._evaluate_model_cv(
            ensemble, X, y, "Ensemble (Top 3)"
        )
        ensemble_result['model'] = ensemble
        self.results.append(ensemble_result)

        # ── Select best by ROC-AUC ───────────────────────────────
        best_result = max(self.results, key=lambda r: r['roc_auc'])
        self.best_model = best_result['model']
        self.best_model_name = best_result['model_name']
        self.optimal_threshold = best_result.get('optimal_threshold', 0.5)

        print(f"\n{'='*70}")
        print(f"  🏆 BEST MODEL: {self.best_model_name}")
        print(f"     ROC-AUC:    {best_result['roc_auc']:.4f}")
        print(f"     F1:         {best_result['f1']:.4f}")
        print(f"     Precision:  {best_result['precision']:.4f}")
        print(f"     Recall:     {best_result['recall']:.4f}")
        print(f"     Accuracy:   {best_result['accuracy']:.4f}")
        print(f"     Threshold:  {self.optimal_threshold:.4f}")
        print(f"{'='*70}")

        # ── Save artifacts ───────────────────────────────────────

        # Best model
        joblib.dump(self.best_model, self.models_dir / "best_model.joblib")
        print(f"\n[SAVED] Best model → best_model.joblib")

        # All individual models (for ensemble reconstruction)
        for name, model in models.items():
            safe_name = name.lower().replace(" ", "_")
            joblib.dump(model, self.models_dir / f"{safe_name}.joblib")
        print(f"[SAVED] All {len(models)} individual models")

        # Model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'optimal_threshold': self.optimal_threshold,
            'roc_auc': best_result['roc_auc'],
            'f1': best_result['f1'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'accuracy': best_result['accuracy'],
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'confusion_matrix': best_result['confusion_matrix'],
            'class_balance': {
                'n_negative': n_neg,
                'n_positive': n_pos,
                'scale_pos_weight': round(scale_pos_weight, 4),
            },
        }
        with open(self.models_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[SAVED] Model metadata → model_metadata.json")

        # Comparison table
        exclude_keys = {
            'model', 'confusion_matrix',
            'calibration_fraction_positives',
            'calibration_mean_predicted',
        }
        comparison_df = pd.DataFrame([
            {k: v for k, v in r.items() if k not in exclude_keys}
            for r in self.results
        ])
        comparison_df.to_csv(
            self.models_dir / "model_comparison.csv", index=False
        )
        print(f"[SAVED] Model comparison → model_comparison.csv")

        # Pretty print comparison
        print(f"\n{'='*70}")
        print("  FINAL COMPARISON TABLE")
        print(f"{'='*70}")
        display_cols = [
            'model_name', 'roc_auc', 'f1', 'precision',
            'recall', 'accuracy', 'optimal_threshold',
        ]
        avail_cols = [c for c in display_cols if c in comparison_df.columns]
        print(
            comparison_df[avail_cols].to_string(
                index=False, float_format="%.4f"
            )
        )

        # Calibration data
        calibration_data = {
            r['model_name']: {
                'fraction_positives': r['calibration_fraction_positives'],
                'mean_predicted': r['calibration_mean_predicted'],
            }
            for r in self.results
        }
        with open(self.models_dir / "calibration_data.json", "w") as f:
            json.dump(calibration_data, f, indent=2)
        print(f"[SAVED] Calibration data → calibration_data.json")

        # ── SHAP computation ─────────────────────────────────────
        # Find which tree model is the best individual performer
        tree_model_names = ['XGBoost', 'LightGBM', 'CatBoost']
        tree_results = [
            r for r in self.results
            if r['model_name'] in tree_model_names
        ]
        tree_results.sort(key=lambda r: r['roc_auc'], reverse=True)

        if tree_results:
            # Best tree model gets primary SHAP (saved to default files)
            best_tree = tree_results[0]['model_name']
            print(f"\n[SHAP] Primary model: {best_tree}")
            self._compute_shap(
                models[best_tree], X, feature_names,
                best_tree, is_primary=True,
            )

            # Second-best tree model gets secondary SHAP
            if len(tree_results) > 1:
                second_tree = tree_results[1]['model_name']
                print(f"[SHAP] Secondary model: {second_tree}")
                self._compute_shap(
                    models[second_tree], X, feature_names,
                    second_tree, is_primary=False,
                )

        print(f"\n[DONE] All artifacts saved to {self.models_dir}/")
        return self.best_model, self.best_model_name


def main():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "bookings.csv"

    if not csv_path.exists():
        print(f"[ERROR] Data file not found at {csv_path}")
        print("  Run `python data/generate_data.py` first.")
        return

    # Run data pipeline
    print(f"\n{'='*70}")
    print(f"  SALON NO-SHOW AI — MODEL TRAINING PIPELINE v2.0")
    print(f"{'='*70}")

    pipeline = DataPipeline()
    df, X, y = pipeline.run_full_pipeline(str(csv_path))

    # Train all models
    trainer = ModelTrainer()
    best_model, best_name = trainer.train_all(X, y, pipeline.feature_names)

    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Best model:     {best_name}")
    print(f"  ROC-AUC:        {trainer.results[-1]['roc_auc']:.4f}")
    print(f"  Threshold:      {trainer.optimal_threshold:.4f}")
    print(f"  Features:       {len(pipeline.feature_names)}")
    print(f"  Artifacts:      {trainer.models_dir}")
    print(f"{'='*70}")
    print(f"\n  Next steps:")
    print(f"    1. streamlit run dashboard/app.py")
    print(f"    2. pytest tests/test_predictor.py -v")


if __name__ == "__main__":
    main()