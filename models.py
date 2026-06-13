"""
Model definitions and training utilities.
All models are wrapped in sklearn Pipelines (StandardScaler → classifier).
Evaluation metric: UAR (Unweighted Average Recall) = balanced accuracy.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, recall_score
)
from xgboost import XGBClassifier
from config import RANDOM_STATE, N_JOBS, LABEL_NAMES


def uar(y_true, y_pred) -> float:
    """Unweighted Average Recall (= balanced accuracy)."""
    return balanced_accuracy_score(y_true, y_pred)


def build_svm(C: float = 1.0, gamma: str = "scale") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=C, kernel="rbf", gamma=gamma,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            probability=True,
        ))
    ])


def build_random_forest(n_estimators: int = 300) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        ))
    ])


def build_xgboost(scale_pos_weight: float = 8.8) -> Pipeline:
    """scale_pos_weight ≈ n_NC / n_C to handle class imbalance."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ))
    ])


def evaluate(model, X_val: np.ndarray, y_val: np.ndarray,
             model_name: str = "") -> float:
    """Print full evaluation report and return UAR."""
    y_pred = model.predict(X_val)
    score  = uar(y_val, y_pred)

    print(f"\n{'='*55}")
    print(f" {model_name}  |  Devel UAR: {score:.4f}")
    print('='*55)
    print(classification_report(y_val, y_pred, target_names=LABEL_NAMES, digits=4))

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"          {'  '.join(LABEL_NAMES)}")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:4s}   {row}")

    per_class = recall_score(y_val, y_pred, average=None)
    print(f"\n  Recall NC={per_class[0]:.4f}  C={per_class[1]:.4f}  → UAR={score:.4f}")
    return score
