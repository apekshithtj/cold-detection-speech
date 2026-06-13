"""
Generate all 5 competition submission CSV files.

Naming: thirnahalli_jayadeva_cheng_submission_X.csv
Format: file_name, label (C/NC)

Submission strategy:
  1 - SVM C=1,  threshold=0.50  (baseline)
  2 - SVM C=1,  threshold=0.30  (boost Cold recall)
  3 - SVM C=10, threshold=0.50  (stronger SVM)
  4 - SVM C=10, threshold=0.30  (stronger + lower threshold)
  5 - SVM C=10, train+devel combined, threshold=0.30  (max data)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

from data_loader import load_labels, get_split, get_test_files
from feature_extraction import extract_all
from config import OUTPUT_DIR

PREFIX  = "thirnahalli_jayadeva_cheng"
REVERSE = {1: "C", 0: "NC"}


def load_features():
    """Load all splits from cache (fast — already extracted)."""
    print("Loading features from cache...")
    df = load_labels()
    train_df = get_split(df, "train")
    devel_df  = get_split(df, "devel")
    test_files = get_test_files()

    X_train = extract_all(train_df["file_name"].tolist(), desc="Train")
    y_train = train_df["label"].values

    X_devel = extract_all(devel_df["file_name"].tolist(), desc="Devel")
    y_devel = devel_df["label"].values

    X_test  = extract_all(test_files, desc="Test ")

    print(f"Train: {X_train.shape}  Devel: {X_devel.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_devel, y_devel, X_test, test_files


def build_svm(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", gamma="scale",
                    class_weight="balanced",
                    probability=True, random_state=42)),
    ])


def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """Use model probability for class 1 (Cold) with custom threshold."""
    proba = model.predict_proba(X)[:, 1]   # P(Cold)
    return (proba >= threshold).astype(int)


def uar(y_true, y_pred) -> float:
    return balanced_accuracy_score(y_true, y_pred)


def save_submission(test_files, y_pred_int, submission_num: int, note: str):
    labels = [REVERSE[p] for p in y_pred_int]
    df = pd.DataFrame({"file_name": test_files, "label": labels})
    fname = f"{PREFIX}_submission_{submission_num}.csv"
    out_path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(out_path, index=False)
    n_c  = sum(1 for l in labels if l == "C")
    n_nc = sum(1 for l in labels if l == "NC")
    print(f"  Saved: {fname}  [C={n_c}, NC={n_nc}]  ({note})")
    return out_path


def main():
    print("=" * 62)
    print("  Generating 5 Submission Files")
    print("=" * 62)

    X_train, y_train, X_devel, y_devel, X_test, test_files = load_features()
    X_train_full = np.vstack([X_train, X_devel])
    y_train_full  = np.concatenate([y_train, y_devel])

    submissions = []

    # ── Sub 1: SVM C=1, threshold=0.50 ────────────────────────────────
    print("\n[1/5] SVM C=1, threshold=0.50 (baseline)")
    m1 = build_svm(C=1.0)
    m1.fit(X_train, y_train)
    devel_uar = uar(y_devel, predict_with_threshold(m1, X_devel, 0.50))
    print(f"  Devel UAR: {devel_uar:.4f}")
    y_pred = predict_with_threshold(m1, X_test, 0.50)
    submissions.append(save_submission(test_files, y_pred, 1, "SVM C=1 thr=0.50"))

    # ── Sub 2: SVM C=1, threshold=0.30 ────────────────────────────────
    print("\n[2/5] SVM C=1, threshold=0.30")
    devel_uar = uar(y_devel, predict_with_threshold(m1, X_devel, 0.30))
    print(f"  Devel UAR: {devel_uar:.4f}")
    y_pred = predict_with_threshold(m1, X_test, 0.30)
    submissions.append(save_submission(test_files, y_pred, 2, "SVM C=1 thr=0.30"))

    # ── Sub 3: SVM C=10, threshold=0.50 ───────────────────────────────
    print("\n[3/5] SVM C=10, threshold=0.50")
    m3 = build_svm(C=10.0)
    m3.fit(X_train, y_train)
    devel_uar = uar(y_devel, predict_with_threshold(m3, X_devel, 0.50))
    print(f"  Devel UAR: {devel_uar:.4f}")
    y_pred = predict_with_threshold(m3, X_test, 0.50)
    submissions.append(save_submission(test_files, y_pred, 3, "SVM C=10 thr=0.50"))

    # ── Sub 4: SVM C=10, threshold=0.30 ───────────────────────────────
    print("\n[4/5] SVM C=10, threshold=0.30")
    # Also sweep thresholds on devel to find the optimal one
    best_thr, best_uar = 0.30, 0.0
    for thr in np.arange(0.10, 0.50, 0.05):
        u = uar(y_devel, predict_with_threshold(m3, X_devel, thr))
        if u > best_uar:
            best_uar, best_thr = u, thr
    print(f"  Best devel threshold: {best_thr:.2f}  UAR: {best_uar:.4f}")
    y_pred = predict_with_threshold(m3, X_test, best_thr)
    submissions.append(save_submission(test_files, y_pred, 4,
                                       f"SVM C=10 thr={best_thr:.2f} (opt)"))

    # ── Sub 5: SVM C=10 on train+devel, threshold=best_thr ────────────
    print("\n[5/5] SVM C=10, train+devel combined, threshold optimised")
    m5 = build_svm(C=10.0)
    m5.fit(X_train_full, y_train_full)
    print(f"  Trained on {len(y_train_full)} samples (train+devel)")
    print(f"  Using threshold: {best_thr:.2f} (from sub 4 devel sweep)")
    y_pred = predict_with_threshold(m5, X_test, best_thr)
    submissions.append(save_submission(test_files, y_pred, 5,
                                       f"SVM C=10 train+devel thr={best_thr:.2f}"))

    # Save best model (sub 5)
    joblib.dump(m5, os.path.join(OUTPUT_DIR, "best_model.pkl"))

    print("\n" + "=" * 62)
    print("  All 5 submissions saved to output/")
    print("=" * 62)
    for p in submissions:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
