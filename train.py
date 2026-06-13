"""
Main training script.

Usage:
    python train.py [--force-extract]

Steps:
  1. Load labels
  2. Extract / load cached features for train + devel
  3. Train SVM, Random Forest, XGBoost
  4. Evaluate each on devel set (UAR)
  5. Save the best model to output/best_model.pkl
"""

import os
import sys
import argparse
import joblib
import numpy as np

from data_loader import load_labels, get_split
from feature_extraction import extract_all
from models import build_svm, build_random_forest, build_xgboost, evaluate, uar
from config import OUTPUT_DIR, RANDOM_STATE


def main(force_extract: bool = False):
    print("=" * 60)
    print("  Cold Detection from Speech — Training Pipeline")
    print("=" * 60)

    # ── 1. Labels ─────────────────────────────────────────────────────────
    df = load_labels()
    train_df = get_split(df, "train")
    devel_df  = get_split(df, "devel")

    print(f"\nTrain: {len(train_df)} samples  "
          f"(C={train_df.label.sum()}, NC={(train_df.label==0).sum()})")
    print(f"Devel: {len(devel_df)} samples  "
          f"(C={devel_df.label.sum()}, NC={(devel_df.label==0).sum()})")

    # ── 2. Feature extraction ─────────────────────────────────────────────
    print("\n[1/3] Extracting train features …")
    X_train = extract_all(train_df["file_name"].tolist(),
                          desc="Train", force=force_extract)
    y_train = train_df["label"].values

    print("\n[2/3] Extracting devel features …")
    X_devel = extract_all(devel_df["file_name"].tolist(),
                          desc="Devel", force=force_extract)
    y_devel = devel_df["label"].values

    print(f"\nFeature matrix — Train: {X_train.shape}  Devel: {X_devel.shape}")

    # ── 3. Train & evaluate models ────────────────────────────────────────
    print("\n[3/3] Training models …")
    n_nc = (y_train == 0).sum()
    n_c  = (y_train == 1).sum()
    spw  = round(n_nc / n_c, 2)

    candidates = {
        "SVM (RBF, balanced)":   build_svm(),
        "Random Forest (300)":   build_random_forest(300),
        f"XGBoost (spw={spw})":  build_xgboost(scale_pos_weight=spw),
    }

    results = {}
    for name, model in candidates.items():
        print(f"\n  Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done.")
        score = evaluate(model, X_devel, y_devel, model_name=name)
        results[name] = (score, model)

    # ── 4. Save best model ────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k][0])
    best_score, best_model = results[best_name]

    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)

    print("\n" + "=" * 60)
    print(f"  Best model : {best_name}")
    print(f"  Devel UAR  : {best_score:.4f}")
    print(f"  Saved to   : {model_path}")
    print("=" * 60)

    # Summary table
    print("\n  Model Comparison (Devel UAR):")
    for name, (score, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = " ← BEST" if name == best_name else ""
        print(f"    {score:.4f}  {name}{marker}")

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract features even if cached")
    args = parser.parse_args()
    main(force_extract=args.force_extract)
