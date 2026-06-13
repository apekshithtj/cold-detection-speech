"""
Generate test-set predictions using the best saved model.

Usage:
    python predict.py [--model output/best_model.pkl]

Output:
    output/test_predictions.tsv  — file_name, predicted_label (C/NC)
"""

import os
import argparse
import joblib
import pandas as pd

from data_loader import get_test_files
from feature_extraction import extract_all
from config import OUTPUT_DIR

REVERSE_LABEL = {1: "C", 0: "NC"}


def main(model_path: str):
    print("=" * 55)
    print("  Generating test-set predictions")
    print("=" * 55)

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    test_files = get_test_files()
    print(f"Test files: {len(test_files)}")

    X_test = extract_all(test_files, desc="Test")

    y_pred_int   = model.predict(X_test)
    y_pred_label = [REVERSE_LABEL[p] for p in y_pred_int]

    out_df = pd.DataFrame({
        "file_name": test_files,
        "prediction": y_pred_label,
    })

    out_path = os.path.join(OUTPUT_DIR, "test_predictions.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"\nPredictions saved to: {out_path}")
    print(f"\nPrediction distribution:")
    print(out_df["prediction"].value_counts().to_string())

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))
    args = parser.parse_args()
    main(args.model)
