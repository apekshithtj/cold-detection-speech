"""
End-to-end pipeline runner.

Usage:
    python run_pipeline.py              # full run (uses cache if available)
    python run_pipeline.py --fresh      # re-extract all features from scratch
    python run_pipeline.py --skip-train # only generate test predictions
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Cold Detection Pipeline")
    parser.add_argument("--fresh",      action="store_true", help="Re-extract features")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only predict")
    args = parser.parse_args()

    if not args.skip_train:
        print("\n>>> Step 1: Explore dataset")
        import explore
        explore.main()

        print("\n>>> Step 2: Train models & evaluate")
        import train
        train.main(force_extract=args.fresh)

    print("\n>>> Step 3: Generate test predictions")
    import predict
    import os
    from config import OUTPUT_DIR
    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run without --skip-train first.")
        sys.exit(1)
    predict.main(model_path)

    print("\n>>> Pipeline complete.")


if __name__ == "__main__":
    main()
