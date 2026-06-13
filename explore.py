"""
EDA script — prints dataset statistics and saves a summary report.
Run before training to understand the data.

Usage:
    python explore.py
"""

import os
import json
import numpy as np
import pandas as pd
import soundfile as sf

from data_loader import load_labels, get_split, get_test_files, wav_path
from config import OUTPUT_DIR


def audio_stats(file_names: list, n_sample: int = 200) -> dict:
    """Sample n_sample files and return duration statistics."""
    sample = file_names[:n_sample]
    durations = []
    for fn in sample:
        try:
            info = sf.info(wav_path(fn))
            durations.append(info.duration)
        except Exception:
            pass
    d = np.array(durations)
    return {"n": len(d), "mean_s": d.mean(), "std_s": d.std(),
            "min_s": d.min(), "max_s": d.max()}


def main():
    df = load_labels()
    train_df = get_split(df, "train")
    devel_df  = get_split(df, "devel")
    test_files = get_test_files()

    print("=" * 60)
    print("  ComParE 2017 Cold Task — Dataset Overview")
    print("=" * 60)

    for split, sdf in [("train", train_df), ("devel", devel_df)]:
        n_c  = (sdf.label == 1).sum()
        n_nc = (sdf.label == 0).sum()
        print(f"\n{split.upper()}: {len(sdf)} total")
        print(f"   Cold (C):     {n_c:5d}  ({100*n_c/len(sdf):.1f}%)")
        print(f"   Non-Cold (NC):{n_nc:5d}  ({100*n_nc/len(sdf):.1f}%)")
        print(f"   Imbalance ratio: 1:{n_nc/n_c:.1f}")

    print(f"\nTEST: {len(test_files)} files (labels withheld)")

    print("\nSampling audio durations (first 200 train files) …")
    stats = audio_stats(train_df["file_name"].tolist())
    print(f"  Duration — mean: {stats['mean_s']:.2f}s  "
          f"std: {stats['std_s']:.2f}s  "
          f"min: {stats['min_s']:.2f}s  "
          f"max: {stats['max_s']:.2f}s")

    # Save summary to output
    summary = {
        "train_total": len(train_df),
        "train_C": int((train_df.label == 1).sum()),
        "train_NC": int((train_df.label == 0).sum()),
        "devel_total": len(devel_df),
        "devel_C": int((devel_df.label == 1).sum()),
        "devel_NC": int((devel_df.label == 0).sum()),
        "test_total": len(test_files),
        "audio_mean_duration_s": round(stats["mean_s"], 2),
    }
    out_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
