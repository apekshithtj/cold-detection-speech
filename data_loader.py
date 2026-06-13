"""Load labels and file lists from the ComParE 2017 Cold dataset."""
import os
import pandas as pd
from config import LAB_FILE, WAV_DIR, LABEL_MAP


def load_labels() -> pd.DataFrame:
    """Return a DataFrame with columns: file_name, label (int), split."""
    df = pd.read_csv(LAB_FILE, sep="\t")
    df.columns = ["file_name", "label_str"]
    df["label"] = df["label_str"].map(LABEL_MAP)
    df["split"] = df["file_name"].str.split("_").str[0]   # train / devel
    return df


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)


def get_test_files() -> list[str]:
    """Return sorted list of test wav filenames (no labels available)."""
    files = sorted(f for f in os.listdir(WAV_DIR) if f.startswith("test_"))
    return files


def wav_path(filename: str) -> str:
    return os.path.join(WAV_DIR, filename)


if __name__ == "__main__":
    df = load_labels()
    print(df.groupby(["split", "label_str"]).size().unstack(fill_value=0))
    print(f"\nTest files: {len(get_test_files())}")
