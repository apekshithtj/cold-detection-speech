"""Central configuration for the Cold Detection project."""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\win10\Downloads\ComParE2017_Cold_4students\ComParE2017_Cold_4students"
WAV_DIR      = os.path.join(DATASET_ROOT, "wav")
LAB_FILE     = os.path.join(DATASET_ROOT, "lab", "ComParE2017_Cold.tsv")

PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR    = os.path.join(PROJECT_DIR, "cache")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "output")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Audio / Feature settings ───────────────────────────────────────────────
SAMPLE_RATE   = 16000      # resample all audio to 16 kHz
N_MFCC        = 40         # number of MFCC coefficients
HOP_LENGTH    = 512
N_FFT         = 1024
FMIN          = 75.0       # min F0 for pitch (Hz)
FMAX          = 600.0      # max F0 for pitch (Hz)

# ── Labels ─────────────────────────────────────────────────────────────────
LABEL_MAP   = {"C": 1, "NC": 0}   # Cold=1, Non-cold=0
LABEL_NAMES = ["NC", "C"]

# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS       = -1          # use all CPU cores
