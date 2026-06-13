"""
Extract acoustic features from WAV files using scipy/numpy only (no librosa).

Feature set per file → fixed-length 257-d vector:
  MFCCs (40) mean+std          80
  Delta MFCCs mean+std         80
  Delta-Delta MFCCs mean+std   80
  Spectral centroid  mean+std   2
  Spectral bandwidth mean+std   2
  Spectral rolloff   mean+std   2
  ZCR                mean+std   2
  RMS energy         mean+std   2
  F0 pitch stats                5  (mean, std, min, max, voiced_fraction)
  HNR proxy          mean+std   2
  ─────────────────────────────────
  Total                       257
"""

import os
import warnings
import numpy as np
import scipy.signal
import scipy.fft
import scipy.linalg.blas as _blas
import soundfile as sf
from tqdm import tqdm

from config import WAV_DIR, CACHE_DIR, SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, FMIN, FMAX

warnings.filterwarnings("ignore")

N_FEAT = 257


# ── Audio loading ───────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)           # stereo → mono
    if sr != target_sr:
        # Simple polyphase resample
        ratio = target_sr / sr
        n_out = int(len(y) * ratio)
        y = scipy.signal.resample(y, n_out)
    return y.astype(np.float32)


# ── Mel filterbank & MFCCs ──────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                    fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_bins))
    for i in range(1, n_mels + 1):
        lo, mid, hi = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if mid > lo:
            filters[i - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            filters[i - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return filters


def _stft_frames(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Return power spectrogram, shape (n_fft//2+1, n_frames)."""
    window = scipy.signal.windows.hann(n_fft)
    n_bins = n_fft // 2 + 1
    n_frames = 1 + (len(y) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(y.strides[0] * hop_length, y.strides[0])
    ).copy()
    spec = scipy.fft.rfft(frames * window, n=n_fft, axis=1)
    return np.abs(spec[:, :n_bins]).T ** 2   # (n_bins, n_frames)


def compute_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Return (n_mfcc, n_frames) MFCC matrix."""
    power = _stft_frames(y, n_fft, hop_length)            # (n_bins, n_frames)
    mel_fb = _mel_filterbank(sr, n_fft, n_mels=n_mfcc).astype(np.float64)   # (n_mfcc, n_bins)
    mel_spec = _blas.dgemm(1.0, mel_fb, power.astype(np.float64))            # (n_mfcc, n_frames)
    log_mel  = np.log(mel_spec + 1e-10)
    # DCT-II along mel axis
    mfcc = scipy.fft.dct(log_mel, axis=0, norm="ortho")   # (n_mfcc, n_frames)
    return mfcc


def compute_delta(feat: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute delta features via polynomial regression."""
    n_feat, n_frames = feat.shape
    half = width // 2
    denom = 2.0 * sum(i ** 2 for i in range(1, half + 1))
    # Pad edges
    padded = np.pad(feat, ((0, 0), (half, half)), mode="edge")
    delta  = np.zeros_like(feat)
    for t in range(n_frames):
        for i in range(1, half + 1):
            delta[:, t] += i * (padded[:, t + half + i] - padded[:, t + half - i])
    delta /= denom
    return delta


# ── Spectral features ───────────────────────────────────────────────────────

def spectral_centroid(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """power: (n_bins, n_frames), returns (n_frames,)."""
    return np.sum(freqs[:, None] * power, axis=0) / (np.sum(power, axis=0) + 1e-10)


def spectral_bandwidth(power: np.ndarray, freqs: np.ndarray,
                       centroid: np.ndarray) -> np.ndarray:
    diff = freqs[:, None] - centroid[None, :]
    return np.sqrt(np.sum(diff ** 2 * power, axis=0) / (np.sum(power, axis=0) + 1e-10))


def spectral_rolloff(power: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
    """Frequency below which roll_percent of energy is contained."""
    cum = np.cumsum(power, axis=0)
    threshold = roll_percent * cum[-1, :]
    idx = np.argmax(cum >= threshold, axis=0)
    return idx.astype(np.float32)


# ── Pitch estimation (autocorrelation-based) ────────────────────────────────

def _autocorr_pitch_frame(frame: np.ndarray, sr: int,
                           fmin: float, fmax: float) -> tuple[float, bool]:
    """Return (f0_hz, is_voiced) for a single frame via autocorrelation."""
    n = len(frame)
    # Normalize
    frame = frame - frame.mean()
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-4:
        return 0.0, False

    # Autocorrelation via FFT
    fft_size = 2 * n
    F = np.fft.rfft(frame, n=fft_size)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    ac /= ac[0] + 1e-10

    # Search for peak in valid lag range
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    lag_max = min(lag_max, n - 1)
    if lag_min >= lag_max:
        return 0.0, False

    region = ac[lag_min:lag_max + 1]
    peak_idx = np.argmax(region)
    peak_val = region[peak_idx]

    if peak_val < 0.3:    # below voiced threshold
        return 0.0, False
    lag = lag_min + peak_idx
    f0  = sr / lag
    return float(f0), True


def compute_pitch(y: np.ndarray, sr: int = SAMPLE_RATE,
                  hop_length: int = HOP_LENGTH,
                  fmin: float = FMIN, fmax: float = FMAX) -> tuple:
    """Return (f0_array, voiced_flags) per frame."""
    frame_len = 1024
    n_frames  = 1 + (len(y) - frame_len) // hop_length
    f0     = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_len]
        if len(frame) < frame_len:
            break
        f0[i], voiced[i] = _autocorr_pitch_frame(frame, sr, fmin, fmax)
    return f0, voiced


# ── HNR (Harmonics-to-Noise Ratio) ─────────────────────────────────────────

def compute_hnr(y: np.ndarray, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Approximate per-frame HNR via autocorrelation (Boersma method)."""
    frame_len = 1024
    n_frames  = 1 + (len(y) - frame_len) // hop_length
    hnr_vals  = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame - frame.mean()
        fft_size = 2 * frame_len
        F  = np.fft.rfft(frame, n=fft_size)
        ac = np.fft.irfft(F * np.conj(F))[:frame_len]
        if ac[0] < 1e-10:
            continue
        ac /= ac[0]
        # HNR from peak of ac
        r = max(0.0, min(ac[1:].max(), 0.9999))
        hnr_vals[i] = 10 * np.log10(r / (1 - r + 1e-10))
    return hnr_vals


# ── Stats helper ────────────────────────────────────────────────────────────

def _stats(x: np.ndarray) -> np.ndarray:
    return np.array([x.mean(), x.std()])


# ── Full feature extraction ──────────────────────────────────────────────────

def extract_features(wav_path: str) -> np.ndarray:
    """Extract 257-d feature vector from a WAV file."""
    y = load_audio(wav_path)

    # ── MFCCs + deltas ────────────────────────────────────────────────────
    mfcc   = compute_mfcc(y)                # (40, T)
    d_mfcc = compute_delta(mfcc)            # (40, T)
    dd_mfcc = compute_delta(d_mfcc)         # (40, T)

    mfcc_feat  = np.concatenate([mfcc.mean(axis=1),  mfcc.std(axis=1)])   # 80
    d_feat     = np.concatenate([d_mfcc.mean(axis=1), d_mfcc.std(axis=1)]) # 80
    dd_feat    = np.concatenate([dd_mfcc.mean(axis=1), dd_mfcc.std(axis=1)]) # 80

    # ── Power spectrogram for spectral features ───────────────────────────
    power = _stft_frames(y, N_FFT, HOP_LENGTH)               # (n_bins, T)
    freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)

    cent = spectral_centroid(power, freqs)
    bw   = spectral_bandwidth(power, freqs, cent)
    roll = spectral_rolloff(power)
    spec_feat = np.concatenate([_stats(cent), _stats(bw), _stats(roll)])   # 6

    # ── ZCR & RMS ─────────────────────────────────────────────────────────
    n_frames = 1 + (len(y) - N_FFT) // HOP_LENGTH
    zcr  = np.array([np.mean(np.abs(np.diff(np.sign(
               y[i*HOP_LENGTH:i*HOP_LENGTH+N_FFT])))) / 2
               for i in range(n_frames)])
    rms  = np.array([np.sqrt(np.mean(y[i*HOP_LENGTH:i*HOP_LENGTH+N_FFT]**2))
               for i in range(n_frames)])
    zcr_feat = _stats(zcr)   # 2
    rms_feat = _stats(rms)   # 2

    # ── Pitch ──────────────────────────────────────────────────────────────
    f0, voiced = compute_pitch(y)
    voiced_f0  = f0[voiced] if voiced.any() else np.array([0.0])
    pitch_feat = np.array([
        voiced_f0.mean(),
        voiced_f0.std(),
        voiced_f0.min(),
        voiced_f0.max(),
        voiced.mean(),        # voiced fraction
    ])                        # 5

    # ── HNR ───────────────────────────────────────────────────────────────
    hnr      = compute_hnr(y)
    hnr_feat = _stats(hnr)   # 2

    feat = np.concatenate([
        mfcc_feat, d_feat, dd_feat,   # 240
        spec_feat,                     # 6
        zcr_feat, rms_feat,            # 4
        pitch_feat,                    # 5
        hnr_feat,                      # 2
    ])                                 # = 257

    # Replace any NaN/inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ── Batch extraction with per-file disk cache ───────────────────────────────

def extract_all(file_names: list, desc: str = "Extracting",
                force: bool = False) -> np.ndarray:
    per_file_cache = os.path.join(CACHE_DIR, "per_file")
    os.makedirs(per_file_cache, exist_ok=True)

    features = []
    for fname in tqdm(file_names, desc=desc, unit="file"):
        cache_path = os.path.join(per_file_cache, fname.replace(".wav", ".npy"))
        if not force and os.path.exists(cache_path):
            feat = np.load(cache_path)
        else:
            wav = os.path.join(WAV_DIR, fname)
            try:
                feat = extract_features(wav)
            except Exception as e:
                print(f"\nWARNING: failed on {fname}: {e}")
                feat = np.zeros(N_FEAT)
            np.save(cache_path, feat)
        features.append(feat)

    return np.vstack(features)


if __name__ == "__main__":
    import time, os
    wav = os.path.join(WAV_DIR, "train_0001.wav")
    print(f"Testing on: {wav}")
    t0 = time.time()
    feat = extract_features(wav)
    print(f"Shape : {feat.shape}")
    print(f"NaN   : {np.isnan(feat).any()}")
    print(f"Time  : {time.time()-t0:.2f}s")
    print(f"First 10: {feat[:10].round(3)}")
