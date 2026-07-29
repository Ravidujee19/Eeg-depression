import numpy as np
import matplotlib
matplotlib.use("Agg")        
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from typing import Optional, Tuple
from src.utils import get_logger, ensure_dirs

logger = get_logger()


# Filter Design
def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 1.0,
    highcut: float = 40.0,
    fs: float = 256.0,
    order: int = 4
) -> np.ndarray:
    # Apply a Butterworth bandpass filter to a 1-D or 2-D signal.

    # Parameters
    # signal  : (n_samples,) or (n_channels, n_samples)
    # lowcut  : lower cutoff frequency in Hz
    # highcut : upper cutoff frequency in Hz
    # fs      : sampling frequency in Hz
    # order   : filter order

    # Returns
    # filtered signal, same shape as input
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")

    if signal.ndim == 1:
        return sosfiltfilt(sos, signal)
    else:
        return np.array([sosfiltfilt(sos, ch) for ch in signal])


# Normalization
def zscore_normalize(signal: np.ndarray) -> np.ndarray:
   
    # Z-score normalize along the last axis (time axis).
    # Clips extreme outliers (±5 σ) before normalizing.

    # Parameters
    # signal : (n_channels, n_samples) or (n_subjects, n_channels, n_samples)
  
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True)
    std  = np.where(std == 0, 1.0, std)       # avoid divide-by-zero
    normed = (signal - mean) / std
    normed = np.clip(normed, -5.0, 5.0)       # clip extreme artifacts
    return normed


# Preprocessing Pipeline
def preprocess_signals(
    X_raw: np.ndarray,
    fs: float = 256.0,
    lowcut: float = 1.0,
    highcut: float = 40.0
) -> np.ndarray:
    # Parameters
    # X_raw : (n_subjects, n_channels, n_samples)
    # fs    : sampling frequency

    # Returns
    # X_proc : (n_subjects, n_channels, n_samples) — filtered + normalized
    
    n_subjects, n_channels, n_samples = X_raw.shape
    X_proc = np.zeros_like(X_raw)

    logger.info(f"Preprocessing {n_subjects} subjects ({n_channels} ch × {n_samples} samples) …")

    for i in range(n_subjects):
        filtered = bandpass_filter(X_raw[i], lowcut=lowcut, highcut=highcut, fs=fs)
        X_proc[i] = zscore_normalize(filtered)

    logger.info("Preprocessing complete done")
    return X_proc

def segment_into_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 256,
    overlap: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    # Segment EEG recordings into overlapping windows.

    # Parameters
    # X           : (n_subjects, n_channels, n_samples)
    # y           : (n_subjects,)
    # window_size : samples per window
    # overlap     : fraction of overlap between consecutive windows

    # Returns
    # X_win : (n_windows, n_channels, window_size)
    # y_win : (n_windows,)

    step = int(window_size * (1 - overlap))
    X_win_list, y_win_list = [], []

    for i in range(len(X)):
        n_samples = X[i].shape[-1]
        starts = range(0, n_samples - window_size + 1, step)
        for s in starts:
            X_win_list.append(X[i, :, s:s + window_size])
            y_win_list.append(y[i])

    return np.array(X_win_list), np.array(y_win_list)


# Visualization 
def plot_eeg_comparison(
    raw: np.ndarray,
    processed: np.ndarray,
    fs: float = 256.0,
    channel_idx: int = 0,
    subject_idx: int = 0,
    save_path: Optional[str] = "results/eeg_comparison.png"
) -> None:
    ensure_dirs("results")
    n_samples = raw.shape[-1]
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"EEG Signal — Subject {subject_idx}, Channel {channel_idx}",
        fontsize=14, fontweight="bold"
    )

    axes[0].plot(t, raw[subject_idx, channel_idx], color="#e74c3c", linewidth=0.7, alpha=0.85)
    axes[0].set_title("Raw EEG Signal")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, processed[subject_idx, channel_idx], color="#2ecc71", linewidth=0.7, alpha=0.85)
    axes[1].set_title("Filtered & Normalized Signal (1–40 Hz)")
    axes[1].set_ylabel("Amplitude (z-score)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved -> {save_path}")
    plt.close()
