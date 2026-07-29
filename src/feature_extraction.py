import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from typing import Tuple, Optional, List
from src.utils import get_logger, ensure_dirs

logger = get_logger()

# Band Definitions
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


# Per-Channel Features 
def band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    # Integrate PSD within [low, high] Hz to estimate band power
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() == 0:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_channel_features(signal: np.ndarray, fs: float = 256.0) -> dict:
    # Extract a feature dictionary from a single-channel EEG signal.

    # Parameters
    # signal : (n_samples,) 1-D array
    # fs     : sampling frequency in Hz
    
    feats = {}

    # Time domain 
    feats["mean"]       = float(np.mean(signal))
    feats["std"]        = float(np.std(signal))
    feats["variance"]   = float(np.var(signal))
    feats["skewness"]   = float(skew(signal))
    feats["kurtosis"]   = float(kurtosis(signal))
    feats["p2p"]        = float(np.ptp(signal))       # peak-to-peak amplitude

    # Frequency domain (Welch PSD)
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    total_power = float(np.trapezoid(psd, freqs)) or 1.0

    for band_name, (lo, hi) in BANDS.items():
        bp = band_power(psd, freqs, lo, hi)
        feats[f"{band_name}_power"]          = bp
        feats[f"{band_name}_relative_power"] = bp / total_power   # ratio

    return feats


# Subject Level Feature Matrix
def extract_features(
    X_proc: np.ndarray,
    y: np.ndarray,
    fs: float = 256.0,
    ch_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Extract features for every subject.

    # Parameters
    # X_proc   : (n_subjects, n_channels, n_samples)
    # y        : (n_subjects,)
    # fs       : sampling frequency

    # Returns
    # X_features : (n_subjects, n_features)
    # y          : (n_subjects,)
    # feat_names : list of feature names
    
    n_subjects, n_channels, _ = X_proc.shape
    if ch_names is None:
        ch_names = [f"ch_{i}" for i in range(n_channels)]

    all_rows = []

    for i in range(n_subjects):
        row = {}
        for ch_idx, ch_name in enumerate(ch_names):
            ch_feats = extract_channel_features(X_proc[i, ch_idx], fs=fs)
            for feat_name, val in ch_feats.items():
                row[f"{ch_name}_{feat_name}"] = val
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.fillna(0, inplace=True)

    feat_names = list(df.columns)
    X_features = df.values.astype(np.float32)

    logger.info(f"Feature matrix: {X_features.shape}  ({len(feat_names)} features per subject)")
    return X_features, y, feat_names


# Visualization 
def plot_feature_distributions(
    X_features: np.ndarray,
    y: np.ndarray,
    feat_names: List[str],
    top_n: int = 12,
    save_path: str = "results/feature_distributions.png"
) -> None:
    ensure_dirs("results")

    # Choose the most discriminative features (highest variance)
    variances = X_features.var(axis=0)
    top_idx   = np.argsort(variances)[::-1][:top_n]
    top_feats = [feat_names[i] for i in top_idx]

    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()
    fig.suptitle("Top Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)

    healthy_mask    = y == 0
    depressed_mask  = y == 1

    for ax, idx, fname in zip(axes, top_idx, top_feats):
        ax.hist(X_features[healthy_mask, idx],   bins=20, alpha=0.65, color="#2196F3", label="Healthy",   density=True)
        ax.hist(X_features[depressed_mask, idx],  bins=20, alpha=0.65, color="#F44336", label="Depressed", density=True)
        ax.set_title(fname.replace("_", " "), fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[top_n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Feature distribution plot saved -> {save_path}")
    plt.close()
