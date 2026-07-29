import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils import get_logger

logger = get_logger()

SAMPLE_RATE = 256          # Hz — standard EEG sample rate
N_CHANNELS  = 14           # EEG channels
DURATION_S  = 10           # seconds per recording
LABEL_COL   = "label"      
LABEL_MAP   = {"healthy": 0, "depressed": 1, 0: 0, 1: 1}


# Synthetic Data Generator 
def generate_synthetic_eeg(
    n_subjects: int = 60,
    n_channels: int = N_CHANNELS,
    duration_s: int = DURATION_S,
    sample_rate: int = SAMPLE_RATE,
    save_path: str = "data/raw/synthetic_eeg.csv",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = duration_s * sample_rate
    t = np.linspace(0, duration_s, n_samples)

    X = np.zeros((n_subjects, n_channels, n_samples))
    y = np.array([0] * (n_subjects // 2) + [1] * (n_subjects - n_subjects // 2))

    for i in range(n_subjects):
        is_depressed = y[i] == 1
        for ch in range(n_channels):
            noise = rng.normal(0, 4.0, n_samples)

            # Frequency components per clinical literature
            delta = (1.05 if is_depressed else 0.95) * np.sin(2 * np.pi * 2 * t)
            theta = (0.95 if is_depressed else 0.85) * np.sin(2 * np.pi * 6 * t)
            alpha = (0.9 if is_depressed else 1.05) * np.sin(2 * np.pi * 10 * t)
            beta  = (0.95 if is_depressed else 1.0) * np.sin(2 * np.pi * 20 * t)

            X[i, ch] = delta + theta + alpha + beta + noise

    # Persist to CSV 
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    records = []
    for i in range(n_subjects):
        for t_idx in range(n_samples):
            row = {f"ch_{c}": X[i, c, t_idx] for c in range(n_channels)}
            row["subject_id"] = i
            row[LABEL_COL]    = int(y[i])
            records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    logger.info(f"Synthetic EEG saved -> {save_path}  ({n_subjects} subjects, {n_samples} samples each)")

    return X, y


# Real / CSV Loader 
def load_from_csv(
    filepath: str,
    label_col: str = LABEL_COL
) -> Tuple[np.ndarray, np.ndarray, list]:
    logger.info(f"Loading CSV -> {filepath}")
    df = pd.read_csv(filepath)

    # Handle missing values
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning(f"Found {n_missing} missing values — forward-filling")
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    # Identify channel columns
    exclude_cols = {label_col, "subject_id"}
    ch_cols = [c for c in df.columns if c not in exclude_cols]

    # Map labels to integers
    df[label_col] = df[label_col].map(lambda v: LABEL_MAP.get(v, int(v)))

    if "subject_id" not in df.columns:
        # Treat the whole dataframe as a single subject
        df["subject_id"] = 0

    subjects  = sorted(df["subject_id"].unique())
    X_list, y_list = [], []

    for sid in subjects:
        sub_df = df[df["subject_id"] == sid]
        label  = int(sub_df[label_col].mode()[0])
        signals = sub_df[ch_cols].values.T 
        X_list.append(signals)
        y_list.append(label)

    X_raw = np.array(X_list)   
    y     = np.array(y_list)   

    logger.info(f"Loaded {len(subjects)} subjects | "
                f"Depressed={y.sum()} | Healthy={(y==0).sum()} | "
                f"Channels={len(ch_cols)}")

    return X_raw, y, ch_cols

def load_data(
    data_path: str = "data/raw/synthetic_eeg.csv",
    use_synthetic: bool = True,
    n_subjects: int = 60
) -> Tuple[np.ndarray, np.ndarray, list]:
    ch_names = [f"ch_{i}" for i in range(N_CHANNELS)]

    if use_synthetic or not Path(data_path).exists():
        logger.info("Generating synthetic EEG dataset …")
        X, y = generate_synthetic_eeg(n_subjects=n_subjects, save_path=data_path)
        return X, y, ch_names

    return load_from_csv(data_path)
