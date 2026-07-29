import os
import random
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# Reproducibility 
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed yet


# Logging 
def get_logger(name: str = "eeg_depression", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Directory Helpers
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# Result Saving
def save_results(results: dict, filepath: str) -> None:
    ensure_dirs(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"[OK] Results saved -> {filepath}")

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    ensure_dirs(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"[OK] DataFrame saved -> {filepath}")

class Timer:
    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start).total_seconds()
        print(f"[Timer] Elapsed: {self.elapsed:.2f}s")


# Device Selector
def get_device() -> str:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return device
