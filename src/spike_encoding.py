import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from src.utils import get_logger, ensure_dirs

logger = get_logger()

# Normalization
def minmax_scale(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom   = np.where((col_max - col_min) < eps, 1.0, col_max - col_min)
    return (X - col_min) / denom


# Rate Coding
def rate_encode(
    X_features: np.ndarray,
    T: int = 50,
    seed: int = 42
) -> np.ndarray:
  
    # Rate-code feature vectors into binary spike trains.

    # Parameters
    # X_features : (n_samples, n_features)  — min-max-scaled features
    # T          : number of time steps (SNN simulation duration)
    # seed       : random seed

    # Returns
    # spikes : (T, n_samples, n_features)
    #          spikes[t, i, j] = 1 if neuron j fired for sample i at time t

    rng = np.random.default_rng(seed)

    # Ensure features are in [0, 1]
    X_scaled = minmax_scale(X_features)

    # Bernoulli sampling: each feature value = firing probability
    # Shape: (T, n_samples, n_features)
    rand_vals = rng.random((T,) + X_scaled.shape)
    spikes    = (rand_vals < X_scaled[np.newaxis, :, :]).astype(np.float32)

    n_samples, n_features = X_features.shape
    logger.info(f"Spike encoding: {n_samples} samples × {n_features} features × {T} time steps")
    logger.info(f"  Mean firing rate: {spikes.mean():.3f}")

    return spikes


# Visualization 
def plot_spike_trains(
    spikes: np.ndarray,
    sample_idx: int = 0,
    n_neurons: int = 30,
    save_path: Optional[str] = "results/spike_trains.png"
) -> None:
    
    # Raster plot of spike trains for a single sample
    ensure_dirs("results")
    T   = spikes.shape[0]
    n   = min(n_neurons, spikes.shape[2])

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for neuron in range(n):
        spike_times = np.where(spikes[:, sample_idx, neuron] == 1)[0]
        ax.scatter(
            spike_times,
            np.full_like(spike_times, neuron),
            marker="|",
            s=20,
            color=plt.cm.plasma(neuron / n),
            linewidths=0.8
        )

    ax.set_xlabel("Time Step", color="white", fontsize=11)
    ax.set_ylabel("Neuron Index", color="white", fontsize=11)
    ax.set_title(
        f"Spike Train Raster — Sample {sample_idx} (first {n} neurons)",
        color="white", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlim(0, T)
    ax.set_ylim(-1, n)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Spike raster plot saved -> {save_path}")
    plt.close()
