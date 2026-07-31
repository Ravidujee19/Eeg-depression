import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from typing import Dict, List, Optional, Tuple
import torch
from src.utils import get_logger, ensure_dirs

logger = get_logger()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_snn(
    model,
    spikes: np.ndarray,
    y_true: np.ndarray,
    device: str = "cpu",
    model_name: str = "SNN"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    
    from src.snn_model import predict_snn
    model.eval()
    spk_t = torch.tensor(spikes, dtype=torch.float32).to(device)

    preds, logits_sum = predict_snn(model, spk_t, device=device)
    y_pred   = preds.cpu().numpy()

    # Softmax score for class=1
    scores   = torch.softmax(logits_sum, dim=1)[:, 1].cpu().numpy()

    metrics  = compute_metrics(y_true, y_pred)

    logger.info(f"\n{model_name} -> " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Healthy','Depressed'])}")
    return y_pred, scores, metrics


# Confusion Matrix 
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = "results/confusion_matrix.png"
) -> None:
    # Heatmap confusion matrix
    ensure_dirs("results")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Depressed"],
        yticklabels=["Healthy", "Depressed"],
        ax=ax, linewidths=0.5
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved -> {save_path}")
    plt.close()


# ROC Curve 
def plot_roc_curve(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "results/roc_curve.png"
) -> None:
    ensure_dirs("results")
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")

    for (name, (y_true, y_scores)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"ROC curve saved -> {save_path}")
    plt.close()


# Training Loss Curve
def plot_training_curve(
    history: Dict[str, List[float]],
    save_path: str = "results/training_loss.png"
) -> None:
    ensure_dirs("results")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("SNN Training History", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], "#3498db", lw=2, label="Train Loss")
    if any(v != 0 for v in history["val_loss"]):
        ax1.plot(epochs, history["val_loss"], "#e74c3c", lw=2, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "#2ecc71", lw=2, label="Train Acc")
    if any(v != 0 for v in history["val_acc"]):
        ax2.plot(epochs, history["val_acc"], "#f39c12", lw=2, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training curve saved -> {save_path}")
    plt.close()


# Model Comparison Bar Chart
def plot_model_comparison(
    all_metrics: Dict[str, Dict[str, float]],
    save_path: str = "results/model_comparison.png"
) -> None:
    ensure_dirs("results")
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    model_names = list(all_metrics.keys())
    n_models    = len(model_names)
    x           = np.arange(len(metrics_to_plot))
    width       = 0.8 / n_models
    colors      = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, metrics) in enumerate(all_metrics.items()):
        vals   = [metrics.get(m, 0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_to_plot], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Comparison chart saved -> {save_path}")
    plt.close()


# Adaptation Summary Table 
def print_adaptation_table(
    metrics_before: Dict[str, float],
    metrics_after:  Dict[str, float]
) -> None:
    border = "=" * 50
    print(f"\n{border}")
    print("  Patient Adaptation Results")
    print(border)
    print(f"  {'Metric':<14} | {'Before':>10} | {'After':>10} | {'Delta':>8}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        before = metrics_before.get(key, 0)
        after  = metrics_after.get(key, 0)
        delta  = after - before
        sign   = "+" if delta >= 0 else ""
        print(f"  {key.capitalize():<14} | {before:>10.4f} | {after:>10.4f} | {sign}{delta:>7.4f}")
    print(border)
