import os
import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from src.utils         import set_seed, get_logger, ensure_dirs, save_results, Timer
from src.data_loader   import load_data
from src.preprocessing import preprocess_signals, plot_eeg_comparison
from src.feature_extraction import extract_features, plot_feature_distributions
from src.spike_encoding import rate_encode, plot_spike_trains
from src.baseline_model import (train_baseline, evaluate_model,
                                 get_feature_importance, save_model)
from src.snn_model      import EEGDepression_SNN, save_snn
from src.train          import train_snn, patient_adaptation
from src.evaluate       import (evaluate_snn, plot_confusion_matrix,
                                 plot_roc_curve, plot_training_curve,
                                 plot_model_comparison, print_adaptation_table,
                                 compute_metrics)

logger = get_logger()

def parse_args():
    p = argparse.ArgumentParser(description="EEG Depression Detection – SNN Pipeline")
    p.add_argument("--n-subjects",  type=int,   default=60,     help="Subjects to generate (synthetic)")
    p.add_argument("--epochs",      type=int,   default=25,     help="SNN training epochs")
    p.add_argument("--lr",          type=float, default=1e-3,   help="Learning rate")
    p.add_argument("--batch-size",  type=int,   default=16,     help="Batch size")
    p.add_argument("--T",           type=int,   default=50,     help="SNN time steps")
    p.add_argument("--seed",        type=int,   default=42,     help="Random seed")
    p.add_argument("--no-synthetic",action="store_true",        help="Use real CSV data")
    p.add_argument("--data-path",   type=str,   default="data/raw/synthetic_eeg.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    ensure_dirs("results", "models", "data/raw", "data/processed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    all_results = {}

    # Load Data
    with Timer():
        X_raw, y, ch_names = load_data(
            data_path=args.data_path,
            use_synthetic=not args.no_synthetic,
            n_subjects=args.n_subjects
        )
    logger.info(f"Dataset: {X_raw.shape}  |  Classes: Healthy={( y==0).sum()}, Depressed={(y==1).sum()}")

    # Preprocess
    with Timer():
        X_proc = preprocess_signals(X_raw, fs=256.0)

    # Save processed data
    np.save("data/processed/X_processed.npy", X_proc)
    np.save("data/processed/y_labels.npy",    y)

    # Visualize EEG
    plot_eeg_comparison(X_raw, X_proc, save_path="results/eeg_comparison.png")

    # Feature Extraction 
    with Timer():
        X_features, y_feat, feat_names = extract_features(X_proc, y, fs=256.0, ch_names=ch_names)

    np.save("data/processed/X_features.npy", X_features)
    logger.info(f"Feature matrix: {X_features.shape}")

    # Feature Visualization 
    plot_feature_distributions(X_features, y_feat, feat_names,
                                save_path="results/feature_distributions.png")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_features, y_feat, test_size=0.25, random_state=args.seed, stratify=y_feat
    )

    with Timer():
        rf_model = train_baseline(X_tr, y_tr, model_type="random_forest", seed=args.seed)

    rf_metrics = evaluate_model(rf_model, X_te, y_te, model_name="Random Forest")
    all_results["Random Forest"] = rf_metrics

    # RF confusion matrix + ROC
    rf_pred   = rf_model.predict(X_te)
    rf_proba  = rf_model.predict_proba(X_te)[:, 1]
    plot_confusion_matrix(y_te, rf_pred, title="Random Forest — Confusion Matrix",
                           save_path="results/rf_confusion_matrix.png")

    # Save RF model
    save_model(rf_model, "models/random_forest.joblib")

    # Spike Encoding 
    with Timer():
        spikes = rate_encode(X_features, T=args.T, seed=args.seed)

    plot_spike_trains(spikes, sample_idx=0, n_neurons=30, save_path="results/spike_trains.png")

    # Split spikes (same indices as features)
    n_total  = len(y_feat)
    n_test   = int(n_total * 0.25)
    indices  = np.arange(n_total)
    np.random.shuffle(indices)
    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    spikes_train = spikes[:, train_idx, :]
    spikes_test  = spikes[:, test_idx,  :]
    y_snn_train  = y_feat[train_idx]
    y_snn_test   = y_feat[test_idx]

    # Adapt SNN input size to feature count
    n_features = X_features.shape[1]

    # Train SNN 
    snn_model = EEGDepression_SNN(n_inputs=n_features, n_hidden=128, n_outputs=2, beta=0.9)
    logger.info(f"SNN parameters: {sum(p.numel() for p in snn_model.parameters()):,}")

    with Timer():
        history = train_snn(
            snn_model, spikes_train, y_snn_train,
            spikes_val=spikes_test, y_val=y_snn_test,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed
        )

    plot_training_curve(history, save_path="results/training_loss.png")

    # Evaluate pre-adaptation SNN
    snn_pred_before, snn_scores_before, snn_metrics_before = evaluate_snn(
        snn_model, spikes_test, y_snn_test, device=device, model_name="SNN (before adapt)"
    )
    all_results["SNN"] = snn_metrics_before
    plot_confusion_matrix(y_snn_test, snn_pred_before,
                           title="SNN — Confusion Matrix (Before Adaptation)",
                           save_path="results/snn_confusion_matrix.png")

    save_snn(snn_model, "models/snn_model.pt")

    # Patient Adaptation
    # Simulate a new patient (last 5 subjects from test set)
    n_patient = min(5, len(y_snn_test))
    spikes_patient = spikes_test[:, :n_patient, :]
    y_patient      = y_snn_test[:n_patient]

    adapted_model, adapt_history = patient_adaptation(
        snn_model, spikes_patient, y_patient, n_epochs=3, lr=5e-4, device=device
    )

    # Evaluate adapted SNN on full test set
    snn_pred_after, snn_scores_after, snn_metrics_after = evaluate_snn(
        adapted_model, spikes_test, y_snn_test, device=device, model_name="SNN (after adapt)"
    )
    all_results["SNN (Adapted)"] = snn_metrics_after

    plot_confusion_matrix(y_snn_test, snn_pred_after,
                           title="SNN — Confusion Matrix (After Adaptation)",
                           save_path="results/adapted_snn_confusion_matrix.png")

    print_adaptation_table(snn_metrics_before, snn_metrics_after)
    save_snn(adapted_model, "models/snn_adapted.pt")

    print("\n Final comparison plots …")

    # ROC Curve 
    roc_data = {
        "Random Forest": (y_te.astype(int),         rf_proba),
        "SNN":           (y_snn_test.astype(int),    snn_scores_before),
        "SNN (Adapted)": (y_snn_test.astype(int),    snn_scores_after),
    }
    plot_roc_curve(roc_data, save_path="results/roc_curve.png")

    # Model comparison bar chart
    plot_model_comparison(all_results, save_path="results/model_comparison.png")

    # Save JSON summary
    save_results(all_results, "results/metrics_summary.json")

    # Print Final Summary 
    print("  FINAL RESULTS SUMMARY")
    print(f"  {'Model':<20} | {'Accuracy':>8} | {'F1':>8} | {'Precision':>9} | {'Recall':>8}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")
    for model_name, metrics in all_results.items():
        print(f"  {model_name:<20} | {metrics['accuracy']:>8.4f} | "
              f"{metrics['f1']:>8.4f} | {metrics['precision']:>9.4f} | "
              f"{metrics['recall']:>8.4f}")
    print("=" * 60)

    print("\n Saved files:")
    print("  models/random_forest.joblib")
    print("  models/snn_model.pt")
    print("  models/snn_adapted.pt")
    print("  results/eeg_comparison.png")
    print("  results/feature_distributions.png")
    print("  results/spike_trains.png")
    print("  results/training_loss.png")
    print("  results/rf_confusion_matrix.png")
    print("  results/snn_confusion_matrix.png")
    print("  results/adapted_snn_confusion_matrix.png")
    print("  results/roc_curve.png")
    print("  results/model_comparison.png")
    print("  results/metrics_summary.json")
    print("\n[OK] Pipeline complete!\n")


if __name__ == "__main__":
    main()
