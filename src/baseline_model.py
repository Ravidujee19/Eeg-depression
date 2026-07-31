import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from typing import Tuple, Dict
from src.utils import get_logger, ensure_dirs

logger = get_logger()


# Random Forest 
def build_random_forest(n_estimators: int = 100, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])


def build_logistic_regression(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced"
        ))
    ])


# Training 
def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    seed: int = 42
) -> Pipeline:
    if model_type == "random_forest":
        model = build_random_forest(seed=seed)
    elif model_type == "logistic_regression":
        model = build_logistic_regression(seed=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Training {model_type} on {X_train.shape[0]} samples …")
    model.fit(X_train, y_train)
    logger.info("Training complete done")
    return model


# Evaluation 
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Baseline"
) -> Dict[str, float]:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Results")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k:<12}: {v:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Healthy','Depressed'])}")

    return metrics


# Feature Importance 
def get_feature_importance(model: Pipeline, feat_names: list, top_n: int = 20) -> dict:
    if not hasattr(model.named_steps["clf"], "feature_importances_"):
        return {}

    importances = model.named_steps["clf"].feature_importances_
    sorted_idx  = np.argsort(importances)[::-1][:top_n]

    return {feat_names[i]: float(importances[i]) for i in sorted_idx}


# Persistence 
def save_model(model, filepath: str = "models/random_forest.joblib") -> None:
    ensure_dirs(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    logger.info(f"Model saved -> {filepath}")


def load_model(filepath: str) -> Pipeline:
    model = joblib.load(filepath)
    logger.info(f"Model loaded <- {filepath}")
    return model
