"""Train the FraudShield credit card fraud detection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
DROP_COLUMNS = ["Time"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a credit card fraud detection model.")
    parser.add_argument(
        "--data",
        default="data/creditcard.csv",
        help="Path to creditcard.csv or a zip file containing the CSV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for saved model and metrics.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Train a smaller Random Forest for fast local verification.",
    )
    return parser.parse_args()


def load_transactions(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download the Kaggle dataset and place it there."
        )

    if data_path.suffix.lower() == ".zip":
        return pd.read_csv(data_path, compression="zip")

    return pd.read_csv(data_path)


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    normalized = data.rename(columns=lambda name: name.strip()).copy()
    if TARGET_COLUMN not in normalized:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    y = normalized[TARGET_COLUMN].astype(int)
    columns_to_drop = [TARGET_COLUMN] + [column for column in DROP_COLUMNS if column in normalized]
    X = normalized.drop(columns=columns_to_drop)
    return X, y


def make_pipeline(quick: bool) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=80 if quick else 300,
        max_depth=7,
        min_samples_leaf=4,
        min_samples_split=5,
        max_features="log2",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("transform", PowerTransformer()),
            ("balance", SMOTE(random_state=RANDOM_STATE)),
            ("model", classifier),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    fraud_probability = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, fraud_probability),
        "pr_auc": average_precision_score(y_test, fraud_probability),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1_score": f1_score(y_test, prediction),
        "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        "classification_report": classification_report(y_test, prediction, output_dict=True),
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = load_transactions(data_path)
    X, y = split_features_and_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = make_pipeline(quick=args.quick)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    metrics["rows"] = int(len(data))
    metrics["fraud_rows"] = int(y.sum())
    metrics["fraud_rate"] = float(y.mean())
    metrics["quick_mode"] = bool(args.quick)
    metrics["model"] = "PowerTransformer + SMOTE + RandomForestClassifier"

    model_path = artifacts_dir / "fraudshield_pipeline.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(
        "Metrics: "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )


if __name__ == "__main__":
    main()
