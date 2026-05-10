"""Score a single credit card transaction with a trained FraudShield model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud risk for one transaction.")
    parser.add_argument("--model", default="artifacts/fraudshield_pipeline.joblib")
    parser.add_argument(
        "--transaction",
        required=True,
        help="JSON object containing V1-V28 and Amount fields.",
    )
    return parser.parse_args()


def risk_label(probability: float) -> str:
    if probability >= 0.80:
        return "high"
    if probability >= 0.45:
        return "medium"
    return "low"


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    transaction = json.loads(args.transaction)
    model = joblib.load(model_path)
    frame = pd.DataFrame([transaction])
    probability = float(model.predict_proba(frame)[:, 1][0])

    print(
        json.dumps(
            {
                "fraud_probability": probability,
                "risk_label": risk_label(probability),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
