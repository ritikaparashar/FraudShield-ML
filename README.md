# FraudShield ML

FraudShield ML is a reproducible credit card fraud detection pipeline built with Python, scikit-learn, and imbalanced-learn. It trains a model for detecting suspicious transactions in a highly imbalanced dataset, saves the trained pipeline, exports evaluation metrics, and provides a command-line script for scoring new transactions.

## Overview

Fraud detection data is usually imbalanced: legitimate transactions heavily outnumber fraudulent ones. This project handles that imbalance directly with stratified splitting, SMOTE resampling, class-weighted modeling, and metrics that are more useful than accuracy alone.

The pipeline focuses on:

- repeatable training from a local CSV or ZIP dataset
- consistent preprocessing with `PowerTransformer`
- class imbalance handling with `SMOTE`
- fraud-focused evaluation using ROC-AUC, PR-AUC, precision, recall, and F1 score
- reusable model export with `joblib`
- JSON-based single-transaction inference

## Features

- Loads `creditcard.csv` or a ZIP file containing the dataset.
- Validates the expected `Class` target column.
- Drops `Time` by default while keeping anonymized PCA features and `Amount`.
- Uses stratified train/test splitting to preserve the rare fraud class distribution.
- Builds an imbalanced-learn pipeline:
  - `PowerTransformer`
  - `SMOTE`
  - `RandomForestClassifier`
- Supports `--quick` mode for faster local verification.
- Saves the trained model to `artifacts/fraudshield_pipeline.joblib`.
- Saves metrics to `artifacts/metrics.json`.
- Scores a single transaction from a JSON payload.

## Dataset

This project is designed for the public Kaggle dataset:

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains European cardholder transactions from September 2013. The available fields are:

- `Time` - seconds elapsed between the transaction and the first transaction in the dataset
- `V1` to `V28` - anonymized PCA-transformed transaction features
- `Amount` - transaction amount
- `Class` - target label, where `0` means legitimate and `1` means fraudulent

The dataset is not committed to this repository. Place it in one of these paths:

```text
data/creditcard.csv
data/creditcard_dataset.zip
```

## Project Structure

```text
FraudShield-ML/
  data/
    .gitkeep
  src/
    train_model.py
    predict_transaction.py
  requirements.txt
  README.md
  LICENSE
  .gitignore
```

Generated files are written to `artifacts/` and ignored by Git.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run a faster local training pass:

```bash
python src/train_model.py --data data/creditcard.csv --quick
```

Run the full training pipeline:

```bash
python src/train_model.py --data data/creditcard.csv
```

Use a ZIP dataset directly:

```bash
python src/train_model.py --data data/creditcard_dataset.zip
```

Write outputs to a custom directory:

```bash
python src/train_model.py --data data/creditcard.csv --artifacts-dir artifacts
```

## Training Outputs

After training, the project writes:

```text
artifacts/fraudshield_pipeline.joblib
artifacts/metrics.json
```

The metrics file includes:

- ROC-AUC
- PR-AUC
- precision
- recall
- F1 score
- confusion matrix
- classification report
- row count
- fraud row count
- fraud rate
- quick/full mode indicator
- model pipeline summary

## Score a Transaction

After training, score one transaction with `predict_transaction.py`.

```bash
python src/predict_transaction.py \
  --model artifacts/fraudshield_pipeline.joblib \
  --transaction '{"V1": -1.3, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
```

Example output:

```json
{
  "fraud_probability": 0.12,
  "risk_label": "low"
}
```

Risk labels are assigned from the predicted fraud probability:

- `high` - probability greater than or equal to `0.80`
- `medium` - probability greater than or equal to `0.45`
- `low` - probability below `0.45`

## Model Design

The training script uses a single pipeline object so preprocessing, resampling, and classification stay tied together.

```text
PowerTransformer -> SMOTE -> RandomForestClassifier
```

This structure helps keep training and inference reproducible. The saved pipeline can be loaded later without manually rebuilding the preprocessing steps.

## Evaluation Notes

Accuracy is not the main metric for this dataset because the fraud class is rare. A model can achieve high accuracy while missing many fraudulent transactions. This project reports metrics that better describe fraud detection behavior:

- `recall` - how many fraudulent transactions were caught
- `precision` - how many predicted fraud cases were actually fraudulent
- `F1 score` - balance between precision and recall
- `ROC-AUC` - ranking quality across thresholds
- `PR-AUC` - precision-recall performance under class imbalance

## Limitations

- The default prediction thresholds are simple probability cutoffs and should be tuned for the target operating environment.
- The current pipeline is batch-oriented and does not include a web API.
- Model explanations are not included yet.
- The Kaggle dataset is anonymized and historical, so live deployment would require fresh data validation and monitoring.

## Future Improvements

- Add threshold tuning based on false-positive and false-negative costs.
- Add SHAP or permutation-importance explanations.
- Add a FastAPI inference service.
- Add a Streamlit monitoring dashboard.
- Add automated tests for data loading, feature validation, and inference output.
- Add experiment tracking for comparing model versions.

## License

This project is available under the MIT License.
