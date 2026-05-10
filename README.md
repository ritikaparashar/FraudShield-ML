# FraudShield ML

FraudShield ML is a credit card fraud detection project built by **Ritika Parashar**. It trains a reproducible machine learning pipeline for identifying suspicious transactions in a highly imbalanced financial dataset.

The project focuses on the real business problem behind fraud detection: fraudulent transactions are rare, but missing them can be expensive. Because of that, the model is evaluated with fraud recall, precision, F1 score, ROC-AUC, and PR-AUC instead of accuracy alone.

## What This Project Does

- Loads credit card transaction data from a local CSV or ZIP file.
- Separates features from the fraud label.
- Uses stratified train/test splitting so the rare fraud class is represented correctly.
- Applies `PowerTransformer` to stabilize numeric feature distributions.
- Uses `SMOTE` to handle class imbalance during training.
- Trains a `RandomForestClassifier`.
- Saves a reusable model pipeline.
- Saves evaluation metrics as JSON.
- Includes a prediction script for scoring a single transaction from JSON.

## Dataset

This project is designed for the public Kaggle credit card fraud detection dataset:

[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains European cardholder transactions from September 2013. It includes anonymized PCA features `V1` to `V28`, transaction `Amount`, transaction `Time`, and a binary `Class` label:

- `0`: legitimate transaction
- `1`: fraudulent transaction

For repository cleanliness and data ownership, the dataset is not committed here. Download it from Kaggle and place it in one of these locations:

```text
data/creditcard.csv
data/creditcard_dataset.zip
```

## Project Structure

```text
.
├── src/
│   ├── predict_transaction.py
│   └── train_model.py
├── data/
│   └── .gitkeep
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a fast local model:

```bash
python src/train_model.py --data data/creditcard.csv --quick
```

Train the full model:

```bash
python src/train_model.py --data data/creditcard.csv
```

The training command writes:

```text
artifacts/fraudshield_pipeline.joblib
artifacts/metrics.json
```

## Example Prediction

After training, score one transaction:

```bash
python src/predict_transaction.py \
  --model artifacts/fraudshield_pipeline.joblib \
  --transaction '{"V1": -1.3, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62}'
```

The output includes a fraud probability and a simple risk label.

## Why This Is Recruiter-Friendly

FraudShield ML shows practical machine learning work beyond a notebook:

- It handles severe class imbalance.
- It avoids misleading accuracy-only evaluation.
- It saves a reusable model artifact.
- It provides a command-line prediction path.
- It separates training, inference, dependencies, and generated artifacts.

## Interview Summary

FraudShield ML is a machine learning project for credit card fraud detection. I built a reproducible training pipeline using Python, scikit-learn, imbalanced-learn, SMOTE, and Random Forest. Since fraud data is highly imbalanced, I focused on recall, PR-AUC, ROC-AUC, precision, and F1 score instead of accuracy alone. The project can train a model, save the pipeline, export metrics, and score new transactions from JSON.

## Future Improvements

- Add threshold tuning based on business cost.
- Add SHAP explanations for model interpretability.
- Add a FastAPI inference service.
- Add a Streamlit dashboard for fraud monitoring.
- Export the model for mobile or backend deployment.
