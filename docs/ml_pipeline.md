# PropLens Price Model

## Design

The price model is a single scikit-learn pipeline:

```text
Validated feature DataFrame
        |
        v
Numeric median imputation and scaling
        |
        v
Categorical mode imputation and one-hot encoding
        |
        v
XGBoost regression on log-transformed price
```

All preprocessing is fitted inside cross-validation and training pipelines to
avoid leakage. Saved processed datasets preserve readable category names.

## Commands

Evaluate candidate models:

```powershell
python -m scripts.validate_models
```

Tune XGBoost:

```powershell
python -m scripts.tune_price_model
```

Train without saving:

```powershell
python -m scripts.train_ml --dry-run
```

Train and save model artifacts:

```powershell
python -m scripts.train_ml
```

Regenerate processed datasets and save the model together:

```powershell
python -m scripts.train_ml --save-data
```

## Artifacts

```text
artifacts/models/price_prediction/
├── model.joblib
├── reference_data.joblib
└── metadata.json
```

Metadata includes training time, feature schema, model parameters, holdout
metrics, and residual quantiles used for an approximate estimate range.
