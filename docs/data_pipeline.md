# PropLens Data Pipeline

## Purpose

The data pipeline converts raw flat, house, and apartment metadata into a
validated modelling dataset while preserving human-readable categorical values.
Encoding is deferred to the model pipeline to prevent leakage and keep
inference behavior consistent.

## Flow

```text
Raw flat and house CSVs
        |
        v
Property cleaning and merge
        |
        v
Sector extraction and standardization
        |
        v
Area, room, furnishing, age, and luxury features
        |
        v
Outlier treatment
        |
        v
Missing-value treatment
        |
        v
Feature evaluation and selection
        |
        v
Model-ready tabular dataset
```

## Commands

Validate every stage without overwriting processed files:

```powershell
python -m scripts.run_data_pipeline --dry-run
```

Run and save all configured outputs:

```powershell
python -m scripts.run_data_pipeline
```

Use the non-dry command together with model retraining because the new output
preserves categorical strings instead of legacy numeric labels.

## Outputs

| Stage | Output |
|---|---|
| Property cleaning | `data/processed/cleaned_properties.csv` |
| Sector cleaning | `data/processed/gurgaon_properties_cleaned_v1.csv` |
| Feature engineering | `data/processed/gurgaon_properties_cleaned_v2.csv` |
| Outlier treatment | `data/processed/gurgaon_properties_outlier_treated.csv` |
| Missing-value treatment | `data/processed/gurgaon_properties_missing_value_imputation.csv` |
| Feature selection | `data/processed/gurgaon_properties_post_feature_selection.csv` |
| Feature report | `reports/feature_importance.csv` |

## Current validation baseline

- Raw merged properties: 3,965 rows
- Standardized-sector properties: 3,811 rows
- Outlier-treated properties: 3,685 rows
- Final selected features: 14 plus the `price` target
- Unit tests: 20 passing

## Known technical debt

- Some sector and outlier corrections still depend on legacy DataFrame indexes.
  Retaining a stable source listing ID will allow these rules to become robust.
- Current input files are static snapshots. Production use requires a versioned
  ingestion process and data-quality monitoring.
