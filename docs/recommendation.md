# PropLens Recommendation System

## Components

- Facility similarity using TF-IDF and cosine similarity
- Price/configuration similarity using structured and scaled listing details
- Location similarity using advertised landmark distances
- Weighted hybrid ranking with explainable component scores

## Training

Validate without saving:

```powershell
python -m scripts.train_recommender --dry-run
```

Train and save:

```powershell
python -m scripts.train_recommender
```

## Artifacts

```text
artifacts/models/recommendation/
├── hybrid_recommender.joblib
└── metadata.json
```

## Data quality

The source apartment CSV contains one duplicated header row. The pipeline
removes it using content validation rather than a brittle row-index rule.
