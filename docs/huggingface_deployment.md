# Deploy PropLens on Hugging Face Spaces

PropLens uses a Docker Space with the free CPU Basic environment. The
container listens on port `7860`, which is Hugging Face's standard Docker
Space port.

## Features in the free demo

- Property price prediction
- Hybrid property recommendations
- Gurgaon market analysis
- Rule-routed AI agent with live web search
- PDF upload and session-based RAG

Deep Vision is disabled in the free deployment because the current visual
embedding database references a large local source-image library that is not
included in the repository. The Docker image installs `requirements-core.txt`
and therefore avoids shipping PyTorch and TorchVision in the free demo.

## Free-tier persistence

The Space writes SQLite, uploaded documents, Chroma data, and logs under
`/tmp/proplens`. Free Space storage is not persistent, so this runtime data
resets when the Space restarts or goes to sleep. Trained price and
recommendation artifacts are bundled in the repository and remain available.

## Required repository files

Before pushing to the Space, verify these files are committed:

```text
artifacts/models/price_prediction/model.joblib
artifacts/models/price_prediction/reference_data.joblib
artifacts/models/price_prediction/metadata.json
artifacts/models/recommendation/hybrid_recommender.joblib
artifacts/models/recommendation/metadata.json
data/raw/apartments.csv
data/raw/gurgaon_property.csv
data/raw/latlongs.csv
```

## Space configuration

Create a new Hugging Face Space with:

- SDK: Docker
- Hardware: CPU Basic - Free
- Visibility: Public

Optional Space variables:

```text
APP_ENV=production
LOG_LEVEL=INFO
ENABLE_VISION=false
USE_OLLAMA_ROUTER=false
RUNTIME_DIR=/tmp/proplens
```

The Space URL will follow this pattern:

```text
https://YOUR_USERNAME-proplens.hf.space
```
