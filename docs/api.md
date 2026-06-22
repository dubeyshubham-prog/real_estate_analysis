# Application API

PropLens uses a FastAPI application factory in `app/main.py`. Route modules
live under `app/api/routes`, request schemas live under `app/schemas`, and
business logic remains in `app/services` or `src`.

## Runtime routes

- `/` and `/predict`: property price prediction
- `/recommend` and `/recommendations`: hybrid property recommendations
- `/analysis`: market-analysis dashboard
- `/deep-vision`: visual similarity search
- `/estate-agent`: agent assistant
- `/rag`: PDF ingestion and retrieval
- `/health/live`: process liveness
- `/health/ready`: required artifact readiness

## Validation and errors

FastAPI validates form fields before business logic executes. Cross-field
rules, such as a property floor not exceeding the building's total floors,
are implemented in Pydantic schemas. Expected application errors return
consistent JSON responses through the handlers in `app/api/error_handlers.py`.

Uploads use server-generated filenames and enforce content type, extension,
non-empty content, and the configured maximum size before loading heavy
vision or RAG services.

## Configuration

Runtime directories and model paths come from `config/settings.py`. This
keeps route and service code independent of the machine's working directory.
Environment-variable examples are available in `.env.example`.
