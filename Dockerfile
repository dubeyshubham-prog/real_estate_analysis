FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    RUNTIME_DIR=/tmp/proplens \
    ENABLE_VISION=false \
    USE_OLLAMA_ROUTER=false

RUN useradd --create-home --uid 1000 user

WORKDIR /home/user/app

COPY requirements-core.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements-core.txt

COPY --chown=user:user . .

USER user

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
