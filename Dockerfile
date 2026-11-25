FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY sentiment/ ./sentiment/
COPY --chown=10001:10001 models/sentiment.joblib ./models/sentiment.joblib
COPY --chown=10001:10001 reports/metrics.json ./reports/metrics.json
COPY README.md ./
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home app \
    && mkdir -p data \
    && chown 10001:10001 data models reports

USER 10001:10001
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--no-control-socket", "sentiment.web:create_app()"]
