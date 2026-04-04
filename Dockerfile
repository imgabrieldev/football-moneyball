FROM python:3.14-slim

WORKDIR /app

# Install system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
COPY football_moneyball/ football_moneyball/
RUN pip install --no-cache-dir .

EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "football_moneyball.api:app", "--host", "0.0.0.0", "--port", "8000"]
