FROM python:3.12-slim

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# App code
COPY app/ ./app/
COPY i18n/ ./i18n/

WORKDIR /workspace/app

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8000"]
