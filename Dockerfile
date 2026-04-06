FROM python:3.11-slim

WORKDIR /app

# System deps — minimal set
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# App code
COPY . .

# Download model at build time (not runtime — avoids cold-start OOM)
ARG HF_MODEL_REPO
ARG HF_TOKEN
RUN if [ -n "$HF_MODEL_REPO" ]; then python download_model.py; fi

EXPOSE 8000

# Single worker — multiple workers would each load the model = instant OOM
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"]