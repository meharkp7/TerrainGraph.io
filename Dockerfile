FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

COPY . .

# Download model — uses HF_MODEL_REPO env var set in Railway dashboard
# Safe to fail: if no repo set, model loads from /runs/ on disk
RUN python download_model.py || echo "Model download skipped"

EXPOSE 8000

# Single worker is critical — multiple workers = multiple model copies = OOM
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]