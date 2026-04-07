FROM python:3.10-slim

WORKDIR /app

# Only the libs actually needed:
#   libgomp1  → PyTorch parallel ops
#   libglib2.0-0 → opencv-python-headless
# libgl1 is NOT needed when using opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (avoids pip pulling CUDA builds by accident)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest (torch/torchvision already satisfied, pip skips them)
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

COPY . .

# Download model weights from HuggingFace at build time.
# Fails gracefully if HF_MODEL_REPO is not set.
RUN python download_model.py || echo "⚠️  Model download skipped — mount or set HF_MODEL_REPO"

EXPOSE 8000

# Single worker is mandatory: multiple workers = multiple model copies = OOM
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75"]