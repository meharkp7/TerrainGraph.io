FROM python:3.11-slim

WORKDIR /app

# System deps
# NOTE: libgl1 (not libgl1-mesa-glx) — the -glx transitional package was
# removed in Debian trixie, which python:3.11-slim is now based on.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
# Upgrade pip first: base-image pip rejects the typing_extensions wheel
# (metadata Name mismatch) and falls back to an sdist that needs flit_core.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Download model at build time from HuggingFace.
# ARG alone is not visible to `python download_model.py` (it reads the
# environment), so promote it to ENV with the public repo as default.
ARG HF_MODEL_REPO=mk1647/terrain-intelligence
ENV HF_MODEL_REPO=${HF_MODEL_REPO} \
    MODEL_CHECKPOINT=/app/runs/deployed/best.pth
RUN python download_model.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
