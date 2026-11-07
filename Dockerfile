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

# Python deps — the LEAN set (no torch): segmentation runs on the
# HuggingFace Space, so this image fits Railway's 512MB tier.
# Upgrade pip first: base-image pip rejects the typing_extensions wheel
# (metadata Name mismatch) and falls back to an sdist that needs flit_core.
COPY requirements-railway.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-railway.txt

# Lean runtime: unbuffered logs + inference via the HF Space.
# (HF_SPACE_ID must be set in Railway env vars — see docs/DEPLOYMENT.md.)
ENV PYTHONUNBUFFERED=1 \
    INFERENCE_BACKEND=space

# App code (no model download: weights live on the Space, not here)
COPY . .

# Railway assigns PORT=8080 and routes public traffic using the exposed port,
# so EXPOSE must match — 8000 here caused 502 connection-refused while the
# internal healthcheck on 8080 passed. (Plain docker runs still default to
# ${PORT:-8000} via the CMD below.)
EXPOSE 8080

# Same PORT handling as railway.toml, so plain `docker run -e PORT=...`
# works too (Railway's startCommand overrides this CMD when deployed there).
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
