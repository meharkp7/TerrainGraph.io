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

# Lean runtime: unbuffered logs + capped BLAS threads (each thread arena
# costs RAM; 2 is plenty for single-request inference on shared vCPU).
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# App code
COPY . .

# Download model at build time from HuggingFace.
# ARG alone is not visible to `python download_model.py` (it reads the
# environment), so promote it to ENV with the public repo as default.
ARG HF_MODEL_REPO=mk1647/terrain-intelligence
ENV HF_MODEL_REPO=${HF_MODEL_REPO} \
    MODEL_CHECKPOINT=/app/runs/deployed/best.pth
RUN python download_model.py

# Railway assigns PORT=8080 and routes public traffic using the exposed port,
# so EXPOSE must match — 8000 here caused 502 connection-refused while the
# internal healthcheck on 8080 passed. (Plain docker runs still default to
# ${PORT:-8000} via the CMD below.)
EXPOSE 8080

# Same PORT handling as railway.toml, so plain `docker run -e PORT=...`
# works too (Railway's startCommand overrides this CMD when deployed there).
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
