# Use Python 3.11 slim image for better performance and security
FROM python:3.11-slim

# Standard Quail ships native 8 kHz and 16 kHz artifacts. Voice Focus ships
# only a 16 kHz artifact; the AIC SDK resamples 8 kHz telephony input internally.
# Do not add a quail_vf_*_8khz path: ai-coustics does not publish that model.
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    NLTK_DATA=/usr/local/nltk_data\
    AIC_MODEL_PATH=/app/models/voice/aic/quail_l_8khz.aicmodel \
    AIC_MODEL_PATH_16KHZ=/app/models/voice/aic/quail_l_16khz.aicmodel \
    AIC_VOICE_FOCUS_MODEL_PATH=/app/models/voice/aic/quail_vf_2_2_l_16khz.aicmodel \
    AIC_MODEL_PATH_S=/app/models/voice/aic/quail_s_8khz.aicmodel \
    AIC_MODEL_PATH_S_16KHZ=/app/models/voice/aic/quail_s_16khz.aicmodel \
    BREEZE_BUDDY_NOISE_MODEL_DIR=/app/models/voice/noizzen \
    UV_CACHE_DIR=/app/.uv-cache

# Install system dependencies required for audio processing and compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libffi-dev \
    libssl-dev \
    pkg-config \
    portaudio19-dev \
    python3-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Create app and model directories
WORKDIR /app
RUN mkdir -p /app/models/voice/aic

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
# Use --no-install-project to avoid installing the app/ package at this stage
# This allows optimal Docker layer caching - dependencies layer is cached separately
# --compile-bytecode: write .pyc at BUILD time. Without it every bot subprocess
# recompiles ~12k source files on each call (PYTHONDONTWRITEBYTECODE=1 means the
# result is never cached). Measured 1.4-1.8x faster bot startup under CPU pressure.
RUN uv sync --frozen --no-dev --no-install-project --compile-bytecode && \
    uv pip show pipecat-ai

# AIC (ai-coustics) Quail models, from ai-coustics' public CDN (artifacts.ai-coustics.io).
# The installed SDK picks the model-file version compatible with itself and verifies
# the checksum; files are saved under the fixed names in AIC_*_PATH above.
# No cloud credentials needed. The build fails if any model can't be downloaded.
RUN /app/.venv/bin/python -c "import shutil, aic_sdk; [shutil.move(aic_sdk.Model.download(m, '/tmp/aic'), '/app/models/voice/aic/' + n) for m, n in [('quail-l-8khz', 'quail_l_8khz.aicmodel'), ('quail-l-16khz', 'quail_l_16khz.aicmodel'), ('quail-vf-2.2-l-16khz', 'quail_vf_2_2_l_16khz.aicmodel'), ('quail-s-8khz', 'quail_s_8khz.aicmodel'), ('quail-s-16khz', 'quail_s_16khz.aicmodel')]]" && \
    ls -l /app/models/voice/aic && rm -rf /tmp/aic

# Create NLTK data directory and download required data
RUN mkdir -p /usr/local/nltk_data && \
    uv run python -m nltk.downloader punkt punkt_tab -d /usr/local/nltk_data

# Noizzen noise filter: Ceva DPDFNet models (Apache-2.0; docs/NOISE_FILTER.md).
# The build fails if a model can't be downloaded or doesn't match.
ARG DPDFNET_URL=https://sdk.breezesdk.store/breezebuddy/models
ARG DPDFNET_CHECKSUMS="6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d  /app/models/voice/noizzen/dpdfnet2_8khz.onnx\n371d26182aff0e1e0d31354e24c81f79cd57458f5a7dd003fc10a7ebf64255e0  /app/models/voice/noizzen/dpdfnet_baseline_16khz.onnx"
RUN mkdir -p /app/models/voice/noizzen && \
    curl -fsSL --retry 5 --retry-all-errors -o /app/models/voice/noizzen/dpdfnet2_8khz.onnx ${DPDFNET_URL}/dpdfnet2_8khz.onnx && \
    curl -fsSL --retry 5 --retry-all-errors -o /app/models/voice/noizzen/dpdfnet_baseline_16khz.onnx ${DPDFNET_URL}/dpdfnet_baseline_16khz.onnx && \
    printf "${DPDFNET_CHECKSUMS}\n" | sha256sum -c -
COPY app/ai/voice/noise_filter/noizzen/engine/ /app/app/ai/voice/noise_filter/noizzen/engine/
RUN uv run --no-project --no-cache --with onnx==1.19.1 --with numpy==2.2.6 \
        --with protobuf==5.29.6 --with ml-dtypes==0.5.4 --with typing-extensions==4.15.0 \
        python app/ai/voice/noise_filter/noizzen/engine/build.py /app/models/voice/noizzen

# Copy application code
COPY . .

# Precompile the app's own bytecode too (same reason as --compile-bytecode above).
RUN /app/.venv/bin/python -m compileall -q -j 0 /app/app || true

# Set proper permissions
RUN chmod +x run.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app/.uv-cache && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /usr/local/nltk_data
USER appuser

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["uv", "run", "python", "run.py"]
