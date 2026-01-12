# Dockerfile for FunASR Triton Inference Server
# Based on NVIDIA Triton Inference Server with Python backend

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    transformers \
    funasr \
    kaldifeat \
    vllm \
    soundfile \
    librosa \
    numpy \
    tritonclient[all]

# Create model repository directory
RUN mkdir -p /models

# Copy model repository
COPY model_repo_funasr /models/funasr_repo

# Set working directory
WORKDIR /workspace

# Copy client scripts
COPY http_client.py /workspace/
COPY grpc_client.py /workspace/

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Default command
CMD ["tritonserver", "--model-repository=/models/funasr_repo"]
