# Anti-UAV Detection and Tracking
# CUDA-enabled Docker image for training and inference

# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent timezone prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src/detection:/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev --frozen

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY Codes/ ./Codes/

# Create directories for data and outputs
RUN mkdir -p data weights runs

# Set default command
CMD ["uv", "run", "python", "-c", "print('Anti-UAV Detection Ready. Use docker run with appropriate command.')"]

# Example usage:
# Build: docker build -t anti-uav .
# Train: docker run --gpus all -v /path/to/data:/app/data anti-uav uv run python scripts/train.py --data data/drone.yaml
# Inference: docker run --gpus all -v /path/to/images:/app/input anti-uav uv run python scripts/infer.py --source /app/input
# Export: docker run --gpus all -v /path/to/weights:/app/weights anti-uav uv run python scripts/export.py --weights weights/best.pt
