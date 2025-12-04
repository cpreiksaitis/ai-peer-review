# AI Peer Review - Production Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including git for pip git dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config.yaml .

# Create data directory
RUN mkdir -p data

# Expose port (Railway uses $PORT)
EXPOSE 8000

# Run production server - explicit shell for variable expansion
CMD ["/bin/sh", "-c", "uvicorn src.web.production.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

