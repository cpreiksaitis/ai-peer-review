# AI Peer Review - Production Docker Image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including git for pip, libgl for PyMuPDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config.yaml .
COPY start.sh .

# Create data directory and make start script executable
RUN mkdir -p data && chmod +x start.sh

# Expose port (Railway uses $PORT)
EXPOSE 8000

# Run production server via start script
CMD ["./start.sh"]

