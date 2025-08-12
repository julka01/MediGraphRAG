# Use official Python image for macOS compatibility
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OLLAMA_HOST="0.0.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (CPU version)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip list | grep neo4j \
    && echo "Neo4j package installed successfully" \
    || echo "Neo4j package installation failed"
# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000 || exit 1

# Command to run both Ollama and the app
CMD sh -c "ollama serve & uvicorn app:app --host 0.0.0.0 --port 8000"
