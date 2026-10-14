# Multi-stage Dockerfile for FreightTiger Smart Shipping Cost Assistant
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, tests, and dataset
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY README.md STUDY_GUIDE.md /app/
COPY AI\ Intern\ Case\ Study/ /app/AI\ Intern\ Case\ Study/

# Default entrypoint runs verification and CLI help
CMD ["python3", "src/cli.py", "verify-reproducibility"]
