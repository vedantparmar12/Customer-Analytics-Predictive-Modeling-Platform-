# Multi-stage build for production deployment
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements_advanced.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements_advanced.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Copy spaCy model
COPY --from=builder /usr/local/lib/python3.9/site-packages/en_core_web_sm /usr/local/lib/python3.9/site-packages/en_core_web_sm

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create directories for artifacts and logs
RUN mkdir -p artifacts logs mlruns && \
    chown -R appuser:appuser artifacts logs mlruns

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden)
CMD ["streamlit", "run", "app.py", "--server.maxUploadSize=200"]