FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8080

# Run Streamlit app
CMD bash -lc 'streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0 --server.enableCORS false --server.headless true'
