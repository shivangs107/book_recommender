FROM python:3.12-slim

WORKDIR /app

# Install system-level dependencies for numpy, pandas, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Install remaining Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Create necessary folders
RUN mkdir -p data/processed data/assets app

# Copy application files
COPY app/gradio_dashboard.py app/

# Copy necessary data files
COPY data/processed/tagged_description.txt data/processed/
COPY data/processed/books_with_emotions.csv data/processed/
COPY data/assets/cover-not-found.jpg data/assets/

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app/gradio_dashboard.py"]
